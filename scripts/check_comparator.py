#!/usr/bin/env python3
"""Replay checked-in proof targets with the official comparator.

Repository audits freeze the current trusted sources and replay their proofs;
they do not compare against a historical Git revision. prove.py separately
compares an agent's candidate against the pre-agent challenge.
"""
import argparse
import fcntl
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from comparator_common import AXIOMS, comparator_command, require_tools, snapshot_project, write_config
from audit_process import DEFAULT_TIMEOUT, positive_seconds, run_logged, service_unit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench.audit_trust_prep import add_import, prepare_source
from scripts.bench.audit_source import lean_code_mask

# Alias actual Lean names instead of trying to parse private, quoted, or
# hygienic names through comparator's dotted-string config format.
INVENTORY = r'''
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  -- Enumerate and look up in the same completed kernel environment. The
  -- elaborator's prefix-based lookup can miss realized private match equations
  -- even though they are already present in `env.constants`.
  let kernelEnv := env.toKernelEnv
  let selected : Array Name := __SELECTED__
  let mut names : Array Name := #[]
  if selected.isEmpty then
    for (name, info) in kernelEnv.constants.toList do
      if (env.getModuleIdxFor? name).isSome then continue
      if let .thmInfo _ := info then names := names.push name
  else
    names := selected
  names := names.qsort (·.toString < ·.toString)
__AXIOM_AUDIT__
  let mut aliases : Array String := #[]
  for i in [:names.size] do
    let name := names[i]!
    let some (.thmInfo info) := kernelEnv.find? name
      | throwError "Comparator target is not a theorem: {name}"
    let wrapperName := Name.mkSimple s!"veritile_comparator_target_{i}"
    if env.contains wrapperName then throwError "Reserved comparator target name: {wrapperName}"
    liftCoreM <| addDecl <| .thmDecl {
      name := wrapperName, levelParams := info.levelParams, type := info.type,
      value := mkConst name (info.levelParams.map Level.param)
    }
    aliases := aliases.push wrapperName.toString
  -- Definition-only regression fixtures still run comparator's primitive
  -- checks and kernel replay. Report zero original theorem targets explicitly.
  if aliases.isEmpty then
    let wrapperName := `veritile_comparator_empty_module
    if env.contains wrapperName then throwError "Reserved comparator target name: {wrapperName}"
    liftCoreM <| addDecl <| .thmDecl {
      name := wrapperName, levelParams := [], type := mkConst ``True,
      value := mkConst ``True.intro
    }
    aliases := #[wrapperName.toString]
  let report := Json.mkObj [
    ("theorems", toJson (names.map Name.toString)),
    ("aliases", toJson aliases)
  ]
  liftIO <| IO.FS.writeFile __REPORT__ report.compress
'''

ALL_THEOREM_AXIOMS = r'''
  let allowed : Array Name := __ALLOWED__
  let mut pending := names
  let mut seen : NameSet := {}
  while !pending.isEmpty do
    let name := pending.back!
    pending := pending.pop
    if seen.contains name then continue
    seen := seen.insert name
    let some info := kernelEnv.find? name
      | throwError "Missing proof dependency: {name}"
    if let .axiomInfo _ := info then
      unless allowed.contains name do throwError "DISALLOWED axiom: {name}"
    pending := pending ++ info.getUsedConstantsAsSet.toList.toArray
  logInfo m!"Full theorem axiom audit: theorems={names.size}"
'''


def prepare(workspace):
    """One independent cache/source copy per batch, shared by its workers."""
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = workspace / 'ready.json'
        if ready.exists():
            state = json.loads(ready.read_text())
            if state['root'] != str(ROOT):
                raise ValueError('Comparator workspace belongs to another project')
            return state
        if (workspace / 'failed').exists():
            raise ValueError('Comparator workspace initialization previously failed')
        try:
            comparator = require_tools()
            judge = workspace / 'project'
            judge.mkdir()
            hashes = snapshot_project(ROOT, judge, libraries=('VeriTileComparator',), submodules=True)
            (judge / 'VeriTileComparator').mkdir()
            manifest = ROOT / 'scripts/kernel-manifest.tsv'
            (judge / 'scripts').mkdir(exist_ok=True)
            if manifest.exists():
                shutil.copyfile(manifest, judge / 'scripts/kernel-manifest.tsv')
                hashes['scripts/kernel-manifest.tsv'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (ROOT / 'Logs').mkdir(exist_ok=True)
            logs = Path(tempfile.mkdtemp(prefix='comparator-check-', dir=ROOT / 'Logs'))
            (logs / 'inputs.json').write_text(json.dumps(hashes, indent=2) + '\n')
            judge_inputs = logs / 'judge-inputs.json'
            judge_inputs.write_text(json.dumps({name: hashlib.sha256((judge / name).read_bytes()).hexdigest()
                                               for name in hashes}, sort_keys=True) + '\n')
            state = {'root': str(ROOT), 'comparator': comparator, 'logs': str(logs),
                     'comparator_sha256': hashlib.sha256(Path(comparator).read_bytes()).hexdigest(),
                     'judge_inputs': str(judge_inputs),
                     'judge_inputs_sha256': hashlib.sha256(judge_inputs.read_bytes()).hexdigest()}
            ready.write_text(json.dumps(state))
            return state
        except Exception:
            (workspace / 'failed').touch()
            raise


def library_targets(manifest):
    modules, names = set(), []
    for line in manifest.read_text().splitlines():
        fields = line.split('\t')
        if len(fields) >= 5 and fields[1].startswith('VeriTile/') and fields[4] == 'proven':
            modules.add(fields[1][:-5].replace('/', '.'))
            names.append(fields[2])
    if not names:
        raise ValueError('No proven library targets in the kernel manifest')
    return sorted(modules), sorted(set(names))


def prepare_target(state, judge, source=None, trust=False, library=False, all_axioms=False):
    """Write one audited inventory module without claiming proof acceptance."""
    logs = Path(state['logs'])
    if library:
        modules, selected = library_targets(judge / 'scripts/kernel-manifest.tsv')
        text = ''.join(f'import {module}\n' for module in modules)
        label, tag = 'proven library manifest', 'Library'
    else:
        relative = source.resolve().relative_to(ROOT).as_posix()
        original = judge / relative
        if source.suffix != '.lean' or not original.is_file():
            raise ValueError(f'Not a snapshotted Lean source: {relative}')
        text = original.read_text()
        if trust:
            text = prepare_source(text, relative, judge / 'scripts/kernel-manifest.tsv')
        selected = []
        label = relative
        tag = 'File_' + hashlib.sha256((relative + str(trust)).encode()).hexdigest()[:24]
    module = 'VeriTileComparator.' + tag
    job_logs = Path(tempfile.mkdtemp(prefix=tag + '-', dir=logs))
    metadata_dir = judge / '.lake' / 'comparator' / tag
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata_dir / 'targets.json'
    selection = '#[' + ', '.join(f'{json.dumps(n)}.toName' for n in selected) + ']'
    footer = INVENTORY.replace('__SELECTED__', selection).replace(
        '__REPORT__', json.dumps(str(metadata)))
    standalone = bool(re.search(r'^\s*module\s*$', lean_code_mask(text), re.M))
    audit = ALL_THEOREM_AXIOMS.replace('__ALLOWED__', '#[' + ', '.join(
        json.dumps(name) + '.toName' for name in AXIOMS) + ']') if all_axioms and not standalone else ''
    footer = footer.replace('__AXIOM_AUDIT__', audit)
    program = add_import(text, 'Lean') + '\n' + footer
    (judge / 'VeriTileComparator' / (tag + '.lean')).write_text(program)
    (job_logs / 'Source.lean').write_text(program)
    return {'module': module, 'logs': job_logs, 'metadata': metadata,
            'source': label, 'tag': tag, 'all_axioms': all_axioms, 'requires_individual': standalone}


def build_inventory(judge, target, timeout=DEFAULT_TIMEOUT, heartbeat=30, cancel=None):
    """Compile one audited source with its own deadline and compiler trace."""
    job_logs = target['logs']
    build, elapsed = run_logged(['lake', 'build', target['module']], cwd=judge,
        log=job_logs / 'build-all.log', source=target['source'], stage='audit-compile',
        timeout=timeout, heartbeat=heartbeat, cancel=cancel)
    if build.returncode or not target['metadata'].is_file():
        print((job_logs / 'build-all.log').read_text(), file=sys.stderr)
        raise ValueError(f'{target["source"]}: elaboration/inventory failed; logs: {job_logs}')
    trace = judge / '.lake/build/lib/lean' / Path(*target['module'].split('.')).with_suffix('.trace')
    if not trace.is_file():
        raise ValueError('Missing per-module audit trace: ' + target['source'])
    entries = json.loads(trace.read_text())['log']
    if any(entry['level'] == 'error' for entry in entries):
        raise ValueError('Per-module compiler trace contains an error: ' + target['source'])
    (job_logs / 'build.log').write_text('\n'.join(entry['message'] for entry in entries) + '\n')
    shutil.copyfile(trace, job_logs / 'build.trace.json')
    inventory = json.loads(target['metadata'].read_text())
    shutil.copyfile(target['metadata'], job_logs / 'targets.json')
    target['audit_seconds'] = elapsed
    return inventory


def replay_target(state, judge, target, inventory, timeout=DEFAULT_TIMEOUT, heartbeat=30):
    job_logs, module = target['logs'], target['module']
    label, tag = target['source'], target['tag']
    config = write_config(judge, inventory['aliases'], challenge=module, solution=module,
                          filename=f'.lake/comparator/{tag}/config.json')
    shutil.copyfile(config, job_logs / 'config.json')
    unit = service_unit()
    result, elapsed = run_logged(comparator_command(judge, state['comparator'], config, unit=unit),
        cwd=judge, log=job_logs / 'comparator.log', source=label, stage='individual-replay',
        timeout=timeout, heartbeat=heartbeat, unit=unit)
    report = {**state, 'source': label, 'mode': 'source-replay',
              'theorem_count': len(inventory['theorems']),
              'theorems': inventory['theorems'], 'evidence_dir': str(job_logs),
              'audit_seconds': target.get('audit_seconds', 0), 'replay_seconds': elapsed,
              'all_theorems_axioms_checked': target.get('all_axioms', False),
              'comparator_exit': result.returncode, 'accepted': result.returncode == 0}
    (job_logs / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if result.returncode:
        print((job_logs / 'comparator.log').read_text(), file=sys.stderr)
        raise ValueError(f'{label}: comparator rejected; logs: {job_logs}')
    print(f'Comparator: {label}: {len(inventory["theorems"])} theorem(s) accepted; logs: {job_logs}', flush=True)
    return report


def check(workspace, source=None, trust=False, library=False, timeout=DEFAULT_TIMEOUT, heartbeat=30):
    timeout = positive_seconds(timeout)
    state = prepare(workspace)
    judge = workspace / 'project'
    target = prepare_target(state, judge, source, trust, library, all_axioms=trust)
    inventory = build_inventory(judge, target, timeout, heartbeat)
    remaining = timeout - target['audit_seconds']
    if remaining <= 0:
        raise ValueError(f'{target["source"]}: file audit deadline exhausted')
    report = replay_target(state, judge, target, inventory, remaining, heartbeat)
    if trust:
        for line in (target['logs'] / 'build.log').read_text().splitlines():
            for marker in ('Axiom audit:', 'Spec audit:'):
                if marker in line:
                    print(line[line.index(marker):], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=Path)
    group.add_argument('--library', action='store_true')
    group.add_argument('--check-tools', action='store_true')
    parser.add_argument('--trust', action='store_true', help='also run the standalone trust gates')
    parser.add_argument('--workspace', type=Path, help='share one snapshot within a batch')
    parser.add_argument('--timeout', type=positive_seconds, default=DEFAULT_TIMEOUT,
                        help='active compile/replay seconds per file (default: 1800)')
    parser.add_argument('--heartbeat', type=positive_seconds, default=30)
    args = parser.parse_args()
    if args.check_tools:
        require_tools()
        print('Official comparator and sandbox tools are available.')
    elif args.workspace:
        check(args.workspace.resolve(), args.file, args.trust, args.library, args.timeout, args.heartbeat)
    else:
        with tempfile.TemporaryDirectory(prefix='veritile-comparator-') as temp:
            check(Path(temp), args.file, args.trust, args.library, args.timeout, args.heartbeat)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        raise SystemExit(1)
