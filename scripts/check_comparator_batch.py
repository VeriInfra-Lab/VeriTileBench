#!/usr/bin/env python3
"""Bounded batch replay with complete per-module trust audits.

Every per-file inventory is retained. Its target union is resolved in the
original imported environment, supplemented by every original-module theorem.
Lazy equation theorems can change module attribution under combined imports.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from check_comparator import ROOT, build_inventory, prepare, prepare_target, replay_target
from audit_process import DEFAULT_TIMEOUT, positive_seconds, progress, run_logged, service_unit
from comparator_common import comparator_command, write_config
from scripts.bench.audit_source import lean_code_mask


BATCH_INVENTORY = r'''
open Lean Elab Command in
run_cmd do
  let modules : Array Name := __MODULES__
  let requested : Array String := __TARGETS__
  let mut targets : Array Name := #[]
  let mut seen : NameSet := {}
  for text in requested do
    let some name := Syntax.decodeNameLit ("`" ++ text)
      | throwError "Cannot decode theorem name: {text}"
    if ((← getEnv).toKernelEnv.find? name).isNone then
      let realized ← liftCoreM <| realizeGlobalConstNoOverloadCore name
      unless realized == name do throwError "Theorem resolved to another name: {name}"
    unless seen.contains name do
      targets := targets.push name
      seen := seen.insert name
  let env ← getEnv
  let kernelEnv := env.toKernelEnv
  for module in modules do
    unless env.header.modules.any (·.module == module) do
      throwError "Batch module is not imported: {module}"
  let mut owned : Array (Name × Name) := #[]
  for (name, info) in kernelEnv.constants.toList do
    let some idx := env.getModuleIdxFor? name | continue
    let module := env.header.modules[idx.toNat]!.module
    unless modules.contains module do continue
    if let .thmInfo _ := info then
      owned := owned.push (module, name)
      unless seen.contains name do
        targets := targets.push name
        seen := seen.insert name
  targets := targets.qsort (·.toString < ·.toString)
  let mut aliases : Array String := #[]
  for i in [:targets.size] do
    let name := targets[i]!
    let some (.thmInfo info) := kernelEnv.find? name
      | throwError "Batch target is not a theorem: {name}"
    let wrapperName := Name.mkSimple s!"veritile_batch_target_{i}"
    if env.contains wrapperName then throwError "Reserved batch name: {wrapperName}"
    liftCoreM <| addDecl <| .thmDecl {
      name := wrapperName, levelParams := info.levelParams, type := info.type,
      value := mkConst name (info.levelParams.map Level.param)
    }
    aliases := aliases.push wrapperName.toString
  if aliases.isEmpty then
    let wrapperName := `veritile_batch_empty_module
    if env.contains wrapperName then throwError "Reserved batch name: {wrapperName}"
    liftCoreM <| addDecl <| .thmDecl {
      name := wrapperName, levelParams := [], type := mkConst ``True,
      value := mkConst ``True.intro
    }
    aliases := #[wrapperName.toString]
  let entries := modules.map fun module => Json.mkObj [
    ("module", toJson module.toString),
    ("theorems", toJson (((owned.filter (·.1 == module)).map (·.2.toString)).qsort (· < ·)))
  ]
  liftIO <| IO.FS.writeFile __REPORT__ (Json.mkObj [
    ("owned_modules", toJson entries), ("theorems", toJson (targets.map Name.toString)),
    ("aliases", toJson aliases)
  ]).compress
'''


def task_files(root, task, *, allow_individual=False):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', task):
        raise ValueError('Expected one TritonBench-G task name')
    directory = root / 'bench/tritonbench_g' / task
    if not directory.resolve().is_relative_to((root / 'bench/tritonbench_g').resolve()):
        raise ValueError('Task directory escapes the corpus')
    files = sorted(p.relative_to(root).as_posix() for p in directory.rglob('*.lean'))
    if not files:
        raise ValueError('No Lean modules for task: ' + task)
    for file in files:
        if not (root / file).resolve().is_relative_to(directory.resolve()):
            raise ValueError('Source escapes the selected task: ' + file)
        if any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', part)
               for part in file[:-5].split('/')):
            raise ValueError('Unsupported Lean module name: ' + file)
        if not allow_individual and re.search(r'^\s*module\s*$', lean_code_mask((root / file).read_text()), re.M):
            raise ValueError('Batch imports require ordinary modules; use individual replay: ' + file)
    return files


def canonical_name(name, copied_module, original_module):
    # Preserve the private counter and complete name. Erasing private names or
    # macro scopes would collapse distinct helpers and could conceal omissions.
    prefix = '_private.' + copied_module + '.'
    if name.startswith(prefix):
        return '_private.' + original_module + '.' + name[len(prefix):]
    return name


def require_coverage(reference, entries):
    discovered = {}
    for entry in entries:
        file = entry['module'].replace('.', '/') + '.lean'
        if file in discovered:
            raise ValueError('Duplicate batch module: ' + file)
        names = entry['theorems']
        if len(names) != len(set(names)):
            raise ValueError('Duplicate batch theorem: ' + file)
        discovered[file] = sorted(names)
    if set(reference) != set(discovered):
        raise ValueError('Batch module coverage differs: ' + json.dumps({
            'missing': sorted(set(reference) - set(discovered)),
            'extra': sorted(set(discovered) - set(reference))}))
    differences = {}
    for file, names in reference.items():
        if len(names) != len(set(names)):
            raise ValueError('Duplicate reference theorem: ' + file)
        missing = sorted(set(names) - set(discovered[file]))
        extra = sorted(set(discovered[file]) - set(names))
        if missing or extra:
            differences[file] = {'missing': missing, 'extra': extra}
    if differences:
        raise ValueError('Batch theorem coverage differs: ' + json.dumps(differences))
    return discovered


def require_batch_targets(reference, inventory):
    """Module attribution can move for lazily generated equation theorems."""
    owned = {}
    for entry in inventory['owned_modules']:
        file = entry['module'].replace('.', '/') + '.lean'
        if file in owned or len(entry['theorems']) != len(set(entry['theorems'])):
            raise ValueError('Duplicate original-module inventory: ' + file)
        owned[file] = entry['theorems']
    if set(owned) != set(reference):
        raise ValueError('Batch original-module membership differs')
    # Validate the per-file lists too; duplicate occurrences inside one file
    # must never disappear under a set union.
    require_coverage(reference, [{'module': f[:-5].replace('/', '.'), 'theorems': ns}
                                 for f, ns in reference.items()])
    requested = {n for ns in reference.values() for n in ns}
    original = {n for ns in owned.values() for n in ns}
    actual = inventory['theorems']
    if len(actual) != len(set(actual)) or set(actual) != requested | original:
        raise ValueError('Batch target union differs: ' + json.dumps({
            'missing': sorted((requested | original) - set(actual)),
            'extra': sorted(set(actual) - (requested | original))}))
    if (len(inventory['aliases']) != max(1, len(actual)) or
            len(set(inventory['aliases'])) != len(inventory['aliases'])):
        raise ValueError('Comparator aliases do not cover the unique batch targets')
    return sorted(original - requested)


def selected_files(root, files):
    if not files or len(set(files)) != len(files):
        raise ValueError('Selected files must be nonempty and unique')
    for file in files:
        if not isinstance(file, str):
            raise ValueError('Invalid selected Lean module: ' + str(file))
        path = root / file
        if (not file.endswith('.lean') or Path(file).is_absolute() or
                any(part in {'', '.', '..'} for part in file.split('/')) or
                not path.resolve().is_relative_to(root.resolve()) or not path.is_file()):
            raise ValueError('Invalid selected Lean module: ' + str(file))
    return list(files)


def require_sources(state, judge, files, task=None):
    selected_files(ROOT, files)
    selected_files(judge, files)
    if task is not None and (task_files(ROOT, task, allow_individual=True) != files or
                             task_files(judge, task, allow_individual=True) != files):
        raise ValueError('Task membership changed during batch audit')
    hashes = json.loads((Path(state['logs']) / 'inputs.json').read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed during batch audit: ' + name)
    judge_inputs = Path(state['judge_inputs'])
    if hashlib.sha256(judge_inputs.read_bytes()).hexdigest() != state['judge_inputs_sha256']:
        raise ValueError('Judge source fingerprint changed during batch audit')
    for name, expected in json.loads(judge_inputs.read_text()).items():
        if hashlib.sha256((judge / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Judge source changed during batch audit: ' + name)
    if hashlib.sha256(Path(state['comparator']).read_bytes()).hexdigest() != state['comparator_sha256']:
        raise ValueError('Comparator binary changed during batch audit')


def build_inventories(judge, targets, batch_size=2, timeout=DEFAULT_TIMEOUT, heartbeat=30):
    """Bound parallel audit compilers; each file owns its trace and deadline."""
    if batch_size < 1:
        raise ValueError('Audit batch size must be positive')
    inventories = []
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=batch_size) as workers:
        for start in range(0, len(targets), batch_size):
            wave = targets[start:start+batch_size]
            futures = {workers.submit(build_inventory, judge, target, timeout, heartbeat, cancel): i
                       for i, target in enumerate(wave)}
            found = {}
            try:
                for future in as_completed(futures):
                    found[futures[future]] = future.result()
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
            inventories.extend(found[i] for i in range(len(wave)))
            progress('wave-complete', source='audit-selection', stage='audit-compile',
                     completed=start+len(wave), total=len(targets))
    return inventories


def validate_audit(target, inventory):
    text = (target['logs'] / 'build.log').read_text()
    if ('Axiom audit:' not in text or 'Spec audit:' not in text or
            (not target.get('requires_individual') and
             f'Full theorem axiom audit: theorems={len(inventory["theorems"])}' not in text)):
        raise ValueError('Missing per-module trust audit: ' + target['source'])


def replay_batch(state, judge, files, targets, inventories, *, task=None,
                 timeout=DEFAULT_TIMEOUT, heartbeat=30):
    start = time.monotonic()
    label = task or files[0].split('/')[2]
    logs = Path(tempfile.mkdtemp(prefix='Batch_' + label + '-', dir=state['logs']))
    reference = {file: [canonical_name(name, target['module'], file[:-5].replace('/', '.'))
                        for name in inventory['theorems']]
                 for file, target, inventory in zip(files, targets, inventories, strict=True)}
    (logs / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    modules = [file[:-5].replace('/', '.') for file in files]
    metadata = logs / 'targets.json'
    source = ('import Lean\n' + ''.join('import ' + module + '\n' for module in modules) +
              BATCH_INVENTORY.replace('__MODULES__', '#[' + ', '.join(
                  json.dumps(module) + '.toName' for module in modules) + ']').replace(
                  '__REPORT__', json.dumps(str(metadata))).replace('__TARGETS__', '#[' +
                  ', '.join(json.dumps(name, ensure_ascii=False) for name in sorted({
                      n for names in reference.values() for n in names})) + ']'))
    tag = 'Batch_' + hashlib.sha256('\n'.join(files).encode()).hexdigest()[:24]
    module = 'VeriTileComparator.' + tag
    (judge / 'VeriTileComparator' / (tag + '.lean')).write_text(source)
    (logs / 'Source.lean').write_text(source)
    progress('batch-members', source=label, stage='batch-inventory', files=files)
    result, _ = run_logged(['lake', 'build', module], cwd=judge, log=logs / 'build.log',
        source=label, stage='batch-inventory', timeout=timeout, heartbeat=heartbeat)
    if result.returncode or not metadata.is_file():
        raise ValueError('Batch inventory build failed; logs: ' + str(logs))
    inventory = json.loads(metadata.read_text())
    additional = require_batch_targets(reference, inventory)
    config = write_config(judge, inventory['aliases'], challenge=module, solution=module,
                          filename=f'.lake/comparator/{tag}.json')
    shutil.copyfile(config, logs / 'config.json')
    replay_start = time.monotonic()
    remaining = timeout-(replay_start-start)
    if remaining <= 0:
        raise ValueError('Batch inventory exhausted the replay deadline: ' + label)
    unit = service_unit()
    result, _ = run_logged(comparator_command(judge, state['comparator'], config, unit=unit),
        cwd=judge, log=logs / 'comparator.log', source=label, stage='batch-replay',
        timeout=remaining, heartbeat=heartbeat, unit=unit)
    report = {**state, 'mode': 'task-batch-replay' if task else 'file-batch-replay', 'task': label,
              'module_count': len(files), 'theorem_count': sum(map(len, reference.values())),
              'unique_theorem_count': len(inventory['theorems']),
              'batch_targets': inventory['theorems'], 'owned_modules': inventory['owned_modules'],
              'additional_targets': additional, 'all_theorems_axioms_checked': True,
              'coverage_matches': True, 'modules': reference,
              'per_file_audits': [{'file': file, 'inventory': str(target['logs'] / 'targets.json'),
                                  'audit_log': str(target['logs'] / 'build.log')}
                                 for file, target in zip(files, targets, strict=True)],
              'comparator_exit': result.returncode, 'accepted': result.returncode == 0,
              'evidence_dir': str(logs), 'wall_seconds': time.monotonic()-start,
              'batch_inventory_seconds': replay_start-start,
              'comparator_seconds': time.monotonic()-replay_start}
    (logs / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if result.returncode:
        raise ValueError('Official comparator rejected batch; logs: ' + str(logs))
    print(f'Batch comparator: {len(files)} modules, {report["theorem_count"]} theorem occurrences, '
          f'{report["unique_theorem_count"]} unique targets accepted; logs: {logs}', flush=True)
    return report


def replay_groups(files, batch_size):
    if batch_size < 1:
        raise ValueError('Replay batch size must be positive')
    groups, current, key = [], [], None
    for file in files:
        parts = file.split('/')
        owner = tuple(parts[:2] if parts[:2] == ['tests', 'lean'] else parts[:3])
        if current and (owner != key or len(current) >= batch_size):
            groups.append(current); current = []
        current.append(file); key = owner
    if current:
        groups.append(current)
    return groups


def check_files(workspace, files, *, task=None, audit_batch_size=2, batch_size=32,
                file_timeout=DEFAULT_TIMEOUT, batch_timeout=DEFAULT_TIMEOUT,
                heartbeat=30, results=None):
    file_timeout, batch_timeout = positive_seconds(file_timeout), positive_seconds(batch_timeout)
    heartbeat = positive_seconds(heartbeat)
    if audit_batch_size < 1 or batch_size < 1:
        raise ValueError('Audit and replay batch sizes must be positive')
    start = time.monotonic()
    files = selected_files(ROOT, files)
    progress('selection', source=task or 'explicit-files', stage='snapshot', modules=len(files))
    state = prepare(workspace)
    judge = workspace / 'project'
    require_sources(state, judge, files, task)
    snapshot_done = time.monotonic()
    targets = [prepare_target(state, judge, ROOT / file, trust=True, all_axioms=True) for file in files]
    inventories = build_inventories(judge, targets, audit_batch_size, file_timeout, heartbeat)
    for target, inventory in zip(targets, inventories, strict=True):
        validate_audit(target, inventory)
    audit_done = time.monotonic()
    for target in targets:
        target['remaining_seconds'] = file_timeout-target.get('audit_seconds', 0)
    lookup = {file: (target, inventory) for file, target, inventory in zip(files, targets, inventories, strict=True)}
    individual, compatible = [], []
    for file in files:
        target, inventory = lookup[file]
        if (any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', part) for part in file[:-5].split('/')) or
                re.search(r'^\s*module\s*$', lean_code_mask((ROOT / file).read_text()), re.M) or
                any('._hyg' in name for name in inventory['theorems'])):
            individual.append(file)
            progress('individual-required', source=file, stage='selection',
                     reason='standalone path, module header or source-copy hygienic theorem names')
        else:
            compatible.append(file)
    groups = replay_groups(compatible, batch_size)
    accepted = []
    batches = []
    for group in groups:
        group_targets, group_inventories = zip(*(lookup[file] for file in group), strict=True)
        remaining = min(t['remaining_seconds'] for t in group_targets)
        group_start = time.monotonic()
        # A batch timeout or failed comparator is an audit failure. Only combined
        # import/coverage incompatibility may use the complete individual path.
        try:
            report = replay_batch(state, judge, group, group_targets, group_inventories,
                task=task, timeout=min(batch_timeout, remaining), heartbeat=heartbeat)
        except ValueError as error:
            if not str(error).startswith('Batch inventory build failed'):
                raise
            for target in group_targets:
                target['remaining_seconds'] -= time.monotonic()-group_start
            progress('individual-fallback', source=group[0], stage='batch-inventory',
                     files=group, reason=str(error))
            individual.extend(group)
        else:
            accepted.extend(group); batches.append(report)
            for file in group:
                print('  ok    ' + file, file=sys.stderr, flush=True)
    fallbacks = []
    for file in individual:
        target, inventory = lookup[file]
        report = replay_target(state, judge, target, inventory,
                               target['remaining_seconds'], heartbeat)
        accepted.append(file); fallbacks.append(report)
        print('  ok    ' + file, file=sys.stderr, flush=True)
    if len(accepted) != len(files) or set(accepted) != set(files):
        raise ValueError('Batch worker coverage differs from the explicit selection')
    require_sources(state, judge, files, task)
    logs = Path(tempfile.mkdtemp(prefix='Selection_', dir=state['logs']))
    report = {**state, 'mode': 'selected-files-batch-replay', 'accepted': True,
              'files': files, 'module_count': len(files),
              'theorem_occurrences': sum(len(i['theorems']) for i in inventories),
              'batches': batches, 'individual': fallbacks, 'all_theorems_axioms_checked': True,
              'snapshot_seconds': snapshot_done-start, 'audit_inventory_seconds': audit_done-snapshot_done,
              'audit_batch_size': audit_batch_size, 'audit_build_invocations': len(files),
              'wall_seconds': time.monotonic()-start, 'file_timeout_seconds': file_timeout,
              'batch_timeout_seconds': batch_timeout, 'evidence_dir': str(logs)}
    (logs / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if results:
        results.write_text(json.dumps(report, indent=2) + '\n')
    progress('accepted', source=task or 'explicit-files', stage='complete',
             modules=len(files), batches=len(batches), individual=len(fallbacks),
             elapsed_seconds=report['wall_seconds'], evidence_dir=str(logs))
    return report


def check_task(workspace, task, audit_batch_size=2, **kwargs):
    files = task_files(ROOT, task, allow_individual=True)
    return check_files(workspace, files, task=task, audit_batch_size=audit_batch_size,
                       batch_size=len(files), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--task')
    selection.add_argument('--files-from', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--audit-batch-size', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--file-timeout', type=positive_seconds, default=DEFAULT_TIMEOUT)
    parser.add_argument('--batch-timeout', type=positive_seconds, default=DEFAULT_TIMEOUT)
    parser.add_argument('--heartbeat', type=positive_seconds, default=30)
    args = parser.parse_args()
    files = (task_files(ROOT, args.task, allow_individual=True) if args.task else
             args.files_from.read_text().splitlines())
    options = dict(task=args.task, audit_batch_size=args.audit_batch_size,
                   batch_size=len(files) if args.task else args.batch_size,
                   file_timeout=args.file_timeout, batch_timeout=args.batch_timeout,
                   heartbeat=args.heartbeat, results=args.results)
    if args.workspace:
        check_files(args.workspace.resolve(), files, **options)
    else:
        with tempfile.TemporaryDirectory(prefix='veritile-batch-') as temp:
            check_files(Path(temp), files, **options)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print('[FAIL] ' + str(error), file=sys.stderr)
        raise SystemExit(1)
