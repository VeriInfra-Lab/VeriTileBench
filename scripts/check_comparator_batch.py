#!/usr/bin/env python3
"""Opt-in task batch replay; retain the existing per-module trust audits.

Every per-file inventory is retained. Its target union is resolved in the
original imported environment, supplemented by every original-module theorem.
Lazy equation theorems can change module attribution under combined imports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from check_comparator import ROOT, build_inventory, prepare, prepare_target
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


def task_files(root, task):
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
        if re.search(r'^\s*module\s*$', lean_code_mask((root / file).read_text()), re.M):
            raise ValueError('Batch pilot requires ordinary modules; use individual replay: ' + file)
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


def require_sources(state, judge, task, files):
    if task_files(ROOT, task) != files or task_files(judge, task) != files:
        raise ValueError('Task membership changed during batch audit')
    hashes = json.loads((Path(state['logs']) / 'inputs.json').read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed during batch audit: ' + name)


def check_task(workspace, task):
    start = time.monotonic()
    files = task_files(ROOT, task)
    state = prepare(workspace)
    judge = workspace / 'project'
    require_sources(state, judge, task, files)
    logs = Path(tempfile.mkdtemp(prefix='Batch_' + task + '-', dir=state['logs']))
    reference = {}
    per_file = []
    for file in files:
        target = prepare_target(state, judge, ROOT / file, trust=True, all_axioms=True)
        inventory = build_inventory(judge, target)
        if any('._hyg' in name for name in inventory['theorems']):
            raise ValueError('Hygienic source-copy targets require individual replay in this pilot: ' + file)
        audit_log = (target['logs'] / 'build.log').read_text()
        if ('Axiom audit:' not in audit_log or 'Spec audit:' not in audit_log or
                f'Full theorem axiom audit: theorems={len(inventory["theorems"])}' not in audit_log):
            raise ValueError('Missing per-module trust audit: ' + file)
        original_module = file[:-5].replace('/', '.')
        reference[file] = [canonical_name(name, target['module'], original_module)
                           for name in inventory['theorems']]
        per_file.append({'file': file, 'inventory': str(target['logs'] / 'targets.json'),
                         'audit_log': str(target['logs'] / 'build.log')})
        print(f'Audit inventory: {file}: {len(reference[file])} theorem(s)', flush=True)
    (logs / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    audit_done = time.monotonic()
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
    with (logs / 'build.log').open('w') as output:
        result = subprocess.run(['lake', 'build', module], cwd=judge,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode or not metadata.is_file():
        print((logs / 'build.log').read_text(), file=sys.stderr)
        raise ValueError('Batch inventory build failed; logs: ' + str(logs))
    inventory = json.loads(metadata.read_text())
    additional = require_batch_targets(reference, inventory)
    discovered = {file: sorted(names) for file, names in reference.items()}
    require_sources(state, judge, task, files)
    config = write_config(judge, inventory['aliases'], challenge=module, solution=module,
                          filename=f'.lake/comparator/{tag}.json')
    shutil.copyfile(config, logs / 'config.json')
    replay_start = time.monotonic()
    with (logs / 'comparator.log').open('w') as output:
        result = subprocess.run(comparator_command(judge, state['comparator'], config),
                                stdout=output, stderr=subprocess.STDOUT)
    require_sources(state, judge, task, files)
    report = {**state, 'mode': 'task-batch-replay', 'task': task,
              'module_count': len(files), 'theorem_count': sum(map(len, discovered.values())),
              'unique_theorem_count': len(inventory['theorems']),
              'batch_targets': inventory['theorems'], 'owned_modules': inventory['owned_modules'],
              'additional_targets': additional, 'all_theorems_axioms_checked': True,
              'coverage_matches': True, 'modules': discovered, 'per_file_audits': per_file,
              'comparator_exit': result.returncode, 'accepted': result.returncode == 0,
              'evidence_dir': str(logs), 'wall_seconds': time.monotonic() - start,
              'audit_inventory_seconds': audit_done - start,
              'batch_inventory_seconds': replay_start - audit_done,
              'comparator_seconds': time.monotonic() - replay_start}
    (logs / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if result.returncode:
        print((logs / 'comparator.log').read_text(), file=sys.stderr)
        raise ValueError('Official comparator rejected batch; logs: ' + str(logs))
    print(f'Batch comparator: {len(files)} modules, {report["theorem_count"]} theorem occurrences, '
          f'{report["unique_theorem_count"]} unique targets accepted; logs: {logs}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--workspace', type=Path)
    args = parser.parse_args()
    if args.workspace:
        check_task(args.workspace.resolve(), args.task)
    else:
        with tempfile.TemporaryDirectory(prefix='veritile-batch-') as temp:
            check_task(Path(temp), args.task)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print('[FAIL] ' + str(error), file=sys.stderr)
        raise SystemExit(1)
