#!/usr/bin/env python3
"""Select complete changed cases, reverse dependents and owned proof dependencies.

Both sides of a Git diff contribute edges: deleting or renaming a shared source
must not erase its consumers from the PR plan. Missing history fails closed to
the complete verified corpus, never to an empty successful check.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

from scripts.bench.final_inventory import modules

MANIFEST = 'validation/body-migration-batch.json'
LEDGER = 'validation/task_progress.json'
OWNED = ('bench/tritonbench_g/', 'bench/examples/', 'tests/lean/')
PENDING = ('bench/bug_detection/', 'bench/pretrain_kernels/')
CONFIG = {'lean-toolchain', 'lakefile.toml', 'lake-manifest.json'}


def source_files(root: Path) -> dict[str, str]:
    files = {}
    for folder in ('bench', 'tests/lean', 'scripts'):
        for path in (root / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.lean', '.py', '.sh', '.json', '.patch'}:
                files[path.relative_to(root).as_posix()] = path.read_text()
    for suffix in ('*.lean', '*.py'):
        for path in root.glob(suffix):
            files[path.name] = path.read_text()
    for name in (MANIFEST, LEDGER, *CONFIG):
        path = root / name
        if path.is_file():
            files[name] = path.read_text()
    # Configuration and the plan producer/consumer also belong to the snapshot.
    for path in (root / '.github').rglob('*'):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_text()
    return files


def fingerprint(files: dict[str, str]) -> dict[str, str]:
    return {p: hashlib.sha256(s.encode()).hexdigest() for p, s in sorted(files.items())}


def git_changes(root: Path, base: str, head: str) -> tuple[set[str], dict[str, str] | None]:
    """Read NUL-delimited A/M/D/R/C records and the old source graph."""
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError('Expected a full Git head SHA')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if actual != head:
        raise ValueError('Checkout differs from the requested head')
    if not re.fullmatch(r'[0-9a-f]{40}', base) or base == '0' * 40:
        return set(), None
    try:
        diff = subprocess.check_output(['git', 'diff', '--name-status', '-z', '--find-renames',
                                        base, head, '--'], cwd=root, stderr=subprocess.PIPE)
        archive = subprocess.check_output(['git', 'archive', base], cwd=root, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return set(), None
    parts, changed, i = diff.decode().split('\0'), set(), 0
    while i < len(parts) and parts[i]:
        status = parts[i]
        count = 2 if status.startswith(('R', 'C')) else 1
        changed.update(parts[i + 1:i + 1 + count])
        i += 1 + count
    # Local invocation must not certify a diff while overlooking dirty sources.
    # In Actions these sets are normally empty; generated/ignored logs stay out.
    for command in (['git', 'diff', '--name-only', '-z', head, '--'],
                    ['git', 'ls-files', '--others', '--exclude-standard', '-z']):
        changed.update(p for p in subprocess.check_output(command, cwd=root).decode().split('\0') if p)
    old = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in tar:
            if entry.isfile() and Path(entry.name).suffix in {'.lean', '.py', '.json'}:
                old[entry.name] = tar.extractfile(entry).read().decode()
    return changed, old


@lru_cache(maxsize=32768)
def lean_dependencies(text: str) -> frozenset[str]:
    return frozenset(m.replace('.', '/') + '.lean' for m in modules(text))


@lru_cache(maxsize=4096)
def python_imports(path: str, text: str) -> tuple[str, ...]:
    names = []
    for node in ast.walk(ast.parse(text, filename=path)):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                parent = Path(path).parent.parts
                module = '.'.join((*parent[:len(parent) - node.level + 1], module)).rstrip('.')
            names.extend([module, *(module + '.' + a.name for a in node.names)])
    return tuple(names)


def dependencies(files: dict[str, str]) -> dict[str, set[str]]:
    """Local Lean imports and Python imports, including source-reference aliases."""
    graph = defaultdict(set)
    python_modules = defaultdict(set)
    for path in files:
        if path.endswith('.py'):
            python_modules[path[:-3].replace('/', '.')].add(path)
            python_modules[Path(path).stem].add(path)
            if path.endswith('/__init__.py'):
                python_modules[path[:-12].replace('/', '.')].add(path)
    for path, text in files.items():
        if path.endswith('source-references.json'):
            for alias, source in json.loads(text).items():
                python_modules[alias].add(source)
    for path, text in files.items():
        if path.endswith('.lean'):
            graph[path].update(lean_dependencies(text))
        elif path.endswith('.py'):
            for name in python_imports(path, text):
                graph[path].update(python_modules.get(name, ()))
                graph[path].update(python_modules.get(name.rsplit('.', 1)[-1], ()))
    return graph


def closure(seeds: set[str], edges: dict[str, set[str]]) -> set[str]:
    result, queue = set(seeds), list(seeds)
    while queue:
        for path in edges.get(queue.pop(), ()):
            if path not in result:
                result.add(path)
                queue.append(path)
    return result


def select(root: Path, changed: set[str], old: dict[str, str] | None,
           *, base: str = '', head: str = '', full: bool = False) -> dict:
    current = source_files(root)
    manifest = json.loads(current[MANIFEST])
    cases = {c['path']: c for c in manifest['cases']}
    if not cases or len(cases) != len(manifest['cases']):
        raise ValueError('Expected unique, nonempty verified cases')
    owner = lambda p: next((c for c in cases if p.startswith(c + '/')), None)
    broad, reasons = old is None, []
    if broad:
        reasons.append('base history unavailable: select every verified case and regression')
    graph = defaultdict(set)
    for side in (old or {}, current):
        for path, imports in dependencies(side).items():
            graph[path].update(imports)
    reverse = defaultdict(set)
    for path, imports in graph.items():
        for dep in imports:
            reverse[dep].add(path)
    seeds = {p for p in changed if Path(p).suffix in {'.py', '.lean', '.json'}
             and not p.startswith(('scripts/', 'validation/'))}
    auditor = full or old is None or any(
        p.startswith(('scripts/', '.github/')) or p in CONFIG for p in changed)
    for path in sorted(changed):
        if path in CONFIG or path.startswith('.github/actions/'):
            broad = True
            reasons.append('shared build/toolchain input: ' + path)
        elif path in (MANIFEST, LEDGER):
            key = 'cases' if path == MANIFEST else 'tasks'
            before = json.loads((old or {}).get(path, '{}'))
            after = json.loads(current[path])
            ident = (lambda row: row['path']) if key == 'cases' else (
                lambda row: 'bench/tritonbench_g/' + row['task'])
            a = {ident(row): row for row in before.get(key, [])}
            b = {ident(row): row for row in after[key]}
            seeds.update(c + '/Kernels.lean' for c in a.keys() | b.keys() if a.get(c) != b.get(c))
            if {k: v for k, v in before.items() if k != key} != {
                    k: v for k, v in after.items() if k != key}:
                # Accounting-only ledger changes do not alter proof selection.
                if path == MANIFEST:
                    broad = True
                    reasons.append('public-target policy changed')
        elif path.endswith('source-references.json'):
            broad = True
            reasons.append('shared Python source bindings changed')
        elif path.endswith('.lean') and not path.startswith((*OWNED, *PENDING)):
            broad = True
            reasons.append('shared Lean entry changed: ' + path)
        elif path.endswith('.py') and not path.startswith((*OWNED, *PENDING, 'scripts/', 'tests/')):
            broad = True
            reasons.append('shared Python source changed: ' + path)
    # Python-to-Lean correspondence is an ownership edge, not a Lean import.
    # Expand entire impacted cases before traversing their consumers.
    impact = closure(seeds, reverse)
    while True:
        owners = {c for p in impact if (c := owner(p))}
        expanded = closure(impact | {p for p in graph if owner(p) in owners}, reverse)
        if expanded == impact:
            break
        impact = expanded
    selected = set(cases) if broad else {c for p in impact if (c := owner(p))}
    # A case's Python source selects all of its proofs even without a Lean edge.
    owned = {p for p in current if p.endswith('.lean') and p.startswith(OWNED)}
    targets = {p for p in owned if owner(p) in selected or
               p.startswith('tests/lean/') and (broad or p in impact)}
    # Replay local dependencies too; include retained helpers not imported by headlines.
    targets = closure(targets, graph) & owned
    if full:
        selected, targets = set(), set()  # Daily shards cover the complete corpus.
    if targets:
        reasons.append('changed sources and reverse import consumers, with complete owned proof dependencies')
    return {'schema': 1, 'base': base, 'head': head, 'full_audit': str(full).lower(),
            'affected_required': str(bool(targets)).lower(),
            'auditor_required': str(bool(auditor)).lower(),
            'changed_files': sorted(changed), 'reasons': reasons,
            'cases': [cases[c] for c in sorted(selected)], 'targets': sorted(targets),
            'shard_count': min(64, max(1, (len(targets) + 127) // 128)),
            'source_sha256': fingerprint(current)}


def shard(plan: dict, index: int, count: int) -> dict:
    """Partition every module exactly once; keep each public goal with its file."""
    if count != plan['shard_count'] or not 0 <= index < count:
        raise ValueError('Affected shard configuration differs from the plan')
    # Sorted paths keep a case's proof files together. Round-robin file slices
    # spread nearly every case across workers and repeatedly build its closure.
    start = len(plan['targets']) * index // count
    stop = len(plan['targets']) * (index + 1) // count
    paths = plan['targets'][start:stop]
    selected = set(paths)
    cases = []
    for case in plan['cases']:
        public = {p: names for p, names in case['public_theorems'].items() if p in selected}
        if public:
            cases.append({**case, 'public_theorems': public})
    return {**plan, 'targets': paths, 'cases': cases,
            'shard': {'index': index, 'count': count}}


def validate(root: Path, plan: dict) -> None:
    if plan.get('schema') != 1 or plan.get('source_sha256') != fingerprint(source_files(root)):
        raise ValueError('Affected plan source snapshot differs from this checkout')
    if plan.get('head'):
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        if head != plan['head']:
            raise ValueError('Affected plan Git revision differs from this checkout')
    # Recompute selection, not just hashes: a truncated plan must fail closed.
    changed, old = git_changes(root, plan['base'], plan['head'])
    expected = select(root, changed, old, base=plan['base'], head=plan['head'],
                      full=plan['full_audit'] == 'true')
    if expected != plan:
        raise ValueError('Affected plan selection differs from the current Git diff')
