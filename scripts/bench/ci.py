#!/usr/bin/env python3
"""Shared local/CI entry points for source checks, affected proofs and daily audits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# One build command includes up to 32 targets and their cold dependencies.
# This budget is separate from each file/batch's 30-minute proof-replay limit.
BUILD_BATCH_TIMEOUT = 3600
TOOL_TESTS = [
    'scripts.test_audit_source', 'scripts.test_audit_shards',
    'scripts.test_comparator_preflight', 'scripts.test_audit_process',
    'scripts.test_spec_sheets', 'scripts.test_coverage_review',
    'scripts.test_bench_ci', 'scripts.test_bench_affected',
]
AUDITOR_TARGETS = [
    'VeriTile.Meta.StatementAudit', 'VeriTile.Meta.Disprove',
    'bench.examples.VectorAdd.Kernels',
    'bench.tritonbench_g.softmax_triton1.Kernels',
]


def plan(root: Path, event: str, requested: str) -> dict:
    if event not in {'pull_request', 'push', 'schedule', 'workflow_dispatch'}:
        raise ValueError('Unsupported CI event: ' + event)
    if requested not in {'', 'true', 'false'}:
        raise ValueError('full_audit must be true or false')
    manifest = json.loads((root / 'lake-manifest.json').read_text())
    package, = (p for p in manifest['packages'] if p['name'] == 'veritile')
    repository = re.fullmatch(r'https://github.com/([\w.-]+/[\w.-]+)\.git', package['url'])
    if repository is None or not re.fullmatch(r'[0-9a-f]{40}', package['rev']):
        raise ValueError('Expected a locked GitHub VeriTile source revision')
    if manifest['packagesDir'] != '.lake/packages':
        raise ValueError('Unexpected Lake package directory')
    full = event == 'schedule' or event == 'workflow_dispatch' and requested == 'true'
    return {'full_audit': str(full).lower(), 'shard_count': 64,
            'shards': json.dumps(list(range(64))),
            'library_repository': repository[1], 'library_revision': package['rev'],
            'library_path': '.lake/packages/veritile'}


def require_results(env: dict) -> None:
    for name in ['CHECKS_RESULT']:
        if env.get(name) != 'success':
            raise ValueError(name + ' did not succeed')
    full = env.get('FULL_AUDIT')
    if full not in {'true', 'false'}:
        raise ValueError('Missing or invalid full-audit plan')
    for flag, name in [('FULL_AUDIT', 'CORPUS_RESULT'),
                       ('AFFECTED_REQUIRED', 'AFFECTED_RESULT'),
                       ('AUDITOR_REQUIRED', 'AUDITOR_RESULT')]:
        if env.get(flag) not in {'true', 'false'}:
            raise ValueError('Missing or invalid plan: ' + flag)
        expected = 'success' if env[flag] == 'true' else 'skipped'
        if env.get(name) != expected:
            raise ValueError(name + ' must be ' + expected)
    if full == 'true' and (env['AUDITOR_REQUIRED'] != 'true' or env['AFFECTED_REQUIRED'] != 'false'):
        raise ValueError('Full audit must run auditor regressions and the complete corpus')


def require_replay(plan: dict, report: dict) -> None:
    """Require exact file coverage and every public goal in the theorem inventory."""
    if (report.get('accepted') is not True or report.get('all_theorems_axioms_checked') is not True
            or sorted(report.get('files', [])) != plan['targets']):
        raise ValueError('Affected replay did not cover every planned module')
    covered, names = [], set()
    for batch in report['batches']:
        if not all(batch.get(k) is True for k in ('accepted', 'coverage_matches', 'all_theorems_axioms_checked')):
            raise ValueError('Affected batch was not independently accepted')
        covered.extend(batch['modules'])
        names.update(batch['batch_targets'])
    for item in report['individual']:
        if item.get('accepted') is not True or item.get('all_theorems_axioms_checked') is not True:
            raise ValueError('Affected individual replay was not accepted')
        covered.append(item['source'])
        # Ordinary public names retain their full names in copied source modules.
        names.update(item['theorems'])
    if sorted(covered) != plan['targets']:
        raise ValueError('Affected batch/individual coverage differs from the plan')
    required = {n for c in plan['cases'] for ns in c['public_theorems'].values() for n in ns}
    if required - names:
        raise ValueError('Missing public theorem replay: ' + repr(sorted(required - names)))


def build_targets(data: dict, index: int = 0, count: int | None = None) -> list[str]:
    from scripts.bench.affected import shard
    part = shard(data, index, count if count is not None else data['shard_count'])
    if not part['targets']:
        raise ValueError('No affected build targets')
    return ['VeriTile', 'VeriTile.Meta.StatementAudit', 'VeriTile.Meta.Disprove',
            *(p[:-5].replace('/', '.') for p in part['targets'])]


def verify_affected(plan_path: Path, results: Path, workspace: Path, index: int = 0,
                    count: int | None = None) -> None:
    from scripts.bench.affected import validate, shard
    complete_plan = json.loads(plan_path.read_text())
    validate(ROOT, complete_plan)
    data = shard(complete_plan, index, count if count is not None else complete_plan['shard_count'])
    if data['affected_required'] != 'true' or not data['targets']:
        raise ValueError('This plan does not request affected proof replay')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.bench.replay_body_migrations import verify_public_targets
    from scripts.check_comparator_batch import check_files
    from scripts.audit_process import run_logged
    # Bounded command lines and build deadlines also cover large dependency bumps.
    logs = workspace / 'build-logs'
    logs.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(data['targets']), 32):
        targets = [p[:-5].replace('/', '.') for p in data['targets'][i:i + 32]]
        result, _ = run_logged(['lake', 'build', *targets], cwd=ROOT,
                              log=logs / f'{i // 32}.log', source='affected-proofs',
                              stage='build', timeout=BUILD_BATCH_TIMEOUT, heartbeat=30)
        if result.returncode:
            raise ValueError('Affected proof build failed: ' + str(logs))
    if data['cases']:
        verify_public_targets(ROOT, data['cases'])
    validate(ROOT, complete_plan)
    report = check_files(workspace / 'replay', data['targets'], results=workspace / 'replay.json')
    require_replay(data, report)
    validate(ROOT, complete_plan)
    receipt = {'schema': 1, 'scope': 'affected-proofs', 'accepted': True,
               'plan': data, 'replay': report,
               'boundary': 'Accepts this shard only. Every planned shard must succeed to accept the selection.'}
    results.write_text(json.dumps(receipt, indent=2) + '\n')


def output(values: dict, path: Path | None) -> None:
    if path:
        with path.open('a') as stream:
            for key, value in values.items():
                print(f'{key}={value}', file=stream)
    print(json.dumps(values))


def run(command: list[str]) -> None:
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def audit_regressions() -> None:
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.comparator_common import require_tools
    require_tools()
    # Lean imports in the test fixtures need built oleans, including in a fresh
    # checkout. Library-only bootstrap does not build local benchmark modules.
    run(['lake', 'build', *AUDITOR_TARGETS])
    run([sys.executable, '-B', '-m', 'unittest', 'scripts.test_audit_gates',
         'scripts.test_check_comparator'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--event', required=True)
    p.add_argument('--full-audit', default='')
    p.add_argument('--github-output', type=Path)
    p = sub.add_parser('affected')
    p.add_argument('--base', default='')
    p.add_argument('--head', required=True)
    p.add_argument('--full-audit', choices=['true', 'false'], default='false')
    p.add_argument('--write', type=Path, required=True)
    p.add_argument('--github-output', type=Path)
    p = sub.add_parser('verify-affected')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int)
    p = sub.add_parser('build-targets')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int)
    p.add_argument('--github-output', type=Path)
    for name in ['checks', 'auditor', 'complete']:
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == 'plan':
        values = plan(ROOT, args.event, args.full_audit)
        output(values, args.github_output)
    elif args.command == 'affected':
        from scripts.bench.affected import git_changes, select
        changed, old = git_changes(ROOT, args.base, args.head)
        data = select(ROOT, changed, old, base=args.base, head=args.head, full=args.full_audit == 'true')
        args.write.write_text(json.dumps(data, indent=2) + '\n')
        values = {k: data[k] for k in ['affected_required', 'auditor_required']}
        values['affected_shards'] = json.dumps(list(range(data['shard_count'])))
        output(values, args.github_output)
        print(f"Selected {len(data['cases'])} cases and {len(data['targets'])} modules", file=sys.stderr)
    elif args.command == 'verify-affected':
        verify_affected(args.plan, args.results, args.workspace, args.shard_index, args.shard_count)
    elif args.command == 'build-targets':
        from scripts.bench.affected import validate
        data = json.loads(args.plan.read_text())
        validate(ROOT, data)
        output({'build_args': ' '.join(build_targets(data, args.shard_index, args.shard_count))},
               args.github_output)
    elif args.command == 'checks':
        run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/python', '-p', 'test_*.py'])
        run([sys.executable, '-B', '-m', 'unittest', *TOOL_TESTS])
        run(['bash', 'scripts/bench/audit_tritonbench_g.sh', '--source-only'])
    elif args.command == 'auditor':
        audit_regressions()
    else:
        require_results(os.environ)
        print('All required benchmark CI gates passed')


if __name__ == '__main__':
    main()
