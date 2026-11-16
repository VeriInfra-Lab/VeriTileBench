"""Keep PR work bounded and require every scheduled audit job to succeed."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

from scripts.bench.ci import plan, require_results, require_replay

ROOT = Path(__file__).resolve().parents[1]


class BenchCITests(unittest.TestCase):
    def test_full_audit_only_runs_daily_or_when_explicitly_dispatched(self):
        for event in ['pull_request', 'push', 'schedule', 'workflow_dispatch']:
            for requested in ['', 'false', 'true']:
                with self.subTest(event=event, requested=requested):
                    p = plan(ROOT, event, requested)
                    full = event == 'schedule' or event == 'workflow_dispatch' and requested == 'true'
                    self.assertEqual(p['full_audit'], str(full).lower())
                    shards = json.loads(p['shards'])
                    self.assertEqual(shards, list(range(p['shard_count'])))
                    self.assertEqual(p['shard_count'], 64)

    def test_plan_reads_the_exact_dependency_lock(self):
        p = plan(ROOT, 'pull_request', '')
        manifest = json.loads((ROOT/'lake-manifest.json').read_text())
        package, = (p for p in manifest['packages'] if p['name'] == 'veritile')
        self.assertEqual(p['library_revision'], package['rev'])
        self.assertEqual(p['library_repository'], 'VeriInfra-Lab/VeriTile')
        self.assertEqual(p['library_path'], '.lake/packages/veritile')

    def test_completion_rejects_failure_cancellation_and_unexpected_skips(self):
        for full, affected, auditor in [('true', 'false', 'true'),
                                        ('false', 'true', 'false'),
                                        ('false', 'false', 'false'),
                                        ('false', 'false', 'true')]:
            env = {'FULL_AUDIT': full, 'AFFECTED_REQUIRED': affected,
                   'AUDITOR_REQUIRED': auditor, 'CHECKS_RESULT': 'success',
                   **{k: 'success' if flag == 'true' else 'skipped' for k, flag in [
                       ('CORPUS_RESULT', full), ('AFFECTED_RESULT', affected),
                       ('AUDITOR_RESULT', auditor)]}}
            require_results(env)
            for key in env:
                for wrong in ['', 'failure', 'cancelled', 'unknown',
                              'skipped' if env[key] == 'success' else 'success']:
                    with self.subTest(full=full, key=key, wrong=wrong):
                        with self.assertRaises(ValueError):
                            require_results({**env, key: wrong})

    def test_replay_requires_auxiliary_files_public_goals_and_negative_headline(self):
        import copy
        p = {'targets': ['a.lean', 'helper.lean'], 'cases': [
            {'public_theorems': {'a.lean': ['N.real', 'N.fp', 'N.disproved']}}]}
        report = {'accepted': True, 'all_theorems_axioms_checked': True,
                  'files': p['targets'], 'individual': [], 'batches': [{
                      'accepted': True, 'coverage_matches': True,
                      'all_theorems_axioms_checked': True,
                      'modules': {'a.lean': [], 'helper.lean': []},
                      'batch_targets': ['N.real', 'N.fp', 'N.disproved', 'N.aux']}]}
        require_replay(p, report)
        for mutation in [lambda r: r.update(accepted=False),
                         lambda r: r.update(files=['a.lean']),
                         lambda r: r['batches'][0].update(coverage_matches=False),
                         lambda r: r['batches'][0]['modules'].pop('helper.lean'),
                         lambda r: r['batches'][0]['batch_targets'].remove('N.disproved')]:
            bad = copy.deepcopy(report)
            mutation(bad)
            with self.assertRaises(ValueError):
                require_replay(p, bad)
        fallback = copy.deepcopy(report)
        fallback['batches'] = []
        fallback['individual'] = [
            {'source': 'a.lean', 'accepted': True, 'all_theorems_axioms_checked': True,
             'theorems': ['N.real', 'N.fp', 'N.disproved']},
            {'source': 'helper.lean', 'accepted': True,
             'all_theorems_axioms_checked': True, 'theorems': ['N.aux']}]
        require_replay(p, fallback)

    def test_plan_cli_outputs_match_the_job_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'github-output'
            result = subprocess.run([sys.executable, 'scripts/bench/ci.py', 'plan',
                                     '--event', 'pull_request', '--full-audit', 'true',
                                     '--github-output', str(output)], cwd=ROOT,
                                    capture_output=True, text=True, check=True)
            values = json.loads(result.stdout)
            self.assertEqual(values['full_audit'], 'false')
            self.assertEqual(dict(line.split('=', 1) for line in output.read_text().splitlines()),
                             {k: str(v) for k, v in values.items()})

    def test_invalid_ci_plan_is_rejected(self):
        for event, requested in [('pull_request_target', ''), ('push', 'yes')]:
            with self.assertRaises(ValueError):
                plan(ROOT, event, requested)
