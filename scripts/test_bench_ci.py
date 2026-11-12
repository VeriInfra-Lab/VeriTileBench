"""Audit cadence and required-gate regressions, without running proof replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench.ci_audit import DAILY_BUILD, WEEKLY_REPLAY, require_results, select_mode


class BenchCiTests(unittest.TestCase):
    def test_pr_and_push_keep_smoke_scope(self):
        for event in ('pull_request', 'push'):
            for requested in ('', 'build', 'replay'):
                self.assertEqual(select_mode(event, requested), 'smoke')

    def test_daily_build_and_weekly_replay_have_separate_schedules(self):
        self.assertEqual(select_mode('schedule', schedule=DAILY_BUILD), 'build')
        self.assertEqual(select_mode('schedule', schedule=WEEKLY_REPLAY), 'replay')
        # 20:00 UTC on Saturday is Sunday 04:00 in Singapore.
        self.assertEqual(DAILY_BUILD, '0 20 * * 0-5')
        self.assertEqual(WEEKLY_REPLAY, '0 20 * * 6')

    def test_manual_default_is_build_and_replay_is_explicit(self):
        self.assertEqual(select_mode('workflow_dispatch'), 'build')
        for mode in ('smoke', 'build', 'replay'):
            self.assertEqual(select_mode('workflow_dispatch', mode), mode)

    def test_unknown_inputs_do_not_silently_reduce_coverage(self):
        for event, requested, schedule in (
            ('workflow_dispatch', 'typo', ''),
            ('schedule', '', ''),
            ('schedule', '', '0 20 * * *'),
            ('unknown', '', ''),
        ):
            with self.subTest(event=event, requested=requested, schedule=schedule):
                with self.assertRaises(ValueError):
                    select_mode(event, requested, schedule)

    def test_full_build_is_not_reported_as_comparator_acceptance(self):
        summary = require_results('build', 'success', 'success', 'success')
        self.assertIn('Full Lean build and 334 imported interfaces: passed.', summary)
        self.assertIn('Independent comparator replay: not requested.', summary)

    def test_replay_requires_all_gates_and_reports_replay(self):
        summary = require_results('replay', 'success', 'success', 'success')
        self.assertIn('Independent comparator replay: passed.', summary)

    def test_full_modes_reject_missing_cancelled_skipped_or_failed_jobs(self):
        for mode in ('build', 'replay'):
            for index in range(3):
                for result in ('', 'skipped', 'cancelled', 'failure', 'timed_out'):
                    results = ['success'] * 3
                    results[index] = result
                    with self.subTest(mode=mode, index=index, result=result):
                        with self.assertRaises(ValueError):
                            require_results(mode, *results)

    def test_smoke_expects_successful_global_and_skipped_corpus_jobs(self):
        summary = require_results('smoke', 'success', 'skipped', 'skipped')
        self.assertIn('Independent comparator replay: not requested.', summary)
        for results in (('failure', 'skipped', 'skipped'),
                        ('success', '', 'skipped'),
                        ('success', 'skipped', 'cancelled'),
                        ('success', 'success', 'success')):
            with self.assertRaises(ValueError):
                require_results('smoke', *results)

    def test_missing_mode_cannot_complete(self):
        for mode in ('', 'typo'):
            with self.assertRaises(ValueError):
                require_results(mode, 'success', 'success', 'success')

    def test_scope_cli_emits_the_complete_worker_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            for schedule, mode, count in ((DAILY_BUILD, 'build', 16),
                                           (WEEKLY_REPLAY, 'replay', 64)):
                output.write_text('')
                env = {**os.environ, 'GITHUB_EVENT_NAME': 'schedule',
                       'GITHUB_OUTPUT': str(output), 'BENCH_AUDIT_SCHEDULE': schedule}
                result = subprocess.run([sys.executable, 'scripts/bench/ci_audit.py', 'scope'],
                                        cwd=ROOT, env=env, text=True, capture_output=True,
                                        timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = dict(line.split('=', 1) for line in output.read_text().splitlines())
                self.assertEqual(values['mode'], mode)
                self.assertEqual(int(values['count']), count)
                self.assertEqual(json.loads(values['shards']), list(range(count)))


if __name__ == '__main__':
    unittest.main()
