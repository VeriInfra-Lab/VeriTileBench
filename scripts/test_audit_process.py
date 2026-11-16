"""Real wall-clock deadlines, live progress and descendant/service cleanup."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_process import AuditTimeout, positive_seconds, progress, run_logged


class AuditProcessTests(unittest.TestCase):
    def test_nonfinite_or_nonpositive_limits_fail_before_launch(self):
        for value in [0, -1, float('inf'), float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                positive_seconds(value)

    def test_parallel_progress_records_remain_one_json_record_per_line(self):
        class YieldingStream(io.StringIO):
            def write(self, text):
                super().write(text)
                time.sleep(.0001)
                return len(text)
        with contextlib.redirect_stdout(YieldingStream()) as stream:
            with ThreadPoolExecutor(max_workers=8) as workers:
                list(workers.map(lambda n: progress('running',source=f'{n}.lean',stage='fixture'),range(80)))
        rows=[json.loads(line.removeprefix('Audit ')) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows),80)
        self.assertEqual({row['source'] for row in rows},{f'{n}.lean' for n in range(80)})

    def test_live_heartbeat_and_complete_output(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as stream:
            log = Path(temp) / 'run.log'
            result, elapsed = run_logged([sys.executable, '-c', 'import time; print("payload"); time.sleep(.15)'],
                cwd=temp, log=log, source='example.lean', stage='fixture', timeout=3, heartbeat=.03)
            self.assertEqual(result.returncode, 0)
            self.assertGreater(elapsed, .1)
            self.assertIn('payload', log.read_text())
            rows = [json.loads(s.removeprefix('Audit ')) for s in stream.getvalue().splitlines()]
            self.assertEqual(rows[0]['event'], 'begin')
            self.assertTrue(any(r['event'] == 'running' for r in rows))
            self.assertEqual(rows[-1]['event'], 'end')
            self.assertTrue(all(r['source'] == 'example.lean' for r in rows))

    def test_timeout_kills_spawned_children_and_names_the_file(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            child = root / 'child.pid'
            source = ('import subprocess,sys,time; from pathlib import Path; '
                      'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                      f'Path({str(child)!r}).write_text(str(p.pid)); time.sleep(60)')
            start = time.monotonic()
            with self.assertRaisesRegex(AuditTimeout, 'slow.lean: audit-compile timed out'):
                run_logged([sys.executable, '-c', source], cwd=root, log=root/'run.log',
                           source='slow.lean', stage='audit-compile', timeout=.3, heartbeat=.03)
            self.assertLess(time.monotonic()-start, 3)
            pid = int(child.read_text())
            proc = Path(f'/proc/{pid}/stat')
            for _ in range(30):
                if not proc.exists() or proc.read_text().split()[2] == 'Z':
                    break
                time.sleep(.02)
            else:
                self.fail('descendant survived the audit timeout')

    def test_comparator_timeout_stops_the_named_systemd_service(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            with patch('audit_process.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as cleanup:
                with self.assertRaises(AuditTimeout):
                    run_logged([sys.executable, '-c', 'import time; time.sleep(60)'], cwd=temp,
                        log=Path(temp)/'run.log', source='example.lean', stage='replay',
                        timeout=.05, heartbeat=.01, unit='veritile-test.service')
            commands = [c.args[0] for c in cleanup.call_args_list]
            self.assertIn(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=KILL', 'veritile-test.service'], commands)
            self.assertIn(['systemctl', '--user', 'stop', 'veritile-test.service'], commands)


if __name__ == '__main__':
    unittest.main()
