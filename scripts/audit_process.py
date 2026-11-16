"""Bounded audit subprocesses with live progress and complete timeout cleanup."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import threading
import sys
import uuid

DEFAULT_TIMEOUT = 1800.0
DEFAULT_HEARTBEAT = 30.0
_LOG_LOCK = threading.Lock()


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Audit timeouts and heartbeat intervals must be finite and positive')
    return value


def progress(event, *, source, stage, **fields):
    record = 'Audit ' + json.dumps({'event': event, 'source': str(source), 'stage': stage,
                                   **fields}, ensure_ascii=False) + '\n'
    with _LOG_LOCK:
        sys.stdout.write(record)
        sys.stdout.flush()


class AuditTimeout(ValueError):
    pass


def terminate(process, unit=None):
    # systemd comparator/exporter workers are outside the launcher's process
    # group. Kill that explicit service as well as Lake and all its children.
    if unit is not None:
        for action in (['kill', '--kill-whom=all', '--signal=KILL'], ['stop']):
            try:
                subprocess.run(['systemctl', '--user', *action, unit],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(command, *, cwd, log, source, stage, timeout=DEFAULT_TIMEOUT,
               heartbeat=DEFAULT_HEARTBEAT, unit=None, cancel=None):
    timeout, heartbeat = positive_seconds(timeout), positive_seconds(heartbeat)
    start = time.monotonic()
    deadline, next_heartbeat = start + timeout, start + heartbeat
    progress('begin', source=source, stage=stage, timeout_seconds=timeout, log=str(log))
    with Path(log).open('w') as output:
        process = subprocess.Popen(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while True:
                now = time.monotonic()
                if cancel is not None and cancel.is_set():
                    terminate(process, unit)
                    progress('cancelled', source=source, stage=stage, elapsed_seconds=now-start)
                    raise ValueError(f'{source}: {stage} cancelled after another audit failed; log: {log}')
                if now >= deadline:
                    terminate(process, unit)
                    elapsed = time.monotonic()-start
                    progress('timeout', source=source, stage=stage, elapsed_seconds=elapsed)
                    raise AuditTimeout(f'{source}: {stage} timed out after {timeout:g}s; log: {log}')
                try:
                    code = process.wait(timeout=min(deadline-now, max(.001, next_heartbeat-now), .25))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= next_heartbeat:
                        progress('running', source=source, stage=stage,
                                 elapsed_seconds=round(time.monotonic()-start, 3))
                        next_heartbeat = time.monotonic()+heartbeat
        except BaseException:
            if process.poll() is None:
                terminate(process, unit)
            raise
    elapsed = time.monotonic()-start
    progress('end', source=source, stage=stage, elapsed_seconds=round(elapsed, 3), exit_code=code)
    if code:
        for line in Path(log).read_text(errors='replace').splitlines()[-40:]:
            print(line[:2000], flush=True)
    return subprocess.CompletedProcess(command, code), elapsed


def service_unit():
    return 'veritile-proof-audit-' + uuid.uuid4().hex + '.service'
