"""Measure a Linux replay command, including its separate systemd services.

RSS is sampled, summed across processes, and can count shared pages more than
once. It is not a cgroup memory limit or an exact physical-memory high-water
mark. The private workspace also identifies comparator workers launched by the
user service manager, which are not descendants of the measured command.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def sample(root_pid, workspace):
    processes = {}
    selected = {root_pid}
    page_size = os.sysconf('SC_PAGE_SIZE')
    prefix = str(workspace.resolve()) + '/'
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            pid = int(entry.name)
            processes[pid] = (int(fields[1]), int(fields[21]) * page_size)
            cwd = os.readlink(entry / 'cwd')
            if cwd == prefix[:-1] or cwd.startswith(prefix):
                selected.add(pid)
        except (OSError, ValueError, IndexError):
            continue
    while True:
        children = {pid for pid, (parent, _) in processes.items() if parent in selected}
        if children <= selected:
            break
        selected.update(children)
    rss = {pid: processes[pid][1] for pid in selected if pid in processes}
    services = set()
    for pid in rss:
        try:
            group = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
        except OSError:
            continue
        for line in group:
            path = line.split(':', 2)[-1]
            if any(part.startswith('run-') and part.endswith('.service') for part in path.split('/')):
                services.add(path)
    return rss, services


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=0.1)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.interval <= 0:
        parser.error('provide a command and a positive sampling interval')
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'TMPDIR': str(args.workspace.resolve())}
    peak = 0
    peak_processes = {}
    services = set()
    gaps = []
    samples = 0
    start = previous = time.monotonic()
    with args.log.open('w') as log:
        process = subprocess.Popen(command, cwd=args.cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            rss, seen = sample(process.pid, args.workspace)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now
            samples += 1
            services.update(seen)
            if sum(rss.values()) > peak:
                peak = sum(rss.values())
                peak_processes = rss
            code = process.poll()
            if code is not None:
                break
            time.sleep(args.interval)
    result = {
        'command': command, 'cwd': str(args.cwd.resolve()), 'exit_code': code,
        'wall_seconds': time.monotonic() - start,
        'sampled_peak_total_rss_bytes': peak, 'peak_process_rss_bytes': peak_processes,
        'rss_method': 'sum of driver descendants and processes in the private judge workspace; includes separate systemd services; shared pages can be counted repeatedly',
        'requested_sampling_interval_seconds': args.interval,
        'maximum_sampling_gap_seconds': max(gaps), 'samples': samples,
        'observed_systemd_service_cgroups': sorted(services),
        'cpu_affinity_count': len(os.sched_getaffinity(0)),
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
