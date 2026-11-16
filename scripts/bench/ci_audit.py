"""Select the CI audit scope and require every gate belonging to that scope."""
import argparse
import json
import os
from pathlib import Path


# GitHub cron uses UTC; these cover 04:00 Asia/Singapore without overlap.
DAILY_BUILD = '0 20 * * 0-5'
WEEKLY_REPLAY = '0 20 * * 6'
MODES = ('smoke', 'build', 'replay')


def select_mode(event, requested='', schedule=''):
    if event in ('pull_request', 'push'):
        return 'smoke'
    if event == 'schedule':
        if schedule == DAILY_BUILD:
            return 'build'
        if schedule == WEEKLY_REPLAY:
            return 'replay'
        raise ValueError('Unknown audit schedule: ' + schedule)
    if event == 'workflow_dispatch':
        mode = requested or 'build'
        if mode in MODES:
            return mode
        raise ValueError('Unknown audit mode: ' + mode)
    raise ValueError('Unknown audit event: ' + event)


def require_results(mode, global_result, drafts_result, corpus_result):
    if mode not in MODES:
        raise ValueError('Missing or unknown audit mode: ' + mode)
    expected = 'skipped' if mode == 'smoke' else 'success'
    for gate, actual, wanted in (
        ('source/regression', global_result, 'success'),
        ('imported interfaces', drafts_result, expected),
        ('corpus workers', corpus_result, expected),
    ):
        if actual != wanted:
            raise ValueError(f'{gate}: expected {wanted}, got {actual!r}')
    return '\n'.join([
        f'Bench checks passed ({mode}).',
        'Source and regression checks: passed.',
        'Full Lean build and 334 imported interfaces: ' +
        ('not requested.' if mode == 'smoke' else 'passed.'),
        'Independent comparator replay: ' +
        ('passed.' if mode == 'replay' else 'not requested.'),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('scope', 'complete'))
    args = parser.parse_args()
    if args.command == 'scope':
        mode = select_mode(os.environ['GITHUB_EVENT_NAME'],
                           os.environ.get('BENCH_AUDIT_MODE', ''),
                           os.environ.get('BENCH_AUDIT_SCHEDULE', ''))
        count = 64 if mode == 'replay' else 16
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
            print('mode=' + mode, file=output)
            print(f'count={count}', file=output)
            print('shards=' + json.dumps(list(range(count))), file=output)
        print(f'Requested scope: {mode}; corpus workers: {0 if mode == "smoke" else count}')
    else:
        summary = require_results(os.environ.get('BENCH_AUDIT_MODE', ''),
                                  os.environ.get('GLOBAL_RESULT', ''),
                                  os.environ.get('DRAFTS_RESULT', ''),
                                  os.environ.get('CORPUS_RESULT', ''))
        print(summary)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as output:
                print(summary.replace('\n', '\n\n'), file=output)


if __name__ == '__main__':
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error))
