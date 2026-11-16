#!/usr/bin/env python3
"""Reject missing, duplicate, unexpected, or malformed worker results."""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from collections import Counter
import json
from pathlib import Path
import re
import sys


def validate(expected: str, results: str) -> None:
    targets = expected.splitlines()
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("expected targets must be nonempty and unique")
    actual = []
    for line in results.splitlines():
        match = re.fullmatch(r"  (?:ok    |FAIL  )(\S+)(?:   \(prep error\))?", line)
        if not match:
            raise ValueError(f"malformed worker result: {line!r}")
        actual.append(match[1])
    if Counter(actual) != Counter(targets):
        missing = sorted((Counter(targets) - Counter(actual)).elements())
        extra = sorted((Counter(actual) - Counter(targets)).elements())
        raise ValueError(f"incomplete worker results: missing={missing}, extra={extra}")


def validate_batch(expected, report):
    files = expected.splitlines()
    if (not files or len(set(files)) != len(files) or
            report.get('mode') != 'selected-files-batch-replay' or report.get('accepted') is not True or
            report.get('all_theorems_axioms_checked') is not True or
            report.get('files') != files or report.get('module_count') != len(files)):
        raise ValueError('incomplete worker results: batch selection or acceptance differs')
    actual = []
    for group in report.get('batches', []):
        if (group.get('accepted') is not True or group.get('comparator_exit') != 0 or
                group.get('coverage_matches') is not True or group.get('all_theorems_axioms_checked') is not True or
                group.get('module_count') != len(group.get('modules', {}))):
            raise ValueError('incomplete worker results: unaccepted batch')
        actual.extend(group['modules'])
    for row in report.get('individual', []):
        if (row.get('mode') != 'source-replay' or row.get('accepted') is not True or
                row.get('comparator_exit') != 0 or row.get('all_theorems_axioms_checked') is not True):
            raise ValueError('incomplete worker results: unaccepted individual replay')
        actual.append(row['source'])
    results = '\n'.join('  ok    ' + file for file in actual)
    validate(expected, results)
    return results


if __name__ == "__main__":
    try:
        if len(sys.argv) == 4 and sys.argv[1] == '--batch-report':
            print(validate_batch(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))
        else:
            validate(*sys.argv[1:])
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
