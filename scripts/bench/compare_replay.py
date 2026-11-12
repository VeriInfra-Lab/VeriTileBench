"""Compare successful per-file and batch runs over exactly the same sources."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_comparator_batch import canonical_name, require_coverage


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def audit_summaries(path):
    result = []
    for line in path.read_text().splitlines():
        for marker in ('Axiom audit:', 'Spec audit:'):
            if marker in line:
                result.append(line[line.index(marker):])
    if not any(x.startswith('Axiom audit:') for x in result) or not any(
            x.startswith('Spec audit:') for x in result):
        raise ValueError('Missing per-file trust audit: ' + str(path))
    return result


def compare(baseline, batch_file, old_metrics, new_metrics):
    batch = json.loads(batch_file.read_text())
    if not batch.get('accepted') or batch.get('comparator_exit') != 0:
        raise ValueError('Batch was not accepted by the official comparator')
    old_inputs = json.loads((baseline / 'inputs.json').read_text())
    new_inputs = json.loads((batch_file.parent.parent / 'inputs.json').read_text())
    if old_inputs != new_inputs:
        raise ValueError('Old and new proof inputs differ')
    old = {}
    old_audits = {}
    old_tool_hashes = set()
    for path in sorted(baseline.glob('*/result.json')):
        row = json.loads(path.read_text())
        if row.get('mode') != 'source-replay' or not row.get('accepted') or row.get('comparator_exit') != 0:
            raise ValueError('Baseline contains an unaccepted replay: ' + str(path))
        file = row['source']
        if file in old:
            raise ValueError('Duplicate baseline file: ' + file)
        names = row['theorems']
        if row['theorem_count'] != len(names):
            raise ValueError('Baseline theorem count differs: ' + file)
        copied = 'VeriTileComparator.File_' + hashlib.sha256((file + 'True').encode()).hexdigest()[:24]
        module = file[:-5].replace('/', '.')
        old[file] = [canonical_name(name, copied, module) for name in names]
        old_audits[file] = audit_summaries(path.parent / 'build.log')
        inventory = json.loads((path.parent / 'targets.json').read_text())
        if inventory['theorems'] != names:
            raise ValueError('Baseline receipt and inventory differ: ' + file)
        old_tool_hashes.add(row['comparator_sha256'])
    if old_tool_hashes != {batch['comparator_sha256']}:
        raise ValueError('Comparator binary differs between runs')
    canonical = require_coverage(old, [
        {'module': file[:-5].replace('/', '.'), 'theorems': names}
        for file, names in batch['modules'].items()])
    if batch['module_count'] != len(canonical) or batch['theorem_count'] != sum(map(len, canonical.values())):
        raise ValueError('Batch receipt counts differ from its inventory')
    audits = batch['per_file_audits']
    if len(audits) != len(canonical) or {a['file'] for a in audits} != set(canonical):
        raise ValueError('Batch did not retain every per-file trust audit')
    for audit in audits:
        if audit_summaries(Path(audit['audit_log'])) != old_audits[audit['file']]:
            raise ValueError('Old and new trust audits differ: ' + audit['file'])
    for metrics, invocations in ((old_metrics, len(canonical)), (new_metrics, 1)):
        if metrics['exit_code'] != 0:
            raise ValueError('Measured command failed')
        if len(metrics['observed_systemd_service_cgroups']) < invocations:
            raise ValueError('RSS sampling did not observe every comparator service')
    if (old_metrics['rss_method'] != new_metrics['rss_method'] or
            old_metrics['requested_sampling_interval_seconds'] != new_metrics['requested_sampling_interval_seconds']):
        raise ValueError('Memory measurement methods differ')
    total = sum(map(len, canonical.values()))
    summary = {
        'schema': 1, 'task': batch['task'], 'proof_inputs_identical': True,
        'proof_inputs_sha256': digest(old_inputs), 'inventory_sha256': digest(canonical),
        'module_count': len(canonical), 'theorem_count': total,
        'private_theorem_count': sum(n.startswith('_private.') for ns in canonical.values() for n in ns),
        'proofs_directory_theorem_count': sum(len(ns) for f, ns in canonical.items() if '/Proofs/' in f),
        'missing_theorems': [], 'extra_theorems': [], 'per_file_trust_audits_passed': len(audits),
        'per_file_trust_audit_summaries_identical': True,
        'audit_summaries_sha256': digest(old_audits),
        'baseline_accepted': True, 'batch_accepted': True,
        'comparator_sha256': batch['comparator_sha256'],
        'old': {'comparator_invocations': len(canonical), 'wall_seconds': old_metrics['wall_seconds'],
                'sampled_peak_total_rss_bytes': old_metrics['sampled_peak_total_rss_bytes']},
        'new': {'comparator_invocations': 1, 'wall_seconds': new_metrics['wall_seconds'],
                'sampled_peak_total_rss_bytes': new_metrics['sampled_peak_total_rss_bytes'],
                'audit_inventory_seconds': batch['audit_inventory_seconds'],
                'batch_inventory_seconds': batch['batch_inventory_seconds'],
                'comparator_seconds': batch['comparator_seconds']},
        'wall_speedup': old_metrics['wall_seconds'] / new_metrics['wall_seconds'],
        'measurement': {'rss_method': old_metrics['rss_method'],
                        'sampling_interval_seconds': old_metrics['requested_sampling_interval_seconds'],
                        'old_maximum_sampling_gap_seconds': old_metrics['maximum_sampling_gap_seconds'],
                        'new_maximum_sampling_gap_seconds': new_metrics['maximum_sampling_gap_seconds'],
                        'cpu_affinity_count': old_metrics['cpu_affinity_count'],
                        'trials': 1, 'parallel_trials': False,
                        'cache': 'Both runs start with copies of the same built dependency cache; fresh independent judge workspaces. Wall time includes snapshot preparation and cleanup.',
                        'boundary': 'Local-host measurement; sampled aggregate RSS can count shared pages repeatedly. Not a GitHub runner capacity or speed guarantee.'},
    }
    return summary, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--old-metrics', type=Path, required=True)
    parser.add_argument('--new-metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    summary, inventory = compare(args.baseline, args.batch,
                                 json.loads(args.old_metrics.read_text()),
                                 json.loads(args.new_metrics.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    args.inventory.write_bytes(gzip.compress(json.dumps(inventory, sort_keys=True,
                                                        ensure_ascii=False).encode(), mtime=0))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
