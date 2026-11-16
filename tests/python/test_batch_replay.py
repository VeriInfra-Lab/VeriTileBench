"""Require exact module/theorem coverage before batch proof acceptance."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_comparator_batch import build_inventories, canonical_name, require_batch_targets, require_coverage, task_files, selected_files, replay_groups, validate_audit
from scripts.bench.compare_replay import compare, compare_previous_batch


class BatchCoverageTests(unittest.TestCase):
    def setUp(self):
        self.file = 'bench/tritonbench_g/fixture/Proofs/Unused.lean'
        self.module = self.file[:-5].replace('/', '.')
        self.private = '_private.' + self.module + '.0.hidden'
        self.reference = {self.file: [self.private, 'helper']}

    def test_complete_inventory_preserves_private_and_unused_theorems(self):
        result = require_coverage(self.reference, [
            {'module': self.module, 'theorems': ['helper', self.private]}])
        self.assertEqual(result[self.file], sorted([self.private, 'helper']))

    def test_missing_private_theorem_is_a_failure(self):
        with self.assertRaisesRegex(ValueError, 'theorem coverage differs'):
            require_coverage(self.reference, [{'module': self.module, 'theorems': ['helper']}])

    def test_extra_or_duplicate_theorems_are_a_failure(self):
        for names in ([self.private, 'helper', 'extra'], [self.private, 'helper', 'helper']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                require_coverage(self.reference, [{'module': self.module, 'theorems': names}])

    def test_missing_extra_or_duplicate_modules_are_a_failure(self):
        entry = {'module': self.module, 'theorems': [self.private, 'helper']}
        for entries in ([], [entry, entry], [entry, {'module': 'Unexpected', 'theorems': []}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                require_coverage(self.reference, entries)

    def test_zero_theorem_modules_still_require_coverage(self):
        result = require_coverage({self.file: []}, [{'module': self.module, 'theorems': []}])
        self.assertEqual(result, {self.file: []})
        with self.assertRaises(ValueError):
            require_coverage({self.file: []}, [])

    def test_private_normalization_retains_distinct_counters_and_names(self):
        copied = 'VeriTileComparator.File_123'
        for counter in ('0', '1'):
            name = f'_private.{copied}.{counter}.hidden'
            self.assertEqual(canonical_name(name, copied, self.module),
                             f'_private.{self.module}.{counter}.hidden')
        self.assertEqual(canonical_name('publicName', copied, self.module), 'publicName')

    def test_task_discovery_includes_unimported_helpers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for file in ('RealCorrect.lean', 'FPEquiv.lean', 'Proofs/Unused.lean'):
                path = root / 'bench/tritonbench_g/fixture' / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            files = task_files(root, 'fixture')
            self.assertEqual(len(files), 3)
            self.assertIn(self.file, files)
            for invalid in ('../fixture', '', 'missing'):
                with self.assertRaises(ValueError):
                    task_files(root, invalid)

    def test_escaping_source_is_a_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'bench/tritonbench_g/fixture'
            folder.mkdir(parents=True)
            outside = root / 'outside.lean'
            outside.touch()
            (folder / 'Escape.lean').symlink_to(outside)
            # The file is inside the repository, but outside its declared task.
            with self.assertRaises(ValueError):
                task_files(root, 'fixture')

    def test_module_header_requires_the_existing_individual_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'bench/tritonbench_g/fixture/Settings.lean'
            path.parent.mkdir(parents=True)
            path.write_text('module\npublic import Init\n')
            with self.assertRaisesRegex(ValueError, 'use individual replay'):
                task_files(root, 'fixture')

    def test_lazy_equations_can_move_module_attribution_without_being_dropped(self):
        other = 'bench/tritonbench_g/fixture/Other.lean'
        reference = {self.file: ['local_helper', 'imported.eq_1'], other: ['other_helper']}
        inventory = {'owned_modules': [
            {'module': self.module, 'theorems': ['local_helper']},
            {'module': other[:-5].replace('/', '.'), 'theorems': ['other_helper', 'imported.eq_1']}],
            'theorems': ['local_helper', 'other_helper', 'imported.eq_1'],
            'aliases': ['target0', 'target1', 'target2']}
        self.assertEqual(require_batch_targets(reference, inventory), [])
        inventory['theorems'].remove('imported.eq_1')
        with self.assertRaisesRegex(ValueError, 'target union differs'):
            require_batch_targets(reference, inventory)

    def test_original_module_additional_theorems_must_also_be_replayed(self):
        inventory = {'owned_modules': [{'module': self.module, 'theorems': ['extra']}],
                     'theorems': ['helper', self.private], 'aliases': ['target0', 'target1']}
        with self.assertRaises(ValueError):
            require_batch_targets(self.reference, inventory)
        inventory['theorems'].append('extra')
        inventory['aliases'].append('target2')
        self.assertEqual(require_batch_targets(self.reference, inventory), ['extra'])


class AuditWaveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.targets = []
        for i in range(5):
            module = f'VeriTileComparator.Case{i}'
            logs = self.root / 'logs' / str(i)
            logs.mkdir(parents=True)
            metadata = logs / 'metadata.json'
            metadata.write_text(json.dumps({'theorems': [f'theorem{i}']}))
            trace = self.root / '.lake/build/lib/lean/VeriTileComparator' / f'Case{i}.trace'
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({'log': [{'level': 'info', 'message': f'audit of module {i}'}]}))
            self.targets.append({'module': module, 'source': f'Case{i}.lean',
                                 'logs': logs, 'metadata': metadata})

    def test_parallel_audits_preserve_order_and_individual_limits(self):
        def inventory(judge, target, timeout, heartbeat, cancel):
            return json.loads(target['metadata'].read_text())
        with patch('check_comparator_batch.build_inventory', side_effect=inventory) as run:
            rows = build_inventories(self.root, self.targets, 2, timeout=12, heartbeat=1)
        self.assertEqual([r['theorems'] for r in rows], [[f'theorem{i}'] for i in range(5)])
        self.assertEqual(run.call_count, 5)
        self.assertTrue(all(call.args[2:4] == (12, 1) for call in run.call_args_list))

    def test_a_failed_file_stops_future_waves(self):
        with patch('check_comparator_batch.build_inventory', side_effect=ValueError('failed file')) as run:
            with self.assertRaisesRegex(ValueError, 'failed file'):
                build_inventories(self.root, self.targets, 2)
        self.assertLessEqual(run.call_count, 2)

    def test_invalid_batch_size_is_rejected(self):
        for size in (0, -1):
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                build_inventories(self.root, self.targets, size)

    def test_missing_own_trace_or_trace_error_cannot_pass(self):
        from check_comparator import build_inventory
        target=self.targets[0]
        trace=self.root/'.lake/build/lib/lean/VeriTileComparator/Case0.trace'
        with patch('check_comparator.run_logged', return_value=(SimpleNamespace(returncode=0),.1)):
            trace.unlink()
            with self.assertRaisesRegex(ValueError,'Missing per-module audit trace'):
                build_inventory(self.root,target)
            trace.write_text(json.dumps({'log':[{'level':'error','message':'bad proof'}]}))
            with self.assertRaisesRegex(ValueError,'compiler trace contains an error'):
                build_inventory(self.root,target)

    def test_failed_compiler_cannot_reuse_stale_inventory(self):
        from check_comparator import build_inventory
        target=self.targets[0]
        (target['logs']/'build-all.log').write_text('failed compiler')
        with patch('check_comparator.run_logged', return_value=(SimpleNamespace(returncode=1),.1)):
            with self.assertRaisesRegex(ValueError,'elaboration/inventory failed'):
                build_inventory(self.root,target)

    def test_per_file_trust_markers_are_mandatory(self):
        target=self.targets[0]
        (target['logs']/'build.log').write_text('Axiom audit: ok\nSpec audit: ok\n')
        with self.assertRaisesRegex(ValueError,'Missing per-module trust audit'):
            validate_audit(target,{'theorems':['a']})


class SelectionTests(unittest.TestCase):
    def test_selected_paths_reject_duplicates_missing_files_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'Safe.lean').touch()
            self.assertEqual(selected_files(root,['Safe.lean']),['Safe.lean'])
            for files in [[],['Safe.lean']*2,['missing.lean'],['../Safe.lean'],[str(root/'Safe.lean')]]:
                with self.subTest(files=files),self.assertRaises(ValueError):
                    selected_files(root,files)

    def test_groups_preserve_exact_selection_and_bound_each_owner(self):
        files=['bench/tritonbench_g/a/F1.lean','bench/tritonbench_g/a/F2.lean',
               'bench/tritonbench_g/a/F3.lean','bench/tritonbench_g/b/F1.lean',
               'tests/lean/F1.lean','tests/lean/F2.lean']
        groups=replay_groups(files,2)
        self.assertEqual([f for group in groups for f in group],files)
        self.assertTrue(all(len(g)<=2 for g in groups))
        self.assertEqual(groups[-1],files[-2:])


class ReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.baseline = root / 'old'
        old = self.baseline / 'File_case'
        new = root / 'new/Batch_case'
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        self.batch_file = new / 'result.json'
        file = 'bench/tritonbench_g/fixture/Proofs/Unused.lean'
        module = file[:-5].replace('/', '.')
        copied = 'VeriTileComparator.File_' + hashlib.sha256((file + 'True').encode()).hexdigest()[:24]
        names = ['helper', '_private.' + copied + '.0.hidden']
        self.old_row = {'mode': 'source-replay', 'accepted': True, 'comparator_exit': 0,
                        'source': file, 'theorems': names, 'theorem_count': 2,
                        'comparator_sha256': 'a' * 64}
        self.old_file = old / 'result.json'
        self.old_file.write_text(json.dumps(self.old_row))
        (old / 'targets.json').write_text(json.dumps({'theorems': names}))
        audit = 'Axiom audit: headlines=0\nSpec audit: kernels=0\n'
        (old / 'build.log').write_text(audit)
        (new / 'audit.log').write_text(audit + 'Full theorem axiom audit: theorems=2\n')
        inputs = {'Input.lean': 'b' * 64}
        (self.baseline / 'inputs.json').write_text(json.dumps(inputs))
        (new.parent / 'inputs.json').write_text(json.dumps(inputs))
        self.batch = {'task': 'fixture', 'accepted': True, 'comparator_exit': 0,
                      'comparator_sha256': 'a' * 64, 'module_count': 1, 'theorem_count': 2,
                      'modules': {file: ['helper', '_private.' + module + '.0.hidden']},
                      'per_file_audits': [{'file': file, 'audit_log': str(new / 'audit.log')}],
                      'audit_inventory_seconds': 0.3, 'batch_inventory_seconds': 0.2,
                      'comparator_seconds': 0.5}
        self.batch.update({'batch_targets': self.batch['modules'][file],
                           'owned_modules': [{'module': module, 'theorems': self.batch['modules'][file]}],
                           'unique_theorem_count': 2, 'additional_targets': [],
                           'all_theorems_axioms_checked': True})
        (new / 'targets.json').write_text(json.dumps({
            'theorems': self.batch['batch_targets'], 'owned_modules': self.batch['owned_modules'],
            'aliases': ['target0', 'target1']}))
        self.batch_file.write_text(json.dumps(self.batch))
        self.metrics = {'exit_code': 0, 'observed_systemd_service_cgroups': ['run-fixture.service'],
                        'wall_seconds': 1, 'sampled_peak_total_rss_bytes': 100,
                        'rss_method': 'fixture', 'requested_sampling_interval_seconds': 0.1,
                        'maximum_sampling_gap_seconds': 0.11, 'cpu_affinity_count': 1}

    def test_comparison_preserves_every_auxiliary_target(self):
        summary, _ = compare(self.baseline, self.batch_file, self.metrics, self.metrics)
        self.assertEqual(summary['per_file_theorem_occurrences'], 2)
        self.assertEqual(summary['private_theorem_count'], 1)
        self.assertTrue(summary['per_file_trust_audit_summaries_identical'])

    def test_unaccepted_baseline_cannot_produce_a_success_report(self):
        self.old_row['accepted'] = False
        self.old_file.write_text(json.dumps(self.old_row))
        with self.assertRaisesRegex(ValueError, 'unaccepted replay'):
            compare(self.baseline, self.batch_file, self.metrics, self.metrics)

    def test_changed_proof_inputs_cannot_be_compared(self):
        (self.batch_file.parent.parent / 'inputs.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'proof inputs differ'):
            compare(self.baseline, self.batch_file, self.metrics, self.metrics)

    def test_dropped_private_helper_cannot_produce_a_success_report(self):
        file = next(iter(self.batch['modules']))
        self.batch['modules'][file] = ['helper']
        self.batch['theorem_count'] = 1
        self.batch_file.write_text(json.dumps(self.batch))
        with self.assertRaisesRegex(ValueError, 'theorem coverage differs'):
            compare(self.baseline, self.batch_file, self.metrics, self.metrics)

    def previous_batch(self):
        previous = self.batch_file.parent.parent.parent / 'previous/Batch_case/result.json'
        previous.parent.mkdir(parents=True)
        (previous.parent.parent / 'inputs.json').write_text(
            (self.batch_file.parent.parent / 'inputs.json').read_text())
        self.batch.update(mode='task-batch-replay', audit_batch_size=4,
                          audit_build_invocations=1, snapshot_seconds=0.1)
        self.batch_file.write_text(json.dumps(self.batch))
        previous.write_text(json.dumps(self.batch))
        return previous

    def test_previous_batch_must_preserve_exact_helpers_and_audits(self):
        previous = self.previous_batch()
        row = compare_previous_batch(previous, self.batch_file, self.metrics, self.metrics)
        self.assertTrue(row['batch_targets_identical'])
        self.assertEqual(row['audit_batch_size'], 4)

    def test_previous_batch_different_helper_cannot_be_compared(self):
        previous = self.previous_batch()
        self.batch['batch_targets'] = ['different_helper']
        self.batch_file.write_text(json.dumps(self.batch))
        with self.assertRaisesRegex(ValueError, 'batch_targets'):
            compare_previous_batch(previous, self.batch_file, self.metrics, self.metrics)

    def test_previous_batch_different_measurement_conditions_cannot_be_compared(self):
        previous = self.previous_batch()
        metrics = {**self.metrics, 'cpu_affinity_count': 2}
        with self.assertRaisesRegex(ValueError, 'measurement conditions differ'):
            compare_previous_batch(previous, self.batch_file, self.metrics, metrics)


if __name__ == '__main__':
    unittest.main()
