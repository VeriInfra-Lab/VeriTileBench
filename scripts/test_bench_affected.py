"""Exercise real Git changes and both dependency graphs without Lean caches."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest

from scripts.bench.affected import MANIFEST, git_changes, select, validate, shard


class AffectedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'CI fixture')
        self.a = 'bench/examples/A'
        self.b = 'bench/tritonbench_g/B'
        self.c = 'bench/examples/C'
        cases = []
        for case in [self.a, self.b, self.c]:
            self.write(case + '/source.py', 'def kernel():\n    return 1\n')
            self.write(case + '/Kernels.lean', 'import Mathlib\n')
            self.write(case + '/RealCorrect.lean', 'import ' + case.replace('/', '.') + '.Kernels\n')
            self.write(case + '/FPEquiv.lean', 'import ' + case.replace('/', '.') + '.Kernels\n')
            # Even an unimported retained helper must be replayed.
            self.write(case + '/Proofs/Unused.lean', 'theorem aux : True := True.intro\n')
            cases.append({'path': case, 'python': case + '/source.py', 'public_theorems': {
                case + '/RealCorrect.lean': ['N.real'], case + '/FPEquiv.lean': ['N.fp']}})
        self.write(MANIFEST, json.dumps({'schema': 1, 'cases': cases}))
        self.write('tests/lean/Consumer.lean', 'import bench.examples.A.Kernels\n')
        self.write('tests/lean/Independent.lean', 'import Mathlib\n')
        self.write('lake-manifest.json', '{}')
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True).strip()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def plan(self):
        head = self.commit()
        changed, old = git_changes(self.root, self.base, head)
        return select(self.root, changed, old, base=self.base, head=head)

    def case_names(self, plan):
        return {c['path'] for c in plan['cases']}

    def test_python_change_replays_whole_case_unimported_helper_and_regression(self):
        self.write(self.a + '/source.py', 'def kernel():\n    return 2\n')
        plan = self.plan()
        self.assertEqual(self.case_names(plan), {self.a})
        self.assertIn(self.a + '/Proofs/Unused.lean', plan['targets'])
        self.assertIn('tests/lean/Consumer.lean', plan['targets'])
        self.assertNotIn('tests/lean/Independent.lean', plan['targets'])
        self.assertEqual(plan['auditor_required'], 'false')
        validate(self.root, plan)

    def test_base_import_edges_survive_rename_and_removed_import(self):
        self.write(self.b + '/Kernels.lean', 'import bench.examples.A.Proofs.Unused\n')
        self.base = self.commit()
        self.git('mv', self.a + '/Proofs/Unused.lean', self.a + '/Proofs/Renamed.lean')
        self.write(self.b + '/Kernels.lean', 'import Mathlib\n')
        plan = self.plan()
        self.assertEqual(self.case_names(plan), {self.a, self.b})
        self.assertIn(self.a + '/Proofs/Unused.lean', plan['changed_files'])
        self.assertIn(self.a + '/Proofs/Renamed.lean', plan['changed_files'])
        self.assertNotIn(self.a + '/Proofs/Unused.lean', plan['targets'])

    def test_deleted_shared_module_keeps_old_consumers(self):
        self.write('tests/lean/Shared.lean', 'theorem x : True := True.intro\n')
        self.write(self.b + '/Kernels.lean', 'import tests.lean.Shared\n')
        self.base = self.commit()
        (self.root / 'tests/lean/Shared.lean').unlink()
        plan = self.plan()
        self.assertEqual(self.case_names(plan), {self.b})
        self.assertNotIn('tests/lean/Shared.lean', plan['targets'])

    def test_python_alias_and_relative_helper_dependencies(self):
        self.write(self.b + '/source.py', 'from alias_a import kernel\n')
        self.write(self.a + '/source.py', 'from .helper import f\n')
        self.write(self.a + '/helper.py', 'def f():\n    return 1\n')
        self.write('bench/examples/source-references.json', json.dumps({'alias_a': self.a + '/source.py'}))
        self.base = self.commit()
        self.write(self.a + '/helper.py', 'def f():\n    return 2\n')
        self.assertEqual(self.case_names(self.plan()), {self.a, self.b})

    def test_lean_reverse_dependents_and_forward_owned_dependencies(self):
        self.write(self.b + '/Kernels.lean', 'import bench.examples.A.Kernels\n')
        self.write(self.a + '/Kernels.lean', 'import bench.examples.C.Kernels\n')
        self.base = self.commit()
        self.write(self.a + '/source.py', 'def kernel():\n    return 2\n')
        plan = self.plan()
        self.assertEqual(self.case_names(plan), {self.a, self.b})
        self.assertIn(self.c + '/Kernels.lean', plan['targets'])
        self.assertNotIn(self.c + '/RealCorrect.lean', plan['targets'])

    def test_docs_and_pending_changes_do_not_request_verified_proofs(self):
        self.write('README.md', 'docs\n')
        self.write('bench/bug_detection/Pending/Kernel.lean', 'import Mathlib\n')
        plan = self.plan()
        self.assertEqual(plan['affected_required'], 'false')
        self.assertEqual(plan['auditor_required'], 'false')
        self.assertFalse(plan['targets'])

    def test_tool_changes_request_actual_auditor_regressions(self):
        self.write('scripts/check_comparator_batch.py', '# changed audit tool\n')
        plan = self.plan()
        self.assertEqual(plan['affected_required'], 'false')
        self.assertEqual(plan['auditor_required'], 'true')

    def test_pending_source_used_by_a_verified_case_still_propagates(self):
        shared = 'bench/bug_detection/Shared/Kernel.lean'
        self.write(shared, 'theorem x : True := True.intro\n')
        self.write(self.b + '/Kernels.lean', 'import bench.bug_detection.Shared.Kernel\n')
        self.base = self.commit()
        self.write(shared, 'theorem x : True := by trivial\n')
        plan = self.plan()
        self.assertEqual(self.case_names(plan), {self.b})
        self.assertNotIn(shared, plan['targets'])

    def test_local_dirty_and_untracked_sources_are_included(self):
        self.write(self.a + '/source.py', 'def kernel():\n    return 2\n')
        self.write(self.b + '/Proofs/New.lean', 'import Mathlib\n')
        changed, old = git_changes(self.root, self.base, self.base)
        plan = select(self.root, changed, old, base=self.base, head=self.base)
        self.assertEqual(self.case_names(plan), {self.a, self.b})
        self.assertIn(self.b + '/Proofs/New.lean', plan['targets'])
        validate(self.root, plan)

    def test_locked_library_change_selects_all_cases_and_regressions(self):
        self.write('lake-manifest.json', '{"rev": "new"}')
        plan = self.plan()
        self.assertEqual(self.case_names(plan), {self.a, self.b, self.c})
        self.assertIn('tests/lean/Independent.lean', plan['targets'])
        self.assertEqual(plan['auditor_required'], 'true')

    def test_missing_base_falls_back_to_full_verified_scope(self):
        head = self.git('rev-parse', 'HEAD')
        changed, old = git_changes(self.root, '0' * 40, head)
        self.assertIsNone(old)
        plan = select(self.root, changed, old, base='0' * 40, head=head)
        self.assertEqual(self.case_names(plan), {self.a, self.b, self.c})
        self.assertIn('tests/lean/Independent.lean', plan['targets'])
        validate(self.root, plan)
        with self.assertRaises(ValueError):
            git_changes(self.root, self.base, 'f' * 40)

    def test_manifest_goal_change_selects_the_affected_case(self):
        manifest = json.loads((self.root / MANIFEST).read_text())
        manifest['cases'][1]['public_theorems'][self.b + '/FPEquiv.lean'].append('N.disproved')
        self.write(MANIFEST, json.dumps(manifest))
        self.assertEqual(self.case_names(self.plan()), {self.b})

    def test_plan_rejects_removed_helper_forged_scope_and_source_drift(self):
        self.write(self.a + '/source.py', 'def kernel():\n    return 2\n')
        plan = self.plan()
        validate(self.root, plan)
        bad = copy.deepcopy(plan)
        bad['targets'].remove(self.a + '/Proofs/Unused.lean')
        with self.assertRaises(ValueError):
            validate(self.root, bad)
        self.write(self.a + '/Proofs/Added.lean', 'import Mathlib\n')
        with self.assertRaises(ValueError):
            validate(self.root, plan)

    def test_daily_plan_uses_complete_shards_and_always_runs_auditor(self):
        head = self.git('rev-parse', 'HEAD')
        plan = select(self.root, set(), None, head=head, full=True)
        self.assertEqual(plan['affected_required'], 'false')
        self.assertEqual(plan['auditor_required'], 'true')
        self.assertFalse(plan['targets'])

    def test_large_selection_shards_cover_every_helper_and_public_goal_once(self):
        for i in range(260):
            self.write(self.a + f'/Proofs/Aux{i}.lean', 'theorem aux : True := True.intro\n')
        self.write('lake-manifest.json', '{"rev": "new"}')
        plan = self.plan()
        self.assertGreater(plan['shard_count'], 1)
        selected, goals = [], []
        for index in range(plan['shard_count']):
            part = shard(plan, index, plan['shard_count'])
            self.assertTrue(part['targets'])
            selected.extend(part['targets'])
            for case in part['cases']:
                for path, names in case['public_theorems'].items():
                    self.assertIn(path, part['targets'])
                    goals.extend((path, name) for name in names)
        expected = [(p, n) for c in plan['cases'] for p, ns in c['public_theorems'].items() for n in ns]
        self.assertEqual(sorted(selected), plan['targets'])
        self.assertEqual(sorted(goals), sorted(expected))
        with self.assertRaises(ValueError):
            shard(plan, 0, 1)

    def test_plan_is_identical_across_independent_python_hash_seeds(self):
        self.write('lean-toolchain', 'leanprover/lean4:v4.24.0\n')
        self.write('lakefile.toml', 'name = "fixture"\n')
        script = ('from pathlib import Path; import json; '
                  'from scripts.bench.affected import source_files, select; '
                  'root=Path.cwd(); '
                  'print(json.dumps(select(root, set(["lean-toolchain", "lakefile.toml", '
                  '"lake-manifest.json"]), source_files(root)), sort_keys=True))')
        outputs = []
        for seed in ['1', '19', '53']:
            env = {**os.environ, 'PYTHONHASHSEED': seed,
                   'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
            outputs.append(subprocess.check_output([sys.executable, '-B', '-c', script],
                           cwd=self.root, env=env, text=True))
        self.assertEqual(len(set(outputs)), 1)
