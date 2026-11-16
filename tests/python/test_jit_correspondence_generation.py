import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.bench.generate_jit_correspondence import generate, snapshot_matches
from scripts.bench.jit_correspondence import check_records
from scripts.bench.library_source import library_root
from tests.python.test_body_corpus import PYTHON, LEAN
from tests.python.layout_fixtures import pinned_cases, split_swiglu_fixture

ROOT = Path(__file__).resolve().parents[2]


class CorrespondenceGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = Path('bench/tritonbench_g/sample')
        self.directory = self.root / self.case
        self.directory.mkdir(parents=True)
        self.python = self.directory / 'sample.py'
        self.lean = self.directory / 'Kernels.lean'
        self.python.write_text(PYTHON)
        self.lean.write_text(LEAN)
        self.manifest = self.directory / 'jit-correspondence.json'

    def cli(self, *arguments):
        return subprocess.run([sys.executable, '-m', 'scripts.bench.generate_jit_correspondence',
                               str(self.case), '--root', str(self.root), *arguments],
                              cwd=ROOT, text=True, capture_output=True)

    def test_no_manifest_is_needed_or_read(self):
        with patch('scripts.bench.jit_correspondence.check_manifest', side_effect=AssertionError('must not read JSON')):
            rows = generate(self.root, self.case)
            self.manifest.write_text('invalid JSON: discovery must never use this')
            self.assertEqual(generate(self.root, self.case), rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(check_records(self.root, rows)[0]['status'], 'PASS')

    def test_default_cli_checks_without_writing(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.manifest.exists())
        self.assertFalse(json.loads(result.stdout)['proof_acceptance'])

    def test_write_preserves_an_equivalent_existing_snapshot_byte_for_byte(self):
        saved = json.dumps(generate(self.root, self.case), separators=(',', ':'))
        self.manifest.write_text(saved)
        result = self.cli('--write')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['manifest_written'])
        self.assertEqual(self.manifest.read_text(), saved)

    def test_explicit_write_and_check_do_not_silently_refresh_types(self):
        result = self.cli('--write')
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = self.manifest.read_bytes()
        self.assertEqual(self.cli('--check').returncode, 0)
        self.lean.write_text(LEAN.replace('("out_ptr",.fp32)', '("out_ptr",.fp16)'))
        result = self.cli('--check')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('reviewed launch types', result.stderr)
        self.assertEqual(self.manifest.read_bytes(), saved)

    def test_body_cast_index_and_store_mutations_are_rejected(self):
        python = PYTHON.replace('tl.arange(0, B)', 'tl.arange(0, B).to(tl.int64)')
        lean = LEAN.replace('tl.arange(0, B)', 'tl.arange(0, B).to(tl.int64)')
        self.python.write_text(python)
        self.lean.write_text(lean)
        self.assertEqual(len(generate(self.root, self.case)), 1)
        for changed in [lean.replace('.to(tl.int64)', ''),
                        lean.replace('tl.arange(0, B)', 'tl.arange(1, B)'),
                        lean.replace('tl.store(out_ptr + offsets, x)', 'tl.store(out_ptr + offsets, x + x)')]:
            with self.subTest(source=changed):
                self.lean.write_text(changed)
                with self.assertRaisesRegex(ValueError, 'No exact source pair'):
                    generate(self.root, self.case)

    def test_unmatched_added_jit_is_not_omitted(self):
        self.python.write_text(PYTHON + '\n@triton.jit\ndef unmatched(x):\n    return x + x\n')
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            generate(self.root, self.case)

    def test_ambiguous_pair_requires_a_small_explicit_binding(self):
        self.lean.write_text(LEAN + LEAN[LEAN.index('def body '):].replace('def body ', 'def duplicate '))
        with self.assertRaisesRegex(ValueError, 'Ambiguous source pair'):
            generate(self.root, self.case)
        selection = {(self.python.relative_to(self.root).as_posix(), 'first'):
                     {(self.lean.relative_to(self.root).as_posix(), 'body')}}
        self.assertEqual(generate(self.root, self.case, selection)[0]['lean_definition'], 'body')

    def test_distinct_input_types_generate_distinct_variants(self):
        half = LEAN.replace('callSignature', 'halfSignature').replace('def body ', 'def halfBody ')
        half = half.replace('("x_ptr",.fp32)', '("x_ptr",.fp16)')
        self.lean.write_text(LEAN + half)
        rows = generate(self.root, self.case)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r['invocation_variant'] for r in rows}), 2)

    def test_source_directory_and_shared_python_reference(self):
        source = self.directory / 'Sources' / 'Body.lean'
        source.parent.mkdir()
        source.write_text(LEAN)
        self.lean.write_text('import bench.tritonbench_g.sample.Sources.Body\n')
        external = self.root / 'bench/tritonbench_g/shared/source.py'
        external.parent.mkdir()
        external.write_text(self.python.read_text())
        self.python.unlink()
        (self.directory.parent/'source-references.json').write_text(json.dumps({'sample': external.relative_to(self.root).as_posix()}))
        rows = generate(self.root, self.case)
        self.assertEqual(rows[0]['lean'], source.relative_to(self.root).as_posix())
        self.assertEqual(rows[0]['python'], external.relative_to(self.root).as_posix())

    def test_source_paths_cannot_escape_root(self):
        with self.assertRaisesRegex(ValueError, 'outside repository'):
            generate(self.root, self.root.parent)

    def test_rms_source_is_discovered_from_one_python_file_without_a_snapshot(self):
        directory = ROOT / 'bench/tritonbench_g/rms_matmul_rbe'
        self.assertEqual([path.name for path in directory.glob('*.py')], ['rms_matmul_rbe.py'])
        self.assertFalse((directory / 'jit-correspondence.json').exists())
        rows = generate(ROOT, directory)
        self.assertEqual({row['python_function']: row['lean_definition'] for row in rows}, {
            'rms_matmul_rbe': 'rms_matmul_rbeSource',
            'rms_matmul_rbe_qkv': 'rms_matmul_rbe_qkvKernel',
            'rms_matmul_rbe_index_commuted': 'rms_matmul_rbe_index_commutedSource',
            'rms_matmul_rbe_qkv_index_commuted': 'rms_matmul_rbe_qkv_index_commutedKernel',
        })

    def copy_case(self, name, files):
        self.python.unlink()
        source = ROOT / 'bench/tritonbench_g' / name
        for filename in files:
            shutil.copyfile(source/filename, self.directory/filename)

    def test_rms_parent_cannot_select_the_wrong_callee(self):
        self.copy_case('rms_matmul_rbe', ['Kernels.lean', 'rms_matmul_rbe.py'])
        self.assertEqual(len(generate(self.root, self.case)), 4)
        text = self.lean.read_text()
        old = 'source_procedures [rms_matmul_rbe_index_commuted := rms_matmul_rbe_index_commutedSource]'
        self.assertEqual(text.count(old), 1)
        self.lean.write_text(text.replace(old, 'source_procedures [rms_matmul_rbe_index_commuted := rms_matmul_rbeSource]'))
        with self.assertRaisesRegex(ValueError, 'No exact source pair|Ambiguous source pair'):
            generate(self.root, self.case)

    def test_cross_file_helper_import_is_checked_without_importing_triton(self):
        self.copy_case('swiglu_triton', ['Kernels.lean', 'swiglu_triton.py'])
        candidate = split_swiglu_fixture(self.directory)
        self.assertEqual(len(generate(self.root, self.case)), 5)
        candidate.write_text(candidate.read_text().replace('from swiglu_triton import silu',
                                                          'from swiglu_triton import missing as silu'))
        with self.assertRaisesRegex(ValueError, 'No exact source pair'):
            generate(self.root, self.case)

    def test_distinct_kernel_names_keep_parameterized_signature_scopes(self):
        self.copy_case('triton_argmax', ['Kernels.lean', 'triton_argmax.py'])
        rows = generate(self.root, self.case)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row['status'] == 'PASS' for row in check_records(self.root, rows)))
        stages = [row for row in rows if row['python_function'].startswith('argmax_kernel_1_')]
        self.assertEqual(len(stages), 2)
        self.assertTrue(all('::' in row['lean_definition'] for row in stages))

    def test_matching_bodies_cannot_hide_unrelated_source_names(self):
        from scripts.bench.check_source_naming import check_case
        self.copy_case('fp4_to_bf16_conversion', ['Kernels.lean', 'fp4_to_bf16_conversion.py'])
        self.assertEqual(check_case(self.root, self.case)['status'], 'PASS')
        self.lean.write_text(self.lean.read_text().replace('_fp4_packed_to_bf16Source', 'unrelatedSource'))
        self.assertEqual(len(generate(self.root, self.case)), 2)
        with self.assertRaisesRegex(ValueError, 'expected _fp4_packed_to_bf16Source'):
            check_case(self.root, self.case)

    def test_source_procedure_returned_jit_helper_is_checked(self):
        self.copy_case('fp4_to_bf16_conversion', ['Kernels.lean', 'fp4_to_bf16_conversion.py'])
        rows = generate(self.root, self.case)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['status']=='PASS' for r in check_records(self.root, rows)))
        text = self.lean.read_text()
        self.lean.write_text(text.replace('_fp4_packed_to_bf16 := _fp4_packed_to_bf16Source',
                                         '_fp4_packed_to_bf16 := missingSource'))
        with self.assertRaisesRegex(ValueError, 'No exact source pair'):
            generate(self.root, self.case)
        with self.assertRaisesRegex(ValueError, 'helper bindings differ'):
            check_records(self.root, rows)

    def test_comment_gate_retains_leading_and_inline_python_comments(self):
        from scripts.bench.check_source_naming import check_case
        python=PYTHON.replace('    offsets =', '    # retain the original index comment\n    offsets =')
        lean=LEAN.replace('def body ', 'def firstKernel ').replace(
            '    offsets =', '    # retain the original index comment\n    offsets =')
        python=python.replace('x = tl.load(x_ptr + offsets)', 'x = tl.load(x_ptr + offsets)  # keep the load')
        lean=lean.replace('x = tl.load(x_ptr + offsets)', 'x = tl.load(x_ptr + offsets)  # keep the load')
        self.python.write_text(python)
        self.lean.write_text(lean)
        self.assertEqual(check_case(self.root,self.case,check_comments=True)['status'],'PASS')
        for comment in ['    # retain the original index comment\n','  # keep the load']:
            self.lean.write_text(lean.replace(comment,''))
            self.assertEqual(check_case(self.root,self.case)['status'],'PASS')
            with self.assertRaisesRegex(ValueError,'pasted Python comments differ'):
                check_case(self.root,self.case,check_comments=True)

    def test_comment_gate_distinguishes_literals_and_accepts_comment_spacing(self):
        from scripts.bench.check_source_naming import check_case, comment_tokens, python_suite
        import ast
        source='def helper(\n    x: "type: annotation",\n):\n    # comment   \n    return "# literal: not a comment"\n'
        body=python_suite(source,ast.parse(source).body[0])
        self.assertEqual(comment_tokens(body),['# comment'])
        self.assertEqual(comment_tokens('return "# literal"\n'),[])
        self.python.write_text(PYTHON.replace('    offsets =','    # original   \n    offsets ='))
        self.lean.write_text(LEAN.replace('def body ','def firstKernel ').replace(
            '    offsets =','    # original\n    offsets ='))
        self.assertEqual(check_case(self.root,self.case,check_comments=True)['status'],'PASS')

    def test_comment_gate_requires_trailing_suite_comments(self):
        from scripts.bench.check_source_naming import check_case
        for ending in ['\n    # trailing comment\n', '\n    # trailing comment',
                       '\n\n    # trailing comment\n\n']:
            with self.subTest(ending=ending):
                self.python.write_text(PYTHON.rstrip()+ending+'\ndef sibling():\n    pass\n')
                lean=LEAN.replace('def body ','def firstKernel ')
                self.lean.write_text(lean)
                with self.assertRaisesRegex(ValueError,'pasted Python comments differ'):
                    check_case(self.root,self.case,check_comments=True)
                self.lean.write_text(lean.replace('  }','    # trailing comment\n  }'))
                self.assertEqual(check_case(self.root,self.case,check_comments=True)['status'],'PASS')

    def test_suite_comments_stop_before_siblings_and_handle_tabs(self):
        from scripts.bench.check_source_naming import comment_tokens, python_suite
        import ast
        samples=[
            ('def outer():\n    def nested():\n        return 1\n        # nested tail\n    # sibling comment\n    return 2\n',1,['# nested tail']),
            ('def function():\n\treturn 1\n\t# tabbed tail\n# module comment\ndef sibling():\n    pass\n',0,['# tabbed tail']),
            ('def function():\n    return 1\n    # EOF tail',0,['# EOF tail']),
        ]
        for source,index,expected in samples:
            with self.subTest(source=source):
                function=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)
                              and n.name==('nested' if index else 'function'))
                self.assertEqual(comment_tokens(python_suite(source,function)),expected)

    def test_reverse_coverage_rejects_python_shaped_and_legacy_lean_only_candidates(self):
        from scripts.bench.check_source_naming import check_case
        original=LEAN.replace('def body ','def firstKernel ')
        extra=LEAN[LEAN.index('def body '):].replace('def body ','def extraKernel ')
        extra=extra.replace('tl.store(out_ptr + offsets, x)','tl.store(out_ptr + offsets, x + x)')
        legacy=extra.replace('    x = tl.load(x_ptr + offsets)',
                             '    for i in range(0, B) {\n      x = tl.load(x_ptr + offsets)\n    }')
        for candidate in [extra,legacy]:
            with self.subTest(candidate=candidate):
                self.lean.write_text(original+'\n'+candidate)
                self.assertEqual(len(generate(self.root,self.case)),1)
                with self.assertRaisesRegex(ValueError,'Lean source body has no Python JIT pair'):
                    check_case(self.root,self.case,check_lean_coverage=True)

    def test_reverse_coverage_checks_shared_modules_against_all_validated_pairs(self):
        from scripts.bench.check_source_naming import validate_lean_coverage
        first=LEAN.replace('def body ','def firstKernel ')
        second=LEAN[LEAN.index('def body '):].replace('def body ','def secondKernel ')
        second=second.replace('tl.store(out_ptr + offsets, x)','tl.store(out_ptr + offsets, x + x)')
        self.lean.write_text(first+'\n'+second)
        other=Path('bench/tritonbench_g/other')
        (self.root/other).mkdir()
        (self.root/other/'Kernels.lean').write_text('import bench.tritonbench_g.sample.Kernels\n')
        (self.root/other/'other.py').write_text(PYTHON.replace('def first(','def second(').replace(
            'tl.store(out_ptr + offsets, x)','tl.store(out_ptr + offsets, x + x)'))
        own=generate(self.root,self.case)
        with self.assertRaisesRegex(ValueError,'Lean source body has no Python JIT pair'):
            validate_lean_coverage(self.root,[self.case],own)
        validate_lean_coverage(self.root,[self.case,other],own+generate(self.root,other))

    def test_standardized_cases_match_reviewed_pairs(self):
        batch = json.loads((ROOT / 'validation/body-migration-batch.json').read_text())
        reviewed = pinned_cases()
        self.assertEqual(len(batch['cases']), 203)
        count = 0
        for entry in batch['cases']:
            root = ROOT if (ROOT / entry['path']).is_dir() else library_root(ROOT)
            case = root / entry['path']
            with self.subTest(case=entry['path']):
                entries = {p.name for p in case.iterdir() if not p.name.startswith('.')
                           and (p.is_file() or any(f.is_file() and '__pycache__' not in f.parts
                                                   for f in p.rglob('*')))}
                self.assertEqual(entries, {case.name + '.py', 'Kernels.lean', 'RealCorrect.lean',
                                           'FPEquiv.lean', 'IO.lean', 'Contract.lean', 'Proofs'})
                rows = generate(root, case)
                if entry['path'] in reviewed:
                    self.assertTrue(snapshot_matches(reviewed[entry['path']], rows))
                count += len(rows)
        self.assertEqual(count, 601)


if __name__ == '__main__':
    unittest.main()
