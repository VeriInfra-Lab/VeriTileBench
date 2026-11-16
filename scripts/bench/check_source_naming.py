#!/usr/bin/env python3
"""Check Python-derived source names and their direct kernel instances.

The existing correspondence checker validates bodies, call signatures and
helper links. This gate adds the naming convention from AGENTS.md; it does
not certify Lean proofs or hardware behavior.
"""
from __future__ import annotations
import argparse,ast,io,json,re,sys,tokenize
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.bench.generate_jit_correspondence import generate, quotations
from scripts.bench.audit_source import lean_code_mask
from scripts.bench.jit_correspondence import quotation
ROOT=Path(__file__).resolve().parents[2]
SOURCE_KINDS={'source_procedure','statement_jit_function','procedure_jit_function',
              'pure_jit_function','binary_pure_jit_function','uniform_jit_function','control_source'}
IDENT=r"[A-Za-z_][\w.']*"
def kernel_name(python_name: str) -> str:
    return python_name if python_name.endswith(('_kernel','Kernel')) else python_name+'Kernel'
def python_suite(text: str, function: ast.FunctionDef) -> str:
    """Keep the complete suite, including comments after its last statement."""
    lines=text.splitlines(keepends=True)
    header=lines[function.lineno-1]
    indentation=len(header[:len(header)-len(header.lstrip())].expandtabs(8))
    end=function.end_lineno
    # AST end_lineno omits trailing comments. Indented comment-only lines
    # still belong to this function; a module/sibling comment does not.
    while end<len(lines):
        line=lines[end]
        if line.strip():
            prefix=line[:len(line)-len(line.lstrip())]
            if not line.lstrip().startswith('#') or len(prefix.expandtabs(8))<=indentation:
                break
        end+=1
    fragment=''.join(lines[function.lineno-1:end])
    starts=[0]
    for line in fragment.splitlines(keepends=True):starts.append(starts[-1]+len(line))
    depth=0
    for token in tokenize.generate_tokens(io.StringIO(fragment).readline):
        if token.type!=tokenize.OP:continue
        if token.string in '([{':depth+=1
        elif token.string in ')]}':depth-=1
        elif token.string==':' and depth==0:
            return fragment[starts[token.end[0]-1]+token.end[1]:]
    raise ValueError('missing Python function suite')

def comment_tokens(body: str) -> list[str]:
    return [t.string.rstrip() for t in tokenize.generate_tokens(io.StringIO(body).readline)
            if t.type==tokenize.COMMENT]

def validate_records(root: Path, records: list[dict], *, check_comments: bool=False) -> dict:
    sources={}
    for row in records:
        kind=row.get('kind')
        if kind is not None and kind not in SOURCE_KINDS:
            raise ValueError('Unknown source kind for the naming gate: '+str(kind))
        expected=row['python_function']+'Source' if kind in SOURCE_KINDS else kernel_name(row['python_function'])
        actual=row['lean_definition'].split('::')[-1].split('.')[-1]
        if actual!=expected:
            raise ValueError(f"{row['lean']}::{row['lean_definition']}: expected {expected} for Python {row['python_function']}")
        if check_comments:
            python=(root/row['python']).read_text()
            function=next(n for n in ast.walk(ast.parse(python))
                          if isinstance(n,ast.FunctionDef) and n.name==row['python_function'])
            _,body=quotation((root/row['lean']).read_text(),row['lean_definition'],raw_python=True)
            if comment_tokens(python_suite(python,function))!=comment_tokens(body):
                raise ValueError(f"{row['lean']}::{row['lean_definition']}: pasted Python comments differ")
        if kind=='source_procedure':sources[actual]=row['python_function']
    # Check thin SourceProcedure instances too: they contain no literal body,
    # so they are intentionally absent from the correspondence pair list.
    for file in {r['lean'] for r in records}:
        code=lean_code_mask((root/file).read_text())
        decls=list(re.finditer(r'(?m)^def\s+('+IDENT+r')\b',code))
        for i,decl in enumerate(decls):
            body=code[decl.end():decls[i+1].start() if i+1<len(decls) else len(code)]
            call=re.search(r'\binstantiate_source_procedure\b(.*?)\bsignature\b',body,re.S)
            if call is None:continue
            referenced={word.split('.')[-1] for word in re.findall(IDENT,call[1])}&sources.keys()
            if not referenced:continue  # A generic Lean adapter, with no fixed Python callee.
            if len(referenced)!=1:raise ValueError('Ambiguous source instance: '+decl[1])
            source=next(iter(referenced));expected=kernel_name(sources[source])
            if decl[1].split('.')[-1]!=expected:
                raise ValueError(f'{file}::{decl[1]}: expected kernel instance {expected}')
    return {'status':'PASS','pairs':len(records)}
def validate_lean_coverage(root: Path, cases: list[Path | str], records: list[dict]) -> None:
    # A shared kernel module may also contain a source used by another selected
    # task. Check the union of validated pairs, never exempt an unpaired body.
    paired={(row['lean'],row['lean_definition']) for row in records}
    for case in cases:
        directory=(root/case).resolve()
        for quote in quotations(root,directory,include_unparsed=True):
            if (root/quote.path).resolve().is_relative_to(directory) and (quote.path,quote.definition) not in paired:
                raise ValueError(f'{quote.path}::{quote.definition}: Lean source body has no Python JIT pair')

def check_case(root: Path, case: Path | str, *, check_comments: bool=False,
               check_lean_coverage: bool=False) -> dict:
    records=generate(root,case)
    result=validate_records(root,records,check_comments=check_comments)
    if check_lean_coverage:validate_lean_coverage(root,[case],records)
    return result
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases',nargs='*');parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--check-comments',action='store_true',help='also retain all Python body comments')
    parser.add_argument('--check-lean-coverage',action='store_true',help='also require a Python pair for every owned Lean source body')
    args=parser.parse_args();root=args.root.resolve()
    cases=args.cases or [c['path'] for c in json.loads((root/'validation/body-migration-batch.json').read_text())['cases']]
    pairs=0;records=[]
    try:
        for case in cases:
            selected=generate(root,case)
            pairs+=validate_records(root,selected,check_comments=args.check_comments)['pairs']
            records.extend(selected)
        if args.check_lean_coverage:validate_lean_coverage(root,cases,records)
    except (ValueError,OSError,KeyError) as error:parser.exit(1,str(error)+'\n')
    print(json.dumps({'status':'PASS','cases':len(cases),'pairs':pairs}))
if __name__=='__main__':main()
