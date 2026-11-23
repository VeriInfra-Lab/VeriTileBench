import models.NanoGPT.Track1.ArtifactTraining
import models.NanoGPT.Track1.ValueAdamNativeEvent
import models.NanoGPT.Track1.ValueAdamRsqrtContract

namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver.CombinedTraining

open VeriTile.Triton
open scoped VeriTile.Spec
open ValueAdam.Elementwise (CombinedPrograms NativeDivider)

def guardedAdamSource : String :=
  candidateNativeAdam.replace "    direction = m_new / (gl.sqrt_rn(v_new) + 1e-10)\n" r#"    if BLOCK == 1024:
        direction = gl.inline_asm_elementwise(
            """{
    .reg .b32 %mOr, %denAnd, %denOr, %shifted, %moments, %rootAnd, %rootOr;
    .reg .f32 %inverse, %root, %den<8>, %rsq, %rough, %negativeRough, %halfRsq, %residual;
    .reg .pred %invalid, %slow, %hasMoment, %warpHasMoment, %specialRoot, %warpSpecialRoot;
    and.b32 %rootAnd, $16, $17;
    lop3.b32 %rootAnd, %rootAnd, $18, $19, 0x80;
    lop3.b32 %rootAnd, %rootAnd, $20, $21, 0x80;
    lop3.b32 %rootAnd, %rootAnd, $22, $23, 0x80;
    or.b32 %rootOr, $16, $17;
    lop3.b32 %rootOr, %rootOr, $18, $19, 0xfe;
    lop3.b32 %rootOr, %rootOr, $20, $21, 0xfe;
    lop3.b32 %rootOr, %rootOr, $22, $23, 0xfe;
    and.b32 %rootAnd, %rootAnd, 0x20000000;
    and.b32 %rootOr, %rootOr, 0xc0000000;
    xor.b32 %rootAnd, %rootAnd, 0x20000000;
    or.b32 %shifted, %rootAnd, %rootOr;
    setp.ne.u32 %specialRoot, %shifted, 0;
    or.b32 %mOr, $8, $9;
    lop3.b32 %mOr, %mOr, $10, $11, 0xfe;
    lop3.b32 %mOr, %mOr, $12, $13, 0xfe;
    lop3.b32 %mOr, %mOr, $14, $15, 0xfe;
    and.b32 %mOr, %mOr, 0x7f800000;
    setp.eq.or.u32 %specialRoot, %mOr, 0x7f800000, %specialRoot;
    vote.sync.any.pred %warpSpecialRoot, %specialRoot, 0xffffffff;
    @%warpSpecialRoot bra.uni L__adam_root_original;
    rsqrt.approx.ftz.f32 %rsq, $16;
    mul.rn.ftz.f32 %rough, $16, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $16;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $17;
    mul.rn.ftz.f32 %rough, $17, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $17;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $18;
    mul.rn.ftz.f32 %rough, $18, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $18;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den2, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $19;
    mul.rn.ftz.f32 %rough, $19, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $19;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den3, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $20;
    mul.rn.ftz.f32 %rough, $20, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $20;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den4, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $21;
    mul.rn.ftz.f32 %rough, $21, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $21;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den5, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $22;
    mul.rn.ftz.f32 %rough, $22, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $22;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den6, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $23;
    mul.rn.ftz.f32 %rough, $23, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $23;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den7, %root, 0f2edbe6ff;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $8;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $9;
    rcp.approx.ftz.f32 %inverse, %den2;
    mul.rn.f32 $2, %inverse, $10;
    rcp.approx.ftz.f32 %inverse, %den3;
    mul.rn.f32 $3, %inverse, $11;
    rcp.approx.ftz.f32 %inverse, %den4;
    mul.rn.f32 $4, %inverse, $12;
    rcp.approx.ftz.f32 %inverse, %den5;
    mul.rn.f32 $5, %inverse, $13;
    rcp.approx.ftz.f32 %inverse, %den6;
    mul.rn.f32 $6, %inverse, $14;
    rcp.approx.ftz.f32 %inverse, %den7;
    mul.rn.f32 $7, %inverse, $15;
    bra.uni L__adam_divide_join;
    L__adam_root_original:
    or.b32 %moments, $16, $17;
    lop3.b32 %moments, %moments, $18, $19, 0xfe;
    lop3.b32 %moments, %moments, $20, $21, 0xfe;
    lop3.b32 %moments, %moments, $22, $23, 0xfe;
    and.b32 %moments, %moments, 0x7fffffff;
    setp.ne.u32 %hasMoment, %moments, 0;
    vote.sync.any.pred %warpHasMoment, %hasMoment, 0xffffffff;
    @!%warpHasMoment bra.uni L__adam_zero_direction;
    sqrt.rn.f32 %root, $16;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $17;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $18;
    add.rn.f32 %den2, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $19;
    add.rn.f32 %den3, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $20;
    add.rn.f32 %den4, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $21;
    add.rn.f32 %den5, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $22;
    add.rn.f32 %den6, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $23;
    add.rn.f32 %den7, %root, 0f2edbe6ff;
    L__adam_roots_done:
    and.b32 %denAnd, %den0, %den1;
    lop3.b32 %denAnd, %denAnd, %den2, %den3, 0x80;
    lop3.b32 %denAnd, %denAnd, %den4, %den5, 0x80;
    lop3.b32 %denAnd, %denAnd, %den6, %den7, 0x80;
    or.b32 %denOr, %den0, %den1;
    lop3.b32 %denOr, %denOr, %den2, %den3, 0xfe;
    lop3.b32 %denOr, %denOr, %den4, %den5, 0xfe;
    lop3.b32 %denOr, %denOr, %den6, %den7, 0xfe;
    and.b32 %denAnd, %denAnd, 0x20000000;
    and.b32 %denOr, %denOr, 0xc0000000;
    xor.b32 %denAnd, %denAnd, 0x20000000;
    or.b32 %shifted, %denAnd, %denOr;
    setp.ne.u32 %invalid, %shifted, 0;
    setp.eq.or.u32 %invalid, %mOr, 0x7f800000, %invalid;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_divide_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $8;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $9;
    rcp.approx.ftz.f32 %inverse, %den2;
    mul.rn.f32 $2, %inverse, $10;
    rcp.approx.ftz.f32 %inverse, %den3;
    mul.rn.f32 $3, %inverse, $11;
    rcp.approx.ftz.f32 %inverse, %den4;
    mul.rn.f32 $4, %inverse, $12;
    rcp.approx.ftz.f32 %inverse, %den5;
    mul.rn.f32 $5, %inverse, $13;
    rcp.approx.ftz.f32 %inverse, %den6;
    mul.rn.f32 $6, %inverse, $14;
    rcp.approx.ftz.f32 %inverse, %den7;
    mul.rn.f32 $7, %inverse, $15;
    bra.uni L__adam_divide_join;
    L__adam_divide_fallback:
    div.full.f32 $0, $8, %den0;
    div.full.f32 $1, $9, %den1;
    div.full.f32 $2, $10, %den2;
    div.full.f32 $3, $11, %den3;
    div.full.f32 $4, $12, %den4;
    div.full.f32 $5, $13, %den5;
    div.full.f32 $6, $14, %den6;
    div.full.f32 $7, $15, %den7;
    bra.uni L__adam_divide_join;
    L__adam_zero_direction:
    add.rn.f32 %den0, $16, 0f2edbe6ff;
    setp.eq.u32 %invalid, %mOr, 0x7f800000;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_zero_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $8;
    mul.rn.f32 $1, %inverse, $9;
    mul.rn.f32 $2, %inverse, $10;
    mul.rn.f32 $3, %inverse, $11;
    mul.rn.f32 $4, %inverse, $12;
    mul.rn.f32 $5, %inverse, $13;
    mul.rn.f32 $6, %inverse, $14;
    mul.rn.f32 $7, %inverse, $15;
    bra.uni L__adam_divide_join;
    L__adam_zero_fallback:
    div.full.f32 $0, $8, %den0;
    div.full.f32 $1, $9, %den0;
    div.full.f32 $2, $10, %den0;
    div.full.f32 $3, $11, %den0;
    div.full.f32 $4, $12, %den0;
    div.full.f32 $5, $13, %den0;
    div.full.f32 $6, $14, %den0;
    div.full.f32 $7, $15, %den0;
    L__adam_divide_join:
    }""",
            constraints="=f,=f,=f,=f,=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f", args=[m_new, v_new],
            dtype=gl.float32, is_pure=True, pack=8,
        )
    elif BLOCK == 512:
        direction = gl.inline_asm_elementwise(
            """{
    .reg .b32 %mOr, %denAnd, %denOr, %shifted, %moments, %rootAnd, %rootOr;
    .reg .f32 %inverse, %root, %den<2>, %rsq, %rough, %negativeRough, %halfRsq, %residual;
    .reg .pred %invalid, %slow, %hasMoment, %warpHasMoment, %specialRoot, %warpSpecialRoot;
    and.b32 %rootAnd, $4, $5;
    or.b32 %rootOr, $4, $5;
    and.b32 %rootAnd, %rootAnd, 0x20000000;
    and.b32 %rootOr, %rootOr, 0xc0000000;
    xor.b32 %rootAnd, %rootAnd, 0x20000000;
    or.b32 %shifted, %rootAnd, %rootOr;
    setp.ne.u32 %specialRoot, %shifted, 0;
    or.b32 %mOr, $2, $3;
    and.b32 %mOr, %mOr, 0x7f800000;
    setp.eq.or.u32 %specialRoot, %mOr, 0x7f800000, %specialRoot;
    vote.sync.any.pred %warpSpecialRoot, %specialRoot, 0xffffffff;
    @%warpSpecialRoot bra.uni L__adam_root_original;
    rsqrt.approx.ftz.f32 %rsq, $4;
    mul.rn.ftz.f32 %rough, $4, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $4;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $5;
    mul.rn.ftz.f32 %rough, $5, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $5;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $2;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $3;
    bra.uni L__adam_divide_join;
    L__adam_root_original:
    or.b32 %moments, $4, $5;
    and.b32 %moments, %moments, 0x7fffffff;
    setp.ne.u32 %hasMoment, %moments, 0;
    vote.sync.any.pred %warpHasMoment, %hasMoment, 0xffffffff;
    @!%warpHasMoment bra.uni L__adam_zero_direction;
    sqrt.rn.f32 %root, $4;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $5;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    L__adam_roots_done:
    and.b32 %denAnd, %den0, %den1;
    or.b32 %denOr, %den0, %den1;
    and.b32 %denAnd, %denAnd, 0x20000000;
    and.b32 %denOr, %denOr, 0xc0000000;
    xor.b32 %denAnd, %denAnd, 0x20000000;
    or.b32 %shifted, %denAnd, %denOr;
    setp.ne.u32 %invalid, %shifted, 0;
    setp.eq.or.u32 %invalid, %mOr, 0x7f800000, %invalid;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_divide_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $2;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $3;
    bra.uni L__adam_divide_join;
    L__adam_divide_fallback:
    div.full.f32 $0, $2, %den0;
    div.full.f32 $1, $3, %den1;
    bra.uni L__adam_divide_join;
    L__adam_zero_direction:
    add.rn.f32 %den0, $4, 0f2edbe6ff;
    setp.eq.u32 %invalid, %mOr, 0x7f800000;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_zero_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $2;
    mul.rn.f32 $1, %inverse, $3;
    bra.uni L__adam_divide_join;
    L__adam_zero_fallback:
    div.full.f32 $0, $2, %den0;
    div.full.f32 $1, $3, %den0;
    L__adam_divide_join:
    }""",
            constraints="=f,=f,f,f,f,f", args=[m_new, v_new],
            dtype=gl.float32, is_pure=True, pack=2,
        )
    else:
        direction = m_new / (gl.sqrt_rn(v_new) + 1e-10)
"#

/-- Admit the exact guarded kernel and project its file onto the packed
reference expected by the common driver linker. The native certificates
supply the execution of the eight checked launch profiles. -/
def packedSource (source : SourceImage) : Option SourceImage := do
  guard (source.lookup privateNativeAdam == some guardedAdamSource)
  return source.map fun (name, text) =>
    (name, if name == privateNativeAdam then candidateNativeAdam else text)

structure Sources where
  actual : SourceImage
  linked : LinkedSources
  recognized : packedSource actual = some linked.candidate

def Sources.check (actual : SourceImage) (linked : LinkedSources) : Option Sources :=
  if recognized : packedSource actual = some linked.candidate then
    some ⟨actual, linked, recognized⟩
  else none

structure Programs where
  remainder : ArtifactTraining.Kernels
  valueAdam : CombinedPrograms

inductive Program where
  | baseline (source : SourceImage) (valueAdam : CombinedPrograms)
  | optimized (source : SourceImage) (programs : Programs)

private def Program.source : Program → SourceProgram
  | .baseline source _ => ⟨.reference, source⟩
  | .optimized source _ => ⟨.candidate, source⟩

private def withValueEvent (calls : PatchedCalls) (value : ValueAdam.EventKernel) : PatchedCalls :=
  { calls with sparseLanding := {calls.sparseLanding with valueEvent := value} }

private def Program.calls (program : Program) (divider : NativeDivider) : PatchedCalls :=
  match program with
  | .baseline _ valueAdam => withValueEvent referenceCalls (valueAdam.run true divider)
  | .optimized _ kernels => withValueEvent kernels.remainder.calls (kernels.valueAdam.run false divider)

private theorem checked_calls_eq (source : SourceImage) (programs : Programs) (divider : NativeDivider) :
    (Program.optimized source programs).calls divider =
      withValueEvent referenceCalls (programs.valueAdam.run true divider) := by
  simp only [Program.calls, programs.remainder.calls_eq]
  congr 1
  funext pull indices order arithmetic launch memory
  exact programs.valueAdam.run_eq divider pull indices order arithmetic launch memory

/-- Both executions share the reciprocal, reciprocal square root, the
hardware NaN-payload effects. The
checked lowering evaluates each side's control, helper and floating-point terms. -/
def Program.run {State Value : Type} (program : Program) (world : World State Value)
    (divider : NativeDivider) (input : TrainingInput) : Option (Result State) :=
  match program with
  | .baseline .. => runWith program.source world input (program.calls divider)
  | .optimized source _ => do
    let packed ← packedSource source
    runWith ⟨.candidate, packed⟩ world input (program.calls divider)

inductive Statement where
  | training (statement : ArtifactTraining.Statement)
  | hopperValueAdam (period : ValueAdam.Elementwise.Period) (tile : ValueAdam.Tile) (masked : Bool)
      (mode : ValueAdam.DivisorBand.ZeroMode)
      (before after beforeSASS afterSASS : String)

private def Programs.body (programs : Programs) : List Statement :=
  let ordinary := programs.remainder.body.filter fun stmt =>
    match stmt with
    | .valueAdamNativeUpdate _ _ true _ _ => false
    | _ => true
  ordinary.map .training ++
    [.two, .four].flatMap fun period =>
      [.x512, .x1024].flatMap fun tile =>
        [false, true].map fun masked =>
          let p := programs.valueAdam.consumer period tile masked
          .hopperValueAdam period tile masked p.mode p.oldPTX p.ptx p.oldSASS p.sass

def sameRun (left right : Program) : Prop :=
  ∀ (State Value : Type) (world : World State Value) (divider : NativeDivider) (input : TrainingInput),
    left.run world divider input = right.run world divider input

instance : Spec.ProgramSyntax Program where
  Statement := Statement
  Signature := List TrainingIO × List TrainingIO
  signature := fun program => Spec.ProgramSyntax.signature program.source
  body := fun program =>
    (Spec.ProgramSyntax.body program.source).map (Statement.training ∘ ArtifactTraining.Statement.source) ++
      match program with
      | .baseline _ _ => []
      | .optimized _ kernels => kernels.body
  structural := some sameRun

theorem linked_run_eq (src : Sources) (programs : Programs)
    {State Value : Type} (world : World State Value) (divider : NativeDivider)
    (input : TrainingInput) :
    (Program.optimized src.actual programs).run world divider input =
      (Program.baseline src.linked.reference programs.valueAdam).run world divider input := by
  simp only [Program.run, src.recognized]
  change runWith ⟨.candidate, src.linked.candidate⟩ world input
      ((Program.optimized src.actual programs).calls divider) =
    runWith ⟨.reference, src.linked.reference⟩ world input
      ((Program.baseline src.linked.reference programs.valueAdam).calls divider)
  rw [checked_calls_eq]
  simp only [runWith, src.linked.candidateOK, src.linked.referenceOK, Program.calls]

/-- The dispatch uses separate content caches for the wrapper shape,
wrapper fingerprint, and PTX fingerprint. Substituting any cache's
inspection trace preserves the downstream training observation. -/
theorem cached_dispatch_run_eq {State Value ε Check : Type}
    (src : Sources) (programs : Programs) (inspect : String → Except ε Check)
    (memo : DispatchMemo.Memo inspect) (reads : List String)
    (worldFrom : List (Except ε Check) → World State Value)
    (divider : NativeDivider) (input : TrainingInput) :
    (Program.optimized src.actual programs).run (worldFrom (memo.trace reads)) divider input =
      (Program.baseline src.linked.reference programs.valueAdam).run
        (worldFrom (reads.map inspect)) divider input := by
  rw [DispatchMemo.Memo.trace_exact]
  exact linked_run_eq src programs _ divider input

specification combined_training_end_to_end (src : Sources) (programs : Programs) :
    Program.optimized src.actual programs ≡[([] : Spec.Assumptions Statement)]
      Program.baseline src.linked.reference programs.valueAdam := by
  refine Spec.FloatingPoint.ofStructural (structural := sameRun) rfl rfl ?_
  exact fun _ _ world divider input => linked_run_eq src programs world divider input

theorem final_validation_eq (src : Sources) (programs : Programs)
    {State Value : Type} (world : World State Value) (divider : NativeDivider) (input : TrainingInput) :
    ((Program.optimized src.actual programs).run world divider input).bind Result.finalValidation =
      ((Program.baseline src.linked.reference programs.valueAdam).run world divider input).bind Result.finalValidation := by
  rw [linked_run_eq src programs]

#print_fp_assumptions combined_training_end_to_end
#axiomsClean cached_dispatch_run_eq
#axiomsClean combined_training_end_to_end
#axiomsClean final_validation_eq

end VeriTile.Bench.Models.NanoGPT.Track1.Driver.CombinedTraining
