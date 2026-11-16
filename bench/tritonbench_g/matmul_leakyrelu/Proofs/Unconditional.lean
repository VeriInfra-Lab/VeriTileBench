import bench.tritonbench_g.matmul_leakyrelu.Proofs.SourceIO
import VeriTile.Triton.Float.ScalarOps

/-! A countermodel of unconditional equivalence for the actual source pair.

The interpreter truncates FP32 inputs to TF32 and accumulates each product in
FP32 with round-toward-zero. The separate scalar add and binary16 output use
round-to-nearest-even. This explicitly defined software interpretation is one
of the models quantified by SourcePair.IOEquivalent; no GPU-conformance claim
or measured output is assumed.

The legal 1x96 and 96x1 inputs use three K32 blocks. The first creates
65536+3/128, the second adds 65536 and exposes different rounding boundaries,
and the third subtracts 131072. The final normal binary16 values differ:
1/32 (encoding 0x2800) and 1/64 (encoding 0x2400). Existing execution theorems connect
these arithmetic recurrences to the complete, unchanged source kernels.
-/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Unconditional
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 8000000
abbrev Word := FP.Value .fp32

def zero : Word := FP.Value.zero .fp32
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def scale : Word := FP.Value.ofBits .fp32 0x47800000 -- 65536
-- The third block cancels the large common part of the accumulator.
def negativeDoubleScale : Word := FP.Value.ofBits .fp32 0xc8000000 -- -131072
def cross : Word := FP.Value.ofBits .fp32 0x42400000 -- 48
def reciprocal2048 : Word := FP.Value.ofBits .fp32 0x3a000000 -- 2^-11
def rtz : FP.Config := ⟨.towardZero,Bool.false,Bool.false⟩
local instance : Inhabited Word := ⟨zero⟩

def tf32 (x : Word) : Word :=
  if x.exponent=255 then x else FP.Value.ofBits .fp32 (x.raw/8192*8192)

def dotSeed (a b : Fin 32 → Word) (acc : Word) : Word :=
  (List.finRange 32).foldl (fun value k => FP.fma rtz (tf32 (a k)) (tf32 (b k)) value) acc

def nearestQuotient (n d : Nat) : Nat :=
  let q := n/d
  let r := n%d
  if 2*r>d ∨ (2*r=d ∧ q%2=1) then q+1 else q

/-- Nearest-even binary16 value, represented by its exact binary32 widening.
NaNs use the software model's canonical payload. -/
def half (x : Word) : Word :=
  let e := x.exponent
  if e = 255 then if x.fraction = 0 then FP.Value.infinity .fp32 x.sign else FP.Value.qNaN .fp32
  else if 142 < e then FP.Value.infinity .fp32 x.sign
  else if e < 113 then
    let m := nearestQuotient (if e=0 then x.fraction else 2^23+x.fraction) (2^(126-e))
    FP.round .fp32 .ieeeValue ((if x.sign then -(m : Rat) else (m : Rat))*FP.pow2 (-24)) x.sign
  else
    let m := nearestQuotient (2^23+x.fraction) (2^13)
    if e=142 ∧ m=2048 then FP.Value.infinity .fp32 x.sign
    else FP.round .fp32 .ieeeValue ((if x.sign then -(m : Rat) else (m : Rat))*FP.pow2 ((e : Int)-137)) x.sign
def nonnegative (x : Word) := FP.eq .ieeeValue zero x || FP.lt .ieeeValue zero x

def numbers : Whole.Numerics Word where
  zero := zero
  add := FP.add .ieeeValue
  dot := fun a b => dotSeed a b zero
  nonnegative := nonnegative
  slopeMul := FP.mul .ieeeValue (FP.Value.ofBits .fp32 0x3c23d70a)
  castHalf := half

def fusedNumbers : Whole.Contracted.Numerics Word where
  zero := zero
  dotAcc := dotSeed
  nonnegative := nonnegative
  rightSlope := fun x => FP.mul .ieeeValue x (FP.Value.ofBits .fp32 0x3c23d70a)
  castHalf := half

def request : Whole.Request where
  A := "a";B := "b";C := "c"
  rows := 1;columns := 1;inner := 96;innerB := 96
  strideAM := 96;strideAK := 1;strideBK := 1;strideBN := 1
  aContiguous := Bool.true;activation := ""

def a (k : Nat) : Word :=
  if k=0 ∨ k=32 then scale else if k=1 then cross else if k=64 then negativeDoubleScale else zero
def b (k : Nat) : Word :=
  if k=0 ∨ k=32 ∨ k=64 then one else if k=1 then reciprocal2048 else zero

theorem a_tf32_exact (k : Nat) : tf32 (a k) = a k := by
  unfold a
  split_ifs <;> decide +kernel

theorem b_tf32_exact (k : Nat) : tf32 (b k) = b k := by
  unfold b
  split_ifs <;> decide +kernel

def initial : State Word where
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype
  mem := fun region k => Cell.mk .fp32 (if region="a" then a k else if region="b" then b k else zero)

theorem original_value : Whole.reference numbers request initial 0 0 =
    FP.Value.ofBits .fp32 0x3d000000 := by
  decide +kernel

theorem contracted_value : Whole.Contracted.reference fusedNumbers request initial 0 0 =
    FP.Value.ofBits .fp32 0x3c800000 := by
  decide +kernel

theorem values_different : Whole.reference numbers request initial 0 0 ≠
    Whole.Contracted.reference fusedNumbers request initial 0 0 := by
  rw [original_value,contracted_value]
  decide

/-- A fully specified FP32 source interpreter. Dot is a left-to-right fused
FP32 accumulation with round-toward-zero at each term. The standalone scalar
add and final binary16 conversion use round-to-nearest-even. This is an
explicit software countermodel, not an assertion of GPU conformance. -/
def scalar : Algebra Word where
  literal := fun _ _ r => if r = 0 then zero else FP.Value.ofBits .fp32 0x3c23d70a
  negInf := FP.Value.infinity .fp32 Bool.true
  binary := fun _ _ op => match op with
    | .add => FP.add .ieeeValue
    | .sub => FP.sub .ieeeValue
    | .mul => FP.mul .ieeeValue
    | .div => FP.div .ieeeValue
    | _ => fun _ _ => zero
  unary := fun _ _ x => x
  cast := fun _ _ dst x => if dst = .fp16 then half x else x
  fromNat := fun _ n => FP.round .fp32 .ieeeValue (n : Rat)
  fromInt := fun _ n => FP.round .fp32 .ieeeValue (Rat.ofInt n)
  fp32Bits := fun bits => ⟨bits.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => zero
  reduceSum := fun _ {_} _ _ _ _ => zero

def algebra : Algebra Word := Pair.fp32Algebra .ieeeValue scalar

def dot : FPValue.Dot Word where
  apply := fun _ _ _ _ _ {batch} {m} {k} {n} =>
    match batch,m,k,n with
    | [],32,32,32 => fun a b i => numbers.dot (fun j => a (i.1,j,())) (fun j => b (j,i.2.1,()))
    | _,_,_,_ => fun _ _ _ => zero
  le := fun _ _ x y => FP.eq .ieeeValue x y || FP.lt .ieeeValue x y

def fusedDot : FPFused.DotAcc Word where
  apply := fun _ _ _ _ _ {batch} {m} {k} {n} =>
    match batch,m,k,n with
    | [],32,32,32 => fun a b acc i => dotSeed (fun j => a (i.1,j,())) (fun j => b (j,i.2.1,())) (acc i)
    | _,_,_,_ => fun _ _ _ _ => zero
  le := dot.le

theorem original_primitives : Whole.PrimitiveLaws algebra dot numbers := by
  constructor
  · simp [algebra,Pair.fp32Algebra,scalar,numbers]
  · simp [algebra,Pair.fp32Algebra,scalar,numbers]
    norm_num
  · intro x y; rfl
  · intro a b i; simp [dot,numbers]
  · intro x; norm_num [dot,algebra,Pair.fp32Algebra,scalar,numbers,nonnegative]
  · intro x; norm_num [algebra,Pair.fp32Algebra,scalar,numbers]
  · intro x; rfl

theorem contracted_primitives : Whole.Contracted.PrimitiveLaws algebra fusedDot fusedNumbers := by
  constructor
  · simp [algebra,Pair.fp32Algebra,scalar,fusedNumbers]
  · simp [algebra,Pair.fp32Algebra,scalar,fusedNumbers]
    norm_num
  · intro a b acc i; simp [fusedDot,fusedNumbers]
  · intro x; norm_num [fusedDot,dot,algebra,Pair.fp32Algebra,scalar,fusedNumbers,nonnegative]
  · intro x; norm_num [algebra,Pair.fp32Algebra,scalar,fusedNumbers]
  · intro x; rfl

def schedule : Schedule request where
  order := [0]
  complete := by decide

theorem source_counterexample : ∃ left right,
    Whole.runPrograms algebra dot request schedule.order initial = some left ∧
    Whole.Contracted.runPrograms algebra fusedDot request schedule.order initial = some right ∧
    left.mem "c" 0 = Cell.mk .fp16 (FP.Value.ofBits .fp32 0x3d000000) ∧
    right.mem "c" 0 = Cell.mk .fp16 (FP.Value.ofBits .fp32 0x3c800000) ∧
    left.mem "c" 0 ≠ right.mem "c" 0 := by
  obtain ⟨left,hl,vl,fl⟩ := OriginalReal.scheduled_grid algebra dot numbers original_primitives
    request schedule initial (by decide) (by decide)
  obtain ⟨right,hr,vr,fr⟩ := scheduled_grid algebra fusedDot fusedNumbers contracted_primitives
    request schedule initial (by decide) (by decide)
  have ll := vl 0 (by decide) 0 (by decide)
  have rr := vr 0 (by decide) 0 (by decide)
  change left.mem "c" 0 = Cell.mk .fp16 (Whole.reference numbers request initial 0 0) at ll
  change right.mem "c" 0 = Cell.mk .fp16 (Whole.Contracted.reference fusedNumbers request initial 0 0) at rr
  rw [original_value] at ll
  rw [contracted_value] at rr
  refine ⟨left,right,hl,hr,ll,rr,?_⟩
  rw [ll,rr]
  intro h
  have payload : (FP.Value.ofBits .fp32 0x3d000000 : Word) = FP.Value.ofBits .fp32 0x3c800000 := by
    injection h
  exact (by decide : (FP.Value.ofBits .fp32 0x3d000000 : Word) ≠ FP.Value.ofBits .fp32 0x3c800000) payload


abbrev parameters : KernelInterface.Parameters where
  A := "a";B := "b";C := "c"
  rows := 1;inner := 96;columns := 1
  strideAM := 96;strideAK := 1;strideBK := 1;strideBN := 1
  activation := ""

def order : Schedule parameters.request := schedule

theorem index_domain : Whole.IndexDomain parameters.request := by
  constructor <;> decide

def inputs : Interface.Inputs Word parameters :=
  Fin.cases (fun j => a j.2.1.val) (Fin.cases (fun j => b j.1.val) (fun i => i.elim0))

def bounds (r : RegionName) : Nat :=
  if r="a" ∨ r="b" then 96 else if r="c" then 1 else 0

theorem input_binding : (Original.io algebra dot parameters order).InputBinding inputs initial := by
  intro port j _
  fin_cases port
  · rcases j with ⟨row,k,⟨⟩⟩
    have hz : row.val = 0 := congrArg Fin.val (Subsingleton.elim row (0 : Fin 1))
    simp [Original.io,Interface.memoryIO,Interface.inputs,parameters,inputs,initial,Cell.read,hz,mapValue] <;> rfl
  · rcases j with ⟨k,col,⟨⟩⟩
    have hz : col.val = 0 := congrArg Fin.val (Subsingleton.elim col (0 : Fin 1))
    simp [Original.io,Interface.memoryIO,Interface.inputs,parameters,inputs,initial,Cell.read,hz,mapValue] <;> rfl

theorem input_bounds : (Original.io algebra dot parameters order).InBounds bounds inputs := by
  constructor
  · intro r hr
    simp only [Original.io,Interface.memoryIO] at hr
    simp only [parameters,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
    simp [bounds,hr.1,hr.2.1,hr.2.2]
  constructor
  · intro port j _
    fin_cases port
    · rcases j with ⟨row,k,⟨⟩⟩
      have hz : row.val=0 := congrArg Fin.val (Subsingleton.elim row (0 : Fin 1))
      simpa [Original.io,Interface.memoryIO,Interface.inputs,parameters,bounds,hz] using k.isLt
    · rcases j with ⟨k,col,⟨⟩⟩
      have hz : col.val=0 := congrArg Fin.val (Subsingleton.elim col (0 : Fin 1))
      simpa [Original.io,Interface.memoryIO,Interface.inputs,parameters,bounds,hz] using k.isLt
  constructor
  · intro port j _
    fin_cases port
    rcases j with ⟨row,col,⟨⟩⟩
    have hr : row.val=0 := congrArg Fin.val (Subsingleton.elim row (0 : Fin 1))
    have hc : col.val=0 := congrArg Fin.val (Subsingleton.elim col (0 : Fin 1))
    simp [Original.io,Interface.memoryIO,Interface.outputs,parameters,bounds,hr,hc,
      Whole.outputOffset,KernelInterface.Parameters.request]
  · intro port
    exact port.elim0

theorem source_schedules_safe :
    ScheduleSafe bounds ((Original.io algebra dot parameters order).schedule inputs) initial ∧
    ScheduleSafe bounds ((Interface.io algebra fusedDot parameters order).schedule inputs) initial :=
  ⟨OriginalIOSafety.whole algebra dot parameters order inputs bounds input_bounds initial,
   IOSafety.whole algebra fusedDot parameters order inputs bounds input_bounds initial⟩

theorem empty_derivation {left right : List ComputeStmt}
    (h : VeriTile.Spec.Derivation ([] : VeriTile.Spec.Assumptions ComputeStmt) left right) : left = right := by
  induction h with
  | refl => rfl
  | atom entry member _ => simp at member
  | symm _ ih => exact ih.symm
  | trans _ _ ih1 ih2 => exact ih1.trans ih2
  | frame before after _ ih => rw [ih]
  | strict certificate => exact False.elim certificate

def opSeeded {dtype : TileDType} {shape : TileShape} (op : Op dtype shape) : Bool :=
  match op with
  | .dotAcc .. => Bool.true
  | _ => Bool.false

def statementSeeded (stmt : ComputeStmt) : Bool :=
  match stmt with
  | .assign _ _ _ (.compute (.alg _ value)) =>
    opSeeded value
  | _ => Bool.false

theorem empty_model : SourcePair.Models [] algebra dot fusedDot := by
  intro h
  have same := empty_derivation h
  have different : SourcePair.separateFragment ≠ SourcePair.contractedFragment := by
    intro eq
    have h := congrArg (List.map statementSeeded) eq
    change [Bool.false] = [Bool.true] at h
    cases h
  exact False.elim (different same)

theorem mapCell_id (c : Cell Word) : mapCell id c = c := by
  cases c with
  | mk dtype value => cases dtype <;> rfl

theorem concrete_io_disproved :
    ¬(Original.io algebra dot parameters order).Equivalent (Pair.variant algebra dot fusedDot parameters order) := by
  intro h
  obtain ⟨left,right,hl,hr,ll,rr,ne⟩ := source_counterexample
  obtain ⟨_,_,left',right',hl',hr',out,_,_⟩ := h inputs trivial bounds input_bounds initial input_binding
  have sl : runSchedule ((Original.io algebra dot parameters order).schedule inputs) initial = some left :=
    (OriginalFacts.schedule_run algebra dot parameters order.order initial).trans hl
  have sr : runSchedule ((Pair.variant algebra dot fusedDot parameters order).schedule inputs) initial = some right :=
    (IOFacts.schedule_run algebra fusedDot parameters order.order initial).trans hr
  have el := Option.some.inj (sl.symm.trans hl')
  have er := Option.some.inj (sr.symm.trans hr')
  subst left';subst right'
  have observed := out (0 : Fin 1) ((0 : Fin 1),(0 : Fin 1),()) trivial
  change mapCell id (left.mem "c" 0) = mapCell id (right.mem "c" 0) at observed
  rw [mapCell_id,mapCell_id] at observed
  exact ne observed

theorem unconditional_io_disproved : ¬ SourcePair.IOEquivalent []
    (SourcePair.original parameters order) (SourcePair.contracted parameters order) := by
  intro h
  have result := h .ieeeValue scalar dot fusedDot numbers original_primitives rfl empty_model
  change (Original.io algebra dot parameters order).Equivalent (Pair.variant algebra dot fusedDot parameters order) at result
  exact concrete_io_disproved result


def observed (program : SourcePair.IOProgram parameters order) : Option (Cell Word) :=
  (runSchedule ((SourcePair.programIO .ieeeValue scalar dot fusedDot program).schedule inputs) initial).map
    (fun final => final.mem "c" 0)

theorem numerical_preserves_observed {left right : SourcePair.IOProgram parameters order}
    (h : SourcePair.IOEquivalent [] left right) : observed left = observed right := by
  have eqv := h .ieeeValue scalar dot fusedDot numbers original_primitives rfl empty_model
  obtain ⟨_,_,l,r,hl,hr,out,_,_⟩ := eqv inputs trivial bounds input_bounds initial input_binding
  have point := out (0 : Fin 1) ((0 : Fin 1),(0 : Fin 1),()) trivial
  change mapCell id (l.mem "c" 0) = mapCell id (r.mem "c" 0) at point
  rw [mapCell_id,mapCell_id] at point
  change (runSchedule ((SourcePair.programIO .ieeeValue scalar dot fusedDot left).schedule inputs) initial).map
      (fun final => final.mem "c" 0) =
    (runSchedule ((SourcePair.programIO .ieeeValue scalar dot fusedDot right).schedule inputs) initial).map
      (fun final => final.mem "c" 0)
  have lmap := congrArg (fun state : Option (State Word) => state.map (fun final => final.mem "c" 0)) hl
  have rmap := congrArg (fun state : Option (State Word) => state.map (fun final => final.mem "c" 0)) hr
  exact lmap.trans ((congrArg some point).trans rmap.symm)

theorem derivation_preserves_observed {left right : SourcePair.IOProgram parameters order}
    (h : VeriTile.Spec.ProgramDerivation (fun _ _ : SourcePair.IOProgram parameters order => False) [] left right) :
    observed left = observed right := by
  induction h with
  | refl => rfl
  | «syntax» signature context proof =>
    change _ = _ at context
    cases context
    rfl
  | structural signature proof => exact False.elim proof
  | numerical signature proof => exact numerical_preserves_observed proof
  | symm _ ih => exact ih.symm
  | trans _ _ ih1 ih2 => exact ih1.trans ih2

open scoped VeriTile.Spec

theorem unconditional_source_pair_disproved :
    ¬(SourcePair.original parameters order ≡[([] : VeriTile.Spec.Assumptions ComputeStmt)]
      SourcePair.contracted parameters order) := by
  intro h
  have derivation : VeriTile.Spec.ProgramDerivation
      (fun _ _ : SourcePair.IOProgram parameters order => False) []
      (SourcePair.original parameters order) (SourcePair.contracted parameters order) := h.derivation
  have same := derivation_preserves_observed derivation
  obtain ⟨left,right,hl,hr,ll,rr,ne⟩ := source_counterexample
  have sl : runSchedule
      ((SourcePair.programIO .ieeeValue scalar dot fusedDot (SourcePair.original parameters order)).schedule inputs)
      initial = some left :=
    (OriginalFacts.schedule_run algebra dot parameters order.order initial).trans hl
  have sr : runSchedule
      ((SourcePair.programIO .ieeeValue scalar dot fusedDot (SourcePair.contracted parameters order)).schedule inputs)
      initial = some right :=
    (IOFacts.schedule_run algebra fusedDot parameters order.order initial).trans hr
  have lmap := congrArg (fun state : Option (State Word) => state.map (fun final : State Word => final.mem "c" 0)) sl
  have rmap := congrArg (fun state : Option (State Word) => state.map (fun final : State Word => final.mem "c" 0)) sr
  exact ne (Option.some.inj (lmap.symm.trans (same.trans rmap)))

#axiomsClean source_counterexample
#axiomsClean unconditional_source_pair_disproved

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Unconditional
