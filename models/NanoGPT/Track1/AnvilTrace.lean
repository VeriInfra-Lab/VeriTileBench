import models.NanoGPT.Track1.AnvilEpilogue
import models.NanoGPT.Track1.SourceAnchors
import models.NanoGPT.Track1.ValueAdam

/-! Two Hopper launches reduce the diagonal through a scratch row of FP32
partial sums. The fused version preserves the ordered FP32 additions and
publishes the rounded denominator and square in a two-word row per matrix. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilTrace
open VeriTile.Triton FP
open AnvilEpilogue (Bank)
open TailNorm (F32 BF16 zero widen narrow)

inductive Tree where
  | zero
  | diagonal (offset : Nat)
  | add (left right : Tree)
  deriving DecidableEq

def butterfly (masks : List Nat) (values : Nat → Tree) : Nat → Tree :=
  masks.foldl (fun prior bit lane => .add (prior lane) (prior (lane ^^^ bit))) values

def lanePair (offset : Nat) : Tree :=
  .add
    (.add (.add (.diagonal offset) .zero) (.diagonal (offset + 64)))
    (.add (.add (.diagonal (offset + 32)) .zero) (.diagonal (offset + 96)))

def originalPartial : Tree :=
  let warp := butterfly [16, 8, 4, 2] fun tid => lanePair (tid / 2)
  .add (warp 0) (warp 32)

def warpPartial : Tree := butterfly [8, 4, 2, 1, 16] lanePair 0

set_option maxRecDepth 4096 in
theorem same_partial_tree : warpPartial = originalPartial := by
  decide

def Tree.eval (load : Nat → F32) : Tree → F32
  | .zero => TailNorm.zero
  | .diagonal i => load i
  | .add a b => FP.add Config.ieeeValue (a.eval load) (b.eval load)

def width : Bank → Nat
  | .qk => 256
  | .vo | .mlp => 768

def chunks (bank : Bank) : Nat := width bank / 128

abbrev Trace := (bank : Bank) → (Nat → BF16) → Fin bank.dimensions.1 → F32

def partialSum (tree : Tree) (bank : Bank) (gram : Nat → BF16) (batch chunk : Nat) : F32 :=
  let d := width bank
  tree.eval fun lane => widen (gram (batch * d * d + (chunk * 128 + lane) * (d + 1)))

def lastTree : Bank → Tree
  | .qk => .add (.diagonal 0) (.diagonal 1)
  | .vo | .mlp => butterfly [4, 2, 1] (fun i => if i < 6 then .diagonal i else .zero) 0

def scratchRow (tree : Tree) (bank : Bank) (gram : Nat → BF16) (batch : Nat) : List F32 :=
  (List.range (chunks bank)).map (partialSum tree bank gram batch)

def reference : Trace := fun bank gram batch =>
  let scratch := scratchRow originalPartial bank gram batch.val
  (lastTree bank).eval fun chunk => scratch[chunk]?.getD zero

def fused : Trace := fun bank gram batch =>
  (lastTree bank).eval fun chunk => partialSum warpPartial bank gram batch.val chunk

theorem last_tree_reads (bank : Bank) (f g : Nat → F32)
    (h : ∀ i, i < chunks bank → f i = g i) :
    (lastTree bank).eval f = (lastTree bank).eval g := by
  cases bank <;> simp [lastTree, Tree.eval, butterfly, chunks, width] at h ⊢
  all_goals simp [h]

theorem fused_equal : fused = reference := by
  funext bank gram batch
  unfold fused reference
  rw [same_partial_tree]
  apply last_tree_reads
  intro i hi
  simp [scratchRow, hi]

def denominator (trace : F32) : F32 :=
  let (margin, epsilon) := Driver.anvilFrobeniusWords
  FP.fma Config.ieeeValue (ValueAdam.sqrtRN trace)
    (Value.ofBits .fp32 margin) (Value.ofBits .fp32 epsilon)

structure Divisors where
  denominator : F32
  square : F32

def divisorWords (trace : F32) : Divisors :=
  let d := denominator trace
  ⟨d, FP.mul Config.ieeeValue d d⟩

/-- The contiguous FP32 scratch table has shape `[batch, 2, 1]`. -/
def packed {batch : Nat} (rows : Fin batch → Divisors) (offset : Fin (2 * batch)) : F32 :=
  let row := rows ⟨offset.val / 2, by have := offset.isLt; omega⟩
  if offset.val % 2 == 0 then row.denominator else row.square

def readDivisors {batch : Nat} (memory : Fin (2 * batch) → F32) (b : Fin batch) : Divisors :=
  ⟨memory ⟨2 * b.val, by have := b.isLt; omega⟩,
    memory ⟨2 * b.val + 1, by have := b.isLt; omega⟩⟩

theorem packed_roundtrip {batch : Nat} (rows : Fin batch → Divisors) (b : Fin batch) :
    readDivisors (packed rows) b = rows b := by
  have h₀ : 2 * b.val / 2 = b.val := by omega
  have h₁ : (2 * b.val + 1) / 2 = b.val := by omega
  have he : 2 * b.val % 2 = 0 := by omega
  have ho : (2 * b.val + 1) % 2 = 1 := by omega
  cases b
  simp [readDivisors, packed, h₀, h₁, he, ho]

abbrev DivisorKernel := (bank : Bank) → (Nat → BF16) → Fin bank.dimensions.1 → Divisors
abbrev Dispatch := Bool → DivisorKernel

def original : Dispatch := fun _ bank gram b => divisorWords (reference bank gram b)

def publish : DivisorKernel := fun bank gram b =>
  readDivisors (packed fun i => divisorWords (fused bank gram i)) b

def optimized : Dispatch := fun eligible => if eligible then publish else original false

theorem dispatch_equal : optimized = original := by
  funext eligible bank gram b
  cases eligible
  · rfl
  · simp [optimized, publish, original, fused_equal, packed_roundtrip]

/-- Denominators and both BF16 outputs of the first-Gram normalization.
The unchanged Hopper full divide is supplied by the numerical runtime. -/
structure Input where
  eligible : Bool
  prepared : Nat → BF16
  gram : Nat → BF16
  divide : F32 → F32 → F32

structure Output (bank : Bank) where
  divisors : Fin bank.dimensions.1 → Divisors
  prepared : Fin bank.dimensions.1 → Fin (bank.dimensions.2.1 * bank.dimensions.2.2) → BF16
  gram : Fin bank.dimensions.1 → Fin (width bank * width bank) → BF16

abbrev Normalization := (bank : Bank) → Input → Output bank

def normalize (run : Dispatch) : Normalization := fun bank input =>
  let divisors := run input.eligible bank input.gram
  let cells := bank.dimensions.2.1 * bank.dimensions.2.2
  let gramCells := width bank * width bank
  ⟨divisors,
    fun b i => narrow (input.divide (widen (input.prepared (b.val * cells + i.val))) (divisors b).denominator),
    fun b i => narrow (input.divide (widen (input.gram (b.val * gramCells + i.val)))
      (divisors b).square)⟩

theorem normalized_equal (bank : Bank) (input : Input) :
    normalize optimized bank input = normalize original bank input := by
  rw [dispatch_equal]

#axiomsClean same_partial_tree
#axiomsClean fused_equal
#axiomsClean packed_roundtrip
#axiomsClean dispatch_equal
#axiomsClean normalized_equal

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilTrace
