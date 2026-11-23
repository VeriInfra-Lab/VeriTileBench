import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Models.NanoGPT.Track1
open VeriTile Triton.FP

abbrev Scalar := Value .fp32

def zero : Scalar := Value.zero .fp32
def literal (q : Rat) : Scalar := round .fp32 Config.ieeeValue q
def mul32 (a b : Scalar) : Scalar := Triton.FP.mul Config.ieeeValue a b
def orderedLt (a b : Scalar) : Bool := lt Config.ieeeValue a b
def roundedLambda (a : Scalar) : Scalar := cast .fp32 Config.ieeeValue (cast .bf16 Config.ieeeValue a)

def min32 (a b : Scalar) : Scalar := if orderedLt b a then b else a
def max32 (a b : Scalar) : Scalar := if orderedLt a b then b else a

/-- Python host gains are folded to the actual fp32 scalars at the pointwise
operations. The theorem holds over every bit pattern of the four scalars. -/
structure Scalars where
  xScale : Scalar
  gradScale : Scalar
  postGain : Scalar
  dpreGain : Scalar

/-- Track 1 has twelve MLP bank slots and eleven rows of residual lambdas.
The last ordinary slot is skipped and the parallel slot uses layer eight. -/
abbrev Slot := Fin 12
abbrev Layer := Fin 11

def referenceLayer (slot : Slot) : Option Layer :=
  if h : slot.val < 10 then some ⟨slot.val, by omega⟩
  else if slot.val = 11 then some 8 else none

def lambdaLayer (slot : Nat) : Nat := if slot = 11 then 8 else slot

def writesFold (slot : Nat) : Bool := slot < 12 && (slot < 10 || slot == 11)

theorem lambda_read_in_bounds (slot : Nat) (h : writesFold slot = true) :
    2 * lambdaLayer slot + 1 < 22 := by
  unfold writesFold at h
  obtain ⟨_,h⟩ := Bool.and_eq_true_iff.mp h
  have h' : slot < 10 ∨ slot = 11 := by simpa using h
  rcases h' with h₀ | h₁
  · have hn : slot ≠ 11 := by omega
    simp [lambdaLayer, hn]
    omega
  · simp [lambdaLayer, h₁]

theorem padding_skips_fold (slot : Nat) (h : 12 ≤ slot) : writesFold slot = false := by
  simp [writesFold, Nat.not_lt.mpr h]

def candidateLayer (slot : Slot) : Option Layer :=
  let row := lambdaLayer slot.val
  if writesFold slot.val then
    some (⟨row % 11, Nat.mod_lt _ (by decide)⟩ : Layer)
  else none

/-- The modulo in the total Lean model is redundant on all active lanes of
the actual masked load. Padding lanes have no store. -/
theorem layer_address (slot : Slot) : candidateLayer slot = referenceLayer slot := by
  by_cases hp : slot.val = 11
  · have hn : ¬ slot.val < 10 := by omega
    simp [candidateLayer, referenceLayer, writesFold, lambdaLayer, hp]
  · by_cases hn : slot.val < 10
    · have h : slot.val < 11 := by omega
      simp [candidateLayer, referenceLayer, writesFold, lambdaLayer, slot.isLt, hp, hn, Nat.mod_eq_of_lt h]
    · simp [candidateLayer, referenceLayer, writesFold, hp, hn]

def sanitizeReference (raw : Scalar) : Scalar :=
  if raw.isNaN || raw.isInf then zero else raw

def sanitizeCandidate (raw : Scalar) : Scalar :=
  if (raw.raw >>> 23) &&& (2^8 - 1) == 255 then zero else raw

theorem sanitize_equal (raw : Scalar) : sanitizeCandidate raw = sanitizeReference raw := by
  unfold sanitizeCandidate sanitizeReference Value.isNaN Value.isInf
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]
  change (if raw.exponent == 255 then zero else raw) = _
  by_cases h : raw.exponent = 255
  · by_cases hf : raw.fraction = 0 <;> simp [h, hf]
  · simp [h]

def refreshReference (raw previous gain floor : Scalar) : Scalar :=
  let a := min32 (sanitizeReference raw) (literal 1000000000000)
  let new := max32 (mul32 a gain) floor
  if orderedLt zero a then new else previous

def refreshCandidate (raw previous gain floor : Scalar) : Scalar :=
  let a := min32 (sanitizeCandidate raw) (literal 1000000000000)
  let new := max32 (mul32 a gain) floor
  if orderedLt zero a then new else previous

theorem refresh_equal (raw previous gain floor : Scalar) :
    refreshCandidate raw previous gain floor = refreshReference raw previous gain floor := by
  simp only [refreshCandidate, refreshReference, sanitize_equal]

/-- FP32 buffers at the boundary after the two weight quantizers. Each field
is an independently allocated buffer in the upstream GPT module. -/
structure Buffers where
  upScales : Slot → Scalar
  downScales : Slot → Scalar
  postAmax : Slot → Scalar
  postScales : Slot → Scalar
  dpreAmax : Slot → Scalar
  dpreScales : Slot → Scalar
  postLambdas : Layer → Scalar
  fold : Slot → Scalar
  upDequant : Slot → Scalar
  downDequant : Slot → Scalar
  downFolded : Slot → Scalar
  downDequantFolded : Slot → Scalar

/-- ATen sequence in GPT._quantize_mlp_weights_and_scales. Both amax
reductions have width one. Their only non-finite outputs are zeroed by
nan_to_num before any result is used. -/
def staged (c : Scalars) (s : Buffers) : Buffers :=
  let d := { s with upDequant := fun i => mul32 (s.upScales i) c.xScale,
                    downDequant := fun i => mul32 (s.downScales i) c.gradScale }
  let a := { d with postScales := fun i =>
                     refreshReference (d.postAmax i) (d.postScales i) c.postGain (literal (1 / 1000000)),
                    postAmax := fun _ => zero,
                    dpreScales := fun i =>
                     refreshReference (d.dpreAmax i) (d.dpreScales i) c.dpreGain (literal (1 / 1000000000000)),
                    dpreAmax := fun _ => zero }
  let f := { a with fold := fun i => (referenceLayer i).elim (a.fold i)
                                              (fun j => roundedLambda (a.postLambdas j)) }
  { f with downFolded := fun i => mul32 (f.downScales i) (f.fold i),
           downDequantFolded := fun i => mul32 (f.downDequant i) (f.fold i) }

def forwardFold (s : Buffers) (i : Slot) : Scalar :=
  (candidateLayer i).elim (s.fold i) (fun j => roundedLambda (s.postLambdas j))

/-- One Triton lane owns each live slot. Both amax clears are after their
loads, and the two consumers share the same rounded fold/dequant registers. -/
def fused (c : Scalars) (s : Buffers) : Buffers :=
  { s with
    postScales := fun i =>
      refreshCandidate (s.postAmax i) (s.postScales i) c.postGain (literal (1 / 1000000))
    dpreScales := fun i =>
      refreshCandidate (s.dpreAmax i) (s.dpreScales i) c.dpreGain (literal (1 / 1000000000000))
    postAmax := fun _ => zero
    dpreAmax := fun _ => zero
    fold := fun i => forwardFold s i
    upDequant := fun i => mul32 (s.upScales i) c.xScale
    downDequant := fun i => mul32 (s.downScales i) c.gradScale
    downFolded := fun i => mul32 (s.downScales i) (forwardFold s i)
    downDequantFolded := fun i => mul32 (mul32 (s.downScales i) c.gradScale) (forwardFold s i) }

theorem buffer_equality (c : Scalars) (s : Buffers) : fused c s = staged c s := by
  simp only [fused, staged, forwardFold, layer_address, refresh_equal]

/-- One live slot is uniquely owned by its same-numbered lane of the 16-lane
launch. Padding lanes 12 to 15 write none of the buffers. -/
theorem launch_ownership (slot : Slot) :
    ∃ lane : Fin 16, (lane.val = slot.val ∧ lane.val < 12) ∧
      ∀ other : Fin 16, other.val = slot.val ∧ other.val < 12 → other = lane := by
  refine ⟨⟨slot.val, by have := slot.isLt; omega⟩, ⟨rfl, slot.isLt⟩, ?_⟩
  intro lane h
  exact Fin.ext h.1

/-- Substituting the refresh at any occurrence in a trainer expressed at
this buffer boundary preserves its entire state at any number of steps.
The common prefix and suffix are unchanged, including optimizer and data
selection. This theorem composes the proved equality; it does not assert a
Lean semantics for the trainer's Python, distributed runtime or GPU lowering. -/
theorem replacement_in_loop {State : Type}
    (before : Nat → State → Buffers × State)
    (after : Nat → Buffers → State → State) (c : Scalars) (state : State) (steps : List Nat) :
    steps.foldl (fun s step => let (b, rest) := before step s; after step (fused c b) rest) state =
      steps.foldl (fun s step => let (b, rest) := before step s; after step (staged c b) rest) state := by
  simp only [buffer_equality]

#axiomsClean sanitize_equal
#axiomsClean layer_address
#axiomsClean lambda_read_in_bounds
#axiomsClean padding_skips_fold
#axiomsClean buffer_equality
#axiomsClean launch_ownership
#axiomsClean replacement_in_loop
end VeriTile.Bench.Models.NanoGPT.Track1
