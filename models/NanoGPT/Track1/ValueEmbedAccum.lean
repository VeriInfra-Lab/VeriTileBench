import models.NanoGPT.Track1.ValueAdam

/-! The value embedding's autograd callback adds BF16 gradients to its
persistent FP16 table. The fused launch assigns a warp to each plane.
The model preserves the observed scalar atomic order at every address. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueEmbedAccum
open VeriTile.Triton FP
open SparsePackets (FP16Rows Channel)

abbrev Plane := Fin 4
abbrev Entry (tokens : Nat) := Plane × Fin tokens

structure Batch where
  tokens : Nat
  vocab : Nat
  token : Fin tokens → Fin vocab
  grad : Plane → Fin tokens → Channel → Value .bf16

def Batch.row (b : Batch) (e : Entry b.tokens) : Nat := e.1.val * b.vocab + (b.token e.2).val

def Batch.operand (b : Batch) (e : Entry b.tokens) (col : Channel) : UInt16 :=
  ValueAdam.narrow (FP.cast .fp32 Config.ieeeValue (b.grad e.1 e.2 col))

/-- The trace records the order at each channel, including repeated token
IDs. Every logical `(plane, token)` contribution is issued once. -/
structure Snapshot where
  address : Nat
  batch : Batch
  prior : FP16Rows
  trace : Channel → List (Entry batch.tokens)
  complete : ∀ col, (trace col).Perm ((List.finRange 4).product (List.finRange batch.tokens))

private def inPlane {tokens : Nat} (p : Plane) (entries : List (Entry tokens)) : List (Entry tokens) :=
  entries.filter fun e => decide (e.1 = p)

private theorem nested_plane {tokens} (p q : Plane) (entries : List (Entry tokens)) :
    inPlane p (inPlane q entries) = if p = q then inPlane p entries else [] := by
  induction entries with
  | nil => simp [inPlane]
  | cons e rest ih =>
    by_cases ep : e.1 = p <;> by_cases eq : e.1 = q <;> by_cases same : p = q <;>
      simp_all [inPlane, List.filter_cons]

private theorem ordered_planes {tokens} (p : Plane) (planes : List Plane) (entries : List (Entry tokens))
    (unique : planes.Nodup) :
    inPlane p (planes.flatMap fun q => inPlane q entries) =
      if p ∈ planes then inPlane p entries else [] := by
  induction planes with
  | nil => simp [inPlane]
  | cons q rest ih =>
    obtain ⟨fresh, tail⟩ := List.nodup_cons.mp unique
    have splitPlane : inPlane p (inPlane q entries ++ rest.flatMap fun q => inPlane q entries) =
      inPlane p (inPlane q entries) ++ inPlane p (rest.flatMap fun q => inPlane q entries) := List.filter_append ..
    rw [List.flatMap_cons, splitPlane, nested_plane, ih tail]
    by_cases same : p = q
    · subst q
      simp [fresh]
    · simp [same]

def stagedTrace {tokens} (events : List (Entry tokens)) : List (Entry tokens) :=
  (List.finRange 4).flatMap fun p => inPlane p events

theorem staged_keeps_plane_order {tokens} (events : List (Entry tokens)) (p : Plane) :
    inPlane p (stagedTrace events) = inPlane p events := by
  simp [stagedTrace, ordered_planes p _ _ (List.nodup_finRange _)]

def observe (s : Snapshot) (trace : Channel → List (Entry s.batch.tokens)) : FP16Rows :=
  fun row col =>
    if inside : row / s.batch.vocab < 4 then
      (inPlane ⟨row / s.batch.vocab, inside⟩ (trace col)).foldl (fun prior e =>
        if s.batch.row e = row then ValueAdam.halfAdd prior (s.batch.operand e col) else prior)
          (s.prior row col)
    else s.prior row col

abbrev Accumulate := Snapshot → FP16Rows

def staged : Accumulate := fun s => observe s (fun col => stagedTrace (s.trace col))
def fused : Accumulate := fun s => observe s s.trace

theorem accum_equal : fused = staged := by
  funext s row col
  simp only [fused, staged, observe, staged_keeps_plane_order]

def accumulateSteps (kernel : Accumulate) (prior : FP16Rows) (steps : List Snapshot) : FP16Rows :=
  steps.foldl (fun bank step => kernel { step with prior := bank }) prior

theorem untouched (s : Snapshot) (row : Nat) (col : Channel) (misses : ∀ e, s.batch.row e ≠ row) :
    fused s row col = s.prior row col := by
  have flat (events : List (Entry s.batch.tokens)) (prior : UInt16) :
      events.foldl (fun a e => if s.batch.row e = row then ValueAdam.halfAdd a (s.batch.operand e col) else a)
        prior = prior := by
    induction events with
    | nil => rfl
    | cons e rest ih => simp [misses e, ih]
  simp only [fused, observe]
  split
  · exact flat ..
  · rfl

def planeOf (thread : Fin 128) : Plane := ⟨thread.val / 32, by have := thread.isLt; omega⟩
def directCol (thread : Fin 128) (stripe : Fin 3) (word : Fin 8) : Channel :=
  ⟨stripe.val * 256 + thread.val % 32 * 8 + word.val, by
    have := stripe.isLt; have := word.isLt; omega⟩
def originalCol (thread : Fin 128) (stripe : Fin 3) (word : Fin 2) : Channel :=
  ⟨stripe.val * 256 + thread.val * 2 + word.val, by
    have := stripe.isLt; have := thread.isLt; have := word.isLt; omega⟩

theorem thread_coordinates (thread : Fin 128) :
    (thread.val >>> 5) &&& 3 = (planeOf thread).val ∧
    (thread.val <<< 3) &&& 248 = thread.val % 32 * 8 := by
  exact SparsePackets.lowered_thread_coordinates thread

theorem covers (plane : Plane) (col : Channel) :
    ∃ thread stripe word, planeOf thread = plane ∧ directCol thread stripe word = col := by
  have lanes := plane.isLt
  have columns := col.isLt
  let thread : Fin 128 := ⟨plane.val * 32 + col.val % 256 / 8, by omega⟩
  refine ⟨thread, ⟨col.val / 256, by omega⟩, ⟨col.val % 8, by omega⟩, Fin.ext ?_, Fin.ext ?_⟩
  · simp only [planeOf, thread, Fin.val_mk]
    omega
  · simp only [directCol, thread, Fin.val_mk]
    omega

theorem reference_lane (thread : Fin 128) (stripe : Fin 3) (word : Fin 8) :
    originalCol ⟨thread.val % 32 * 4 + word.val / 2, by have := word.isLt; omega⟩ stripe
      ⟨word.val % 2, by omega⟩ = directCol thread stripe word := by
  apply Fin.ext
  simp only [originalCol, directCol, Fin.val_mk]
  omega

theorem unique_lane {t u : Fin 128} {s r : Fin 3} {w v : Fin 8}
    (plane : planeOf t = planeOf u) (col : directCol t s w = directCol u r v) :
    t = u ∧ s = r ∧ w = v := by
  have a := congrArg Fin.val plane
  have b := congrArg Fin.val col
  simp only [planeOf, directCol, Fin.val_mk] at a b
  have := t.isLt; have := u.isLt; have := s.isLt; have := r.isLt; have := w.isLt; have := v.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega), Fin.ext (by omega)⟩

theorem Batch.row_within (b : Batch) (entry : Entry b.tokens) : b.row entry < 4 * b.vocab := by
  have oneMore : b.row entry < (entry.1.val + 1) * b.vocab := by
    rw [Nat.add_mul, Nat.one_mul]
    exact Nat.add_lt_add_left (b.token entry.2).isLt _
  exact oneMore.trans_le (Nat.mul_le_mul_right _ entry.1.isLt)

theorem addresses (b : Batch) (token : Fin b.tokens) (thread : Fin 128) (stripe : Fin 3) :
    let col := directCol thread stripe ⟨0, by decide⟩
    let row := b.row (planeOf thread, token)
    (2 * (token.val * 768 + col.val)) % 16 = 0 ∧
    (2 * (row * 768 + col.val)) % 16 = 0 ∧
    row * 768 + col.val + 7 < 4 * b.vocab * 768 := by
  intro col row
  have live : row < 4 * b.vocab := b.row_within (planeOf thread, token)
  have edge : col.val = stripe.val * 256 + thread.val % 32 * 8 := by simp [col, directCol]
  have := stripe.isLt
  omega

#axiomsClean accum_equal
#axiomsClean untouched
#axiomsClean staged_keeps_plane_order
#axiomsClean covers
#axiomsClean reference_lane
#axiomsClean unique_lane
#axiomsClean addresses
end VeriTile.Bench.Models.NanoGPT.Track1.ValueEmbedAccum
