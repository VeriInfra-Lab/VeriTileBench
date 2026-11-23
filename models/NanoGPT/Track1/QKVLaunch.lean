import models.NanoGPT.Track1.QKVAmax

/-! Address and launch contract for the aligned device code at the three
widths used by the pinned trainer's GPT.quantize_attn_fp8. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVLaunch
open QKVAmax

inductive Width where
  | narrow | halfValue | wide
  deriving DecidableEq, Repr

structure Layout where
  batches : Nat
  firstTiles : Nat
  secondTiles : Nat
  firstStride : Nat
  secondStride : Nat
  deriving DecidableEq, Repr

/-- Element strides of qk_heads[..., :qk_dim] and v_heads[..., :v_dim]
after the actual reshape in the pinned GPT module. A full V slice retains
the V/O pitch; a cropped and packed V slice does not. -/
def Layout.ofWidth : Width → Layout
  | .narrow => ⟨3, 288, 288, 589824, 1179648⟩
  | .halfValue => ⟨2, 288, 144, 589824, 294912⟩
  | .wide => ⟨2, 576, 288, 1179648, 1179648⟩

def word (n : Nat) : BitVec 32 := BitVec.ofNat 32 n

/-- Signed arith.divsi of the two i32 loop bounds in the compiled TTIR. -/
def visitsI32 (tiles : Nat) (part : Part) : Nat :=
  ((word (tiles + 31) - word part.val).sdiv (word 32)).toNat

theorem loop_counts (w : Width) :
    ∀ part : Part,
      let l := Layout.ofWidth w
      visitsI32 l.firstTiles part = visitCount part l.firstTiles ∧
      visitsI32 (l.firstTiles+l.secondTiles) part = visitCount part (l.firstTiles+l.secondTiles) := by
  cases w <;> decide +kernel

/-- Trip count of the reference scf.for, whose induction variable is the
element offset and advances by 32 tiles. The bounds are signed i32. -/
def referenceVisitsI32 (l : Layout) (part : Part) : Nat :=
  let lower := (word part.val * word Tile).toInt
  let upper := (word ((l.firstTiles + l.secondTiles) * Tile)).toInt
  ((upper - lower + (32 * Tile - 1)) / (32 * Tile)).toNat

theorem reference_loop_count (w : Width) : ∀ part : Part,
    let l := Layout.ofWidth w
    referenceVisitsI32 l part = visitCount part (l.firstTiles + l.secondTiles) := by
  cases w <;> decide +kernel

def elementOffset (batch stride tile : Nat) (lane : Lane) : Nat :=
  batch * stride + tile * Tile + lane.val

/-- The compiler computes these products in i32, then sign extends the
offsets for tt.addptr. The sum is split into successive tt.addptrs. -/
def offsetI32 (batch stride tile : Nat) (lane : Lane) : BitVec 32 :=
  word batch * word stride + word tile * word Tile + word lane.val

theorem offset_word (batch stride tile : Nat) (lane : Lane) :
    offsetI32 batch stride tile lane = word (elementOffset batch stride tile lane) := by
  simp only [offsetI32,elementOffset,word,BitVec.ofNat_add,BitVec.ofNat_mul]

theorem offset_sign_extend (batch stride tile : Nat) (lane : Lane)
    (hb : elementOffset batch stride tile lane < 2^31) :
    (offsetI32 batch stride tile lane).toInt = (elementOffset batch stride tile lane : Int) := by
  have h32 : elementOffset batch stride tile lane < 2^32 := by omega
  rw [offset_word]
  unfold word
  have hn : (BitVec.ofNat 32 (elementOffset batch stride tile lane)).toNat =
      elementOffset batch stride tile lane := by
    exact (BitVec.toNat_ofNat _ _).trans (Nat.mod_eq_of_lt h32)
  rw [BitVec.toInt_eq_toNat_cond,hn,if_pos]
  omega

theorem qk_bounds (w : Width) (batch tile : Nat) (lane : Lane)
    (hb : batch < (Layout.ofWidth w).batches) (ht : tile < (Layout.ofWidth w).firstTiles) :
    let l := Layout.ofWidth w
    elementOffset batch l.firstStride tile lane < (l.batches-1)*l.firstStride+l.firstTiles*Tile ∧
    elementOffset batch l.firstStride tile lane < 2^31 := by
  have hl := lane.isLt
  change lane.val < 2048 at hl
  cases w <;> simp only [Layout.ofWidth,elementOffset,Tile] at * <;> omega

theorem v_bounds (w : Width) (batch tile : Nat) (lane : Lane)
    (hb : batch < (Layout.ofWidth w).batches) (ht : tile < (Layout.ofWidth w).secondTiles) :
    let l := Layout.ofWidth w
    elementOffset batch l.secondStride tile lane < (l.batches-1)*l.secondStride+l.secondTiles*Tile ∧
    elementOffset batch l.secondStride tile lane < 2^31 := by
  have hl := lane.isLt
  change lane.val < 2048 at hl
  cases w <;> simp only [Layout.ofWidth,elementOffset,Tile] at * <;> omega

theorem v_tile_word (w : Width) (part : Part) (i : Nat)
    (lo : visitCount part (Layout.ofWidth w).firstTiles ≤ i)
    (hi : i < visitCount part ((Layout.ofWidth w).firstTiles + (Layout.ofWidth w).secondTiles)) :
    let l := Layout.ofWidth w
    (word part.val + word i * word 32 - word l.firstTiles).toNat = globalTile part i - l.firstTiles := by
  let l := Layout.ofWidth w
  have hb : l.firstTiles ≤ globalTile part i := by
    apply Nat.le_of_not_gt
    intro ht
    exact (Nat.not_lt.mpr lo) ((visited_iff part l.firstTiles i).mpr ht)
  have he := (visited_iff part (l.firstTiles+l.secondTiles) i).mp hi
  have hfirst : l.firstTiles < 2^32 := by cases w <;> decide
  have htile : globalTile part i < 2^32 :=
    Nat.lt_trans he (by cases w <;> decide)
  have heq : word part.val + word i * word 32 = word (globalTile part i) := by
    simp only [globalTile, word, BitVec.ofNat_add, BitVec.ofNat_mul]
  change (word part.val + word i * word 32 - word l.firstTiles).toNat = _
  rw [heq,BitVec.toNat_sub_of_not_usubOverflow]
  · simp only [word,BitVec.toNat_ofNat,Nat.mod_eq_of_lt hfirst,Nat.mod_eq_of_lt htile,l]
  · rw [BitVec.usubOverflow_eq,decide_eq_true_eq]
    change ¬ (word (globalTile part i)).toNat < (word l.firstTiles).toNat
    simpa only [word,BitVec.toNat_ofNat,Nat.mod_eq_of_lt hfirst,Nat.mod_eq_of_lt htile] using Nat.not_lt.mpr hb

def partialOffset (batch : Nat) (part : Part) : Nat := batch * 32 + part.val

theorem store_disjoint {b c : Nat} {p q : Part} (h : partialOffset b p = partialOffset c q) :
    b = c ∧ p = q := by
  have hp := p.isLt
  have hq := q.isLt
  change p.val < 32 at hp
  change q.val < 32 at hq
  unfold partialOffset at h
  exact ⟨by omega, Fin.ext (by omega)⟩

theorem partial_store_bounds (w : Width) (batch : Nat) (part : Part)
    (h : batch < (Layout.ofWidth w).batches) :
    partialOffset batch part < (Layout.ofWidth w).batches * 32 := by
  have hp := part.isLt
  change part.val < 32 at hp
  unfold partialOffset
  omega

#axiomsClean loop_counts
#axiomsClean reference_loop_count
#axiomsClean offset_sign_extend
#axiomsClean qk_bounds
#axiomsClean v_bounds
#axiomsClean v_tile_word
#axiomsClean store_disjoint
#axiomsClean partial_store_bounds
end VeriTile.Bench.Models.NanoGPT.Track1.QKVLaunch
