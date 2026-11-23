import models.NanoGPT.Track1.FineWebBatches
import VeriTile.Meta.StatementAudit

/-! Four independent CPU arrays occupy disjoint, aligned intervals in the
single pinned H2D packet. The GPU exposes the same typed words as separate
transfers. Bytes that exist solely for alignment are never observed. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.PinnedBatch
open FineWeb

inductive Field where
  | inputs | targets | cumulative | ngram
  deriving DecidableEq

def Field.width : Field → Nat
  | .targets => 8
  | _ => 4

abbrev Counts := Field → Nat

def size (counts : Counts) (f : Field) : Nat := counts f * f.width

def align (endByte : Nat) : Nat := (endByte + 15) / 16 * 16

theorem le_align (byte : Nat) : byte ≤ align byte := by
  unfold align
  omega

def start (counts : Counts) : Field → Nat
  | .inputs => 0
  | .targets => align (size counts .inputs)
  | .cumulative => align (align (size counts .inputs) + size counts .targets)
  | .ngram => align (align (align (size counts .inputs) + size counts .targets) + size counts .cumulative)

theorem aligned (counts : Counts) (f : Field) : start counts f % 16 = 0 := by
  cases f <;> simp [start, align]

def stop (counts : Counts) (f : Field) : Nat := start counts f + size counts f

theorem aligned_offset_in_field (counts : Counts) (offset : Nat) (aligned : offset % 16 = 0)
    (inside : offset < stop counts .ngram) :
    ∃ field, start counts field ≤ offset ∧ offset < stop counts field := by
  have round (x : Nat) (le : x ≤ offset) : align x ≤ offset := by
    unfold align
    omega
  by_cases inputs : offset < stop counts .inputs
  · exact ⟨.inputs, Nat.zero_le _, inputs⟩
  by_cases targets : offset < stop counts .targets
  · refine ⟨.targets, ?_, targets⟩
    simpa only [stop, start, Nat.zero_add] using round _ (Nat.le_of_not_gt inputs)
  by_cases cumulative : offset < stop counts .cumulative
  · exact ⟨.cumulative, round _ (Nat.le_of_not_gt targets), cumulative⟩
  exact ⟨.ngram, round _ (Nat.le_of_not_gt cumulative), inside⟩

theorem disjoint (counts : Counts) (a b : Field) (different : a ≠ b) :
    stop counts a ≤ start counts b ∨ stop counts b ≤ start counts a := by
  have first := le_align (stop counts .inputs)
  have second := le_align (stop counts .targets)
  have third := le_align (stop counts .cumulative)
  simp only [stop, start, Nat.zero_add] at *
  cases a <;> cases b <;> simp_all <;> omega

theorem packet_fits (actual capacity : Counts) (fits : ∀ f, actual f ≤ capacity f) :
    stop actual .ngram ≤ stop capacity .ngram := by
  have widthBound (f : Field) : size actual f ≤ size capacity f :=
    Nat.mul_le_mul_right f.width (fits f)
  have alignBound (a b : Nat) (bound : a ≤ b) : align a ≤ align b := by
    unfold align
    omega
  simp only [stop, start]
  exact Nat.add_le_add (alignBound _ _ (Nat.add_le_add
    (alignBound _ _ (Nat.add_le_add (alignBound _ _ (widthBound .inputs)) (widthBound .targets)))
    (widthBound .cumulative))) (widthBound .ngram)

abbrev Bytes (counts : Counts) (α : Type) := (f : Field) → Fin (size counts f) → α

def write (counts : Counts) (f : Field) (bytes : Bytes counts α)
    (prior : Nat → α) (address : Nat) : α :=
  if h : start counts f ≤ address ∧ address < stop counts f then
    bytes f ⟨address - start counts f, by simp only [stop] at h; omega⟩
  else prior address

def packet (counts : Counts) (bytes : Bytes counts α) (padding : Nat → α) : Nat → α :=
  write counts .ngram bytes (write counts .cumulative bytes
    (write counts .targets bytes (write counts .inputs bytes padding)))

theorem write_own (counts : Counts) (f : Field) (bytes : Bytes counts α) (prior : Nat → α)
    (i : Fin (size counts f)) : write counts f bytes prior (start counts f + i.val) = bytes f i := by
  have atF : start counts f ≤ start counts f + i.val ∧ start counts f + i.val < stop counts f := by
    have h := i.isLt
    unfold stop
    omega
  simp [write, atF]

theorem write_other (counts : Counts) (a b : Field) (different : a ≠ b) (bytes : Bytes counts α)
    (prior : Nat → α) (i : Fin (size counts b)) :
    write counts a bytes prior (start counts b + i.val) = prior (start counts b + i.val) := by
  have outside : ¬ (start counts a ≤ start counts b + i.val ∧ start counts b + i.val < stop counts a) := by
    have h := i.isLt
    have separated := disjoint counts a b different
    unfold stop at *
    omega
  simp [write, outside]

theorem packet_exact (counts : Counts) (bytes : Bytes counts α) (padding : Nat → α)
    (f : Field) (i : Fin (size counts f)) :
    packet counts bytes padding (start counts f + i.val) = bytes f i := by
  cases f <;> simp [packet, write_own, write_other, reduceCtorEq]

def counts (b : TokenBatch) : Counts
  | .inputs => b.inputs.size
  | .targets => b.targets.size
  | .cumulative => b.cumulative.size
  | .ngram => b.ngramRows.size

abbrev Words (s : Counts) := (f : Field) → Fin (s f) → UInt64

def words (b : TokenBatch) : Words (counts b)
  | .inputs, i => b.inputs[i.val]'i.isLt |>.toUInt64
  | .targets, i => b.targets[i.val]'i.isLt
  | .cumulative, i => b.cumulative[i.val]'i.isLt |>.toUInt64
  | .ngram, i => b.ngramRows[i.val]'i.isLt |>.toUInt64

private theorem width_positive (f : Field) : 0 < f.width := by cases f <;> decide

/-- Label each transferred byte with the word it belongs to. The region
proof applies to every byte and is independent of the bytes' contents. -/
def wordLabels (s : Counts) (raw : Words s) : Bytes s UInt64 :=
  fun f i => raw f ⟨i.val / f.width, (Nat.div_lt_iff_lt_mul (width_positive f)).2 i.isLt⟩

def exposeWord (s : Counts) (raw : Words s) (f : Field) (i : Fin (s f)) : UInt64 :=
  packet s (wordLabels s raw) (fun _ => 0) (start s f + i.val * f.width)

@[simp] theorem expose_word_exact (s : Counts) (raw : Words s) (f : Field) (i : Fin (s f)) :
    exposeWord s raw f i = raw f i := by
  have index := Nat.mul_lt_mul_of_pos_right i.isLt (width_positive f)
  unfold exposeWord
  rw [packet_exact s (wordLabels s raw) (fun _ => 0) f ⟨i.val * f.width, index⟩]
  simp [wordLabels, Nat.mul_div_cancel _ (width_positive f)]

def uploadPacked (b : TokenBatch) : TokenBatch :=
  let word := exposeWord (counts b) (words b)
  { inputs := Array.ofFn fun i => (word .inputs i).toUInt32
    targets := Array.ofFn fun i => word .targets i
    cumulative := Array.ofFn fun i => (word .cumulative i).toUInt32
    ngramRows := Array.ofFn fun i => (word .ngram i).toUInt32 }

@[simp] theorem upload_packed_exact (b : TokenBatch) : uploadPacked b = b := by
  cases b with
  | mk inputs targets cumulative ngramRows =>
    simp only [uploadPacked, expose_word_exact, words, UInt32.toUInt32_toUInt64, TokenBatch.mk.injEq]
    exact ⟨Array.ofFn_getElem, Array.ofFn_getElem, Array.ofFn_getElem, Array.ofFn_getElem⟩

def wordBytes (s : Counts) (raw : Words s) : Bytes s UInt8 :=
  fun f i =>
    let within := (Nat.div_lt_iff_lt_mul (width_positive f)).2 i.isLt
    UInt8.ofNat ((raw f ⟨i.val / f.width, within⟩).toNat / 256 ^ (i.val % f.width))

private def copyBytes (source : Nat → UInt8) (count base : Nat) (prior : Nat → UInt8) : Nat → UInt8 :=
  fun address =>
    if base ≤ address ∧ address < base + count then source (address - base) else prior address

private theorem copied_byte (source : Nat → UInt8) (count base : Nat) (prior : Nat → UInt8)
    (offset : Nat) (within : offset < count) :
    copyBytes source count base prior (base + offset) = source offset := by
  have written : base ≤ base + offset ∧ base + offset < base + count := by omega
  simp only [copyBytes, if_pos written, Nat.add_sub_cancel_left]

/-- Padding comes from the pinned allocation. The copy leaves addresses
outside the uploaded packet unchanged. -/
def uploadBytes (b : TokenBatch) (base : Nat) (padding prior : Nat → UInt8) : Nat → UInt8 :=
  copyBytes (packet (counts b) (wordBytes _ (words b)) padding) (stop (counts b) .ngram) base prior

def uploadFieldBytes (b : TokenBatch) (f : Field) (base : Nat) (prior : Nat → UInt8) : Nat → UInt8 :=
  let field := fun offset =>
    if h : offset < size (counts b) f then wordBytes _ (words b) f ⟨offset, h⟩ else 0
  copyBytes field (size (counts b) f) base prior

theorem uploadBytes_frame (b : TokenBatch) (base : Nat) (padding prior : Nat → UInt8)
    (address : Nat) (outside : ¬ (base ≤ address ∧ address < base + stop (counts b) .ngram)) :
    uploadBytes b base padding prior address = prior address := by
  simp only [uploadBytes, copyBytes, if_neg outside]

theorem uploadFieldBytes_frame (b : TokenBatch) (f : Field) (base : Nat) (prior : Nat → UInt8)
    (address : Nat) (outside : ¬ (base ≤ address ∧ address < base + size (counts b) f)) :
    uploadFieldBytes b f base prior address = prior address := by
  simp only [uploadFieldBytes, copyBytes, if_neg outside]

private theorem stop_le_packet_end (s : Counts) (f : Field) : stop s f ≤ stop s .ngram := by
  have first := le_align (stop s .inputs)
  have second := le_align (stop s .targets)
  have third := le_align (stop s .cumulative)
  cases f <;> simp only [stop, start, Nat.zero_add] at * <;> omega

private theorem word_byte_fits (s : Counts) (f : Field) (i : Fin (s f)) (j : Fin f.width) :
    i.val * f.width + j.val < size s f := by
  have next := Nat.mul_le_mul_right f.width (Nat.succ_le_of_lt i.isLt)
  simp only [Nat.succ_mul] at next
  have hj := j.isLt
  unfold size
  omega

private theorem word_byte_exact (s : Counts) (raw : Words s) (f : Field)
    (i : Fin (s f)) (j : Fin f.width) :
    wordBytes s raw f ⟨i.val * f.width + j.val, word_byte_fits s f i j⟩ =
      UInt8.ofNat ((raw f i).toNat / 256 ^ j.val) := by
  have div : (i.val * f.width + j.val) / f.width = i.val := by
    rw [Nat.mul_comm, Nat.mul_add_div (width_positive f), Nat.div_eq_of_lt j.isLt, Nat.add_zero]
  have mod : (i.val * f.width + j.val) % f.width = j.val := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt j.isLt]
  simp only [wordBytes, div, mod]

theorem upload_byte_exact (b : TokenBatch) (base : Nat) (padding prior : Nat → UInt8)
    (f : Field) (i : Fin (counts b f)) (j : Fin f.width) :
    uploadBytes b base padding prior (base + start (counts b) f + i.val * f.width + j.val) =
      UInt8.ofNat ((words b f i).toNat / 256 ^ j.val) := by
  let offset := i.val * f.width + j.val
  have within := word_byte_fits (counts b) f i j
  have inField : start (counts b) f + offset < stop (counts b) f := by
    simpa only [stop, Nat.add_lt_add_iff_left] using within
  have fits := Nat.lt_of_lt_of_le inField (stop_le_packet_end (counts b) f)
  rw [show base + start (counts b) f + i.val * f.width + j.val =
        base + (start (counts b) f + offset) by simp only [offset]; omega]
  rw [uploadBytes, copied_byte _ _ _ _ _ fits, packet_exact _ _ _ f ⟨offset, within⟩]
  exact word_byte_exact _ _ f i j

theorem upload_field_byte_exact (b : TokenBatch) (f : Field) (base : Nat) (prior : Nat → UInt8)
    (i : Fin (counts b f)) (j : Fin f.width) :
    uploadFieldBytes b f base prior (base + i.val * f.width + j.val) =
      UInt8.ofNat ((words b f i).toNat / 256 ^ j.val) := by
  have within := word_byte_fits (counts b) f i j
  rw [Nat.add_assoc, uploadFieldBytes, copied_byte _ _ _ _ _ within, dif_pos within]
  exact word_byte_exact _ _ f i j

#axiomsClean upload_byte_exact
#axiomsClean upload_field_byte_exact
#axiomsClean expose_word_exact
#axiomsClean upload_packed_exact
#axiomsClean disjoint
#axiomsClean packet_fits
#axiomsClean packet_exact
end VeriTile.Bench.Models.NanoGPT.Track1.PinnedBatch
