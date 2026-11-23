import models.NanoGPT.Track1.GraphStageMemory
import Batteries.Data.List.Perm
import Batteries.Data.List.Lemmas

/-! Word effects of the copy branch of the fused SM90a launch. A vector
operation addresses two consecutive words. The scalar MTP and prefix
stores have stricter predicates than their loads. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphCopyPTX
open GraphStage GraphStageMemory CudaStores

abbrev Word := Field × Nat

def index (pid tid : UInt32) : UInt32 :=
  ((tid &&& 127) <<< 1) ||| (pid <<< 8)

theorem index_exact (pid tid : Nat) (block : pid < 192) (lane : tid < 128) :
    (index (UInt32.ofNat pid) (UInt32.ofNat tid)).toNat = 256 * pid + 2 * tid := by
  have p : (UInt32.ofNat pid).toNat = pid := UInt32.toNat_ofNat_of_lt' (by change pid < 4294967296; omega)
  have t : (UInt32.ofNat tid).toNat = tid := UInt32.toNat_ofNat_of_lt' (by change tid < 4294967296; omega)
  rw [index, UInt32.toNat_or, UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, UInt32.toNat_and, p, t]
  change ((tid &&& 127) <<< 1) % 4294967296 ||| ((pid <<< 8) % 4294967296) = 256 * pid + 2 * tid
  have mask : tid &&& 127 = tid := Nat.and_two_pow_sub_one_of_lt_two_pow (n := 7) lane
  have pshift : pid <<< 8 = 256 * pid := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  have tshift : tid <<< 1 = 2 * tid := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  rw [mask, Nat.mod_eq_of_lt (by rw [tshift]; omega), Nat.mod_eq_of_lt (by rw [pshift]; omega), Nat.or_comm,
    ← Nat.shiftLeft_add_eq_or_of_lt (i := 8) (by rw [tshift]; omega), pshift, tshift]

def masked (guard : Bool) (word : Word) : List Word := if guard then [word] else []

def mtpRead (s : Shape) (row : UInt32) : Bool :=
  match s with
  | .initial => row != 3
  | .middle => decide (row < 2)
  | .bulk | .taper | .extension => row == 0

def scalarWords (s : Shape) (tid : Fin 128) (stores : Bool) : List Word :=
  let thread := UInt32.ofNat tid.val
  let lane := thread &&& 127
  let warp := (thread >>> 5) &&& 3
  let row := thread &&& 3
  let firstLane := (warp ||| (thread &&& 31)) == 0
  let mtp := if s.mtp == 1 then firstLane else (warp ||| (thread &&& 28)) == 0
  masked (s.ends == 128 || decide (lane.toNat < s.ends)) (.ends, lane.toNat) ++
    masked (mtpRead s row && (!stores || mtp)) (.mtp, row.toNat) ++
    masked (!stores || firstLane) (.prefix, 0)

def words (s : Shape) (pid : Nat) (tid : Fin 128) (stores : Bool) : List Word :=
  let first := (index (UInt32.ofNat pid) (UInt32.ofNat tid.val)).toNat
  [(.inputs, first), (.inputs, first + 1), (.targets, first), (.targets, first + 1)] ++
    if pid = 0 then scalarWords s tid stores else []

set_option maxRecDepth 8192 in
private theorem scalar_store_words (s : Shape) : ∀ tid : Fin 128,
    scalarWords s tid true =
      masked (decide (tid.val < s.ends)) (.ends, tid.val) ++
      masked (decide (tid.val < s.mtp)) (.mtp, tid.val) ++
      masked (decide (tid.val = 0)) (.prefix, 0) := by
  cases s <;> decide

set_option maxRecDepth 8192 in
private theorem scalar_reads_bounded (s : Shape) : ∀ tid : Fin 128,
    (scalarWords s tid false).all (fun word => decide (word.2 < extent s word.1)) = true := by
  cases s <;> decide

set_option maxRecDepth 8192 in
private theorem scalar_stores_read (s : Shape) : ∀ tid : Fin 128,
    (scalarWords s tid true).all (fun word => decide (word ∈ scalarWords s tid false)) = true := by
  cases s <;> decide

private theorem block_limit (s : Shape) : s.blocks ≤ 192 := by cases s <;> decide

private theorem tile_count (s : Shape) : 256 * s.blocks = s.tokens := by cases s <;> decide

private theorem geometry (s : Shape) : s.ends ≤ 128 ∧ s.mtp ≤ 4 := by cases s <;> decide

theorem reads_bounded (s : Shape) (pid : Nat) (live : pid < s.blocks) (tid : Fin 128)
    (word : Word) (read : word ∈ words s pid tid false) : word.2 < extent s word.1 := by
  have lane := tid.isLt
  have idx := index_exact pid tid.val (Nat.lt_of_lt_of_le live (block_limit s)) lane
  have count := tile_count s
  unfold words at read
  simp only [idx, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at read
  rcases read with (rfl | rfl | rfl | rfl) | head
  all_goals try (change _ < s.tokens; omega)
  by_cases first : pid = 0
  · simp only [if_pos first] at head
    exact of_decide_eq_true (List.all_eq_true.mp (scalar_reads_bounded s tid) word head)
  · simp [first] at head

theorem stores_read (s : Shape) (pid : Nat) (tid : Fin 128) (word : Word)
    (written : word ∈ words s pid tid true) : word ∈ words s pid tid false := by
  unfold words at written ⊢
  rcases List.mem_append.mp written with vector | head
  · exact List.mem_append_left _ vector
  · by_cases first : pid = 0
    · simp only [if_pos first] at head ⊢
      exact List.mem_append_right _ (of_decide_eq_true (List.all_eq_true.mp (scalar_stores_read s tid) word head))
    · simp [first] at head

def wordLane (f : Field) (word : Nat) : Nat :=
  match f with
  | .inputs | .targets => word % 256 / 2
  | _ => word

private theorem mem_masked (p : Prop) [Decidable p] (x y : Word) :
    x ∈ masked (decide p) y ↔ p ∧ x = y := by
  by_cases h : p <;> simp [masked, h]

theorem stores_word_iff (s : Shape) (pid : Nat) (copy : pid < s.blocks) (tid : Fin 128)
    (f : Field) (word : Nat) :
    (f, word) ∈ words s pid tid true ↔
      f ≠ .slots ∧ word < extent s f ∧ word / f.tile = pid ∧ wordLane f word = tid.val := by
  have lane := tid.isLt
  have idx := index_exact pid tid.val (Nat.lt_of_lt_of_le copy (block_limit s)) lane
  have count := tile_count s
  have ⟨ends, mtp⟩ := geometry s
  simp only [words, idx, scalar_store_words, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
  by_cases first : pid = 0
  · subst pid
    cases f <;> simp [mem_masked, wordLane, Field.tile, extent] <;> omega
  · simp only [if_neg first, List.not_mem_nil, or_false]
    cases f <;> simp [wordLane, Field.tile, extent] <;> omega

private theorem width_pos (f : Field) : 0 < width f := by cases f <;> decide

def wordBytes (word : Word) : List (Field × Nat) :=
  (List.range (width word.1)).map fun b => (word.1, word.2 * width word.1 + b)

theorem byte_of_word (f : Field) (offset : Nat) (word : Word) :
    (f, offset) ∈ wordBytes word ↔ (f, offset / width f) = word := by
  rcases word with ⟨field, i⟩
  rw [wordBytes, List.mem_map]
  constructor
  · rintro ⟨b, live, same⟩
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj same
    have bound := List.mem_range.mp live
    rw [Nat.mul_comm i (width field), Nat.mul_add_div (width_pos field), Nat.div_eq_of_lt bound, Nat.add_zero]
  · intro same
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj same
    exact ⟨offset % width f, List.mem_range.mpr (Nat.mod_lt _ (width_pos f)),
      congrArg (Prod.mk f) (Nat.div_add_mod' offset (width f))⟩

def laneBytes (s : Shape) (pid : Nat) (tid : Fin 128) : List (Field × Nat) :=
  (words s pid tid true).flatMap wordBytes

theorem lane_byte_iff (s : Shape) (pid : Nat) (tid : Fin 128) (f : Field) (offset : Nat) :
    (f, offset) ∈ laneBytes s pid tid ↔ (f, offset / width f) ∈ words s pid tid true := by
  simp only [laneBytes, List.mem_flatMap, byte_of_word]
  exact ⟨fun ⟨_, present, rfl⟩ => present, fun present => ⟨_, present, rfl⟩⟩

set_option maxRecDepth 8192 in
private theorem scalar_nodup (s : Shape) : ∀ tid : Fin 128, (scalarWords s tid true).Nodup := by
  cases s <;> decide

private theorem vector_distinct (i : Nat) :
    ([(Field.inputs, i), (.inputs, i + 1), (.targets, i), (.targets, i + 1)]).Nodup := by simp

private theorem words_nodup (s : Shape) (pid : Nat) (tid : Fin 128) :
    (words s pid tid true).Nodup := by
  unfold words
  split
  · apply List.nodup_append.mpr
    refine ⟨vector_distinct _, scalar_nodup s tid, ?_⟩
    intro a present b tail same
    subst b
    rw [scalar_store_words] at tail
    have fields : a.1 = .inputs ∨ a.1 = .targets := by
      rcases List.mem_cons.mp present with rfl | present
      · exact Or.inl rfl
      rcases List.mem_cons.mp present with rfl | present
      · exact Or.inl rfl
      rcases List.mem_cons.mp present with rfl | present
      · exact Or.inr rfl
      have h := List.mem_singleton.mp present
      exact Or.inr (congrArg Prod.fst h)
    rcases a with ⟨field, offset⟩
    rcases fields with h | h <;> cases h <;> simp [mem_masked] at tail
  · simp only [List.append_nil]
    exact vector_distinct _

private theorem lane_bytes_nodup (s : Shape) (pid : Nat) (tid : Fin 128) :
    (laneBytes s pid tid).Nodup := by
  unfold laneBytes
  refine List.pairwise_flatMap.mpr ⟨?_, ?_⟩
  · intro word _
    rw [wordBytes, List.pairwise_map]
    apply List.nodup_range.imp
    intro a b unequal same
    exact unequal (Nat.add_left_cancel (congrArg Prod.snd same))
  · apply (words_nodup s pid tid).imp
    intro a b different x hx y hy same
    subst y
    rcases x with ⟨f, offset⟩
    exact different (((byte_of_word f offset a).mp hx).symm.trans ((byte_of_word f offset b).mp hy))

def ctaBytes (s : Shape) (pid : Nat) : List (Field × Nat) :=
  (List.finRange 128).flatMap (laneBytes s pid)

private theorem cta_bytes_nodup (s : Shape) (pid : Nat) (copy : pid < s.blocks) : (ctaBytes s pid).Nodup := by
  unfold ctaBytes
  refine List.pairwise_flatMap.mpr ⟨fun tid _ => lane_bytes_nodup s pid tid, ?_⟩
  apply (List.nodup_finRange 128).imp
  intro a b different x hx y hy same
  subst y
  rcases x with ⟨f, offset⟩
  have left := (stores_word_iff s pid copy a f _).mp ((lane_byte_iff s pid a f offset).mp hx)
  have right := (stores_word_iff s pid copy b f _).mp ((lane_byte_iff s pid b f offset).mp hy)
  exact different (Fin.ext (left.2.2.2.symm.trans right.2.2.2))

private theorem cta_byte_iff (s : Shape) (pid : Nat) (copy : pid < s.blocks) (f : Field) (offset : Nat) :
    (f, offset) ∈ ctaBytes s pid ↔
      f ≠ .slots ∧ offset < (Output.graph f).bytes s ∧ offset / width f / f.tile = pid := by
  simp only [ctaBytes, List.mem_flatMap, List.mem_finRange, true_and, lane_byte_iff, stores_word_iff s pid copy]
  rw [Output.bytes, ← Nat.div_lt_iff_lt_mul (width_pos f)]
  constructor
  · rintro ⟨_, h, range, assigned, _⟩
    exact ⟨h, range, assigned⟩
  · rintro ⟨copyPort, bound, assigned⟩
    have limit : wordLane f (offset / width f) < 128 := by
      have h := geometry s
      cases f <;> simp [wordLane, extent, width] at copyPort bound ⊢ <;> omega
    exact ⟨⟨_, limit⟩, copyPort, bound, assigned, rfl⟩

private def toCell {s : Shape} (byte : Field × Nat) (valid : byte.2 < (Output.graph byte.1).bytes s) :
    Cell s := ⟨.graph byte.1, ⟨byte.2, valid⟩⟩

def copyCells (s : Shape) (pid : Nat) (copy : pid < s.blocks) : List (Cell s) :=
  (ctaBytes s pid).attach.map fun b =>
    toCell b.val ((cta_byte_iff s pid copy b.val.1 b.val.2).mp b.property).2.1

private theorem copy_cells_nodup (s : Shape) (pid : Nat) (copy : pid < s.blocks) :
    (copyCells s pid copy).Nodup := by
  unfold copyCells
  rw [List.Nodup, List.pairwise_map]
  have original : ((ctaBytes s pid).attach.map Subtype.val).Nodup := by
    simpa only [List.attach_map_subtype_val] using cta_bytes_nodup s pid copy
  rw [List.Nodup, List.pairwise_map] at original
  apply original.imp
  intro a b different same
  apply different
  apply Prod.ext
  · exact Output.graph.inj (congrArg Cell.out same)
  · exact congrArg (fun cell : Cell s => cell.offset.val) same

theorem copy_cells_permute {s n} (layout : Layout s n) (pid : Nat) (copy : pid < s.blocks) :
    (copyCells s pid copy).Perm (assigned s pid) := by
  apply (List.perm_ext_iff_of_nodup (copy_cells_nodup s pid copy) (assigned_nodup layout pid)).2
  intro cell
  rw [assigned, List.mem_filter]
  simp only [mem_cells, true_and, decide_eq_true_eq]
  constructor
  · intro present
    obtain ⟨byte, _, rfl⟩ := List.mem_map.mp present
    have ⟨field, _, assigned⟩ := (cta_byte_iff s pid copy byte.val.1 byte.val.2).mp byte.property
    rcases byte with ⟨⟨f, i⟩, _⟩
    cases f <;> first | contradiction | exact assigned
  · intro owned
    rcases cell with ⟨out, offset⟩
    cases out with
    | pending => have h := copy; simp [owner, queryCTA] at owned; omega
    | graph f =>
      have copied : f ≠ .slots := by
        intro same
        subst f
        simp [owner, queryCTA] at owned
        omega
      have exactOwner : offset.val / width f / f.tile = pid := by
        cases f <;> first | contradiction | exact owned
      let byte : {b // b ∈ ctaBytes s pid} :=
        ⟨(f, offset.val), (cta_byte_iff s pid copy f offset.val).mpr ⟨copied, offset.isLt, exactOwner⟩⟩
      exact List.mem_map.mpr ⟨byte, List.mem_attach _ _, rfl⟩

#axiomsClean index_exact
#axiomsClean reads_bounded
#axiomsClean stores_word_iff
#axiomsClean copy_cells_permute
end VeriTile.Bench.Models.NanoGPT.Track1.GraphCopyPTX
