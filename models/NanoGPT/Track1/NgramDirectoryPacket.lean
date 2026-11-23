import models.NanoGPT.Track1.GraphStageMemory

/-! The host packet stores sorted rows, padding through the next aligned
word index, and the left insertion indices for the directory boundaries. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramDirectoryPacket
open CudaStores GraphStageMemory NgramSlots

def boundedWant (want : Want) : Want where
  size := want.size
  read i := if i < want.size then want.read i else 0
  ordered i j order inside := by
    have earlier := Nat.lt_of_le_of_lt order inside
    simpa only [if_pos earlier, if_pos inside] using want.ordered i j order inside
  inVocab i inside := by simpa only [if_pos inside] using want.inVocab i inside

def boundary (want : Want) (i : Nat) : Nat := nativeNat (boundedWant want) (i * 32768)

structure Prepared (want : Want) : Type where
  nonempty : 0 < want.size
  fits : want.size < 2147483648
  widths : ∀ i < bucketCount, boundary want (i + 1) - boundary want i < 256

def prepare (want : Want) : Option (Prepared want) :=
  if geometry : 0 < want.size ∧ want.size < 2147483648 then
    if fits : (List.range bucketCount).all (fun i => decide (boundary want (i + 1) - boundary want i < 256)) then
      some ⟨geometry.1, geometry.2,
        fun i inside => of_decide_eq_true (List.all_eq_true.mp fits i (List.mem_range.mpr inside))⟩
    else none
  else none

def packetWord (want : Want) (i : Nat) : Nat :=
  if i < directoryStart want.size then (boundedWant want).read i
  else boundary want (i - directoryStart want.size)

def upload (want : Want) (base : Nat) (prior : Memory) : Memory := fun address =>
  if base ≤ address ∧ address < base + 4 * (directoryStart want.size + bucketCount + 1) then
    let byte := address - base
    slotByte (packetWord want (byte / 4)) byte
  else prior address

private theorem uploaded_word_bytes (want : Want) (base : Nat) (prior : Memory)
    (word : Nat) (fits : word < directoryStart want.size + bucketCount + 1)
    (j : Nat) (byte : j < 4) :
    upload want base prior (base + 4 * word + j) = slotByte (packetWord want word) j := by
  have live : base ≤ base + 4 * word + j ∧
      base + 4 * word + j < base + 4 * (directoryStart want.size + bucketCount + 1) := by omega
  have offsets : base + 4 * word + j - base = 4 * word + j := by omega
  have index : (4 * word + j) / 4 = word := by omega
  have part : (4 * word + j) % 4 = j % 4 := by omega
  rw [upload, if_pos live, offsets]
  simp only [index, slotByte, part]

theorem uploaded_want {s} (want : Want) (layout : Layout s want.size) (prior : Memory) :
    readWord32 layout (upload want (layout.base .want) prior) .want = (boundedWant want).read := by
  funext i
  by_cases live : i < want.size
  · have bytes : 4 * (i + 1) ≤ Buffer.bytes s want.size .want := by simp [Buffer.bytes]; omega
    have start := (directory_fits want.size).1
    have raw : packetWord want i = want.read i := by
      simp [packetWord, boundedWant, live, show i < directoryStart want.size by omega]
    have range := want.inVocab i live
    rw [read_word32_of_bytes layout _ .want i (want.read i) bytes (by omega)]
    · simp [boundedWant, live]
    · intro j jb
      rw [uploaded_word_bytes want (layout.base .want) prior i (by omega) j jb, raw]
  · have outside : ¬ 4 * (i + 1) ≤ Buffer.bytes s want.size .want := by simp [Buffer.bytes]; omega
    simp [readWord32, outside, boundedWant, live]

set_option maxRecDepth 16384 in
theorem uploaded_boundary {s} (want : Want) (packet : Prepared want)
    (layout : Layout s want.size) (prior : Memory)
    (placement : layout.base .directory = layout.base .want + 4 * directoryStart want.size)
    (i : Nat) (live : i ≤ bucketCount) :
    readWord32 layout (upload want (layout.base .want) prior) .directory i = boundary want i := by
  have bytes : 4 * (i + 1) ≤ Buffer.bytes s want.size .directory := by
    simp only [Buffer.bytes, Nat.mul_comm (bucketCount + 1)]
    exact Nat.mul_le_mul_left 4 (Nat.succ_le_succ live)
  have limit : boundary want i ≤ want.size := native_nat_within (boundedWant want) (i * 32768)
  have small : boundary want i < 4294967296 :=
    Nat.lt_of_le_of_lt limit (Nat.lt_trans packet.fits (by decide))
  apply read_word32_of_bytes layout _ .directory i (boundary want i) bytes small
  intro j jb
  have inPacket : directoryStart want.size + i < directoryStart want.size + bucketCount + 1 := by omega
  have address : layout.base .directory + 4 * i + j =
      layout.base .want + 4 * (directoryStart want.size + i) + j := by rw [placement]; omega
  rw [address, uploaded_word_bytes want (layout.base .want) prior _ inPacket j jb]
  have lower : ¬ directoryStart want.size + i < directoryStart want.size := by omega
  simp only [packetWord, if_neg lower, Nat.add_sub_cancel_left]

theorem upload_ready {s} (want : Want) (packet : Prepared want)
    (layout : Layout s want.size) (prior : Memory)
    (placement : layout.base .directory = layout.base .want + 4 * directoryStart want.size) :
    Ready layout (upload want (layout.base .want) prior) := by
  have contents := uploaded_want want layout prior
  have sorted : ∀ i j, i ≤ j → j < want.size →
      readWord32 layout (upload want (layout.base .want) prior) .want i ≤
        readWord32 layout (upload want (layout.base .want) prior) .want j := by
    rw [contents]
    exact (boundedWant want).ordered
  have valid : ∀ i, i < want.size →
      readWord32 layout (upload want (layout.base .want) prior) .want i < 84602880 := by
    rw [contents]
    exact (boundedWant want).inVocab
  refine ⟨sorted, valid, ?_⟩
  have same : (⟨want.size, readWord32 layout (upload want (layout.base .want) prior) .want, sorted, valid⟩ : Want) =
      boundedWant want := by
    simp only [boundedWant, Want.mk.injEq, true_and]
    exact contents
  rw [same]
  constructor
  · exact uploaded_boundary want packet layout prior placement
  · intro i inside
    rw [uploaded_boundary want packet layout prior placement (i + 1) (by omega),
      uploaded_boundary want packet layout prior placement i (by omega)]
    exact packet.widths i inside

/-- Every aligned position in the directory packet belongs to the wanted
IDs or the directory. The gap is shorter than one alignment unit. -/
theorem upload_preserves_aligned_view {s} (want : Want) (layout : Layout s want.size)
    (prior : Memory) (placement : layout.base .directory = layout.base .want + 4 * directoryStart want.size)
    (wantAligned : layout.base .want % 16 = 0) (buffer : Buffer)
    (aligned : layout.base buffer % 16 = 0) (differentWant : buffer ≠ .want)
    (differentDirectory : buffer ≠ .directory) :
    readWord32 layout (upload want (layout.base .want) prior) buffer = readWord32 layout prior buffer := by
  funext word
  apply word_read_agrees
  intro address here
  unfold upload
  apply if_neg
  intro packet
  let witness := max (layout.base .want) (layout.base buffer)
  have inView : Contains layout buffer witness :=
    ⟨Nat.le_max_right _ _, Nat.lt_of_le_of_lt (Nat.max_le.mpr ⟨packet.1, here.1⟩) here.2⟩
  have fromWant := Nat.le_max_left (layout.base .want) (layout.base buffer)
  have beforeStop := Nat.lt_of_le_of_lt (Nat.max_le.mpr ⟨packet.1, here.1⟩) packet.2
  by_cases first : witness < layout.base .directory
  · have witnessAligned : witness % 16 = 0 := by
      rcases Nat.le_total (layout.base .want) (layout.base buffer) with h | h
      · simp only [witness, Nat.max_eq_right h, aligned]
      · simp only [witness, Nat.max_eq_left h, wantAligned]
    obtain ⟨_, finish, multiple⟩ := directory_fits want.size
    have inside : Contains layout .want witness := by
      refine ⟨fromWant, ?_⟩
      simp only [Buffer.bytes]
      omega
    exact separated_addresses layout differentWant inView inside
  · have inside : Contains layout .directory witness := by
      refine ⟨Nat.le_of_not_gt first, ?_⟩
      simpa only [placement, Buffer.bytes, Nat.mul_add, Nat.mul_one, Nat.add_assoc,
        Nat.add_comm 4 (bucketCount * 4), Nat.mul_comm bucketCount 4] using beforeStop
    exact separated_addresses layout differentDirectory inView inside

def BatchDisjoint {s n} (layout : Layout s n) (transfer : BatchTransfer) (batch : FineWeb.TokenBatch) : Prop :=
  ∀ buffer : Buffer, (buffer = .want ∨ buffer = .directory) →
    ∀ address, Contains layout buffer address → ¬ transfer.touches layout batch address

def sourceField : PinnedBatch.Field → GraphStage.Field
  | .inputs => .inputs
  | .targets => .targets
  | .cumulative => .ends
  | .ngram => .slots

def BatchViews {s n} (layout : Layout s n) (transfer : BatchTransfer) (batch : FineWeb.TokenBatch) : Prop :=
  match transfer with
  | .separate => True
  | .packed base _ => ∀ field,
    layout.base (.source (sourceField field)) = base + PinnedBatch.start (PinnedBatch.counts batch) field ∧
    (Buffer.source (sourceField field)).bytes s n = PinnedBatch.size (PinnedBatch.counts batch) field

/-- At a 16-byte-aligned address the coalesced batch occupies a typed
view if it occupies the packed allocation at all. The launcher checks
alignment and separation of the views. -/
theorem batch_disjoint_of_layout {s n} (layout : Layout s n) (transfer : BatchTransfer)
    (batch : FineWeb.TokenBatch) (count : batch.ngramRows.size = GraphStage.extent s .slots)
    (views : BatchViews layout transfer batch)
    (inputAligned : layout.base (.source .inputs) % 16 = 0)
    (wantAligned : layout.base .want % 16 = 0) (directoryAligned : layout.base .directory % 16 = 0) :
    BatchDisjoint layout transfer batch := by
  intro buffer isDirectory address here touches
  have distinct (f : GraphStage.Field) : buffer ≠ .source f := by
    rcases isDirectory with rfl | rfl <;> cases f <;> decide
  cases transfer with
  | separate =>
    have source : Contains layout (.source .slots) address := by
      simpa only [BatchTransfer.touches, PinnedBatch.size, PinnedBatch.Field.width,
        PinnedBatch.counts, Contains, Buffer.bytes, GraphStageMemory.width, count] using touches
    exact separated_addresses layout (distinct .slots) here source
  | packed base padding =>
    have baseAligned : base % 16 = 0 := by
      have inputs : layout.base (.source .inputs) = base := by
        simpa only [sourceField, PinnedBatch.start, Nat.add_zero] using (views .inputs).1
      rw [inputs] at inputAligned
      exact inputAligned
    have bufferAligned : layout.base buffer % 16 = 0 := by
      rcases isDirectory with rfl | rfl
      · exact wantAligned
      · exact directoryAligned
    obtain ⟨fromPacket, inPacket⟩ := touches
    let witness := max base (layout.base buffer)
    have witnessAligned : witness % 16 = 0 := by
      rcases Nat.le_total base (layout.base buffer) with h | h
      · simp only [witness, Nat.max_eq_right h, bufferAligned]
      · simp only [witness, Nat.max_eq_left h, baseAligned]
    have point : witness ≤ address := Nat.max_le.mpr ⟨fromPacket, here.1⟩
    have fromBase : base ≤ witness := Nat.le_max_left _ _
    have relative : (witness - base) % 16 = 0 := by omega
    have limit : witness - base < PinnedBatch.stop (PinnedBatch.counts batch) .ngram := by omega
    obtain ⟨field, first, last⟩ := PinnedBatch.aligned_offset_in_field (PinnedBatch.counts batch) _ relative limit
    have read : Contains layout buffer witness := ⟨Nat.le_max_right _ _, Nat.lt_of_le_of_lt point here.2⟩
    have source : Contains layout (.source (sourceField field)) witness := by
      unfold Contains
      rw [(views field).1, (views field).2]
      unfold PinnedBatch.stop at last
      omega
    exact separated_addresses layout (distinct _) read source

theorem batch_preserves_word {s n} (layout : Layout s n) (memory : Memory)
    (transfer : BatchTransfer) (batch : FineWeb.TokenBatch) (disjoint : BatchDisjoint layout transfer batch)
    (buffer : Buffer) (isDirectory : buffer = .want ∨ buffer = .directory) :
    readWord32 layout (transfer.upload layout batch memory) buffer = readWord32 layout memory buffer :=
  funext fun word => word_read_agrees layout buffer word _ _ fun address here =>
    transfer.frame layout batch memory address (disjoint buffer isDirectory address here)

theorem ready_after_batch {s n} (layout : Layout s n) (memory : Memory) (ready : Ready layout memory)
    (transfer : BatchTransfer) (batch : FineWeb.TokenBatch) (disjoint : BatchDisjoint layout transfer batch) :
    Ready layout (transfer.upload layout batch memory) := by
  have unchanged := batch_preserves_word layout memory transfer batch disjoint
  have want := unchanged .want (Or.inl rfl)
  have directory := unchanged .directory (Or.inr rfl)
  have ordered (i j) (before : i ≤ j) (within : j < n) :
      readWord32 layout (transfer.upload layout batch memory) .want i ≤
        readWord32 layout (transfer.upload layout batch memory) .want j := by
    rw [want]
    exact ready.ordered i j before within
  have vocab (i) (within : i < n) :
      readWord32 layout (transfer.upload layout batch memory) .want i < 84602880 := by
    rw [want]
    exact ready.inVocab i within
  refine ⟨ordered, vocab, ?_⟩
  have rows : (⟨n, readWord32 layout (transfer.upload layout batch memory) .want, ordered, vocab⟩ : Want)
      = ready.want := by simpa only [Ready.want, Want.mk.injEq, true_and] using want
  rw [rows, directory]
  exact ready.directory

#axiomsClean uploaded_want
#axiomsClean upload_ready
#axiomsClean upload_preserves_aligned_view
#axiomsClean ready_after_batch
#axiomsClean batch_disjoint_of_layout

end VeriTile.Bench.Models.NanoGPT.Track1.NgramDirectoryPacket
