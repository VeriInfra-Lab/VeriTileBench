import VeriTile.Meta.StatementAudit

/-! The live pull's local slice precedes the IDs received from peers in
the gradient merge. The candidate shares the aligned packed prefix with
the local reader and writes an aligned mirror for the peer reader. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AdamEntries

structure Input where
  want : Array Int32
  asked : Array Int32
  lo : Nat
  hi : Nat
  first : Int32
  ordered : lo ≤ hi
  valid : hi ≤ want.size

def Input.localCount (s : Input) : Nat := s.hi - s.lo
def Input.peerCount (s : Input) : Nat := s.asked.size
def Input.count (s : Input) : Nat := s.localCount + s.peerCount

structure Indices (s : Input) where
  localRows : Fin s.localCount → Int32
  peerRows : Fin s.peerCount → Int32
  entries : Fin s.count → Int32

def localRead (s : Input) (i : Fin s.localCount) : Int32 :=
  s.want[s.lo + i.val]'(by
    have := s.ordered
    have := s.valid
    have := i.isLt
    unfold Input.localCount at *
    omega) - s.first

def peerRead (s : Input) (i : Fin s.peerCount) : Int32 :=
  s.asked[i.val]'i.isLt - s.first

def concatenate (s : Input) (left : Fin s.localCount → Int32) (right : Fin s.peerCount → Int32)
    (i : Fin s.count) : Int32 :=
  if h : i.val < s.localCount then left ⟨i.val, h⟩
  else right ⟨i.val - s.localCount, by have := i.isLt; unfold Input.count at *; omega⟩

def staged (s : Input) : Indices s :=
  let own := localRead s
  let peer := peerRead s
  ⟨own, peer, concatenate s own peer⟩

def packedRead (s : Input) (i : Fin s.count) : Int32 :=
  (if h : i.val < s.localCount then
    s.want[s.lo + i.val]'(by
      have := s.valid
      have := s.ordered
      unfold Input.localCount at h
      omega)
  else s.asked[i.val - s.localCount]'(by
    have := i.isLt
    unfold Input.count Input.peerCount at *
    omega)) - s.first

def localSlot (s : Input) (i : Fin s.localCount) : Fin s.count :=
  ⟨i.val, by have := i.isLt; unfold Input.count; omega⟩

def peerSlot (s : Input) (i : Fin s.peerCount) : Fin s.count :=
  ⟨s.localCount + i.val, by have := i.isLt; unfold Input.count; omega⟩

def fused (s : Input) : Indices s :=
  ⟨fun i => packedRead s (localSlot s i),
   fun i => packedRead s (peerSlot s i),
   packedRead s⟩

theorem packed_exact (s : Input) (i : Fin s.count) : packedRead s i = (staged s).entries i := by
  simp only [packedRead, staged, concatenate, localRead, peerRead]
  split <;> rfl

theorem local_exact (s : Input) (i : Fin s.localCount) :
    (fused s).localRows i = (staged s).localRows i := by
  simp [fused, localSlot, packedRead, localRead, staged]

theorem peer_exact (s : Input) (i : Fin s.peerCount) :
    (fused s).peerRows i = (staged s).peerRows i := by
  have peer : ¬s.localCount + i.val < s.localCount := by omega
  simp [fused, peerSlot, packedRead, peerRead, staged, peer]
  rfl

abbrev Pack := (s : Input) → Indices s

theorem pack_equal : (fused : Pack) = staged := by
  funext s
  simp only [fused, staged, Indices.mk.injEq]
  exact ⟨funext (local_exact s), funext (peer_exact s), funext (packed_exact s)⟩

def peerStart (count : Nat) : Nat := (count + 3) / 4 * 4

theorem mirror_layout (count : Nat) :
    count ≤ peerStart count ∧ peerStart count < count + 4 ∧
      (4 * peerStart count) % 16 = 0 := by
  unfold peerStart
  omega

theorem peer_mirror_separate (count peerCount : Nat) (i : Fin peerCount) :
    count ≤ peerStart count + i.val ∧ peerStart count + i.val < peerStart count + peerCount := by
  have := (mirror_layout count).1
  have := i.isLt
  omega

theorem slot_coordinate (slot : Nat) : slot / 128 * 128 + slot % 128 = slot := by omega

/-- The gradient initializer's three 16-byte stores per lane. A tail
row whose entry is missing issues no write. -/
def gradientTile (count block row stripe lane piece : Nat) : Option (Nat × UInt16) :=
  let slot := 4 * block + row
  if slot < count then
    some (slot * 768 + stripe * 256 + lane * 8 + piece, 0)
  else none

def gradientWord {count : Nat} (slot : Fin count) (channel : Fin 768) : Option (Nat × UInt16) :=
  gradientTile count (slot.val / 4) (slot.val % 4)
    (channel.val / 256) (channel.val % 256 / 8) (channel.val % 8)

theorem gradient_word_exact {count : Nat} (slot : Fin count) (channel : Fin 768) :
    gradientWord slot channel = some (slot.val * 768 + channel.val, 0) := by
  have entry : 4 * (slot.val / 4) + slot.val % 4 = slot.val := by omega
  have column : channel.val / 256 * 256 + channel.val % 256 / 8 * 8 + channel.val % 8 =
      channel.val := by omega
  have combined : slot.val * 768 + channel.val / 256 * 256 +
      channel.val % 256 / 8 * 8 + channel.val % 8 = slot.val * 768 + channel.val := by
    calc
      _ = slot.val * 768 +
        (channel.val / 256 * 256 + channel.val % 256 / 8 * 8 + channel.val % 8) := by ac_rfl
      _ = _ := congrArg (slot.val * 768 + ·) column
  simp only [gradientWord, gradientTile, entry, if_pos slot.isLt, combined]

theorem gradient_addresses_unique {count : Nat} (slot another : Fin count) (column other : Fin 768)
    (same : gradientWord slot column = gradientWord another other) : slot = another ∧ column = other := by
  simp only [gradient_word_exact, Option.some.injEq, Prod.mk.injEq, and_true] at same
  have c := column.isLt
  have d := other.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

#axiomsClean pack_equal
#axiomsClean peer_mirror_separate
#axiomsClean slot_coordinate
#axiomsClean gradient_word_exact
#axiomsClean gradient_addresses_unique
end VeriTile.Bench.Models.NanoGPT.Track1.AdamEntries
