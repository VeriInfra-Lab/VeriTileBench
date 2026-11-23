import VeriTile.Meta.StatementAudit

/-! Warp-private 512-byte exchange slots for the Gluon QKV packer.
The two 64-column phases reuse the same slots. The PTX row pointers for
ldmatrix.x4 are contributed by lanes 8*matrix + row, and each group of
four destination lanes receives the sixteen bytes of that matrix row. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVPack

structure ThreadValue where
  warp : Fin 8
  lane : Fin 32
  reg : Fin 32

abbrev TileIndex := Fin 64 × Fin 128

namespace GluonShared

abbrev Phase := Fin 2

structure ByteLocation where
  phase : Phase
  address : Fin 4096
  deriving DecidableEq

def phase (p : ThreadValue) : Phase :=
  ⟨p.reg.val / 16, by have := p.reg.isLt; omega⟩

/-- The generated pointer is global_smem + ((tid.x << 4) & 4080).
For 256 threads this is exactly the index of a 16-byte slot. -/
def slot (warp : Fin 8) (lane : Fin 32) : Nat := 512 * warp.val + 16 * lane.val

/-- cvt packs the even column in the low byte. The subsequent mov
interleaves the two rows before the vector store. -/
def storeLocation (p : ThreadValue) : ByteLocation :=
  ⟨phase p, ⟨slot p.warp p.lane + 4 * ((p.reg.val % 8) / 2) +
    2 * ((p.reg.val / 8) % 2) + p.reg.val % 2, by
      have := p.warp.isLt
      have := p.lane.isLt
      unfold slot
      omega⟩⟩

/-- The matrix number selects one of the four ldmatrix.x4 results.
The four-lane group selects the matrix row. -/
def ldmatrixRow (w : Fin 8) (lane : Fin 32) (matrix : Fin 4) : Nat :=
  512 * w.val + 16 * (8 * matrix.val + lane.val / 4)

theorem ldmatrix_row_aligned (w : Fin 8) (lane : Fin 32) (m : Fin 4) :
    ldmatrixRow w lane m % 16 = 0 ∧
      512 * w.val ≤ ldmatrixRow w lane m ∧ ldmatrixRow w lane m + 16 ≤ 512 * (w.val + 1) := by
  have hl := lane.isLt
  have hm := m.isLt
  unfold ldmatrixRow
  omega

/-- prmt uses the even or odd byte positions of pairs of matrix
result registers. Inside one word, the destination lane owns four
bytes beginning at 4*(lane % 4). -/
def loadLocation (p : ThreadValue) : ByteLocation :=
  let m : Fin 4 := ⟨(p.reg.val % 8) / 2, by omega⟩
  ⟨phase p, ⟨ldmatrixRow p.warp p.lane m + 4 * (p.lane.val % 4) +
    2 * (p.reg.val % 2) + (p.reg.val / 8) % 2, by
      have hb := (ldmatrix_row_aligned p.warp p.lane m).2.2
      have hw := p.warp.isLt
      omega⟩⟩

theorem store_in_warp (p : ThreadValue) :
    512 * p.warp.val ≤ (storeLocation p).address.val ∧
      (storeLocation p).address.val < 512 * (p.warp.val + 1) := by
  have hl := p.lane.isLt
  simp only [storeLocation, slot]
  omega

theorem load_in_warp (p : ThreadValue) :
    512 * p.warp.val ≤ (loadLocation p).address.val ∧
      (loadLocation p).address.val < 512 * (p.warp.val + 1) := by
  have hb := (ldmatrix_row_aligned p.warp p.lane ⟨(p.reg.val % 8) / 2, by omega⟩).2
  simp only [loadLocation]
  omega

theorem separate_warps (p q : ThreadValue) (hne : p.warp ≠ q.warp) :
    (storeLocation p).address ≠ (loadLocation q).address := by
  intro h
  have ha := congrArg Fin.val h
  have hw := store_in_warp p
  have hr := load_in_warp q
  exact hne (Fin.ext (by omega))

/-- Decode the logical position from the phase and the physical byte in
one warp-private slot. -/
def decode (s : ByteLocation) : TileIndex :=
  let a := s.address.val
  let lane := (a / 16) % 32
  let byte := a % 16
  (⟨8 * (a / 512) + 2 * (lane / 8) + (byte / 2) % 2, by have := s.address.isLt; omega⟩,
   ⟨64 * s.phase.val + 8 * (lane % 8) + 2 * (byte / 4) + byte % 2, by have := s.phase.isLt; omega⟩)

theorem decode_injective {p q : ByteLocation} (e : decode p = decode q) : p = q := by
  have er := congrArg (fun i : TileIndex => i.1.val) e
  have ec := congrArg (fun i : TileIndex => i.2.val) e
  have hp := p.address.isLt
  have hq := q.address.isLt
  have hpp : p.phase.val < 2 := p.phase.isLt
  have hpq : q.phase.val < 2 := q.phase.isLt
  simp only [decode] at er ec
  have phase : p.phase = q.phase := Fin.ext (by omega)
  have address : p.address = q.address := Fin.ext (by omega)
  cases p
  cases q
  congr

inductive Event (phases : Nat) where
  | write (p : Fin phases)
  | barrier
  | read (p : Fin phases)

/-- The PTX has one uniform sequence on every warp. In particular, the
first read has its own barrier before the second phase overwrites it. -/
def warpEvents : List (Event 2) :=
  [.write 0, .barrier, .read 0, .barrier, .write 1, .barrier, .read 1]

private structure ExchangeState (phases : Nat) where
  current : Option (Fin phases) := none
  visible : Bool := false
  reading : Bool := false
  completed : List (Fin phases) := []

private def step {phases : Nat} (state : ExchangeState phases) : Event phases → Option (ExchangeState phases)
  | .write p =>
      if state.reading then none
      else some { state with current := some p, visible := false }
  | .barrier => some { state with visible := true, reading := false }
  | .read p =>
      if state.visible && state.current == some p then
        some { state with reading := true, completed := state.completed ++ [p] }
      else none

def checkedPhases {phases : Nat} (events : List (Event phases)) : Option (List (Fin phases)) := do
  let final ← events.foldlM step {}
  pure final.completed

theorem phase_order : checkedPhases warpEvents = some [0, 1] := rfl

#axiomsClean ldmatrix_row_aligned
#axiomsClean separate_warps
#axiomsClean decode_injective
#axiomsClean phase_order
end GluonShared
end VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
