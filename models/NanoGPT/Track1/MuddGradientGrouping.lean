import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! An Inductor persistent row reduction reads one BF16 plane and
writes one column of a row-major BF16 destination. A grouped launch
adds `program_id(1)` as the plane selector and folds its constant factor
into the output pointer. The reduction for every launched tile is unchanged. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientGrouping
open VeriTile.Spec
open scoped VeriTile.Spec
open VeriTile.Triton.FP

abbrev BF16 := Value .bf16
abbrev F32 := Value .fp32
abbrev Memory := Nat → BF16

structure Packet (rows planes : Nat) where
  input : Fin planes → Fin rows → Fin 12 → BF16
  reduce : (Fin 16 → F32) → F32
  base : Nat
  first : Nat
  memory : Memory
  tile : Nat := 1
  warps : Nat := 2

def reduceRow {rows planes : Nat} (packet : Packet rows planes) (plane : Fin planes) (row : Fin rows) : BF16 :=
  cast .bf16 Config.ieeeValue (packet.reduce fun c =>
    if h : c.val < 12 then cast .fp32 Config.ieeeValue (packet.input plane row ⟨c.val, h⟩)
    else Value.zero .fp32)

def rowSlot (base addr : Nat) : Nat := (addr - base) / 2 / 10

def colSlot (base addr : Nat) : Nat := (addr - base) / 2 % 10

def address {rows planes : Nat} (packet : Packet rows planes) (row : Fin rows) (plane : Fin planes) : Nat :=
  packet.base + 2 * (row.val * 10 + packet.first + plane.val)

structure Cell (rows planes : Nat) where
  row : Fin rows
  plane : Fin planes

def locate {rows planes : Nat} (packet : Packet rows planes) (addr : Nat) : Option (Cell rows planes) :=
  let row := rowSlot packet.base addr
  let col := colSlot packet.base addr
  if h : packet.base ≤ addr ∧ (addr - packet.base) % 2 = 0 ∧ row < rows ∧
      packet.first ≤ col ∧ col - packet.first < planes then
    some ⟨⟨row, h.2.2.1⟩, ⟨col - packet.first, h.2.2.2.2⟩⟩
  else none

theorem locate_address {rows planes : Nat} (packet : Packet rows planes) (row : Fin rows) (plane : Fin planes)
    (within : packet.first + planes ≤ 10) : locate packet (address packet row plane) = some ⟨row, plane⟩ := by
  have r := row.isLt
  have p := plane.isLt
  have col : packet.first + plane.val < 10 := by omega
  have hrow : rowSlot packet.base (address packet row plane) = row.val := by
    unfold rowSlot address
    omega
  have hcol : colSlot packet.base (address packet row plane) = packet.first + plane.val := by
    unfold colSlot address
    omega
  have base : packet.base ≤ address packet row plane := by unfold address; omega
  have even : (address packet row plane - packet.base) % 2 = 0 := by unfold address; omega
  unfold locate
  dsimp only
  split
  · congr 2
    · exact Fin.ext hrow
    · apply Fin.ext
      change colSlot packet.base (address packet row plane) - packet.first = plane.val
      omega
  · omega

def selectPointer {planes : Nat} (bases : Fin planes → Nat) (first chosen : Fin planes) : Nat :=
  (List.finRange planes).foldl (fun selected plane => if chosen = plane then bases plane else selected) (bases first)

theorem select_pointer {planes : Nat} (bases : Fin planes → Nat) (first chosen : Fin planes) :
    selectPointer bases first chosen = bases chosen := by
  have select : ∀ (lanes : List (Fin planes)) (prior : Nat),
      lanes.foldl (fun selected plane => if chosen = plane then bases plane else selected) prior =
        if chosen ∈ lanes then bases chosen else prior := by
    intro lanes
    induction lanes with
    | nil => simp
    | cons hd tl ih =>
      intro prior
      rw [List.foldl_cons, ih]
      by_cases h : chosen = hd
      · subst hd
        simp
      · by_cases m : chosen ∈ tl <;> simp [h, m]
  rw [selectPointer, select]
  simp

theorem row_in_launch {rows tile : Nat} (row : Fin rows) (positive : 0 < tile) (covered : rows % tile = 0) :
    row.val / tile < rows / tile ∧ row.val % tile < tile ∧
      row.val / tile * tile + row.val % tile = row.val := by
  have rowBound := row.isLt
  have extent : rows / tile * tile = rows := by
    rw [Nat.mul_comm, ← Nat.zero_add (tile * (rows / tile)), ← covered, Nat.mod_add_div]
  refine ⟨?_, Nat.mod_lt _ positive, ?_⟩
  · rw [Nat.div_lt_iff_lt_mul positive, extent]
    exact rowBound
  · rw [Nat.mul_comm, Nat.add_comm, Nat.mod_add_div]

def writeColumn {rows planes : Nat} (packet : Packet rows planes) (addr : Nat)
    (prior : BF16) (plane : Fin planes) : BF16 :=
  match locate packet addr with
  | some cell => if cell.plane = plane then reduceRow packet plane cell.row else prior
  | none => prior

def separate {rows planes : Nat} (packet : Packet rows planes) : Memory := fun addr =>
  (List.finRange planes).reverse.foldl (writeColumn packet addr) (packet.memory addr)

def grouped {rows planes : Nat} (packet : Packet rows planes) : Memory := fun addr =>
  (locate packet addr).elim (packet.memory addr) fun cell => reduceRow packet cell.plane cell.row

private theorem fold_write {n : Nat} (xs : List (Fin n)) (expected : Fin n) (value prior : BF16) :
    (xs.foldl (fun current plane => if expected = plane then value else current) prior) =
      if expected ∈ xs then value else prior := by
  induction xs generalizing prior with
  | nil => simp
  | cons hd tl ih =>
    rw [List.foldl_cons, ih]
    by_cases h : expected = hd
    · subst hd
      simp
    · by_cases m : expected ∈ tl <;> simp [h, m]

theorem run_eq {rows planes : Nat} (packet : Packet rows planes) : grouped packet = separate packet := by
  funext addr
  unfold grouped separate writeColumn
  cases h : locate packet addr with
  | none =>
    simp only [Option.elim]
    induction (List.finRange planes).reverse with
    | nil => rfl
    | cons p rest ih => exact ih
  | some cell =>
    simp only [Option.elim]
    have eq : (fun (old : BF16) (plane : Fin planes) =>
        if cell.plane = plane then reduceRow packet plane cell.row else old) =
      (fun (old : BF16) (plane : Fin planes) =>
        if cell.plane = plane then reduceRow packet cell.plane cell.row else old) := by
      funext old plane
      by_cases same : cell.plane = plane
      · subst plane
        rfl
      · simp [same]
    rw [eq, fold_write]
    simp

inductive Program where
  | separate
  | grouped

abbrev Run := (rows planes : Nat) → Packet rows planes → Memory

def Program.run (p : Program) : Run := fun _ _ packet =>
  match p with
  | .separate => MuddGradientGrouping.separate packet
  | .grouped => MuddGradientGrouping.grouped packet

def same (p q : Program) : Prop := p.run = q.run

instance : Spec.ProgramSyntax Program where
  Statement := Unit
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some same

theorem grouped_run_eq : Program.grouped.run = Program.separate.run := by
  funext rows planes packet
  exact run_eq packet

specification grouped_equiv : Program.grouped ≡[([] : Spec.Assumptions Unit)] Program.separate := by
  exact Spec.FloatingPoint.ofStructural (structural := same) rfl rfl grouped_run_eq

#print_fp_assumptions grouped_equiv
#axiomsClean locate_address
#axiomsClean select_pointer
#axiomsClean row_in_launch
#axiomsClean grouped_equiv
end VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientGrouping
