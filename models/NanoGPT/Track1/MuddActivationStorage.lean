import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Spec
import Mathlib.Logic.Equiv.Fin.Basic

/-! The full Inductor graph writes the same E4M3 tile to two arenas. Its
logit GEMM reads the first arena, then that arena is reused for the
transpose destination. The optimized wrapper swaps the source and
destination arenas of that transpose and elides the second write. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddActivationStorage
open VeriTile.Spec
open scoped VeriTile.Spec

set_option maxRecDepth 2048

abbrev RowBytes (rows : Nat) := Fin rows → Fin 768 → UInt8
abbrev ColumnBytes (rows : Nat) := Fin 768 → Fin rows → UInt8
abbrev Arena (rows : Nat) := Fin (rows * 768) → UInt8

def packRows {rows : Nat} (words : RowBytes rows) : Arena rows := fun address =>
  let (row, col) := finProdFinEquiv.symm address
  words row col

def readRows {rows : Nat} (arena : Arena rows) : RowBytes rows := fun row col =>
  arena (finProdFinEquiv (row, col))

def transposeOffset (rows : Nat) : Fin 768 × Fin rows ≃ Fin (rows * 768) :=
  finProdFinEquiv.trans (finCongr (Nat.mul_comm 768 rows))

def writeTranspose {rows : Nat} (arena : Arena rows) : Arena rows := fun address =>
  let (col, row) := (transposeOffset rows).symm address
  readRows arena row col

def readColumns {rows : Nat} (arena : Arena rows) : ColumnBytes rows := fun col row =>
  arena (transposeOffset rows (col, row))

structure Scratch (rows : Nat) where
  first : Arena rows
  second : Arena rows
  surrounding : Nat → UInt8

inductive Program where
  | duplicate
  | single

def Program.forward {rows : Nat} (program : Program) (encoded : RowBytes rows) (scratch : Scratch rows) :=
  match program with
  | .duplicate => {scratch with first := packRows encoded, second := packRows encoded}
  | .single => {scratch with first := packRows encoded}

def Program.source {rows : Nat} (program : Program) (scratch : Scratch rows) : Arena rows :=
  match program with
  | .duplicate => scratch.second
  | .single => scratch.first

def Program.storeTranspose {rows : Nat} (program : Program) (scratch : Scratch rows) : Scratch rows :=
  let words := writeTranspose (program.source scratch)
  match program with
  | .duplicate => {scratch with first := words}
  | .single => {scratch with second := words}

def Program.readTranspose {rows : Nat} (program : Program) (scratch : Scratch rows) : ColumnBytes rows :=
  match program with
  | .duplicate => readColumns scratch.first
  | .single => readColumns scratch.second

structure Observed (rows : Nat) where
  logitInput : RowBytes rows
  weightGradientInput : ColumnBytes rows
  surrounding : Nat → UInt8

def Program.run {rows : Nat} (program : Program) (encoded : RowBytes rows) (scratch : Scratch rows) :
    Observed rows :=
  let written := program.forward encoded scratch
  let backward := program.storeTranspose written
  ⟨readRows written.first, program.readTranspose backward, backward.surrounding⟩

theorem transpose_exact {rows : Nat} (words : RowBytes rows) (feature : Fin 768) (row : Fin rows) :
    readColumns (writeTranspose (packRows words)) feature row = words row feature := by
  simp [readColumns, writeTranspose, readRows, packRows]

def sameRun (left right : Program) : Prop :=
  ∀ (rows : Nat) encoded scratch, @Program.run rows left encoded scratch = @Program.run rows right encoded scratch

instance : Spec.ProgramSyntax Program where
  Statement := Unit
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some sameRun

specification single_equiv :
    Program.single ≡[([] : Spec.Assumptions Unit)] Program.duplicate := by
  refine Spec.FloatingPoint.ofStructural (structural := sameRun) rfl rfl ?_
  intro rows encoded scratch
  rfl

abbrev Read := (rows : Nat) → RowBytes rows → RowBytes rows

def Program.read (program : Program) : Read := fun _ data =>
  readRows (program.source (program.forward data ⟨fun _ => 0, fun _ => 0, fun _ => 0⟩))

theorem read_eq : Program.single.read = Program.duplicate.read := rfl

theorem read_exact (p : Program) : p.read = fun _ data => data := by
  funext rows data row col
  cases p <;> simp [Program.read, Program.source, Program.forward, readRows, packRows]

#print_fp_assumptions single_equiv
#axiomsClean single_equiv
#axiomsClean read_eq
#axiomsClean read_exact
#axiomsClean transpose_exact
end VeriTile.Bench.Models.NanoGPT.Track1.MuddActivationStorage
