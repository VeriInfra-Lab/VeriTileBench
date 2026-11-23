import Mathlib.Data.Fintype.Basic
import Lean.Elab.Tactic.Omega
import Init.Data.BitVec.Lemmas
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddFourRowReduction

private def steps : List (Fin 4) := [3, 2, 1, 0]

def shuffleDistances : List Nat := steps.map fun bit => 2 ^ bit.val

private def warpLane (half : Fin 2) (lane : Fin 16) : Fin 32 :=
  ⟨16 * half.val + lane.val, by
    have := half.isLt
    have := lane.isLt
    omega⟩

private def shuffle {lanes : Nat} [NeZero lanes] (lane : Fin lanes) (step : Fin 4) : Fin lanes :=
  ⟨(lane.val ^^^ (2 ^ step.val)) % lanes, Nat.mod_lt _ (Nat.pos_of_ne_zero (NeZero.ne lanes))⟩

private theorem shuffle_stays_in_row (half : Fin 2) (lane : Fin 16) (step : Fin 4) :
    shuffle (warpLane half lane) step = warpLane half (shuffle lane step) := by
  revert half lane step
  decide +kernel

private def combine {α : Type} {lanes : Nat} [NeZero lanes] (add : α → α → α)
    (words : Fin lanes → α) (step : Fin 4) : Fin lanes → α :=
  fun lane => add (words lane) (words (shuffle lane step))

private def getRow {α : Type} (words : Fin 32 → α) (half : Fin 2) : Fin 16 → α :=
  fun lane => words (warpLane half lane)

private theorem step_commutes {α : Type} (add : α → α → α) (words : Fin 32 → α)
    (half : Fin 2) (step : Fin 4) :
    getRow (combine add words step) half = combine add (getRow words half) step := by
  funext lane
  simp only [getRow, combine, shuffle_stays_in_row]

private theorem reduce_commutes {α : Type} (add : α → α → α) (words : Fin 32 → α)
    (half : Fin 2) (steps : List (Fin 4)) :
    getRow (steps.foldl (combine add) words) half = steps.foldl (combine add) (getRow words half) := by
  induction steps generalizing words with
  | nil => rfl
  | cons step steps ih =>
    simp only [List.foldl_cons, ih, step_commutes]

theorem four_shuffles_exact {α : Type} (add : α → α → α) (words : Fin 32 → α) (half : Fin 2) :
    getRow (steps.foldl (combine add) words) half =
      steps.foldl (combine add) (getRow words half) :=
  reduce_commutes add words half _

theorem row_coverage (rows : Nat) (covered : rows % 4 = 0) (row : Fin rows) :
    row.val / 4 < rows / 4 ∧ row.val = 4 * (row.val / 4) + 2 * (row.val / 2 % 2) + row.val % 2 := by
  have := row.isLt
  omega

theorem ptx_thread_coordinates (tid : Fin 64) :
    let word := BitVec.ofNat 32 tid.val
    ((word >>> 4) &&& 3).toNat = tid.val / 16 ∧ (word &&& 15).toNat = tid.val % 16 := by
  revert tid
  decide +kernel

theorem ptx_row_join (cta : BitVec 32) (within : BitVec 2) :
    ((cta <<< 2) ||| within.zeroExtend 32) = (cta <<< 2) + within.zeroExtend 32 := by
  have shift : (cta <<< 2).toNat = (cta.toNat % 1073741824) <<< 2 := by
    rw [BitVec.toNat_shiftLeft]
    simp only [Nat.shiftLeft_eq]
    change cta.toNat * 4 % 4294967296 = (cta.toNat % 1073741824) * 4
    omega
  have lower : (within.zeroExtend 32).toNat = within.toNat :=
    BitVec.toNat_setWidth_of_le (by decide)
  have small : within.toNat < 2 ^ 2 := within.isLt
  have valid : (cta <<< 2).toNat + (within.zeroExtend 32).toNat < 2 ^ 32 := by
    rw [shift, lower, Nat.shiftLeft_eq]
    have high := Nat.mod_lt cta.toNat (by decide : 0 < 1073741824)
    omega
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_or, BitVec.toNat_add_of_lt valid, shift, lower]
  exact (Nat.shiftLeft_add_eq_or_of_lt small _).symm

private def threadWord {α : Type} (input : Fin 2 → Fin 16 → α) (thread : Fin 32) : α :=
  input ⟨thread.val / 16, by have := thread.isLt; omega⟩
    ⟨thread.val % 16, Nat.mod_lt _ (by decide)⟩

private theorem row_of_threads {α : Type} (input : Fin 2 → Fin 16 → α) (half : Fin 2) :
    getRow (threadWord input) half = input half := by
  funext lane
  have row : (16 * half.val + lane.val) / 16 = half.val := by
    have := lane.isLt
    omega
  have column : (16 * half.val + lane.val) % 16 = lane.val := by
    have := lane.isLt
    omega
  unfold getRow threadWord warpLane
  have x : (⟨(16 * half.val + lane.val) / 16, by have := half.isLt; have := lane.isLt; omega⟩ : Fin 2) = half :=
    Fin.ext row
  have y : (⟨(16 * half.val + lane.val) % 16, Nat.mod_lt _ (by decide)⟩ : Fin 16) = lane :=
    Fin.ext column
  rw [x, y]

def referenceRow {α : Type} (add : α → α → α) (values : Fin 16 → α) : α :=
  steps.foldl (combine add) values 0

def tiledWarp {α : Type} (add : α → α → α) (input : Fin 2 → Fin 16 → α) (half : Fin 2) : α :=
  getRow (steps.foldl (combine add) (threadWord input)) half 0

theorem row_output_eq {α : Type} (add : α → α → α) (input : Fin 2 → Fin 16 → α) (half : Fin 2) :
    tiledWarp add input half = referenceRow add (input half) := by
  unfold tiledWarp referenceRow
  rw [four_shuffles_exact, row_of_threads]

theorem row_unique {blocks : Nat} (a b : Fin blocks) (warpA warpB halfA halfB : Fin 2)
    (same : 4 * a.val + 2 * warpA.val + halfA.val = 4 * b.val + 2 * warpB.val + halfB.val) :
    a = b ∧ warpA = warpB ∧ halfA = halfB := by
  have := warpA.isLt
  have := warpB.isLt
  have := halfA.isLt
  have := halfB.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega), Fin.ext (by omega)⟩

theorem eight_row_coverage {rows : Nat} (covered : rows % 8 = 0) (row : Fin rows) :
    row.val / 8 < rows / 8 ∧
      row.val = 8 * (row.val / 8) + 4 * (row.val / 4 % 2) + 2 * (row.val / 2 % 2) + row.val % 2 := by
  have := row.isLt
  omega

theorem eight_row_unique {blocks : Nat} (a b : Fin blocks) (regA regB warpA warpB halfA halfB : Fin 2)
    (same : 8 * a.val + 4 * regA.val + 2 * warpA.val + halfA.val =
      8 * b.val + 4 * regB.val + 2 * warpB.val + halfB.val) :
    a = b ∧ regA = regB ∧ warpA = warpB ∧ halfA = halfB := by
  have := regA.isLt
  have := regB.isLt
  have := warpA.isLt
  have := warpB.isLt
  have := halfA.isLt
  have := halfB.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega), Fin.ext (by omega), Fin.ext (by omega)⟩

def eightRow (reg warp half : Fin 2) : Fin 8 :=
  ⟨4 * reg.val + 2 * warp.val + half.val, by
    have := reg.isLt
    have := warp.isLt
    have := half.isLt
    omega⟩

theorem eight_row_output_eq {α : Type} (add : α → α → α) (words : Fin 8 → Fin 16 → α)
    (reg warp half : Fin 2) :
    tiledWarp add (fun lane => words (eightRow reg warp lane)) half =
      referenceRow add (words (eightRow reg warp half)) :=
  row_output_eq add (fun lane => words (eightRow reg warp lane)) half

#axiomsClean eight_row_output_eq
#axiomsClean eight_row_unique
#axiomsClean eight_row_coverage
#axiomsClean row_unique
#axiomsClean row_output_eq
#axiomsClean ptx_thread_coordinates
#axiomsClean ptx_row_join
#axiomsClean four_shuffles_exact
#axiomsClean row_coverage

end VeriTile.Bench.Models.NanoGPT.Track1.MuddFourRowReduction
