import VeriTile.Triton.Float.ScalarOpsLaws
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization
import Mathlib.Data.Fintype.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic
open VeriTile.Triton.FP

/-- The packer starts with column 9, then adds columns 8 through 0. -/
def total (selected : Fin 10 → Value .fp32) : Value .fp32 :=
  (List.finRange 9).reverse.foldl (fun sum i => add Config.ieeeValue sum (selected i.castSucc)) (selected 9)

private def select (x : Value .fp32) (plane slot : Fin 10) : Value .fp32 :=
  if plane = slot then x else Value.zero .fp32

set_option maxRecDepth 4096 in
private theorem selected_total (value : Value .bf16) (plane : Fin 10) :
    total (select (cast .fp32 Config.ieeeValue value) plane) = addZero (cast .fp32 Config.ieeeValue value) := by
  let x := cast .fp32 Config.ieeeValue value
  have right : add Config.ieeeValue x (Value.zero .fp32) = addZero x := rfl
  have left : add Config.ieeeValue (Value.zero .fp32) x = addZero x := ScalarOpsLaws.add_comm Config.ieeeValue _ _
  have afterFirst : add Config.ieeeValue (addZero x) (Value.zero .fp32) = addZero x := add_zero_twice value
  have onlyZeros : add Config.ieeeValue (Value.zero .fp32) (Value.zero .fp32) = Value.zero .fp32 := by decide +kernel
  have columns : (List.finRange 9).reverse = [8, 7, 6, 5, 4, 3, 2, 1, 0] := by decide +kernel
  change (List.finRange 9).reverse.foldl
    (fun sum i => add Config.ieeeValue sum (select x plane i.castSucc)) (select x plane 9) = addZero x
  rw [columns]
  fin_cases plane <;> simp [select, List.foldl, right, left, afterFirst, onlyZeros, Fin.ext_iff]

theorem add_zero_selected (value : Value .bf16) (plane : Fin 10) :
    total (fun slot => if plane = slot then cast .fp32 Config.ieeeValue value else Value.zero .fp32) =
      addZero (cast .fp32 Config.ieeeValue value) :=
  selected_total value plane

#axiomsClean add_zero_selected
end VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic
