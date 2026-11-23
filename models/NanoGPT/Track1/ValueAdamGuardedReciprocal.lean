import models.NanoGPT.Track1.ValueAdamElementwise

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.GuardedReciprocal

open VeriTile.Triton FP

def epsilon : F32 := Value.ofBits .fp32 0x2edbe6ff

def reference (reciprocal : F32 → F32) (moment : F32) : F32 :=
  FP.mul Config.ieeeValue (reciprocal epsilon) moment

def direct (reciprocal : F32 → F32) (moment : F32) : F32 :=
  FP.mul Config.ieeeValue moment (reciprocal epsilon)

def warpHasNaN (warp : List F32) : Bool :=
  warp.any Value.isNaN

def selected (reciprocal : F32 → F32) (warp : List F32) (moment : F32) : F32 :=
  if warpHasNaN warp then reference reciprocal moment else direct reciprocal moment

theorem no_nan_on_direct_path (warp : List F32) (moment : F32) (inWarp : moment ∈ warp)
    (directPath : warpHasNaN warp = false) : moment.isNaN = false := by
  cases code : moment.isNaN
  · rfl
  · have yes : warpHasNaN warp = true := List.any_eq_true.mpr ⟨moment, inWarp, code⟩
    rw [directPath] at yes
    contradiction

private theorem mul_commute (a b : F32) : FP.mul Config.ieeeValue a b = FP.mul Config.ieeeValue b a := by
  have sign_comm (x y : Bool) : (x != y) = (y != x) := by cases x <;> cases y <;> rfl
  unfold FP.mul
  cases (prepare Config.ieeeValue a).decode <;> cases (prepare Config.ieeeValue b).decode <;>
    simp [Rat.mul_comm, sign_comm]

/-- The comparison takes place after the reference SASS has selected the
ordinary divider for the already-checked epsilon word. Both use the same
hardware reciprocal. The NaN route keeps the reference operand order. -/
theorem selected_exact (reciprocal : F32 → F32) (warp : List F32) (moment : F32) :
    selected reciprocal warp moment = reference reciprocal moment := by
  unfold selected
  split
  · rfl
  · exact mul_commute _ _

#axiomsClean no_nan_on_direct_path
#axiomsClean selected_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.GuardedReciprocal
