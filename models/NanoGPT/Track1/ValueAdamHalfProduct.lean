import models.NanoGPT.Track1.ValueAdamHalfProductCodes0
import models.NanoGPT.Track1.ValueAdamHalfProductCodes1
import models.NanoGPT.Track1.ValueAdamHalfProductCodes2
import models.NanoGPT.Track1.ValueAdamHalfProductCodes3

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

private theorem checked_half_codes : HalfProductSlice 0 65536 :=
  (HalfProductSlice.append (HalfProductSlice.append halfProductCodes0 halfProductCodes1) (HalfProductSlice.append halfProductCodes2 halfProductCodes3))

theorem half_code_exact (word : Half) : selectHalfProduct word = scaleRead word := by
  have check := checked_half_codes ⟨word.toNat, word.toNat_lt⟩
  simpa [HalfProductSlice] using check

/-- The vote chooses the scalar path for every warp containing a NaN.
No NaN payload is supplied to the packed product by the fast path. -/
def guardedScale (words : List Half) : List Half :=
  if words.any halfNaN then words.map scaleRead else words.map finiteHalfProduct

theorem guarded_scale_exact (words : List Half) : guardedScale words = words.map scaleRead := by
  unfold guardedScale
  split
  · rfl
  · apply List.map_congr_left
    intro word present
    rename_i check
    have clear : halfNaN word = false := by
      cases h : halfNaN word
      · rfl
      · exact False.elim (check (List.any_eq_true.mpr ⟨word, present, h⟩))
    simpa [selectHalfProduct, clear] using half_code_exact word

#axiomsClean half_code_exact
#axiomsClean guarded_scale_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
