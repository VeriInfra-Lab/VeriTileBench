import models.NanoGPT.Track1.ValueAdamRootBandMask

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand

open VeriTile.Triton FP

private def ftz : Config := ⟨.nearestEven, true, true⟩
private def zero : F32 := Value.zero .fp32
private def one : F32 := Value.ofBits .fp32 0x3f800000
private def half : F32 := Value.ofBits .fp32 0x3f000000
private def scaleUp : F32 := Value.ofBits .fp32 0x5f800000
private def scaleDown : F32 := Value.ofBits .fp32 0x2f800000
private def negativeRoot : F32 := Value.ofBits .fp32 0x7fffffff

/-- Evaluate the verified nonzero suffix of the original square-root
helper. The FTZ ordered comparison implements the complement of the
first unordered predicate. The NaN branch retains the native addition
payload behavior. -/
def helperAfterZero (rsqrt : F32 → F32) (nanAdd : F32 → F32 → F32) (word : F32) : F32 :=
  if FP.lt ftz word zero then negativeRoot
  else if word.isNaN then nanAdd word one
  else if word.isInf then word
  else
    let scaled := FP.fma Config.ieeeValue word scaleUp zero
    let reciprocal := rsqrt scaled
    let rough := FP.mul ftz scaled reciprocal
    let halfReciprocal := FP.mul ftz reciprocal half
    let minusRough := FP.add ftz rough.neg (Value.zero .fp32 true)
    let residual := FP.fma Config.ieeeValue rough minusRough scaled
    let corrected := FP.fma Config.ieeeValue residual halfReciprocal rough
    FP.mul ftz corrected scaleDown

theorem helper_special_words (rsqrt : F32 → F32) (nanAdd : F32 → F32 → F32) :
    helperWithZero (helperAfterZero rsqrt nanAdd) (Value.zero .fp32) = Value.zero .fp32 ∧
    helperWithZero (helperAfterZero rsqrt nanAdd) (Value.zero .fp32 true) = Value.zero .fp32 true ∧
    helperAfterZero rsqrt nanAdd (Value.infinity .fp32 true) = negativeRoot ∧
    helperAfterZero rsqrt nanAdd (Value.infinity .fp32) = Value.infinity .fp32 ∧
    helperAfterZero rsqrt nanAdd (Value.ofBits .fp32 0x7f800001) =
      nanAdd (Value.ofBits .fp32 0x7f800001) one := by
  have negInf : FP.lt ftz (Value.infinity .fp32 true) zero = true := by decide +kernel
  have posInf : FP.lt ftz (Value.infinity .fp32 false) zero = false := by decide +kernel
  have isInf : (Value.infinity .fp32 false).isInf = true := by decide +kernel
  have notNaN : (Value.infinity .fp32 false).isNaN = false := by decide +kernel
  have nanLt : FP.lt ftz (Value.ofBits .fp32 0x7f800001) zero = false := by decide +kernel
  have nan : (Value.ofBits .fp32 0x7f800001).isNaN = true := by decide +kernel
  simp only [helperAfterZero, negInf, posInf, isInf, notNaN, nanLt, nan, Bool.false_eq_true,
    if_false, if_true, and_self]
  exact ⟨rfl, rfl, True.intro⟩

theorem helper_exact_power :
    helperAfterZero (fun _ => Value.ofBits .fp32 0x4f000000) (FP.add ftz)
      (Value.ofBits .fp32 0x00800000) = Value.ofBits .fp32 0x20000000 := by
  decide +kernel

#axiomsClean helper_special_words
#axiomsClean helper_exact_power

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand
