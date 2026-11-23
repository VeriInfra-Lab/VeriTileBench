import models.NanoGPT.Track1.ValueAdamDivisorBand

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand

open VeriTile.Triton FP

private def flush : Config := ⟨.nearestEven, true, true⟩

theorem fma_product_commutes (cfg : Config) (x y z : F32) :
    FP.fma cfg x y z = FP.fma cfg y x z := by
  let a := prepare cfg x
  let b := prepare cfg y
  have nan : (a.isNaN || b.isNaN) = (b.isNaN || a.isNaN) := Bool.or_comm _ _
  have invalid : ((a.isInf && b.isZero) || (a.isZero && b.isInf)) =
      ((b.isInf && a.isZero) || (b.isZero && a.isInf)) := by
    rw [Bool.and_comm a.isInf b.isZero, Bool.and_comm a.isZero b.isInf, Bool.or_comm]
  have inf : (a.isInf || b.isInf) = (b.isInf || a.isInf) := Bool.or_comm _ _
  have sign : (a.sign != b.sign) = (b.sign != a.sign) := bne_comm
  dsimp only [a, b] at nan invalid inf sign
  simp only [FP.fma, nan, invalid, inf, sign]
  split
  · rfl
  · split
    · rfl
    · split
      · rfl
      · cases (prepare cfg x).decode <;> cases (prepare cfg y).decode <;>
          cases (prepare cfg z).decode <;> simp only [Rat.mul_comm, bne_comm]

/-- The decoded fast square-root instructions use the common hardware
reciprocal square root followed by two rounded FTZ multiplies and the
two explicit FP32 fused operations. -/
def fastRoot (rsqrt : F32 → F32) (second : F32) (swapped : Bool) : F32 :=
  let inv := rsqrt second
  let rough := FP.mul flush second inv
  let half := FP.mul flush inv (Value.ofBits .fp32 0x3f000000)
  let residual := FP.fma Config.ieeeValue rough.neg rough second
  if swapped then FP.fma Config.ieeeValue half residual rough
  else FP.fma Config.ieeeValue residual half rough

theorem correction_equal (rsqrt : F32 → F32) (second : F32) :
    fastRoot rsqrt second true = fastRoot rsqrt second false := by
  unfold fastRoot
  dsimp only
  exact fma_product_commutes _ _ _ _

def oldRoot (rsqrt helper : F32 → F32) (second : F32) : F32 :=
  if DivisorBand.originalRootHelper second then helper second else fastRoot rsqrt second false

def selectedRoot (rsqrt helper : F32 → F32) (warp : List (List F32)) (second : F32) : F32 :=
  if DivisorBand.rootReject warp then oldRoot rsqrt helper second else fastRoot rsqrt second true

theorem selected_root_exact (rsqrt helper : F32 → F32) (warp : List (List F32))
    (lane : List F32) (second : F32) (inWarp : lane ∈ warp) (inLane : second ∈ lane) :
    selectedRoot rsqrt helper warp second = oldRoot rsqrt helper second := by
  cases rejected : DivisorBand.rootReject warp
  · have scalar := DivisorBand.root_helper_unneeded warp lane second inWarp inLane rejected
    simp [selectedRoot, oldRoot, rejected, scalar, correction_equal]
  · simp [selectedRoot, rejected]

def scalarHelperMask (lane : Nat → F32) (width : Nat := 8) : Fin (2^width) :=
  let mask := (List.finRange width).foldl (fun (bits : BitVec width) slot =>
    if DivisorBand.originalRootHelper (lane slot.val)
    then bits ||| BitVec.ofNat width (2 ^ slot.val) else bits) 0
  ⟨mask.toNat, mask.isLt⟩

theorem scalar_mask_zero (warp : List (List F32)) (lane : Nat → F32)
    (present : (List.range 8).map lane ∈ warp) (clear : DivisorBand.rootReject warp = false) :
    scalarHelperMask lane = 0 := by
  have each (slot : Fin 8) : DivisorBand.originalRootHelper (lane slot.val) = false := by
    apply DivisorBand.root_helper_unneeded warp _ _ present _ clear
    exact List.mem_map.mpr ⟨slot.val, List.mem_range.mpr slot.isLt, rfl⟩
  apply Fin.ext
  unfold scalarHelperMask
  simp only [each, Bool.false_eq_true, if_false]
  have preserve (initial : BitVec 8) (slots : List (Fin 8)) :
      slots.foldl (fun bits _ => bits) initial = initial := by
    induction slots with
    | nil => rfl
    | cons slot rest induction => simpa only [List.foldl_cons] using induction
  rw [preserve]
  rfl

#axiomsClean correction_equal
#axiomsClean selected_root_exact
#axiomsClean scalar_mask_zero

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand
