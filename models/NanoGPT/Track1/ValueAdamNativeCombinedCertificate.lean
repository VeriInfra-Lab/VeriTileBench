import models.NanoGPT.Track1.ValueAdamNativeCombinedTrace
import models.NanoGPT.Track1.ValueAdamCombinedDivisorSpec
import models.NanoGPT.Track1.ValueAdamDivisorBandPTX
import models.NanoGPT.Track1.ValueAdamRootHelper
import models.NanoGPT.Track1.ValueAdamFastDenominator

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeCombinedTrace

open VeriTile.Triton FP
open CombinedDivisor
open DivisorBand (ZeroMode)

private def ZeroMode.layout : ZeroMode → Layout
  | .dynamic => .dynamicZero
  | .reciprocal => .sharedReciprocal
  | .guardedReciprocal => .guardedReciprocal
  | .rootFirst => .rootFirst
  | .directAfterRoot => .directAfterRoot

private def expected (width : Nat) (scaled inverseFirst shared : Bool) (root : RootPlan := .absent) : Output :=
  ⟨(List.range width).map fun index => ⟨index, inverseFirst, scaled, shared⟩,
    if shared then 1 else width, root⟩

private def allWarm (image : Image) (layout : Layout) (reject : Bool) (period : Nat) (masked : Bool) (width : Nat)
    (result : Output) : Bool :=
  (List.finRange (2^width)).all fun mask =>
    [false,true].all fun gradient =>
      decide (image.decode layout ⟨gradient, reject, mask.val, false, true⟩ period masked width = some {result with root := .classified mask.val})

private def fastScale (mode : ZeroMode) (reject : Bool) : Bool :=
  mode != .directAfterRoot && reject

private def allFastRoot (image : Image) (mode : ZeroMode) (period : Nat) (masked : Bool) (width : Nat) : Bool :=
  if mode.guardsRoot then
    [false,true].all fun reject =>
      [false,true].all fun gradient =>
        decide (image.decode (ZeroMode.layout mode) ⟨gradient, reject, 0, false, false⟩ period masked width =
          some (expected width (fastScale mode reject) (fastScale mode reject) false .swappedCorrection))
  else true

private def allCold (image : Image) (layout : Layout) (reject : Bool) (period : Nat) (masked : Bool) (width : Nat)
    (result : Output) : Bool :=
  [false,true].all fun gradient =>
    decide (image.decode layout ⟨gradient, reject, 0, true, true⟩ period masked width = some result)

private theorem checked_warm {masked : Bool} {width : Nat} (image : Image) (layout : Layout) (reject : Bool) (period : Nat)
    (result : Output) (ok : allWarm image layout reject period masked width result = true)
    (mask : Fin (2^width)) (gradient : Bool) :
    image.decode layout ⟨gradient, reject, mask.val, false, true⟩ period masked width = some {result with root := .classified mask.val} := by
  have rows := List.all_eq_true.mp ok mask (List.mem_finRange mask)
  exact of_decide_eq_true (List.all_eq_true.mp rows gradient (by cases gradient <;> simp))

private theorem checked_cold {masked : Bool} {width : Nat} (image : Image) (layout : Layout) (reject : Bool) (period : Nat)
    (result : Output) (ok : allCold image layout reject period masked width result = true) (gradient : Bool) :
    image.decode layout ⟨gradient, reject, 0, true, true⟩ period masked width = some result := by
  exact of_decide_eq_true (List.all_eq_true.mp ok gradient (by cases gradient <;> simp))

private def zeroUsesDynamic (mode : ZeroMode) (reject : Bool) : Bool :=
  match mode with
  | .dynamic => true
  | .reciprocal | .guardedReciprocal | .rootFirst | .directAfterRoot => reject

structure Certificate (oldPTX newPTX oldSASS newSASS : String)
    (mode : ZeroMode) (period : Nat) (masked : Bool := true) (width : Nat := 8) : Type where
  private mk ::
  private ptx : (DivisorBand.compareCombinedPTX oldPTX newPTX mode width).isSome = true
  private original : allWarm (load oldSASS) .reference false period masked width (expected width true true false) = true
  private warmReject : allWarm (load newSASS) (ZeroMode.layout mode) true period masked width (expected width true true false) = true
  private warmDirect : allWarm (load newSASS) (ZeroMode.layout mode) false period masked width (expected width false false false) = true
  private fastRoot : allFastRoot (load newSASS) mode period masked width = true
  private coldReject : allCold (load newSASS) (ZeroMode.layout mode) true period masked width (expected width true true true) = true
  private coldClear : allCold (load newSASS) (ZeroMode.layout mode) false period masked width
    (expected width (zeroUsesDynamic mode false) (zeroUsesDynamic mode false) true) = true

def certify (oldPTX newPTX oldSASS newSASS : String) (mode : ZeroMode) (period : Nat) (masked : Bool := true) (width : Nat := 8) :
    Option (Certificate oldPTX newPTX oldSASS newSASS mode period masked width) := do
  guard ((period == 2 || period == 4) && (width == 2 || width == 8))
  let original := load oldSASS
  let candidate := load newSASS
  let method := ZeroMode.layout mode
  if ptx : (DivisorBand.compareCombinedPTX oldPTX newPTX mode width).isSome = true then
    if base : allWarm original .reference false period masked width (expected width true true false) = true then
      if warmReject : allWarm candidate method true period masked width (expected width true true false) = true then
        if warmDirect : allWarm candidate method false period masked width (expected width false false false) = true then
          if root : allFastRoot candidate mode period masked width = true then
            if coldReject : allCold candidate method true period masked width (expected width true true true) = true then
              if coldClear : allCold candidate method false period masked width
                    (expected width (zeroUsesDynamic mode false) (zeroUsesDynamic mode false) true) = true then
                return ⟨ptx, base, warmReject, warmDirect, root, coldReject, coldClear⟩
              else none
            else none
          else none
        else none
      else none
    else none
  else none

private def rootWarp (warp : List (List Moments)) : List (List F32) :=
  warp.map (List.map Prod.snd)

private def rootFast (mode : ZeroMode) (warp : List (List Moments)) : Bool :=
  mode.guardsRoot && !(DivisorBand.rootReject (rootWarp warp) ||
    (mode == .directAfterRoot && CombinedDivisor.coldReject warp))

private theorem root_clear (mode : ZeroMode) (warp : List (List Moments))
    (fast : rootFast mode warp = true) :
    DivisorBand.rootReject (rootWarp warp) = false := by
  simp only [rootFast, Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_false_iff] at fast
  exact fast.2.1

private theorem numerator_clear (warp : List (List Moments))
    (fast : rootFast .directAfterRoot warp = true) : CombinedDivisor.coldReject warp = false := by
  simp only [rootFast, Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_false_iff] at fast
  exact fast.2.2

private def planFor (mode : ZeroMode) (warp : List (List Moments)) (mask : Fin (2^width)) : RootPlan :=
  if rootFast mode warp then .swappedCorrection else .classified mask.val

private def warmRoute (mode : ZeroMode) (warp : List (List Moments)) (gradient rejected : Bool)
    (rootMask : Fin (2^width)) : Route :=
  if rootFast mode warp then
    ⟨gradient, rejected, 0, false, false⟩
  else ⟨gradient, rejected, rootMask.val, false, true⟩

private def warmScale (mode : ZeroMode) (warp : List (List Moments)) (reject : Bool) : Bool :=
  if rootFast mode warp then fastScale mode reject else reject

private theorem warm_decoded (certificate : Certificate oldPTX newPTX oldSASS newSASS mode period masked width)
    (warp : List (List Moments)) (gradient rejected : Bool) (rootMask : Fin (2^width)) :
    (load newSASS).decode (ZeroMode.layout mode) (warmRoute mode warp gradient rejected rootMask) period masked width =
      some (expected width (warmScale mode warp rejected) (warmScale mode warp rejected)
        false (planFor mode warp rootMask)) := by
  unfold warmRoute
  split
  · rename_i takesFast
    have parts := takesFast
    simp only [rootFast, Bool.and_eq_true] at parts
    have guarded := parts.1
    have casesFast := certificate.fastRoot
    unfold allFastRoot at casesFast
    rw [if_pos guarded] at casesFast
    simp only [planFor, warmScale, if_pos takesFast]
    have guardCase := List.all_eq_true.mp casesFast rejected (by cases rejected <;> simp)
    exact of_decide_eq_true (List.all_eq_true.mp guardCase gradient (by cases gradient <;> simp))
  · rename_i notFast
    simp only [planFor, warmScale, if_neg notFast]
    cases rejected
    · exact checked_warm _ _ _ _ _ certificate.warmDirect rootMask gradient
    · exact checked_warm _ _ _ _ _ certificate.warmReject rootMask gradient

/-- Evaluate the correction format recorded by the accepted native trace.
The words used by the helper and reciprocal square root are common to
the two artifacts. -/
def RootPlan.eval (plan : RootPlan) (rsqrt helper : F32 → F32) (position : Nat)
    (second : F32) : Option F32 :=
  match plan with
  | .absent => none
  | .classified mask corrections =>
    some (if mask.testBit position then helper second else
      RootBand.fastRoot rsqrt second (corrections.testBit position))
  | .swappedCorrection => some (RootBand.fastRoot rsqrt second true)

theorem RootPlan.eval_normalized (plan : RootPlan) (rsqrt helper : F32 → F32)
    (position : Nat) (second : F32) :
    plan.normalized.eval rsqrt helper position second = plan.eval rsqrt helper position second := by
  cases plan with
  | absent | swappedCorrection => rfl
  | classified mask corrections =>
    simp only [RootPlan.normalized, RootPlan.eval, Nat.zero_testBit]
    cases corrections.testBit position <;> simp [RootBand.correction_equal]

/-- The native reciprocal square root and FTZ addition's NaN payload
behavior are shared by both decoded executions. -/
structure RootOps where
  approximation : RootBand.ApproximateRsqrt
  nanAdd : F32 → F32 → F32

def RootOps.rsqrt (ops : RootOps) : F32 → F32 := ops.approximation.run

private def RootOps.helper (ops : RootOps) : F32 → F32 :=
  RootBand.helperAfterZero ops.rsqrt ops.nanAdd

def RootOps.reference (ops : RootOps) : F32 → F32 :=
  RootBand.observedRoot ops.rsqrt ops.helper

def RootOps.denominator (ops : RootOps) (second : F32) : F32 :=
  FP.add Config.ieeeValue (ops.reference second) epsilon

private def nativeDivisionWarp (ops : RootOps) (warp : List (List Moments)) :
    List (List DivisorBand.Pair) :=
  warp.map (List.map (fun point => (point.1, ops.denominator point.2)))

private def nativeRoute (mode : ZeroMode) (ops : RootOps) (warp : List (List Moments))
    (lane : Nat → Moments) (gradient : Bool) (width : Nat) : Route :=
  if hasMagnitude warp then
    warmRoute mode warp gradient (DivisorBand.warpReject (nativeDivisionWarp ops warp))
      (RootBand.scalarHelperMask (fun i => (lane i).2) width)
  else ⟨gradient, coldReject warp, 0, true, DivisorBand.rootReject (rootWarp warp)⟩

private def evalDirection (out : Output) (ops : RootOps) (reciprocal : F32 → F32)
    (nanProduct : Bool → F32 → F32 → F32) (lane : Nat → Moments) (slot : Fin width) :
    Option F32 :=
  out.terms[slot.val]?.bind fun term =>
    let point := lane term.slot
    let root := out.root.eval ops.rsqrt (RootBand.helperWithZero ops.helper) term.slot point.2
    let d := if term.shared then some (sharedDenominator (lane 0).2)
      else root.map (fun value => FP.add Config.ieeeValue value epsilon)
    let divide := if term.scaled then DivisorBand.referenceDivide else DivisorBand.directDivide
    d.map (fun value => divide reciprocal nanProduct point.1 value)

private theorem decoded_evaluation {α : Type} (image : Image) (layout : Layout)
    (route : Route) (period : Nat) (masked : Bool) (width : Nat) (ops : RootOps) (reciprocal : F32 → F32)
    (nanProduct : Bool → F32 → F32 → F32) (lane : Nat → Moments) (slot : Fin width)
    (store : F32 → α) :
    (image.rawDecode layout route period masked width).bind
        (fun out => (evalDirection out ops reciprocal nanProduct lane slot).map store) =
      (image.decode layout route period masked width).bind
        (fun out => (evalDirection out ops reciprocal nanProduct lane slot).map store) := by
  unfold Image.decode
  cases image.rawDecode layout route period masked width <;> simp
    [evalDirection, Output.normalized, RootPlan.eval_normalized]

private theorem evaluated_shape (root : RootPlan) (scaled shared : Bool) (ops : RootOps)
    (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (lane : Nat → Moments) (slot : Fin width) :
    evalDirection (expected width scaled scaled shared root) ops reciprocal nanProduct lane slot =
      let point := lane slot.val
      let d := if shared then some (sharedDenominator (lane 0).2) else
        (root.eval ops.rsqrt (RootBand.helperWithZero ops.helper) slot.val point.2).map
          (fun value => FP.add Config.ieeeValue value epsilon)
      d.map (fun value => (if scaled then DivisorBand.referenceDivide else DivisorBand.directDivide)
        reciprocal nanProduct point.1 value) := by
  simp only [evalDirection, expected, List.getElem?_map, List.getElem?_range slot.isLt, Option.map_some]
  cases scaled <;> cases shared <;> cases root <;> rfl

private theorem root_plan_exact (mode : ZeroMode) (ops : RootOps) (warp : List (List Moments))
    (lane : Nat → Moments) (present : (List.range width).map lane ∈ warp) (slot : Fin width) :
    (planFor mode warp (RootBand.scalarHelperMask (fun i => (lane i).2) width)).eval ops.rsqrt
      (RootBand.helperWithZero ops.helper) slot.val (lane slot.val).2 =
      some (ops.reference (lane slot.val).2) := by
  unfold planFor
  split
  · rename_i fast
    have inWarp : ((List.range width).map lane).map Prod.snd ∈ rootWarp warp :=
      List.mem_map.mpr ⟨_, present, rfl⟩
    have found : (lane slot.val).2 ∈ ((List.range width).map lane).map Prod.snd := by
      exact List.mem_map.mpr ⟨lane slot.val, List.mem_map.mpr
        ⟨slot.val, List.mem_range.mpr slot.isLt, rfl⟩, rfl⟩
    have clear := DivisorBand.root_helper_unneeded _ _ _ inWarp found (root_clear mode warp fast)
    simp [RootPlan.eval, RootOps.reference, RootBand.observedRoot, RootBand.oldRoot, clear,
      RootBand.correction_equal]
  · simp [RootPlan.eval, RootOps.reference, RootBand.observedRoot, RootBand.oldRoot,
      RootBand.scalar_helper_mask_bit]

def Certificate.nativeWeightAt (_certificate : Certificate oldPTX newPTX oldSASS newSASS mode period masked width)
    (ops : RootOps) (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (warp : List (List Moments)) (lane : Nat → Moments) (gradient : Bool)
    (learningRateMul : F32 → F32 → F32) (step decay : F32) (old : Nat → BF16)
    (slot : Fin width) : Option BF16 :=
  ((load newSASS).rawDecode (ZeroMode.layout mode) (nativeRoute mode ops warp lane gradient width) period masked width).bind
    (fun out => (evalDirection out ops reciprocal nanProduct lane slot).map (fun direction =>
      NativeColdTrace.storedWeightWord (old slot.val) (learningRateMul direction step) decay))

theorem Certificate.native_weight_at_exact
    (certificate : Certificate oldPTX newPTX oldSASS newSASS mode period masked width)
    (ops : RootOps) (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (warp : List (List Moments)) (lane : Nat → Moments)
    (inWarp : (List.range width).map lane ∈ warp) (gradient : Bool)
    (learningRateMul : F32 → F32 → F32) (step decay : F32) (old : Nat → BF16)
    (slot : Fin width) :
    certificate.nativeWeightAt ops reciprocal nanProduct warp lane gradient learningRateMul step decay old slot =
      some (NativeColdTrace.storedWeightWord (old slot.val)
        (learningRateMul (DivisorBand.referenceDivide reciprocal nanProduct (lane slot.val).1
          (ops.denominator (lane slot.val).2)) step) decay) := by
  let point := lane slot.val
  let localLane := (List.range width).map lane
  have found : point ∈ localLane :=
    List.mem_map.mpr ⟨slot.val, List.mem_range.mpr slot.isLt, rfl⟩
  unfold Certificate.nativeWeightAt
  rw [decoded_evaluation]
  unfold nativeRoute
  cases magnitude : hasMagnitude warp
  · simp only [Bool.false_eq_true, if_false]
    have first : lane 0 ∈ localLane := List.mem_map.mpr ⟨0, List.mem_range.mpr (lt_of_le_of_lt (Nat.zero_le _) slot.isLt), rfl⟩
    have sourceZero := selected_signed_zero warp localLane (lane 0) inWarp first magnitude
    have targetZero := selected_signed_zero warp localLane point inWarp found magnitude
    have needsRootCheck : DivisorBand.rootReject (rootWarp warp) = true := by
      cases checked : DivisorBand.rootReject (rootWarp warp)
      · have present : localLane.map Prod.snd ∈ rootWarp warp :=
          List.mem_map.mpr ⟨_, inWarp, rfl⟩
        have there : point.2 ∈ localLane.map Prod.snd := List.mem_map.mpr ⟨_, found, rfl⟩
        have helper := DivisorBand.root_helper_unneeded _ _ _ present there checked
        rw [RootBand.root_helper_of_zero _ targetZero] at helper
        contradiction
      · rfl
    rw [needsRootCheck]
    have shared := (zero_denominators (lane 0).2 sourceZero).1
    have den : ops.denominator point.2 = epsilon :=
      RootBand.observed_denominator_of_zero ops.rsqrt ops.helper _ targetZero
    cases rejected : CombinedDivisor.coldReject warp
    · rw [checked_cold _ _ _ _ _ certificate.coldClear gradient]
      have noNaN : point.1.isNaN = false := by
        apply ExponentGuard.no_nan_on_lane_vote (warp.map (List.map Prod.fst)) (localLane.map Prod.fst)
        · exact List.mem_map.mpr ⟨_, inWarp, rfl⟩
        · exact List.mem_map.mpr ⟨_, found, rfl⟩
        · exact rejected
      have divide := DivisorBand.direct_for_pair reciprocal nanProduct (point.1, epsilon) noNaN
        (by change 0x20000000 ≤ epsilon.raw; decide +kernel)
        (by change epsilon.raw < 0x40000000; decide +kernel)
      change _ = some (NativeColdTrace.storedWeightWord _ (learningRateMul
        (DivisorBand.referenceDivide reciprocal nanProduct point.1 (ops.denominator point.2)) step) decay)
      rw [den]
      cases mode <;> simp [evaluated_shape, zeroUsesDynamic, shared, point, divide]
    · rw [checked_cold _ _ _ _ _ certificate.coldReject gradient]
      change _ = some (NativeColdTrace.storedWeightWord _ (learningRateMul
        (DivisorBand.referenceDivide reciprocal nanProduct point.1 (ops.denominator point.2)) step) decay)
      rw [den]
      simp [evaluated_shape, shared, point]
  · simp only [if_true]
    let group := nativeDivisionWarp ops warp
    let pairs := localLane.map (fun item => (item.1, ops.denominator item.2))
    let pair := (point.1, ops.denominator point.2)
    have pairLane : pairs ∈ group := List.mem_map.mpr ⟨_, inWarp, rfl⟩
    have pairFound : pair ∈ pairs := List.mem_map.mpr ⟨_, found, rfl⟩
    rw [warm_decoded certificate]
    simp only [Option.bind_some, evaluated_shape, root_plan_exact _ _ _ _ inWarp, Option.map_some]
    by_cases directRoot : mode = .directAfterRoot ∧ rootFast mode warp = true
    · obtain ⟨rfl, fast⟩ := directRoot
      have rawClear := root_clear .directAfterRoot warp fast
      have firstClear := numerator_clear warp fast
      have inRoot : localLane.map Prod.snd ∈ rootWarp warp := List.mem_map.mpr ⟨_, inWarp, rfl⟩
      have wordFound : point.2 ∈ localLane.map Prod.snd := List.mem_map.mpr ⟨_, found, rfl⟩
      have noNaN : point.1.isNaN = false := by
        apply ExponentGuard.no_nan_on_lane_vote (warp.map (List.map Prod.fst)) (localLane.map Prod.fst)
        · exact List.mem_map.mpr ⟨_, inWarp, rfl⟩
        · exact List.mem_map.mpr ⟨_, found, rfl⟩
        · exact firstClear
      have originalClear := DivisorBand.root_helper_unneeded _ _ _ inRoot wordFound rawClear
      have sharedResult : ops.reference point.2 = RootBand.fastRoot ops.rsqrt point.2 true := by
        simp [RootOps.reference, RootBand.observedRoot, RootBand.oldRoot, originalClear, RootBand.correction_equal]
      obtain ⟨value, finite, minimum, maximum⟩ :=
        RootBand.guarded_denominator_range ops.approximation _ _ _ inRoot wordFound rawClear
      have denominator : (ops.denominator point.2).decode = .finite false value := by
        rw [RootOps.denominator, sharedResult]
        exact finite
      have divide := DivisorBand.direct_for_finite reciprocal nanProduct pair noNaN value denominator minimum maximum
      simp only [warmScale, fast, if_true, fastScale]
      change some (NativeColdTrace.storedWeightWord _ (learningRateMul
        (DivisorBand.directDivide reciprocal nanProduct point.1 (ops.denominator point.2)) step) decay) = _
      change DivisorBand.directDivide reciprocal nanProduct point.1 (ops.denominator point.2) =
        DivisorBand.referenceDivide reciprocal nanProduct point.1 (ops.denominator point.2) at divide
      rw [divide]
    · have scale : warmScale mode warp (DivisorBand.warpReject group) = DivisorBand.warpReject group := by
        unfold warmScale
        split
        · rename_i fast
          have different : mode ≠ .directAfterRoot := fun eq => directRoot ⟨eq, fast⟩
          have notDirect : (mode != ZeroMode.directAfterRoot) = true := by
            cases mode <;> first | rfl | exact (different rfl).elim
          simp [fastScale, notDirect]
        · rfl
      rw [scale]
      cases rejected : DivisorBand.warpReject group
      · have divide := DivisorBand.direct_exact reciprocal nanProduct group pairs pair pairLane pairFound rejected
        change DivisorBand.directDivide reciprocal nanProduct point.1 (ops.denominator point.2) =
          DivisorBand.referenceDivide reciprocal nanProduct point.1 (ops.denominator point.2) at divide
        change some (NativeColdTrace.storedWeightWord _ (learningRateMul
          (DivisorBand.directDivide reciprocal nanProduct point.1 (ops.denominator point.2)) step) decay) = _
        rw [divide]
      · rfl

/-- Evaluate the original instructions on the same raw lane and primitives. -/
def Certificate.referenceWeightAt (_certificate : Certificate oldPTX newPTX oldSASS newSASS mode period masked width)
    (ops : RootOps) (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (lane : Nat → Moments) (gradient : Bool) (learningRateMul : F32 → F32 → F32)
    (step decay : F32) (old : Nat → BF16) (slot : Fin width) : Option BF16 :=
  let mask := RootBand.scalarHelperMask (fun i => (lane i).2) width
  ((load oldSASS).rawDecode .reference ⟨gradient,false,mask.val,false,true⟩ period masked width).bind
    (fun out => (evalDirection out ops reciprocal nanProduct lane slot).map (fun direction =>
      NativeColdTrace.storedWeightWord (old slot.val) (learningRateMul direction step) decay))

theorem Certificate.reference_weight_at_exact
    (certificate : Certificate oldPTX newPTX oldSASS newSASS mode period masked width)
    (ops : RootOps) (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (lane : Nat → Moments) (gradient : Bool) (learningRateMul : F32 → F32 → F32)
    (step decay : F32) (old : Nat → BF16) (slot : Fin width) :
    certificate.referenceWeightAt ops reciprocal nanProduct lane gradient learningRateMul step decay old slot =
      some (NativeColdTrace.storedWeightWord (old slot.val)
        (learningRateMul (DivisorBand.referenceDivide reciprocal nanProduct (lane slot.val).1
          (ops.denominator (lane slot.val).2)) step) decay) := by
  dsimp only [Certificate.referenceWeightAt]
  rw [decoded_evaluation]
  rw [checked_warm _ _ _ _ _ certificate.original]
  change (evalDirection (expected width true true false
    (.classified (RootBand.scalarHelperMask (fun i => (lane i).2) width).val))
      ops reciprocal nanProduct lane slot).map _ = _
  simp [evaluated_shape, RootPlan.eval, RootBand.scalar_helper_mask_bit, RootOps.denominator,
    RootOps.reference, RootBand.observedRoot, RootBand.oldRoot]

#axiomsClean Certificate.reference_weight_at_exact
#axiomsClean Certificate.native_weight_at_exact


end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeCombinedTrace
