import models.NanoGPT.Track1.MuddNormHeadReplay

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddBackward
open VeriTile.Triton.FP
open VeriTile.Spec
open scoped VeriTile.Spec
open MuddNorm (Native BF16 F32 xorReduce)

def layout (launch : MuddNorm.BackwardLayout) : Nat :=
  match launch with | .withQuantization => 16 | .gradientOnly => 8

def channel (launch : MuddNorm.BackwardLayout) (lane word : Nat) : Nat :=
  let width := layout launch
  width * lane + word % width + (32 * width) * (word / width)

structure Input where
  mixed : Fin 768 → F32
  gradient : Fin 768 → BF16

abbrev zero32 := Value.zero .fp32
private def extend (x : BF16) : F32 := cast .fp32 Config.ieeeValue x
private def add32 := @add .fp32 Config.ieeeValue
private def mul32 := @mul .fp32 Config.ieeeValue
private def fma32 := @fma .fp32 Config.ieeeValue

def operands (launch : MuddNorm.BackwardLayout) (input : Input) (lane word : Nat) : F32 × F32 :=
  let col := channel launch lane word
  if live : col < 768 then (extend (input.gradient ⟨col, live⟩), input.mixed ⟨col, live⟩)
  else (zero32, zero32)

private def paddedTailProduct (input : Input) (lane offset : Nat) : F32 :=
  let (g, x) := operands .withQuantization input lane (offset + 16)
  mul32 g x

private def maskedTailProduct (input : Input) (lane offset : Nat) : F32 :=
  if lane < 16 then paddedTailProduct input lane offset else zero32

theorem padding_products_are_zero :
    cast .fp32 Config.ieeeValue (Value.zero .bf16) = zero32 ∧ mul32 zero32 zero32 = zero32 := by
  decide +kernel

private theorem padded_tail_matches_masked (input : Input) (lane offset : Nat) :
    paddedTailProduct input lane offset = maskedTailProduct input lane offset := by
  by_cases live : lane < 16
  · simp only [maskedTailProduct, if_pos live]
  · have outside : ¬channel .withQuantization lane (offset + 16) < 768 := by
      simp only [channel, layout]
      omega
    simpa only [paddedTailProduct, maskedTailProduct, operands, dif_neg outside, if_neg live]
      using padding_products_are_zero.2

private def dotPartialUsing (tail : Input → Nat → Nat → F32)
    (launch : MuddNorm.BackwardLayout) (input : Input) (lane : Nat) : F32 :=
  let get := operands launch input lane
  let seed := mul32 (get 1).1 (get 1).2
  let liveWords := match launch with | .withQuantization => 16 | .gradientOnly => 24
  let partialSum := ((List.range liveWords).erase 1).foldl (fun acc word =>
    let (g, x) := get word
    fma32 g x acc) seed
  match launch with
  | .withQuantization =>
    (List.range 16).foldl (fun acc offset => add32 (tail input lane offset) acc) partialSum
  | .gradientOnly => add32 partialSum zero32

def dotPartial := dotPartialUsing maskedTailProduct

private def dotPartialPadded := dotPartialUsing paddedTailProduct

theorem dotPartialPadded_eq : dotPartialPadded = dotPartial := by
  have tail : paddedTailProduct = maskedTailProduct := funext fun input =>
    funext fun lane => funext (padded_tail_matches_masked input lane)
  simp only [dotPartialPadded, dotPartial, tail]

#axiomsClean padding_products_are_zero
#axiomsClean dotPartialPadded_eq

structure Output where
  gradient : Fin 768 → BF16
  encoded : Option (Fin 768 → UInt8)

private def backwardUsing (partialSum : MuddNorm.BackwardLayout → Input → Nat → F32)
    (launch : MuddNorm.BackwardLayout) (lane : Fin 768 → Nat)
    (native : Native) (input : Input) (rstd : F32) : Output :=
  let dot := xorReduce [16, 8, 4, 2, 1] (partialSum launch input)
  let cube := mul32 rstd (mul32 rstd rstd)
  let dx := fun c =>
    let alpha := mul32 (mul32 (mul32 (dot (lane c)) (Value.ofBits .fp32 0xbf000000)) cube)
      (Value.ofBits .fp32 0x3aaaaaab)
    let x := input.mixed c
    cast .bf16 Config.ieeeValue (fma32 rstd (extend (input.gradient c)) (mul32 (add32 x x) alpha))
  let encoded := fun c =>
    MuddNorm.encodeWithInverse native rstd (input.mixed c)
  ⟨dx, match launch with | .withQuantization => some encoded | .gradientOnly => none⟩

def backwardInLayout := backwardUsing dotPartial

def backwardInPaddedLayout := backwardUsing dotPartialPadded

theorem backwardInPaddedLayout_eq : backwardInPaddedLayout = backwardInLayout := by
  simp only [backwardInPaddedLayout, backwardInLayout, dotPartialPadded_eq]

#axiomsClean backwardInPaddedLayout_eq

abbrev Run := MuddNorm.BackwardLayout → Native → Input → F32 → Output

def backward : Run := fun launch => backwardInLayout launch (fun c => c.val / layout launch % 32)

def localWord (launch : MuddNorm.BackwardLayout) (column : Fin 768) : Nat :=
  let width := layout launch
  column.val % width + width * (column.val / (32 * width))

theorem channel_exact (launch : MuddNorm.BackwardLayout) (column : Fin 768) :
    channel launch (column.val / layout launch % 32) (localWord launch column) = column.val := by
  cases launch
  · have h : ∀ c : Fin 768, channel .withQuantization
        (c.val / layout .withQuantization % 32) (localWord .withQuantization c) = c.val := by decide +kernel
    exact h column
  · have h : ∀ c : Fin 768, channel .gradientOnly
        (c.val / layout .gradientOnly % 32) (localWord .gradientOnly c) = c.val := by decide +kernel
    exact h column

abbrev BatchRun := (rows : Nat) → MuddNorm.Saved rows → (Fin rows → Fin 768 → BF16) →
  (Fin rows → Fin 768 → UInt8) → (Fin rows → Fin 768 → BF16) × (Fin rows → Fin 768 → UInt8)

def execute (run : Run) : BatchRun := fun _ saved gradient activation =>
  let outputs := fun row => run saved.layout saved.native
    ⟨saved.mixedResidual row, gradient row⟩ (saved.inverse row)
  (fun row => (outputs row).gradient, fun row => (outputs row).encoded.getD (activation row))

abbrev GradientRun := MuddNorm.BackwardLayout → Native → Input → F32 → Fin 768 → BF16

def executeRetained (run : GradientRun) : BatchRun := fun _ saved gradient activation =>
  let residual := fun row => run saved.layout saved.native
    ⟨saved.mixedResidual row, gradient row⟩ (saved.inverse row)
  (residual, match saved.layout with | .gradientOnly => activation | .withQuantization => saved.activation)

theorem executeRetained_eq :
    executeRetained (fun layout native input inverse => (backward layout native input inverse).gradient) =
      execute backward := by
  funext rows saved gradient activation
  obtain ⟨layout, native, mixed, inverse, encoded, exactEncoding⟩ := saved
  cases layout
  · exact Prod.ext (by rfl) exactEncoding
  · rfl

abbrev StatisticMemory := Nat → F32

def writeStatistics (program : MuddNorm.Statistic) (native : Native) (shape : MuddCoefficients.Shape)
    (base : Nat) (squares : Fin shape.rows → F32) (memory : StatisticMemory) : StatisticMemory :=
  fun address =>
    if live : base ≤ address ∧ (address - base) % 4 = 0 ∧ (address - base) / 4 < shape.rows then
      program.store native (squares ⟨(address - base) / 4, live.2.2⟩)
    else memory address

theorem statistic_store (program : MuddNorm.Statistic) (native : Native) (shape : MuddCoefficients.Shape)
    (base : Nat) (squares : Fin shape.rows → F32) (memory : StatisticMemory) (row : Fin shape.rows) :
    writeStatistics program native shape base squares memory (base + 4 * row.val) =
      program.store native (squares row) := by
  unfold writeStatistics
  have exactOffset : (base + 4 * row.val - base) / 4 = row.val := by omega
  have aligned : (base + 4 * row.val - base) % 4 = 0 := by omega
  simp [exactOffset, aligned, row.isLt]

theorem statistic_frame (program : MuddNorm.Statistic) (native : Native) (shape : MuddCoefficients.Shape)
    (base : Nat) (squares : Fin shape.rows → F32) (memory : StatisticMemory) (address : Nat)
    (outside : ¬ (base ≤ address ∧ (address - base) % 4 = 0 ∧ (address - base) / 4 < shape.rows)) :
    writeStatistics program native shape base squares memory address = memory address := by
  simp [writeStatistics, outside]

theorem cached_memory_read_eq (native : Native) (shape : MuddCoefficients.Shape)
    (base : Nat) (squares : Fin shape.rows → F32) (memory : StatisticMemory) (row : Fin shape.rows) :
    MuddNorm.Statistic.inverse.load native (writeStatistics .inverse native shape base squares memory
      (base + 4 * row.val)) =
    MuddNorm.Statistic.squareSum.load native (writeStatistics .squareSum native shape base squares memory
      (base + 4 * row.val)) := by
  simp only [statistic_store]
  rfl

#axiomsClean statistic_store
#axiomsClean channel_exact
#axiomsClean executeRetained_eq
#axiomsClean statistic_frame
#axiomsClean cached_memory_read_eq

end VeriTile.Bench.Models.NanoGPT.Track1.MuddBackward
