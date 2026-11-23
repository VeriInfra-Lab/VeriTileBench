import models.NanoGPT.Track1.AnvilEpilogue
import models.NanoGPT.Track1.SourceAnchors

/-! The reference writes FP32 rails and reads them back for the BF16 Gram
input. Each fused thread keeps those same two FP32 words live until the
lookahead is rounded, then publishes both rail updates and the BF16 word. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilRails
open VeriTile.Triton FP
open AnvilEpilogue (Bank)
open TailNorm (F32 BF16 one widen narrow)

def cells (bank : Bank) : Nat :=
  bank.dimensions.1 * bank.dimensions.2.1 * bank.dimensions.2.2

def blockSize : Nat := 1024

theorem full_tiles (bank : Bank) : cells bank / blockSize * blockSize = cells bank := by
  cases bank <;> decide

structure Lane (bank : Bank) where
  block : Fin (cells bank / blockSize)
  lane : Fin blockSize

def laneIndex {bank : Bank} (p : Lane bank) : Fin (cells bank) :=
  ⟨p.block.val * blockSize + p.lane.val, by
    have hb := p.block.isLt
    have hl := p.lane.isLt
    have hc := full_tiles bank
    simp only [blockSize] at *
    omega⟩

def owner {bank : Bank} (i : Fin (cells bank)) : Lane bank where
  block := ⟨i.val / blockSize, by
    have hi := i.isLt
    have hc := full_tiles bank
    simp only [blockSize] at *
    omega⟩
  lane := ⟨i.val % blockSize, by simp [blockSize, Nat.mod_lt, Nat.zero_lt_succ]⟩

@[simp] theorem lane_owner {bank : Bank} (i : Fin (cells bank)) : laneIndex (owner i) = i := by
  apply Fin.ext
  simp only [laneIndex, owner, blockSize]
  omega

theorem owner_lane {bank : Bank} (p : Lane bank) : owner (laneIndex p) = p := by
  rcases p with ⟨⟨block, hb⟩, ⟨lane, hl⟩⟩
  have hLane : lane < 1024 := hl
  simp only [owner, laneIndex, Lane.mk.injEq, Fin.mk.injEq, blockSize]
  omega

def railOffset {bank : Bank} (slow : Bool) (i : Fin (cells bank)) : Nat :=
  (if slow then cells bank else 0) + i.val

theorem rail_word_owner {bank : Bank} (s t : Bool) (i j : Fin (cells bank))
    (h : railOffset s i = railOffset t j) : s = t ∧ i = j := by
  have hi := i.isLt
  have hj := j.isLt
  cases s <;> cases t <;> simp [railOffset] at h ⊢ <;> omega

structure Input (bank : Bank) where
  eligible : Bool
  grad : Fin (cells bank) → BF16
  fast : Fin (cells bank) → F32
  slow : Fin (cells bank) → F32
  momentum : F32
  fastBeta : F32
  fastWeight : F32

structure Output (bank : Bank) where
  fast : Fin (cells bank) → F32
  slow : Fin (cells bank) → F32
  prepared : Fin (cells bank) → BF16

private def cfg := Config.ieeeValue
private def minusOne := Value.ofBits .fp32 0xbf800000
private def half := Value.ofBits .fp32 0x3f000000

/-- The native predicate is ordered, including the branch on an exact half. -/
def orderedLerp (start finish weight : F32) : F32 :=
  let magnitude := Value.abs weight
  let useFinish := FP.lt cfg half magnitude || FP.eq cfg half magnitude
  let scale := if useFinish then FP.add cfg weight minusOne else weight
  let base := if useFinish then finish else start
  FP.fma cfg scale (FP.sub cfg finish start) base

def fastUpdate {bank : Bank} (input : Input bank) (i : Fin (cells bank)) : F32 :=
  orderedLerp (input.fast i) (widen (input.grad i)) (FP.sub cfg one input.fastBeta)

def slowUpdate {bank : Bank} (input : Input bank) (i : Fin (cells bank)) : F32 :=
  let old := input.slow i
  let rate := Value.ofBits .fp32 Driver.anvilSlowWeightBits
  FP.fma cfg rate (FP.sub cfg (widen (input.grad i)) old) old

def prepareGrad {bank : Bank} (input : Input bank) (i : Fin (cells bank))
    (fast slow : F32) : BF16 :=
  let slowPart := FP.mul cfg (FP.sub cfg one input.fastWeight) slow
  let blend := FP.fma cfg input.fastWeight fast slowPart
  narrow (orderedLerp (widen (input.grad i)) blend input.momentum)

abbrev Dispatch := (bank : Bank) → Input bank → Output bank

/-- The lookahead reads the two FP32 buffers produced by the first launch. -/
def reference : Dispatch := fun _ input =>
  let savedFast := fastUpdate input
  let savedSlow := slowUpdate input
  ⟨savedFast, savedSlow, fun i => prepareGrad input i (savedFast i) (savedSlow i)⟩

/-- Three stores by one thread, with the two FP32 results still in registers. -/
def fusedLane {bank : Bank} (input : Input bank) (lane : Lane bank) : F32 × F32 × BF16 :=
  let i := laneIndex lane
  let fast := fastUpdate input i
  let slow := slowUpdate input i
  ⟨fast, slow, prepareGrad input i fast slow⟩

def fused : Dispatch := fun _ input =>
  let output := fun i => fusedLane input (owner i)
  ⟨fun i => (output i).1, fun i => (output i).2.1, fun i => (output i).2.2⟩

theorem fused_equal : fused = reference := by
  funext bank input
  simp only [fused, reference, fusedLane, lane_owner]

def optimized : Dispatch := fun bank input =>
  if input.eligible then fused bank input else reference bank input

theorem dispatch_equal : optimized = reference := by
  funext bank input
  simp [optimized, fused_equal]

#axiomsClean full_tiles
#axiomsClean lane_owner
#axiomsClean owner_lane
#axiomsClean rail_word_owner
#axiomsClean dispatch_equal

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilRails
