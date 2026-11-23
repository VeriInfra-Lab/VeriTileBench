import models.NanoGPT.Track1.ValueAdamNativeCombinedCertificate
import models.NanoGPT.Track1.ValueAdamNativeShard

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam

def Tile.laneWidth : Tile → Nat
  | .x512 => 2
  | .x1024 => 8

namespace CombinedDivisor

open VeriTile.Triton FP
open Elementwise (Inputs Period)
open NativeCombinedTrace (Certificate RootOps)
open DivisorBand (ZeroMode)

private def stepMul : F32 → F32 → F32 := FP.mul Config.ieeeValue

def momentsAt (input : Inputs) (period : Period) (index : Nat) : Moments :=
  let launch := period.launch 0 input.step input.decay
  let old := input.cell index
  let g := SparsePackets.halfToFP32 (input.gradientWord true index)
  let fp := Config.ieeeValue
  (FP.fma fp old.first launch.beta1 (FP.mul fp g launch.oneMinusBeta1),
   FP.fma fp old.second launch.beta2 (FP.mul fp (FP.mul fp g g) launch.oneMinusBeta2))

def nativeArithmetic (tile : Launch → Tile) (reciprocal : F32 → F32)
    (nanProduct : Bool → F32 → F32 → F32) : Arithmetic :=
  ⟨tile, DivisorBand.referenceDivide reciprocal nanProduct⟩

private def flat (tile : Tile) (warp lane slot : Nat) : Nat :=
  (32 * tile.laneWidth) * warp + tile.laneWidth * lane + slot

private theorem flat_index (tile : Tile) (i : Nat) :
    flat tile (i / (32 * tile.laneWidth)) (i % (32 * tile.laneWidth) / tile.laneWidth)
      (i % tile.laneWidth) = i := by
  cases tile
  · have nested : i % 64 % 2 = i % 2 := Nat.mod_mod_of_dvd _ (by decide)
    change 64 * (i / 64) + 2 * (i % 64 / 2) + i % 2 = i
    omega
  · have nested : i % 256 % 8 = i % 8 := Nat.mod_mod_of_dvd _ (by decide)
    change 256 * (i / 256) + 8 * (i % 256 / 8) + i % 8 = i
    omega

private def gradientHasNaN (tile : Tile) (input : Inputs) (warp : Nat) : Bool :=
  (List.range 32).any fun lane =>
    (List.range tile.laneWidth).any fun slot =>
      Elementwise.halfNaN (input.gradient (flat tile warp lane slot))

private def nativeCell (ops : RootOps) (reciprocal : F32 → F32)
    (nanProduct : Bool → F32 → F32 → F32) (input : Inputs) (period : Period) (index : Nat) : Cell :=
  let point := momentsAt input period index
  let direction := DivisorBand.referenceDivide reciprocal nanProduct point.1 (ops.denominator point.2)
  let weight := NativeColdTrace.storedWeightWord (input.cell index).weight (stepMul direction input.step) input.decay
  ⟨weight, point.1, point.2⟩

private def updateTile {oldPTX newPTX oldSASS newSASS mode masked} (period : Period) (tile : Tile)
    (certificate : Certificate oldPTX newPTX oldSASS newSASS mode (if period == .two then 2 else 4) masked tile.laneWidth)
    (original : Bool) (input : Inputs) (ops : RootOps)
    (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (index : Fin tile.width) : Option Cell :=
  let warp := index.val / (32 * tile.laneWidth)
  let lane := index.val % (32 * tile.laneWidth) / tile.laneWidth
  let slot : Fin tile.laneWidth := ⟨index.val % tile.laneWidth,
    Nat.mod_lt _ (by cases tile <;> decide)⟩
  let points := fun thread j => momentsAt input period (flat tile warp thread j)
  let old := fun j => (input.cell (flat tile warp lane j)).weight
  let grad := masked && tile.laneWidth == 8 && gradientHasNaN tile input warp
  let weights := if original then
      certificate.referenceWeightAt ops reciprocal nanProduct (points lane) grad stepMul input.step input.decay old slot
    else
      let group := (List.finRange 32).map (fun thread => (List.range tile.laneWidth).map (points thread.val))
      certificate.nativeWeightAt ops reciprocal nanProduct group (points lane) grad stepMul input.step input.decay old slot
  weights.map (fun weight =>
    let point := momentsAt input period index.val
    ⟨weight, point.1, point.2⟩)

private theorem update_tile_exact {oldPTX newPTX oldSASS newSASS mode masked} (period : Period) (tile : Tile)
    (certificate : Certificate oldPTX newPTX oldSASS newSASS mode (if period == .two then 2 else 4) masked tile.laneWidth)
    (original : Bool) (input : Inputs) (ops : RootOps)
    (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (index : Fin tile.width) :
    updateTile period tile certificate original input ops reciprocal nanProduct index =
      some (nativeCell ops reciprocal nanProduct input period index.val) := by
  dsimp only [updateTile]
  let thread : Fin 32 := ⟨index.val % (32 * tile.laneWidth) / tile.laneWidth, by
    cases tile
    · change index.val % 64 / 2 < 32
      omega
    · change index.val % 256 / 8 < 32
      omega⟩
  cases original
  · simp only [Bool.false_eq_true, if_false]
    rw [certificate.native_weight_at_exact]
    · rw [flat_index]
      rfl
    · exact List.mem_map.mpr ⟨thread, List.mem_finRange thread, rfl⟩
  · simp only [if_true]
    rw [certificate.reference_weight_at_exact, flat_index]
    rfl

def runShard {oldPTX newPTX oldSASS newSASS mode masked} (period : Period) (tile : Tile)
    (certificate : Certificate oldPTX newPTX oldSASS newSASS mode (if period == .two then 2 else 4) masked tile.laneWidth)
    (original : Bool) (input : Inputs) (ops : RootOps)
    (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (index : Fin (25152 * 768)) : Option Cell :=
  updateTile period tile certificate original (input.atCTA tile (index.val / tile.width))
    ops reciprocal nanProduct ⟨index.val % tile.width, Nat.mod_lt _ (by cases tile <;> decide)⟩

theorem run_shard_exact {oldPTX newPTX oldSASS newSASS mode masked} (period : Period) (tile : Tile)
    (certificate : Certificate oldPTX newPTX oldSASS newSASS mode (if period == .two then 2 else 4) masked tile.laneWidth)
    (original : Bool) (input : Inputs) (ops : RootOps)
    (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (index : Fin (25152 * 768)) :
    runShard period tile certificate original input ops reciprocal nanProduct index =
      some (nativeCell ops reciprocal nanProduct input period index.val) := by
  unfold runShard
  rw [update_tile_exact]
  have flatten : index.val / tile.width * tile.width + index.val % tile.width = index.val := by
    rw [Nat.mul_comm, Nat.div_add_mod]
  simp only [nativeCell, momentsAt, Inputs.atCTA, Inputs.gradientWord, if_true, flatten]

#axiomsClean run_shard_exact

end CombinedDivisor
end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
