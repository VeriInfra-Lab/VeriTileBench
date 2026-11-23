import models.NanoGPT.Track1.ValueAdamClaimedClear

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam

open SparsePackets (Channel)

def clearRawLine (previous : PriorMarked) (row : Nat) : Line :=
  let p := previous.pull
  let idx := previous.indices
  let order := previous.order
  let own := order.localAtomics 0
  let peer := order.peerAtomics 0
  (List.finRange p.ids.count).foldl (clearEntry fun i => decide ((scaleEntry idx i).toInt = row))
    (merge p idx order row, scaleClaim p idx own peer row)

theorem clear_raw_line_zero (previous : PriorMarked) (row : Nat) :
    clearRawLine previous row = (fun _ => 0, 0) := by
  let entries : NgramAdam.Order previous.pull.ids.count := ⟨List.finRange _, List.Perm.refl _⟩
  dsimp only [clearRawLine]
  rw [show List.finRange previous.pull.ids.count = entries.slots from rfl,
    clear_scattered_line]
  apply congrArg (fun words : Channel → Half => (words, 0))
  funext col
  cases h : marked previous.pull previous.indices row
  · simp [clearMarked, h, unmarked_zero previous.pull previous.indices previous.order row col h]
  · simp [clearMarked, h]

theorem clear_raw_line_equal (previous : PriorMarked) (row : Nat) :
    clearRawLine previous row = clearIndexedLine previous row := by
  rw [clear_raw_line_zero, clear_indexed_line_exact, marked_event_clears_to_zero]

/-- The fused load widens the scattered word, multiplies in FP32 and
rounds to FP16 before the existing Adam update widens it again. -/
def scaleRead (raw : Half) : Half :=
  narrow (Triton.FP.mul Triton.FP.Config.ieeeValue (SparsePackets.halfToFP32 raw) eighth)

def updateReadScale (ar : Arithmetic) (launch : Launch) (previous : PriorMarked)
    (memory : Memory) : Fin memory.rows → Channel → Cell :=
  fun row col => update ar launch (row.val * 768 + col.val) (memory.read row col)
    (scaleRead (merge previous.pull previous.indices previous.order row.val col))

theorem read_scale_then_clear_equal (ar : Arithmetic) (launch : Launch) (memory : Memory)
    (previous : PriorMarked) (factor : launch.gradientScale = eighth) :
    (updateReadScale ar launch previous memory, clearRawLine previous) =
      ((event dense previous.pull previous.indices previous.order ar launch memory).updated,
        clearIndexedLine previous) := by
  apply Prod.ext
  · funext row col
    simp only [updateReadScale, scaleRead, event, dense, factor, halfScale]
  · funext row
    exact clear_raw_line_equal previous row

def Tile.width : Tile → Nat
  | .x512 => 512
  | .x1024 => 1024

/-- With the default shard shape, both reference schedules launch only
full tiles. Each update CTA's coordinate belongs to a unique gradient word. -/
theorem read_scale_address (tile : Tile) :
    ∀ index : Fin (25152 * 768),
      ∃! address : Fin (25152 * 768 / tile.width) × Fin tile.width,
        address.1.val * tile.width + address.2.val = index.val := by
  intro index
  have length := index.isLt
  cases tile
  · change ∃! address : Fin (25152 * 768 / 512) × Fin 512,
        address.1.val * 512 + address.2.val = index.val
    refine ⟨(⟨index.val / 512, by omega⟩, ⟨index.val % 512, by omega⟩), ?_, ?_⟩
    · change index.val / 512 * 512 + index.val % 512 = index.val
      omega
    · rintro ⟨⟨cta, ctaBound⟩, ⟨lane, laneBound⟩⟩ equal
      change cta * 512 + lane = index.val at equal
      apply Prod.ext <;> apply Fin.ext
      · change cta = index.val / 512
        omega
      · change lane = index.val % 512
        omega
  · change ∃! address : Fin (25152 * 768 / 1024) × Fin 1024,
        address.1.val * 1024 + address.2.val = index.val
    refine ⟨(⟨index.val / 1024, by omega⟩, ⟨index.val % 1024, by omega⟩), ?_, ?_⟩
    · change index.val / 1024 * 1024 + index.val % 1024 = index.val
      omega
    · rintro ⟨⟨cta, ctaBound⟩, ⟨lane, laneBound⟩⟩ equal
      change cta * 1024 + lane = index.val at equal
      apply Prod.ext <;> apply Fin.ext
      · change cta = index.val / 1024
        omega
      · change lane = index.val % 1024
        omega

#axiomsClean read_scale_address
#axiomsClean clear_raw_line_zero
#axiomsClean clear_raw_line_equal
#axiomsClean read_scale_then_clear_equal

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
