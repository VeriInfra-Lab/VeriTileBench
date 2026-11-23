import models.NanoGPT.Track1.ValueAdamElementwise
import models.NanoGPT.Track1.ValueAdamHalfProduct
import models.NanoGPT.Track1.ValueAdamHalfProductVote

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise.GuardedHalf

/-- Observable halfwords delivered by the scoped four-pair PTX fragment. -/
def readWarp (warp : Fin 32 → Lane Half) (lane : Fin 32) : List Half :=
  let words := (List.finRange 32).flatMap (Lane.words ∘ warp)
  if warpVote halfNaN warp lane then words.map scaleRead else words.map finiteHalfProduct

theorem read_warp_exact (warp : Fin 32 → Lane Half) (lane : Fin 32) :
    readWarp warp lane = ((List.finRange 32).flatMap (Lane.words ∘ warp)).map scaleRead := by
  unfold readWarp
  rw [warp_vote_exact]
  exact guarded_scale_exact _

def scalarTerm (index : Nat) : Term :=
  .narrowFP16 (.mul (.widenFP16 (.input .gradient index)) (.constant 0x3e000000))

theorem scalar_term_exact (inputs : Inputs) (index : Nat) :
    (scalarTerm index).eval inputs = some (.half (scaleRead (inputs.gradient index))) := rfl

/-- Connecting the exact guarded instruction sequence to the terms that the
native Adam decoder installs in the receiving registers. -/
theorem read_warp_terms_exact (inputs : Inputs) (slots : Fin 32 → Lane Nat) (lane : Fin 32) :
    (readWarp (fun i => (slots i).map inputs.gradient) lane).map (some ∘ Data.half) =
      ((List.finRange 32).flatMap (Lane.words ∘ slots)).map
        (fun index => (scalarTerm index).eval inputs) := by
  rw [read_warp_exact]
  have words_eq :
      (List.finRange 32).flatMap (fun i => ((slots i).map inputs.gradient).words) =
        ((List.finRange 32).flatMap (Lane.words ∘ slots)).map inputs.gradient := by
    simp only [Lane.words_map, List.map_flatMap, Function.comp_apply]
  change (((List.finRange 32).flatMap (fun i => ((slots i).map inputs.gradient).words)).map
    scaleRead).map (some ∘ Data.half) = _
  rw [words_eq]
  simp only [List.map_map, scalar_term_exact, Function.comp_def]

#axiomsClean read_warp_exact
#axiomsClean read_warp_terms_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise.GuardedHalf
