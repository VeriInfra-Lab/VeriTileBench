import models.NanoGPT.Track1.ValueAdamNativePTX

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

theorem Tile.index_or (tile : Tile) (cta lane : Nat) (inTile : lane < tile.width) :
    (cta * tile.width) ||| lane = cta * tile.width + lane := by
  cases tile
  · have low : lane < 2^9 := inTile
    simpa only [Tile.width, Nat.shiftLeft_eq] using
      (Nat.shiftLeft_add_eq_or_of_lt low cta).symm
  · have low : lane < 2^10 := inTile
    simpa only [Tile.width, Nat.shiftLeft_eq] using
      (Nat.shiftLeft_add_eq_or_of_lt low cta).symm

def Inputs.atCTA (inputs : Inputs) (tile : Tile) (cta : Nat) : Inputs :=
  { inputs with cell := fun lane => inputs.cell (cta * tile.width + lane)
                gradient := fun lane => inputs.gradient (cta * tile.width + lane) }

def Inputs.scaleGradient (inputs : Inputs) : Inputs :=
  {inputs with gradient := fun index => scaleRead (inputs.gradient index)}

private theorem Tile.width_pos (tile : Tile) : 0 < tile.width := by
  cases tile <;> decide

def Certificate.updateShard {profile source} (checked : Certificate profile source)
    (inputs : Inputs) (index : Fin (25152 * 768)) : Option Cell :=
  checked.update (inputs.atCTA profile.tile (index.val / profile.tile.width))
    (index.val % profile.tile.width)

theorem Certificate.update_shard_eq {profile source} (checked : Certificate profile source)
    (inputs : Inputs) (index : Fin (25152 * 768)) :
    checked.updateShard inputs index =
      some (ValueAdam.update ⟨fun _ => profile.tile, inputs.divideFull⟩
        (profile.period.launch 0 inputs.step inputs.decay) index.val
        (inputs.cell index.val) (inputs.gradientWord profile.fused index.val)) := by
  let j := index.val
  let tile := profile.tile
  let lane : Fin tile.width := ⟨j % tile.width, Nat.mod_lt _ (Tile.width_pos tile)⟩
  let cta := j / tile.width
  have flat : cta * tile.width + lane.val = index.val := by
    simpa only [Nat.mul_comm cta tile.width] using Nat.div_add_mod j tile.width
  have cell : (inputs.atCTA tile cta).cell lane.val = inputs.cell index.val :=
    congrArg inputs.cell flat
  have gradient : (inputs.atCTA tile cta).gradientWord profile.fused lane.val =
      inputs.gradientWord profile.fused index.val := by
    simp only [Inputs.gradientWord, Inputs.atCTA, flat]
  change checked.update (inputs.atCTA tile cta) lane = _
  rw [checked.update_eq _ lane, expected_exact _ _ _ 0, cell, gradient]
  simp only [ValueAdam.update, Inputs.atCTA] <;> rfl

theorem native_read_scale_grid (period : Period) (tile : Tile) (reference fused : String)
    (r : Certificate ⟨period, tile, false⟩ reference) (q : Certificate ⟨period, tile, true⟩ fused)
    (input : Inputs) :
    q.updateShard input = r.updateShard input.scaleGradient := by
  funext index
  rw [q.update_shard_eq, r.update_shard_eq]
  rfl

def Inputs.forPull (inputs : Inputs) (previous : PriorMarked) : Inputs :=
  let grad (i : Nat) := merge previous.pull previous.indices previous.order (i / 768)
    ⟨i % 768, Nat.mod_lt _ (by decide)⟩
  {inputs with gradient := grad}

theorem native_read_scale_event (period : Period) (tile : Tile) (reference fused : String)
    (r : Certificate ⟨period, tile, false⟩ reference) (q : Certificate ⟨period, tile, true⟩ fused)
    (input : Inputs) (previous : PriorMarked) :
    (q.updateShard (input.forPull previous), clearRawLine previous) =
      (r.updateShard (input.forPull previous).scaleGradient, clearIndexedLine previous) := by
  refine Prod.ext (native_read_scale_grid period tile reference fused r q _) ?_
  funext row
  exact clear_raw_line_equal previous row

def Certificate.gradientAfterShard {profile source} (checked : Certificate profile source)
    (input : Nat → Half) (index : Fin (25152 * 768)) : Half :=
  let tile := profile.tile
  let block := index.val / tile.width
  checked.gradientAfter (fun lane => input (block * tile.width + lane)) (index.val % tile.width)

theorem Certificate.gradient_after_shard_zero {profile source} (checked : Certificate profile source)
    (used : checked.consumes = true) (input : Nat → Half) (index : Fin (25152 * 768)) :
    checked.gradientAfterShard input index = 0 := by
  let lane : Fin profile.tile.width := ⟨index.val % profile.tile.width, Nat.mod_lt _ (Tile.width_pos _)⟩
  change checked.gradientAfter (fun j => input (index.val / profile.tile.width * profile.tile.width + j))
    lane.val = 0
  exact checked.gradient_after_zero used _ lane

theorem native_read_consume_event (period : Period) (tile : Tile) (reference fused : String)
    (r : Certificate ⟨period, tile, false⟩ reference) (q : Certificate ⟨period, tile, true⟩ fused)
    (used : q.consumes = true) (input : Inputs) (previous : PriorMarked) :
    (q.updateShard (input.forPull previous), q.gradientAfterShard (input.forPull previous).gradient) =
      (r.updateShard (input.forPull previous).scaleGradient, referenceWorkspace previous) := by
  refine Prod.ext (native_read_scale_grid period tile reference fused r q _) ?_
  funext index
  change q.gradientAfterShard (input.forPull previous).gradient index = referenceWorkspace previous index
  rw [q.gradient_after_shard_zero used]
  unfold referenceWorkspace
  rw [← clear_raw_line_equal, clear_raw_line_zero]

#axiomsClean Certificate.gradient_after_shard_zero
#axiomsClean native_read_consume_event
#axiomsClean Tile.index_or
#axiomsClean Certificate.update_shard_eq
#axiomsClean native_read_scale_grid
#axiomsClean native_read_scale_event

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
