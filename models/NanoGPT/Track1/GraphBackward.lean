import models.NanoGPT.Track1.MuddFullPacking
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import models.NanoGPT.Track1.ValueEmbedAccum
import models.NanoGPT.Track1.CETranspose
import models.NanoGPT.Track1.TrainingHead
import models.NanoGPT.Track1.MuddGradientGrouping
import models.NanoGPT.Track1.MuddFusedPacking

/-! Gradient installation after the captured autograd replay. Buffer addresses
record the aliases observed by the optimizer and the n-gram pending list. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphBackward
open VeriTile.Triton FP

structure Tensor (format : Format) where
  address : Nat
  size : Nat
  words : Nat → Value format

structure AdamGrad where
  parameter : Nat
  format : Format
  replayed : Tensor format
  accumulator : Tensor format
  hasGrad : Bool

structure Captured where
  anvil : List (Nat × Nat)
  adam : List AdamGrad
  ngram : Tensor .bf16
  valueGrad : Option ValueEmbedAccum.Snapshot := none

def storageAdd (cfg : Config) : (f : Format) → Value f → Value f → Value f
  | .fp32, old, next => FP.add cfg old next
  | .bf16, old, next =>
    cast .bf16 cfg (FP.add cfg (cast .fp32 cfg old) (cast .fp32 cfg next))

/-- The destination-first foreach sum uses FP32 opmath and writes in the
persistent buffer's storage precision. The first contribution is copied. -/
def AdamGrad.update (cfg : Config) (a : AdamGrad) (i : Nat) : Value a.format :=
  if a.hasGrad then
    storageAdd cfg a.format (a.accumulator.words i) (a.replayed.words i)
  else a.replayed.words i

def AdamGrad.groups (adam : List AdamGrad) : List AdamGrad × List AdamGrad :=
  (adam.filter fun a => !a.hasGrad, adam.filter (·.hasGrad))

theorem first_copy (cfg : Config) (a : AdamGrad) (first : a.hasGrad = false) (i : Nat) :
    a.update cfg i = a.replayed.words i := by simp [AdamGrad.update, first]

theorem later_add (cfg : Config) (a : AdamGrad) (later : a.hasGrad = true) (i : Nat) :
    a.update cfg i = storageAdd cfg a.format (a.accumulator.words i) (a.replayed.words i) := by
  simp [AdamGrad.update, later]

def ringSize := 4

structure Pending where
  gradient : Tensor .bf16
  count : Nat
  ring : Fin ringSize → Nat

inductive Held where
  | direct (gradient : Tensor .bf16)
  | copied (slot : Fin ringSize) (gradient : Tensor .bf16)

def Pending.hold (p : Pending) (updateEvent : Bool) : Option Held :=
  if updateEvent then some (.direct p.gradient)
  else if h : p.count < ringSize then
    let slot : Fin ringSize := ⟨p.count, h⟩
    some (.copied slot {p.gradient with address := p.ring slot})
  else none

theorem hold_event (p : Pending) : p.hold true = some (.direct p.gradient) := by
  simp [Pending.hold]

theorem hold_pending (p : Pending) (room : p.count < ringSize) :
    p.hold false = some (.copied ⟨p.count, room⟩ {p.gradient with address := p.ring ⟨p.count, room⟩}) := by
  simp [Pending.hold, room]

inductive MuddGradientPacket where
  | columns (rows planes : Nat) (input : MuddGradientGrouping.Packet rows planes)
  | packing (rows : Nat) (input : MuddGradientPacking.Packet rows)
  | fusedPacking (rows : Nat) (input : MuddFusedPacking.Inputs rows)
  | fullPacking (rows : Nat) (input : MuddFusedPacking.Inputs rows)
      (projection : MuddFullPacking.ScaledProjection input.group)

structure MuddGradientRuntime (State : Type) where
  prepare : (tape : TrainingHead.Tape) → TrainingHead.Gradients tape.tokens →
    State → Option (MuddGradientPacket × State)
  resume : MuddGradientPacket → MuddGradientGrouping.Memory → State → Option (Captured × State)

structure HeadRuntime (State : Type) where
  prepare : (tape : TrainingHead.Tape) → State → Option (TrainingHead.BackwardLayout tape × State)
  /-- Continue backpropagation from the hidden-state gradient and install the head gradient. -/
  resume : (tape : TrainingHead.Tape) → TrainingHead.Gradients tape.tokens →
    State → Option (Captured × State)
  muddGradients : Option (MuddGradientRuntime State) := none

def HeadRuntime.finish {State : Type} (runtime : HeadRuntime State) (reduce : MuddGradientGrouping.Run)
    (packing : MuddGradientPacking.Run) (fused : MuddFusedPacking.Run)
    (fullPacking : MuddFullPacking.Run)
    (tape : TrainingHead.Tape) (gradients : TrainingHead.Gradients tape.tokens) (state : State) :
    Option (Captured × State) := do
  let some mudd := runtime.muddGradients | runtime.resume tape gradients state
  let (packet, next) ← mudd.prepare tape gradients state
  let written := match packet with
    | .columns rows planes input => reduce rows planes input
    | .packing rows input => packing rows input
    | .fusedPacking rows input => fused rows input
    | .fullPacking rows input projection => fullPacking rows input projection
  mudd.resume packet written next

structure Runtime (State : Type) where
  /-- The captured head backward prepares both weight-gradient GEMM operands using this port. -/
  replay : CETranspose.Dispatch → State → Option (Captured × State)
  writeValueGrad : Nat → SparsePackets.FP16Rows → State → Option State
  linkGradient : Nat → Nat → State → Option State
  writeAdam : (format : Format) → Tensor format → State → Option State
  publishSink : Tensor .bf16 → State → Option State
  readPending : State → Option (Pending × State)
  writeRing : Fin ringSize → Tensor .bf16 → State → Option State
  appendPending : Tensor .bf16 → State → Option State
  head : Option (HeadRuntime State) := none

def replayBackward {State : Type} (r : Runtime State) (ceOperands : CETranspose.Dispatch)
    (head : TrainingHead.Backward) (saved : Option TrainingHead.Tape) (state : State)
    (reduce : MuddGradientGrouping.Run := MuddGradientGrouping.Program.separate.run)
    (packing : MuddGradientPacking.Run := MuddGradientPacking.Program.interleaved.run)
    (fused : MuddFusedPacking.Run := MuddFusedPacking.Program.separate.run)
    (fullPacking : MuddFullPacking.Run := MuddFullPacking.reference) :
    Option (Captured × State) := do
  let some tape := saved | r.replay ceOperands state
  let runtime ← r.head
  let (layout, staged) ← runtime.prepare tape state
  runtime.finish reduce packing fused fullPacking tape (layout.run head) staged

theorem replayBackward_tape {State : Type} (r : Runtime State) (ceOperands : CETranspose.Dispatch)
    (head : TrainingHead.Backward) (tape : TrainingHead.Tape) (state staged : State)
    (runtime : HeadRuntime State) (active : r.head = some runtime)
    (layout : TrainingHead.BackwardLayout tape) (prepared : runtime.prepare tape state = some (layout, staged))
    (reduce : MuddGradientGrouping.Run) (packing : MuddGradientPacking.Run) (fused : MuddFusedPacking.Run)
    (fullPacking : MuddFullPacking.Run) :
    replayBackward r ceOperands head (some tape) state reduce packing fused fullPacking =
      runtime.finish reduce packing fused fullPacking tape (layout.run head) staged := by
  simp only [replayBackward, active, prepared, bind, Option.bind_some]

def execute {State : Type} (r : Runtime State) (valueAccum : ValueEmbedAccum.Accumulate)
    (ceOperands : CETranspose.Dispatch) (head : TrainingHead.Backward) (saved : Option TrainingHead.Tape)
    (state : State) (reduce : MuddGradientGrouping.Run := MuddGradientGrouping.Program.separate.run)
    (packing : MuddGradientPacking.Run := MuddGradientPacking.Program.interleaved.run)
    (fused : MuddFusedPacking.Run := MuddFusedPacking.Program.separate.run)
    (fullPacking : MuddFullPacking.Run := MuddFullPacking.reference) : Option State := do
  let (captured, replayed) ← replayBackward r ceOperands head saved state reduce packing fused fullPacking
  let replayed ← match captured.valueGrad with
    | none => some replayed
    | some value => r.writeValueGrad value.address (valueAccum value) replayed
  let state ← captured.anvil.foldlM (fun s (parameter, address) => r.linkGradient parameter address s) replayed
  let state ← captured.adam.foldlM (fun s a => r.linkGradient a.parameter a.accumulator.address s) state
  let (first, accumulate) := AdamGrad.groups captured.adam
  let write := fun s (a : AdamGrad) =>
    r.writeAdam a.format {a.accumulator with words := a.update Config.ieeeValue} s
  let state ← first.foldlM write state
  let state ← accumulate.foldlM write state
  r.publishSink captured.ngram state

def holdAndAppend {State : Type} (r : Runtime State) (updateEvent : Bool) (state : State) : Option State := do
  let (pending, state) ← r.readPending state
  match ← pending.hold updateEvent with
  | .direct gradient => r.appendPending gradient state
  | .copied slot gradient =>
    let state ← r.writeRing slot gradient state
    r.appendPending gradient state

#axiomsClean first_copy
#axiomsClean later_add
#axiomsClean hold_event
#axiomsClean hold_pending
#axiomsClean replayBackward_tape
end VeriTile.Bench.Models.NanoGPT.Track1.GraphBackward
