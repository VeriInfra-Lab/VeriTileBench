import models.NanoGPT.Track1.CudaStreamOrder
import models.NanoGPT.Track1.GraphSlotPTXBinding

/-! Completion order of the cycle packet and selected batch uploads,
fused staging kernel, and captured forward. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphReplayStream
open CudaStreamOrder CudaStores GraphStageMemory

inductive Work where
  | uploadWant | uploadBatch | stage | forwardReplay
  deriving DecidableEq

def commands (uploaded : Event) : List (Command Work) :=
  [.work .uploadBatch, .record uploaded, .work .stage, .work .forwardReplay]

def enqueueCommands (queues : Queues Work) (compute : Nat) (ops : List (Command Work)) : Queues Work :=
  ⟨fun stream => if stream = compute then ops else queues.pending stream,
    queues.completed⟩

def enqueue (queues : Queues Work) (compute : Nat) (uploaded : Event) : Queues Work :=
  enqueueCommands queues compute (commands uploaded)

def cycleCommands (wantUploaded batchUploaded : Event) : List (Command Work) :=
  [.work .uploadBatch, .record batchUploaded, .work .uploadWant, .record wantUploaded, .work .stage, .work .forwardReplay]

def enqueueCycle (queues : Queues Work) (compute : Nat) (wantUploaded batchUploaded : Event) : Queues Work :=
  enqueueCommands queues compute (cycleCommands wantUploaded batchUploaded)

structure Replay (s : GraphStage.Shape) where
  memory : Memory
  forwards : List (GraphStage.Observation s)

def executeReplay {s n} (layout : Layout s n) (writeWant : Memory → Memory) (stage : Memory → Option Memory)
    (batch : FineWeb.TokenBatch) (transfer : BatchTransfer) :
    List (Command Work) → Replay s → Option (Replay s)
  | [], state => some state
  | op :: tail, state => do
    let next ← match op with
      | .work .uploadWant => some {state with memory := writeWant state.memory}
      | .work .uploadBatch => some {state with memory := transfer.upload layout batch state.memory}
      | .work .stage => (stage state.memory).map (fun memory => {state with memory})
      | .work .forwardReplay => some {state with forwards := observe layout state.memory :: state.forwards}
      | .record _ | .wait _ => some state
    executeReplay layout writeWant stage batch transfer tail next

def execute {s n} (layout : Layout s n) (ptx : String)
    (batch : FineWeb.TokenBatch) (transfer : BatchTransfer) (ctas : List Nat) :
    List (Command Work) → Replay s → Option (Replay s) :=
  executeReplay layout id (GraphSlotPTXBinding.interpretLaunch s ptx layout ctas) batch transfer

theorem replay_ordered {s n} (layout : Layout s n) (ptx : String)
    (batch : FineWeb.TokenBatch) (transfer : BatchTransfer) (ctas : List Nat) (prior : Memory)
    (queues final : Queues Work) (compute : Nat) (uploaded : Event) (issued : List (Nat × Command Work))
    (run : Run (enqueue queues compute uploaded) issued final) (complete : final.pending compute = []) :
    execute layout ptx batch transfer ctas (issuedOn compute issued) ⟨prior, []⟩ =
      (GraphSlotPTXBinding.interpretLaunch s ptx layout ctas (transfer.upload layout batch prior)).map
        (fun memory => ⟨memory, [observe layout memory]⟩) := by
  rw [fifo_complete run compute complete]
  simp [enqueue, enqueueCommands, commands, execute, executeReplay]
  cases GraphSlotPTXBinding.interpretLaunch s ptx layout ctas (transfer.upload layout batch prior) <;> rfl

theorem replay_cycle_ordered {s n} (layout : Layout s n) (writeWant : Memory → Memory)
    (stage : Memory → Option Memory) (batch : FineWeb.TokenBatch) (transfer : BatchTransfer) (prior : Memory)
    (queues final : Queues Work) (compute : Nat) (wantUploaded batchUploaded : Event)
    (issued : List (Nat × Command Work))
    (run : Run (enqueueCycle queues compute wantUploaded batchUploaded) issued final)
    (complete : final.pending compute = []) :
    executeReplay layout writeWant stage batch transfer (issuedOn compute issued) ⟨prior, []⟩ =
      (stage (writeWant (transfer.upload layout batch prior))).map
        (fun memory => ⟨memory, [observe layout memory]⟩) := by
  rw [fifo_complete run compute complete]
  simp [enqueueCycle, enqueueCommands, cycleCommands, executeReplay]
  cases stage (writeWant (transfer.upload layout batch prior)) <;> rfl

private theorem recorded_of_pending {initial final : Queues Work} {issued : List (Nat × Command Work)}
    (run : Run initial issued final) (compute : Nat) (event : Event) (complete : final.pending compute = [])
    (member : Command.record event ∈ initial.pending compute) : event ∈ final.completed := by
  have order := fifo_complete run compute complete
  have recorded := (issued_on_member compute (.record event) issued).mp (order.symm ▸ member)
  rw [completed_events run]
  exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨(compute, .record event), recorded, by simp [Command.recorded]⟩)

theorem upload_recorded (queues final : Queues Work) (compute : Nat) (uploaded : Event)
    (issued : List (Nat × Command Work))
    (run : Run (enqueue queues compute uploaded) issued final) (complete : final.pending compute = []) :
    uploaded ∈ final.completed := by
  exact recorded_of_pending run compute uploaded complete (by simp [enqueue, enqueueCommands, commands])

theorem cycle_uploads_recorded (queues final : Queues Work) (compute : Nat)
    (wantUploaded batchUploaded : Event) (issued : List (Nat × Command Work))
    (run : Run (enqueueCycle queues compute wantUploaded batchUploaded) issued final)
    (complete : final.pending compute = []) : wantUploaded ∈ final.completed ∧ batchUploaded ∈ final.completed := by
  constructor <;> apply recorded_of_pending run compute _ complete <;>
    simp [enqueueCycle, enqueueCommands, cycleCommands]

/-- A host synchronization on the latest recording may return before the
forward completes. That recording already certifies the H2D read of the
pinned slot, which is the condition needed to refill the slot. -/
theorem reuse_after_upload (queues final : Queues Work) (compute : Nat) (uploaded : Event)
    (issued : List (Nat × Command Work))
    (run : Run (enqueue queues compute uploaded) issued final)
    (generation : uploaded ∉ queues.completed)
    (owner : ∀ other ≠ compute, Command.record uploaded ∉ queues.pending other)
    (synchronized : uploaded ∈ final.completed) :
    ∃ remaining, issuedOn compute issued = .work .uploadBatch :: .record uploaded :: remaining := by
  have completed := completed_events run
  rw [completed, List.mem_append] at synchronized
  obtain ⟨⟨stream, cmd⟩, present, recorded⟩ := List.mem_flatMap.mp (synchronized.resolve_left generation)
  have same : cmd = .record uploaded := by
    cases cmd <;> simp [Command.recorded] at recorded ⊢
    exact recorded.symm
  subst cmd
  have earlier := (issued_on_member stream (.record uploaded) issued).mpr present
  have onCompute : stream = compute := by
    by_cases same : stream = compute
    · exact same
    · have localOrder := fifo_projection run stream
      have member : Command.record uploaded ∈ (enqueue queues compute uploaded).pending stream := by
        rw [← localOrder]
        exact List.mem_append_left _ earlier
      exact False.elim (owner stream same (by simpa [enqueue, enqueueCommands, same] using member))
  subst stream
  have order := fifo_projection run compute
  simp only [enqueue, enqueueCommands, commands, if_true] at order
  cases seen : issuedOn compute issued with
  | nil => simp [seen] at earlier
  | cons first rest =>
    rw [seen] at order
    have compared := List.cons.inj order
    cases rest with
    | nil => simp [seen, compared.1] at earlier
    | cons second rest =>
      have secondEq := (List.cons.inj compared.2).1
      exact ⟨rest, by rw [compared.1, secondEq]⟩

#axiomsClean replay_ordered
#axiomsClean replay_cycle_ordered
#axiomsClean upload_recorded
#axiomsClean cycle_uploads_recorded
#axiomsClean reuse_after_upload

end VeriTile.Bench.Models.NanoGPT.Track1.GraphReplayStream
