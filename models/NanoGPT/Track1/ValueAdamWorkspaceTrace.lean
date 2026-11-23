import models.NanoGPT.Track1.ValueAdamNativeEvent

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

structure WorkspaceState where
  history : History
  bytes : Workspace
  valid : history.represents bytes

def WorkspaceState.materialize (state : WorkspaceState) (next : PriorMarked)
    (sparse : Bool) (scale : F32) : WorkspaceState :=
  if sparse then
    if next.pull.ids.count = 0 then
      ⟨.clean, state.bytes.beforeScatter state.history,
        sparse_preparation_zero state.bytes state.history state.valid⟩
    else ⟨.indexed next, Workspace.indexed next, rfl⟩
  else
    ⟨.shared, ⟨dense next.pull next.indices next.order scale next.rows, state.bytes.markers⟩, trivial⟩

structure NativeCall (period : Period) where
  next : PriorMarked
  arithmetic : Arithmetic
  launch : Launch
  memory : Memory
  fullShard : memory.rows = 25152
  matched : matchesPeriod launch period
  sparse : Bool

def NativeCall.reference (call : NativeCall period) : Output call.memory :=
  event dense call.next.pull call.next.indices call.next.order call.arithmetic call.launch call.memory

def NativeCall.output (call : NativeCall period) (programs : NativePrograms)
    (state : WorkspaceState) : Output call.memory × Workspace :=
  consumingEvent programs period call.next.pull call.next.indices call.next.order
    call.arithmetic call.launch call.memory call.fullShard state.history state.bytes call.sparse

theorem NativeCall.output_exact (call : NativeCall period) (programs : NativePrograms)
    (state : WorkspaceState) :
    call.output programs state = (call.reference, state.bytes.consumed state.history call.sparse) :=
  consuming_event_exact programs period call.next.pull call.next.indices call.next.order
    call.arithmetic call.launch call.memory call.fullShard state.history state.bytes call.sparse
    state.valid call.matched

def NativeCall.advance (call : NativeCall period) (programs : NativePrograms)
    (state : WorkspaceState) : WorkspaceState :=
  ⟨state.history.afterConsume call.sparse, (call.output programs state).2, by
    rw [call.output_exact programs state]
    exact consume_keeps_history state.history state.bytes call.sparse state.valid⟩

inductive WorkspaceEvent where
  | materialized (next : PriorMarked) (sparse : Bool) (scale : F32)
  | consumed (period : Period) (call : NativeCall period)

def WorkspaceEvent.advance (programs : NativePrograms) (state : WorkspaceState) : WorkspaceEvent → WorkspaceState
  | .materialized next sparse scale => state.materialize next sparse scale
  | .consumed _ call => call.advance programs state

theorem WorkspaceEvent.trace_starts_clean (programs : NativePrograms) (state : WorkspaceState)
    (events : List WorkspaceEvent) :
    let next := events.foldl (WorkspaceEvent.advance programs) state
    next.bytes.beforeScatter next.history = Workspace.zero ∧
      (next.bytes.beforeDense next.history).gradient = Workspace.zero.gradient := by
  intro next
  exact ⟨sparse_preparation_zero _ _ next.valid, dense_preparation_zero _ _ next.valid⟩

#axiomsClean NativeCall.output_exact
#axiomsClean NativeCall.advance
#axiomsClean WorkspaceEvent.trace_starts_clean

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
