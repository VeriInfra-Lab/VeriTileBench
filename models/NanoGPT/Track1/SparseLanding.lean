import models.NanoGPT.Track1.ValueAdam
import models.NanoGPT.Track1.NgramSlots

/-! Row exchanges carry BF16 payloads without performing arithmetic on them.
The cache writes use the incoming peer order with the owner's block inserted. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SparseLanding
open SparsePackets (Channel Cut)

abbrev Rows := Nat → Channel → UInt16
abbrev ReadRows := Nat → Channel → Option UInt16

structure ServeInput where
  count : Nat
  asked : Nat → Int32
  first : Int32
  shardRows : Nat
  shard : Rows

def shardRead (s : ServeInput) (id : Int32) (col : Channel) : Option UInt16 :=
  let offset := id.toInt
  if 0 ≤ offset ∧ offset < s.shardRows then some (s.shard offset.toNat col) else none

def stagedServe (s : ServeInput) : ReadRows :=
  let offsets := fun row => s.asked row - s.first
  fun row col => if row < s.count then shardRead s (offsets row) col else none

def directServe (s : ServeInput) : ReadRows := fun row col =>
  if row < s.count then shardRead s (s.asked row - s.first) col else none

theorem serve_exact (s : ServeInput) : directServe s = stagedServe s := rfl

structure LandInput where
  cut : Cut
  shard : ServeInput
  received : Rows
  previous : Rows

def stagedLand (s : LandInput) : ReadRows :=
  let copyPrefix := fun row col =>
    if row < s.cut.lo then some (s.received row col) else some (s.previous row col)
  let suffix := fun row col =>
    if s.cut.hi ≤ row ∧ row < s.cut.rows then some (s.received (s.cut.lo + (row - s.cut.hi)) col)
    else copyPrefix row col
  fun row col =>
    if s.cut.lo ≤ row ∧ row < s.cut.hi then shardRead s.shard (s.shard.asked row - s.shard.first) col
    else suffix row col

def directLand (s : LandInput) : ReadRows := fun row col =>
  if row < s.cut.rows then
    if s.cut.lo ≤ row ∧ row < s.cut.hi then shardRead s.shard (s.shard.asked row - s.shard.first) col
    else some (s.received (row - (if s.cut.hi ≤ row then s.cut.own else 0)) col)
  else some (s.previous row col)

theorem land_exact (s : LandInput) (row : Nat) (col : Channel) :
    directLand s row col = stagedLand s row col := by
  have ordered := s.cut.ordered
  have within := s.cut.within
  by_cases live : row < s.cut.rows
  · by_cases before : row < s.cut.lo
    · have hn : ¬s.cut.lo ≤ row := by omega
      have hh : ¬s.cut.hi ≤ row := by omega
      simp [directLand, stagedLand, live, before, hn, hh]
    · have hlo : s.cut.lo ≤ row := by omega
      by_cases own : row < s.cut.hi
      · simp [directLand, stagedLand, live, hlo, own]
      · have hhi : s.cut.hi ≤ row := by omega
        have slot : row - s.cut.own = s.cut.lo + (row - s.cut.hi) := by
          unfold Cut.own
          omega
        simp [directLand, stagedLand, live, own, hhi, slot]
  · have hb : ¬row < s.cut.lo := by omega
    have hh : ¬row < s.cut.hi := by omega
    simp [directLand, stagedLand, live, hb, hh]

theorem cache_tail_exact (s : LandInput) (row : Nat) (col : Channel) (tail : s.cut.rows ≤ row) :
    directLand s row col = some (s.previous row col) := by simp [directLand, Nat.not_lt.mpr tail]

theorem wire_slot_bounded (cut : Cut) (row : Nat) (live : row < cut.rows)
    (remote : row < cut.lo ∨ cut.hi ≤ row) :
    row - (if cut.hi ≤ row then cut.own else 0) < cut.peers := by
  have := cut.ordered
  have := cut.within
  unfold Cut.peers Cut.own
  split <;> omega

theorem shard_address_i64 {rows : Nat} (row : Fin rows) (col : Channel) (bound : rows < 2 ^ 31) :
    (Int64.ofNat (row.val * 768 + col.val)).toInt = row.val * 768 + col.val := by
  apply Int64.toInt_ofNat_of_lt
  have := SparsePackets.address_range row col
  omega

theorem default_shard_needs_wide_address : 2 ^ 31 < (84602880 / 8 : Nat) * 768 := by decide

theorem tile_covers (row : Fin 4) (col : Channel) :
    ∃ (thread : Fin 128) (stripe : Fin 3) (word : Fin 8),
      ((thread.val >>> 5) &&& 3) = row.val ∧
      stripe.val * 256 + ((thread.val <<< 3) &&& 248) + word.val = col.val := by
  let lane : Fin 256 := ⟨col.val % 256, by omega⟩
  refine ⟨⟨row.val * 32 + lane.val / 8, by have := row.isLt; have := lane.isLt; omega⟩,
    ⟨col.val / 256, by have := col.isLt; omega⟩, ⟨lane.val % 8, by omega⟩, ?_, ?_⟩
  · exact (SparsePackets.lowered_thread_owner row lane).1
  · have h := (SparsePackets.lowered_thread_owner row lane).2
    rw [Nat.add_assoc, h]
    dsimp [lane]
    omega

structure Kernels where
  serve : ServeInput → ReadRows
  land : LandInput → ReadRows
  moments : MomentReplay.Replay
  adamEntries : AdamEntries.Pack
  ngramPrepare : NgramAdam.Prepare
  ngramUpdate : NgramAdam.RowKernel
  ngramOwnedWords : NgramAdam.OwnedWords
  ngramPeerWords : NgramAdam.PeerWords
  valueNormalize : ValueAdam.WorkspaceNormalize
  valueEvent : ValueAdam.EventKernel
  slots : NgramSlots.Lookup

def staged : Kernels :=
  ⟨stagedServe, stagedLand, MomentReplay.staged, AdamEntries.staged, NgramAdam.stagedPrepare, NgramAdam.rowUpdate,
    NgramAdam.Pull.ownGradient, NgramAdam.Pull.peerGradient, ValueAdam.clearDense,
    ValueAdam.event ValueAdam.dense, NgramSlots.native⟩
def direct : Kernels :=
  ⟨directServe, directLand, MomentReplay.selected, AdamEntries.fused, NgramAdam.fusedPrepare,
    NgramAdam.selectedRowUpdate, NgramAdam.Pull.ownGradient, NgramAdam.Pull.peerGradient,
    ValueAdam.clearSelected,
    (fun p idx order ar launch memory =>
      ValueAdam.event (ValueAdam.clearSelected memory.prior) p idx order ar launch memory),
    NgramSlots.bucketed⟩

theorem kernels_equal : direct = staged := by
  have send : directServe = stagedServe := funext serve_exact
  have land : directLand = stagedLand := by
    funext s row col
    exact land_exact s row col
  simp only [direct, staged, send, land, MomentReplay.selected_equal, AdamEntries.pack_equal,
    NgramAdam.prepare_equal, NgramAdam.selected_row_update_equal, ValueAdam.workspace_normalization_equal,
    ValueAdam.clearDense, NgramSlots.bucketed_exact]

inductive Kind where
  | value | ngram

inductive ForwardPull where
  | cold | restore | ready

/-- The wait operations expose the inputs only after the existing async
work and landing state have been consumed. The send and cache callbacks
retain the original pull metadata and buffer lifetimes in the host state. -/
structure Runtime (State : Type) where
  waitIDs : Kind → State → Option (ServeInput × State)
  send : Kind → ReadRows → State → Option State
  waitNgram : State → Option (LandInput × State)
  writeCache : ReadRows → State → Option State
  hasNext : State → Option Bool
  valueAdam : ValueAdam.Runtime State
  ngramAdam : NgramAdam.Runtime State
  /-- Read the next pull's local slice and the history after the current
      Adam update. The absent snapshot is the original zero-event guard. -/
  readMoments : State → Option (Option MomentReplay.Input)
  writeMoments : MomentReplay.Memory → State → Option State
  consumeNext : State → Option State
  makeLive : State → Option State
  forwardPull : State → Option ForwardPull
  prepareColdNgram : State → Option State
  startColdNgram : State → Option State
  prepareColdValue : State → Option State
  completeColdValue : State → Option State
  prepareRestore : State → Option State
  /-- Read the live pull and current row IDs after any next-cycle request. -/
  readSlots : State → Option (NgramSlots.Request × State)
  writeSlots : Array Nat → State → Option State

def serve {State : Type} (r : Runtime State) (k : Kernels) (kind : Kind)
    (state : State) : Option State := do
  let (input, ready) ← r.waitIDs kind state
  r.send kind (k.serve input) ready

def finishLand {State : Type} (r : Runtime State) (k : Kernels) (state : State) : Option State := do
  let (input, ready) ← r.waitNgram state
  r.writeCache (k.land input) ready

def land {State : Type} (r : Runtime State) (k : Kernels) (state : State) : Option State := do
  finishLand r k (← serve r k .ngram state)

def forward {State : Type} (r : Runtime State) (k : Kernels) (state : State) :
    Option (NgramSlots.Request × State) := do
  let ready ← match ← r.forwardPull state with
    | .cold => do
        let ngram ← r.prepareColdNgram state
        let landed ← land r k ngram
        let live ← r.startColdNgram landed
        let value ← r.prepareColdValue live
        r.completeColdValue (← serve r k .value value)
    | .restore => land r k (← r.prepareRestore state)
    | .ready => pure state
  r.readSlots ready

def updateEvent {State : Type} (r : Runtime State) (k : Kernels) (step : Nat) (state : State) :
    Option State := do
  let valueUpdated ← ValueAdam.execute r.valueAdam k.adamEntries k.valueEvent step state
  let updated ← NgramAdam.execute r.ngramAdam k.ngramPrepare step valueUpdated
    k.ngramUpdate k.ngramOwnedWords k.ngramPeerWords
  if ← r.hasNext updated then
    let value ← serve r k .value updated
    let ngram ← serve r k .ngram value
    match ← r.readMoments ngram with
    | none => pure ngram
    | some snapshot => r.writeMoments (k.moments snapshot) ngram
  else pure updated

def finishEvent {State : Type} (r : Runtime State) (k : Kernels) (state : State) : Option State := do
  if ← r.hasNext state then
    let consumed ← r.consumeNext state
    r.makeLive (← finishLand r k consumed)
  else pure state

structure EventHooks (State : Type) where
  launch : State → Option State
  update : State → Option State
  finish : State → Option State

#axiomsClean land_exact
#axiomsClean cache_tail_exact
#axiomsClean wire_slot_bounded
#axiomsClean shard_address_i64
#axiomsClean default_shard_needs_wide_address
#axiomsClean tile_covers
#axiomsClean kernels_equal
end VeriTile.Bench.Models.NanoGPT.Track1.SparseLanding
