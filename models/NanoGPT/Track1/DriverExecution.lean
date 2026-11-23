import models.NanoGPT.Track1.DriverSource
import models.NanoGPT.Track1.FPEquiv
import models.NanoGPT.Track1.GraphBackward
import models.NanoGPT.Track1.TailShip
import models.NanoGPT.Track1.ValidationCE
import models.NanoGPT.Track1.FineWebBatches

/-! Numerical execution of the pinned Python driver's control flow. The
host operations are interpreted by a shared runtime; their complete source
expressions are passed to that runtime, and the patched interfaces
are linked at the same points in both programs. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver
open TailNorm (F32)

structure Validation where
  step : Nat
  totalSteps : Nat
  loss : F32

inductive ValidationProgress where
  | idle
  | checked (batches batchSize : Nat) (inputs : Option (List FineWeb.TokenBatch))
  | accumulating (batches : Nat) (remaining : List FineWeb.TokenBatch) (sum : Option F32)
  | scaled (mean : F32)
  | reduced (root : Option F32)

structure Result (State : Type) where
  state : State
  /-- Most recent event first. In the record run there is one event. -/
  validations : List Validation
  validation : ValidationProgress := .idle
  schedule : Option HostSchedule.Plan := none
  pendingSlots : Option NgramSlots.Request := none
  headTape : Option TrainingHead.Tape := none

def Result.finalValidation {State : Type} (r : Result State) : Option Validation :=
  r.validations.head?.filter (fun v => v.step == v.totalSteps)

inductive Flow (State : Type) where
  | next (state : State)
  | stopLoop (state : State)

abbrev Exec (State : Type) := State → Option (Flow State)

structure ValidationRuntime (State : Type) where
  /-- `args.val_tokens` and `args.val_batch_size`. -/
  shape : State → Option (Nat × Nat)
  bindCount : Nat → State → Option State
  /-- Resolve the sorted validation files and distributed rank. The loader
  below owns decoding, windows, and hash ids. -/
  readFiles : State → Option (FineWeb.ValidationSource × State)
  /-- Upload the four arrays of each batch and bind `val_batches`. -/
  bindBatches : List FineWeb.TokenBatch → State → Option State
  nativeCE : ValidationCE.Native
  /-- The model forward exposes the validation head inputs. The driver
  binds the actual FineWeb tokens and targets when executing the head. -/
  forward : PatchedCalls → String → FineWeb.TokenBatch → State → Option (ValidationCE.BatchInput × State)
  /-- Execute the collective transport with this rank's computed scalar.
  The payload supplies peer values and the ring schedule. -/
  exchange : F32 → State → Option (ValidationLoss.Exchange × State)
  writeRoot : F32 → State → Option State
  /-- `step` and `training_schedule.total_steps` at the report site. -/
  position : State → Option (Nat × Nat)

structure TrainingSourcePlan where
  files : List ByteArray
  readiness : List Bool
  worldSize : Nat
  rank : Nat
  steps : List Nat

def trainingGeometries (schedule : HostSchedule.Plan) (steps : List Nat) :
    Option (List (Nat × FineWeb.TrainingGeometry)) :=
  steps.mapM fun step => do
    let (stage, _) ← schedule.lookup step
    some (step, ⟨stage.batchSize, stage.maxLength⟩)

theorem training_geometries_bounded (schedule : HostSchedule.Plan) (steps : List Nat)
    (result : List (Nat × FineWeb.TrainingGeometry)) (valid : trainingGeometries schedule steps = some result)
    (entry : Nat × FineWeb.TrainingGeometry) (lives : entry ∈ result) :
    entry.2.globalTokens ≤ 393216 := by
  induction steps generalizing result with
  | nil =>
    have same := Option.some.inj valid
    simp [← same] at lives
  | cons step later ih =>
    simp only [trainingGeometries, List.mapM_cons, Option.bind_eq_bind,
      Option.bind_eq_some_iff, Option.some.injEq, Prod.mk.injEq] at valid
    obtain ⟨_, ⟨⟨stage, atProgress⟩, lookup, _, rfl⟩, remaining, rest, made⟩ := valid
    change some ((step, ⟨stage.batchSize, stage.maxLength⟩) :: remaining) = some result at made
    have exactResult := Option.some.inj made
    subst result
    rcases List.mem_cons.mp lives with rfl | more
    · exact HostSchedule.lookup_batch_bound schedule step stage atProgress lookup
    · exact ih remaining rest more

structure TrainingBatchRuntime (State : Type) where
  /-- The ordered files, BOS-index readiness events, and selected step IDs. -/
  readPlan : Bool → State → Option (TrainingSourcePlan × State)
  bindSource : FineWeb.ScheduledStream → State → Option State
  /-- Return the current step and the batch cache, including any lookahead. -/
  read : State → Option (Nat × FineWeb.ScheduledStream × State)
  /-- Bind the current batch or retire the consumed batch. -/
  commit : Bool → FineWeb.TokenBatch → FineWeb.ScheduledStream → State → Option State

def executeTrainingBatch {State : Type} (r : TrainingBatchRuntime State) (kernels : PatchedCalls)
    (op : TrainingBatchOp) (s : Result State) : Option (Result State) := do
  match op with
  | .openSource warmup =>
    let schedule ← s.schedule
    let (plan, next) ← r.readPlan warmup s.state
    let steps ← trainingGeometries schedule plan.steps
    let stream ← FineWeb.ScheduledStream.open plan.files plan.readiness plan.worldSize plan.rank steps
    let state ← r.bindSource stream next
    pure {s with state}
  | .access consume =>
    let (step, stream, next) ← r.read s.state
    let (batch, stream) ← stream.access consume step kernels.packTraining kernels.ngramRowHash
    let state ← r.commit consume batch stream next
    pure {s with state}

structure ScheduleRuntime (State : Type) where
  iterations : State → Option (Nat × Nat)
  install : HostSchedule.Plan → HostSchedule.WeightTables → State → Option State
  advance : PatchedCalls → Nat → Nat → HostSchedule.Forward → State → Option State
  parameterGroups : State → Option (List HostSchedule.ParameterGroup)

structure SampledGatherInput where
  count : SampledRows.Count
  candidates : SampledInverse.Candidates count
  bank : Nat → UInt8
  hopper : Bool

structure SampledGatherRuntime (State : Type) where
  /-- Resolve full versus sampled softmax, waiting on the candidate upload in the sampled case. -/
  prepare : Nat → State → Option (Option SampledGatherInput × State)
  commit : (count : SampledRows.Count) →
    (SampledRows.Point count → (Nat × UInt8) × (Nat × UInt8)) →
    Densify.PositionMap count → State → Option State

def executeSampledGather {State : Type} (r : SampledGatherRuntime State)
    (kernels : PatchedCalls) (line : Nat) (state : State) : Option State := do
  let (input, next) ← r.prepare line state
  match input with
  | none => some next
  | some s =>
    let (weights, positions) := kernels.sampledGather s.hopper s.count s.bank s.candidates
    r.commit s.count weights positions next

structure QKVRefreshInput where
  grouped : Bool
  first : QKVTTIR.Bank
  second : QKVTTIR.Bank
  reduce : (QKVAmax.Lane → QKVTTIR.Scalar) → QKVTTIR.Scalar
  arithmetic : QKVPack.Arithmetic

structure QKVRefreshRuntime (State : Type) where
  read : State → Option (QKVRefreshInput × State)
  commit : QKVRefreshOutput → State → Option State

def QKVRefreshRuntime.execute {State : Type} (r : QKVRefreshRuntime State)
    (kernels : PatchedCalls) (state : State) : Option State := do
  let (input, ready) ← r.read state
  r.commit (refreshCache kernels input.grouped input.arithmetic input.first input.second input.reduce) ready

structure HeadRefreshInput where
  hopper : Bool
  weight : QKVTTIR.Bank
  inverseScale : QKVTTIR.Scalar

structure HeadRefreshRuntime (State : Type) where
  read : State → Option (HeadRefreshInput × State)
  commit : (LMHead.HeadPoint → Option QKVPack.Writes) → State → Option State

def HeadRefreshRuntime.execute {State : Type} (r : HeadRefreshRuntime State)
    (kernels : PatchedCalls) (state : State) : Option State := do
  let (input, ready) ← r.read state
  r.commit (kernels.headRefresh input.hopper input.weight input.inverseScale) ready

structure Fp8RefreshRuntime (State : Type) where
  qkv : QKVRefreshRuntime State
  head : HeadRefreshRuntime State
  /-- Read and clear the refresh owed by the last optimizer. -/
  takeOwed : Nat → State → Option (Option Bool × State)
  waitAttention : State → Option State
  waitBanks : State → Option State
  completeValueLanding : State → Option State
  refreshAttention : (State → Option State) → State → Option State
  refreshMLP : PatchedCalls → Bool → (State → Option State) → State → Option State
  eagerMLP : PatchedCalls → (State → Option State) → Nat → State → Option State
  /-- Snapshot, warm and record the three pinned graph variants. -/
  capture : PatchedCalls → (State → Option State) → (State → Option State) →
    Nat → State → Option State

def Fp8RefreshRuntime.flush {State : Type} (r : Fp8RefreshRuntime State)
    (kernels : PatchedCalls) (line : Nat) (state : State) : Option State := do
  let (owed, state) ← r.takeOwed line state
  let state ← r.waitAttention state
  let state ← if owed.isSome then r.refreshAttention (r.qkv.execute kernels) state else some state
  let state ← r.completeValueLanding state
  let state ← r.waitBanks state
  match owed with
  | some refreshHead => r.refreshMLP kernels refreshHead (r.head.execute kernels) state
  | none => pure state

def Fp8RefreshRuntime.flushFinal {State : Type} (r : Fp8RefreshRuntime State)
    (line : Nat) (state : State) : Option State := do
  let (_, state) ← r.takeOwed line state
  let state ← r.waitBanks state
  r.completeValueLanding state

/-- Per-operation interface of the common Python and tensor runtime.
Filesystem contents, RNG inputs and asynchronous event choices live in
its input state. The runtime is used on both sides with the same inputs.
No relation between two executions is supplied to this interface. -/
structure Runtime (State Value : Type) where
  statement : PatchedCalls → Nat → String → State → Option State
  sparseRuntime : SparsePackets.Runtime State
  readNgramTape : State → Option (NgramAccumulation.Cycle × State)
  sparseLanding : SparseLanding.Runtime State
  replicatedAdam : ReplicatedAdam.Runtime State
  anvilGram : AnvilGram.Runtime State
  anvilEpilogue : AnvilEpilogue.Runtime State
  graphStage : GraphStage.Runtime State
  graphBackward : GraphBackward.Runtime State
  sampledGather : SampledGatherRuntime State
  fp8Refresh : Fp8RefreshRuntime State
  schedule : ScheduleRuntime State
  optimizerPosition : State → Option (Nat × Bool)
  tiedEmbedding : Bf16Transpose.Runtime State
  /-- Dispatch the sparse hooks at their three call sites, and the
  replicated Adam hook at the first replicated label in phase two.
  The ANVIL callback runs the whitening cascade and its epilogue.
  At the pinned tied-embedding sites, the first transpose hook runs
  before launching the head gradient reduction and the second after
  waiting for the head gather. Both retain the source's parameter and
  gradient presence checks. -/
  optimizerStep : HostSchedule.OptimizerValues → Option (SparseLanding.EventHooks State) →
    (State → Option State) → (AnvilEpilogue.Bank → State → Option State) →
    Option (Bf16Transpose.OptimizerHooks State) → Nat → String → State → Option State
  readTailTick : State → Option TailLerp.Snapshot
  writeTailTick : Nat → TailLerp.Output → State → Option State
  /-- The aligned Hopper case uses the explicit numerical ship model.
  Other configurations retain the original runtime path. -/
  readTailShip : State → Option TailShip.Snapshot
  gatherTailShard : TailShip.Gather State
  writeTailShip : Nat → TailShip.Weights → State → Option State
  test : Nat → String → State → Option (Bool × State)
  iterable : Nat → String → State → Option (List Value × State)
  bind : String → Value → State → Option State
  enterContext : List String → State → Option State
  exitContext : List String → State → Option State
  enterStep : String → State → Option State
  exitStep : State → Option State
  enterMain : State → Option State
  exitMain : State → Option State
  trainingBatches : TrainingBatchRuntime State
  validation : ValidationRuntime State

def Result.update {State : Type} (r : Result State) (f : State → Option State) : Option (Result State) :=
  (f r.state).map (fun s => {r with state := s})

def executeValidation {State : Type} (r : ValidationRuntime State) (kernels : PatchedCalls)
    (op : ValidationCalc) (s : Result State) : Option (Result State) := do
  match op, s.validation with
  | .checkBatchCount, .idle =>
    let (tokens, batchSize) ← r.shape s.state
    let count ← ValidationLoss.batchCount tokens batchSize
    pure {s with validation := .checked count batchSize none}
  | .bindBatchCount, .checked count _ none => s.update (r.bindCount count)
  | .loadBatches, .checked count batchSize none =>
    let (batches, next) ← if count = 0 then some ([], s.state) else do
      let (source, next) ← r.readFiles s.state
      let batches ← source.load batchSize count kernels.ngramRowHash
      some (batches, next)
    let next ← r.bindBatches batches next
    pure {s with state := next, validation := .checked count batchSize (some batches)}
  | .resetLoss, .checked count _ (some batches) => pure {s with validation := .accumulating count batches none}
  | .accumulate call, .accumulating count (input :: remaining) sum =>
    let (head, next) ← r.forward kernels call input s.state
    let mean ← kernels.validationCE r.nativeCE head input.inputs input.targets
    let total := TailNorm.add (sum.getD TailNorm.zero) mean
    pure {s with state := next, validation := .accumulating count remaining (some total)}
  | .scale, .accumulating count [] (some total) =>
    if count = 0 then none else pure {s with validation := .scaled (ValidationLoss.scaleBatches total count)}
  | .reduce, .scaled mean =>
    let (payload, next) ← r.exchange mean s.state
    let root := ValidationLoss.rootLoss payload mean
    let committed ← match root with
      | none => some next
      | some value => r.writeRoot value next
    pure {s with state := committed, validation := .reduced root}
  | _, _ => none

def reportValidation {State : Type} (r : ValidationRuntime State) (s : Result State) : Option (Result State) := do
  match s.validation with
  | .reduced (some loss) =>
    let (step, totalSteps) ← r.position s.state
    pure {s with validation := .idle, validations := ⟨step, totalSteps, loss⟩ :: s.validations}
  | .reduced none => pure {s with validation := .idle}
  | _ => none

def sequence {State : Type} (first rest : Exec State) : Exec State := fun s =>
  match first s with
  | none => none
  | some (.stopLoop t) => some (.stopLoop t)
  | some (.next t) => rest t

def runLoop {State Value : Type}
    (bind : Value → State → Option State) (body : Exec State) : List Value → Exec State
  | [], s => some (.next s)
  | v :: vs, s => do
      let s ← bind v s
      match ← body s with
      | .stopLoop t => pure (.next t)
      | .next t => runLoop bind body vs t

def procedure {State : Type} (enter leave : State → Option State) (body : Exec State) :
    State → Option State := fun s => do
  let s ← enter s
  match ← body s with
  | .next t => leave t
  | .stopLoop _ => none

mutual

def compileBlock {State Value : Type} (r : Runtime State Value) (kernels : PatchedCalls)
    (invoke : String → Result State → Option (Result State)) : List Stmt → Exec (Result State)
  | [] => fun s => some (.next s)
  | stmt :: rest => sequence (compileStmt r kernels invoke stmt) (compileBlock r kernels invoke rest)
termination_by code => sizeOf code

def compileStmt {State Value : Type} (r : Runtime State Value) (kernels : PatchedCalls)
    (invoke : String → Result State → Option (Result State)) : Stmt → Exec (Result State)
  | .shared line code => fun s =>
      let refresh := r.fp8Refresh
      let action :=
        match code with
        | "model.quantize_attn_fp8()" => refresh.qkv.execute kernels
        | "model.quantize_mlp_fp8(refresh_lm=True)" => refresh.eagerMLP kernels (refresh.head.execute kernels) line
        | "deferred_gathers.flush()" => refresh.flush kernels line
        | "deferred_gathers.flush_final()" => refresh.flushFinal line
        | "fp8_refresh.capture()" => refresh.capture kernels (refresh.qkv.execute kernels)
            (refresh.head.execute kernels) line
        | _ => r.statement kernels line code
      (s.update action).map .next
  | .diagnostics _ _ => fun s => some (.next s)
  | .validationCalc _ op => fun s => (executeValidation r.validation kernels op s).map .next
  | .validation _ _ => fun s => (reportValidation r.validation s).map .next
  | .trainStep _ code => fun s => (invoke code s).map .next
  | .trainingBatch _ op => fun s => (executeTrainingBatch r.trainingBatches kernels op s).map .next
  | .setupSchedule _ => fun s => do
      let (base, extension) ← r.schedule.iterations s.state
      let schedule ← HostSchedule.build base extension
      let weights ← schedule.weights
      let state ← r.schedule.install schedule (kernels.scheduleUpload weights) s.state
      pure (.next {s with state, schedule := some schedule})
  | .advanceSchedule line => fun s => do
      let schedule ← s.schedule
      let (step, _) ← r.optimizerPosition s.state
      let next ← schedule.forward step
      (s.update (r.schedule.advance kernels line step next)).map .next
  | .optimizerStep line code => fun s => do
      let (step, warmup) ← r.optimizerPosition s.state
      let schedule ← s.schedule
      let controls ← schedule.optimizer step
      let groups ← r.schedule.parameterGroups s.state
      let prepareNgram state : Option (SparsePackets.NgramInput × State) := do
        let (cycle, state) ← r.readNgramTape state
        pure (⟨cycle.cut, kernels.ngramAccumulation cycle⟩, state)
      let event : SparseLanding.EventHooks State := {
        launch := SparsePackets.launch r.sparseRuntime kernels.sparsePackets prepareNgram
        update := SparseLanding.updateEvent r.sparseLanding kernels.sparseLanding step
        finish := SparseLanding.finishEvent r.sparseLanding kernels.sparseLanding }
      let hooks := if SparsePackets.eventDue warmup step then some event else none
      let replicated := if controls.updateAdam then ReplicatedAdam.execute r.replicatedAdam @kernels.replicatedAdam
        else some
      let anvil bank state := do
        let next ← AnvilGram.cascade r.anvilGram kernels.anvilRails kernels.anvilGram
          kernels.anvilNormalize bank state (runRound := kernels.anvilRound) (runInitial := kernels.anvilInitial)
        AnvilEpilogue.execute r.anvilEpilogue kernels.anvilEpilogue bank next
      let tied : Option (Bf16Transpose.OptimizerHooks _) :=
        if controls.updateAdam then
          some ⟨Bf16Transpose.execute r.tiedEmbedding kernels.bf16Transpose .add,
            Bf16Transpose.execute r.tiedEmbedding kernels.bf16Transpose .copy⟩
        else none
      (s.update (r.optimizerStep (controls.values step groups) hooks replicated anvil tied line code)).map .next
  | .rowForward _ => fun s => do
      let (request, ready) ← SparseLanding.forward r.sparseLanding kernels.sparseLanding s.state
      -- A prepared bucket lookup has no readers before graph staging.
      -- Both presentations bind its value at that boundary.
      if request.useDirectory then
        pure (.next {s with state := ready, pendingSlots := some request})
      else
        let state ← r.sparseLanding.writeSlots (request.resolve kernels.sparseLanding.slots) ready
        pure (.next {s with state, pendingSlots := none})
  | .sampledGather line => fun s => (s.update (executeSampledGather r.sampledGather kernels line)).map .next
  | .graphForward _ => fun s => do
      let head := TrainingHead.forward kernels.headActivation kernels.trainingCE kernels.muddResidual
      let (state, tape) ← GraphStage.execute r.graphStage kernels.graphStage kernels.graphLaunch kernels.graphCopyLaunch
        kernels.sparseLanding.slots kernels.trainingCE head r.sparseLanding.writeSlots s.pendingSlots s.state
        kernels.muddCoefficients kernels.muddNorm
      pure (.next {s with state, pendingSlots := none, headTape := tape})
  | .graphBackward _ => fun s => do
      let head := TrainingHead.backwardWithNorm kernels.ceTranspose kernels.sampledDensify kernels.muddBackward
        kernels.muddActivation
      let state ← GraphBackward.execute r.graphBackward kernels.valueAccumulation kernels.ceTranspose head
        s.headTape s.state kernels.muddGradients kernels.muddPacking kernels.muddFusedPacking kernels.muddFullPacking
      pure (.next {s with state, headTape := none})
  | .graphHoldNgram _ => fun s => do
      let (step, warmup) ← r.optimizerPosition s.state
      (s.update (GraphBackward.holdAndAppend r.graphBackward (SparsePackets.eventDue warmup step))).map .next
  | .rowLand _ => fun s =>
      (s.update (SparseLanding.land r.sparseLanding kernels.sparseLanding)).map .next
  | .tailTick line => fun s => do
      let snapshot ← r.readTailTick s.state
      (s.update (r.writeTailTick line (TailLerp.tick kernels.tailUpdate snapshot))).map .next
  | .tailShip line => fun s =>
      match r.readTailShip s.state with
      | none => (s.update (r.statement kernels line "tail_averages.ship()")).map .next
      | some snapshot => do
        let (weights, next) ← TailShip.execute kernels.tailRestore r.gatherTailShard snapshot s.state
        (s.update (fun _ => r.writeTailShip line weights next)).map .next
  | .breakLoop _ => fun s => some (.stopLoop s)
  | .branch line cond yes no => fun s => do
      let (b, t) ← r.test line cond s.state
      if b then compileBlock r kernels invoke yes {s with state := t}
      else compileBlock r kernels invoke no {s with state := t}
  | .loop line target source body => fun s => do
      let (items, t) ← r.iterable line source s.state
      runLoop (fun v u => u.update (r.bind target v)) (compileBlock r kernels invoke body) items {s with state := t}
  | .within _ items body => fun s => do
      let t ← s.update (r.enterContext items)
      let ctrl ← compileBlock r kernels invoke body t
      match ctrl with
      | .next u => (u.update (r.exitContext items)).map .next
      | .stopLoop u => (u.update (r.exitContext items)).map .stopLoop
termination_by stmt => sizeOf stmt

end

/-- The actual, nonrecursive train_step. Its two driver call sites are the
warmup loop and the timed loop. The runtime evaluates the source call's
arguments when it enters the frame. -/
def invokeTraining {State Value : Type} (r : Runtime State Value) (kernels : PatchedCalls)
    (call : String) : Result State → Option (Result State) :=
  procedure (fun s => s.update (r.enterStep call)) (fun s => s.update r.exitStep)
    (compileBlock r kernels (fun _ _ => none) trainingStep)

/-- Entry point for numerical observations. It runs module initialization,
model creation, warmup, reset, training, validation and process teardown.
The entry guard is the script case of Python's __main__. -/
def execute {State Value : Type} (r : Runtime State Value) (kernels : PatchedCalls)
    (initial : State) : Option (Result State) := do
  let init ← procedure some some (compileBlock r kernels (fun _ _ => none) moduleInitialization)
    { state := initial, validations := [] }
  procedure (fun s => s.update r.enterMain) (fun s => s.update r.exitMain)
    (compileBlock r kernels (invokeTraining r kernels) main) init

/-- Matching control, failures and validation observations are obtained by
substituting the proved ports into this fixed driver. -/
theorem execution_equal {State Value : Type} (r : Runtime State Value) (s : State) :
    execute r candidateCalls s = execute r referenceCalls s := by
  rw [calls_equal]

/-- A timed n-gram cycle encounters an update in at most four consecutive steps,
also when a cycle crosses the change from period two to period four. -/
theorem timed_ngram_cycle (step : Nat) :
    ∃ offset : Fin GraphBackward.ringSize, SparsePackets.eventDue false (step + offset.val) = true := by
  let offset := 3 - step % 4
  have inRange : offset < GraphBackward.ringSize := by
    unfold offset GraphBackward.ringSize
    omega
  refine ⟨⟨offset, inRange⟩, ?_⟩
  have odd : (step + offset) % 2 = 1 := by unfold offset; omega
  have fourth : (step + offset) % 4 = 3 := by unfold offset; omega
  simp [SparsePackets.eventDue, odd, fourth]

end VeriTile.Bench.Models.NanoGPT.Track1.Driver
