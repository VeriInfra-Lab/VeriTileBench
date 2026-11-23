import models.NanoGPT.Track1.GraphPlainMemory
import models.NanoGPT.Track1.TrainingHead
import models.NanoGPT.Track1.MuddNorm

/-! Run the checked fused input staging on the captured buffers. The
launcher guards and the uploaded arrays determine the launch packet. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphStage
open CudaStores (Memory)
open GraphStageMemory (Layout Ready UploadedHashes)

structure Packet (s : Shape) where
  wantSize : Nat
  layout : Layout s wantSize
  memory : Memory
  ready : Ready layout memory
  fits : wantSize < 2147483648
  batch : FineWeb.TokenBatch
  hashed : batch.hashedRows
  upload : UploadedHashes layout memory batch
  schedule : List Nat
  complete : (GraphStageMemory.grid s).Perm schedule
  member : Bool
  rowsPresent : member = true → GraphMemberLaunch.RowsPresent layout memory ready

def Packet.request {s : Shape} (p : Packet s) : NgramSlots.Request where
  useDirectory := true
  want := p.ready.want
  rows := Array.ofFn fun i : Fin (extent s .slots) => GraphStageMemory.query p.layout p.memory i.val

theorem Packet.request_rows {s : Shape} (p : Packet s) :
    p.request.rows = p.batch.ngramRows.map (fun row => row.toInt32.toInt) := by
  apply Array.ext
  · simp [Packet.request, p.upload.count]
  · intro i left right
    have live : i < extent s .slots := by simpa [Packet.request] using left
    let index : Fin (extent s .slots) := ⟨i, live⟩
    have within : i < p.batch.ngramRows.size := by rw [p.upload.count]; exact live
    let row := p.batch.ngramRows[i]'within
    have range := FineWeb.hash_rows_range p.batch.inputs p.batch.ngramRows p.hashed row (Array.getElem_mem within)
    simp only [Packet.request, Array.getElem_ofFn, Array.getElem_map, GraphStageMemory.query]
    rw [p.upload.word index, Int32.toInt_ofNat_of_lt (Nat.lt_trans range (by decide)),
      FineWeb.decode_hashed_row row range]
    rfl

abbrev Launch := (s : Shape) → Packet s → Option (Observation s)

def referenceLaunch : Launch := fun _ p =>
  some (separateLookup _ (GraphStageMemory.readField p.layout p.memory .source)
    p.ready.want (fun i => GraphStageMemory.query p.layout p.memory i.val))

theorem reference_pending_exact {s : Shape} (p : Packet s) :
    (referenceLaunch s p).map (fun observed => Array.ofFn observed.pendingSlots) =
      some (p.request.resolve NgramSlots.native) := by
  simp [referenceLaunch, separateLookup, Packet.request, NgramSlots.Request.resolve, NgramSlots.native]
  rfl

structure Program (s : Shape) where
  plain : GraphPlainPTX.Program s
  hashedPTX : String
  memberPTX : String
  hashed : GraphSlotPTXBinding.Certificate s hashedPTX
  member : GraphSlotPTXBinding.MemberCertificate s memberPTX

abbrev Programs := (s : Shape) → Program s

def Programs.copy (programs : Programs) : Copy :=
  GraphPlainPTX.Programs.run fun s => (programs s).plain

theorem Programs.copy_eq (programs : Programs) : programs.copy = eager :=
  GraphPlainPTX.Programs.run_eq _

structure CopyPacket (s : Shape) where
  layout : GraphPlainMemory.Layout s
  memory : Memory
  schedule : List Nat
  complete : (List.range s.blocks).Perm schedule

abbrev CopyLaunch := (s : Shape) → CopyPacket s → Option (StaticInputs s)

def referenceCopyLaunch : CopyLaunch := fun _ p => some (GraphPlainMemory.observe p.layout false p.memory)

def Programs.copyLaunch (programs : Programs) : CopyLaunch := fun s p =>
  (GraphPlainMemory.interpret (programs s).plain p.layout p.schedule p.memory).map
    (GraphPlainMemory.observe p.layout true)

theorem Programs.copyLaunch_eq (programs : Programs) : programs.copyLaunch = referenceCopyLaunch := by
  funext s p
  simp only [Programs.copyLaunch, referenceCopyLaunch, GraphPlainMemory.interpret, (programs s).plain.recognized,
    if_true, Option.map_some, GraphPlainMemory.launch_observes_copy p.layout p.schedule p.complete p.memory]

def Program.runMemory {s : Shape} (program : Program s) (packet : Packet s) : Option Memory :=
  if packet.member then
    GraphSlotPTXBinding.interpretMemberLaunch s program.memberPTX packet.layout packet.schedule packet.memory
  else
    GraphSlotPTXBinding.interpretLaunch s program.hashedPTX packet.layout packet.schedule packet.memory

theorem Program.runMemory_eq {s : Shape} (program : Program s) (p : Packet s) :
    program.runMemory p = some (GraphStageMemory.launch p.layout p.schedule p.memory) := by
  unfold Program.runMemory
  split
  · rename_i member
    exact program.member.replay _ p.layout p.memory p.ready p.fits p.batch p.hashed p.upload
      (p.rowsPresent member) p.schedule
  · exact program.hashed.replay _ p.layout p.memory p.ready p.fits p.batch p.hashed p.upload p.schedule

def Programs.run (programs : Programs) : Launch := fun s packet =>
  ((programs s).runMemory packet).map (GraphStageMemory.observe packet.layout)

theorem Programs.run_eq (programs : Programs) : programs.run = referenceLaunch := by
  funext s packet
  simp only [Programs.run, Program.runMemory_eq, Option.map_some, referenceLaunch]
  exact congrArg some (GraphStageMemory.launch_matches_observation packet.layout packet.memory
    packet.ready packet.schedule packet.complete packet.fits)

theorem Program.runMemory_frame {s : Shape} (program : Program s) (p : Packet s)
    (address : Nat) (other : ¬ GraphStageMemory.writes p.layout address) :
    (program.runMemory p).map (· address) = some (p.memory address) := by
  rw [program.runMemory_eq, Option.map_some]
  exact congrArg some ((GraphStageMemory.launch_exact _ _ p.ready p.schedule p.complete).2 address other)

inductive Capture (request : NgramSlots.Request) where
  | words (input : CapturedInputs)
  | memory (s : Shape) (p : Packet s) (bound : p.request = request)

inductive Input where
  | eagerCopy
  | words (s : Shape) (input : StaticInputs s)
  | memory (s : Shape) (packet : CopyPacket s)

structure MuddRuntime (State : Type) where
  /-- Read the two projections and bias at the coefficient launch boundary. -/
  read : State → Option (MuddCoefficients.Packet × State)
  /-- Resume the graph with the ten BF16 coefficient tensors. -/
  resume : (packet : MuddCoefficients.Packet) → MuddCoefficients.Bank packet.shape →
    State → Option (TrainingHead.ForwardPacket × State)

structure MuddNormRuntime (State : Type) where
  read : State → Option (MuddNorm.Packet × MuddNorm.Native × MuddNorm.BackwardLayout × State)
  /-- Bind the coefficient bank and mixed residual. The statistic is consumed by the saved head packet. -/
  resume : (packet : MuddNorm.Packet) → MuddCoefficients.Bank packet.shape →
    (Fin packet.shape.rows → Fin 768 → MuddNorm.F32) →
    State → Option (TrainingHead.ForwardPacket × State)

def MuddNormRuntime.prepareHead {State : Type} (runtime : MuddNormRuntime State)
    (norm : MuddNorm.TrainingRun) (state : State) : Option (TrainingHead.ForwardPacket × State) := do
  let (packet, native, layout, ready) ← runtime.read state
  let output := norm native packet
  let saved := output.saved layout
  let (head, next) ← runtime.resume packet output.coefficients saved.mixedResidual ready
  if tokens : head.tokens = packet.shape.rows then
    let activations : TrainingHead.View head.tokens 768 UInt8 := tokens.symm ▸ output.activations
    let norm : MuddNorm.Saved head.tokens := tokens.symm ▸ saved
    return ({head with input := {head.input with encoded := some activations}, norm := some norm}, next)
  else none

structure ForwardRuntime (State : Type) where
  /-- Replay the trunk up to the final normalized BF16 activation. -/
  prepare : State → Option (TrainingHead.ForwardPacket × State)
  /-- Publish the row losses. The driver owns the saved head operands. -/
  publish : (packet : TrainingHead.ForwardPacket) → (Fin packet.tokens → TrainingCE.F32) → State → Option State
  mudd : Option (MuddRuntime State) := none
  muddNorm : Option (MuddNormRuntime State) := none

def ForwardRuntime.prepareHead {State : Type} (r : ForwardRuntime State)
    (coefficients : MuddCoefficients.Run) (state : State)
    (norm : MuddNorm.TrainingRun := (MuddNorm.RowProgram.staged.training .squareSum)) : Option (TrainingHead.ForwardPacket × State) := do
  if let some runtime := r.muddNorm then
    runtime.prepareHead norm state
  else
    let some mudd := r.mudd | r.prepare state
    let (packet, next) ← mudd.read state
    mudd.resume packet (coefficients packet.shape packet.input) next

structure Runtime (State : Type) where
  /-- A live capture can expose the launch packet. Missing captures and
      unsupported layouts take the existing eager copy path. -/
  tryCaptured : (request : NgramSlots.Request) → State → Option (Option (Capture request))
  prepare : State → Option (Input × State)
  eagerCopy : State → Option State
  write : (s : Shape) → StaticInputs s → State → Option State
  /-- The training forward computes the loss and saves the FP8 logit gradient. -/
  replay : TrainingCE.Run → State → Option State
  head : Option (ForwardRuntime State) := none

def replayForward {State : Type} (r : Runtime State) (crossEntropy : TrainingCE.Run)
    (head : TrainingHead.Forward) (staged : State)
    (coefficients : MuddCoefficients.Run := MuddCoefficients.reference)
    (norm : MuddNorm.TrainingRun := (MuddNorm.RowProgram.staged.training .squareSum)) : Option (State × Option TrainingHead.Tape) := do
  let some runtime := r.head | return (← r.replay crossEntropy staged, none)
  let (packet, state) ← runtime.prepareHead coefficients staged norm
  let output := packet.run head
  let next ← runtime.publish packet output.loss state
  return (next, some (packet.tape output.saved))

theorem replayForward_tape {State : Type} (r : Runtime State) (crossEntropy : TrainingCE.Run)
    (head : TrainingHead.Forward) (runtime : ForwardRuntime State)
    (active : r.head = some runtime) (staged next : State) (packet : TrainingHead.ForwardPacket)
    (prepared : runtime.prepareHead MuddCoefficients.reference staged = some (packet, next)) :
    (replayForward r crossEntropy head staged).map Prod.snd =
      (runtime.publish packet (packet.run head).loss next).map
        (fun _ => some (packet.tape (packet.run head).saved)) := by
  simp only [replayForward, active, prepared, bind, Option.bind_some]
  cases runtime.publish packet (packet.run head).loss next <;> rfl

theorem prepare_head_mudd {State : Type} (rt : ForwardRuntime State) (coefficients : MuddCoefficients.Run)
    (mudd : MuddRuntime State) (active : rt.mudd = some mudd) (separate : rt.muddNorm = none) (state ready : State)
    (packet : MuddCoefficients.Packet) (read : mudd.read state = some (packet, ready)) :
    rt.prepareHead coefficients state = mudd.resume packet (coefficients packet.shape packet.input) ready := by
  simp [ForwardRuntime.prepareHead, separate, active, read]

theorem prepare_head_norm {State : Type} (rt : ForwardRuntime State) (coefficients : MuddCoefficients.Run)
    (norm : MuddNorm.TrainingRun) (runtime : MuddNormRuntime State)
    (active : rt.muddNorm = some runtime) (state : State) :
    rt.prepareHead coefficients state norm = runtime.prepareHead norm state := by
  simp [ForwardRuntime.prepareHead, active]

def executeCopy {State : Type} (r : Runtime State) (copy : Copy) (launch : CopyLaunch)
    (crossEntropy : TrainingCE.Run) (head : TrainingHead.Forward) (state : State)
    (coefficients : MuddCoefficients.Run := MuddCoefficients.reference)
    (norm : MuddNorm.TrainingRun := (MuddNorm.RowProgram.staged.training .squareSum)) : Option (State × Option TrainingHead.Tape) := do
  let (input, prepared) ← r.prepare state
  let staged ← match input with
    | .eagerCopy => r.eagerCopy prepared
    | .words s words => r.write s (copy s words) prepared
    | .memory s packet => r.write s (← launch s packet) prepared
  replayForward r crossEntropy head staged coefficients norm

def execute {State : Type} (r : Runtime State) (k : Kernels) (launch : Launch) (copyLaunch : CopyLaunch)
    (lookup : NgramSlots.Lookup)
    (crossEntropy : TrainingCE.Run) (head : TrainingHead.Forward) (writeSlots : Array Nat → State → Option State)
    (pending : Option NgramSlots.Request) (state : State)
    (coefficients : MuddCoefficients.Run := MuddCoefficients.reference)
    (norm : MuddNorm.TrainingRun := (MuddNorm.RowProgram.staged.training .squareSum)) : Option (State × Option TrainingHead.Tape) := do
  let some request := pending | executeCopy r k.copy copyLaunch crossEntropy head state coefficients norm
  let publish := fun s (observed : Observation s) => do
    let withSlots ← writeSlots (Array.ofFn observed.pendingSlots) state
    replayForward r crossEntropy head (← r.write s observed.graph withSlots) coefficients norm
  match ← r.tryCaptured request state with
  | some (.memory s packet _) => publish s (← launch s packet)
  | some (.words captured) =>
    let s := captured.shape
    if h : request.rows.size = extent s .slots then
      publish s (k.withLookup s captured.fields request.want
        (fun i => request.rows[i.val]'(by rw [h]; exact i.isLt)))
    else executeCopy r k.copy copyLaunch crossEntropy head (← writeSlots (request.resolve lookup) state) coefficients norm
  | none => executeCopy r k.copy copyLaunch crossEntropy head (← writeSlots (request.resolve lookup) state) coefficients norm

theorem execute_captured_packet {State : Type} (r : Runtime State) (k : Kernels)
    (programs : Programs) (copyLaunch : CopyLaunch) (lookup : NgramSlots.Lookup) (crossEntropy : TrainingCE.Run)
    (head : TrainingHead.Forward)
    (writeSlots : Array Nat → State → Option State) {s : Shape} (p : Packet s) (state : State)
    (captured : r.tryCaptured p.request state = some (some (.memory s p rfl))) :
    execute r k programs.run copyLaunch lookup crossEntropy head writeSlots (some p.request) state = (do
      let memory ← (programs s).runMemory p
      let observed := GraphStageMemory.observe p.layout memory
      let pending ← writeSlots (Array.ofFn observed.pendingSlots) state
      replayForward r crossEntropy head (← r.write s observed.graph pending)) := by
  simp only [execute, captured, bind, Option.bind_some, Programs.run]
  cases (programs s).runMemory p <;> rfl

theorem execute_compiled_copy {State : Type} (r : Runtime State) (programs : Programs)
    (crossEntropy : TrainingCE.Run) (head : TrainingHead.Forward)
    (state prepared : State) (s : Shape) (words : StaticInputs s)
    (input : r.prepare state = some (.words s words, prepared)) :
    executeCopy r programs.copy programs.copyLaunch crossEntropy head state = (do
      let copied ← GraphPlainPTX.interpret s (programs s).plain.ptx words
      replayForward r crossEntropy head (← r.write s copied prepared)) := by
  rw [GraphPlainPTX.interpret_exact (programs s).plain.recognized, Programs.copy_eq]
  simp only [executeCopy, input, bind, Option.bind_some]

theorem execute_copy_packet {State : Type} (r : Runtime State) (programs : Programs)
    (crossEntropy : TrainingCE.Run) (head : TrainingHead.Forward)
    (state prepared : State) (s : Shape) (p : CopyPacket s)
    (input : r.prepare state = some (.memory s p, prepared)) :
    executeCopy r programs.copy programs.copyLaunch crossEntropy head state = (do
      let memory ← GraphPlainMemory.interpret (programs s).plain p.layout p.schedule p.memory
      replayForward r crossEntropy head (← r.write s (GraphPlainMemory.observe p.layout true memory) prepared)) := by
  simp only [executeCopy, input, bind, Option.bind_some, Programs.copyLaunch]
  cases GraphPlainMemory.interpret (programs s).plain p.layout p.schedule p.memory <;> rfl

#axiomsClean reference_pending_exact
#axiomsClean Packet.request_rows
#axiomsClean Program.runMemory_eq
#axiomsClean Programs.run_eq
#axiomsClean Program.runMemory_frame
#axiomsClean execute_captured_packet
#axiomsClean Programs.copy_eq
#axiomsClean Programs.copyLaunch_eq
#axiomsClean execute_compiled_copy
#axiomsClean execute_copy_packet
#axiomsClean replayForward_tape
#axiomsClean prepare_head_mudd
#axiomsClean prepare_head_norm
end VeriTile.Bench.Models.NanoGPT.Track1.GraphStage
