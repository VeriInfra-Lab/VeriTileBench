import VeriTile.Meta.StatementAudit

/-! FIFO execution across CUDA streams. An event identifies one recording
of an event object. A later recording can reuse the object with a new
generation. A wait can run only when its recording has completed. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.CudaStreamOrder

structure Event where
  slot : Nat
  generation : Nat
  deriving DecidableEq

inductive Command (Work : Type) where
  | work (operation : Work)
  | record (event : Event)
  | wait (event : Event)
  deriving DecidableEq

structure Queues (Work : Type) where
  pending : Nat → List (Command Work)
  completed : List Event

def Command.enabled (op : Command Work) (completed : List Event) : Prop :=
  match op with
  | .wait event => event ∈ completed
  | _ => True

def Command.recorded (op : Command Work) : List Event :=
  match op with
  | .record event => [event]
  | _ => []

def Queues.dequeue (queues : Queues Work) (stream : Nat)
    (op : Command Work) (rest : List (Command Work)) : Queues Work :=
  ⟨fun i => if i = stream then rest else queues.pending i, queues.completed ++ op.recorded⟩

inductive Run {Work : Type} :
    Queues Work → List (Nat × Command Work) → Queues Work → Prop where
  | done (queues : Queues Work) : Run queues [] queues
  | head (queues : Queues Work) (stream : Nat) (op : Command Work) (rest : List (Command Work))
      (issued : List (Nat × Command Work)) (final : Queues Work)
      (front : queues.pending stream = op :: rest) (enabled : op.enabled queues.completed)
      (tail : Run (queues.dequeue stream op rest) issued final) :
      Run queues ((stream, op) :: issued) final

def issuedOn (stream : Nat) : List (Nat × Command Work) → List (Command Work)
  | [] => []
  | (s, op) :: rest => if s = stream then op :: issuedOn stream rest else issuedOn stream rest

theorem issued_on_member {Work : Type} (stream : Nat) (op : Command Work)
    (issued : List (Nat × Command Work)) :
    op ∈ issuedOn stream issued ↔ (stream, op) ∈ issued := by
  induction issued with
  | nil => simp [issuedOn]
  | cons first rest ih =>
    obtain ⟨current, command⟩ := first
    by_cases onStream : current = stream
    · subst current
      simp [issuedOn, ih]
    · simp [issuedOn, onStream, Ne.symm onStream, ih]

theorem fifo_projection {Work : Type} {initial final : Queues Work}
    {issued : List (Nat × Command Work)} (run : Run initial issued final) (stream : Nat) :
    issuedOn stream issued ++ final.pending stream = initial.pending stream := by
  induction run with
  | done => rfl
  | head queues current op rest issued final front _ _ ih =>
    by_cases same : current = stream
    · subst current
      simp [issuedOn, ih, Queues.dequeue, front]
    · simpa only [issuedOn, if_neg same, Queues.dequeue, if_neg (Ne.symm same)] using ih

theorem fifo_complete {Work : Type} {initial final : Queues Work}
    {issued : List (Nat × Command Work)} (run : Run initial issued final) (stream : Nat)
    (empty : final.pending stream = []) :
    issuedOn stream issued = initial.pending stream := by
  simpa only [empty, List.append_nil] using fifo_projection run stream

theorem completed_events {Work : Type} {initial final : Queues Work}
    {issued : List (Nat × Command Work)} (run : Run initial issued final) :
    final.completed = initial.completed ++ issued.flatMap (fun entry => entry.2.recorded) := by
  induction run with
  | done => simp
  | head _ _ _ _ _ _ _ _ _ ih =>
    simpa only [Queues.dequeue, List.flatMap_cons, List.append_assoc] using ih

theorem wait_has_record {Work : Type} {initial final : Queues Work}
    {issued : List (Nat × Command Work)} (run : Run initial issued final)
    (stream : Nat) (event : Event) (before after : List (Nat × Command Work))
    (atWait : issued = before ++ (stream, .wait event) :: after)
    (fresh : event ∉ initial.completed) :
    ∃ source, (source, .record event) ∈ before := by
  induction run generalizing before after with
  | done => cases before <;> simp at atWait
  | head queues current op rest issued final front enabled tail ih =>
    cases before with
    | nil =>
      simp only [List.nil_append, List.cons.injEq, Prod.mk.injEq] at atWait
      rcases atWait with ⟨⟨_, rfl⟩, _⟩
      exact False.elim (fresh enabled)
    | cons first earlier =>
      simp only [List.cons_append, List.cons.injEq] at atWait
      obtain ⟨rfl, atWait⟩ := atWait
      by_cases recorded : event ∈ op.recorded
      · have isRecord : op = .record event := by
          cases op <;> simp [Command.recorded] at recorded ⊢
          exact recorded.symm
        exact ⟨current, by simp [isRecord]⟩
      · have stillFresh : event ∉ (queues.dequeue current op rest).completed := by
          simp [Queues.dequeue, fresh, recorded]
        obtain ⟨stream, hit⟩ := ih earlier after atWait stillFresh
        exact ⟨stream, List.mem_cons_of_mem _ hit⟩

#axiomsClean fifo_projection
#axiomsClean completed_events
#axiomsClean wait_has_record

end VeriTile.Bench.Models.NanoGPT.Track1.CudaStreamOrder
