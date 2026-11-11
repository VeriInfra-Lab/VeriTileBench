import bench.tritonbench_g.decay_cumsum.Proofs.Sources.Mathematical_Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def dtype : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def assignType? (code : List ComputeStmt) (name : RegName) : Option TileDType :=
  match code with
  | [] => none
  | .assign d _ out _ :: rest => if out = name then some d else assignType? rest name
  | .alg (.assign d _ out _) :: rest => if out = name then some d else assignType? rest name
  | _ :: rest => assignType? rest name

def loadType? (previous : List (RegName × TileDType)) {d : TileDType} {sh : TileShape} :
    Op d sh → Option TileDType
  | .load dtype _ _ _ => some dtype
  | .ref _ _ name => previous.lookup name
  | .castFloat _ _ e => loadType? previous e
  | .withPrecision _ _ e => loadType? previous e
  | .mul _ _ a _ => loadType? previous a
  | _ => none

/-- Track load origins through the frontend's explicit temporary assignments.
A register overwrite replaces its earlier origin. -/
def assignedLoadType? (code : List ComputeStmt) (name : RegName) : Option TileDType := go code []
where
  go (code : List ComputeStmt) (previous : List (RegName × TileDType)) : Option TileDType :=
    match code with
    | [] => none
    | .assign _ _ out e :: rest =>
      let found := e.toAlgorithm?.toOption.bind (loadType? previous)
      if out = name then found else
        let previous := previous.filter (fun pair => pair.1 != out)
        go rest (found.elim previous fun dtype => (out, dtype) :: previous)
    | .alg (.assign _ _ out e) :: rest =>
      let found := loadType? previous e
      if out = name then found else
        let previous := previous.filter (fun pair => pair.1 != out)
        go rest (found.elim previous fun dtype => (out, dtype) :: previous)
    | _ :: rest => go rest previous

def stores (code : List ComputeStmt) : List TileDType :=
  match code with
  | [] => []
  | .store d _ _ _ _ :: rest => d :: stores rest
  | .alg (.store d _ _ _ _) :: rest => d :: stores rest
  | .ifThen _ body :: rest => stores body ++ stores rest
  | .ifThenElse _ a b :: rest => stores a ++ stores b ++ stores rest
  | _ :: rest => stores rest


def loopBody (code : List ComputeStmt) : List ComputeStmt :=
  match code.getLast? with
  | some (.forRangeDyn _ _ _ _ body) => body
  | some (.forRangeInt _ _ _ _ body) => body
  | some (.alg (.forRangeDyn _ _ _ _ body)) => body.map ComputeStmt.alg
  | some (.alg (.forRangeInt _ _ _ _ body)) => body.map ComputeStmt.alg
  | _ => []

def fwd (storage : Storage) (swapped : Bool) :=
  forward storage swapped "g" "go" 16 3 1 2 2 4 2 2 3 1

def prep (source gate : Storage) (swapped : Bool) :=
  prepare source gate swapped "q" "k" "g" "qg" "kg" 16 3 1 2 2 4 2 2 3 1

def bwd (source gate : Storage) (swapped : Bool) :=
  backward source gate swapped "dq_inner" "dq_inter" "dk_inner" "dk_inter" "q" "k" "g" "dg" 16 3 1 2 2 4 2 2 3 1

/-- The load keeps its storage type before the fp32 conversion and accumulation;
the store converts to the declared destination element type. -/
theorem forward_types (storage : Storage) (swapped : Bool) :
    let code := (fwd storage swapped).surfaceBody
    assignedLoadType? (loopBody code) "_g" = some (dtype storage) ∧
    assignType? (loopBody code) "_g" = some .real ∧
    assignType? code "cum_decay" = some .real ∧
    stores (loopBody code) = [dtype storage] := by
  cases storage <;> cases swapped
  all_goals refine ⟨?_,rfl,rfl,?_⟩
  all_goals simp [stores,loopBody,fwd,forward,VeriTile.Bench.Repairs.DecayCumsum.forwardKernel,VeriTile.Bench.Repairs.DecayCumsum.forwardCandidateKernel,VeriTile.Bench.Repairs.DecayCumsum.prepareKernel,VeriTile.Bench.Repairs.DecayCumsum.prepareCandidateKernel,VeriTile.Bench.Repairs.DecayCumsum.backwardKernel,VeriTile.Bench.Repairs.DecayCumsum.backwardCandidateKernel,ComputeKernel.surfaceBody,assignedLoadType?,assignedLoadType?.go,loadType?,dtype,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

/-- Query/key and gate storage are independent, including the unmasked final
gate load. Neither the explicit fp32 conversion nor output casts are erased. -/
theorem prepare_types (source gate : Storage) (swapped : Bool) :
    let code := (prep source gate swapped).surfaceBody
    assignedLoadType? code "last_decay" = some (dtype gate) ∧
    assignedLoadType? (loopBody code) "_q" = some (dtype source) ∧
    assignedLoadType? (loopBody code) "_k" = some (dtype source) ∧
    assignedLoadType? (loopBody code) "_g" = some (dtype gate) ∧
    assignType? (loopBody code) "_g" = some .real ∧
    stores (loopBody code) = [dtype source,dtype source] := by
  cases source <;> cases gate <;> cases swapped
  all_goals refine ⟨?_,?_,?_,?_,rfl,?_⟩
  all_goals simp [stores,loopBody,prep,prepare,VeriTile.Bench.Repairs.DecayCumsum.forwardKernel,VeriTile.Bench.Repairs.DecayCumsum.forwardCandidateKernel,VeriTile.Bench.Repairs.DecayCumsum.prepareKernel,VeriTile.Bench.Repairs.DecayCumsum.prepareCandidateKernel,VeriTile.Bench.Repairs.DecayCumsum.backwardKernel,VeriTile.Bench.Repairs.DecayCumsum.backwardCandidateKernel,ComputeKernel.surfaceBody,assignedLoadType?,assignedLoadType?.go,loadType?,dtype,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

theorem backward_types (source gate : Storage) (swapped : Bool) :
    let code := (bwd source gate swapped).surfaceBody
    assignedLoadType? (loopBody code) "_g" = some (dtype gate) ∧
    assignedLoadType? (loopBody code) "_dq1" = some (dtype source) ∧
    assignedLoadType? (loopBody code) "_dq2" = some (dtype source) ∧
    assignedLoadType? (loopBody code) "_dk1" = some (dtype source) ∧
    assignedLoadType? (loopBody code) "_dk2" = some (dtype source) ∧
    assignedLoadType? (loopBody code) "_q" = some (dtype source) ∧
    assignedLoadType? (loopBody code) "_k" = some (dtype source) ∧
    assignType? code "cum_grad_dg" = some .real ∧
    stores (loopBody code) = [dtype source,dtype source,dtype gate] := by
  cases source <;> cases gate <;> cases swapped
  all_goals refine ⟨?_,?_,?_,?_,?_,?_,?_,rfl,?_⟩
  all_goals simp [stores,loopBody,bwd,backward,VeriTile.Bench.Repairs.DecayCumsum.forwardKernel,VeriTile.Bench.Repairs.DecayCumsum.forwardCandidateKernel,VeriTile.Bench.Repairs.DecayCumsum.prepareKernel,VeriTile.Bench.Repairs.DecayCumsum.prepareCandidateKernel,VeriTile.Bench.Repairs.DecayCumsum.backwardKernel,VeriTile.Bench.Repairs.DecayCumsum.backwardCandidateKernel,ComputeKernel.surfaceBody,assignedLoadType?,assignedLoadType?.go,loadType?,dtype,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.SourceTypes
