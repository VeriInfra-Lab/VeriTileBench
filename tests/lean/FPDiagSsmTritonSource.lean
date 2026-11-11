import bench.tritonbench_g.diag_ssm_triton.Proofs.Sources.Mathematical_Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def storageType : Storage → TileDType
  | .fp32 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

def loaded {d : TileDType} {sh : TileShape} : Op d sh → List (TileDType × Bool)
  | .load dt _ (.maskOther _ _) _ => [(dt,Bool.true)]
  | .load dt _ _ _ => [(dt,Bool.false)]
  | .castFloat _ _ e => loaded e
  | .withPrecision _ _ e => loaded e
  | .negFloat _ e => loaded e
  | .sub _ _ _ b => loaded b
  | _ => []

def algorithmAccesses (code : List Stmt) : List (String × TileDType × Bool) :=
  match code with
  | [] => []
  | .assign _ _ _ e :: rest =>
      (loaded e).map (fun (d,m) => ("load",d,m)) ++ algorithmAccesses rest
  | .store d _ _ _ .none :: rest => ("store",d,Bool.false) :: algorithmAccesses rest
  | .store d _ _ _ _ :: rest => ("store",d,Bool.true) :: algorithmAccesses rest
  | .forLoop _ _ body :: rest | .forRange _ _ _ _ body :: rest
  | .forRangeDyn _ _ _ _ body :: rest | .forRangeInt _ _ _ _ body :: rest =>
      algorithmAccesses body ++ algorithmAccesses rest
  | .ifThen _ body :: rest => algorithmAccesses body ++ algorithmAccesses rest
  | .ifThenElse _ left right :: rest =>
      algorithmAccesses left ++ algorithmAccesses right ++ algorithmAccesses rest
  | _ :: rest => algorithmAccesses rest
termination_by sizeOf code

/-- The structural projection retains storage dtypes and masks, including
loops represented directly as algorithm statements by the frontend. -/
def accesses (code : List ComputeStmt) : List (String × TileDType × Bool) :=
  algorithmAccesses ((ComputeStmt.listToAlgorithm? code).toOption.getD [])

/-- Storage precision and masks survive the typed transcription and loops. -/
theorem real_forward_accesses (storage : Storage) (swapped : Bool) :
    accesses (forward storage swapped "s" "x" "lambda" "y" 2 1 3 4).surfaceBody =
      [("load",storageType storage,Bool.true),("load",storageType storage,Bool.true),
       ("load",storageType storage,Bool.true),("store",storageType storage,Bool.true)] := by
  cases storage <;> cases swapped <;>
    simp (config := {maxSteps := 500000}) [forward,
      VeriTile.Bench.Repairs.DiagSsmTriton.forwardKernel,
      VeriTile.Bench.Repairs.DiagSsmTriton.forwardCandidateKernel,
      ComputeKernel.surfaceBody, accesses, algorithmAccesses, loaded, storageType,
      ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
      bind, pure, Except.bind, Except.pure, Except.toOption]

theorem real_backward_accesses (storage : Storage) (swapped : Bool) :
    accesses (backward storage swapped "s" "lambda" "y" "gs" "gx" "gl" "gy" 2 1 3 4).surfaceBody =
      List.replicate 4 ("load",storageType storage,Bool.true) ++
      List.replicate 3 ("store",storageType storage,Bool.true) := by
  cases storage <;> cases swapped <;>
    simp (config := {maxSteps := 500000}) [backward,
      VeriTile.Bench.Repairs.DiagSsmTriton.backwardKernel,
      VeriTile.Bench.Repairs.DiagSsmTriton.backwardCandidateKernel,
      ComputeKernel.surfaceBody, accesses, algorithmAccesses, loaded, storageType,
      ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
      bind, pure, Except.bind, Except.pure, Except.toOption]

theorem complex_forward_accesses (swapped : Bool) :
    accesses (complexForward swapped "s" "x" "y" "lambda" 2 1 3 4).surfaceBody =
      List.replicate 6 ("load",TileDType.real,Bool.true) ++
      List.replicate 2 ("store",TileDType.real,Bool.true) := by
  cases swapped <;>
    simp (config := {maxSteps := 500000}) [complexForward,
      VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardKernel,
      VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardCandidateKernel,
      ComputeKernel.surfaceBody, accesses, algorithmAccesses, loaded, storageType,
      ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
      bind, pure, Except.bind, Except.pure, Except.toOption]

theorem complex_backward_accesses (swapped : Bool) :
    accesses (complexBackward swapped "s" "lambda" "y" "gs" "gx" "gl" "gy" 2 1 3 4).surfaceBody =
      List.replicate 8 ("load",TileDType.real,Bool.true) ++
      List.replicate 6 ("store",TileDType.real,Bool.true) := by
  cases swapped <;>
    simp (config := {maxSteps := 500000}) [complexBackward,
      VeriTile.Bench.Repairs.DiagSsmTriton.complexBackwardKernel,
      VeriTile.Bench.Repairs.DiagSsmTriton.complexBackwardCandidateKernel,
      ComputeKernel.surfaceBody, accesses, algorithmAccesses, loaded, storageType,
      ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
      bind, pure, Except.bind, Except.pure, Except.toOption]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.SourceTypes
