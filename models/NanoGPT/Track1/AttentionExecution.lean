import models.NanoGPT.Track1.FPEquiv
import models.NanoGPT.Track1.RopeShiftBinding
import models.NanoGPT.Track1.RopeBackwardBinding
import models.NanoGPT.Track1.RopeNarrowBinding
import models.NanoGPT.Track1.ByteTransposeBinding

/-! The programs bound at each attention call site carry their emitted
IR and PTX. The training driver selects the decoded store equations at
the exact length and width of the call. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver.AttentionExecution

open VeriTile.Triton FP
open RopeShift (Length)

structure Forward (n : Length) where
  artifacts : RopeShift.Artifacts
  checked : RopeShift.Certificate n artifacts

structure Backward (n : Length) where
  artifacts : RopeBackward.Artifacts
  checked : RopeBackward.Certificate n artifacts

structure Narrow (s : RopeBackward.Narrow.Shape) (n : Length) where
  artifacts : RopeBackward.Narrow.Artifacts
  checked : RopeBackward.Narrow.Certificate s n artifacts

structure Bytes where
  artifacts : ByteTranspose.Artifacts
  checked : ByteTranspose.Certificate artifacts

structure Programs where
  forward : (n : Length) → Forward n
  backward : (n : Length) → Option (Backward n)
  narrow : (s : RopeBackward.Narrow.Shape) → (n : Length) → Option (Narrow s n)
  bytes : Bytes

def Programs.forwardStore (k : Programs) (fast : Bool) (op : RopeShift.Arithmetic) (read : RopeShift.Inputs)
    (length : Length) (point : RopeShift.Point length) : Option (RopeShift.Plane × Nat × Value .bf16) :=
  if fast && RopeShift.matchesArtifacts length (k.forward length).artifacts then
    RopeShift.emittedWrite op read point
  else RopeShift.referenceWrite op read point

theorem Programs.forwardStore_eq (k : Programs) : k.forwardStore = candidateCalls.ropeForward := by
  funext fast op read length point
  cases fast <;> simp only [Programs.forwardStore, candidateCalls, (k.forward length).checked.matched,
    Bool.and_true, Bool.false_eq_true, if_true, if_false]

def Programs.backwardStores (k : Programs) (fast : Bool) (op : RopeBackward.Target)
    (length : Length) (read : RopeBackward.Inputs length) (point : RopeBackward.Point length) :
    (Nat × UInt8) × (Nat × UInt8) :=
  match k.backward length with
  | some program =>
    RopeBackward.writes (fast && RopeBackward.matchesArtifacts length program.artifacts) op read point
  | none => candidateCalls.ropeBackward fast op length read point

theorem Programs.backwardStores_eq (k : Programs) : k.backwardStores = candidateCalls.ropeBackward := by
  funext fast op length read point
  unfold Programs.backwardStores
  split
  · rename_i program _
    simp only [program.checked.matched, Bool.and_true, candidateCalls]
  · rfl

def Programs.narrowStores (k : Programs) (op : RopeBackward.Target) (read : RopeBackward.Narrow.Inputs)
    (shape : RopeBackward.Narrow.Shape) (length : Length) (point : RopeBackward.Narrow.Point shape length) :
    (Nat × UInt8) × (Nat × UInt8) :=
  match k.narrow shape length with
  | some program =>
    RopeBackward.Narrow.writes (RopeBackward.Narrow.matchesArtifacts shape length program.artifacts) op read point
  | none => candidateCalls.ropeBackwardNarrow op read shape length point

theorem Programs.narrowStores_eq (k : Programs) : k.narrowStores = candidateCalls.ropeBackwardNarrow := by
  funext op read shape length point
  unfold Programs.narrowStores
  split
  · rename_i program _
    simp only [program.checked.matched, candidateCalls]
  · rfl

def Programs.transposeBytes (k : Programs) (fast : Bool) (shape : ByteTranspose.Shape)
    (read : Nat → UInt8) (point : ByteTranspose.Point shape) : Option (Nat × UInt8) :=
  if fast && ByteTranspose.matchesArtifacts k.bytes.artifacts then ByteTranspose.emitted read point
  else ByteTranspose.reference read point

theorem Programs.transposeBytes_eq (k : Programs) : k.transposeBytes = candidateCalls.byteTranspose := by
  funext fast shape read point
  cases fast <;> simp only [Programs.transposeBytes, candidateCalls, k.bytes.checked.recognized,
    Bool.and_true, Bool.false_eq_true, if_true, if_false]

#axiomsClean Programs.forwardStore_eq
#axiomsClean Programs.backwardStores_eq
#axiomsClean Programs.narrowStores_eq
#axiomsClean Programs.transposeBytes_eq

end VeriTile.Bench.Models.NanoGPT.Track1.Driver.AttentionExecution
