import models.NanoGPT.Track1.FPEquiv
import models.NanoGPT.Track1.QKVGroupedBinding
import models.NanoGPT.Track1.QKVNativeAmaxBinding
import models.NanoGPT.Track1.QKVGluonBinding
import models.NanoGPT.Track1.QKVScaleBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVRefresh
open VeriTile.Triton.FP QKVLaunch
open QKVTTIR (Bank Scalar)

structure GroupedProgram where
  artifacts : Grouped.Artifacts
  certified : Grouped.Certificate artifacts

def GroupedProgram.check (a : Grouped.Artifacts) : Option GroupedProgram :=
  (Grouped.certify a).map fun proof => ⟨a, proof⟩

structure NativeProgram (w : Width) where
  scan : QKVNative.ScanProgram w
  referencePack : String
  packIR : String
  packPTX : String
  pack : QKVPack.GluonCertificate w referencePack packIR packPTX

def NativeProgram.check (w : Width) (scan : QKVNative.ScanArtifacts)
    (referencePack packIR packPTX : String) : Option (NativeProgram w) := do
  let scan ← QKVNative.ScanProgram.check w scan
  let pack ← QKVPack.certifyGluon w referencePack packIR packPTX
  pure ⟨scan, referencePack, packIR, packPTX, pack⟩

structure ScaleProgram where
  ttir : String
  ptx : String
  certified : QKVScale.Certificate ttir ptx

def ScaleProgram.check (ttir ptx : String) : Option ScaleProgram :=
  (QKVScale.certify ttir ptx).map fun proof => ⟨ttir, ptx, proof⟩

structure Programs where
  grouped : GroupedProgram
  perWidth : (w : Width) → NativeProgram w
  perWidthScale : ScaleProgram

def Programs.scan (programs : Programs) (grouped : Bool) (first second : Bank)
    (reduce : (QKVAmax.Lane → Scalar) → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : QKVAmax.Part) : Nat × Scalar :=
  if grouped then
    match programs.grouped with
    | ⟨_, _⟩ => (Grouped.ScanSchedule.store first second reduce (Grouped.encodeSlot w batch) part).2
  else
    match (programs.perWidth w).scan with
    | ⟨_, _, _, _⟩ =>
      QKVNative.nativeScan (QKVNative.bankView (Grouped.slotBase w) first)
        (QKVNative.bankView (Grouped.slotBase w) second) reduce w batch part

def Programs.scale (programs : Programs) (grouped : Bool) (scratch : Nat → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) : Nat × Scalar :=
  if grouped then
    match programs.grouped with
    | ⟨_, _⟩ => (Grouped.scaleStore (fun _ => scratch) (Grouped.encodeSlot w batch)).2
  else
    (QKVScale.decodeStore programs.perWidthScale.ttir programs.perWidthScale.ptx scratch w batch).getD
      (0, Value.zero .fp32)

def Programs.pack (programs : Programs) (grouped : Bool) (op : QKVPack.Arithmetic)
    (raw : QKVPack.Inputs) (w : Width) (point : QKVPack.Point (QKVPack.Geometry.ofWidth w)) :
    Option QKVPack.Writes :=
  if grouped then
    match programs.grouped with
    | ⟨_, _⟩ =>
      let (band, lane) := Grouped.encodeTile ⟨w, point.batch, point.rowTile⟩
      (Grouped.vectorPackStore op raw.first raw.second (fun _ => raw.scales)
        band lane point.colTile point.rowLane point.colLane).2
  else
    match programs.perWidth w with
    | ⟨_, _, _, _, _⟩ =>
      QKVPack.observeGluon
        (QKVNative.nativePack op (QKVNative.inputsView (Grouped.slotBase w) raw) w) point

theorem Programs.scan_eq (programs : Programs) : programs.scan = referenceCalls.qkvAmax := by
  funext grouped first second reduce w batch part
  cases grouped
  · exact (programs.perWidth w).scan.ir.stores
      (QKVNative.bankView (Grouped.slotBase w) first)
      (QKVNative.bankView (Grouped.slotBase w) second) reduce batch part
  · exact congrArg Prod.snd (programs.grouped.certified.scan first second reduce w batch part)

theorem Programs.scale_eq (programs : Programs) : programs.scale = referenceCalls.qkvScale := by
  funext grouped scratch w batch
  cases grouped
  · change (QKVScale.decodeStore _ _ scratch w batch).getD _ = QKVScale.referenceStore scratch w batch
    rw [programs.perWidthScale.certified.stores]
    rfl
  · exact congrArg Prod.snd (programs.grouped.certified.scale (fun _ => scratch) w batch)

theorem Programs.pack_eq (programs : Programs) : programs.pack = referenceCalls.qkvPack := by
  funext grouped op raw w point
  cases grouped
  · exact (programs.perWidth w).pack.stores op (QKVNative.inputsView (Grouped.slotBase w) raw) point
  · exact congrArg Prod.snd (programs.grouped.certified.pack op raw.first raw.second (fun _ => raw.scales) w point)

#axiomsClean Programs.scan_eq
#axiomsClean Programs.scale_eq
#axiomsClean Programs.pack_eq
end VeriTile.Bench.Models.NanoGPT.Track1.QKVRefresh
