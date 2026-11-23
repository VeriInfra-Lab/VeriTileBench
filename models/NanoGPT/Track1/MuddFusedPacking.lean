import models.NanoGPT.Track1.Proofs.MuddPackingFma
import models.NanoGPT.Track1.MuddGradientPacking
import models.NanoGPT.Track1.MuddPackingArithmetic
import models.NanoGPT.Track1.CudaStreamOrder

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddFusedPacking
open MuddGradientPacking
open MuddGradientGrouping (Memory BF16 F32)
open VeriTile.Triton.FP
open VeriTile.Spec
open scoped VeriTile.Spec

def reduceZero {rows : Nat} (p : Packet rows) (reduce : (Fin 16 → F32) → F32) (m : Memory) : Memory := fun addr =>
  if h : inWindow p.columnsBase (rows * 10) addr ∧ (addr - p.columnsBase) % 20 = 0 then
    let row : Fin rows := ⟨(addr - p.columnsBase) / 20, by unfold inWindow at h; omega⟩
    cast .bf16 Config.ieeeValue (reduce fun c =>
      if live : c.val < 12 then cast .fp32 Config.ieeeValue (m (sourceAddress p 0 row ⟨c.val, live⟩))
      else Value.zero .fp32)
  else m addr

theorem zero_frame {rows : Nat} (p : Packet rows) (reduce : (Fin 16 → F32) → F32) (m : Memory) (addr : Nat)
    (out : ¬ inWindow p.columnsBase (rows * 10) addr) :
    reduceZero p reduce m addr = m addr := by simp [reduceZero, out]

theorem zero_sources {rows : Nat} (p : Packet rows) (reduce : (Fin 16 → F32) → F32) (m : Memory) :
    SameSources p (reduceZero p reduce m) m := by
  intro plane row col
  exact zero_frame p reduce m _ (fun h => p.columnsDisjoint plane _ h (source_in_window p plane row col))

def isZeroPlane {rows : Nat} (p : Packet rows) (addr : Nat) : Bool :=
  ((addr - p.packedBase) / 2 / 12) % 10 == 0

private def packedPlane {rows : Nat} (p : Packet rows) (m result : Memory) (zero : Bool) : Memory := fun addr =>
  if isZeroPlane p addr == zero then result addr else m addr

def packPart {rows : Nat} (p : Packet rows) (m : Memory) (zero : Bool) : Memory :=
  packedPlane p m (pack p m true) zero

theorem part_frame {rows : Nat} (p : Packet rows) (m : Memory) (zero : Bool) (addr : Nat)
    (out : ¬ inWindow p.packedBase (rows * 120) addr) : packPart p m zero addr = m addr := by
  unfold packPart packedPlane
  split
  · exact congrFun (pack_routed_eq p m) addr |>.trans (packing_frame p m addr out)
  · rfl

private theorem part_sources {rows : Nat} (p : Packet rows) (m : Memory) (zero : Bool) :
    SameSources p (packPart p m zero) m := by
  intro plane row col
  exact part_frame p m zero _ (fun h => p.packingDisjoint plane _ h (source_in_window p plane row col))

private theorem same_packValue {rows : Nat} (p : Packet rows) (a b : Memory) (same : SameSources p a b)
    (row : Fin rows) (plane : Fin 10) (col : Fin 12) :
    packValue p a row plane col = packValue p b row plane col := by
  unfold SameSources at same
  unfold MuddGradientPacking.packValue
  congr 1
  funext i
  split
  · congr 1
    exact same i row col
  · rfl

private theorem pack_uses_sources {rows : Nat} (p : Packet rows) (a b : Memory) (addr : Nat)
    (s : SameSources p a b) (same : a addr = b addr) :
    pack p a true addr = pack p b true addr := by
  rw [pack_routed_eq, pack_routed_eq]
  unfold pack writePacked
  split
  · exact same_packValue p a b s _ _ _
  · exact same

set_option maxHeartbeats 1000000 in
/-- Finish the grouped nine-column reduction and column-zero reduction.
Each produces its own part of the packing buffer from the same source
values. Finishing the two launches is equivalent to the separate packer. -/
theorem dual_outputs_eq {rows : Nat} (p : Packet rows) (reduce : (Fin 16 → F32) → F32) (m : Memory) :
    packPart p (reduceZero p reduce (packPart p m false)) true =
      reduceZero p reduce (pack p m true) := by
  funext addr
  by_cases packed : inWindow p.packedBase (rows * 120) addr
  · have h : ¬ inWindow p.columnsBase (rows * 10) addr := fun writes => p.outputsDisjoint addr writes packed
    rw [zero_frame p reduce (pack p m true) addr h]
    by_cases zero : isZeroPlane p addr = true
    · have chosen : packPart p (reduceZero p reduce (packPart p m false)) true addr =
          pack p (reduceZero p reduce (packPart p m false)) true addr := by simp [packPart, packedPlane, zero]
      rw [chosen]
      apply pack_uses_sources p _ _ addr
      · intro i row c
        exact (zero_sources p reduce _ i row c).trans (part_sources p m false i row c)
      · rw [zero_frame p reduce (packPart p m false) addr h]
        simp [packPart, packedPlane, zero]
    · have plane : isZeroPlane p addr = false := Bool.eq_false_iff.mpr zero
      have bypass : packPart p (reduceZero p reduce (packPart p m false)) true addr =
          reduceZero p reduce (packPart p m false) addr := by simp [packPart, packedPlane, plane]
      rw [bypass, zero_frame p reduce (packPart p m false) addr h]
      simp [packPart, packedPlane, plane]
  · rw [part_frame p _ _ _ packed]
    have reads : SameSources p (packPart p m false) (pack p m true) := by
      rw [pack_routed_eq]
      intro plane row col
      exact (part_sources p m false plane row col).trans (pack_sources p m plane row col).symm
    unfold SameSources at reads
    unfold reduceZero
    split
    · simp only [reads]
    · rw [part_frame p _ _ _ packed, pack_routed_eq, packing_frame p m addr packed]

theorem fused_block_eq {rows : Nat} (p : Packet rows) (reduce : (Fin 16 → F32) → F32) :
    packPart p (reduceZero p reduce (packPart p (MuddGradientGrouping.grouped (reduction p p.memory)) false)) true =
      reduceZero p reduce (MuddGradientPacking.Program.interleaved.run rows p) := by
  rw [dual_outputs_eq, ← MuddGradientPacking.run_eq]
  rfl

structure Inputs (rows : Nat) where
  group : Packet rows
  zeroReduce : (Fin 16 → F32) → F32
  zeroTile : Nat
  zeroWarps : Nat

inductive Program where
  | separate
  | fused
  | single

abbrev Run := (rows : Nat) → Inputs rows → Memory

inductive Launch where
  | column (plane : Fin 9)
  | pack
  | zero
  | groupAndPack
  | zeroAndPack
  | allAndPack
deriving DecidableEq

def Program.launches : Program → List Launch
  | .separate => earlyColumns.map .column ++ [.pack] ++ lateColumns.map .column ++ [.zero]
  | .fused => [.groupAndPack, .zeroAndPack]
  | .single => [.allAndPack]

def Launch.writeWith {rows : Nat} (p : Inputs rows) (packGradient : Packet rows → Memory → Memory)
    (m : Memory) : Launch → Memory
  | .column plane => reduceColumns p.group [plane] m
  | .pack => MuddGradientPacking.pack p.group m
  | .zero => reduceZero p.group p.zeroReduce m
  | .groupAndPack =>
    let reduced := MuddGradientGrouping.grouped (reduction p.group m)
    packedPlane p.group reduced (packGradient p.group reduced) false
  | .zeroAndPack =>
    let reduced := reduceZero p.group p.zeroReduce m
    packedPlane p.group reduced (packGradient p.group reduced) true
  | .allAndPack => fun addr =>
    if inWindow p.group.packedBase (rows * 120) addr then packGradient p.group m addr
    else if inWindow p.group.columnsBase (rows * 10) addr ∧ (addr - p.group.columnsBase) % 20 = 0 then
      reduceZero p.group p.zeroReduce m addr
    else MuddGradientGrouping.grouped (reduction p.group m) addr

private def Launch.writeOrdered {rows : Nat} (p : Inputs rows) (m : Memory) : Launch → Memory :=
  Launch.writeWith p (fun packet memory => MuddGradientPacking.pack packet memory true) m

private def Program.runOrdered (program : Program) : Run := fun _ p =>
  program.launches.foldl (Launch.writeOrdered p) p.group.memory

def packOneAdd {rows : Nat} (p : Packet rows) (m : Memory) : Memory :=
  writePacked p m fun row plane col =>
    let value := cast .fp32 Config.ieeeValue (m (sourceAddress p plane row col))
    scalePacked (MuddPackingArithmetic.addZero value)

private theorem pack_one_add_eq {rows : Nat} (p : Packet rows) (m : Memory) :
    packOneAdd p m = pack p m true := by
  rw [pack_routed_eq]
  apply congrArg (writePacked p m)
  funext row plane col
  let value := m (sourceAddress p plane row col)
  let loaded := cast .fp32 Config.ieeeValue value
  let selected := fun i : Fin 10 =>
    if plane = i then cast .fp32 Config.ieeeValue (m (sourceAddress p i row col)) else Value.zero .fp32
  let routed := fun i : Fin 10 => if plane = i then loaded else Value.zero .fp32
  have same : routed = selected := by
    funext i
    unfold routed selected
    split
    · rename_i chosen
      subst i
      rfl
    · rfl
  change scalePacked (MuddPackingArithmetic.addZero loaded) = scalePacked (MuddPackingArithmetic.total selected)
  exact congrArg scalePacked ((MuddPackingArithmetic.add_zero_selected value plane).symm.trans
    (congrArg MuddPackingArithmetic.total same))

def Launch.write {rows : Nat} (p : Inputs rows) (m : Memory) : Launch → Memory :=
  Launch.writeWith p packOneAdd m

/-- Replay the reduction schedule with the packing arithmetic of the selected
kernel. The independent writes in a combined launch use the same input memory. -/
def Program.runWithPacking {rows : Nat} (program : Program) (packer : Packet rows → Memory → Memory)
    (p : Inputs rows) : Memory :=
  program.launches.foldl (Launch.writeWith p packer) p.group.memory

def Program.run (program : Program) : Run := fun _ p =>
  program.runWithPacking packOneAdd p

def packFma {rows : Nat} (p : Packet rows) (m : Memory) : Memory :=
  writePacked p m fun row plane col =>
    let value := cast .fp32 Config.ieeeValue (m (sourceAddress p plane row col))
    cast .bf16 Config.ieeeValue (fma Config.ieeeValue value MuddPackingFma.tenth (Value.zero .fp32))

theorem pack_fma_eq {rows : Nat} :
    (@packFma rows) = @packOneAdd rows := by
  funext packet memory
  apply congrArg (writePacked packet memory)
  funext row plane col
  exact congrArg (cast .bf16 Config.ieeeValue)
    (MuddPackingFma.packing_fma_eq (memory (sourceAddress packet plane row col)))

private def Launch.directPacking (groupDirect zeroDirect : Bool) : Launch → Bool
  | .groupAndPack | .allAndPack => groupDirect
  | .zeroAndPack => zeroDirect
  | .column _ | .pack | .zero => false

/-- A split launch can choose the FMA packer independently for its own columns. -/
def Program.runSelected {rows : Nat} (program : Program) (p : Inputs rows)
    (groupDirect zeroDirect : Bool) : Memory :=
  program.launches.foldl (fun m launch =>
    launch.writeWith p (if launch.directPacking groupDirect zeroDirect then packFma else packOneAdd) m) p.group.memory

theorem Program.runSelected_eq {rows : Nat} (program : Program) (p : Inputs rows)
    (groupDirect zeroDirect : Bool) : program.runSelected p groupDirect zeroDirect = program.run rows p := by
  simp only [Program.runSelected, Program.run, Program.runWithPacking, pack_fma_eq, ite_self]

private theorem run_ordered_eq (program : Program) : program.run = program.runOrdered := by
  funext rows p
  unfold Program.run Program.runWithPacking Program.runOrdered Launch.writeOrdered
  have same : (@packOneAdd rows) = fun packet memory => pack packet memory true := by
    funext packet memory
    exact pack_one_add_eq packet memory
  rw [same]

private theorem append_columns {rows : Nat} (p : Packet rows) (first rest : List (Fin 9)) (m : Memory) :
    reduceColumns p rest (reduceColumns p first m) = reduceColumns p (first ++ rest) m := by
  have reads := reduce_sources p first m
  have input : (reduction p (reduceColumns p first m)).input = (reduction p m).input := by
    funext plane row col
    exact reads plane.succ row col
  have same : reduction p (reduceColumns p first m) = {reduction p m with memory := reduceColumns p first m} := by
    simp only [reduction, MuddGradientGrouping.Packet.mk.injEq, and_true] at input ⊢
    exact input
  have store (addr : Nat) :
      MuddGradientGrouping.writeColumn (reduction p (reduceColumns p first m)) addr =
        MuddGradientGrouping.writeColumn (reduction p m) addr := by
    rw [same]
    funext old plane
    rfl
  funext addr
  change rest.foldl (MuddGradientGrouping.writeColumn (reduction p (reduceColumns p first m)) addr)
    (reduceColumns p first m addr) = (first ++ rest).foldl (MuddGradientGrouping.writeColumn (reduction p m) addr) (m addr)
  rw [store, List.foldl_append]
  rfl

private theorem replay_columns {rows : Nat} (p : Inputs rows) (planes : List (Fin 9)) (m : Memory) :
    (planes.map Launch.column).foldl (Launch.writeOrdered p) m = reduceColumns p.group planes m := by
  induction planes generalizing m with
  | nil => rfl
  | cons plane rest ih =>
    simp only [List.map_cons, List.foldl_cons, ih, Launch.writeOrdered, Launch.writeWith]
    exact append_columns p.group [plane] rest m

theorem separate_run {rows : Nat} (p : Inputs rows) :
    Program.separate.run rows p = reduceZero p.group p.zeroReduce
      (MuddGradientPacking.Program.interleaved.run rows p.group) := by
  rw [run_ordered_eq]
  simp only [Program.runOrdered, Program.launches, List.foldl_append, replay_columns, List.foldl_cons,
    List.foldl_nil, Launch.writeOrdered, Launch.writeWith, MuddGradientPacking.Program.run]

theorem run_eq : Program.fused.run = Program.separate.run := by
  funext rows p
  rw [run_ordered_eq, separate_run]
  exact fused_block_eq p.group p.zeroReduce

theorem single_run_eq : Program.single.run = Program.fused.run := by
  rw [run_ordered_eq, run_ordered_eq]
  funext rows p
  let packet := p.group
  let cols := MuddGradientGrouping.grouped (reduction packet packet.memory)
  have reads : SameSources packet cols packet.memory := by
    unfold cols
    rw [← join_columns]
    intro plane row col
    exact (reduce_sources packet lateColumns _ plane row col).trans (reduce_sources packet earlyColumns _ plane row col)
  change Launch.allAndPack.writeOrdered p packet.memory =
    packPart packet (reduceZero packet p.zeroReduce (packPart packet cols false)) true
  rw [dual_outputs_eq]
  funext addr
  by_cases packed : inWindow packet.packedBase (rows * 120) addr
  · have noCols : ¬ inWindow packet.columnsBase (rows * 10) addr :=
      fun h => packet.outputsDisjoint addr h packed
    rw [zero_frame packet _ _ _ noCols]
    have framed : cols addr = packet.memory addr := by
      unfold cols
      rw [← join_columns, columns_frame packet _ _ _ noCols, columns_frame packet _ _ _ noCols]
    simp only [Launch.writeOrdered, Launch.writeWith, show p.group = packet from rfl, if_pos packed]
    exact (pack_uses_sources packet cols packet.memory addr reads framed).symm
  · have unpacked : pack packet cols true addr = cols addr := by
      rw [pack_routed_eq, packing_frame packet _ _ packed]
    have sameReads : SameSources packet (pack packet cols true) packet.memory := by
      rw [pack_routed_eq]
      intro i r c
      exact (pack_sources packet cols i r c).trans (reads i r c)
    unfold SameSources at sameReads
    simp only [Launch.writeOrdered, Launch.writeWith, show p.group = packet from rfl, if_neg packed]
    unfold reduceZero
    split
    · simp only [sameReads]
    · exact unpacked.symm

open CudaStreamOrder in
/-- Once the stream has drained the captured region, replaying its issued
work yields the two output windows and the same values at all other addresses. -/
theorem issued_memory_eq {rows : Nat} (p : Inputs rows) (program : Program) (stream : Nat)
    {initial final : Queues Launch} {trace : List (Nat × Command Launch)}
    (issued : CudaStreamOrder.Run initial trace final)
    (enqueued : initial.pending stream = program.launches.map Command.work)
    (drained : final.pending stream = []) :
    ((issuedOn stream trace).foldl (fun m op =>
      match op with
      | .work work => work.write p m
      | .record _ | .wait _ => m) p.group.memory) = Program.separate.run rows p := by
  rw [fifo_complete issued stream drained, enqueued, List.foldl_map]
  change program.run rows p = _
  cases program
  · rfl
  · exact congrFun (congrFun run_eq rows) p
  · exact congrFun (congrFun (single_run_eq.trans run_eq) rows) p

def same (a b : Program) : Prop := a.run = b.run

instance : Spec.ProgramSyntax Program where
  Statement := Launch
  Signature := Unit
  signature := fun _ => ()
  body := Program.launches
  structural := some same

specification fused_equiv : Program.fused ≡[([] : Spec.Assumptions Launch)] Program.separate := by
  exact Spec.FloatingPoint.ofStructural (structural := same) rfl rfl run_eq

specification single_equiv : Program.single ≡[([] : Spec.Assumptions Launch)] Program.separate := by
  exact Spec.FloatingPoint.ofStructural (structural := same) rfl rfl (single_run_eq.trans run_eq)

#axiomsClean fused_block_eq
#axiomsClean run_ordered_eq
#axiomsClean issued_memory_eq
#axiomsClean pack_fma_eq
#axiomsClean Program.runSelected_eq
#print_fp_assumptions fused_equiv
#axiomsClean fused_equiv
#print_fp_assumptions single_equiv
#axiomsClean single_equiv
end VeriTile.Bench.Models.NanoGPT.Track1.MuddFusedPacking
