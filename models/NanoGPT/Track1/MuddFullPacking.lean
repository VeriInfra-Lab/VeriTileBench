import models.NanoGPT.Track1.MuddFusedPacking

/-! MUDD backward for the full-vocabulary graphs. The two reduction
autotuners determine whether the columns share one launch. The launch
writes the rounded column sums, their BF16 projection operand, and the
packed gradient to disjoint windows. The unchanged projection consumes
the same BF16 words before the driver resumes the backward graph. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddFullPacking
open MuddGradientPacking MuddFusedPacking
open MuddGradientGrouping (Memory BF16 F32)
open VeriTile.Triton.FP

theorem zero_pack_commute {rows : Nat} (p : Packet rows) (reduce : (Fin 16 → F32) → F32) (m : Memory) :
    reduceZero p reduce (pack p m) = pack p (reduceZero p reduce m) := by
  funext addr
  by_cases inPack : inWindow p.packedBase (rows * 120) addr
  · have outside : ¬ inWindow p.columnsBase (rows * 10) addr := fun h => p.outputsDisjoint _ h inPack
    rw [zero_frame p reduce _ _ outside]
    simp only [pack, writePacked, Bool.false_eq_true, if_false]
    split
    · exact same_packValue p _ _ (fun i r c => (zero_sources p reduce m i r c).symm) _ _ _
    · exact (zero_frame p reduce _ _ outside).symm
  · rw [packing_frame p _ _ inPack]
    unfold reduceZero
    split
    · have reads := pack_sources p m
      unfold SameSources at reads
      simp only [reads]
    · exact packing_frame p _ _ inPack

def reduceAll {rows : Nat} (p : Inputs rows) : Memory :=
  reduceZero p.group p.zeroReduce (MuddGradientGrouping.grouped (reduction p.group p.group.memory))

theorem packed_after_reductions {rows : Nat} (p : Inputs rows) :
    Program.single.run rows p = pack p.group (reduceAll p) := by
  rw [single_run_eq, MuddFusedPacking.run_eq, separate_run]
  simp only [MuddGradientPacking.Program.run, reduceAll]
  rw [pack_reduce_commute, join_columns, zero_pack_commute]

structure ScalePort {rows : Nat} (p : Packet rows) where
  base : Nat
  columnsDisjoint : ∀ addr, inWindow base (rows * 10) addr → ¬ inWindow p.columnsBase (rows * 10) addr
  packedDisjoint : ∀ addr, inWindow base (rows * 10) addr → ¬ inWindow p.packedBase (rows * 120) addr
  sourcesDisjoint : ∀ plane addr, inWindow base (rows * 10) addr → ¬ inWindow (p.source plane) (rows * 12) addr

structure ScaledProjection {rows : Nat} (p : Packet rows) where
  scale : ScalePort p
  stores : Nat → Bool
  result : (Fin (rows * 10) → BF16) → Nat → BF16
  packedIndependent : ∀ addr, stores addr = true → ¬ inWindow p.packedBase (rows * 120) addr
  inputsIndependent : ∀ plane addr, stores addr = true → ¬ inWindow (p.source plane) (rows * 12) addr

/-- The captured projection consumes the scaled BF16 operand, and its weight
operand is fixed by the unchanged wrapper call. -/
def scaledProjection {rows : Nat} (p : Packet rows) (project : ScaledProjection p) (m : Memory) : Memory := fun addr =>
  if project.stores addr then project.result (fun index => m (project.scale.base + 2 * index.val)) addr else m addr

def scaleWord (word : BF16) : BF16 :=
  let cfg := Config.ieeeValue
  cast .bf16 cfg (mul cfg (cast .fp32 cfg word) (Value.ofBits .fp32 0x3dcccccd))

def scaleColumns {rows : Nat} (p : Inputs rows) (scaled : Nat) (m : Memory) : Memory := fun addr =>
  if inWindow scaled (rows * 10) addr ∧ (addr - scaled) % 2 = 0 then
    scaleWord (m (p.group.columnsBase + (addr - scaled)))
  else m addr

theorem scale_preserves_columns {rows : Nat} (p : Inputs rows) (port : ScalePort p.group)
    (m : Memory) (addr : Nat) (column : inWindow p.group.columnsBase (rows * 10) addr) :
    scaleColumns p port.base m addr = m addr := by
  have outside : ¬ inWindow port.base (rows * 10) addr :=
    fun scaled => port.columnsDisjoint addr scaled column
  simp [scaleColumns, outside]

private theorem scaling_commutes {rows : Nat} (p : Inputs rows) (port : ScalePort p.group) (m : Memory) :
    scaleColumns p port.base (pack p.group m) = pack p.group (scaleColumns p port.base m) := by
  funext addr
  unfold scaleColumns
  split
  · rename_i scaled
    have absent := port.packedDisjoint addr scaled.1
    rw [packing_frame _ _ _ absent]
    have readColumn : inWindow p.group.columnsBase (rows * 10) (p.group.columnsBase + (addr - port.base)) := by
      unfold inWindow at *
      omega
    rw [packing_frame _ _ _ (p.group.outputsDisjoint _ readColumn)]
    simp [scaled]
  · rename_i noScale
    unfold pack writePacked
    split
    · rename_i packed
      apply same_packValue
      intro plane row col
      have noRead : ¬ inWindow port.base (rows * 10) (sourceAddress p.group plane row col) := by
        intro h
        exact port.sourcesDisjoint plane _ h (source_in_window p.group plane row col)
      simp [noRead]
    · simp [noScale]

private theorem scaled_projection_commutes {rows : Nat} (p : Packet rows) (project : ScaledProjection p) (m : Memory) :
    scaledProjection p project (pack p m) = pack p (scaledProjection p project m) := by
  have scaledReads : (fun i : Fin (rows * 10) => pack p m false (project.scale.base + 2 * i.val)) =
      (fun i : Fin (rows * 10) => m (project.scale.base + 2 * i.val)) := by
    funext i
    apply packing_frame
    apply project.scale.packedDisjoint
    have := i.isLt
    unfold inWindow
    omega
  funext addr
  cases store : project.stores addr
  · simp only [scaledProjection, store, Bool.false_eq_true, if_false]
    unfold pack writePacked
    split
    · apply same_packValue
      intro i r c
      have noWrite : project.stores (sourceAddress p i r c) = false := by
        cases write : project.stores (sourceAddress p i r c)
        · rfl
        · exact (project.inputsIndependent i _ write (source_in_window p i r c)).elim
      simp [scaledProjection, noWrite]
    · simp [scaledProjection, store]
  · rw [packing_frame p _ _ (project.packedIndependent addr store)]
    simp only [scaledProjection, store, if_true, scaledReads]

def reduceAllSeparately {rows : Nat} (p : Inputs rows) : Memory :=
  reduceZero p.group p.zeroReduce
    (reduceColumns p.group lateColumns (reduceColumns p.group earlyColumns p.group.memory))

def fullReference {rows : Nat} (p : Inputs rows) (project : ScaledProjection p.group) : Memory :=
  pack p.group
    (scaledProjection p.group project (scaleColumns p project.scale.base (reduceAllSeparately p)))

private theorem run_fma_eq {rows : Nat} (program : MuddFusedPacking.Program) (p : Inputs rows) :
    program.runWithPacking packFma p = program.run rows p := by
  rw [pack_fma_eq]
  rfl

/-- The split launch writes only its own columns to the projection operand. -/
private def scaleBank {rows : Nat} (p : Inputs rows) (scaled : Nat) (zero : Bool) (m : Memory) : Memory :=
  fun addr => if ((addr - scaled) / 2 % 10 == 0) == zero then scaleColumns p scaled m addr else m addr

private theorem scaleBank_frame {rows : Nat} (p : Inputs rows) (scaled : Nat) (zero : Bool) (m : Memory)
    (addr : Nat) (outside : ¬ inWindow scaled (rows * 10) addr) : scaleBank p scaled zero m addr = m addr := by
  simp [scaleBank, scaleColumns, outside]

private theorem scaleBank_sources {rows : Nat} (p : Inputs rows) (port : ScalePort p.group)
    (zero : Bool) (m : Memory) : SameSources p.group (scaleBank p port.base zero m) m := by
  intro plane row col
  apply scaleBank_frame
  exact fun scaled => port.sourcesDisjoint plane _ scaled (source_in_window p.group plane row col)

private theorem scaleBanks_eq {rows : Nat} (p : Inputs rows) (port : ScalePort p.group) (m : Memory) :
    scaleBank p port.base true (scaleBank p port.base false m) = scaleColumns p port.base m := by
  funext addr
  by_cases zero : (addr - port.base) / 2 % 10 = 0
  · simp only [scaleBank, zero, beq_self_eq_true, ↓reduceIte]
    unfold scaleColumns
    split
    · rename_i aligned
      have column : inWindow p.group.columnsBase (rows * 10)
          (p.group.columnsBase + (addr - port.base)) := by
        unfold inWindow at *
        omega
      have outside : ¬ inWindow port.base (rows * 10) (p.group.columnsBase + (addr - port.base)) :=
        fun h => port.columnsDisjoint _ h column
      rw [scaleBank_frame p _ false m _ outside]
    · simp [scaleBank, zero]
  · simp [scaleBank, zero]

private def zeroPacked {rows : Nat} (p : Inputs rows) (m : Memory) : Memory :=
  let reduced := reduceZero p.group p.zeroReduce m
  fun addr => if isZeroPlane p.group addr == true then packFma p.group reduced addr else reduced addr

private theorem zeroPacked_frame {rows : Nat} (p : Inputs rows) (m : Memory) (addr : Nat)
    (outsidePack : ¬ inWindow p.group.packedBase (rows * 120) addr)
    (outsideZero : ¬ (inWindow p.group.columnsBase (rows * 10) addr ∧ (addr - p.group.columnsBase) % 20 = 0)) :
    zeroPacked p m addr = m addr := by
  simp [zeroPacked, packFma, writePacked, outsidePack, reduceZero, outsideZero]

private theorem zeroPacked_congr_at {rows : Nat} (p : Inputs rows) (a b : Memory) (addr : Nat)
    (same : SameSources p.group a b) (framed : a addr = b addr) :
    zeroPacked p a addr = zeroPacked p b addr := by
  have reduced : SameSources p.group (reduceZero p.group p.zeroReduce a) (reduceZero p.group p.zeroReduce b) := by
    intro plane row col
    exact (zero_sources _ _ _ plane row col).trans ((same plane row col).trans (zero_sources _ _ _ plane row col).symm)
  unfold SameSources at same reduced
  have atAddr : reduceZero p.group p.zeroReduce a addr = reduceZero p.group p.zeroReduce b addr := by
    unfold reduceZero
    split
    · simp only [same]
    · exact framed
  unfold zeroPacked
  split
  · unfold packFma writePacked
    split
    · simp only [reduced]
    · exact atAddr
  · exact atAddr

private theorem scaleBank_zeroPacked {rows : Nat} (p : Inputs rows) (port : ScalePort p.group) (m : Memory) :
    zeroPacked p (scaleBank p port.base false m) = scaleBank p port.base false (zeroPacked p m) := by
  funext addr
  by_cases early : (addr - port.base) / 2 % 10 ≠ 0 ∧ inWindow port.base (rows * 10) addr ∧ (addr - port.base) % 2 = 0
  · have scalePack := port.packedDisjoint addr early.2.1
    have scaleRaw := port.columnsDisjoint addr early.2.1
    have frame := zeroPacked_frame p (scaleBank p port.base false m) addr scalePack (fun h => scaleRaw h.1)
    rw [frame]
    let raw := p.group.columnsBase + (addr - port.base)
    have onRaw : inWindow p.group.columnsBase (rows * 10) raw := by
      have window := early.2.1
      dsimp [raw]
      unfold inWindow at *
      omega
    have notZero : (raw - p.group.columnsBase) % 20 ≠ 0 := by
      have notFirst := early.1
      dsimp [raw]
      omega
    have keepsRaw := zeroPacked_frame p m raw (p.group.outputsDisjoint raw onRaw) (fun h => notZero h.2)
    simp [scaleBank, scaleColumns, early.1, early.2.1, early.2.2]
    change scaleWord (m raw) = scaleWord (zeroPacked p m raw)
    rw [keepsRaw]
  · have noWrite (input : Memory) : scaleBank p port.base false input addr = input addr := by
      unfold scaleBank scaleColumns
      split
      · rename_i nonzero
        split
        · rename_i aligned
          have notFirst : (addr - port.base) / 2 % 10 ≠ 0 := by simpa using nonzero
          exact (early ⟨notFirst, aligned⟩).elim
        · rfl
      · rfl
    rw [noWrite]
    exact zeroPacked_congr_at p _ _ addr (scaleBank_sources p port false m) (noWrite m)

private def groupPacked {rows : Nat} (p : Inputs rows) : Memory :=
  Launch.groupAndPack.writeWith p packFma p.group.memory

private theorem splitLaunch_eq {rows : Nat} (p : Inputs rows) (port : ScalePort p.group) :
    scaleBank p port.base true (zeroPacked p (scaleBank p port.base false (groupPacked p))) =
      scaleColumns p port.base (MuddFusedPacking.Program.fused.runWithPacking packFma p) := by
  rw [scaleBank_zeroPacked, scaleBanks_eq]
  rfl

def fullOptimized {rows : Nat} (p : Inputs rows) (project : ScaledProjection p.group)
    (ready : Bool) : Memory :=
  if ready then
    let output :=
      if p.group.tile == p.zeroTile && p.group.warps == p.zeroWarps then
        scaleColumns p project.scale.base (MuddFusedPacking.Program.single.runWithPacking packFma p)
      else
        let group := scaleBank p project.scale.base false (groupPacked p)
        scaleBank p project.scale.base true (zeroPacked p group)
    scaledProjection p.group project output
  else fullReference p project

theorem full_optimized_eq {rows : Nat} (p : Inputs rows) (project : ScaledProjection p.group) (ready : Bool) :
    fullOptimized p project ready = fullReference p project := by
  have writes :
      scaledProjection p.group project (scaleColumns p project.scale.base (Program.single.run rows p)) =
        fullReference p project := by
    rw [packed_after_reductions, scaling_commutes, scaled_projection_commutes]
    unfold fullReference reduceAllSeparately reduceAll
    rw [join_columns]
  cases ready
  · rfl
  · by_cases same : (p.group.tile == p.zeroTile && p.group.warps == p.zeroWarps) = true
    · simpa only [fullOptimized, if_pos same, run_fma_eq] using writes
    · simpa only [fullOptimized, if_neg same, splitLaunch_eq, run_fma_eq, single_run_eq] using writes

/-- One CTA-uniform dispatch chooses the original reduction per plane while
issuing both banks from the same launch. -/
def fullJoined {rows : Nat} (p : Inputs rows) (project : ScaledProjection p.group) : Memory :=
  scaledProjection p.group project
    (scaleColumns p project.scale.base (MuddFusedPacking.Program.single.runWithPacking packFma p))

theorem full_joined_eq {rows : Nat} (p : Inputs rows) (project : ScaledProjection p.group) :
    fullJoined p project = fullReference p project := by
  let joined := {p with zeroTile := p.group.tile, zeroWarps := p.group.warps}
  simpa only [fullJoined, fullOptimized, joined, beq_self_eq_true, Bool.and_self, ↓reduceIte] using
    full_optimized_eq joined project true

#axiomsClean full_joined_eq

abbrev Run := (rows : Nat) → (p : Inputs rows) → ScaledProjection p.group → Memory

def reference : Run := fun _ p project => fullReference p project

def candidate : Run := fun _ p project => fullOptimized p project true

theorem run_eq : candidate = reference := by
  funext rows p project
  exact full_optimized_eq p project true

#axiomsClean scale_preserves_columns
#axiomsClean splitLaunch_eq
#axiomsClean run_fma_eq
#axiomsClean full_optimized_eq
#axiomsClean run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.MuddFullPacking
