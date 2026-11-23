import models.NanoGPT.Track1.MuddGradientGrouping

/-! The sampled graph packs ten read-only gradient planes between two
batches of row reductions. Packing and the reductions write separate
BF16 allocations. The delayed reductions can be moved before packing
and joined to the early batch on the same stream. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientPacking
open VeriTile.Spec
open scoped VeriTile.Spec
open VeriTile.Triton.FP
open MuddGradientGrouping (BF16 F32 Memory)

def inWindow (base words addr : Nat) : Prop :=
  base ≤ addr ∧ addr < base + 2 * words

instance (base words addr : Nat) : Decidable (inWindow base words addr) := by
  unfold inWindow
  infer_instance

structure Packet (rows : Nat) where
  source : Fin 10 → Nat
  columnsBase : Nat
  packedBase : Nat
  memory : Memory
  reduce : (Fin 16 → F32) → F32
  tile : Nat
  warps : Nat
  columnsDisjoint : ∀ plane addr, inWindow columnsBase (rows * 10) addr →
    ¬ inWindow (source plane) (rows * 12) addr
  packingDisjoint : ∀ plane addr, inWindow packedBase (rows * 120) addr →
    ¬ inWindow (source plane) (rows * 12) addr
  outputsDisjoint : ∀ addr, inWindow columnsBase (rows * 10) addr →
    ¬ inWindow packedBase (rows * 120) addr

def sourceAddress {rows : Nat} (p : Packet rows) (plane : Fin 10) (row : Fin rows) (col : Fin 12) : Nat :=
  p.source plane + 2 * (row.val * 12 + col.val)

theorem source_in_window {rows : Nat} (p : Packet rows) (plane : Fin 10) (row : Fin rows) (col : Fin 12) :
    inWindow (p.source plane) (rows * 12) (sourceAddress p plane row col) := by
  have hr := row.isLt
  have hc := col.isLt
  unfold inWindow sourceAddress
  omega

def SameSources {rows : Nat} (p : Packet rows) (left right : Memory) : Prop :=
  ∀ plane row col, left (sourceAddress p plane row col) = right (sourceAddress p plane row col)

private def read {rows : Nat} (p : Packet rows) (m : Memory) (plane : Fin 10) (row : Fin rows)
    (column : Fin 12) : BF16 := m (sourceAddress p plane row column)

def reduction {rows : Nat} (p : Packet rows) (m : Memory) : MuddGradientGrouping.Packet rows 9 where
  input := fun plane row col => read p m plane.succ row col
  reduce := p.reduce
  base := p.columnsBase
  first := 1
  memory := m
  tile := p.tile
  warps := p.warps

private def allColumns := (List.finRange 9).reverse
def earlyColumns := allColumns.take 6
def lateColumns := allColumns.drop 6

def reduceColumns {rows : Nat} (p : Packet rows) (columns : List (Fin 9)) (m : Memory) : Memory := fun addr =>
  columns.foldl (MuddGradientGrouping.writeColumn (reduction p m) addr) (m addr)

def scalePacked (sum : F32) : BF16 :=
  cast .bf16 Config.ieeeValue (mul Config.ieeeValue sum (Value.ofBits .fp32 0x3dcccccd))

private def packScale (selected : Fin 10 → F32) : BF16 :=
  let sum := allColumns.foldl (fun total i => add Config.ieeeValue total (selected i.castSucc)) (selected 9)
  scalePacked sum

def packValue {rows : Nat} (p : Packet rows) (m : Memory) (row : Fin rows) (plane : Fin 10) (col : Fin 12) :
    BF16 := packScale fun i =>
  if plane = i then cast .fp32 Config.ieeeValue (read p m i row col) else Value.zero .fp32

def routedValue {rows : Nat} (p : Packet rows) (m : Memory) (row : Fin rows) (plane : Fin 10) (col : Fin 12) :
    BF16 :=
  let pointer := MuddGradientGrouping.selectPointer p.source 0 plane
  let loaded := cast .fp32 Config.ieeeValue (m (pointer + 2 * (row.val * 12 + col.val)))
  packScale fun i => if plane = i then loaded else Value.zero .fp32

theorem routed_eq {rows : Nat} (p : Packet rows) (m : Memory) (row : Fin rows) (plane : Fin 10) (col : Fin 12) :
    routedValue p m row plane col = packValue p m row plane col := by
  unfold routedValue packValue
  dsimp only
  rw [MuddGradientGrouping.select_pointer]
  congr 1
  funext i
  by_cases same : plane = i
  · subst i
    rfl
  · simp [same]

def writePacked {rows : Nat} (p : Packet rows) (m : Memory) (value : Fin rows → Fin 10 → Fin 12 → BF16) :
    Memory := fun addr =>
  if h : inWindow p.packedBase (rows * 120) addr ∧ (addr - p.packedBase) % 2 = 0 then
    let index := (addr - p.packedBase) / 2
    let row : Fin rows := ⟨index / 120, by unfold inWindow at h; omega⟩
    let plane : Fin 10 := ⟨index / 12 % 10, Nat.mod_lt _ (by decide)⟩
    let column : Fin 12 := ⟨index % 12, Nat.mod_lt _ (by decide)⟩
    value row plane column
  else m addr

def pack {rows : Nat} (p : Packet rows) (m : Memory) (routed : Bool := false) : Memory :=
  writePacked p m (if routed then routedValue p m else packValue p m)

theorem pack_routed_eq {rows : Nat} (p : Packet rows) (m : Memory) : pack p m true = pack p m := by
  funext addr
  simp [pack, writePacked, routed_eq]

theorem packing_frame {rows : Nat} (p : Packet rows) (m : Memory) (addr : Nat)
    (out : ¬ inWindow p.packedBase (rows * 120) addr) :
    pack p m false addr = m addr := by simp [pack, writePacked, out]

theorem columns_frame {rows : Nat} (p : Packet rows) (columns : List (Fin 9)) (m : Memory) (addr : Nat)
    (out : ¬ inWindow p.columnsBase (rows * 10) addr) :
    reduceColumns p columns m addr = m addr := by
  have h : MuddGradientGrouping.locate (reduction p m) addr = none := by
    unfold MuddGradientGrouping.locate MuddGradientGrouping.rowSlot
    dsimp [reduction]
    split
    · rename_i h
      have base := h.1
      have bound := h.2.2.1
      exfalso
      apply out
      unfold inWindow
      omega
    · rfl
  unfold reduceColumns MuddGradientGrouping.writeColumn
  simp only [h]
  induction columns with
  | nil => rfl
  | cons hd tl ih => exact ih

theorem pack_sources {rows : Nat} (p : Packet rows) (m : Memory) : SameSources p (pack p m) m := by
  intro plane row col
  exact packing_frame p _ _ (fun h => p.packingDisjoint plane _ h (source_in_window p plane row col))

theorem reduce_sources {rows : Nat} (p : Packet rows) (columns : List (Fin 9)) (m : Memory) :
    SameSources p (reduceColumns p columns m) m := by
  intro plane row col
  exact columns_frame p _ _ _ (fun h => p.columnsDisjoint plane _ h (source_in_window p plane row col))

private theorem same_reduceRow {rows : Nat} (p : Packet rows) (a b : Memory) (same : SameSources p a b)
    (plane : Fin 9) (row : Fin rows) :
    MuddGradientGrouping.reduceRow (reduction p a) plane row =
      MuddGradientGrouping.reduceRow (reduction p b) plane row := by
  have input : (reduction p a).input plane row = (reduction p b).input plane row := by
    funext col
    exact same plane.succ row col
  simp only [MuddGradientGrouping.reduceRow, input]
  rfl

private theorem same_column_write {rows : Nat} (p : Packet rows) (a b : Memory) (same : SameSources p a b)
    (addr : Nat) :
    MuddGradientGrouping.writeColumn (reduction p a) addr = MuddGradientGrouping.writeColumn (reduction p b) addr := by
  have locate : MuddGradientGrouping.locate (reduction p a) addr =
      MuddGradientGrouping.locate (reduction p b) addr := rfl
  funext prior plane
  simp only [MuddGradientGrouping.writeColumn, locate]
  split
  · rw [same_reduceRow p a b same]
  · rfl

theorem same_packValue {rows : Nat} (p : Packet rows) (a b : Memory) (same : SameSources p a b)
    (row : Fin rows) (plane : Fin 10) (col : Fin 12) :
    packValue p a row plane col = packValue p b row plane col := by
  unfold SameSources at same
  simp only [packValue, read, same]

theorem pack_reduce_commute {rows : Nat} (p : Packet rows) (columns : List (Fin 9)) (m : Memory) :
    reduceColumns p columns (pack p m) = pack p (reduceColumns p columns m) := by
  funext addr
  by_cases h : inWindow p.packedBase (rows * 120) addr
  · have out : ¬ inWindow p.columnsBase (rows * 10) addr := fun output => p.outputsDisjoint addr output h
    rw [columns_frame p _ _ _ out]
    by_cases aligned : (addr - p.packedBase) % 2 = 0
    · have inside : inWindow p.packedBase (rows * 120) addr ∧ (addr - p.packedBase) % 2 = 0 := ⟨h, aligned⟩
      simp only [pack, writePacked, dif_pos inside]
      exact same_packValue p _ _ (fun plane row col => (reduce_sources p columns m plane row col).symm) _ _ _
    · have outside : ¬ (inWindow p.packedBase (rows * 120) addr ∧ (addr - p.packedBase) % 2 = 0) :=
        fun bad => aligned bad.2
      simp only [pack, writePacked, dif_neg outside, columns_frame p _ _ _ out]
  · rw [packing_frame p _ _ h]
    change columns.foldl (MuddGradientGrouping.writeColumn (reduction p (pack p m)) addr) (pack p m false addr) =
      columns.foldl (MuddGradientGrouping.writeColumn (reduction p m) addr) (m addr)
    rw [packing_frame p _ _ h, same_column_write p _ _ (pack_sources p m)]

theorem join_columns {rows : Nat} (p : Packet rows) (m : Memory) :
    reduceColumns p lateColumns (reduceColumns p earlyColumns m) = MuddGradientGrouping.grouped (reduction p m) := by
  rw [MuddGradientGrouping.run_eq]
  funext addr
  change lateColumns.foldl (MuddGradientGrouping.writeColumn (reduction p (reduceColumns p earlyColumns m)) addr)
    (reduceColumns p earlyColumns m addr) = _
  rw [same_column_write p _ _ (reduce_sources p earlyColumns m)]
  unfold MuddGradientGrouping.separate
  change _ = allColumns.foldl _ (m addr)
  unfold reduceColumns
  rw [← List.foldl_append]
  rfl

inductive Program where
  | interleaved
  | hoisted

abbrev Run := (rows : Nat) → Packet rows → Memory

def Program.run (program : Program) : Run := fun _ p =>
  match program with
  | .interleaved => reduceColumns p lateColumns (pack p (reduceColumns p earlyColumns p.memory))
  | .hoisted => pack p (MuddGradientGrouping.grouped (reduction p p.memory)) true

theorem run_eq : Program.hoisted.run = Program.interleaved.run := by
  funext rows p
  change pack p _ true = reduceColumns p lateColumns (pack p (reduceColumns p earlyColumns p.memory))
  rw [pack_routed_eq, pack_reduce_commute, join_columns]

def same (p q : Program) : Prop := p.run = q.run

instance : Spec.ProgramSyntax Program where
  Statement := Unit
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some same

specification hoisted_equiv :
    Program.hoisted ≡[([] : Spec.Assumptions Unit)] Program.interleaved := by
  exact Spec.FloatingPoint.ofStructural (structural := same) rfl rfl run_eq

def parseLaunch (line : String) (nameStart : String) : Option (String × List String) := do
  if !line.startsWith nameStart || !line.endsWith ")" then return ← none
  let [name, args] := ((line.dropEnd 1).toString).splitOn ".run(" | none
  pure (name, args.splitOn ", ")

def baseOfColumn (rowCount column : Nat) (alias dst : String) : Option String := do
  let [left, args] := alias.splitOn " = reinterpret_tensor(" | none
  let base := (args.splitOn ", ").head!
  let ending := s!"{base}, (1, {rowCount}, 1), ({rowCount * 10}, 10, 1), {column})  # alias"
  if left == dst && args == ending then some base else none

private structure Wrapper where
  setup : Nat
  packLaunch : Nat
  zeroInput : String
  lastColumnLaunch : Nat
  sourcePorts : List String
  packedRoot : String
  columns : String
  reduction : String
  stream : String

private def verifyWrapper (rows : Nat) (before after : String) : Option Wrapper := do
  let oldLines : Array String := (before.splitOn "\n").toArray
  let newLines : Array String := (after.splitOn "\n").toArray
  let old : Array String := oldLines.map (fun (line : String) => (String.trimAscii line).toString)
  let new : Array String := newLines.map (fun (line : String) => (String.trimAscii line).toString)
  if old.size != new.size then return ← none
  let helper := "from track_1_short.perf.kernels.mudd_coefficients import try_grouped_gradients"
  let flag := "_mudd_column_reduction_0"
  let mut launches : Array Nat := #[]
  let mut setupIndex : Option Nat := none
  let mut importIndex : Option Nat := none
  for i in [:old.size] do
    let a := old[i]!
    let b := new[i]!
    if oldLines[i]! == newLines[i]! then continue
    let indent := (oldLines[i]!.takeWhile (· == ' ')).toString
    if newLines[i]! != indent ++ b then return ← none
    if b == helper && a.startsWith "# Topologically Sorted Source Nodes: " && importIndex.isNone then
      importIndex := some i
    else if b.startsWith (a ++ s!"; {flag} = try_grouped_gradients(") && setupIndex.isNone then
      setupIndex := some i
    else if b == (s!"if not {flag}: " ++ a) then
      launches := launches.push i
    else return ← none
  let setup ← setupIndex
  let imported ← importIndex
  if launches.size != 9 || setup + 1 != launches[0]! || imported + 1 != setup then return ← none
  let mut kernel := ""
  let mut base := ""
  let mut stream := ""
  let mut sources : List String := []
  for j in [:launches.size] do
    let i := launches[j]!
    let (name, [src, dst, rowCount, "12", streamArg]) ← parseLaunch old[i]! "triton_per_fused_squeeze_sum_"
      | none
    let ["stream", streamName] := streamArg.splitOn "=" | none
    let output ← baseOfColumn rows (9 - j) old[i - 3]! dst
    if rowCount != toString rows || streamName.isEmpty || !old[i - 1]!.startsWith
        (streamName ++ " = get_raw_stream(") || old[i - 1]! != old[setup]! then return ← none
    if j == 0 then
      kernel := name
      base := output
      stream := streamName
    if name != kernel || output != base || streamName != stream || src == base || sources.contains src then
      return ← none
    if j != 0 && j != 6 then
      let previous := launches[j - 1]!
      let afterPrevious := i - 3
      if afterPrevious != previous + 1 &&
          !(afterPrevious == previous + 2 && old[previous + 1]! == "del " ++ sources.head!) then return ← none
    if !old[i - 2]!.startsWith "# Topologically Sorted Source Nodes: " then return ← none
    sources := src :: sources
  let route := s!"{old[setup]!}; {flag} = try_grouped_gradients({kernel}, {rows}, 1, {base}, " ++
    "(" ++ String.intercalate ", " sources ++ s!"), {stream})"
  if route != new[setup]! then return ← none
  let middle := (old.toList.drop (launches[5]! + 1)).take (launches[6]! - launches[5]! - 1)
  let commands := middle.filter fun (line : String) => !line.isEmpty && !line.startsWith "#"
  if commands.length != 12 then return ← none
  let packCall := commands[3]!
  let (_, output :: args) ← parseLaunch packCall "triton_poi_fused_add_mul_select_backward_squeeze_" | none
  if args.take 9 != sources.reverse || args.length != 12 || args[10]! != toString (rows * 120) ||
      args[11]! != s!"stream={stream}" || output == base || sources.contains output ||
      args[9]! == base || args[9]! == output || sources.contains args[9]! then return ← none
  let [view, packView] := commands[0]!.splitOn " = reinterpret_tensor(" | none
  let packedRoot := (packView.splitOn ", ").head!
  let packedLayout := s!"{packedRoot}, (1, {rows}, 10, 12), ({rows * 120}, 120, 12, 1), 0); del {packedRoot}  # reuse"
  let beforePack := [commands[0]!, s!"{output} = {view}; del {view}  # reuse", old[setup]!, packCall]
  let releases := (sources.reverse.take 6).map ("del " ++ ·)
  if packView != packedLayout || commands != beforePack ++ releases ++ [old[launches[6]! - 3]!, old[setup]!] then
    return ← none
  let prior : List String := old.toList.take (launches[0]! - 3)
  let [columnsView] := prior.filter (fun (line : String) => line.startsWith (base ++ " = ")) | none
  let [_, columnsArgs] := columnsView.splitOn " = reinterpret_tensor(" | none
  let columnsRoot := (columnsArgs.splitOn ", ").head!
  let columnsLayout := s!"{columnsRoot}, (1, {rows}, 10), ({rows * 10}, 10, 1), 0); del {columnsRoot}  # reuse"
  let hasAllocation (name call : String) :=
    prior.filter (fun (line : String) => line.startsWith (name ++ " = ")) == [name ++ " = " ++ call]
  let sourceLayout := s!"empty_strided_cuda((1, {rows}, 12, 1), ({rows * 12}, 12, 1, 1), torch.bfloat16)"
  let packedAllocation := s!"empty_strided_cuda((1, {rows}, 120), ({rows * 120}, 120, 1), torch.bfloat16)"
  let columnsAllocation := s!"empty_strided_cuda(({rows}, 10), (10, 1), torch.bfloat16)"
  let allInputs := args.take 10
  if columnsArgs != columnsLayout || !(hasAllocation columnsRoot columnsAllocation) ||
      !(hasAllocation packedRoot packedAllocation) || !(allInputs.all (hasAllocation · sourceLayout)) ||
      ((columnsRoot :: packedRoot :: allInputs).eraseDups).length != 12 then return ← none
  let packLaunch := launches[5]! + 1 + (middle.takeWhile (· != packCall)).length
  pure ⟨setup, packLaunch, args[9]!, launches[8]!, sources, packedRoot, base, kernel, stream⟩

private def verifyFusedWrapper (rows : Nat) (before grouped fused : String) : Option Unit := do
  let w ← verifyWrapper rows before grouped
  let lines := (grouped.splitOn "\n").toArray
  let old := lines.map fun line => line.trimAscii.toString
  let last := w.lastColumnLaunch
  let zeroLaunch := last + 5
  let (kernel, [src, dst, rowCount, "12", streamArg]) ←
    parseLaunch old[zeroLaunch]! "triton_per_fused_squeeze_sum_" | none
  let base ← baseOfColumn rows 0 old[zeroLaunch - 3]! dst
  if src != w.zeroInput || base != w.columns || kernel == w.reduction || rowCount != toString rows ||
      streamArg != s!"stream={w.stream}" || old[zeroLaunch - 1]! != (old[w.setup]!.splitOn "; ").head! ||
      old[last + 1]! != "del " ++ w.sourcePorts.head! ||
      !old[zeroLaunch - 2]!.startsWith "# Topologically Sorted Source Nodes: " then return ← none
  let fuseFlag := "_mudd_fused_packing_0"
  let groupFlag := "_mudd_column_reduction_0"
  let [stream, group] := lines[w.setup]!.splitOn s!"; {groupFlag} = " | none
  let ports := String.intercalate ", " (src :: w.sourcePorts)
  let call := s!"try_fused_gradients({w.reduction}, {kernel}, {rows}, {base}, {w.packedRoot}, ({ports}), {w.stream})"
  let setup := s!"{stream}; {fuseFlag} = {call}; {groupFlag} = {fuseFlag} or {group}"
  let guard := fun i => (lines[i]!.takeWhile (· == ' ')).toString ++ s!"if not {fuseFlag}: " ++ old[i]!
  let expected := (lines.set! (w.setup - 1) (lines[w.setup - 1]! ++ ", try_fused_gradients"))
    |>.set! w.setup setup |>.set! w.packLaunch (guard w.packLaunch) |>.set! zeroLaunch (guard zeroLaunch)
  if expected.toList == fused.splitOn "\n" then some () else none

def checksFusedWrapper (rows : Nat) (before grouped fused : String) : Bool :=
  (verifyFusedWrapper rows before grouped fused).isSome

/-- Independently validate the launch-site edit in the captured wrapper.
The nine original launches must be guarded by a single call with the
same stream and reversed input order. The packer stays between the
original groups and still reads their inputs and a tenth plane. -/
def checksWrapper (rows : Nat) (before after : String) : Bool :=
  (verifyWrapper rows before after).isSome

#print_fp_assumptions hoisted_equiv
#axiomsClean pack_reduce_commute
#axiomsClean join_columns
#axiomsClean routed_eq
#axiomsClean hoisted_equiv
end VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientPacking
