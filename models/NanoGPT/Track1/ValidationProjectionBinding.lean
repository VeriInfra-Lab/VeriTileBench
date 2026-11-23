import models.NanoGPT.Track1.ValidationBatchBinding

/-! Decode the captured TorchInductor wrapper which supplies the eight
BF16 projection ports and aliases the FP32 output slabs. The matrix
operation is the same per-slab call used by the source evaluation. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationProjectionBinding
open ValidationCE (slabCount slabRows)

open ValidationCE (channels vocabulary BF16Matrix Matmul)

structure Slab where
  activationOffset : Nat
  lossOffset : Nat
  deriving DecidableEq, Repr

def reference : List Slab :=
  (List.range slabCount).map fun i => ⟨i * slabRows * channels, i * slabRows⟩

private def entryName (ptx : String) : Option String := do
  let [entry] := (QKVTTIR.normalizePTX ptx).filter (·.startsWith ".visible .entry ") | none
  let [_, _, name] := entry.splitOn " " | none
  guard (name.endsWith "(")
  return (name.dropEnd 1).toString

private def callBody (source : String) : Option (List String) := do
  let lines := source.splitOn "\n"
  let signature := "    def call(self, args):"
  let [start] := (lines.zipIdx.filter (fun (line, _) => line == signature)).map Prod.snd | none
  let [runner] := (lines.zipIdx.filter (fun (line, _) => line == "class Runner:")).map Prod.snd | none
  guard (runner < start && ((lines.drop (runner + 1)).take (start - runner - 1)).all fun line =>
    line.isEmpty || line.startsWith " ")
  let code := (lines.drop (start + 1)).takeWhile (fun line => line.isEmpty || line.startsWith "        ")
  let statements := ((code.map fun line => (line.splitOn "#").headD "").filter fun text =>
    !text.trimAscii.toString.isEmpty)
  let indent := fun (line : String) => (line.toList.takeWhile (· == ' ')).length
  guard ((statements.take 8).all (fun line => indent line == 8) &&
    ((statements.drop 8).dropLast).all (fun line => indent line == 12) &&
    statements.getLast?.map indent == some 8)
  return statements.map (·.trimAscii.toString)

private def buffer (name : String) : Option String := do
  guard (name.startsWith "buf")
  let index ← (name.drop 3).toString.toNat?
  guard (name == s!"buf{index}")
  return name

private def projectionCall (lines : List String) : Option (String × Nat) := do
  let [allocation, multiply] := lines | none
  let [lhs, allocate] := allocation.splitOn " = " | none
  let output ← buffer lhs
  guard (allocate == "empty_strided_cuda((32768, 50304), (50304, 1), torch.bfloat16)")
  let start := "extern_kernels.mm(reinterpret_tensor(arg0_1, (32768, 768), (768, 1), "
  let suffix := s!"), arg1_1, out={output})"
  guard (multiply.startsWith start && multiply.endsWith suffix)
  let offset ← ((multiply.drop start.length).dropEnd suffix.length).toString.toNat?
  guard (multiply == start ++ toString offset ++ suffix)
  return (output, offset)

private def lossAlias (root line : String) : Option (String × Nat) := do
  let [lhs, view] := line.splitOn " = " | none
  let output ← buffer lhs
  let start := s!"reinterpret_tensor({root}, (32768, ), (1, ), "
  guard (view.startsWith start && view.endsWith ")")
  let offset ← ((view.drop start.length).dropEnd 1).toString.toNat?
  guard (view == start ++ toString offset ++ ")")
  return (output, offset)

private def header : List String :=
  ["arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args", "args.clear()",
   "assert_size_stride(arg0_1, (262144, 768), (768, 1))",
   "assert_size_stride(arg1_1, (768, 50304), (50304, 1))",
   "assert_size_stride(arg2_1, (262144, ), (1, ))",
   "assert_size_stride(arg3_1, (50304, 6288), (6288, 1))",
   "assert_size_stride(arg4_1, (262144, ), (1, ))",
   "with torch.cuda._DeviceGuard(0):", "torch.cuda.set_device(0)"]

def decode (source ptx : String) : Option (List Slab) := do
  let kernel ← entryName ptx
  let body ← callBody source
  guard (body.take header.length == header)
  let work := body.drop header.length
  let projections ← (List.range slabCount).mapM fun i => projectionCall ((work.drop (i * 2)).take 2)
  let "del arg0_1" :: "del arg1_1" :: allocation :: tail := work.drop (2 * slabCount) | none
  let [root, "empty_strided_cuda((262144, ), (1, ), torch.float32)"] := allocation.splitOn " = " | none
  let root ← buffer root
  let outputs ← (tail.take slabCount).mapM (lossAlias root)
  let keys := root :: (projections ++ outputs).map Prod.fst
  guard (keys.eraseDups.length == 1 + 2 * slabCount && keys.length == 1 + 2 * slabCount)
  let "stream0 = get_raw_stream(0)" :: launch :: finish := tail.drop slabCount | none
  let start := kernel ++ ".run("
  guard (launch.startsWith start && launch.endsWith ")")
  let args := ((launch.drop start.length).dropEnd 1).toString.splitOn ", "
  guard (args.length == 2 * slabCount + 6 &&
    args.take 2 == ["arg2_1", "arg3_1"] && args[2 + slabCount]? == some "arg4_1" &&
    args.drop (3 + 2 * slabCount) == ["32768", "50304", "stream=stream0"])
  let drops := ["arg2_1", "arg3_1", "arg4_1"] ++ projections.map Prod.fst
  guard (finish == drops.map (fun name => "del " ++ name) ++ [s!"return ({root}, )"])
  let projectionNames := (args.drop 2).take slabCount
  let outputNames := (args.drop (3 + slabCount)).take slabCount
  (projectionNames.zip outputNames).mapM fun (projected, returned) => do
    return ⟨← projections.lookup projected, ← outputs.lookup returned⟩

structure Certificate (source ptx : String) : Type where
  decoded : decode source ptx = some reference

def certify (source ptx : String) : Option (Certificate source ptx) :=
  if h : decode source ptx = some reference then some ⟨h⟩ else none

def projection (multiply : Matmul) (activation : Nat → TailNorm.BF16)
    (weight : BF16Matrix channels vocabulary) (route : Slab) : BF16Matrix slabRows vocabulary :=
  multiply (fun row chan => activation (route.activationOffset + row.val * channels + chan.val)) weight

private def routedProjection (layout : Option (List Slab)) (multiply : Matmul)
    (activation : Nat → TailNorm.BF16) (weight : BF16Matrix channels vocabulary)
    (port : Nat) (row : Fin slabRows) (column : Fin vocabulary) : Option TailNorm.BF16 := do
  let routes ← layout
  guard (2 ≤ port)
  let route ← routes[port - 2]?
  return projection multiply activation weight route row column

def decodedProjection (source ptx : String) (multiply : Matmul)
    (activation : Nat → TailNorm.BF16) (weight : BF16Matrix channels vocabulary)
    (port : Nat) (row : Fin slabRows) (column : Fin vocabulary) : Option TailNorm.BF16 :=
  routedProjection (decode source ptx) multiply activation weight port row column

/-- Launch port 2 through 9 receives the BF16 product of its own
activation slice and the common head weight. The external matmul is
called with the same matrices as the sliced source evaluation. -/
theorem projection_port_exact {source ptx : String} (artifact : Certificate source ptx)
    (multiply : Matmul) (activation : Nat → TailNorm.BF16) (weight : BF16Matrix channels vocabulary)
    (slab : Fin slabCount) (row : Fin slabRows) (column : Fin vocabulary) :
    decodedProjection source ptx multiply activation weight (2 + slab.val) row column =
      some (multiply (fun localRow chan =>
        activation ((slab.val * slabRows + localRow.val) * channels + chan.val)) weight row column) := by
  unfold decodedProjection routedProjection
  rw [artifact.decoded]
  have enough : 2 ≤ 2 + slab.val := Nat.le_add_right _ _
  simp only [bind, Option.bind_some, guard, enough, if_true, Nat.add_sub_cancel_left,
    reference, List.getElem?_map, List.getElem?_range slab.isLt, Option.map_some]
  unfold projection
  simp only [Nat.add_mul]
  rfl

def decodedLossIndex (source ptx : String) (slab : Fin slabCount) (row : Fin slabRows) : Option Nat := do
  let routes ← decode source ptx
  let route ← routes[slab.val]?
  return route.lossOffset + row.val

/-- The FP32 aliases put the CTA losses in the order consumed by the
batch mean, including when Inductor schedules the projections out of order. -/
theorem loss_alias_exact {source ptx : String} (artifact : Certificate source ptx)
    (slab : Fin slabCount) (row : Fin slabRows) :
    decodedLossIndex source ptx slab row = some (ValidationCE.slabRow slab row) := by
  unfold decodedLossIndex
  rw [artifact.decoded]
  simp only [bind, Option.bind_some, reference, List.getElem?_map, List.getElem?_range slab.isLt, Option.map_some]
  rfl

open VeriTile.Triton FP

private def readWord (layout : Option (List Slab)) (multiply : Matmul) (activation : Nat → TailNorm.BF16)
    (weight : BF16Matrix channels vocabulary) (port : Nat) (byte : Int) : Option TailNorm.BF16 :=
  let i := byte.toNat / 2
  if h : i / vocabulary < slabRows then
    routedProjection layout multiply activation weight port ⟨i / vocabulary, h⟩
      ⟨i % vocabulary, Nat.mod_lt _ (by decide)⟩
  else none

private theorem read_word_exact {source ptx : String} (artifact : Certificate source ptx)
    (multiply : Matmul) (activation : Nat → TailNorm.BF16) (weight : BF16Matrix channels vocabulary)
    (slab : Fin slabCount) (row : Fin slabRows) (column : Fin vocabulary) :
    readWord (decode source ptx) multiply activation weight (2 + slab.val)
        (2 * (row.val * vocabulary + column.val)) =
      some (multiply (fun r k => activation ((slab.val * slabRows + r.val) * channels + k.val))
        weight row column) := by
  have address : (2 * (row.val * vocabulary + column.val) : Int).toNat / 2 =
      row.val * vocabulary + column.val := by
    change ((↑(2 * (row.val * vocabulary + column.val) : Nat) : Int).toNat) / 2 = _
    rw [Int.toNat_natCast]
    exact Nat.mul_div_cancel_left _ (by decide)
  have quotient : (row.val * vocabulary + column.val) / vocabulary = row.val := by
    rw [Nat.mul_comm row.val, Nat.mul_add_div (by decide : 0 < vocabulary),
      Nat.div_eq_of_lt column.isLt, Nat.add_zero]
  have offset : (row.val * vocabulary + column.val) % vocabulary = column.val := by
    rw [Nat.mul_comm row.val, Nat.mul_add_mod, Nat.mod_eq_of_lt column.isLt]
  unfold readWord
  simp only [address, quotient, offset, dif_pos row.isLt]
  exact projection_port_exact artifact multiply activation weight slab row column

def ports (source ptx : String) (multiply : Matmul) (activation : Nat → TailNorm.BF16)
    (weight : BF16Matrix channels vocabulary) (mask : Nat → UInt8) (device : TailNorm.Device) :
    ValidationBatchBinding.Ports :=
  let layout := decode source ptx
  ⟨fun port byte => (readWord layout multiply activation weight port byte).getD (Value.ofBits .bf16 0),
    mask, device⟩

def sourceProjection (tile : ValidationCE.Tile) (multiply : Matmul) (activation : Nat → TailNorm.BF16)
    (weight : BF16Matrix channels vocabulary) (mask : Nat → UInt8) (device : TailNorm.Device) :
    ValidationCE.Projection :=
  ValidationCE.HeadInput.projection ⟨multiply, activation, weight, mask, device⟩ tile

/-- The decoded wrapper supplies the source projection at every row and
column consumed by the checked loss. Both use the same BF16 slab GEMMs. -/
theorem projection_exact {source ptx : String} (artifact : Certificate source ptx)
    (tile : ValidationCE.Tile) (multiply : Matmul) (activation : Nat → TailNorm.BF16)
    (weight : BF16Matrix channels vocabulary) (mask : Nat → UInt8) (device : TailNorm.Device) :
    (ports source ptx multiply activation weight mask device).projection tile =
      sourceProjection tile multiply activation weight mask device := by
  unfold ValidationBatchBinding.Ports.projection sourceProjection ValidationCE.HeadInput.projection
  congr 1
  funext i column
  let loc := ValidationBatchBinding.locate tile i
  let row : Fin slabRows := ⟨i.val % slabRows, Nat.mod_lt _ (by decide)⟩
  have localEq : loc.block * ValidationReduceBinding.rowCount tile + loc.row.val = row.val :=
    ValidationBatchBinding.locate_slab_row tile i
  change (readWord (decode source ptx) multiply activation weight (2 + loc.slab.val)
    (2 * ((loc.block * ValidationReduceBinding.rowCount tile + loc.row.val) * vocabulary + column.val))).getD
      (Value.ofBits .bf16 0) = _
  have localInt : ((loc.block : Int) * ValidationReduceBinding.rowCount tile + loc.row.val) = row.val := by
    exact_mod_cast localEq
  rw [localInt, read_word_exact artifact multiply activation weight loc.slab row column]
  rfl

/-- Compose the wrapper's captured BF16 views with the compiled loss
and the driver's local mean, using the source's token binding. -/
theorem compiled_head_batch_mean_exact {source ptx : String} {tile : ValidationCE.Tile}
    (wrapper : Certificate source ptx)
    (compiled : ValidationReduceBinding.Certificate ptx tile)
    (scan : ValidationScanBinding.Certificate ptx tile) (native : ValidationCE.Native)
    (multiply : Matmul) (activation : Nat → TailNorm.BF16) (weight : BF16Matrix channels vocabulary)
    (mask : Nat → UInt8) (device : TailNorm.Device) (inputs : Array UInt32) (targets : Array UInt64)
    (batch : ValidationCE.ProjectedBatch)
    (bound : (sourceProjection tile multiply activation weight mask device).bindTokens inputs targets = some batch) :
    let captured := ports source ptx multiply activation weight mask device
    let layout := (ValidationCE.bound_tokens_are_source _ _ _ _ bound).1.trans
      (projection_exact wrapper tile multiply activation weight mask device).symm
    ValidationBatchBinding.runMean compiled scan native captured batch layout =
      some (ValidationLoss.batchMean (ValidationCE.losses native batch)) := by
  have capture : ((ports source ptx multiply activation weight mask device).projection tile).bindTokens
      inputs targets = some batch := by
    rw [projection_exact wrapper tile multiply activation weight mask device]
    exact bound
  exact ValidationBatchBinding.decoded_batch_mean_exact compiled scan native _ inputs targets batch capture

structure HeadProgram (tile : ValidationCE.Tile) where
  source : String
  ptx : String
  wrapper : Certificate source ptx
  reduction : ValidationReduceBinding.Certificate ptx tile
  scan : ValidationScanBinding.Certificate ptx tile

def HeadProgram.check (tile : ValidationCE.Tile) (source ptx : String) : Option (HeadProgram tile) := do
  let wrapper ← certify source ptx
  let reduction ← ValidationReduceBinding.certify ptx tile
  let scan ← ValidationScanBinding.certifyFrom reduction.scan
  return ⟨source, ptx, wrapper, reduction, scan⟩

def HeadProgram.run {tile : ValidationCE.Tile} (program : HeadProgram tile) (native : ValidationCE.Native)
    (head : ValidationCE.HeadInput) (inputs : Array UInt32) (targets : Array UInt64) : Option TailNorm.F32 :=
  let captured := ports program.source program.ptx head.multiply head.activation head.weight head.mask head.device
  match bound : (captured.projection tile).bindTokens inputs targets with
  | none => none
  | some batch =>
    ValidationBatchBinding.runMean program.reduction program.scan native captured batch
      (ValidationCE.bound_tokens_are_source _ _ _ _ bound).1

theorem HeadProgram.run_eq {tile : ValidationCE.Tile} (program : HeadProgram tile) (native : ValidationCE.Native)
    (head : ValidationCE.HeadInput) (inputs : Array UInt32) (targets : Array UInt64) :
    program.run native head inputs targets = ValidationCE.runBatch native (.head tile head) inputs targets := by
  let captured := ports program.source program.ptx head.multiply head.activation head.weight head.mask head.device
  have projection : captured.projection tile = head.projection tile :=
    projection_exact program.wrapper tile head.multiply head.activation head.weight head.mask head.device
  unfold HeadProgram.run ValidationCE.runBatch
  change (match bound : (captured.projection tile).bindTokens inputs targets with
    | none => none
    | some batch => ValidationBatchBinding.runMean program.reduction program.scan native captured batch
        (ValidationCE.bound_tokens_are_source _ _ _ _ bound).1) = _
  change _ = (head.projection tile).bindTokens inputs targets >>= fun batch =>
    some (ValidationLoss.batchMean (ValidationCE.losses native batch))
  rw [← projection]
  split
  · rename_i bound
    rw [bound]
    rfl
  · rename_i batch bound
    calc
      _ = some (ValidationLoss.batchMean (ValidationCE.losses native batch)) :=
        ValidationBatchBinding.decoded_batch_mean_exact program.reduction program.scan native captured
          inputs targets batch bound
      _ = _ := by rw [bound]; rfl

abbrev Programs := (tile : ValidationCE.Tile) → HeadProgram tile

def runBatch (programs : Programs) : ValidationCE.BatchRun := fun native input tokens targets =>
  match input with
  | .head tile head => (programs tile).run native head tokens targets
  | .projected projection => ValidationCE.runBatch native (.projected projection) tokens targets

theorem runBatch_eq (programs : Programs) : runBatch programs = ValidationCE.runBatch := by
  funext native input tokens targets
  cases input
  · exact HeadProgram.run_eq _ _ _ _ _
  · rfl

#axiomsClean projection_port_exact
#axiomsClean loss_alias_exact
#axiomsClean projection_exact
#axiomsClean compiled_head_batch_mean_exact
#axiomsClean HeadProgram.run_eq
#axiomsClean runBatch_eq
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationProjectionBinding
