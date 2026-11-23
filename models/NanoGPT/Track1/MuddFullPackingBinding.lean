import models.NanoGPT.Track1.MuddFusedPackingBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddFullPacking

open MuddGradientPacking (parseLaunch baseOfColumn)

private def trimming (lines : Array String) : Array String :=
  lines.map fun (line : String) => line.trimAscii.toString

private def assignment (line : String) (starts : String) : Option String := do
  let [name, expr] := line.splitOn " = " | none
  if expr.startsWith starts then some name else none

def verifyFullWrapper (rows : Nat) (before fused : String) : Option Unit := do
  if rows != 16384 && rows != 40960 then return ← none
  let original := (before.splitOn "\n").toArray
  let candidate := (fused.splitOn "\n").toArray
  let raw := trimming original
  let after := trimming candidate
  if original.size != candidate.size then return ← none
  let groupFlag := "_mudd_column_reduction_0"
  let fusedFlag := "_mudd_fused_packing_0"
  let groupGuard := s!"if not {groupFlag}: "
  let fusedGuard := s!"if not {fusedFlag}: "
  let helper := "from track_1_short.perf.kernels.mudd_coefficients import try_grouped_gradients, try_fused_full_gradients"
  let imports := (List.range after.size).filter fun i => after[i]! == helper
  let [importIndex] := imports | none
  let setup := importIndex + 1
  let launch := setup + 1
  let changed := (List.range after.size).filter fun i => original[i]! != candidate[i]!
  let launches := changed.filter fun i => after[i]! == groupGuard ++ raw[i]!
  if launches.length != 9 || launches.head! != launch ||
      !raw[importIndex]!.startsWith "# Topologically Sorted Source Nodes: " then return ← none
  let (firstKernel, [_, firstOut, _, "12", firstStream]) ← parseLaunch raw[launch]! "triton_per_fused_squeeze_sum_" | none
  let base ← baseOfColumn rows 9 raw[launch - 3]! firstOut
  let ["stream", stream] := firstStream.splitOn "=" | none
  if !raw[setup]!.startsWith (stream ++ " = get_raw_stream(") then return ← none
  let mut inputs := []
  let mut views := []
  for column in [:9] do
    let i := launches[column]!
    if i != launch + 4 * column || raw[i - 1]! != raw[setup]! ||
        (column > 0 && !raw[i - 2]!.startsWith "# Topologically Sorted Source Nodes: ") then return ← none
    let (kernel, [input, out, count, "12", streamArg]) ← parseLaunch raw[i]! "triton_per_fused_squeeze_sum_" | none
    let root ← baseOfColumn rows (9 - column) raw[i - 3]! out
    if kernel != firstKernel || root != base || streamArg != firstStream || count != toString rows ||
        inputs.contains input || views.contains out then return ← none
    inputs := input :: inputs
    views := views ++ [out]
  let zeroIndex := launches.getLast! + 4
  let (zeroKernel, [zero, zeroView, zeroCount, "12", zeroStream]) ←
    parseLaunch raw[zeroIndex]! "triton_per_fused_squeeze_sum_" | none
  let zeroBase ← baseOfColumn rows 0 raw[zeroIndex - 3]! zeroView
  if zeroBase != base || zeroKernel == firstKernel || zeroCount != toString rows || inputs.contains zero ||
      views.contains zeroView || zeroStream != firstStream || raw[zeroIndex - 1]! != raw[setup]! then return ← none

  let scaleAllocation := zeroIndex + 1
  let scaleIndex := zeroIndex + 4
  let (_, [input, scaled, count, scaleStream]) ← parseLaunch raw[scaleIndex]! "triton_poi_fused_mul_view_" | none
  if input != base || count != toString (rows * 10) || scaleStream != firstStream ||
      raw[scaleAllocation]! != s!"{scaled} = empty_strided_cuda(({rows}, 10), (10, 1), torch.bfloat16)" ||
      raw[scaleIndex - 1]! != raw[setup]! then return ← none
  let cleanups := (raw.toList.drop (scaleIndex + 1)).take 10
  if cleanups != ((views ++ [zeroView]).map ("del " ++ ·)) then return ← none
  let projection := scaleIndex + 13
  let output ← assignment raw[projection - 2]! "empty_strided_cuda("
  if raw[projection - 2]! != s!"{output} = empty_strided_cuda(({rows}, 64), (64, 1), torch.bfloat16)" then return ← none
  let into := s!"extern_kernels.mm({scaled}, reinterpret_tensor("
  if !raw[projection]!.startsWith into then return ← none
  let weight := (((raw[projection]!.drop into.length).toString).splitOn ", ").head!
  if !weight.startsWith "primals_" || raw[projection]! !=
      s!"{into}{weight}, (10, 64), (64, 1), 896), out={output})" then return ← none

  let pack := projection + 5
  let (_, outputView :: packArgs) ← parseLaunch raw[pack]! "triton_poi_fused_add_mul_select_backward_squeeze_" | none
  let [view, args] := raw[projection + 1]!.splitOn " = reinterpret_tensor(" | none
  let packed := (args.splitOn ", ").head!
  if packArgs != inputs.reverse ++ [zero, toString (rows * 120), firstStream] ||
      args != s!"{packed}, (1, {rows}, 10, 12), ({rows * 120}, 120, 12, 1), 0); del {packed}  # reuse" ||
      raw[projection + 2]! != s!"{outputView} = {view}; del {view}  # reuse" ||
      raw[pack - 1]! != raw[setup]! then return ← none

  let prior : List String := raw.toList.take (importIndex - 1)
  let [baseView] := prior.filter (fun line => line.startsWith (base ++ " = ")) | none
  let [_, columnView] := baseView.splitOn " = reinterpret_tensor(" | none
  let columnRoot := (columnView.splitOn ", ").head!
  let columnLayout := s!"{columnRoot}, (1, {rows}, 10), ({rows * 10}, 10, 1), 0); del {columnRoot}  # reuse"
  let allocations (name call : String) :=
    prior.filter (fun (line : String) => line.startsWith (name ++ " = ")) == [name ++ " = " ++ call]
  let sourceLayout := s!"empty_strided_cuda((1, {rows}, 12, 1), ({rows * 12}, 12, 1, 1), torch.bfloat16)"
  let packedLayout := s!"empty_strided_cuda((1, {rows}, 120), ({rows * 120}, 120, 1), torch.bfloat16)"
  let columnsLayout := s!"empty_strided_cuda(({rows}, 10), (10, 1), torch.bfloat16)"
  let roots := [base, columnRoot, output, scaled, packed, view, outputView, weight, zero] ++ inputs ++ views ++ [zeroView]
  if roots.eraseDups.length != roots.length || columnView != columnLayout ||
      !allocations columnRoot columnsLayout || !allocations packed packedLayout ||
      !(zero :: inputs).all (allocations · sourceLayout) then return ← none

  let fallback := s!"try_grouped_gradients({firstKernel}, {rows}, 1, {base}, " ++
    "(" ++ String.intercalate ", " inputs ++ s!"), {stream})"
  let fusedCall := s!"try_fused_full_gradients({firstKernel}, {zeroKernel}, {rows}, {base}, {packed}, " ++
    "(" ++ String.intercalate ", " (zero :: inputs) ++ s!"), {stream})"
  let dispatch := raw[setup]! ++ s!"; {scaled} = {fusedCall}; {fusedFlag} = {scaled} is not None; " ++
    s!"{groupFlag} = {fusedFlag} or {fallback}"
  let protect := [zeroIndex, scaleAllocation, scaleIndex, pack]
  let mut expected := original
  for (i, text) in ([(importIndex, helper), (setup, dispatch)] ++
      launches.map (fun i => (i, groupGuard ++ raw[i]!)) ++ protect.map (fun i => (i, fusedGuard ++ raw[i]!))) do
    expected := expected.set! i ((original[i]!.takeWhile (· == ' ')).toString ++ text)
  if expected == candidate then some () else none


private def scaleOperations (tile : Nat) (layout : String) : List String :=
  let ty := fun dtype => s!"tensor<{tile}x{dtype}{layout}>"
  let int := ty "i32"
  let f32 := ty "f32"
  let bf16 := ty "bf16"
  let ptr := ty "!tt.ptr<bf16>"
  MuddActivationStorage.Binding.canonicalNames [
    s!"%gain = arith.constant dense<1.000000e-01> : {f32}",
    "%block = tt.get_program_id x : i32",
    s!"%first = arith.muli %block, {tile} : i32",
    s!"%lane = tt.make_range \{end = {tile} : i32, start = 0 : i32} : {int}",
    s!"%begin = tt.splat %first : i32 -> {int}",
    s!"%idx = arith.addi %begin, %lane : {int}",
    s!"%input = tt.splat %in_ptr0 : !tt.ptr<bf16> -> {ptr}",
    s!"%src = tt.addptr %input, %idx : {ptr}, {int}",
    s!"%value = tt.load %src : {ptr}",
    s!"%x = arith.extf %value : {bf16} to {f32}",
    s!"%scale = arith.mulf %x, %gain : {f32}",
    s!"%output = tt.splat %out_ptr0 : !tt.ptr<bf16> -> {ptr}",
    s!"%dst = tt.addptr %output, %idx : {ptr}, {int}",
    s!"%narrow = arith.truncf %scale : {f32} to {bf16}",
    s!"tt.store %dst, %narrow : {ptr}", "tt.return", "}", "}"]

private def scaleBody (tile : Nat) (ir layout : String) (moduleLines : List String) : Bool :=
  (do
    let lines ← QKVTTIR.normalizeTTIR ir
    let header := lines.drop moduleLines.length
    let args := "(!tt.ptr<bf16> {tt.divisibility = 16 : i32}"
    let first := s!"(%in_ptr0: {args.drop 1}, %out_ptr0: {args.drop 1}, %xnumel: i32 \{tt.divisibility = 16 : i32}) attributes \{noinline = false} \{"
    pure (lines.take moduleLines.length == moduleLines && header.head!.startsWith "tt.func public @triton_poi_fused_mul_view_" &&
      header.head!.endsWith first && MuddActivationStorage.Binding.canonicalNames header.tail! == scaleOperations tile layout)).getD false

def verifiesProjectionScaling (rows tile warps : Nat) (ttir ttgir : String) : Bool :=
  let perThread := if tile == 512 then 2 else 8
  let blocked := s!"#blocked = #ttg.blocked<\{sizePerThread = [{perThread}], threadsPerWarp = [32], warpsPerCTA = [{warps}], order = [0]}>"
  let moduleHeader := s!"module attributes \{\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = {warps} : i32, ttg.target = \"cuda:90\", \"ttg.threads-per-warp\" = 32 : i32} \{"
  ((tile, warps) == (512, 8) || (tile, warps) == (1024, 4)) &&
    (rows * 10) % tile == 0 && scaleBody tile ttir "" ["module {"] &&
    scaleBody tile ttgir ", #blocked" [blocked, moduleHeader]

end VeriTile.Bench.Models.NanoGPT.Track1.MuddFullPacking
