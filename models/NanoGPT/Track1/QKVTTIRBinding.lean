import models.NanoGPT.Track1.QKVTTIR

/-! A fail-closed recognizer for the two emitted QKV amax TTIR fragments.
Templates keep the complete SSA graph, attributes, types and regions.
The inputs are the TTIR texts from the offline compiler replay. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVTTIR
open QKVAmax QKVLaunch

/-- Read the rest of a parenthesized location, including nested call sites.
Quotes keep parentheses inside an MLIR string from closing the location. -/
private def endLocation : List Char → Nat → Bool → Option (List Char)
  | [], _, _ => none
  | '\\' :: _ :: rest, n, true => endLocation rest n true
  | '"' :: rest, n, quoted => endLocation rest n (!quoted)
  | '(' :: rest, n, false => endLocation rest (n+1) false
  | ')' :: rest, 0, false => some rest
  | ')' :: rest, n+1, false => endLocation rest n false
  | _ :: rest, n, quoted => endLocation rest n quoted

private def dropLocations (fuel : Nat) (line : List Char) : Option (List Char) := do
  match fuel, line with
  | _, [] => some []
  | 0, _ => none
  | fuel+1, ' ' :: 'l' :: 'o' :: 'c' :: '(' :: rest =>
      dropLocations fuel (← endLocation rest 0 false)
  | fuel+1, c :: rest => return c :: (← dropLocations fuel rest)

private def locationAlias (s : String) : Bool :=
  s.startsWith "#loc" && ((s.drop 4).toString.toList.all Char.isDigit)

private def constantName (s : String) : Bool :=
  s.startsWith "%c" && s.endsWith "_i32" &&
    ((s.drop 2).dropEnd 4).toString.toNat?.isSome

private def replaceOperand (constants : List (String × String)) (s : String) : String :=
  if s.endsWith "," then
    let name := (s.dropEnd 1).toString
    match constants.lookup name with
    | some value => value ++ ","
    | none => s
  else
    (constants.lookup s).getD s

private def normalizeLines (lines : List String) : Option (List String) := do
  let mut constants : List (String × String) := []
  let mut result := []
  for line in lines do
    let raw := line.toList
    let clean ← dropLocations raw.length raw
    let tokens := ((String.ofList clean).splitOn " ").filter (!·.isEmpty)
    match tokens with
    | [] => pure ()
    | [alias, "="] =>
        if !locationAlias alias || !(line.startsWith (alias ++ " = loc(")) then return ← none
    | [name, "=", "arith.constant", value, ":", "i32"] =>
        if constantName name then
          if value.toNat?.isNone || (constants.lookup name).isSome then return ← none
          constants := (name, value) :: constants
        else
          result := String.intercalate " " tokens :: result
    | _ => result := String.intercalate " " (tokens.map (replaceOperand constants)) :: result
  return result.reverse

def normalizeTTIR (text : String) : Option (List String) :=
  normalizeLines (text.splitOn "\n")

/-- Keep instruction operands, labels and assembly directives, dropping
debug locations and the debug sections after the entry point. -/
def normalizePTX (text : String) : List String :=
  let code := ((text.splitOn "\n").map fun line =>
    let body := ((line.splitOn "//").headD "").replace "\t" " "
    String.intercalate " " ((body.splitOn " ").filter (!·.isEmpty)))
  (code.takeWhile (!·.startsWith ".section")).filter fun line =>
    !line.isEmpty && !line.startsWith ".loc " && !line.startsWith ".file "

private def alignedTemplate : String := r#"
module {
tt.func public @_aligned_packed_amax(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %partial_amax: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<2048xf32>
%batch = tt.get_program_id x : i32
%part = tt.get_program_id y : i32
%lane = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
%first_count = arith.subi @first_count@, %part : i32
%first_count_0 = arith.divsi %first_count, 32 : i32
%total_count = arith.subi @total_count@, %part : i32
%total_count_1 = arith.divsi %total_count, 32 : i32
%local_amax = scf.for %i = 0 to %first_count_0 step 1 iter_args(%local_amax_3 = %cst) -> (tensor<2048xf32>) : i32 {
%tile = arith.muli %i, 32 : i32
%tile_4 = arith.addi %part, %tile : i32
%values = arith.muli %batch, @first_stride@ : i32
%values_5 = tt.addptr %first, %values : !tt.ptr<bf16>, i32
%values_6 = arith.muli %tile_4, 2048 : i32
%values_7 = tt.addptr %values_5, %values_6 : !tt.ptr<bf16>, i32
%values_8 = tt.splat %values_7 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_9 = tt.addptr %values_8, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_10 = tt.load %values_9 : tensor<2048x!tt.ptr<bf16>>
%values_11 = arith.extf %values_10 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_12 = arith.addf %values_11, %cst : tensor<2048xf32>
%local_amax_13 = math.absf %local_amax_12 : tensor<2048xf32>
%local_amax_14 = arith.maxnumf %local_amax_3, %local_amax_13 : tensor<2048xf32>
scf.yield %local_amax_14 : tensor<2048xf32>
}
%local_amax_2 = scf.for %i = %first_count_0 to %total_count_1 step 1 iter_args(%local_amax_3 = %local_amax) -> (tensor<2048xf32>) : i32 {
%tile = arith.muli %i, 32 : i32
%tile_4 = arith.addi %part, %tile : i32
%tile_5 = arith.subi %tile_4, @first_tiles@ : i32
%values = arith.muli %batch, @second_stride@ : i32
%values_6 = tt.addptr %second, %values : !tt.ptr<bf16>, i32
%values_7 = arith.muli %tile_5, 2048 : i32
%values_8 = tt.addptr %values_6, %values_7 : !tt.ptr<bf16>, i32
%values_9 = tt.splat %values_8 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_10 = tt.addptr %values_9, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_11 = tt.load %values_10 : tensor<2048x!tt.ptr<bf16>>
%values_12 = arith.extf %values_11 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_13 = arith.addf %values_12, %cst : tensor<2048xf32>
%local_amax_14 = math.absf %local_amax_13 : tensor<2048xf32>
%local_amax_15 = arith.maxnumf %local_amax_3, %local_amax_14 : tensor<2048xf32>
scf.yield %local_amax_15 : tensor<2048xf32>
}
%0 = arith.muli %batch, 32 : i32
%1 = tt.addptr %partial_amax, %0 : !tt.ptr<f32>, i32
%2 = tt.addptr %1, %part : !tt.ptr<f32>, i32
%3 = "tt.reduce"(%local_amax_2) <{axis = 0 : i32}> ({
^bb0(%arg3: f32, %arg4: f32):
%4 = arith.maxnumf %arg3, %arg4 : f32
tt.reduce.return %4 : f32
}) : (tensor<2048xf32>) -> f32
tt.store %2, %3 : !tt.ptr<f32>
tt.return
}
}
"#

private def referenceSecondBank : String := r#"
%second_values_14 = arith.muli %batch, @second_stride@ : i32
%second_values_15 = tt.addptr %second, %second_values_14 : !tt.ptr<bf16>, i32
%second_values_16 = tt.splat %second_values_15 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%second_values_17 = tt.addptr %second_values_16, %second_indices_12 : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%second_values_18 = tt.load %second_values_17, %second_values_13, %cst : tensor<2048x!tt.ptr<bf16>>
%values = arith.extf %first_values_11 : tensor<2048xbf16> to tensor<2048xf32>
%values_19 = arith.extf %second_values_18 : tensor<2048xbf16> to tensor<2048xf32>
%values_20 = arith.addf %values, %values_19 : tensor<2048xf32>
%local_amax_21 = math.absf %values_20 : tensor<2048xf32>
%local_amax_22 = arith.maxnumf %local_amax_5, %local_amax_21 : tensor<2048xf32>
scf.yield %local_amax_22 : tensor<2048xf32>
"#

private def referenceSharedStride : String := r#"
%second_values_14 = tt.addptr %second, %first_values_7 : !tt.ptr<bf16>, i32
%second_values_15 = tt.splat %second_values_14 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%second_values_16 = tt.addptr %second_values_15, %second_indices_12 : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%second_values_17 = tt.load %second_values_16, %second_values_13, %cst : tensor<2048x!tt.ptr<bf16>>
%values = arith.extf %first_values_11 : tensor<2048xbf16> to tensor<2048xf32>
%values_18 = arith.extf %second_values_17 : tensor<2048xbf16> to tensor<2048xf32>
%values_19 = arith.addf %values, %values_18 : tensor<2048xf32>
%local_amax_20 = math.absf %values_19 : tensor<2048xf32>
%local_amax_21 = arith.maxnumf %local_amax_5, %local_amax_20 : tensor<2048xf32>
scf.yield %local_amax_21 : tensor<2048xf32>
"#

private def referenceTemplate : String := r#"
module {
tt.func public @_packed_batched_matrix_amax_partial_kernel(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %partial_amax: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%local_amax = arith.constant dense<0.000000e+00> : tensor<2048xf32>
%cst = arith.constant dense<0.000000e+00> : tensor<2048xbf16>
%cst_0 = arith.constant dense<true> : tensor<2048xi1>
%cst_1 = arith.constant dense<0> : tensor<2048xi32>
%cst_2 = arith.constant dense<@first_elements@> : tensor<2048xi32>
%cst_3 = arith.constant dense<@total_elements@> : tensor<2048xi32>
%batch = tt.get_program_id x : i32
%part = tt.get_program_id y : i32
%offsets = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
%0 = arith.muli %part, 2048 : i32
%local_amax_4 = scf.for %start = %0 to @total_elements@ step 65536 iter_args(%local_amax_5 = %local_amax) -> (tensor<2048xf32>) : i32 {
%indices = tt.splat %start : i32 -> tensor<2048xi32>
%indices_6 = arith.addi %indices, %offsets : tensor<2048xi32>
%valid = arith.cmpi slt, %indices_6, %cst_3 : tensor<2048xi32>
%in_first = arith.cmpi slt, %indices_6, %cst_2 : tensor<2048xi32>
%first_values = arith.andi %valid, %in_first : tensor<2048xi1>
%first_values_7 = arith.muli %batch, @first_stride@ : i32
%first_values_8 = tt.addptr %first, %first_values_7 : !tt.ptr<bf16>, i32
%first_values_9 = tt.splat %first_values_8 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%first_values_10 = tt.addptr %first_values_9, %indices_6 : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%first_values_11 = tt.load %first_values_10, %first_values, %cst : tensor<2048x!tt.ptr<bf16>>
%second_indices = arith.subi %indices_6, %cst_2 : tensor<2048xi32>
%second_indices_12 = arith.maxsi %second_indices, %cst_1 : tensor<2048xi32>
%second_values = arith.xori %in_first, %cst_0 : tensor<2048xi1>
%second_values_13 = arith.andi %valid, %second_values : tensor<2048xi1>
@reference_second_bank@
}
%1 = arith.muli %batch, 32 : i32
%2 = tt.addptr %partial_amax, %1 : !tt.ptr<f32>, i32
%3 = tt.addptr %2, %part : !tt.ptr<f32>, i32
%4 = "tt.reduce"(%local_amax_4) <{axis = 0 : i32}> ({
^bb0(%arg3: f32, %arg4: f32):
%5 = arith.maxnumf %arg3, %arg4 : f32
tt.reduce.return %5 : f32
}) : (tensor<2048xf32>) -> f32
tt.store %3, %4 : !tt.ptr<f32>
tt.return
}
}
"#

inductive Side where
  | reference | aligned
  deriving DecidableEq, Repr

private def template (side : Side) (w : Width) : List String :=
  let l := Layout.ofWidth w
  let text := match side with
    | .reference =>
        let secondBank :=
          if l.firstStride == l.secondStride then referenceSharedStride else referenceSecondBank
        referenceTemplate.replace "@reference_second_bank@" secondBank
    | .aligned => alignedTemplate
  let values :=
    [("@first_count@", l.firstTiles + 31), ("@total_count@", l.firstTiles + l.secondTiles + 31),
     ("@first_tiles@", l.firstTiles), ("@first_stride@", l.firstStride), ("@second_stride@", l.secondStride),
     ("@first_elements@", l.firstTiles * Tile), ("@total_elements@", (l.firstTiles + l.secondTiles) * Tile)]
  let specialized := values.foldl (fun body (key, value) => body.replace key (toString value)) text
  (specialized.splitOn "\n").filter (!·.isEmpty)

def matchesArtifact (side : Side) (w : Width) (text : String) : Bool :=
  normalizeTTIR text == some (template side w)

/-- Resolve the entire matched fragment to its typed store observation.
There is no catch-all interpretation for unrecognized TTIR. -/
def decodeStore (side : Side) (w : Width) (text : String)
    (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (batch : Fin (Layout.ofWidth w).batches) (part : Part) : Option (Nat × Scalar) :=
  if matchesArtifact side w text then
    some (match side with
      | .reference => referenceStore first second reduce w batch part
      | .aligned => emittedStore first second reduce w batch part)
  else none

theorem decoded_stores_equal (w : Width) (oldIR newIR : String)
    (hr : matchesArtifact .reference w oldIR = true) (ha : matchesArtifact .aligned w newIR = true)
    (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    decodeStore .aligned w newIR first second reduce batch part =
      decodeStore .reference w oldIR first second reduce batch part := by
  simp [decodeStore, hr, ha, store_lowering]

structure Certificate (w : Width) (oldIR newIR : String) : Type where
  stores : ∀ (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (batch : Fin (Layout.ofWidth w).batches) (part : Part),
    decodeStore .aligned w newIR first second reduce batch part =
      decodeStore .reference w oldIR first second reduce batch part

def certifyPair (w : Width) (oldIR newIR : String) : Option (Certificate w oldIR newIR) :=
  if hr : matchesArtifact .reference w oldIR = true then
    if ha : matchesArtifact .aligned w newIR = true then
      some ⟨decoded_stores_equal w oldIR newIR hr ha⟩
    else none
  else none

#axiomsClean decoded_stores_equal
#axiomsClean certifyPair
end VeriTile.Bench.Models.NanoGPT.Track1.QKVTTIR
