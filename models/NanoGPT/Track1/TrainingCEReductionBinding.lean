import models.NanoGPT.Track1.TrainingCEGatherBinding

/-! Decode the five warp shuffles and the leader store at the end of the
fixed-count CE forward scan. The input register carries one FP32 sum per
lane. Both the shuffle parameters and the predicate, address and value
of the shared store are recovered from the PTX. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEReductionBinding
open TrainingCE TrainingCEGatherBinding

private def sumsSymbol := "_ZZ17ce_fwd_bwd_kernelE10block_sums"

structure Reduction where
  input : String
  offsets : List Nat
  output : String
  leaderMask : Nat
  warpShift : Nat
  wordBytes : Nat

private abbrev Constants := List (String × Int)

private def constant (values : Constants) (arg : String) : Option Int :=
  (values.lookup arg).orElse fun _ => arg.toInt?

private def readConstants : Constants → List (List String) →
    Option (Constants × List (List String))
  | values, ["mov.u32", dst, lit] :: rest => do
    let value ← lit.toInt?
    readConstants ((dst, value) :: values.filter (fun entry => entry.1 != dst)) rest
  | values, next => some (values, next)

/-- Match dataflow instead of register numbers. The shuffle output is
bitcast to float, added to the current sum, and reinterpreted as bits
before the next shuffle. -/
private def decodeSteps (current : String) (constants : Constants) :
    Nat → List (List String) → Option (List Nat × String × List (List String))
  | 0, rest => some ([], current, rest)
  | steps + 1,
      ["mov.b32", bits, input] :: tail => do
    guard (input == current)
    let (constants, next) ← readConstants (constants.filter (fun entry => entry.1 != bits)) tail
    guard ((constants.lookup bits).isNone)
    let [op, outputs, fromReg, delta, clamp, mask] :: ["mov.b32", shuffled, fromBits] ::
      ["add.ftz.f32", result, left, right] :: rest := next | none
    let [word, predicate] := outputs.splitOn "|" | none
    guard (op == "shfl.sync.down.b32" && fromReg == bits && word == fromBits)
    guard (predicate.startsWith "%p" && bits.startsWith "%r" && word.startsWith "%r" &&
      shuffled.startsWith "%f" && result.startsWith "%f" && shuffled != current)
    guard (left == current && right == shuffled && clamp != bits && delta != bits && mask != bits)
    guard (constant constants clamp == some 31 && constant constants mask == some (-1))
    let offset ← constant constants delta
    guard (0 < offset && offset < 32)
    let nextConstants := constants.filter (fun entry => entry.1 != bits && entry.1 != word)
    let (offsets, final, after) ← decodeSteps result nextConstants steps rest
    return (offset.toNat :: offsets, final, after)
  | _, _ => none

private def storeLayout (result : String) (instructions : List (List String)) :
    Option (String × Nat × Nat × Nat) := do
  let ["and.b32", low, lane, mask] :: ["setp.ne.s32", predicate, test, "0"] ::
    [branch, "bra", skip] :: ["shr.u32", warp, sameLane, shift] ::
    ["mul.wide.u32", displacement, sameWarp, stride] :: ["mov.u64", base, symbol] ::
    ["add.s64", pointer, sameBase, sameDisplacement] :: ["st.shared.f32", toAddr, payload] ::
    [label] :: tail := instructions | none
  guard (test == low && branch == "@" ++ predicate && label == skip ++ ":")
  guard (sameLane == lane && sameWarp == warp && low != lane && warp != lane)
  guard (low.startsWith "%r" && warp.startsWith "%r" && predicate.startsWith "%p" &&
    displacement.startsWith "%rd" && base.startsWith "%rd" && pointer.startsWith "%rd")
  guard (symbol == sumsSymbol && sameBase == base && sameDisplacement == displacement && base != displacement)
  guard (toAddr == "[" ++ pointer ++ "]" && payload == result)
  match tail with
  | [] => pure ()
  | [["cvt.u64.u32", _, tid]] => guard (tid == lane)
  | _ => none
  return (lane, ← mask.toNat?, ← shift.toNat?, ← stride.toNat?)

private def laneSource (lines : List String) (edges : Array (List Nat)) :
    Nat → String → Nat → Option Unit
  | 0, _, _ => none
  | fuel + 1, lane, use => do
    let (insn, pc) ← uniqueDefinition lines edges lane use
    guard (pc < use)
    match insn with
    | ["mov.u32", _, "%tid.x"] => pure ()
    | ["cvt.u32.u64", _, source] =>
      guard (lane.startsWith "%r" && !lane.startsWith "%rd" && source.startsWith "%rd")
      laneSource lines edges fuel source pc
    | ["cvt.u64.u32", _, source] =>
      guard (lane.startsWith "%rd" && source.startsWith "%r" && !source.startsWith "%rd")
      laneSource lines edges fuel source pc
    | _ => none

def decode (ptx : String) : Option Reduction := do
  let lines := QKVTTIR.normalizePTX ptx
  guard (lines.contains s!".shared .align 4 .b8 {sumsSymbol}[32];" ||
    lines.contains s!".shared .align 16 .b8 {sumsSymbol}[32];")
  guard (lines.contains ".maxntid 256, 1, 1")
  let edges ← flow lines
  let [tableReady, sumsReady, _] := lines.zipIdx.filterMap (fun (line, pc) =>
    if line == "bar.sync 0;" then some pc else none) | none
  let firstShuffle ← lines.zipIdx.find? (fun (line, _) => line.startsWith "shfl.sync.down.b32 ")
  -- The straight-line prelude consists of the seed cast and the four
  -- immediates the compiler hoists for the first and fourth shuffles.
  guard (tableReady + 5 < firstShuffle.2 && firstShuffle.2 < sumsReady)
  let seed := firstShuffle.2 - 5
  guard (dominates edges seed sumsReady)
  let body := ((lines.take sumsReady).drop seed).map operands
  let ["mov.b32", _, input] :: _ := body | none
  guard (input.startsWith "%f")
  let (offsets, result, tail) ← decodeSteps input [] 5 body
  let (lane, mask, shift, stride) ← storeLayout result tail
  guard (lane.startsWith "%r" && !lane.startsWith "%rd")
  laneSource lines edges 3 lane seed
  return ⟨input, offsets, result, mask, shift, stride⟩

def Reduction.store (decoded : Reduction) (read : Lane → F32) (lane : Lane) : Option (Nat × F32) :=
  if lane.val % (decoded.leaderMask + 1) = 0 then
    some (lane.val / (2 ^ decoded.warpShift) * decoded.wordBytes,
      (decoded.offsets.foldl shuffleAdd read) lane)
  else none

structure Certificate (ptx : String) where
  reduction : Reduction
  decoded : decode ptx = some reduction
  offsets : reduction.offsets = [16, 8, 4, 2, 1]
  layout : (reduction.leaderMask, reduction.warpShift, reduction.wordBytes) = (31, 5, 4)

def certify (ptx : String) : Option (Certificate ptx) :=
  match decoded : decode ptx with
  | none => none
  | some reduction =>
    if offsets : reduction.offsets = [16, 8, 4, 2, 1] then
      if layout : (reduction.leaderMask, reduction.warpShift, reduction.wordBytes) = (31, 5, 4) then
        some ⟨reduction, decoded, offsets, layout⟩
      else none
    else none

theorem store_exact {ptx : String} (artifact : Certificate ptx) (read : Lane → F32) (lane : Lane) :
    artifact.reduction.store read lane =
      if leader : lane.val % 32 = 0 then
        let warp : Fin 8 := ⟨lane.val / 32, by have := lane.isLt; omega⟩
        some (warp.val * 4, warpLeaders read warp)
      else none := by
  have mask : artifact.reduction.leaderMask = 31 := congrArg Prod.fst artifact.layout
  have shift : artifact.reduction.warpShift = 5 := congrArg (fun layout => layout.2.1) artifact.layout
  have stride : artifact.reduction.wordBytes = 4 := congrArg (fun layout => layout.2.2) artifact.layout
  by_cases aligned : lane.val % 32 = 0
  · simp only [Reduction.store, mask, shift, stride, artifact.offsets, Nat.reduceAdd, Nat.reducePow,
      if_pos aligned, dif_pos aligned, warpLeaders]
    have sameLane : (⟨lane.val / 32 * 32, by have := lane.isLt; omega⟩ : Lane) = lane := by
      apply Fin.ext
      change lane.val / 32 * 32 = lane.val
      omega
    rw [sameLane]
  · simp only [Reduction.store, mask, Nat.reduceAdd, if_neg aligned, dif_neg aligned]

def Reduction.readLeaders (decoded : Reduction) (read : Lane → F32) (warp : Fin 8) : F32 :=
  let lane : Lane := ⟨warp.val * 32, by have := warp.isLt; omega⟩
  ((decoded.store read lane).map Prod.snd).getD (VeriTile.Triton.FP.Value.ofBits .fp32 0)

theorem leaders_exact {ptx : String} (artifact : Certificate ptx) (read : Lane → F32) :
    artifact.reduction.readLeaders read = warpLeaders read := by
  funext warp
  let lane : Lane := ⟨warp.val * 32, by have := warp.isLt; omega⟩
  have aligned : lane.val % 32 = 0 := by simp [lane]
  have index : (⟨lane.val / 32, by have := lane.isLt; omega⟩ : Fin 8) = warp := by
    apply Fin.ext
    change warp.val * 32 / 32 = warp.val
    omega
  change ((artifact.reduction.store read lane).map Prod.snd).getD _ = _
  rw [store_exact artifact, dif_pos aligned, index]
  rfl

#axiomsClean store_exact
#axiomsClean leaders_exact
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEReductionBinding
