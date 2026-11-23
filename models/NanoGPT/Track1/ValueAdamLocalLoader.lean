import models.NanoGPT.Track1.ValueAdamLocalSource

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows

structure Program (direct marked : Bool) where
  ptx : String
  checked : Certificate direct marked ptx

structure Programs where
  get : (direct marked : Bool) → Program direct marked

private def metadataMatches (raw : String) : Bool :=
  let inspected : Except String Bool := do
    let metadata ← Lean.Json.parse raw
    let target ← metadata.getObjVal? "target"
    return (← target.getObjValAs? String "backend") == "cuda" &&
      (← target.getObjValAs? Nat "arch") == 90 &&
      (← target.getObjValAs? Nat "warp_size") == 32 &&
      (← metadata.getObjValAs? Nat "num_warps") == 4 &&
      (← metadata.getObjValAs? Nat "num_ctas") == 1 &&
      (← metadata.getObjValAs? Nat "num_stages") == 1 &&
      (← metadata.getObjValAs? Nat "shared") == 0
  inspected.toOption.getD false

private def readProgram (root : System.FilePath) (direct marked : Bool) : IO (Program direct marked) := do
  let kind := if direct then "direct" else "staged"
  let flag := if marked then "1" else "0"
  let stem := s!"value-local-{kind}-{flag}"
  let metadata ← IO.FS.readFile (root / (stem ++ ".json"))
  if !metadataMatches metadata then throw (IO.userError s!"{stem}: incorrect launch metadata")
  let ptx ← IO.FS.readFile (root / (stem ++ ".ptx"))
  let some checked := certify direct marked ptx
    | throw (IO.userError s!"{stem}: {repr (diagnose direct marked ptx)}")
  let mutations := [("setp.ge.s32", "setp.lt.s32"), ("bra.uni", "bra"),
      ("bra.uni L__local_rows_end", "bra.uni wrong_target"),
      (", 248;", ", 240;"), ("mad.lo.s32", "mad.wide.s32"),
      (".noftz.v8.f16", ".noftz.v4.f16")] ++
      (if marked then [(".max.s32", ".min.s32")] else [])
  for (before, after) in mutations do
    let mutated := ptx.replace before after
    if mutated == ptx || (certify direct marked mutated).isSome then
      throw (IO.userError s!"{stem}: wrong mutation result for {before}")
  return ⟨ptx, checked⟩

def readPrograms (root : System.FilePath) : IO Programs := do
  let stagedUnmarked ← readProgram root false false
  let stagedMarked ← readProgram root false true
  let directUnmarked ← readProgram root true false
  let directMarked ← readProgram root true true
  IO.println "Certified both local value scatter layouts, markers, disabled lanes and positive-zero stores."
  return ⟨fun
    | false, false => stagedUnmarked
    | false, true => stagedMarked
    | true, false => directUnmarked
    | true, true => directMarked⟩

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
