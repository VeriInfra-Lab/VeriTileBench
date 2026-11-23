import models.NanoGPT.Track1.ValueAdamNativeEvent

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

private def readArtifact (root : System.FilePath) (stem : String) (period : Period)
    (tile : Tile) (fused : Bool) : IO (Artifact period tile fused) := do
  let profile : Profile := ⟨period, tile, fused⟩
  let metadata ← IO.FS.readFile (root / (stem ++ ".json"))
  if !profile.metadataMatches metadata then throw (IO.userError s!"Wrong launch metadata for {stem}")
  let ptx ← IO.FS.readFile (root / (stem ++ ".ptx"))
  let some certificate := certify profile ptx
    | throw (IO.userError s!"{stem}: {diagnose profile ptx}")
  let modified := ptx.replace ".param .f32 " ".param .u64 "
  if modified == ptx || (certify profile modified).isSome then
    throw (IO.userError s!"{stem}: accepted a different scalar parameter ABI")
  return ⟨ptx, certificate⟩

private def readConsumer (root : System.FilePath) (stem : String) (period : Period)
    (tile : Tile) (clearAll : Bool) : IO (Consumer period tile clearAll) := do
  let p ← readArtifact root stem period tile true
  if matched : p.2.modeMatches clearAll = true then
    let modified := p.1.replace "cvt.rn.f16.f32" "cvt.rz.f16.f32"
    if modified == p.1 || (certify ⟨period, tile, true⟩ modified).isSome then
      throw (IO.userError s!"{stem}: accepted a different FP16 rounding")
    let lines := QKVTTIR.normalizePTX p.1
    if lines.any (·.startsWith "bar.sync ") then
      throw (IO.userError s!"{stem}: added CTA synchronization")
    let guarded := GuardedHalf.extract lines
    let expectedGuard := tile.width == 1024 && !clearAll
    unless guarded.isSome == expectedGuard do
      throw (IO.userError s!"{stem}: unexpected packed FP16 dispatch")
    let dirtyCode ← match guarded with
      | none => pure lines
      | some (_, _, suffix) => do
        let changes := [("max.NaN.f16x2", "max.f16x2"),
          ("vote.sync.any.pred", "vote.sync.all.pred"), ("0xffffffff", "0x0000ffff"),
          ("@%slow bra.uni", "@!%slow bra.uni"), ("0x30003000", "0x30003001"),
          ("mul.rn.f16x2", "mul.rn.ftz.f16x2"), ("{%a, %b}", "{%b, %a}")]
        for (before, after) in changes do
          let changed := p.1.replace before after
          if changed == p.1 || (certify ⟨period, tile, true⟩ changed).isSome then
            throw (IO.userError s!"{stem}: accepted a different packed FP16 instruction at {before}")
        pure suffix
    if !clearAll then
      let some predicate := dirtyCode.find? (·.startsWith "setp.ne.b32 ")
        | throw (IO.userError s!"{stem}: missing zero predicate")
      if !predicate.endsWith ", 0;" then throw (IO.userError s!"{stem}: malformed predicate")
      let altered := (predicate.dropEnd 2).toString ++ "1;"
      let changed := String.intercalate "\n" (lines.map fun line =>
        if line == predicate then altered else line)
      if (certify ⟨period, tile, true⟩ changed).isSome then
        throw (IO.userError s!"{stem}: accepted a predicate comparing against one")
    return ⟨p, matched⟩
  else throw (IO.userError s!"{stem}: unexpected gradient workspace stores")

private structure Pair (period : Period) (tile : Tile) where
  baseline : Artifact period tile false
  masked : Consumer period tile false
  dense : Consumer period tile true

private def readPair (root : System.FilePath) (period : Period) (tile : Tile) : IO (Pair period tile) := do
  let tick := if period == .two then 2 else 4
  let width := tile.width
  let r ← readArtifact root s!"dense-period{tick}.{width}" period tile false
  if r.2.consumes then throw (IO.userError "Reference gradient workspace is read-only")
  let masked ← readConsumer root s!"consume-masked-{tick}-{width}" period tile false
  let dense ← readConsumer root s!"consume-dense-{tick}-{width}" period tile true
  IO.println s!"Certified value Adam results and both workspace clear modes for period {tick}, tile {width}."
  return ⟨r, masked, dense⟩

def readPrograms (root : System.FilePath) : IO NativePrograms := do
  let a ← readPair root .two .x512
  let b ← readPair root .two .x1024
  let c ← readPair root .four .x512
  let d ← readPair root .four .x1024
  IO.println "Certified the guarded FP16 pack, both branch effects and the distinct workspace clear predicate."
  let select : (period : Period) → (tile : Tile) → Pair period tile :=
    fun | .two, .x512 => a | .two, .x1024 => b | .four, .x512 => c | .four, .x1024 => d
  return {
    reference := fun period tile => (select period tile).baseline,
    fused := fun period tile clearAll => match clearAll with
      | false => (select period tile).masked
      | true => (select period tile).dense
  }

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
