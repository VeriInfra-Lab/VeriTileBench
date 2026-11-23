import models.NanoGPT.Track1.HostFloat64
import VeriTile.Meta.StatementAudit

/-! Numerical execution of the pinned Track 1 host schedule. Ratios denote
exact binary64 values, with rounding at the Python arithmetic sites.
The two upload paths differ only in the placement of the FP32 words. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.HostSchedule
open VeriTile.Triton FP
open HostFloat64 (nearestNat)

private def rn := HostFloat64.round

structure Stage where
  duration : Rat
  batchSize : Nat
  maxLength : Nat
  shortWindow : Nat
  longWindow : Nat
  rate : Rat
  mtpStart : List Rat
  mtpEnd : List Rat
  prefixWeights : List Rat

def normalizedDurations : List Rat :=
  let values := [rn ((2854 : Rat) / 10000), rn ((3209 : Rat) / 10000), rn ((3931 : Rat) / 10000)]
  let total := rn values.sum
  values.map fun d => rn (d / total)

private def taperDuration : Rat := rn ((6 : Rat) / 100)

/-- The binary64 square root of the rounded host division 20/24. -/
def taperRoot : Rat := (8222410352942271 : Rat) / 9007199254740992

theorem taper_root_bracket :
    let input := rn ((20 : Rat) / 24)
    let halfUlp := (1 : Rat) / 18014398509481984
    let lo := taperRoot - halfUlp
    let hi := taperRoot + halfUlp
    rn taperRoot = taperRoot ∧ 0 < lo ∧ lo * lo < input ∧ input < hi * hi := by
  decide +kernel

def firstThreeDurations : List Rat :=
  let d := normalizedDurations
  [d[0]?.getD 0, d[1]?.getD 0, rn (d[2]?.getD 0 - taperDuration)]

def stages : List Stage := [
  ⟨firstThreeDurations[0]?.getD 0, 131072, 896, 1, 3, 1,
    [1, 1/2, 1/4], [1, 1/2, 0], [1/4]⟩,
  ⟨firstThreeDurations[1]?.getD 0, 262144, 2048, 3, 7, rn ((152 : Rat)/100),
    [1, 1/2], [1, 0], [1/5, rn ((15 : Rat)/100), 1/10, 1/20].map rn⟩,
  ⟨firstThreeDurations[2]?.getD 0, 393216, 3072, 5, 11, rn ((173 : Rat)/100), [1], [1], [0]⟩,
  ⟨taperDuration, 327680, 3072, 5, 11, rn (rn ((173 : Rat)/100) * taperRoot), [1], [1], [0]⟩,
  ⟨0, 131072, 3072, 6, 13, 1, [1], [1], [0]⟩]

def boundaries (base extension : Nat) : List (Nat × Nat) :=
  let endFractions := ((stages.dropLast.map (·.duration)).scanl (fun a b => rn (a + b)) 0).drop 1
  let shifted := endFractions.mapIdx fun i x => nearestNat (rn (x * rn (base : Rat))) + if i ≥ 2 then 52 else 0
  let ends := 0 :: shifted ++ [base + 52 + extension]
  ends.zip (ends.drop 1)

theorem default_boundaries : boundaries 1122 20 =
    [(0, 320), (320, 681), (681, 1107), (1107, 1174), (1174, 1194)] := by decide +kernel

structure Plan where
  scheduled : Nat
  total : Nat
  splitStep : Nat
  limits : List (Nat × Nat)

def build (base extension : Nat) : Option Plan := do
  let limits := boundaries base extension
  let split ← limits[4]?
  if split.1 ≠ base + 52 then none else
    some ⟨base + 52, base + 52 + extension, split.1 ||| 1, limits⟩

def Plan.lookup (s : Plan) (step : Nat) : Option (Stage × Rat) :=
  match (stages.zip s.limits).find? (fun x => step < x.2.2) with
  | some (stage, (first, stop)) => some (stage, rn (((step - first : Nat) : Rat) / (stop - first : Nat)))
  | none => stages.getLast?.map (·, 1)

theorem lookup_stage (s : Plan) (step : Nat) (stage : Stage) (progress : Rat)
    (found : s.lookup step = some (stage, progress)) : stage ∈ stages := by
  unfold Plan.lookup at found
  split at found
  · rename_i selected period picked
    obtain ⟨rfl, _⟩ := Prod.mk.inj (Option.some.inj found)
    exact (List.of_mem_zip (List.mem_of_find?_eq_some picked)).1
  · cases last : stages.getLast? with
    | none => simp [last] at found
    | some selected =>
      simp only [last, Option.map_some, Option.some.injEq, Prod.mk.injEq] at found
      obtain ⟨rfl, _⟩ := found
      exact List.mem_of_getLast? last

theorem largest_stage_batch (stage : Stage) (member : stage ∈ stages) :
    stage.batchSize ≤ 393216 := by
  simp only [stages, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;> decide

theorem lookup_batch_bound (s : Plan) (step : Nat) (stage : Stage) (progress : Rat)
    (found : s.lookup step = some (stage, progress)) : stage.batchSize ≤ 393216 :=
  largest_stage_batch stage (lookup_stage s step stage progress found)

private def toF32 (x : Rat) : Value .fp32 := FP.round .fp32 Config.ieeeValue x
private def truncate (x : Rat) : Nat := x.num.natAbs / x.den

structure Forward where
  batchSize : Nat
  sequenceLimit : Nat
  shortBlocks : Nat
  longBlocks : Nat
  mtp : List (Value .fp32)
  prefixWeight : Value .fp32

def Plan.forward (s : Plan) (step : Nat) : Option Forward := do
  let (stage, t) ← s.lookup step
  let mtp := (stage.mtpStart.zip stage.mtpEnd).map fun (a, b) => toF32 (rn (a + rn (rn (b - a) * t)))
  let idx := min (stage.prefixWeights.length - 1) (truncate (rn (t * rn stage.prefixWeights.length)))
  let prefixWeight ← stage.prefixWeights[idx]?
  pure ⟨stage.batchSize, stage.maxLength, stage.shortWindow, stage.longWindow, mtp, toF32 prefixWeight⟩

def lrFloor : Rat := rn ((3 : Rat) / 10)

def Plan.learningRate (s : Plan) (step : Nat) : Option Rat := do
  let (stage, _) ← s.lookup step
  let cutoff := truncate (rn (rn s.scheduled * rn (1 - rn ((8 : Rat)/10))))
  if step < cutoff then some stage.rate else
    let t := min 1 (rn (((step - cutoff : Nat) : Rat) / (s.scheduled - cutoff : Nat)))
    some (rn (rn (stage.rate * rn (1 - t)) + rn (lrFloor * t)))

def railBeta (total step : Nat) : Rat :=
  let low := rn ((85 : Rat)/100)
  let high := rn ((93 : Rat)/100)
  let delta := rn (high - low)
  if step < 240 then rn (low + rn (rn ((step : Rat) / 240) * delta))
  else
    let cooldownStart : Int := (total : Int) - 50
    if (step : Int) > cooldownStart then
      rn (high - rn (rn (((step : Int) - cooldownStart : Int) / (50 : Rat)) * delta))
    else high

structure OptimizerStep where
  rate : Rat
  momentum : Rat
  fastBeta : Rat
  fastWeight : Rat
  updateAdam : Bool
  splitEmbed : Bool

def Plan.optimizer (s : Plan) (step : Nat) : Option OptimizerStep := do
  let rate ← s.learningRate step
  let beta := railBeta s.total step
  let (fastBeta, fastWeight) := if step < 514 then (beta, 1) else (rn ((85 : Rat)/100), rn ((4385 : Rat)/10000))
  pure ⟨rate, beta, fastBeta, fastWeight, step % 2 == 1, step == s.splitStep⟩

def OptimizerStep.parameterRate (s : OptimizerStep) (initial : Rat) : Rat :=
  rn (initial * s.rate)

inductive ParameterKind where
  | anvil | valueEmbedding | other

structure ParameterGroup where
  kind : ParameterKind
  initialRate : Rat

inductive ParameterChange where
  | rateOnly
  | momentum (value : Rat)
  | adam (beta1 beta2 decayMultiplier : Rat)

def OptimizerStep.forGroup (s : OptimizerStep) (step : Nat) (group : ParameterGroup) : Rat × ParameterChange :=
  let change := match group.kind with
    | .anvil => .momentum s.momentum
    | .valueEmbedding =>
      let beta2 := rn ((95 : Rat)/100)
      if step < 336 then .adam (3/4) beta2 5 else .adam (9/16) (rn (beta2 * beta2)) 10
    | .other => .rateOnly
  (s.parameterRate group.initialRate, change)

structure OptimizerValues where
  controls : OptimizerStep
  groups : List (Rat × ParameterChange)
  sparseRate : Rat
  sparseDecay : Rat
  sparseEpsilon : Rat

def OptimizerStep.values (s : OptimizerStep) (step : Nat) (groups : List ParameterGroup) : OptimizerValues :=
  ⟨s, groups.map (s.forGroup step), s.parameterRate (rn ((8 : Rat)/1000)),
    rn ((5 : Rat)/1000), rn ((1 : Rat)/10000000000)⟩

abbrev WeightRows := List (List (Value .fp32))

structure WeightTables where
  mtp : WeightRows
  prefixWeights : WeightRows

def Plan.weights (s : Plan) : Option WeightTables := do
  let rows ← (List.range (s.total + 1)).mapM s.forward
  pure ⟨rows.map (·.mtp), rows.map fun r => [r.prefixWeight]⟩

/-- A row view addresses its own four-word slot in the backing tensor. -/
def uploadRows (rows : WeightRows) : WeightRows :=
  let storage := rows.map fun row => row ++ List.replicate (4 - row.length) (Value.zero .fp32)
  (storage.zip (rows.map List.length)).map fun (row, width) => row.take width

private theorem upload_rows_equal (rows : WeightRows) : uploadRows rows = rows := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
    simp only [uploadRows, List.map_cons, List.zip_cons_cons, List.take_append_length, List.cons.injEq, true_and] at *
    exact ih

abbrev Upload := WeightTables → WeightTables

def uploadReference : Upload := id

def uploadPacked : Upload := fun tables =>
  ⟨uploadRows tables.mtp, uploadRows tables.prefixWeights⟩

theorem upload_equal : uploadPacked = uploadReference := by
  funext tables
  cases tables
  simp [uploadPacked, uploadReference, upload_rows_equal]

theorem packed_row_alignment (step : Nat) : (4 * step * 4) % 16 = 0 := by omega

theorem packed_rows_disjoint (a b : Nat) (p q : Fin 4) (distinct : a ≠ b) :
    4 * a + p.val ≠ 4 * b + q.val := by
  have hp := p.isLt
  have hq := q.isLt
  omega

#axiomsClean taper_root_bracket
#axiomsClean default_boundaries
#axiomsClean upload_equal
#axiomsClean packed_row_alignment
#axiomsClean packed_rows_disjoint

end VeriTile.Bench.Models.NanoGPT.Track1.HostSchedule
