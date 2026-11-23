import models.NanoGPT.Track1.TrainingCEExecution

/-! The launch width follows the candidate schedule in sampled_softmax.py.
The graph's prediction count is the length of the schedule's MTP row. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCELaunch
open TrainingCE

def sampledWidths : Nat → List Width
  | 0 | 1 => [.sampled10]
  | 2 => [.sampled14, .sampled14, .sampled24]
  | _ => []

def widthAt (plan : HostSchedule.Plan) (step : Nat) : Width := Id.run do
  let some (interval, i) := plan.limits.zipIdx.find? (fun x => step < x.1.2) | return .full
  let widths := sampledWidths i
  return widths[((step - interval.1) * widths.length) / (interval.2 - interval.1)]?.getD .full

def defaultPlan : HostSchedule.Plan :=
  ⟨1174, 1194, 1175, [(0, 320), (320, 681), (681, 1107), (1107, 1174), (1174, 1194)]⟩

theorem source_default_plan : HostSchedule.build 1122 20 = some defaultPlan := by
  rw [HostSchedule.build, HostSchedule.default_boundaries]
  rfl

def defaultWidth (step : Nat) : Width :=
  if step < 681 then .sampled10
  else if step < 965 then .sampled14
  else if step < 1107 then .sampled24
  else .full

theorem source_default_width (step : Nat) :
    widthAt defaultPlan step = defaultWidth step := by
  unfold widthAt defaultWidth
  by_cases s0 : step < 320
  · have part : step / 320 = 0 := Nat.div_eq_of_lt s0
    simp [defaultPlan, sampledWidths, s0, part, show step < 681 by omega]
  by_cases s1 : step < 681
  · have part : (step - 320) / 361 = 0 := Nat.div_eq_of_lt (by omega)
    simp [defaultPlan, List.find?, sampledWidths, s0, s1, part]
  by_cases s2 : step < 1107
  · have part : (step - 681) * 3 / 426 < 3 := by omega
    by_cases transition : step < 965
    · have low : (step - 681) * 3 / 426 = 0 ∨ (step - 681) * 3 / 426 = 1 := by omega
      rcases low with low | low <;>
        simp [defaultPlan, List.find?, sampledWidths, s0, s1, s2, transition, low]
    · have high : (step - 681) * 3 / 426 = 2 := by omega
      simp [defaultPlan, List.find?, sampledWidths, s0, s1, s2, transition, high]
  have next : ¬ step < 965 := by omega
  by_cases s3 : step < 1174 <;> by_cases s4 : step < 1194 <;>
    simp [defaultPlan, List.find?, sampledWidths, s0, s1, s2, next, s3, s4]

def defaultPredictions (step : Nat) : Fin 4 :=
  if step < 320 then 3 else if step < 681 then 2 else 1

theorem source_default_predictions (step : Nat) (forward : HostSchedule.Forward)
    (loaded : defaultPlan.forward step = some forward) :
    forward.mtp.length = (defaultPredictions step).val := by
  simp only [HostSchedule.Plan.forward, Option.bind_eq_bind, Option.bind_eq_some_iff] at loaded
  obtain ⟨⟨stage, progress⟩, atStage, compiled⟩ := loaded
  simp only [Option.pure_def, Option.some.injEq] at compiled
  obtain ⟨prefixWeight, _, rfl⟩ := compiled
  simp only [List.length_map, List.length_zip]
  have count : min stage.mtpStart.length stage.mtpEnd.length =
      (defaultPredictions step).val := by
    unfold HostSchedule.Plan.lookup at atStage
    by_cases s0 : step < 320
    · have exactStage := congrArg (fun x => x.map Prod.fst) atStage
      simp [defaultPlan, HostSchedule.stages, s0] at exactStage
      subst stage
      simp [defaultPredictions, s0]
    by_cases s1 : step < 681
    · have exactStage := congrArg (fun x => x.map Prod.fst) atStage
      simp [defaultPlan, HostSchedule.stages, List.find?, s0, s1] at exactStage
      subst stage
      simp [defaultPredictions, s0, s1]
    by_cases s2 : step < 1107 <;> by_cases s3 : step < 1174 <;> by_cases s4 : step < 1194 <;>
      have exactStage := congrArg (fun x => x.map Prod.fst) atStage <;>
      simp [defaultPlan, HostSchedule.stages, List.find?, s0, s1, s2, s3, s4] at exactStage <;>
      subst stage <;> simp [defaultPredictions, s0, s1]
  exact count

private theorem default_supported (step : Nat) :
    (widthAt defaultPlan step).hasFixedPredictions (defaultPredictions step) = true := by
  rw [source_default_width]
  by_cases s0 : step < 320
  · simp [defaultWidth, defaultPredictions, Width.hasFixedPredictions, s0, show step < 681 by omega]
  by_cases s1 : step < 681
  · simp [defaultWidth, defaultPredictions, Width.hasFixedPredictions, s0, s1]
  by_cases s2 : step < 965
  · simp [defaultWidth, defaultPredictions, Width.hasFixedPredictions, s0, s1, s2]
  by_cases s3 : step < 1107 <;>
    simp [defaultWidth, defaultPredictions, Width.hasFixedPredictions, s0, s1, s2, s3]

theorem default_storage_extent (step : Nat) (forward : HostSchedule.Forward)
    (loaded : defaultPlan.forward step = some forward) :
    (forward.batchSize / 8) * (widthAt defaultPlan step).columns < 2^31 := by
  simp only [HostSchedule.Plan.forward, Option.bind_eq_bind, Option.bind_eq_some_iff] at loaded
  obtain ⟨⟨stage, progress⟩, atStage, compiled⟩ := loaded
  simp only [Option.pure_def, Option.some.injEq] at compiled
  obtain ⟨_, _, rfl⟩ := compiled
  change (stage.batchSize / 8) * (widthAt defaultPlan step).columns < 2^31
  rw [source_default_width]
  by_cases sampled : step < 1107
  · have rows : stage.batchSize / 8 ≤ 49152 := by
      have bound := HostSchedule.lookup_batch_bound defaultPlan step stage progress atStage
      omega
    have columns : (defaultWidth step).columns ≤ 24576 := by
      unfold defaultWidth
      split
      · decide
      split
      · decide
      simp [Width.columns]
    exact Nat.lt_of_le_of_lt (Nat.mul_le_mul rows columns) (by decide)
  · have width : defaultWidth step = .full := by
      simp [defaultWidth, sampled, show ¬ step < 965 by omega, show ¬ step < 681 by omega]
    rw [width]
    have selection := congrArg (fun x => x.map Prod.fst) atStage
    have beforeFirst : ¬ step < 320 := by omega
    have beforeSecond : ¬ step < 681 := by omega
    by_cases taper : step < 1174 <;> by_cases cool : step < 1194 <;>
      simp [HostSchedule.Plan.lookup, defaultPlan, HostSchedule.stages, List.find?,
        beforeFirst, beforeSecond, sampled, taper, cool] at selection <;>
      subst stage <;> dsimp only [] <;> decide

theorem selected_all_default_steps (programs : TrainingCEExecution.Programs) (step : Nat) :
    (programs.select (widthAt defaultPlan step) (defaultPredictions step)).isSome = true := by
  rw [TrainingCEExecution.Programs.select_isSome, default_supported]

def defaultParameters (p : Parameters) : Parameters :=
  {Softcap.defaultABC.parameters p with gradScaleParam := Driver.pinnedCEHeadGradBits}

private theorem default_parameters_match (w : Width) (p : Parameters) :
    w.fixedParametersMatch (defaultParameters p) = true := by
  cases w <;> rfl

theorem default_softcap_launch (step : Nat) (parameters : Parameters) :
    let w := widthAt defaultPlan step
    let p := defaultParameters parameters
    w.launchSoftcap (defaultPredictions step) p = w.defaultSoftcap ∧
      w.launchGradScale (defaultPredictions step) p = .lmHead := by
  simp [Width.launchSoftcap, Width.launchGradScale, Width.fixedLaunch,
    default_supported, default_parameters_match]

theorem scheduled_row_runs_fixed (programs : TrainingCEExecution.Programs) (step : Nat)
    (forward : HostSchedule.Forward) (loaded : defaultPlan.forward step = some forward)
    (op : Native) (parameters : Parameters) (row : Row (widthAt defaultPlan step))
    (predictions : row.predictions.val = forward.mtp.length) :
    let p := defaultParameters parameters
    ∃ program, programs.select (widthAt defaultPlan step) row.predictions = some program ∧
      programs.run (widthAt defaultPlan step) op p row = program.run (widthAt defaultPlan step) op p row := by
  let w := widthAt defaultPlan step
  let p := defaultParameters parameters
  have count : row.predictions = defaultPredictions step :=
    Fin.ext (predictions.trans (source_default_predictions step forward loaded))
  have selected : (programs.select w row.predictions).isSome = true := by
    rw [count]
    exact selected_all_default_steps programs step
  have accepted : w.fixedParametersMatch p = true := default_parameters_match w parameters
  cases found : programs.select w row.predictions with
  | none => simp [found] at selected
  | some program =>
    refine ⟨program, rfl, ?_⟩
    change programs.run w op p row = program.run w op p row
    simp only [TrainingCEExecution.Programs.run, accepted, if_pos, found]

#axiomsClean source_default_plan
#axiomsClean source_default_width
#axiomsClean source_default_predictions
#axiomsClean default_storage_extent
#axiomsClean selected_all_default_steps
#axiomsClean default_softcap_launch
#axiomsClean scheduled_row_runs_fixed
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCELaunch
