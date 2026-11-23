import models.NanoGPT.Track1.TrainingEquiv

/-! A control-flow regression for the generated driver. This small runtime
records source sites. It exercises validation and loop exits without GPUs,
tensor evaluation or the FineWeb dataset. -/
namespace VeriTile.Bench.Tests.Track1Driver
open VeriTile.Bench.Models.NanoGPT.Track1
open VeriTile.Bench.Models.NanoGPT.Track1.Driver
open VeriTile.Triton.FP

structure Trace where
  step : Nat := 0
  warmup : Bool := false
  lines : List Nat := []
  replicatedUpdates : Nat := 0
  tiedAdds : Nat := 0
  tiedCopies : Nat := 0
  qkvRefreshes : Nat := 0
  qkvPending : Bool := false
  refreshEvents : List Nat := []
  headRefreshes : Nat := 0
  headPending : Bool := false
  anvilGramLaunches : Nat := 0
  anvilPreparations : Nat := 0
  anvilNormalizations : Nat := 0
  anvilUpdates : Nat := 0
  trainStream : Option FineWeb.ScheduledStream := none
  ngramSlots : Array Nat := #[]

def mark (site : Nat) (s : Trace) : Trace := {s with lines := site :: s.lines}

private def sampledFixture : SampledGatherInput where
  count := .first
  candidates := {
    ids := fun i => ⟨i.val, Nat.lt_trans i.isLt (by decide)⟩
    sorted := fun _ _ less => less }
  bank := fun address => UInt8.ofNat (address + 7)
  hopper := true

private def checkSampledCommit : SampledGatherRuntime Trace where
  prepare line s := some (some sampledFixture, mark line s)
  commit c layout pos s :=
    if h : 10239 < c.rows then
      let p := SampledRows.point c ⟨10239, h⟩ 3
      if layout p == ((7863555, 10), (40959, 10)) && pos 10239 == some ⟨10239, h⟩ then
        some (mark 999 s)
      else none
    else none

#guard (executeSampledGather checkSampledCommit candidateCalls 99 {}).map (·.lines) == some [999, 99]
#guard (executeSampledGather checkSampledCommit referenceCalls 99 {}).map (·.lines) == some [999, 99]

private def fineWebFixture : ByteArray := Id.run do
  let header := (#[(20240520 : Nat), 1, 602]).flatMap fun word =>
    (Array.range 4).map fun i => UInt8.ofNat (word / 256 ^ i)
  let tokens := (Array.range 602).flatMap fun i =>
    let token := i % 4
    #[UInt8.ofNat token, UInt8.ofNat (token / 256)]
  pure ⟨header ++ Array.replicate (1024 - header.size) 0 ++ tokens⟩

#guard FineWeb.hashRows #[65535, 4095, 50256, 0, 65535] ==
  some #[42301439, 8333271, 8372803, 12864848, 30671353, 84602879, 84602879, 72608940, 69824842, 56414001]
#guard (FineWeb.decodeShard (fineWebFixture.extract 0 (fineWebFixture.size - 1))).isNone

private def alignedFixture : ByteArray := Id.run do
  let header := (#[(20240520 : Nat), 1, 802]).flatMap fun word =>
    (Array.range 4).map fun i => UInt8.ofNat (word / 256 ^ i)
  let tokens := (Array.range 802).flatMap fun i =>
    let token := if i % 401 = 0 then 50256 else i % 4
    #[UInt8.ofNat token, UInt8.ofNat (token / 256)]
  pure ⟨header ++ Array.replicate (1024 - header.size) 0 ++ tokens⟩

def shardInput : SparseLanding.ServeInput :=
  ⟨3, fun row => Int32.ofNat (10 + row), 10, 3, fun row _ => UInt16.ofNat (0x4100 + row)⟩

def adamLaunchWords (step : Nat) : List Nat :=
  let p := NgramAdam.configure step ⟨16, 0.001, 0.11, 1e-10, 8⟩
  [p.beta2.raw, p.eps.raw, p.stepSize.raw, p.decay.raw, p.gradMul.raw, p.squareMul.raw]

#guard adamLaunchWords 335 == [0x3f733333, 0x2edbe6ff, 0x3d5ab687, 0x3b309e99, 0x3e000000, 0x3a4ccccd]
#guard adamLaunchWords 336 == [0x3f670a3d, 0x2edbe6ff, 0x3d823a1e, 0x3bb09e99, 0x3e000000, 0x3ac7ae14]

def valueAdamRuntime : ValueAdam.Runtime Trace where
  wait s :=
    if s.lines.head? == some 1103 then
      let ids : AdamEntries.Input := ⟨#[9, 10, 12], #[10, 10, 12], 1, 3, 10, by decide, by decide⟩
      some (⟨ids, fun i _ => if i.val = 0 then 0x3c00 else 1,
        fun i _ => if i.val < 2 then 0x3c00 else 0⟩, mark 1214 s)
    else none
  readAdam s :=
    if s.lines.head? == some 1214 then
      some (⟨2, 0.01, 0.2, 1.0, 8⟩,
        ⟨12, (fun _ _ => ⟨Value.ofBits .bf16 0x3f80, Value.ofBits .fp32 0x3e800000,
          Value.zero .fp32⟩), none⟩)
    else none
  order p _ := some ⟨fun _ => ⟨List.finRange p.ids.localCount, .refl _⟩,
    fun _ => ⟨List.finRange p.ids.peerCount, .refl _⟩⟩
  arithmetic := ⟨fun _ => .x1024, div Config.ieeeValue⟩
  commit memory launch output s :=
    if h : 2 < memory.rows then
      let r₀ : Fin memory.rows := ⟨0, by omega⟩
      let r₁ : Fin memory.rows := ⟨1, by omega⟩
      let r₂ : Fin memory.rows := ⟨2, h⟩
      if output.gradient r₀ 0 == 0x3600 && output.gradient r₁ 0 == 0 &&
          output.gradient r₂ 0 == 0 && launch.event == 3 &&
          (output.updated r₂ 0).first == Value.ofBits .fp32 0x3e400000 &&
          lt Config.ieeeValue (output.updated r₀ 0).weight (Value.ofBits .bf16 0x3f80) then
        some (mark 1210 s)
      else none
    else none

def ngramAdamRuntime : NgramAdam.Runtime Trace where
  wait s :=
    if s.lines.head? == some 1210 then
      let ids : AdamEntries.Input := ⟨#[8, 10, 12, 47], #[12, 13, 10], 1, 3, 10, by decide, by decide⟩
      some (⟨ids, fun i _ => if i.val = 0 then 0x3c00 else 0x3800,
        fun i _ => Value.ofBits .bf16 (if i.val = 0 then 0x3f00 else 0x3f80)⟩, mark 1213 s)
    else none
  readScalars s :=
    if s.lines.head? == some 1213 then some (⟨2, 0.01, 0.2, 1.0, 1⟩,
      ⟨fun _ => ⟨fun _ => Value.ofBits .bf16 0x3f80, Value.zero .fp32, 2⟩,
        fun _ => Value.zero .fp32⟩)
    else none
  order p _ := some ⟨⟨List.finRange p.ids.count, .refl _⟩,
    fun _ => ⟨List.finRange p.ids.localCount, .refl _⟩,
    fun _ => ⟨List.finRange p.ids.peerCount, .refl _⟩⟩
  native := ⟨div Config.ieeeValue, fun _ => Value.zero .fp32⟩
  commit _ launch output s :=
    match output.claims 0, output.claims 2 with
    | some root, some other =>
      let row := output.state.rows 0
      let untouched := output.state.rows 1
      if root.val == 4 && other.val == 2 &&
          output.gradient root 0 == Value.ofBits .bf16 0x4000 &&
          output.gradient other 0 == Value.ofBits .bf16 0x3f80 &&
          launch.event == 3 && output.state.betaHistory 3 == launch.beta2 &&
          row.stamp == 3 && lt Config.ieeeValue (row.weights 0) (Value.ofBits .bf16 0x3f80) &&
          untouched.stamp == 2 && untouched.weights 0 == Value.ofBits .bf16 0x3f80 then
        some (mark 1211 s)
      else none
    | _, _ => none

def rowRuntime : SparseLanding.Runtime Trace where
  waitIDs kind s := some (shardInput, mark (match kind with | .value => 1201 | .ngram => 1203) s)
  send kind rows s :=
    let site := match kind with | .value => 1201 | .ngram => 1203
    if s.lines.head? == some site && rows 2 0 == some 0x4102 && (rows 3 0).isNone then
      some (mark (site + 1) s)
    else none
  waitNgram s :=
    if s.lines.head? == some 1204 || s.lines.head? == some 1206 then
      some (⟨⟨3, 1, 2, by decide, by decide⟩, shardInput,
        fun row _ => if row = 0 then 0xbf80 else 0x8000, fun _ _ => 0x7fc0⟩, mark 1207 s)
    else none
  writeCache rows s :=
    if s.lines.head? == some 1207 && rows 0 0 == some 0xbf80 && rows 1 0 == some 0x4101 &&
        rows 2 0 == some 0x8000 && rows 3 0 == some 0x7fc0 then
      some (mark 1208 s)
    else none
  hasNext _ := some true
  valueAdam := valueAdamRuntime
  ngramAdam := ngramAdamRuntime
  readMoments s :=
    if s.lines.take 5 == [1204, 1203, 1202, 1201, 1211] then
      some (some ⟨#[8, 10, 11, 57], 1, 3, 10, 3, 3, fun _ => Value.ofBits .fp32 0x3f000000,
        fun row => if row = 0 then ⟨Value.ofBits .fp32 0x3f800000, 1⟩
          else ⟨Value.zero .fp32 true, if row = 1 then 3 else 0⟩⟩)
    else none
  writeMoments rows s :=
    if rows 0 == ⟨Value.ofBits .fp32 0x3e800000, 3⟩ &&
        rows 1 == ⟨Value.zero .fp32 true, 3⟩ && rows 2 == ⟨Value.zero .fp32 true, 0⟩ then
      some (mark 1205 s)
    else none
  consumeNext s := if s.lines.head? == some 1205 then some (mark 1206 s) else none
  makeLive s := if s.lines.take 3 == [1208, 1207, 1206] then some (mark 1209 s) else none
  forwardPull s := some (if s.step == 0 then .cold else if s.warmup then .ready else .restore)
  prepareColdNgram := some
  startColdNgram s := if s.lines.head? == some 1208 then some (mark 1212 s) else none
  prepareColdValue s := if s.lines.head? == some 1212 then some s else none
  completeColdValue s := if s.lines.head? == some 1202 then some s else none
  prepareRestore := some
  readSlots s :=
    let rows : Array Int := #[-1, 32766, 32768, 32770, 84602880]
    let query := if s.step == 0 then rows else rows ++ Array.replicate 32763 32768
    some (⟨s.step != 0, ⟨4, (32766 + ·), by intros; dsimp; omega, by intros; dsimp; omega⟩, query⟩, s)
  writeSlots slots s :=
    let exampleSlots := #[0, 0, 2, 4, 4]
    if (s.step == 0 && slots == exampleSlots) ||
        (s.step != 0 && slots.size == 32768 && slots.extract 0 5 == exampleSlots && slots.back? == some 2)
    then some (mark 90 {s with ngramSlots := slots}) else none

private def tiedPoint (mode : Bf16Transpose.Mode) : Bf16Transpose.Point mode :=
  ⟨⟨⟨0, by cases mode <;> decide⟩, ⟨0, by cases mode <;> decide⟩⟩, 63, 127⟩

private def tiedRuntime : Bf16Transpose.Runtime Trace where
  read mode s :=
    let cols := if mode == .add then 768 else 50304
    let rows := if mode == .add then 50304 else 768
    let src address := Value.ofBits .bf16 (if address == 2 * (63 * cols + 127) then 0x3f80 else 0)
    let dst address := Value.ofBits .bf16 (if address == 2 * (127 * rows + 63) then 0x4000 else 0)
    some (⟨true, Config.ieeeValue, src, dst⟩, s)
  write mode stores s :=
    let (address, value) := stores (tiedPoint mode)
    match mode with
    | .add =>
        if s.tiedAdds == s.tiedCopies && address == 2 * (127 * 50304 + 63) && value.raw == 0x4040 then
          some {s with tiedAdds := s.tiedAdds + 1}
        else none
    | .copy =>
        if s.tiedAdds == s.tiedCopies + 1 && address == 2 * (127 * 768 + 63) && value.raw == 0x3f80 then
          some {s with tiedCopies := s.tiedCopies + 1}
        else none

private def qkvRefreshRuntime : QKVRefreshRuntime Trace where
  read s :=
    let first address := Value.ofBits .bf16 (if address == 0 then 0x3f80 else 0)
    let arithmetic : QKVPack.Arithmetic :=
      ⟨cast .fp32 Config.ieeeValue, div Config.ieeeValue,
        fun a b => if lt Config.ieeeValue a b then a else b,
        fun a b => if lt Config.ieeeValue b a then a else b,
        FP8E4M3.clampConvert⟩
    some (⟨true, first, fun _ => Value.zero .bf16, fun lanes => lanes 0, arithmetic⟩, s)
  commit output s := do
    let amax := output.partials .narrow ⟨0, by decide⟩ 0
    let scale := output.scales .narrow ⟨0, by decide⟩
    let write ← output.packed .narrow ⟨⟨0, by decide⟩, ⟨0, by decide⟩, ⟨0, by decide⟩, 0, 0⟩
    if amax.1 == 0 && amax.2.raw == 0x3f800000 && scale.1 == 0 && scale.2.raw == 0x3b124925 &&
      write.rowOffset == 0 && write.transposedOffset == 0 && write.byte == 0x7e then
        some {s with qkvRefreshes := s.qkvRefreshes + 1, refreshEvents := 2 :: s.refreshEvents}
    else none

private def headRefreshRuntime : HeadRefreshRuntime Trace where
  read s :=
    let read address := Value.ofBits .bf16 (if address == 100611 then 0x3f80 else 0)
    some (⟨true, read, Value.ofBits .fp32 0x40400000⟩, s)
  commit stores s := do
    let out ← stores ⟨⟨0, by decide⟩, ⟨0, by decide⟩, ⟨0, by decide⟩, 2, 3⟩
    if out.rowOffset == 100611 && out.transposedOffset == 2306 && out.byte == 0x44 then
      some {s with headRefreshes := s.headRefreshes + 1, refreshEvents := 6 :: s.refreshEvents}
    else none

private def markRefresh (event : Nat) (s : Trace) : Trace :=
  {s with refreshEvents := event :: s.refreshEvents}

private def fp8RefreshRuntime : Fp8RefreshRuntime Trace where
  qkv := qkvRefreshRuntime
  head := headRefreshRuntime
  takeOwed line s :=
    let flag := if s.qkvPending then some s.headPending else none
    some (flag, mark line {s with qkvPending := false, headPending := false})
  waitAttention s := some (markRefresh 1 s)
  completeValueLanding s := some (markRefresh 3 s)
  waitBanks s := some (markRefresh 4 s)
  refreshAttention body := body
  refreshMLP _ refreshHead body s :=
    let state := markRefresh 5 s
    if refreshHead then body state else some state
  eagerMLP _ body line s := body (mark line s)
  capture _ qkv head line s := qkv (mark line s) >>= head

def runtime (intermediate : Bool) (rank : Fin 8 := 0) : Runtime Trace Nat where
  statement _ line _ s :=
    let s := mark line s
    match line with
    | 112 => some {s with qkvPending := true, headPending := s.step % 2 == 1}
    | _ => some s
  fp8Refresh := fp8RefreshRuntime
  sampledGather := {
    prepare := fun line s => some (none, mark line s)
    commit := fun _ _ _ _ => none }
  sparseRuntime := {
    readValue := fun _ =>
      let seen := fun i : Fin 7 => decide (4 ≤ i.val)
      some ⟨⟨3, 1, 2, by decide, by decide⟩, 7,
        (SparsePackets.valueEmbedWant 1 seen).get, SparsePackets.value_embed_want_distinct 1 seen,
        fun i _ => UInt16.ofNat (32768 + i)⟩
    sendValue := fun input packets s =>
      if h : 0 < input.cut.own ∧ 1 < input.cut.peers then
        if packets.own ⟨0, h.1⟩ 0 == 32773 &&
            packets.wire ⟨0, Nat.lt_trans (by decide) h.2⟩ 0 == 32772 &&
            packets.wire ⟨1, h.2⟩ 0 == 32774 then
          some {s with lines := 1101 :: s.lines} else none
      else none
    sendNgram := fun input packets s =>
      if h : 0 < input.cut.own ∧ 0 < input.cut.peers then
        if packets.own ⟨0, h.1⟩ 0 == 0xbc00 &&
            packets.wire ⟨0, h.2⟩ 0 == Value.ofBits .bf16 0x3f80 then
          some {s with lines := 1102 :: s.lines} else none
      else none
    clearValue := fun memory s =>
      if s.lines.head? == some 1102 && memory 4 0 == 0 && memory 3 0 == 32771 then
        some {s with lines := 1103 :: s.lines} else none }
  readNgramTape := fun s =>
    if s.lines.head? == some 1101 then
      let step (neg pos : Nat) : NgramAccumulation.Step 2 :=
        ⟨2, id, fun i _ => Value.ofBits .bf16 (if i.val = 0 then neg else pos),
          fun _ _ => ⟨List.finRange 2, .refl _⟩⟩
      some (⟨⟨2, 0, 1, by decide, by decide⟩, fun _ _ => 0x7e00,
        [step 0xbf80 0x3f80], step 0x8000 0⟩, s)
    else none
  sparseLanding := rowRuntime
  graphStage := {
    tryCaptured request s :=
      some (if s.warmup || request.rows.size != 32768 then none else
        some (.words ⟨.initial, fun _ i => UInt64.ofNat (i.val + 1)⟩))
    prepare s := some (.words .initial (fun field i =>
      match field, i with
      | .slots, i => UInt64.ofNat (s.ngramSlots[i.val]?.getD 0)
      | _, i => UInt64.ofNat (i.val + 1)), s)
    eagerCopy := some
    write shape words s :=
      match shape, words with
      | .initial, words =>
        let lastSlot := if s.step == 0 then 0 else 2
        if words .slots ⟨32767, by decide⟩ == lastSlot && words .mtp ⟨2, by decide⟩ == 3 &&
            words .prefix ⟨0, by decide⟩ == 1 then some (mark 100 s) else none
      | _, _ => none
    replay _ s := if s.lines.head? == some 100 then some s else none }
  graphBackward := {
    replay _ s := some ({
      anvil := [(1, 11)]
      adam := [
        ⟨2, .fp32, ⟨12, 1, fun _ => Value.ofBits .fp32 0x3f800000⟩,
          ⟨22, 1, fun _ => Value.ofBits .fp32 0x40000000⟩, false⟩,
        ⟨3, .bf16, ⟨13, 1, fun _ => Value.ofBits .bf16 0x3f80⟩,
          ⟨23, 1, fun _ => Value.ofBits .bf16 0x4000⟩, true⟩]
      ngram := ⟨14, 1, fun _ => Value.ofBits .bf16 0x8000⟩
      valueGrad := some {
        address := 24
        batch := ⟨1, 2, fun _ => ⟨1, by decide⟩, fun _ _ _ => Value.ofBits .bf16 0x3f80⟩
        prior := fun _ _ => 0
        trace := fun _ => (List.finRange 4).product (List.finRange 1)
        complete := fun _ => .refl _ }
      }, mark 103 s)
    writeValueGrad address words s :=
      if address == 24 && words 1 ⟨0, by decide⟩ == 0x3c00 && words 6 ⟨0, by decide⟩ == 0 &&
          words 7 ⟨767, by decide⟩ == 0x3c00 then some s else none
    linkGradient parameter address s :=
      if (parameter, address, s.lines.head?) == (1, 11, some 103) then some (mark 1041 s)
      else if (parameter, address, s.lines.head?) == (2, 22, some 1041) then some (mark 1042 s)
      else if (parameter, address, s.lines.head?) == (3, 23, some 1042) then some (mark 1043 s)
      else none
    writeAdam _ tensor s :=
      if tensor.address == 22 && (tensor.words 0).raw == 0x3f800000 && s.lines.head? == some 1043 then
        some (mark 1044 s)
      else if tensor.address == 23 && (tensor.words 0).raw == 0x4040 && s.lines.head? == some 1044 then
        some (mark 1045 s)
      else none
    publishSink gradient s :=
      if gradient.address == 14 && (gradient.words 0).raw == 0x8000 && s.lines.head? == some 1045 then
        some (mark 1046 s)
      else none
    readPending s :=
      if s.lines.head? == some 1046 then
        some (⟨⟨14, 1, fun _ => Value.ofBits .bf16 0x8000⟩, 1, fun i => 30 + i.val⟩, s)
      else none
    writeRing slot gradient s :=
      if slot.val == 1 && gradient.address == 31 && gradient.size == 1 &&
          (gradient.words 0).raw == 0x8000 then some (mark 1047 s) else none
    appendPending gradient s :=
      if (gradient.address, s.lines.head?) == (14, some 1046) then some (mark 104 s)
      else if (gradient.address, s.lines.head?) == (31, some 1047) then some (mark 104 s)
      else none }
  schedule := {
    iterations _ := some (1122, 20)
    install plan tables s :=
      if plan.total == 1194 && plan.splitStep == 1175 &&
          tables.mtp.length == 1195 && tables.prefixWeights.length == 1195 then some (mark 138 s) else none
    advance _ line step values s :=
      if step == s.step && values.batchSize == 131072 && values.sequenceLimit == 896 &&
          values.shortBlocks == 1 && values.longBlocks == 3 then some (mark line s) else none
    parameterGroups _ := some [⟨.anvil, 1⟩, ⟨.valueEmbedding, 1⟩, ⟨.other, 1⟩] }
  optimizerPosition s := some (s.step, s.warmup)
  tiedEmbedding := tiedRuntime
  optimizerStep values event replicated anvil tied line _ s := do
    if values.groups.length != 3 || values.controls.rate != 1 ||
        values.controls.updateAdam != (s.step % 2 == 1) || tied.isSome != values.controls.updateAdam then none else do
      let s := mark line s
      let sent ← event.map (·.launch s) |>.getD (some s)
      let sent ← tied.map (·.beforeHeadReduce sent) |>.getD (some sent)
      let next ← anvil .qk sent
      let next ← anvil .vo next
      let updated ← event.map (·.update next) |>.getD (some next)
      let updated ← replicated updated
      let updated ← anvil .mlp updated
      let updated ← tied.map (·.afterHeadGather updated) |>.getD (some updated)
      event.map (·.finish updated) |>.getD (some updated)
  replicatedAdam := {
    read s := some ([{
      format := .bf16
      native := ⟨fun _ => TailNorm.one, div Config.ieeeValue⟩
      sizes := [⟨3, 513⟩, ⟨1, 4⟩]
      scalars := fun i => ⟨TailNorm.zero, TailNorm.one, TailNorm.zero, TailNorm.one,
        TailNorm.zero, TailNorm.one, TailNorm.zero, if i == 3 then TailNorm.one else TailNorm.zero⟩
      gradient := fun _ => Value.ofBits .bf16 0x3f80
      words := fun _ => ⟨Value.ofBits .bf16 0x4000, TailNorm.zero, TailNorm.zero⟩
      eligible := true
    }], s)
    write outputs s := do
      let [output] := outputs | none
      if (output.words 512).parameter.raw == 0x3f80 &&
          (output.words 513).parameter.raw == 0x4000 then
        some {s with replicatedUpdates := s.replicatedUpdates + 1}
      else none }
  anvilGram := {
    readPreparation := fun _ s =>
      if s.anvilGramLaunches % 12 == 0 then some ({
        eligible := true
        grad := fun _ => Value.ofBits .bf16 0x3f80
        fast := fun _ => Value.ofBits .fp32 0x3f000000
        slow := fun _ => Value.ofBits .fp32 0x3e800000
        momentum := Value.ofBits .fp32 0x3f000000
        fastBeta := Value.ofBits .fp32 0x3f000000
        fastWeight := Value.ofBits .fp32 0x3f400000
      }, s) else none
    commitPreparation := fun bank out s =>
      if h : 0 < AnvilRails.cells bank then
        let i : Fin (AnvilRails.cells bank) := ⟨0, h⟩
        if (out.fast i).raw == 0x3f400000 && (out.slow i).raw == 0x3e87ae14 &&
            (out.prepared i).raw == 0x3f50 then
          some {s with anvilPreparations := s.anvilPreparations + 1}
        else none
      else none
    read := fun _ operation s =>
      let gramTurn := s.anvilGramLaunches % 2 == 0
      let valid := match operation with
        | .gram => gramTurn
        | .polynomial alpha beta =>
          match AnvilGram.coefficients[s.anvilGramLaunches % 12 / 2]? with
          | some c => !gramTurn && alpha == c.alpha && beta == c.beta
          | none => false
      if valid && s.anvilPreparations == s.anvilGramLaunches / 12 + 1 then
        some ({
          values := fun _ => Value.zero .bf16
          wgmma16 := fun acc _ => acc
          retainedOutput := fun _ => Value.ofBits .bf16 0x4000
          freshOutput := fun _ => Value.ofBits .bf16 0xbf80
        }, s)
      else none
    commit := fun bank _ stores s =>
      if stores.tiles.length == AnvilGram.compactCount bank then
        some {s with anvilGramLaunches := s.anvilGramLaunches + 1}
      else none
    readNormalization := fun _ s =>
      if s.anvilGramLaunches % 12 == 1 then
        let one := fun _ => Value.ofBits .bf16 0x3f80
        some (⟨true, one, one, div Config.ieeeValue⟩, s)
      else none
    commitNormalization := fun bank out s =>
      if h : 0 < bank.dimensions.1 ∧ 0 < bank.dimensions.2.1 * bank.dimensions.2.2 ∧
          0 < AnvilTrace.width bank * AnvilTrace.width bank then
        let b : Fin bank.dimensions.1 := ⟨0, h.1⟩
        let (d, dSquare, x, a) := if bank == .qk then (0x41866667, 0x438d1eba, 0x3d74, 0x3b68)
          else (0x41e8c9a2, 0x4453ae14, 0x3d0d, 0x3a9b)
        if (out.divisors b).denominator.raw == d && (out.divisors b).square.raw == dSquare &&
            (out.prepared b ⟨0, h.2.1⟩).raw == x &&
            (out.gram b ⟨0, h.2.2⟩).raw == a then
          some {s with anvilNormalizations := s.anvilNormalizations + 1}
        else none
      else none
    project := fun _ alpha s =>
      match AnvilGram.coefficients[(s.anvilGramLaunches - 1) % 12 / 2]? with
      | some c => if alpha == c.projection then some s else none
      | none => none }
  anvilEpilogue := {
    read bank s := some ({
      native := ⟨fun _ => TailNorm.one, div Config.ieeeValue⟩
      value := fun _ => Value.ofBits .bf16 0x3f80
      power := fun _ => bank.laneLength
      laneEnergy := fun _ => TailNorm.one
      preSum := fun _ => div Config.ieeeValue TailNorm.one bank.laneLength
      postSum := fun _ => TailNorm.one
      upper := fun _ => 0x4000
      mantissa := fun _ => 0
      slowRail := fun _ => TailNorm.one
      decay := TailNorm.zero
      learningRate := fun _ => TailNorm.one
    }, s)
    commit _ output s :=
      if output.upper 0 == 0x3f80 && output.mantissa 0 == 0 && output.laneEnergy 0 == TailNorm.one then
        some {s with anvilUpdates := s.anvilUpdates + 1}
      else none }
  readTailTick s := some {
    step := s.step, totalSteps := 2, valueEmbeddingUpdates := false
    elements := fun _ => 1, eligible := fun _ => false
    input := fun _ => {
      weight := fun _ => Value.zero .bf16
      first := fun _ => ⟨Value.zero .fp32, false⟩
      second := fun _ => ⟨Value.zero .fp32, false⟩ } }
  writeTailTick line output s :=
    let vo := output .vo 0
    if vo.1.seeded == (s.step == 1) && vo.2.seeded then some {s with lines := line :: s.lines} else none
  readTailShip _ := some {
    whole := fun _ _ => Value.zero .bf16
    shard := fun _ _ => Value.zero .bf16
    average := fun _ _ _ => TailNorm.one
    seeded := fun _ _ => true
    normMath := ⟨⟨1, 2048⟩, true, id⟩
    lerpContract := true }
  gatherTailShard _ _ s :=
    some (⟨0, 1, fun _ _ => Value.zero .bf16⟩, {s with lines := 1000 :: s.lines})
  writeTailShip line weights s :=
    if weights .valueEmbedding 0 == Value.ofBits .bf16 0x3f80 then
      some {s with lines := line :: s.lines} else none
  test line _ s :=
    let last := s.step == 2
    match line with
    | 108 => some (!s.warmup && s.step == 0, s)
    | 119 | 382 | 395 | 400 => some (false, s)
    | 246 => some (s.step == 0, s)
    | 249 => some (s.step == 1, s)
    | 326 => some (last || intermediate, s)
    | 329 | 380 => some (last, s)
    | _ => none
  iterable line _ s :=
    match line with
    | 168 | 242 | 356 | 364 => some ([0,1], s)
    | 322 => some ([0,1,2], s)
    | _ => none
  bind target v s :=
    if target == "(position, step)" then some {s with step := v, warmup := true}
    else if target == "step" then some {s with step := v, warmup := false}
    else some s
  enterContext _ := some
  exitContext _ := some
  enterStep _ := some
  exitStep := some
  enterMain := some
  exitMain := some
  trainingBatches := {
    readPlan _ s := some (⟨[alignedFixture, alignedFixture], [false, true], 1, 0, [0, 1]⟩, s)
    bindSource stream s := do
      if !stream.pending.all (fun (_, g) => g.globalTokens == 131072 && g.maxLength == 896) then none else do
        -- The fixture caches two small batches after checking the pinned geometry.
        let fixture := {stream with pending := [(0, ⟨300, 400⟩), (1, ⟨300, 400⟩)]}
        let (_, cached) ← fixture.peek 1 FineWeb.packTrainingReference FineWeb.hashRows
        some {s with trainStream := some cached}
    read s := s.trainStream.map fun stream => (s.step, stream, s)
    commit consume batch stream s :=
      if batch.inputs.size == 300 && batch.inputs[0]? == some 50256 &&
          batch.cumulative[1]? == some 300 && batch.ngramRows[0]? == some 42301439 then
        some (mark (if consume then 113 else 88) {s with trainStream := some stream})
      else none }
  validation := {
    shape _ := some (600, 300)
    bindCount count s := if count == 2 then some (mark 351 s) else none
    readFiles s := some (⟨[fineWebFixture], 1, 0⟩, s)
    bindBatches batches s := if batches.length == 2 then some (mark 353 s) else none
    nativeCE := ⟨fun _ _ => TailNorm.one, div Config.ieeeValue TailNorm.one⟩
    forward _ call input s :=
      if call.startsWith "model(batch.inputs, batch.targets," && s.lines.contains 353 &&
          input.inputs.size == 300 && input.inputs[1]? == some 1 && input.targets[0]? == some 1 &&
          input.cumulative[1]? == some 300 && input.ngramRows[0]? == some 42301439 then
        some (.projected {
          rows := 300, vocab := 4
          projection := fun _ _ => Value.ofBits .bf16 0xc0a0
          mask := fun _ _ => 0
          tile := .x64r4
          device := ⟨132, 2048⟩
        }, mark 366 s)
      else none
    exchange own s :=
      if own.raw == 0x3fb17218 then
        let ring : ValidationLoss.Ring := ⟨[1,2,3,4,5,6,7,0], by decide, by decide⟩
        some (⟨rank, ring, fun _ => Value.ofBits .fp32 0x40800000⟩, s)
      else none
    writeRoot value s := if value.raw == 0x406b1722 then some (mark 370 s) else none
    position s := if s.lines.head? == some 370 then some (s.step, 2) else none }

def observed (intermediate : Bool) : Option (List Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat) := do
  let result ← execute (runtime intermediate) candidateCalls {}
  let final ← result.finalValidation
  let sites := result.state.lines
  pure (result.validations.map Validation.step, final.step, final.loss.raw,
        (sites.filter (· == 100)).length,
        (sites.filter (· == 366)).length, (sites.filter (· == 392)).length,
        (sites.filter (· == 1000)).length, (sites.filter (· == 1103)).length,
        (sites.filter (· == 1208)).length, (sites.filter (· == 1209)).length)

-- Two warmup and two timed train_step invocations. No update at the
-- final validation iteration. Each validation consumes two batches.
#guard observed false == some ([2], 2, 0x406b1722, 4, 2, 2, 6, 2, 9, 2)
#guard observed true == some ([2,1,0], 2, 0x406b1722, 4, 6, 2, 6, 2, 17, 2)
#guard ((execute (runtime false 3) candidateCalls {}).map (·.validations.isEmpty)) == some true
#guard ((execute (runtime false) candidateCalls {}).map (·.state.replicatedUpdates)) == some 2
#guard ((execute (runtime false) candidateCalls {}).map (·.state.headRefreshes)) == some 3
#guard ((execute (runtime false) candidateCalls {}).map (·.state.qkvRefreshes)) == some 5
#guard ((execute (runtime false) referenceCalls {}).map (·.state.qkvRefreshes)) == some 5
#guard Id.run do
  let flush := compileStmt (runtime false) candidateCalls (fun _ _ => none)
    (.shared 96 "deferred_gathers.flush()")
  match flush {state := {qkvPending := true, headPending := true}, validations := []} with
  | some (.next result) =>
      result.state.qkvRefreshes == 1 && !result.state.qkvPending &&
      result.state.headRefreshes == 1 && !result.state.headPending &&
      result.state.refreshEvents == [6, 5, 4, 3, 2, 1]
  | _ => false
#guard Id.run do
  let flush := compileStmt (runtime false) candidateCalls (fun _ _ => none)
    (.shared 332 "deferred_gathers.flush_final()")
  match flush {state := {qkvPending := true, headPending := true}, validations := []} with
  | some (.next result) =>
      !result.state.qkvPending && !result.state.headPending && result.state.refreshEvents == [3, 4]
  | _ => false
#guard ((execute (runtime false) candidateCalls {}).map (fun r => (r.state.tiedAdds, r.state.tiedCopies))) == some (2, 2)
#guard ((execute (runtime false) referenceCalls {}).map (fun r => (r.state.tiedAdds, r.state.tiedCopies))) == some (2, 2)
#guard ((execute (runtime false) candidateCalls {}).map (·.state.anvilGramLaunches)) == some 144
#guard ((execute (runtime false) candidateCalls {}).map (·.state.anvilPreparations)) == some 12
#guard ((execute (runtime false) candidateCalls {}).map (·.state.anvilNormalizations)) == some 12
#guard ((execute (runtime false) candidateCalls {}).map (·.state.anvilUpdates)) == some 12

-- The vector path includes an unaligned head and a short tail.
#guard (ValidationLoss.batchMean ⟨Array.replicate 134 (Value.ofBits .fp32 0x3f800000), 1, ⟨132, 2048⟩⟩).raw == 0x3f800000
#guard MuddFusedPacking.Program.separate.launches.length == 11
#guard MuddFusedPacking.Program.fused.launches.length == 2
#guard MuddFusedPacking.Program.single.launches.length == 1

-- Five classes exercise a second online update and pad three slots.
private def fiveClassLoss (mask : Nat → UInt8) : TailNorm.F32 :=
  ValidationCE.rowLoss (runtime false).validation.nativeCE .x64r4
    (fun _ : Fin 5 => Value.ofBits .bf16 0xc0a0) mask (some 4)

#guard (fiveClassLoss fun _ => 0).raw == 0x3fce0210
#guard lt Config.ieeeValue (Value.ofBits .fp32 0x428c0000) (fiveClassLoss fun _ => 0x10)

#guard Id.run do
  let projection : ValidationCE.Projection := {
    rows := 1, vocab := 5
    projection := fun _ _ => Value.ofBits .bf16 0xc0a0
    mask := fun _ _ => 0
    tile := .x64r4
    device := ⟨132, 2048⟩ }
  let native := (runtime false).validation.nativeCE
  let output := (projection.bindTokens #[1] #[ValidationCE.ignoredTarget]).bind fun batch =>
    (ValidationCE.losses native batch).words[0]?.map (·.raw)
  return output == some 0 && (projection.bindTokens #[1] #[5]).isNone

-- Make both gradient products consume the saved activation and E5M2 bytes.
private def operandAt {m n : Nat} (view : TrainingHead.View m n UInt8) (row col : Nat) : UInt8 :=
  if hr : row < m then
    if hc : col < n then view ⟨row, hr⟩ ⟨col, hc⟩ else 0
  else 0

private def tinyHeadFor (tokens : Nat) : TrainingHead.ForwardPacket where
  tokens := tokens
  width := .sampled10
  gemm := {
    logit := fun product _ _ =>
      if operandAt product.left 0 0 == 0x49 && operandAt product.right 0 3 == 0x44 &&
          product.scaleA.raw == 0x3e649249 && product.scaleB.raw == 0x3b924925 &&
          product.fastAccumulation then 0x20 else 0
    gradient := fun product _ _ =>
      let valid := match product.leftType, product.rightType with
        | .e4m3fn, .e5m2 =>
          operandAt product.left 0 0 == 0x49 && operandAt product.right 0 3 == 0x35 &&
            product.scaleA.raw == 0x3e649249 && product.scaleB.raw == 0x395b6db7
        | .e5m2, .e4m3fn =>
          operandAt product.left 0 3 == 0x35 && operandAt product.right 3 0 == 0x30 &&
            product.scaleA.raw == 0x395b6db7 && product.scaleB.raw == 0x3b924925
        | _, _ => false
      Value.ofBits .bf16 (if valid && !product.fastAccumulation then 0x3f80 else 0) }
  native := ⟨id, id, id, id⟩
  input := {
    hidden := fun i => Value.ofBits .bf16 (if i.val == 0 then 0x3f80 else 0)
    activationFast := true
    weightColumns := fun i => if i == 3 * 768 then 0x44 else 0
    weightRows := fun i => if i == 3 then 0x30 else 0
    weightTarget := .sampled .first rfl fun cls => if cls.val = 2 then some ⟨3, by decide⟩ else none
    parameters := ⟨TrainingCE.Softcap.aBits, TrainingCE.Softcap.bBits, TrainingCE.Softcap.cBits, 0,
      Value.zero .fp32⟩
    predictions := 2
    weights := fun _ => Value.ofBits .fp32 0x3f800000
    targets := fun _ => ⟨3, by decide⟩
    prefixTargets := fun _ => none }

private def tinyHead := tinyHeadFor 128

private def headTestCE : TrainingCE.Run := fun w _ parameters row =>
  let column : Fin w.columns := ⟨3, by cases w <;> decide⟩
  if row.logits column == 0x20 && parameters.gradScale.raw == 0x395b6db7 then
    ⟨Value.ofBits .fp32 0x3f800000, fun _ => 0x35⟩
  else ⟨Value.zero .fp32, fun _ => 0⟩

-- The graph must resume from the bank produced by the driver's coefficient call.
#guard Id.run do
  let old := runtime false
  let stage := { old.graphStage with head := some {
    prepare := fun _ => none
    publish := fun _ _ s => some (mark 200 s)
    mudd := some {
      read := fun s => some (⟨.initial, {
        mu := fun i => Value.ofBits .bf16 (if i == 9 then 0x4120 else 0)
        bias := fun i => Value.ofBits .bf16 (if i == 14 then 0x4120 else 0)
        delta := fun i => Value.ofBits .bf16 (if i == 16383 * 120 + 2 * 12 + 11 then 0x41a0 else 0)
      }⟩, s)
      resume := fun packet bank s => by
        rcases packet with ⟨shape, _⟩
        cases shape
        case initial =>
          let first : Fin MuddCoefficients.Shape.initial.plane := ⟨0, by decide⟩
          let lastGroup : Fin MuddCoefficients.Shape.initial.plane := ⟨11, by decide⟩
          let last : Fin MuddCoefficients.Shape.initial.plane := ⟨16384 * 12 - 1, by decide⟩
          exact if (bank 0 first).raw == 0x3f80 &&
              (bank 9 lastGroup).raw == 0x3f80 && (bank 2 last).raw == 0x4000
            then some (tinyHead, mark 202 s) else none
        all_goals exact none
    } } }
  let r := { old with graphStage := stage }
  let calls := { candidateCalls with trainingCE := headTestCE }
  let run := fun kernels =>
    compileStmt r kernels (fun _ _ => none) (.graphForward 99) { state := {}, validations := [] }
  let broken := { calls with muddCoefficients := fun _ _ _ _ => Value.zero .bf16 }
  match run calls with
  | some (.next state) =>
    return state.headTape.isSome && state.state.lines.take 2 == [200, 202] && (run broken).isNone
  | _ => return false

-- The fused norm supplies the FP8 head input and keeps the backward buffers.
#guard Id.run do
  let packet : MuddNorm.Packet := {
    shape := .initial
    projection := {
      mu := fun _ => Value.zero .bf16
      bias := fun i => Value.ofBits .bf16 (if i == 14 then 0x4120 else 0)
      delta := fun _ => Value.zero .bf16 }
    residual := fun _ _ => Value.zero .bf16
    sources := fun plane row col =>
      Value.ofBits .bf16 (if plane.val == 0 && row.val == 0 && col.val == 0 then 0x3f80 else 0) }
  let native : MuddNorm.Native := ⟨fun x _ => x, fun _ => Value.ofBits .fp32 0x3f800000,
    fun x => if x.raw == 0x408f5c29 then 0x49 else 0⟩
  let old := runtime false
  let stage := { old.graphStage with head := some {
    prepare := fun _ => none
    publish := fun head loss state =>
      if positive : 0 < head.tokens then
        if (loss ⟨0, positive⟩).raw == 0x3f800000 then some (mark 220 state) else none
      else none
    muddNorm := some {
      read := fun state => some (packet, native, .withQuantization, state)
      resume := fun p coefficients mixedResidual state => do
        if row : 0 < p.shape.rows then
          if group : 0 < p.shape.plane then
            if (coefficients 0 ⟨0, group⟩).raw == 0x3f80 &&
                (mixedResidual ⟨0, row⟩ 0).raw == 0x3f800000 then
              return (tinyHeadFor p.shape.rows, mark 221 state)
        none
    } } }
  let r := { old with graphStage := stage }
  let calls := { candidateCalls with trainingCE := headTestCE, headActivation := fun _ _ _ _ _ => 0 }
  let run := fun kernels =>
    compileStmt r kernels (fun _ _ => none) (.graphForward 99) { state := {}, validations := [] }
  let broken := { calls with muddNorm := fun ops p =>
    { (calls.muddNorm ops p) with activations := fun _ _ => 0 } }
  match run calls with
  | some (.next state) =>
    let some tape := state.headTape | return false
    let some saved := tape.saved.norm | return false
    if live : 0 < tape.tokens then
      return (saved.inverse ⟨0, live⟩).raw == 0x3f800000 &&
        state.state.lines.take 2 == [220, 221] && (run broken).isNone
    else return false
  | _ => return false

#guard Id.run do
  let packet := tinyHead
  let captured := packet.run (TrainingHead.forward HeadActivation.fused headTestCE)
  let shape : CETranspose.Shape := ⟨1, 80, by decide, by decide⟩
  let grads := TrainingHead.backward candidateCalls.ceTranspose candidateCalls.sampledDensify
    .sampled10 packet.gemm true true shape rfl captured.saved
  return (captured.loss ⟨0, by decide⟩).raw == 0x3f800000 &&
    (grads.activation ⟨0, by decide⟩ 0).raw == 0x3f80 &&
    (grads.weight 0 2).raw == 0x3f80 && (grads.weight 0 3).raw == 0

#guard Id.run do
  let packet := tinyHead
  let one := Value.ofBits .fp32 0x3f800000
  let residual : MuddResidualStorage.Packet packet.tokens := {
    native := ⟨fun x _ => x, fun _ => one, fun x => if x.raw == 0x408f5c29 then 0x49 else 0⟩
    block := 1024
    mixed := fun _ _ => Value.ofBits .fp32 0x3f800001
    sumSquares := fun _ => one
    base := 32
    memory := fun _ => Value.ofBits .bf16 0x4000 }
  let packet := { packet with input := { packet.input with residual := some residual } }
  let result := packet.run (TrainingHead.forward (fun _ _ _ _ _ => 0) headTestCE candidateCalls.muddResidual)
  match result.saved.residualMemory with
  | some memory => return (result.loss ⟨0, by decide⟩).raw == 0x3f800000 &&
      (memory 32).raw == 0x3f80 && (memory 30).raw == 0x4000
  | none => return false

-- The forward hook publishes the loss and the backward consumes its returned tape.
#guard Id.run do
  let norm : MuddNorm.Saved tinyHead.tokens :=
    ⟨.withQuantization, ⟨fun x _ => x, id, fun _ => 0x49⟩,
      fun _ _ => Value.zero .fp32, fun _ => Value.ofBits .fp32 0x40000000, fun _ _ => 0x49, rfl⟩
  let stage := (runtime false).graphStage
  let stage := { stage with head := some {
    prepare := fun s => some (⟨tinyHead.tokens, tinyHead.width, tinyHead.gemm, tinyHead.native,
      tinyHead.input, some norm⟩, s)
    publish := fun packet loss s =>
      if positive : 0 < packet.tokens then
        if (loss ⟨0, positive⟩).raw == 0x3f800000 then some (mark 200 s) else none
      else none } }
  let original := (runtime false).graphBackward
  let shape : CETranspose.Shape := ⟨1, 80, by decide, by decide⟩
  let valid (tape : TrainingHead.Tape) (grads : TrainingHead.Gradients tape.tokens) : Bool :=
    if tokens : 0 < tape.tokens then
      (grads.activation ⟨0, tokens⟩ 0).raw == 0x4000 && (grads.weight 0 2).raw == 0x3f80
    else false
  let headRuntime : GraphBackward.HeadRuntime Trace := {
    prepare := fun tape s =>
      if rows : (shape.operand .gradient).rows = tape.tokens then
        if cols : (shape.operand .gradient).cols = tape.width.columns then
          some (⟨true, true, shape, rows, cols⟩, s)
        else none
      else none
    resume := fun tape grads s =>
      if valid tape grads then original.replay candidateCalls.ceTranspose (mark 201 s) else none }
  let columnRuntime : GraphBackward.MuddGradientRuntime Trace := {
    prepare := fun tape grads s =>
      if valid tape grads then some (.columns 1 6 {
        input := fun plane _ _ => Value.ofBits .bf16 (if plane.val == 0 then 0x3f80 else 0x4000)
        reduce := fun tile => tile 0
        base := 64
        first := 4
        memory := fun _ => Value.ofBits .bf16 0xc000 }, s)
      else none
    resume := fun _ memory s =>
      if (memory 70).raw == 0xc000 && (memory 72).raw == 0x3f80 && (memory 74).raw == 0x4000 then
        original.replay candidateCalls.ceTranspose (mark 202 s)
      else none }
  let forward := TrainingHead.forward HeadActivation.fused headTestCE
  let gradients := TrainingHead.backward candidateCalls.ceTranspose candidateCalls.sampledDensify
  let run (runtime : GraphBackward.HeadRuntime Trace) (reduce : MuddGradientGrouping.Run)
      (packing : MuddGradientPacking.Run)
      (fused : MuddFusedPacking.Run := MuddFusedPacking.Program.separate.run)
      (fullPacking : MuddFullPacking.Run := MuddFullPacking.reference) :=
    (GraphStage.replayForward stage headTestCE forward {}).bind fun (s, saved) =>
      GraphBackward.replayBackward { original with head := some runtime } candidateCalls.ceTranspose
        gradients saved s reduce packing fused fullPacking
  let packet : MuddGradientPacking.Packet 1 := {
        source := fun plane => 1024 + 32 * plane.val
        columnsBase := 64
        packedBase := 128
        memory := fun addr => Value.ofBits .bf16
          (if addr == 1312 then 0x3f80 else if addr == 1026 then 0x4000 else if addr == 7 then 0xc000 else 0)
        reduce := fun tile => tile 0
        tile := 1
        warps := 2
        columnsDisjoint := by
          intro _ _ owned src
          simp only [MuddGradientPacking.inWindow] at *
          omega
        packingDisjoint := by
          intro _ _ owned src
          simp only [MuddGradientPacking.inWindow] at *
          omega
        outputsDisjoint := by
          intro _ owned packed
          simp only [MuddGradientPacking.inWindow] at *
          omega }
  let packingRuntime : GraphBackward.MuddGradientRuntime Trace := {
    prepare := fun tape grads s =>
      if valid tape grads then some (.packing 1 packet, s)
      else none
    resume := fun _ memory s =>
      if (memory 82).raw == 0x3f80 && (memory 344).raw == 0x3dcd && (memory 7).raw == 0xc000 then
        original.replay candidateCalls.ceTranspose (mark 203 s)
      else none }
  let fusedRuntime : GraphBackward.MuddGradientRuntime Trace := {
    prepare := fun tape grads s =>
      if valid tape grads then some (.fusedPacking 1 {
        group := packet
        zeroReduce := fun tile => tile 1
        zeroTile := 8
        zeroWarps := 2 }, s)
      else none
    resume := fun _ memory s =>
      if (memory 64).raw == 0x4000 && (memory 130).raw == 0x3e4d &&
          (memory 82).raw == 0x3f80 && (memory 344).raw == 0x3dcd && (memory 7).raw == 0xc000 then
        original.replay candidateCalls.ceTranspose (mark 204 s)
      else none }
  let ordinary := run headRuntime referenceCalls.muddGradients referenceCalls.muddPacking
  let grouped := run { headRuntime with muddGradients := some columnRuntime }
    candidateCalls.muddGradients candidateCalls.muddPacking
  let packed := run { headRuntime with muddGradients := some packingRuntime }
    candidateCalls.muddGradients candidateCalls.muddPacking
  let fused := run { headRuntime with muddGradients := some fusedRuntime }
    candidateCalls.muddGradients candidateCalls.muddPacking candidateCalls.muddFusedPacking
  let separate := run { headRuntime with muddGradients := some fusedRuntime }
    referenceCalls.muddGradients referenceCalls.muddPacking referenceCalls.muddFusedPacking
  let single := run { headRuntime with muddGradients := some fusedRuntime }
    candidateCalls.muddGradients candidateCalls.muddPacking MuddFusedPacking.Program.single.run
  let selected (groupDirect zeroDirect : Bool) := run { headRuntime with muddGradients := some fusedRuntime }
    candidateCalls.muddGradients candidateCalls.muddPacking
    (fun _ p => MuddFusedPacking.Program.fused.runSelected p groupDirect zeroDirect)
  let projection : MuddFullPacking.ScaledProjection packet := {
    scale := {
      base := 512
      columnsDisjoint := by
        intro _ scaled column
        simp only [packet, MuddGradientPacking.inWindow] at *
        omega
      packedDisjoint := by
        intro _ scaled packed
        simp only [packet, MuddGradientPacking.inWindow] at *
        omega
      sourcesDisjoint := by
        intro _ _ scaled source
        simp only [packet, MuddGradientPacking.inWindow] at *
        omega }
    stores := (· == 600)
    result := fun words _ => words 0
    packedIndependent := by
      intro addr store packed
      simp only [beq_iff_eq] at store
      simp only [packet, MuddGradientPacking.inWindow] at packed
      omega
    inputsIndependent := by
      intro _ addr store source
      simp only [beq_iff_eq] at store
      simp only [packet, MuddGradientPacking.inWindow] at source
      omega }
  let testFull (zeroTile : Nat) (fullPacking : MuddFullPacking.Run) :=
    let gradients : GraphBackward.MuddGradientRuntime Trace := {
      prepare := fun tape grads s => if valid tape grads then some (.fullPacking 1 {
        group := packet
        zeroReduce := fun tile => tile 1
        zeroTile
        zeroWarps := 2 } projection, s) else none
      resume := fun _ memory s =>
        if (memory 64).raw == 0x4000 && (memory 82).raw == 0x3f80 && (memory 130).raw == 0x3e4d &&
            (memory 344).raw == 0x3dcd && (memory 512).raw == 0x3e4d && (memory 530).raw == 0x3dcd &&
            (memory 600).raw == 0x3e4d && (memory 7).raw == 0xc000 then
          original.replay candidateCalls.ceTranspose (mark 205 s)
        else none }
    run { headRuntime with muddGradients := some gradients }
      candidateCalls.muddGradients candidateCalls.muddPacking candidateCalls.muddFusedPacking fullPacking
  return ordinary.map (·.2.lines) == some [103, 201, 200] && grouped.map (·.2.lines) == some [103, 202, 200] &&
    packed.map (·.2.lines) == some [103, 203, 200] && fused.map (·.2.lines) == some [103, 204, 200] &&
    separate.map (·.2.lines) == some [103, 204, 200] && single.map (·.2.lines) == some [103, 204, 200] &&
    (selected true false).map (·.2.lines) == some [103, 204, 200] &&
    (selected false true).map (·.2.lines) == some [103, 204, 200] &&
    (testFull 1 referenceCalls.muddFullPacking).map (·.2.lines) == some [103, 205, 200] &&
    (testFull 1 candidateCalls.muddFullPacking).map (·.2.lines) == some [103, 205, 200] &&
    (testFull 8 candidateCalls.muddFullPacking).map (·.2.lines) == some [103, 205, 200]

end VeriTile.Bench.Tests.Track1Driver
