import models.NanoGPT.Track1.GraphMemberLaunch

/-! Decode both branches of the five specialized SM90a kernels. The
instruction stream includes the vector copies, scalar predicates, all
eight lookup rounds and the two slot stores. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphSlotPTXBinding
open GraphStage GraphStageMemory CudaStores

def normalize (ptx : String) : List String :=
  let line (s : String) :=
    let code := (((s.splitOn "//").headD "").replace "\t" " ").replace "\r" ""
    String.intercalate " " ((code.splitOn " ").filter (!·.isEmpty))
  ((ptx.splitOn "\n").map line |>.takeWhile (!·.startsWith ".section .debug_")).filter
    (fun l => !l.isEmpty && !(l.startsWith ".loc " && !l.contains ';') && !l.startsWith ".file ")

private def param (i : Nat) : String := s!"stage_step_with_lookup_param_{i}"

private def entry (s : Shape) (member : Bool) : List String :=
  [".version 8.7", ".target sm_90a", ".address_size 64", ".visible .entry stage_step_with_lookup("] ++
  (List.range 17).map (fun i => s!".param .u64 .ptr .global .align 1 {param i}" ++
    if i < 16 then "," else "") ++
  [")", ".reqntid 128", "{",
    s!".reg .pred %p<{(if s.mtp = 1 then 22 else 24) - (if member then 8 else 0 : Nat)}>;",
    s!".reg .b32 %r<{(if s.mtp = 1 then 81 else 83) - (if member then 15 else 0 : Nat)}>;",
    ".reg .b64 %rd<44>;", "$L__func_begin0:",
    "mov.u32 %r1, %ctaid.x;", s!"setp.gt.u32 %p1, %r1, {s.blocks - 1};", "@%p1 bra $L__BB0_3;"]

private def r (i : Nat) : String := s!"%r{i}"
private def rd (i : Nat) : String := s!"%rd{i}"
private def pred (i : Nat) : String := s!"%p{i}"

private def select (out yes no test : String) : String := s!"selp.b32 {out}, {yes}, {no}, {test};"

private def copyBlock (s : Shape) : List String :=
  normalize r#"
ld.param.b64 %rd9, [stage_step_with_lookup_param_9];
ld.param.b64 %rd8, [stage_step_with_lookup_param_8];
ld.param.b64 %rd2, [stage_step_with_lookup_param_1];
ld.param.b64 %rd1, [stage_step_with_lookup_param_0];
$L__tmp0:
shl.b32 %r70, %r1, 8;
mov.u32 %r2, %tid.x;
and.b32 %r3, %r2, 127;
shl.b32 %r71, %r3, 1;
or.b32 %r72, %r71, %r70;
mul.wide.u32 %rd34, %r72, 4;
add.s64 %rd29, %rd8, %rd34;
add.s64 %rd28, %rd1, %rd34;
mov.u32 %r68, 0x0;
mov.u32 %r69, 0x0;
ld.global.v2.b32 { %r68, %r69 }, [ %rd28 + 0 ];
st.global.v2.b32 [ %rd29 + 0 ], { %r68, %r69 };
mul.wide.u32 %rd35, %r72, 8;
add.s64 %rd33, %rd9, %rd35;
add.s64 %rd32, %rd2, %rd35;
mov.u64 %rd30, 0x0;
mov.u64 %rd31, 0x0;
ld.global.v2.b64 { %rd30, %rd31 }, [ %rd32 + 0 ];
st.global.v2.b64 [ %rd33 + 0 ], { %rd30, %rd31 };
setp.ne.b32 %p18, %r1, 0;
@%p18 bra $L__BB0_4;
ld.param.b64 %rd41, [stage_step_with_lookup_param_13];
ld.param.b64 %rd12, [stage_step_with_lookup_param_12];
ld.param.b64 %rd10, [stage_step_with_lookup_param_10];
ld.param.b64 %rd40, [stage_step_with_lookup_param_5];
ld.param.b64 %rd5, [stage_step_with_lookup_param_4];
ld.param.b64 %rd3, [stage_step_with_lookup_param_2];
bfe.u32 %r77, %r2, 5, 2;
and.b32 %r78, %r2, 31;
"# ++ (if s.ends = 128 then [] else [s!"setp.lt.u32 %p19, %r3, {s.ends};"]) ++
  ["mul.wide.u32 %rd42, %r3, 4;",
    "add.s64 %rd37, %rd10, %rd42;", "add.s64 %rd36, %rd3, %rd42;"] ++
  (if s.ends = 128 then ["mov.pred %p19, -1;"] else []) ++
  ["mov.u32 %r73, 0x0;",
    "@%p19 ld.global.b32 { %r73 }, [ %rd36 + 0 ];",
    "@%p19 st.global.b32 [ %rd37 + 0 ], { %r73 };",
    "and.b32 %r79, %r2, 3;",
    (match s with
    | .initial => "setp.ne.b32 %p20, %r79, 3;"
    | .middle => "setp.lt.u32 %p20, %r79, 2;"
    | .bulk | .taper | .extension => "setp.eq.b32 %p20, %r79, 0;"),
    "mul.wide.u32 %rd43, %r79, 4;",
    "add.s64 %rd39, %rd12, %rd43;", "add.s64 %rd38, %rd5, %rd43;",
    "mov.b32 %r74, 0;", "mov.u32 %r75, %r74;",
    "@%p20 ld.global.b32 { %r75 }, [ %rd38 + 0 ];"] ++
  (if s.mtp = 1 then ["or.b32 %r80, %r78, %r77;", "setp.eq.b32 %p21, %r80, 0;"]
   else ["and.b32 %r80, %r2, 28;", "or.b32 %r81, %r77, %r80;",
     "setp.eq.b32 %p23, %r81, 0;", "and.pred %p21, %p20, %p23;"]) ++
  ["@%p21 st.global.b32 [ %rd39 + 0 ], { %r75 };",
    "mov.u32 %r76, 0x0;", "ld.global.b32 { %r76 }, [ %rd40 + 0 ];"] ++
  (if s.mtp = 1 then ["@%p21 st.global.b32 [ %rd41 + 0 ], { %r76 };"]
   else ["or.b32 %r82, %r77, %r78;", "setp.eq.b32 %p22, %r82, 0;",
     "@%p22 st.global.b32 [ %rd41 + 0 ], { %r76 };"]) ++
  ["bra.uni $L__BB0_4;", "$L__tmp1:"]

private def probeBlock (round : Nat) : List String :=
  let lo := r (if round = 0 then 4 else 20 + 6 * round)
  let hi := r (if round = 0 then 5 else 21 + 6 * round)
  let active := pred (2 + round)
  let higher := pred (10 + round)
  let value := r (if round = 0 then 6 else 7 + round)
  let address := rd (17 + round)
  let scratch (i : Nat) := r (22 + 6 * round + i)
  let difference := scratch 0
  let half := scratch 1
  let mid := scratch 2
  let afterMid := scratch 3
  let nextLo := if round = 7 then r 15 else scratch 4
  [s!"setp.gt.s32 {active}, {hi}, {lo};",
    s!"sub.s32 {difference}, {hi}, {lo};",
    s!"shr.u32 {half}, {difference}, 1;",
    s!"add.s32 {mid}, {half}, {lo};",
    s!"mad.wide.s32 {address}, {mid}, 4, %rd6;",
    s!"mov.u32 {value}, %r7;",
    s!"@{active} ld.global.b32 \{ {value} }, [ {address} + 0 ];",
    s!"setp.lt.s32 {higher}, {value}, %r7;",
    s!"add.s32 {afterMid}, {mid}, 1;",
    select nextLo afterMid lo higher] ++
  if round = 7 then [] else
    [select (scratch 5) hi mid higher]

private def memberProbeBlock (round : Nat) : List String :=
  let lo := r (if round = 0 then 5 else 22 + 4 * round)
  let test := pred (2 + round)
  let value := r (7 + round)
  let address := rd (17 + round)
  let distance := 128 / 2 ^ round
  let scratch (i : Nat) := r (23 + 4 * round + i)
  let candidate := if round = 7 then lo else scratch 0
  let mid := if round = 7 then scratch 0 else scratch 1
  (if round = 7 then [] else [s!"add.s32 {candidate}, {lo}, {distance - 1};"]) ++
  [s!"min.u32 {mid}, {candidate}, %r22;",
    s!"mad.wide.u32 {address}, {mid}, 4, %rd6;",
    s!"mov.u32 {value}, 0x0;",
    s!"ld.global.b32 \{ {value} }, [ {address} + 0 ];",
    s!"setp.lt.s32 {test}, {value}, %r4;"] ++
  if round = 7 then [select (scratch 1) "1" "0" test, s!"add.s32 %r15, {lo}, {scratch 1};"]
  else [s!"add.s32 {scratch 2}, {lo}, {distance};", select (scratch 3) (scratch 2) lo test]

private def lookupBlock (s : Shape) (member : Bool) : List String :=
  let row := r (if member then 4 else 7)
  let begin := r (if member then 5 else 4)
  let stop := r (if member then 6 else 5)
  ["$L__BB0_3:",
    s!"ld.param.b64 %rd13, [{param 14}];",
    s!"ld.param.b64 %rd11, [{param 11}];",
    s!"ld.param.b64 %rd7, [{param 7}];",
    s!"ld.param.b64 %rd6, [{param 6}];",
    s!"ld.param.b64 %rd4, [{param 3}];",
    "shl.b32 %r16, %r1, 7;",
    "mov.u32 %r17, %tid.x;",
    "and.b32 %r18, %r17, 127;",
    "or.b32 %r19, %r16, %r18;",
    s!"add.s32 %r20, %r19, -{128 * s.blocks};",
    "mul.wide.s32 %rd27, %r20, 4;",
    "add.s64 %rd14, %rd4, %rd27;",
    s!"mov.u32 {row}, 0x0;",
    s!"ld.global.b32 \{ {row} }, [ %rd14 + 0 ];",
    "$L__tmp2:",
    s!"shr.u32 %r21, {row}, 15;",
    "mad.wide.u32 %rd15, %r21, 4, %rd7;",
    s!"mov.u32 {begin}, 0x0;",
    s!"ld.global.b32 \{ {begin} }, [ %rd15 + 0 ];",
    "add.s64 %rd16, %rd15, 4;",
    s!"mov.u32 {stop}, 0x0;",
    s!"ld.global.b32 \{ {stop} }, [ %rd16 + 0 ];",
    "$L__tmp3:"] ++
  (if member then ["add.s32 %r22, %r6, -1;"] else []) ++
  (List.range 8).flatMap (if member then memberProbeBlock else probeBlock) ++
  ["$L__tmp4:", "add.s64 %rd25, %rd11, %rd27;",
    "st.global.b32 [ %rd25 + 0 ], { %r15 };",
    "add.s64 %rd26, %rd13, %rd27;",
    "st.global.b32 [ %rd26 + 0 ], { %r15 };",
    "$L__BB0_4:", "ret;", "$L__tmp5:", "$L__func_end0:", "}"]

def recognizes (s : Shape) (ptx : String) : Bool :=
  normalize ptx == (entry s false ++ copyBlock s ++ lookupBlock s false)

private def memberCopyBlock (s : Shape) : List String :=
  (copyBlock s).map fun line =>
    let registers := (List.range 15).foldl (fun code i => code.replace (r (68 + i)) (r (53 + i))) line
    (List.range 6).foldl (fun code i => code.replace (pred (18 + i)) (pred (10 + i))) registers

def recognizesMember (s : Shape) (ptx : String) : Bool :=
  normalize ptx == (entry s true ++ memberCopyBlock s ++ lookupBlock s true)

def slotStores {s n} (layout : Layout s n) (index slot : Nat) : List Store :=
  [.graph .slots, .pending].flatMap fun out =>
    (List.range 4).map fun i =>
      ⟨layout.base (.destination out) + 4 * index + i, slotByte slot i⟩

private def lookupReadsWith {s n} (member : Bool) (layout : Layout s n) (memory : Memory)
    (index : Nat) : List (Buffer × Nat) :=
  let row := UInt32.ofNat (query layout memory index).toNat
  let bucket := (row >>> 15).toNat
  let offset := readWord32 layout memory .directory
  [(.source .slots, index), (.directory, bucket), (.directory, bucket + 1)] ++
    (if member then (GraphMemberLaunch.lookup layout memory index).2 else
      NgramSlotPTX.readTrace (fun i => UInt32.ofNat (readWord32 layout memory .want i))
        row 8 (NgramSlotPTX.startBounds offset row)).map (fun i => (.want, i))

def lookupReads {s n} (layout : Layout s n) (memory : Memory) (index : Nat) : List (Buffer × Nat) :=
  lookupReadsWith false layout memory index

def memberReads {s n} (layout : Layout s n) (memory : Memory) (index : Nat) : List (Buffer × Nat) :=
  lookupReadsWith true layout memory index

private theorem lookup_reads_with_inside {s n} (member : Bool) (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (word : Fin (extent s .slots))
    (present : member = true → NgramSlots.ContainsRow ready.want (query layout memory word.val).toNat)
    (read : Buffer × Nat) (issued : read ∈ lookupReadsWith member layout memory word.val) :
    4 * (read.2 + 1) ≤ read.1.bytes s n := by
  have inside := word.isLt
  have valid := (hashed_query_in_range layout memory batch hashed upload word).2
  have bucket := NgramSlots.bucket_indices (query layout memory word.val).toNat valid
  simp only [lookupReadsWith, List.mem_append, List.mem_cons, List.not_mem_nil, or_false, List.mem_map] at issued
  rcases issued with (rfl | rfl | rfl) | ⟨probe, issued, rfl⟩
  · simp only [Buffer.bytes, width]
    omega
  · simp only [Buffer.bytes, NgramSlotPTX.bucket_exact _ valid]
    omega
  · simp only [Buffer.bytes, NgramSlotPTX.bucket_exact _ valid]
    omega
  · have limit : probe < n := by
      cases member
      · exact NgramSlotPTX.lookup_reads_fit ready.want _ ready.directory fits _ valid probe issued
      · exact (NgramSlotPTX.member_lookup_exact ready.want _ ready.directory fits _ valid
          (present rfl)).2 probe issued
    change 4 * (probe + 1) ≤ n * 4
    omega

theorem lookup_reads_inside {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (word : Fin (extent s .slots)) (read : Buffer × Nat) (issued : read ∈ lookupReads layout memory word.val) :
    4 * (read.2 + 1) ≤ read.1.bytes s n :=
  lookup_reads_with_inside false layout memory ready fits batch hashed upload word (by simp) read issued

theorem member_reads_inside {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (members : GraphMemberLaunch.RowsPresent layout memory ready)
    (word : Fin (extent s .slots)) (read : Buffer × Nat) (issued : read ∈ memberReads layout memory word.val) :
    4 * (read.2 + 1) ≤ read.1.bytes s n :=
  lookup_reads_with_inside true layout memory ready fits batch hashed upload word (fun _ => members word) read issued

/-- Interpret one lane of the decoded lookup block. The caller supplies the
entry launch shape and addresses. The copy block has its own memory proof. -/
def interpretLane (s : Shape) (ptx : String) {n} (layout : Layout s n)
    (memory : Memory) (pid tid : Nat) : Option (List Store) :=
  if recognizes s ptx && decide (s.blocks ≤ pid) then
    let index := (NgramSlotPTX.signed (NgramSlotPTX.queryIndex s.blocks pid tid)).toNat
    some (slotStores layout index (lookupHashed layout memory index))
  else none

private theorem lookup_cta_fits (s : Shape) (word : Fin (extent s .slots)) :
    queryCTA s word.val < 1024 := by
  have upper := (query_ownership s word).2.1
  cases s <;> simp only [Shape.blocks, Shape.tokens, queryBlocks, extent] at upper <;> omega

theorem lane_exact {s n} (ptx : String) (decoded : recognizes s ptx = true)
    (layout : Layout s n) (memory : Memory) (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (word : Fin (extent s .slots)) :
    interpretLane s ptx layout memory (queryCTA s word.val) (queryLane word.val) =
      some (slotStores layout word.val (NgramSlots.native false ready.want (query layout memory word.val))) := by
  obtain ⟨lookup, _, lane, index, _⟩ := query_ownership s word
  have raw := NgramSlotPTX.query_index_exact s.blocks (queryCTA s word.val)
    (queryLane word.val) lookup (lookup_cta_fits s word) lane
  have address : (NgramSlotPTX.signed (NgramSlotPTX.queryIndex s.blocks
      (queryCTA s word.val) (queryLane word.val))).toNat = word.val := by
    rw [raw, Int.toNat_natCast]
    omega
  have range := hashed_query_in_range layout memory batch hashed upload word
  have machine := NgramSlotPTX.lookup_exact ready.want _ ready.directory fits _ range.2
  have exactWord : lookupHashed layout memory word.val =
      NgramSlots.native false ready.want (query layout memory word.val) := by
    simpa only [lookupHashed, Ready.want, Int.toNat_of_nonneg range.1] using machine
  simp only [interpretLane, decoded, lookup, decide_true, Bool.true_and, if_true, address, exactWord]

/-- The recognized launch runs both branches for the supplied CTA order. -/
def interpretLaunch (s : Shape) (ptx : String) {n} (layout : Layout s n)
    (schedule : List Nat) (memory : Memory) : Option Memory :=
  if recognizes s ptx then some (GraphLaunchPTX.launch layout schedule memory) else none

theorem replay_exact {s n} (ptx : String) (decoded : recognizes s ptx = true)
    (layout : Layout s n) (memory : Memory) (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (schedule : List Nat) :
    interpretLaunch s ptx layout schedule memory = some (GraphStageMemory.launch layout schedule memory) := by
  simp only [interpretLaunch, decoded, if_true, GraphLaunchPTX.launch_equal,
    GraphStageMemory.hashed_launch_equal layout memory ready fits batch hashed upload schedule]

structure Certificate (s : Shape) (ptx : String) : Type where
  recognized : recognizes s ptx = true
  stores : ∀ (n : Nat) (layout : Layout s n) (memory : Memory) (ready : Ready layout memory)
    (_ : n < 2147483648) (batch : FineWeb.TokenBatch), batch.hashedRows →
    UploadedHashes layout memory batch → ∀ (word : Fin (extent s .slots)),
    interpretLane s ptx layout memory (queryCTA s word.val) (queryLane word.val) =
      some (slotStores layout word.val (NgramSlots.native false ready.want (query layout memory word.val)))
  reads : ∀ (n : Nat) (layout : Layout s n) (memory : Memory) (_ : Ready layout memory)
    (_ : n < 2147483648) (batch : FineWeb.TokenBatch), batch.hashedRows →
    UploadedHashes layout memory batch → ∀ (word : Fin (extent s .slots)) (read : Buffer × Nat),
    read ∈ lookupReads layout memory word.val → 4 * (read.2 + 1) ≤ read.1.bytes s n
  copyReads : ∀ (pid : Nat) (_ : pid < s.blocks) (tid : Fin 128) (word : GraphCopyPTX.Word),
    word ∈ GraphCopyPTX.words s pid tid false → word.2 < extent s word.1
  replay : ∀ (n : Nat) (layout : Layout s n) (memory : Memory) (_ : Ready layout memory)
    (_ : n < 2147483648) (batch : FineWeb.TokenBatch), batch.hashedRows →
    UploadedHashes layout memory batch → ∀ (schedule : List Nat),
    interpretLaunch s ptx layout schedule memory = some (GraphStageMemory.launch layout schedule memory)

def certify (s : Shape) (ptx : String) : Option (Certificate s ptx) :=
  if h : recognizes s ptx = true then
    some ⟨h, fun _ => lane_exact ptx h, fun _ => lookup_reads_inside,
      GraphCopyPTX.reads_bounded s, fun _ => replay_exact ptx h⟩
  else none

def interpretMemberLaunch (s : Shape) (ptx : String) {n} (layout : Layout s n)
    (schedule : List Nat) (memory : Memory) : Option Memory :=
  if recognizesMember s ptx then some (GraphMemberLaunch.launch layout schedule memory) else none

theorem member_replay_exact {s n} (ptx : String) (decoded : recognizesMember s ptx = true)
    (layout : Layout s n) (memory : Memory) (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (members : GraphMemberLaunch.RowsPresent layout memory ready) (schedule : List Nat) :
    interpretMemberLaunch s ptx layout schedule memory = some (GraphStageMemory.launch layout schedule memory) := by
  simp only [interpretMemberLaunch, decoded, if_true,
    GraphMemberLaunch.launch_equal layout memory ready fits batch hashed upload members,
    GraphLaunchPTX.launch_equal, GraphStageMemory.hashed_launch_equal layout memory ready fits batch hashed upload]

structure MemberCertificate (s : Shape) (ptx : String) : Type where
  recognized : recognizesMember s ptx = true
  reads : ∀ (n : Nat) (layout : Layout s n) (memory : Memory) (ready : Ready layout memory)
    (_ : n < 2147483648) (batch : FineWeb.TokenBatch), batch.hashedRows →
    UploadedHashes layout memory batch → GraphMemberLaunch.RowsPresent layout memory ready →
    ∀ (word : Fin (extent s .slots)) (read : Buffer × Nat),
    read ∈ memberReads layout memory word.val → 4 * (read.2 + 1) ≤ read.1.bytes s n
  copyReads : ∀ (pid : Nat) (_ : pid < s.blocks) (tid : Fin 128) (word : GraphCopyPTX.Word),
    word ∈ GraphCopyPTX.words s pid tid false → word.2 < extent s word.1
  replay : ∀ (n : Nat) (layout : Layout s n) (memory : Memory) (ready : Ready layout memory)
    (_ : n < 2147483648) (batch : FineWeb.TokenBatch), batch.hashedRows →
    UploadedHashes layout memory batch → GraphMemberLaunch.RowsPresent layout memory ready →
    ∀ (schedule : List Nat),
    interpretMemberLaunch s ptx layout schedule memory = some (GraphStageMemory.launch layout schedule memory)

def certifyMember (s : Shape) (ptx : String) : Option (MemberCertificate s ptx) :=
  if h : recognizesMember s ptx = true then
    some ⟨h, fun _ => member_reads_inside, GraphCopyPTX.reads_bounded s, fun _ => member_replay_exact ptx h⟩
  else none

#axiomsClean lane_exact
#axiomsClean lookup_reads_inside
#axiomsClean replay_exact
#axiomsClean member_reads_inside
#axiomsClean member_replay_exact
end VeriTile.Bench.Models.NanoGPT.Track1.GraphSlotPTXBinding
