import models.NanoGPT.Track1.GraphSlotPTXBinding

/-! The ordinary graph stage copies six buffers. Its final vector load and
store move already resolved slot IDs, including the sentinel values. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphPlainPTX
open GraphStage GraphStageMemory

private def entry (s : Shape) : List String :=
  [".version 8.7", ".target sm_90a", ".address_size 64", ".visible .entry stage_step("] ++
  (List.range 14).map (fun i => s!".param .u64 .ptr .global .align 1 stage_step_param_{i}" ++
    if i < 13 then "," else "") ++
  [")", ".reqntid 128", "{",
    s!".reg .pred %p<{if s.mtp = 1 then 5 else 7}>;",
    s!".reg .b32 %r<{if s.mtp = 1 then 24 else 26}>;",
    ".reg .b64 %rd<30>;", "$L__func_begin0:"]

private def inputs : List String := GraphSlotPTXBinding.normalize r#"
ld.param.b64 %rd5, [stage_step_param_9];
ld.param.b64 %rd2, [stage_step_param_3];
ld.param.b64 %rd13, [stage_step_param_0];
$L__tmp0:
mov.u32 %r1, %ctaid.x;
ld.param.b64 %rd14, [stage_step_param_1];
$L__tmp1:
shl.b32 %r6, %r1, 8;
mov.u32 %r2, %tid.x;
and.b32 %r3, %r2, 127;
shl.b32 %r7, %r3, 1;
or.b32 %r8, %r7, %r6;
ld.param.b64 %rd15, [stage_step_param_6];
ld.param.b64 %rd16, [stage_step_param_7];
mul.wide.s32 %rd17, %r8, 4;
add.s64 %rd8, %rd15, %rd17;
add.s64 %rd7, %rd13, %rd17;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v2.b32 { %r4, %r5 }, [ %rd7 + 0 ];
st.global.v2.b32 [ %rd8 + 0 ], { %r4, %r5 };
mul.wide.s32 %rd18, %r8, 8;
add.s64 %rd12, %rd16, %rd18;
add.s64 %rd11, %rd14, %rd18;
mov.u64 %rd9, 0x0;
mov.u64 %rd10, 0x0;
ld.global.v2.b64 { %rd9, %rd10 }, [ %rd11 + 0 ];
st.global.v2.b64 [ %rd12 + 0 ], { %rd9, %rd10 };
setp.ne.b32 %p1, %r1, 0;
@%p1 bra $L__BB0_2;
"#

private def scalar (s : Shape) : List String :=
  GraphSlotPTXBinding.normalize r#"
ld.param.b64 %rd24, [stage_step_param_11];
ld.param.b64 %rd6, [stage_step_param_10];
ld.param.b64 %rd4, [stage_step_param_8];
ld.param.b64 %rd23, [stage_step_param_5];
ld.param.b64 %rd3, [stage_step_param_4];
ld.param.b64 %rd1, [stage_step_param_2];
bfe.u32 %r13, %r2, 5, 2;
and.b32 %r14, %r2, 31;
"# ++ (if s.ends = 128 then [] else [s!"setp.lt.u32 %p2, %r3, {s.ends};"]) ++
  ["mul.wide.u32 %rd25, %r3, 4;", "add.s64 %rd20, %rd4, %rd25;", "add.s64 %rd19, %rd1, %rd25;"] ++
  (if s.ends = 128 then ["mov.pred %p2, -1;"] else []) ++
  ["mov.u32 %r9, 0x0;", "@%p2 ld.global.b32 { %r9 }, [ %rd19 + 0 ];",
    "@%p2 st.global.b32 [ %rd20 + 0 ], { %r9 };", "and.b32 %r15, %r2, 3;",
    (match s with
    | .initial => "setp.ne.b32 %p3, %r15, 3;"
    | .middle => "setp.lt.u32 %p3, %r15, 2;"
    | .bulk | .taper | .extension => "setp.eq.b32 %p3, %r15, 0;"),
    "mul.wide.u32 %rd26, %r15, 4;", "add.s64 %rd22, %rd6, %rd26;",
    "add.s64 %rd21, %rd3, %rd26;", "mov.b32 %r10, 0;", "mov.u32 %r11, %r10;",
    "@%p3 ld.global.b32 { %r11 }, [ %rd21 + 0 ];"] ++
  (if s.mtp = 1 then ["or.b32 %r16, %r14, %r13;", "setp.eq.b32 %p4, %r16, 0;"]
   else ["and.b32 %r16, %r2, 28;", "or.b32 %r17, %r13, %r16;",
     "setp.eq.b32 %p6, %r17, 0;", "and.pred %p4, %p3, %p6;"]) ++
  ["@%p4 st.global.b32 [ %rd22 + 0 ], { %r11 };", "mov.u32 %r12, 0x0;",
    "ld.global.b32 { %r12 }, [ %rd23 + 0 ];"] ++
  if s.mtp = 1 then ["@%p4 st.global.b32 [ %rd24 + 0 ], { %r12 };"]
  else ["or.b32 %r18, %r13, %r14;", "setp.eq.b32 %p5, %r18, 0;",
    "@%p5 st.global.b32 [ %rd24 + 0 ], { %r12 };"]

private def slots (s : Shape) : List String :=
  let first := if s.mtp = 1 then 17 else 19
  let thread := first + 5
  let block := first + 4
  let combined := first + 6
  let names := "{ " ++ String.intercalate ", " ((List.range 4).map fun d => s!"%r{first + d}") ++ " }"
  ["$L__tmp2:", "$L__BB0_2:", s!"shl.b32 %r{block}, %r1, 9;",
    s!"shl.b32 %r{thread}, %r3, 2;", s!"or.b32 %r{combined}, %r{thread}, %r{block};",
    s!"mul.wide.s32 %rd29, %r{combined}, 4;", "add.s64 %rd28, %rd5, %rd29;",
    "add.s64 %rd27, %rd2, %rd29;"] ++
  (List.range 4).map (fun d => s!"mov.u32 %r{first + d}, 0x0;") ++
  [s!"ld.global.v4.b32 {names}, [ %rd27 + 0 ];", s!"st.global.v4.b32 [ %rd28 + 0 ], {names};",
    "ret;", "$L__tmp3:", "$L__func_end0:", "}"]

def recognizes (s : Shape) (ptx : String) : Bool :=
  GraphSlotPTXBinding.normalize ptx == entry s ++ inputs ++ scalar s ++ slots s

def slotIndex (pid tid : UInt32) : UInt32 :=
  ((tid &&& 127) <<< 2) ||| (pid <<< 9)

theorem slot_index_exact (pid tid : Nat) (block : pid < 192) (lane : tid < 128) :
    (slotIndex (UInt32.ofNat pid) (UInt32.ofNat tid)).toNat = 512 * pid + 4 * tid := by
  have p : (UInt32.ofNat pid).toNat = pid := UInt32.toNat_ofNat_of_lt' (by change pid < 4294967296; omega)
  have t : (UInt32.ofNat tid).toNat = tid := UInt32.toNat_ofNat_of_lt' (by change tid < 4294967296; omega)
  rw [slotIndex, UInt32.toNat_or, UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, UInt32.toNat_and, p, t]
  change ((tid &&& 127) <<< 2) % 4294967296 ||| ((pid <<< 9) % 4294967296) = 512 * pid + 4 * tid
  have mask : tid &&& 127 = tid := Nat.and_two_pow_sub_one_of_lt_two_pow (n := 7) lane
  have pshift : pid <<< 9 = 512 * pid := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  have tshift : tid <<< 2 = 4 * tid := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  rw [mask, Nat.mod_eq_of_lt (by rw [tshift]; omega), Nat.mod_eq_of_lt (by rw [pshift]; omega), Nat.or_comm,
    ← Nat.shiftLeft_add_eq_or_of_lt (i := 9) (by rw [tshift]; omega), pshift, tshift]

def words (s : Shape) (pid : Nat) (tid : Fin 128) (stores : Bool) : List GraphCopyPTX.Word :=
  let slot := (slotIndex (UInt32.ofNat pid) (UInt32.ofNat tid.val)).toNat
  GraphCopyPTX.words s pid tid stores ++ (List.range 4).map (fun j => (.slots, slot + j))

def wordLane (f : Field) (word : Nat) : Nat :=
  if f = .slots then word % 512 / 4 else GraphCopyPTX.wordLane f word

private theorem shape_geometry (s : Shape) :
    s.blocks ≤ 192 ∧ 512 * s.blocks = extent s .slots := by cases s <;> decide

theorem reads_bounded (s : Shape) (pid : Nat) (live : pid < s.blocks) (tid : Fin 128)
    (word : GraphCopyPTX.Word) (read : word ∈ words s pid tid false) : word.2 < extent s word.1 := by
  unfold words at read
  rcases List.mem_append.mp read with read | tail
  · exact GraphCopyPTX.reads_bounded s pid live tid word read
  · obtain ⟨i, within, rfl⟩ := List.mem_map.mp tail
    have ⟨blocks, count⟩ := shape_geometry s
    have lane := tid.isLt
    rw [slot_index_exact pid tid.val (Nat.lt_of_lt_of_le live blocks) lane]
    have bound := List.mem_range.mp within
    change 512 * pid + 4 * tid.val + i < extent s .slots
    omega

theorem stores_read (s : Shape) (pid : Nat) (tid : Fin 128) (word : GraphCopyPTX.Word)
    (written : word ∈ words s pid tid true) : word ∈ words s pid tid false := by
  unfold words at written ⊢
  exact List.mem_append.mpr ((List.mem_append.mp written).imp (GraphCopyPTX.stores_read s pid tid word) id)

theorem stores_word_iff (s : Shape) (pid : Nat) (live : pid < s.blocks) (tid : Fin 128)
    (f : Field) (word : Nat) :
    (f, word) ∈ words s pid tid true ↔
      word < extent s f ∧ word / f.tile = pid ∧ wordLane f word = tid.val := by
  have ⟨blocks, count⟩ := shape_geometry s
  have lane := tid.isLt
  simp only [words, List.mem_append, GraphCopyPTX.stores_word_iff s pid live, List.mem_map,
    List.mem_range, Prod.mk.injEq, slot_index_exact pid tid.val (Nat.lt_of_lt_of_le live blocks) lane]
  by_cases isSlot : f = .slots
  · subst f
    simp only [ne_eq, not_true_eq_false, false_and, true_and, false_or, Field.tile, wordLane, if_true]
    constructor
    · rintro ⟨i, within, rfl⟩
      omega
    · rintro ⟨within, block, thread⟩
      refine ⟨(word % 512) % 4, Nat.mod_lt _ (by decide), ?_⟩
      have tile := Nat.div_add_mod' word 512
      have vector := Nat.div_add_mod' (word % 512) 4
      omega
  · have unequal := Ne.symm isSlot
    simp [isSlot, unequal, wordLane]

theorem word_lane_bound (s : Shape) (f : Field) (word : Fin (extent s f)) :
    wordLane f word.val < 128 := by
  have live := word.isLt
  cases f <;> cases s <;> simp [wordLane, GraphCopyPTX.wordLane, extent, Shape.ends, Shape.mtp] at live ⊢ <;> omega

private def readAfterStore {s : Shape} (input : StaticInputs s) (f : Field)
    (word : Fin (extent s f)) (pid : Nat) (tid : Fin 128) : Option UInt64 :=
  if (f, word.val) ∈ words s pid tid true && (f, word.val) ∈ words s pid tid false then
    some (input f word)
  else none

/-- Decode the emitting CTA and lane for each observable destination word. -/
def interpret (s : Shape) (ptx : String) (input : StaticInputs s) : Option (StaticInputs s) :=
  if recognizes s ptx then
    some fun f word =>
      (readAfterStore input f word (word.val / f.tile) ⟨wordLane f word.val, word_lane_bound s f word⟩).getD 0
  else none

theorem interpret_exact {s : Shape} {ptx : String} (decoded : recognizes s ptx = true)
    (input : StaticInputs s) : interpret s ptx input = some (eager s input) := by
  simp only [interpret, decoded, if_true, Option.some.injEq, eager]
  funext f word
  let pid := word.val / f.tile
  let tid : Fin 128 := ⟨wordLane f word.val, word_lane_bound s f word⟩
  have live : pid < s.blocks := by
    have bound := word.isLt
    cases f <;> cases s <;> simp [pid, extent, Field.tile, Shape.blocks, Shape.tokens, Shape.ends, Shape.mtp]
      at bound ⊢ <;> omega
  have store := (stores_word_iff s pid live tid f word.val).mpr ⟨word.isLt, rfl, rfl⟩
  have read := stores_read s pid tid (f, word.val) store
  change (readAfterStore input f word pid tid).getD 0 = input f word
  simp [readAfterStore, store, read]

structure Program (s : Shape) where
  ptx : String
  recognized : recognizes s ptx = true

def Program.check (s : Shape) (ptx : String) : Option (Program s) :=
  if h : recognizes s ptx = true then some ⟨ptx, h⟩ else none

abbrev Programs := (s : Shape) → Program s

def Programs.run (programs : Programs) : Copy := fun s input =>
  (interpret s (programs s).ptx input).getD (fun _ _ => 0)

theorem Programs.run_eq (programs : Programs) : programs.run = eager := by
  funext s input
  rw [Programs.run, interpret_exact (programs s).recognized]
  rfl

#axiomsClean slot_index_exact
#axiomsClean reads_bounded
#axiomsClean stores_read
#axiomsClean stores_word_iff
#axiomsClean interpret_exact
#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.GraphPlainPTX
