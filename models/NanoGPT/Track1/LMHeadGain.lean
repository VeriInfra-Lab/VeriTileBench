import models.NanoGPT.Track1.LMHeadPack

/-! The pinned FP32 inverse scale is slightly below 224. The Hopper
specialization multiplies in BF16 with 224 and preserves the final E4M3
byte. The computation checks every possible BF16 input, including
subnormals, zeros, infinities and NaNs. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.LMHead.Gain
open VeriTile.Triton.FP

def pinned : Value .fp32 := Value.ofBits .fp32 0x435fffff
def packed : Value .bf16 := Value.ofBits .bf16 0x4360

def scale (x : Value .bf16) : Value .bf16 := mul Config.ieeeValue x packed

private def agrees (raw : Nat) : Bool :=
  let x := Value.ofBits .bf16 raw
  FP8E4M3.clampConvert (cast .fp32 Config.ieeeValue (scale x)) == byte x pinned

private def verifies (page : Nat) : Bool :=
  (List.range 64).all fun i => agrees (page * 64 + i)

private def verifiedGroup (group : Nat) : Prop :=
  ∀ row col : Fin 8, verifies (group * 64 + row.val * 8 + col.val) = true

private theorem use_groups
    (verified : ∀ g : Nat, g < 16 → verifiedGroup g) (x : Value .bf16) :
    satfinite (cast .fp32 Config.ieeeValue (upperMin (scale x))) = byte x pinned := by
  have bound : x.raw < 65536 := x.bits.isLt
  let page := x.raw / 64
  let row : Fin 8 := ⟨page % 64 / 8, by omega⟩
  let col : Fin 8 := ⟨page % 8, by omega⟩
  have confirmed := verified (page / 64) (by omega) row col
  have bitCheck := List.all_eq_true.mp confirmed (x.raw % 64) (List.mem_range.mpr (by omega))
  have join : (page / 64 * 64 + row.val * 8 + col.val) * 64 + x.raw % 64 = x.raw := by
    change (page / 64 * 64 + page % 64 / 8 * 8 + page % 8) * 64 + x.raw % 64 = x.raw
    omega
  rw [join] at bitCheck
  have reencode : Value.ofBits .bf16 x.raw = x := by
    cases x
    simp [Value.ofBits, Value.raw, BitVec.ofNat_toNat]
  simpa only [rounded_byte, agrees, reencode, beq_iff_eq] using bitCheck

-- Each closed leaf checks 64 encodings. The two Fin indices enumerate
-- the 64 slices of a group. Keep the groups separate for kernel reduction.
local macro "check_gain_group" : tactic => `(tactic|
  (unfold verifiedGroup
   repeat
     intro row
     refine Fin.cases ?_ ?_ row
     · repeat
         intro col
         refine Fin.cases ?_ ?_ col
         · decide +kernel
       intro col
       exact Fin.elim0 col
   intro row
   exact Fin.elim0 row))

section
set_option maxRecDepth 8192
set_option maxHeartbeats 0
private theorem group0 : verifiedGroup 0 := by check_gain_group
private theorem group1 : verifiedGroup 1 := by check_gain_group
private theorem group2 : verifiedGroup 2 := by check_gain_group
private theorem group3 : verifiedGroup 3 := by check_gain_group
private theorem group4 : verifiedGroup 4 := by check_gain_group
private theorem group5 : verifiedGroup 5 := by check_gain_group
private theorem group6 : verifiedGroup 6 := by check_gain_group
private theorem group7 : verifiedGroup 7 := by check_gain_group
private theorem group8 : verifiedGroup 8 := by check_gain_group
private theorem group9 : verifiedGroup 9 := by check_gain_group
private theorem group10 : verifiedGroup 10 := by check_gain_group
private theorem group11 : verifiedGroup 11 := by check_gain_group
private theorem group12 : verifiedGroup 12 := by check_gain_group
private theorem group13 : verifiedGroup 13 := by check_gain_group
private theorem group14 : verifiedGroup 14 := by check_gain_group
private theorem group15 : verifiedGroup 15 := by check_gain_group
end

private theorem all_groups (g : Nat) (bound : g < 16) : verifiedGroup g := by
  match g with
  | 0 => exact group0
  | 1 => exact group1
  | 2 => exact group2
  | 3 => exact group3
  | 4 => exact group4
  | 5 => exact group5
  | 6 => exact group6
  | 7 => exact group7
  | 8 => exact group8
  | 9 => exact group9
  | 10 => exact group10
  | 11 => exact group11
  | 12 => exact group12
  | 13 => exact group13
  | 14 => exact group14
  | 15 => exact group15
  | _ + 16 => omega

theorem byte_eq (x : Value .bf16) :
    satfinite (cast .fp32 Config.ieeeValue (upperMin (scale x))) = byte x pinned :=
  use_groups all_groups x

#axiomsClean byte_eq
end VeriTile.Bench.Models.NanoGPT.Track1.LMHead.Gain
