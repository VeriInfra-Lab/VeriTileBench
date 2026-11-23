import models.NanoGPT.Track1.ValueAdamRootInputRange
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Data.Rat.Cast.Order

namespace VeriTile.Triton.FP

theorem pow2_as_zpow (e : Int) : pow2 e = (2 : Rat) ^ e := by
  cases e with
  | ofNat exponent =>
    simp [pow2, Rat.ofInt_eq_cast]
  | negSucc exponent =>
    simp [pow2, Rat.mkRat_eq_divInt, Rat.divInt_eq_div, zpow_negSucc, one_div]

private theorem between_powers (n : Nat) (positive : 0 < n) :
    (2 : Rat) ^ n.log2 ≤ n ∧ (n : Rat) < (2 : Rat) ^ (n.log2 + 1) := by
  have nz := Nat.ne_of_gt positive
  exact ⟨by exact_mod_cast (Nat.le_log2 nz).mp (Nat.le_refl _),
    by exact_mod_cast (Nat.log2_lt nz).mp (Nat.lt_succ_self _)⟩

theorem positive_binade (q : Rat) (positive : 0 < q) :
    let estimate : Int := (q.num.natAbs.log2 : Int) - q.den.log2
    let e := if q < pow2 estimate then estimate - 1 else estimate
    pow2 e ≤ q ∧ q < pow2 (e + 1) := by
  let p := q.num.natAbs
  let d := q.den
  let s : Int := (p.log2 : Int) - d.log2
  have numPos := Rat.num_pos.mpr positive
  have pPos : 0 < p := by
    dsimp [p]
    omega
  have dPos : 0 < d := q.pos
  have nPos : (0 : Rat) < p := by exact_mod_cast pPos
  have denPos : (0 : Rat) < d := by exact_mod_cast dPos
  have pBounds := between_powers p pPos
  have dBounds := between_powers d dPos
  have qFraction : q = (p : Rat) / (d : Rat) := by
    conv_lhs => rw [← Rat.num_divInt_den q, Rat.divInt_eq_div]
    rw [← Int.natAbs_of_nonneg numPos.le]
    simp [p, d]
  have below : pow2 (s - 1) < q := by
    have shift : s - 1 = (p.log2 : Int) - ((d.log2 + 1 : Nat) : Int) := by dsimp [s]; omega
    rw [shift, pow2_as_zpow, zpow_sub₀ (by norm_num : (2 : Rat) ≠ 0), Rat.zpow_natCast,
      Rat.zpow_natCast, qFraction]
    calc
      (2 : Rat) ^ p.log2 / 2 ^ (d.log2 + 1) < (2 : Rat) ^ p.log2 / d :=
        div_lt_div_of_pos_left (by positivity) denPos dBounds.2
      _ ≤ (p : Rat) / d := div_le_div_of_nonneg_right pBounds.1 denPos.le
  have above : q < pow2 (s + 1) := by
    have shift : s + 1 = ((p.log2 + 1 : Nat) : Int) - (d.log2 : Int) := by dsimp [s]; omega
    rw [shift, pow2_as_zpow, zpow_sub₀ (by norm_num : (2 : Rat) ≠ 0), Rat.zpow_natCast,
      Rat.zpow_natCast, qFraction]
    calc
      (p : Rat) / d ≤ p / (2 : Rat) ^ d.log2 :=
        div_le_div_of_nonneg_left nPos.le (by positivity) dBounds.1
      _ < (2 : Rat) ^ (p.log2 + 1) / 2 ^ d.log2 :=
        div_lt_div_of_pos_right pBounds.2 (by positivity)
  dsimp only
  change pow2 (if q < pow2 s then s - 1 else s) ≤ q ∧
    q < pow2 ((if q < pow2 s then s - 1 else s) + 1)
  split
  · rename_i small
    have back : s - 1 + 1 = s := by omega
    rw [back]
    exact ⟨below.le, small⟩
  · rename_i large
    exact ⟨le_of_not_gt large, above⟩

#axiomsClean positive_binade

end VeriTile.Triton.FP
