import Init.Data.SInt.Lemmas
import Init.Data.Int.Bitwise.Lemmas
import VeriTile.Meta.StatementAudit

/-! The CPU row IDs used by the training and validation data loaders. The
optimized remainder follows the signed int64 scratch and int32 output
calculations of the NumPy implementation. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.FineWeb

def rowHashHalf : Nat := 42301440
def rowHashModulus : Nat := rowHashHalf - 1

private def quotient (x : Int) : Int := (51 * x + 1073741824) / 2147483648

private theorem residue_bounds (x : Int) (lo : -2147483648 ≤ x) (hi : x < 2147483648) :
    -42301439 ≤ x - quotient x * 42301439 ∧ x - quotient x * 42301439 < 42301439 := by
  unfold quotient
  let n := 51 * x + 1073741824
  have hn₁ := Int.emod_nonneg n (by decide : (2147483648 : Int) ≠ 0)
  have hn₂ := Int.emod_lt_of_pos n (by decide : (0 : Int) < 2147483648)
  have hq := Int.mul_ediv_add_emod n 2147483648
  change -42301439 ≤ x - (n / 2147483648) * 42301439 ∧ x - (n / 2147483648) * 42301439 < 42301439
  dsimp [n] at *
  omega

private theorem bmod32 {x : Int} (lo : -(2 ^ 31) ≤ x) (hi : x < 2 ^ 31) :
    x.bmod (2 ^ 32) = x := by
  simpa only [Int32.toInt_ofInt] using Int32.toInt_ofInt_of_le lo hi

private theorem bmod64 {x : Int} (lo : -(2 ^ 63) ≤ x) (hi : x < 2 ^ 63) :
    x.bmod (2 ^ 64) = x := by
  simpa only [Int64.toInt_ofInt] using Int64.toInt_ofInt_of_le lo hi

private theorem shift31_int64 (x : Int64) :
    (x >>> (31 : Int64)).toInt = x.toInt / 2147483648 := by
  change (x.toBitVec.sshiftRight' ((31 : BitVec 64).smod 64)).toInt = _
  rw [BitVec.toInt_sshiftRight', Int.shiftRight_eq_div_pow]
  rfl

private theorem shift31_int32 (x : Int32) :
    (x >>> (31 : Int32)) = if x.toInt < 0 then -1 else 0 := by
  have lo := Int32.le_toInt x
  have hi := Int32.toInt_lt x
  apply Int32.toInt_inj.mp
  change (x.toBitVec.sshiftRight' ((31 : BitVec 32).smod 32)).toInt = _
  rw [BitVec.toInt_sshiftRight', Int.shiftRight_eq_div_pow]
  change x.toInt / 2147483648 = (if x.toInt < 0 then (-1 : Int32) else 0).toInt
  split
  · change x.toInt / 2147483648 = -1
    omega
  · change x.toInt / 2147483648 = 0
    omega

/-- Multiply and bias in the int64 scratch, narrow the residual, then add
the modulus exactly when the residual's arithmetic sign shift is -1. -/
def remainderFast (word : Int32) : Int32 :=
  let wide := word.toInt64
  let q := (wide * 51 + 1073741824) >>> (31 : Int64)
  let residual := (wide - q * 42301439).toInt32
  residual + ((residual >>> (31 : Int32)) &&& 42301439)

theorem remainder_fast_exact (word : Int32) :
    (remainderFast word).toInt = word.toInt % 42301439 := by
  have lo := Int32.le_toInt word
  have hi := Int32.toInt_lt word
  let q := (word.toInt64 * 51 + 1073741824) >>> (31 : Int64)
  have qInt : q.toInt = quotient word.toInt := by
    simp only [q, quotient, shift31_int64, Int64.toInt_add, Int64.toInt_mul, Int32.toInt_toInt64]
    change (((word.toInt * 51).bmod (2^64) + 1073741824).bmod (2^64)) / 2147483648 = _
    rw [bmod64 (by omega : -(2^63) ≤ word.toInt * 51) (by omega), bmod64 (by omega) (by omega)]
    rw [Int.mul_comm word.toInt 51]
  let r := word.toInt - quotient word.toInt * 42301439
  have ⟨rlo, rhi⟩ := residue_bounds word.toInt lo hi
  let residual := (word.toInt64 - q * 42301439).toInt32
  have residualInt : residual.toInt = r := by
    have qlo : -51 ≤ quotient word.toInt := by unfold quotient; omega
    have qhi : quotient word.toInt ≤ 51 := by unfold quotient; omega
    simp only [residual, Int64.toInt_toInt32, Int64.toInt_sub, Int32.toInt_toInt64, Int64.toInt_mul, qInt]
    change (((word.toInt - (quotient word.toInt * 42301439).bmod (2^64)).bmod (2^64)).bmod (2^32)) = r
    rw [bmod64 (by omega : -(2^63) ≤ quotient word.toInt * 42301439) (by omega),
        bmod64 (by omega) (by omega), bmod32 (by omega) (by omega)]
  have decompose : r % 42301439 = word.toInt % 42301439 := by simp [r]
  change (residual + ((residual >>> (31 : Int32)) &&& 42301439)).toInt = _
  rw [shift31_int32, residualInt]
  split
  · rename_i negative
    have mask : ((-1 : Int32) &&& 42301439).toInt = 42301439 := rfl
    have low : 0 ≤ r + 42301439 := by omega
    have high : r + 42301439 < 42301439 := by omega
    rw [Int32.toInt_add, residualInt, mask, bmod32 (by omega) (by omega)]
    calc
      r + 42301439 = (r + 42301439) % 42301439 := (Int.emod_eq_of_lt low high).symm
      _ = r % 42301439 := by simp
      _ = _ := decompose
  · rename_i nonnegative
    have mask : ((0 : Int32) &&& 42301439).toInt = 0 := rfl
    rw [Int32.toInt_add, residualInt, mask, Int.add_zero, bmod32 (by omega) (by omega)]
    exact (Int.emod_eq_of_lt (by omega : 0 ≤ r) rhi).symm.trans decompose

#axiomsClean remainder_fast_exact

def reduceRow (x : UInt32) : UInt32 :=
  UInt32.ofNat (x.toInt32.toInt % (rowHashModulus : Int)).toNat

def reduceRowFast (x : UInt32) : UInt32 := UInt32.ofNat (remainderFast x.toInt32).toInt.toNat

theorem reduce_row_fast_exact (x : UInt32) : reduceRowFast x = reduceRow x := by
  rw [reduceRowFast, reduceRow, remainder_fast_exact]
  rfl

abbrev RowHash := Array UInt32 → Option (Array UInt32)

/-- Both paths wrap the products and the XOR in int32. The int32 bit
patterns are represented as UInt32 until the signed remainder. -/
def hashRowsWith (reduce : UInt32 → UInt32) : RowHash := fun inputs => do
  if inputs.size < 2 then none else do
    let indices := Array.range inputs.size
    let read := fun i => inputs[i]?.getD 0
    let reserved := UInt32.ofNat rowHashModulus
    let bigram := indices.map fun i =>
      if i = 0 then reserved else
        let previous := if i = 1 then reserved else read (i - 1)
        reduce ((36313 * read i) ^^^ (27191 * previous))
    let half := UInt32.ofNat rowHashHalf
    let trigram := indices.map fun i =>
      if i < 2 then half + reserved else
        half + reduce ((17351 * read i) ^^^ (60961 * read (i - 1)) ^^^ (45259 * read (i - 2)))
    some (bigram ++ trigram)

def hashRows : RowHash := hashRowsWith reduceRow

def hashRowsCPU : RowHash := fun inputs =>
  if inputs.size ≥ 16384 then hashRowsWith reduceRowFast inputs else hashRows inputs

theorem hash_rows_cpu_equal : hashRowsCPU = hashRows := by
  have reduction : reduceRowFast = reduceRow := funext reduce_row_fast_exact
  funext input
  simp [hashRowsCPU, hashRows, reduction]

theorem hash_rows_size (inputs : Array UInt32) (rows : Array UInt32)
    (h : hashRows inputs = some rows) : rows.size = 2 * inputs.size := by
  unfold hashRows hashRowsWith at h
  split at h
  · contradiction
  · rw [← Option.some.inj h]
    simp
    omega

theorem reduce_row_bound (word : UInt32) : (reduceRow word).toNat < 42301440 := by
  change (word.toInt32.toInt % 42301439).toNat % (2 ^ 32) < 42301440
  have res := Int.emod_lt_of_pos word.toInt32.toInt (by decide : (0 : Int) < 42301439)
  omega

theorem shifted_row_bound (word : UInt32) :
    ((42301440 : UInt32) + reduceRow word).toNat < 84602880 := by
  have lo := reduce_row_bound word
  rw [UInt32.toNat_add]
  change (42301440 + (reduceRow word).toNat) % (2 ^ 32) < 84602880
  omega

theorem hash_rows_range (inputs rows : Array UInt32) (hash : hashRows inputs = some rows)
    (row : UInt32) (member : row ∈ rows) : row.toNat < 84602880 := by
  unfold hashRows hashRowsWith at hash
  split at hash
  · contradiction
  · rw [← Option.some.inj hash] at member
    rcases Array.mem_append.mp member with inFirst | inSecond
    · obtain ⟨i, _, rfl⟩ := Array.mem_map.mp inFirst
      split
      · decide
      · exact Nat.lt_trans (reduce_row_bound _) (by decide)
    · obtain ⟨i, _, rfl⟩ := Array.mem_map.mp inSecond
      split
      · decide
      · exact shifted_row_bound _

theorem hash_rows_cpu_range (inputs rows : Array UInt32) (hash : hashRowsCPU inputs = some rows)
    (row : UInt32) (member : row ∈ rows) : row.toNat < 84602880 := by
  exact hash_rows_range inputs rows (by simpa [hash_rows_cpu_equal] using hash) row member

theorem decode_hashed_row (row : UInt32) (bound : row.toNat < 84602880) :
    row.toInt32.toInt = Int.ofNat row.toNat := by
  change row.toBitVec.toInt = Int.ofNat row.toBitVec.toNat
  rw [BitVec.toInt_eq_toNat_bmod]
  change (↑row.toNat : Int).bmod (2 ^ 32) = (↑row.toNat : Int)
  apply bmod32 <;> omega

#axiomsClean hash_rows_cpu_range
#axiomsClean decode_hashed_row
#axiomsClean hash_rows_cpu_equal
#axiomsClean hash_rows_size

end VeriTile.Bench.Models.NanoGPT.Track1.FineWeb
