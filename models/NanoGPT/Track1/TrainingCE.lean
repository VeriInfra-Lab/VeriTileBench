import VeriTile.Triton.Float.ScalarOpsLaws
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Fin.Basic
import models.NanoGPT.Track1.TrainingCEConversions
import models.NanoGPT.Track1.CudaStores

/-! Training softcapped cross-entropy after the FP8 logit GEMM.
The originating CUDA CTA decodes logits into a halfword row, reduces the
fixed-max exponential sum, writes its loss, then writes the base FP8
gradient and patches the target columns. Each fixed-count CTA
stages bytes and evaluates the sigmoid and sum contribution once per code.
Format conversions use explicit word semantics. The native interface
supplies approximate elementary instructions. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCE
open VeriTile.Triton FP
open TrainingCEConversions

abbrev F32 := Value .fp32
abbrev Half := UInt16

structure Native where
  reciprocal : F32 → F32
  tanh : F32 → F32
  exp2 : F32 → F32
  log2 : F32 → F32

def ftz : Config := ⟨.nearestEven, true, true⟩
private def word : Nat → F32 := Value.ofBits .fp32
private def zero : F32 := word 0
private def one : F32 := word 0x3f800000
private def half : F32 := word 0x3f000000
private def add := FP.add (f := .fp32) ftz
private def sub := FP.sub (f := .fp32) ftz
private def mul := FP.mul (f := .fp32) ftz
private def fma := FP.fma (f := .fp32) ftz

structure Parameters where
  aParam : UInt64
  bParam : UInt64
  cParam : UInt64
  gradScaleParam : UInt64
  prefixWeight : F32

def Parameters.a (p : Parameters) : F32 := launchDoubleToFloat p.aParam
def Parameters.b (p : Parameters) : F32 := launchDoubleToFloat p.bParam
def Parameters.c (p : Parameters) : F32 := launchDoubleToFloat p.cParam
def Parameters.gradScale (p : Parameters) : F32 := launchDoubleToFloat p.gradScaleParam

/-- The full-width kernel keeps A in the launch to preserve its register
budget. The sampled kernels also specialize A. -/
inductive Softcap where
  | runtime | defaultBC | defaultABC
  deriving DecidableEq

def Softcap.aBits : UInt64 := 0x4037000000000000
def Softcap.bBits : UInt64 := 0x4014000000000000
def Softcap.cBits : UInt64 := 0x401e000000000000

def Softcap.parameters (mode : Softcap) (p : Parameters) : Parameters :=
  match mode with
  | .runtime => p
  | .defaultBC => {p with bParam := Softcap.bBits, cParam := Softcap.cBits}
  | .defaultABC =>
    {p with aParam := Softcap.aBits, bParam := Softcap.bBits, cParam := Softcap.cBits}

def Softcap.accepts (mode : Softcap) (p : Parameters) : Bool :=
  let bc := p.bParam == Softcap.bBits && p.cParam == Softcap.cBits
  match mode with
  | .runtime => true
  | .defaultBC => bc
  | .defaultABC => p.aParam == Softcap.aBits && bc

theorem Softcap.parameters_exact (mode : Softcap) (p : Parameters) (accepted : mode.accepts p = true) :
    mode.parameters p = p := by
  cases mode <;> cases p <;> simp_all [Softcap.accepts, Softcap.parameters]

theorem Softcap.a_word :
    launchDoubleToFloat Softcap.aBits = Value.ofBits .fp32 0x41b80000 := by decide

theorem Softcap.b_word :
    launchDoubleToFloat Softcap.bBits = Value.ofBits .fp32 0x40a00000 := by decide

theorem Softcap.c_word :
    launchDoubleToFloat Softcap.cBits = Value.ofBits .fp32 0x40f00000 := by decide

/-- The fixed CE launch embeds the rounded default head scale. The
reciprocal is still evaluated by the same native instruction. -/
inductive GradScale where
  | runtime | lmHead
  deriving DecidableEq

def GradScale.lmHeadBits : UInt64 := 0x3f2b6db6db6db6db

def GradScale.lmHeadWord : Nat := 0x395b6db7

def GradScale.parameters : GradScale → Parameters → Parameters
  | .runtime, p => p
  | .lmHead, p => {p with gradScaleParam := lmHeadBits}

def GradScale.accepts : GradScale → Parameters → Bool
  | .runtime, _ => true
  | .lmHead, p => p.gradScaleParam == lmHeadBits

theorem GradScale.parameters_exact (scale : GradScale) (p : Parameters) (accepted : scale.accepts p = true) :
    scale.parameters p = p := by
  cases scale <;> cases p <;> simp_all [GradScale.accepts, GradScale.parameters]

theorem GradScale.lm_head_word :
    launchDoubleToFloat lmHeadBits = Value.ofBits .fp32 lmHeadWord := by decide +kernel

def gradientFactor (op : Native) (p : Parameters) : F32 :=
  mul (mul (op.reciprocal p.c) p.a) (op.reciprocal p.gradScale)

def decode (b : UInt8) : F32 := halfToFloat (e4m3ToHalf b)

/-- The packed PTX conversion applies the same scalar conversion to the
low and high byte, preserving their order in the halfword word. -/
def unpackLogits (bytes : UInt16) : F32 × F32 :=
  (decode (UInt8.ofNat (bytes.toNat % 256)), decode (UInt8.ofNat (bytes.toNat / 256)))

def packBytes (lo hi : UInt8) : UInt16 := UInt16.ofNat (lo.toNat + 256 * hi.toNat)

theorem unpack_exact (lo hi : UInt8) :
    unpackLogits (packBytes lo hi) = (decode lo, decode hi) := by
  have l : lo.toNat < 256 := UInt8.toNat_lt_size lo
  have h : hi.toNat < 256 := UInt8.toNat_lt_size hi
  have pack : (packBytes lo hi).toNat = lo.toNat + 256 * hi.toNat := by
    change (lo.toNat + 256 * hi.toNat) % 65536 = _
    exact Nat.mod_eq_of_lt (by omega)
  have low : (lo.toNat + 256 * hi.toNat) % 256 = lo.toNat := by omega
  have high : (lo.toNat + 256 * hi.toNat) / 256 = hi.toNat := by omega
  simp only [unpackLogits, pack, low, high, UInt8.ofNat_toNat]

def sigmoid (op : Native) (p : Parameters) (x : F32) : F32 :=
  let inv := op.reciprocal p.c
  fma (op.tanh (mul (fma inv x (mul inv p.b)) half)) half half

def referencePair (op : Native) (p : Parameters) (lo hi : UInt8) : Half × Half :=
  (floatToHalf (sigmoid op p (decode lo)), floatToHalf (sigmoid op p (decode hi)))

def packHalves (lo hi : Half) : UInt32 := UInt32.ofNat (lo.toNat + 65536 * hi.toNat)

def unpackHalves (word : UInt32) : Half × Half :=
  (UInt16.ofNat (word.toNat % 65536), UInt16.ofNat (word.toNat / 65536))

theorem unpack_halves_exact (lo hi : Half) : unpackHalves (packHalves lo hi) = (lo, hi) := by
  have l : lo.toNat < 65536 := UInt16.toNat_lt_size lo
  have h : hi.toNat < 65536 := UInt16.toNat_lt_size hi
  have pack : (packHalves lo hi).toNat = lo.toNat + 65536 * hi.toNat := by
    change (lo.toNat + 65536 * hi.toNat) % 4294967296 = _
    exact Nat.mod_eq_of_lt (by omega)
  have low : (lo.toNat + 65536 * hi.toNat) % 65536 = lo.toNat := by omega
  have high : (lo.toNat + 65536 * hi.toNat) / 65536 = hi.toNat := by omega
  simp only [unpackHalves, pack, low, high, UInt16.ofNat_toNat]

def packedPair (op : Native) (p : Parameters) (lo hi : UInt8) : Half × Half :=
  let (x, y) := unpackLogits (packBytes lo hi)
  unpackHalves (packHalves (floatToHalf (sigmoid op p x)) (floatToHalf (sigmoid op p y)))

theorem pairs_equal : packedPair = referencePair := by
  funext op p lo hi
  simp only [packedPair, referencePair, unpack_exact, unpack_halves_exact]

inductive Width where
  | sampled10 | sampled14 | sampled24 | full
  deriving DecidableEq

def Width.columns : Width → Nat
  | .sampled10 => 10240
  | .sampled14 => 14336
  | .sampled24 => 24576
  | .full => 50304

abbrev Column (w : Width) := Fin w.columns

def Width.chunks (w : Width) : Nat := (w.columns + 2047) / 2048

abbrev Lane := Fin 256
abbrev Byte := Fin 8

def column (chunk : Nat) (lane : Lane) (byte : Byte) : Nat := chunk * 2048 + lane.val * 8 + byte.val

theorem column_unique (l r : Nat) (a b : Lane) (x y : Byte)
    (same : column l a x = column r b y) : l = r ∧ a = b ∧ x = y := by
  have := a.isLt
  have := b.isLt
  have := x.isLt
  have := y.isLt
  unfold column at same
  exact ⟨by omega, Fin.ext (by omega), Fin.ext (by omega)⟩

theorem column_covered (w : Width) (c : Column w) :
    ∃ (chunk : Fin w.chunks) (lane : Lane) (byte : Byte), column chunk lane byte = c.val := by
  have bound := c.isLt
  refine ⟨⟨c.val / 2048, by unfold Width.chunks; omega⟩,
    ⟨c.val % 2048 / 8, by omega⟩, ⟨c.val % 8, by omega⟩, ?_⟩
  simp only [column]
  omega

def columnOwner {w : Width} (c : Column w) : Lane :=
  ⟨c.val / 8 % 256, Nat.mod_lt _ (by decide)⟩

theorem column_owner_exact {w : Width} (c : Column w) (chunk : Nat) (lane : Lane) (byte : Byte)
    (address : column chunk lane byte = c.val) : columnOwner c = lane := by
  have := lane.isLt
  have := byte.isLt
  unfold column at address
  apply Fin.ext
  change c.val / 8 % 256 = lane.val
  omega

structure Row (w : Width) where
  batchRows : Nat
  row : Fin batchRows
  predictions : Fin 4
  targets : Fin batchRows → Column w
  prefixTarget : Option (Column w)
  weights : Nat → F32
  logits : Column w → UInt8

abbrev Convert := Native → Parameters → UInt8 → UInt8 → Half × Half

def sigmoidRow {w : Width} (convert : Convert) (op : Native) (p : Parameters)
    (read : Column w → UInt8) : Column w → Half := fun c =>
  let lo := (c.val / 2) * 2
  have left : lo < w.columns := by have := c.isLt; dsimp [lo]; omega
  have right : lo + 1 < w.columns := by
    have bound := c.isLt
    cases w <;> simp only [Width.columns] at bound ⊢ <;> dsimp [lo] <;> omega
  let both := convert op p (read ⟨lo, left⟩) (read ⟨lo + 1, right⟩)
  if c.val % 2 = 0 then both.1 else both.2

theorem reference_sigmoid {w : Width} (op : Native) (p : Parameters)
    (read : Column w → UInt8) (c : Column w) :
    sigmoidRow referencePair op p read c = floatToHalf (sigmoid op p (decode (read c))) := by
  by_cases even : c.val % 2 = 0
  · have val : c.val / 2 * 2 = c.val := by omega
    have index : (⟨c.val / 2 * 2, by rw [val]; exact c.isLt⟩ : Column w) = c := Fin.ext val
    simp only [sigmoidRow, if_pos even, referencePair, index]
  · have val : c.val / 2 * 2 + 1 = c.val := by omega
    have index : (⟨c.val / 2 * 2 + 1, by rw [val]; exact c.isLt⟩ : Column w) = c := Fin.ext val
    simp only [sigmoidRow, if_neg even, referencePair, index]

private def expf (op : Native) (x : F32) : F32 := op.exp2 (mul x (word 0x3fb8aa3b))

structure ScaledLogit where
  sigma : F32
  scaled : F32

def ScaledLogit.ofHalf (p : Parameters) (sigma : Half) : ScaledLogit :=
  let widened := halfToFloat sigma
  ⟨widened, mul widened p.a⟩

structure ProbabilityEntry where
  sigma : F32
  probability : F32

private def rowProbability (op : Native) (lse scaled : F32) : F32 :=
  expf op (sub scaled lse)

def ScaledLogit.forRow (op : Native) (lse : F32) (logit : ScaledLogit) : ProbabilityEntry :=
  ⟨logit.sigma, rowProbability op lse logit.scaled⟩

structure LogitEntry where
  logit : ScaledLogit
  contribution : F32

def LogitEntry.ofHalf (op : Native) (p : Parameters) (sigma : Half) : LogitEntry :=
  let logit := ScaledLogit.ofHalf p sigma
  ⟨logit, expf op (sub logit.scaled p.a)⟩

/-- One lane writes each of the 256 entries, before any table reads. -/
def logitTable (op : Native) (p : Parameters) : Vector LogitEntry 256 :=
  Vector.ofFn fun lane =>
    let sigma := floatToHalf (sigmoid op p (decode (UInt8.ofNat lane.val)))
    LogitEntry.ofHalf op p sigma

def readLogitTable (table : Vector LogitEntry 256) (code : UInt8) : LogitEntry :=
  table[code.toNat]'(UInt8.toNat_lt_size code)

theorem table_lookup_exact (op : Native) (p : Parameters) (code : UInt8) :
    readLogitTable (logitTable op p) code =
      LogitEntry.ofHalf op p (floatToHalf (sigmoid op p (decode code))) := by
  simp [readLogitTable, logitTable]

def probabilityTable (op : Native) (lse : F32) (table : Vector LogitEntry 256) :
    Vector ProbabilityEntry 256 :=
  Vector.ofFn fun lane => table[lane.val].logit.forRow op lse

def readProbabilityTable (table : Vector ProbabilityEntry 256) (code : UInt8) : ProbabilityEntry :=
  table[code.toNat]'(UInt8.toNat_lt_size code)

theorem probability_lookup_exact (op : Native) (lse : F32) (table : Vector LogitEntry 256) (code : UInt8) :
    readProbabilityTable (probabilityTable op lse table) code =
      (readLogitTable table code).logit.forRow op lse := by
  simp [readProbabilityTable, probabilityTable, readLogitTable]

/-! Shared memory is addressed in bytes. The two static CUDA arrays, the
scalar gradient factor and the dynamic logit row occupy separate regions. -/
private def writeBytes {n : Nat} (address : Fin n → Nat) (byte : Fin n → UInt8) : List CudaStores.Store :=
  (List.finRange n).map fun i => ⟨address i, byte i⟩

private theorem write_bytes_unique {n : Nat} (address : Fin n → Nat) (byte : Fin n → UInt8)
    (uniqueAddress : Function.Injective address) :
    ((writeBytes address byte).map CudaStores.Store.address).Nodup := by
  simp only [writeBytes, List.map_map]
  apply List.pairwise_map.mpr
  apply (List.nodup_finRange _).imp
  exact fun h same => h (uniqueAddress same)

private theorem write_bytes_exact {n : Nat} (address : Fin n → Nat) (byte : Fin n → UInt8)
    (uniqueAddress : Function.Injective address)
    (order : List CudaStores.Store) (schedule : order.Perm (writeBytes address byte))
    (prior : CudaStores.Memory) :
    (∀ i : Fin n, CudaStores.commitAll order prior (address i) = byte i) ∧
    (∀ a, (∀ i, address i ≠ a) → CudaStores.commitAll order prior a = prior a) := by
  have unique := write_bytes_unique address byte uniqueAddress
  rw [← CudaStores.unique_stores_commute _ _ _ unique schedule.symm]
  constructor
  · intro i
    exact CudaStores.unique_store_result _ _ ⟨address i, byte i⟩ unique
      (List.mem_map.mpr ⟨i, List.mem_finRange i, rfl⟩)
  · intro a outside
    apply CudaStores.preserve_unwritten
    intro store listed
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp listed
    exact outside i

private def writeRegion {n : Nat} (base : Nat) (byte : Fin n → UInt8) : List CudaStores.Store :=
  writeBytes (fun i => base + i.val) byte

private theorem region_exact {n : Nat} (base : Nat) (byte : Fin n → UInt8)
    (order : List CudaStores.Store) (schedule : order.Perm (writeRegion base byte))
    (prior : CudaStores.Memory) :
    (∀ i : Fin n, CudaStores.commitAll order prior (base + i.val) = byte i) ∧
    (∀ a, a < base ∨ base + n ≤ a → CudaStores.commitAll order prior a = prior a) := by
  have injective : Function.Injective (fun (i : Fin n) => base + i.val) :=
    fun _ _ same => Fin.ext (Nat.add_left_cancel same)
  obtain ⟨values, frame⟩ := write_bytes_exact _ byte injective order schedule prior
  refine ⟨values, fun a outside => frame a ?_⟩
  intro i
  have := i.isLt
  omega

private def lowByte (value offset : Nat) : UInt8 := UInt8.ofNat (value / 256 ^ offset)

inductive TableField where
  | sigma | scaled | contribution

private def TableField.start (field : TableField) (code : UInt8) : Nat :=
  match field with
  | .sigma => 8 * code.toNat
  | .scaled => 8 * code.toNat + 4
  | .contribution => 2048 + 4 * code.toNat

private def TableField.value (field : TableField) (entry : LogitEntry) : F32 :=
  match field with
  | .sigma => entry.logit.sigma
  | .scaled => entry.logit.scaled
  | .contribution => entry.contribution

private def tableByte (table : Vector LogitEntry 256) (i : Fin 3072) : UInt8 :=
  if packed : i.val < 2048 then
    let entry := table.get ⟨i.val / 8, by omega⟩
    let byte := i.val % 8
    if byte < 4 then lowByte entry.logit.sigma.raw byte
    else lowByte entry.logit.scaled.raw (byte - 4)
  else
    let j := i.val - 2048
    have exp : j / 4 < 256 := by have := i.isLt; dsimp [j]; omega
    lowByte (table.get ⟨j / 4, exp⟩).contribution.raw (j % 4)

private abbrev gradientFactorAddress : Nat := 3072
abbrev codeAddress : Nat := 3076

private def initializationByte (table : Vector LogitEntry 256) (factor : F32) (i : Fin codeAddress) : UInt8 :=
  if inTable : i.val < gradientFactorAddress then tableByte table ⟨i.val, inTable⟩
  else lowByte factor.raw (i.val - gradientFactorAddress)

def initializationStores (table : Vector LogitEntry 256) (factor : F32) : List CudaStores.Store :=
  writeRegion 0 (initializationByte table factor)

/-- The two lookup tables and the scalar published by lane zero. -/
def tableStores (op : Native) (p : Parameters) : List CudaStores.Store :=
  initializationStores (logitTable op p) (gradientFactor op p)

def codeStores {w : Width} (r : Row w) : List CudaStores.Store := writeRegion codeAddress r.logits

/-- At the staging barrier each raw logit has its own column address, and
the staged row preserves every byte in both initialized tables. -/
theorem staged_code_exact {w : Width} (r : Row w) (order : List CudaStores.Store)
    (schedule : order.Perm (codeStores r)) (prior : CudaStores.Memory) :
    (∀ c : Column w, CudaStores.commitAll order prior (codeAddress + c.val) = r.logits c) ∧
    (∀ a, a < codeAddress → CudaStores.commitAll order prior a = prior a) := by
  obtain ⟨written, untouched⟩ := region_exact codeAddress r.logits order schedule prior
  exact ⟨written, fun a before => untouched a (Or.inl before)⟩

private theorem table_byte_offset (field : TableField) (table : Vector LogitEntry 256)
    (code : UInt8) (byte : Fin 4) (within : field.start code + byte.val < 3072) :
    tableByte table ⟨field.start code + byte.val, within⟩ =
      lowByte (field.value (readLogitTable table code)).raw byte.val := by
  have width : code.toNat < 256 := UInt8.toNat_lt_size code
  have hb := byte.isLt
  have indexPacked (offset : Nat) (h : offset < 8) :
      (8 * code.toNat + offset) / 8 = code.toNat := by omega
  have offsetPacked (offset : Nat) (h : offset < 8) :
      (8 * code.toNat + offset) % 8 = offset := by omega
  cases field with
  | sigma =>
    have packed : 8 * code.toNat + byte.val < 2048 := by omega
    simp only [TableField.start, tableByte, dif_pos packed, indexPacked byte.val (by omega),
      offsetPacked byte.val (by omega), if_pos hb, TableField.value, readLogitTable]
    rfl
  | scaled =>
    have packed : 8 * code.toNat + (4 + byte.val) < 2048 := by omega
    have outside : ¬ 4 + byte.val < 4 := by omega
    simp only [TableField.start, tableByte, dif_pos packed, Nat.add_assoc,
      indexPacked (4 + byte.val) (by omega), offsetPacked (4 + byte.val) (by omega), if_neg outside,
      Nat.add_sub_cancel_left, TableField.value, readLogitTable]
    rfl
  | contribution =>
    have separate : ¬ 2048 + 4 * code.toNat + byte.val < 2048 := by omega
    have index : (2048 + 4 * code.toNat + byte.val - 2048) / 4 = code.toNat := by omega
    have offset : (2048 + 4 * code.toNat + byte.val - 2048) % 4 = byte.val := by omega
    simp only [TableField.start, tableByte, dif_neg separate, index, offset,
      TableField.value, readLogitTable]
    rfl

/-- Every table byte is published before the CTA first reads the tables.
The two FP32 gradient operands occupy one consecutive eight-byte entry. -/
theorem published_table_bytes (op : Native) (p : Parameters) (order : List CudaStores.Store)
    (schedule : order.Perm (tableStores op p)) (prior : CudaStores.Memory) (code : UInt8)
    (field : TableField) (byte : Fin 4) :
    CudaStores.commitAll order prior (field.start code + byte.val) =
      lowByte (field.value (readLogitTable (logitTable op p) code)).raw byte.val := by
  have width : code.toNat < 256 := UInt8.toNat_lt_size code
  have hb := byte.isLt
  have within : field.start code + byte.val < 3072 := by
    cases field <;> simp only [TableField.start] <;> omega
  obtain ⟨written, _⟩ := region_exact 0 (initializationByte (logitTable op p) (gradientFactor op p))
    order schedule prior
  have atByte := written ⟨field.start code + byte.val, Nat.lt_trans within (by decide)⟩
  simp only [Nat.zero_add] at atByte
  rw [atByte]
  simp only [initializationByte, dif_pos within, table_byte_offset]

private def readSharedWord (memory : CudaStores.Memory) (base : Nat) : F32 :=
  word ((memory base).toNat + 256 * (memory (base + 1)).toNat +
    65536 * (memory (base + 2)).toNat + 16777216 * (memory (base + 3)).toNat)

def readGradientFactor (memory : CudaStores.Memory) : F32 :=
  readSharedWord memory gradientFactorAddress

/-- The base gradient reads two packed words and selects one byte with a
permutation against the zero word. -/
def readPackedCode (memory : CudaStores.Memory) (address : Nat) : UInt8 :=
  lowByte (readSharedWord memory (address / 4 * 4)).raw (address % 4)

theorem packed_shared_code_exact (memory : CudaStores.Memory) (address : Nat) :
    readPackedCode memory address = memory address := by
  let base := address / 4 * 4
  let byte := address % 4
  have extent : byte < 4 := Nat.mod_lt _ (by decide)
  have index : base + byte = address := by dsimp [base, byte]; omega
  have h₀ : (memory base).toNat < 256 := UInt8.toNat_lt_size _
  have h₁ : (memory (base + 1)).toNat < 256 := UInt8.toNat_lt_size _
  have h₂ : (memory (base + 2)).toNat < 256 := UInt8.toNat_lt_size _
  have h₃ : (memory (base + 3)).toNat < 256 := UInt8.toNat_lt_size _
  have bits : (readSharedWord memory base).raw =
      (memory base).toNat + 256 * (memory (base + 1)).toNat +
        65536 * (memory (base + 2)).toNat + 16777216 * (memory (base + 3)).toNat := by
    unfold readSharedWord
    apply Nat.mod_eq_of_lt
    change _ < 4294967296
    omega
  calc
    readPackedCode memory address = memory (base + byte) := by
      change lowByte (readSharedWord memory base).raw byte = _
      rw [bits]
      apply UInt8.toNat_inj.mp
      have cases : byte = 0 ∨ byte = 1 ∨ byte = 2 ∨ byte = 3 := by omega
      rcases cases with h | h | h | h <;>
        simp only [h, lowByte, UInt8.toNat_ofNat', Nat.reducePow, Nat.div_one, Nat.add_zero] <;> omega
    _ = memory address := congrArg memory index

def readSharedTable (memory : CudaStores.Memory) (code : UInt8) : LogitEntry :=
  ⟨⟨readSharedWord memory (TableField.sigma.start code),
    readSharedWord memory (TableField.scaled.start code)⟩,
    readSharedWord memory (TableField.contribution.start code)⟩

private theorem float_bytes_roundtrip (x : F32) :
    word ((lowByte x.raw 0).toNat + 256 * (lowByte x.raw 1).toNat +
      65536 * (lowByte x.raw 2).toNat + 16777216 * (lowByte x.raw 3).toNat) = x := by
  have bound : x.raw < 4294967296 := x.bits.isLt
  have pack : (lowByte x.raw 0).toNat + 256 * (lowByte x.raw 1).toNat +
      65536 * (lowByte x.raw 2).toNat + 16777216 * (lowByte x.raw 3).toNat = x.raw := by
    simp only [lowByte, UInt8.toNat_ofNat', Nat.pow_zero, Nat.pow_one, Nat.div_one]
    omega
  rw [pack]
  cases x
  simp [word, Value.raw, Value.ofBits]

private theorem read_shared_word_from_bytes (memory : CudaStores.Memory) (base : Nat) (x : F32)
    (bytes : ∀ byte : Fin 4, memory (base + byte.val) = lowByte x.raw byte.val) :
    readSharedWord memory base = x := by
  have first := bytes ⟨0, by decide⟩
  change memory base = _ at first
  simp only [readSharedWord, first, bytes ⟨1, by decide⟩, bytes ⟨2, by decide⟩, bytes ⟨3, by decide⟩,
    float_bytes_roundtrip]

private theorem read_shared_word_congr (source target : CudaStores.Memory) (base : Nat)
    (bytes : ∀ byte : Fin 4, source (base + byte.val) = target (base + byte.val)) :
    readSharedWord source base = readSharedWord target base := by
  have first := bytes ⟨0, by decide⟩
  change source base = target base at first
  simp only [readSharedWord, first, bytes ⟨1, by decide⟩, bytes ⟨2, by decide⟩, bytes ⟨3, by decide⟩]

theorem published_table_words (op : Native) (p : Parameters) (order : List CudaStores.Store)
    (schedule : order.Perm (tableStores op p)) (prior : CudaStores.Memory) (code : UInt8) :
    readSharedTable (CudaStores.commitAll order prior) code = readLogitTable (logitTable op p) code := by
  have read (field : TableField) :
      readSharedWord (CudaStores.commitAll order prior) (field.start code) =
        field.value (readLogitTable (logitTable op p) code) :=
    read_shared_word_from_bytes _ _ _ (published_table_bytes op p order schedule prior code field)
  rw [readSharedTable, read, read, read]
  generalize readLogitTable (logitTable op p) code = entry
  rcases entry with ⟨⟨sigma, scaled⟩, exp⟩
  rfl

theorem published_gradient_factor (op : Native) (p : Parameters) (order : List CudaStores.Store)
    (schedule : order.Perm (tableStores op p)) (prior : CudaStores.Memory) :
    readSharedWord (CudaStores.commitAll order prior) gradientFactorAddress = gradientFactor op p := by
  obtain ⟨written, _⟩ := region_exact 0 (initializationByte (logitTable op p) (gradientFactor op p))
    order schedule prior
  have bytes (byte : Fin 4) :
      CudaStores.commitAll order prior (gradientFactorAddress + byte.val) =
        lowByte (gradientFactor op p).raw byte.val := by
    have atByte := written ⟨gradientFactorAddress + byte.val, Nat.add_lt_add_left byte.isLt _⟩
    simp only [Nat.zero_add] at atByte
    rw [atByte]
    simp only [initializationByte, dif_neg (Nat.not_lt_of_ge (Nat.le_add_right ..)),
      Nat.add_sub_cancel_left]
  exact read_shared_word_from_bytes _ _ _ bytes

theorem staged_gradient_factor {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (tableOrder : List CudaStores.Store) (initialized : tableOrder.Perm (tableStores op p))
    (codeOrder : List CudaStores.Store) (staged : codeOrder.Perm (codeStores r))
    (prior : CudaStores.Memory) :
    readSharedWord (CudaStores.commitAll codeOrder (CudaStores.commitAll tableOrder prior))
        gradientFactorAddress = gradientFactor op p := by
  have frame := (staged_code_exact r codeOrder staged (CudaStores.commitAll tableOrder prior)).2
  rw [← published_gradient_factor op p tableOrder initialized prior]
  simp (disch := decide) only [readSharedWord, frame]

theorem staged_table_read {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (tableOrder : List CudaStores.Store) (initialized : tableOrder.Perm (tableStores op p))
    (codeOrder : List CudaStores.Store) (staged : codeOrder.Perm (codeStores r))
    (prior : CudaStores.Memory) (c : Column w) :
    let memory := CudaStores.commitAll codeOrder (CudaStores.commitAll tableOrder prior)
    readSharedTable memory (memory (codeAddress + c.val)) =
      readLogitTable (logitTable op p) (r.logits c) := by
  have ⟨codes, frame⟩ := staged_code_exact r codeOrder staged (CudaStores.commitAll tableOrder prior)
  change ∀ a, a < 3076 → _ at frame
  have width : (r.logits c).toNat < 256 := UInt8.toNat_lt_size _
  dsimp only
  rw [codes c, ← published_table_words op p tableOrder initialized prior (r.logits c)]
  simp (disch := omega) only [readSharedTable, readSharedWord, TableField.start, frame]

/-- The second FP32 word in each packed entry is replaced in place. -/
def probabilityAddress (i : Fin 1024) : Nat := 8 * (i.val / 4) + 4 + i.val % 4

private theorem probability_address_unique : Function.Injective probabilityAddress := by
  intro i j equal
  apply Fin.ext
  unfold probabilityAddress at equal
  omega

private def probabilityByte (op : Native) (lse : F32) (prior : CudaStores.Memory)
    (i : Fin 1024) : UInt8 :=
  let scaled := readSharedWord prior (8 * (i.val / 4) + 4)
  lowByte (rowProbability op lse scaled).raw (i.val % 4)

def probabilityStores (op : Native) (lse : F32) (prior : CudaStores.Memory) : List CudaStores.Store :=
  writeBytes probabilityAddress (probabilityByte op lse prior)

/-- Before its own store, a lane still sees its original scaled word even
if any number of other lanes have published their probabilities. -/
theorem probability_other_lanes_preserve_scaled (issued : List CudaStores.Store)
    (prior : CudaStores.Memory) (code : UInt8)
    (others : ∀ store ∈ issued, ∃ i : Fin 1024,
      i.val / 4 ≠ code.toNat ∧ store.address = probabilityAddress i) :
    readSharedWord (CudaStores.commitAll issued prior) (TableField.scaled.start code) =
      readSharedWord prior (TableField.scaled.start code) := by
  apply read_shared_word_congr
  intro byte
  apply CudaStores.preserve_unwritten
  intro store listed same
  obtain ⟨i, foreign, addr⟩ := others store listed
  have equal : probabilityAddress i = TableField.scaled.start code + byte.val := addr.symm.trans same
  have bound := byte.isLt
  simp only [probabilityAddress, TableField.start] at equal
  omega

theorem probability_store_frame (op : Native) (lse : F32) (prior : CudaStores.Memory)
    (order : List CudaStores.Store) (schedule : order.Perm (probabilityStores op lse prior))
    (address : Nat) (outside : 2048 ≤ address ∨ address % 8 < 4) :
    CudaStores.commitAll order prior address = prior address := by
  have frame := (write_bytes_exact _ (probabilityByte op lse prior)
    probability_address_unique order schedule prior).2
  apply frame
  intro i
  have bound := i.isLt
  unfold probabilityAddress
  omega

private theorem published_probability (op : Native) (lse : F32) (prior : CudaStores.Memory)
    (order : List CudaStores.Store) (schedule : order.Perm (probabilityStores op lse prior))
    (code : UInt8) :
    readSharedWord (CudaStores.commitAll order prior) (TableField.scaled.start code) =
      rowProbability op lse (readSharedWord prior (TableField.scaled.start code)) := by
  have written := (write_bytes_exact _ (probabilityByte op lse prior)
    probability_address_unique order schedule prior).1
  apply read_shared_word_from_bytes
  intro byte
  have width : code.toNat < 256 := UInt8.toNat_lt_size code
  have bound := byte.isLt
  have div : (4 * code.toNat + byte.val) / 4 = code.toNat := by omega
  have rem : (4 * code.toNat + byte.val) % 4 = byte.val := by omega
  have atByte := written ⟨4 * code.toNat + byte.val, by omega⟩
  simpa only [probabilityAddress, probabilityByte, div, rem, TableField.start] using atByte

def readSharedProbability (memory : CudaStores.Memory) (code : UInt8) : ProbabilityEntry :=
  ⟨readSharedWord memory (TableField.sigma.start code),
    readSharedWord memory (TableField.scaled.start code)⟩

theorem probability_read_exact (op : Native) (lse : F32) (prior : CudaStores.Memory)
    (order : List CudaStores.Store) (schedule : order.Perm (probabilityStores op lse prior))
    (code : UInt8) :
    readSharedProbability (CudaStores.commitAll order prior) code =
      (readSharedTable prior code).logit.forRow op lse := by
  have sigma : readSharedWord (CudaStores.commitAll order prior) (TableField.sigma.start code) =
      readSharedWord prior (TableField.sigma.start code) := by
    apply read_shared_word_congr
    intro byte
    apply probability_store_frame op lse prior order schedule
    right
    have bound := byte.isLt
    simp only [TableField.start]
    omega
  rw [readSharedProbability, sigma, published_probability op lse prior order schedule]
  rfl

theorem probability_stores_commute (op : Native) (lse : F32) (prior : CudaStores.Memory)
    (order : List CudaStores.Store) (schedule : order.Perm (probabilityStores op lse prior)) :
    CudaStores.commitAll order prior = CudaStores.commitAll (probabilityStores op lse prior) prior := by
  have exactBytes := write_bytes_exact _ (probabilityByte op lse prior)
    probability_address_unique order schedule prior
  have canonical := write_bytes_exact _ (probabilityByte op lse prior)
    probability_address_unique _ (List.Perm.refl (probabilityStores op lse prior)) prior
  funext address
  by_cases written : ∃ i, probabilityAddress i = address
  · obtain ⟨i, rfl⟩ := written
    rw [exactBytes.1, canonical.1]
  · have outside (i : Fin 1024) : probabilityAddress i ≠ address := fun h => written ⟨i, h⟩
    rw [exactBytes.2 _ outside, canonical.2 _ outside]

structure Scan where
  chunks : List Nat
  rolled : Bool

def linearScan (w : Width) : Scan := ⟨List.range w.chunks, false⟩

/-- Two consecutive chunks per loop iteration, with the odd tail last. -/
def twoChunks (n : Nat) : List Nat :=
  (List.range (n / 2)).flatMap (fun k => [2 * k, 2 * k + 1]) ++
    (if n % 2 = 0 then [] else [n - 1])

def tableScan (w : Width) : Scan := ⟨twoChunks w.chunks, true⟩

/-- The reference base-gradient loop expands four consecutive chunks at a
time. The shorter sampled loops retain one iteration per chunk. -/
def gradientChunks (w : Width) (rolled : Bool) : List Nat :=
  if rolled then List.range w.chunks else
    (List.range (w.chunks / 4)).flatMap (fun k => [4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3]) ++
      (List.range (w.chunks % 4)).map (fun k => (w.chunks / 4) * 4 + k)

theorem gradient_store_order (w : Width) : gradientChunks w true = gradientChunks w false := by
  cases w <;> decide

private def accContribution (rolled : Bool) (chunk : Nat) (byte : Byte) (acc contribution : F32) : F32 :=
  if !rolled && chunk == 0 && byte.val == 0 then add contribution acc else add acc contribution

private theorem seed_order (rolled : Bool) : accContribution rolled = accContribution false := by
  cases rolled
  · rfl
  · funext chunk byte acc contribution
    unfold accContribution
    simp only [Bool.not_true, Bool.false_and, Bool.not_false, Bool.true_and, Bool.false_eq_true, ↓reduceIte]
    split <;> simp [add, FP.ScalarOpsLaws.add_comm]

def laneSumBy {w : Width} (scan : Scan) (exp : Column w → F32) (tid : Lane) : F32 :=
  scan.chunks.foldl (fun acc chunk =>
    (List.finRange 8).foldl (fun acc byte =>
      let c := column chunk tid byte
      if live : c < w.columns then
        accContribution scan.rolled chunk byte acc (exp ⟨c, live⟩)
      else acc) acc) zero

def laneSum {w : Width} (scan : Scan) (op : Native) (p : Parameters)
    (sigma : Column w → Half) (tid : Lane) : F32 :=
  laneSumBy scan (fun c => (LogitEntry.ofHalf op p (sigma c)).contribution) tid

/-- A full-mask SHFL.DOWN with a clamp of 31 keeps the source lane when
the requested lane would cross the warp boundary. -/
def shuffleAdd (values : Lane → F32) (offset : Nat) (tid : Lane) : F32 :=
  let up := tid.val + offset
  if sameWarp : tid.val % 32 + offset < 32 ∧ up < 256 then
    add (values tid) (values ⟨up, sameWarp.2⟩)
  else add (values tid) (values tid)

/-- Each warp publishes the first lane after the five descending shuffles. -/
def warpLeaders (read : Lane → F32) : Fin 8 → F32 :=
  let reduced := [16, 8, 4, 2, 1].foldl shuffleAdd read
  fun warp => reduced ⟨warp.val * 32, by have := warp.isLt; omega⟩

def sumWarpLeaders (leaders : Fin 8 → F32) : F32 :=
  (List.finRange 8).foldl (fun acc warp =>
    if warp.val == 0 then add (leaders warp) acc else add acc (leaders warp)) zero

def blockSum (read : Lane → F32) : F32 := sumWarpLeaders (warpLeaders read)

def logSumExpFromLeaders (op : Native) (p : Parameters) (leaders : Fin 8 → F32) : F32 :=
  fma (op.log2 (sumWarpLeaders leaders)) (word 0x3f317218) p.a

private def finishLogSumExp (op : Native) (p : Parameters) (read : Lane → F32) : F32 :=
  logSumExpFromLeaders op p (warpLeaders read)

def logSumExp {w : Width} (scan : Scan) (op : Native) (p : Parameters) (sigma : Column w → Half) : F32 :=
  finishLogSumExp op p (laneSum scan op p sigma)

theorem table_scan_exact (w : Width) : laneSumBy (w := w) (tableScan w) = laneSumBy (linearScan w) := by
  have order : (tableScan w).chunks = (linearScan w).chunks := by cases w <;> decide
  funext exp tid
  unfold laneSumBy
  rw [order, seed_order]
  rfl

structure PredictionWalk where
  forward : List Nat
  correction : List Nat
  reverseSeed : Bool

def dynamicWalk (n : Fin 4) : PredictionWalk :=
  ⟨List.range n.val, (List.range 3).filter (fun k => k < n.val), false⟩

def fixedIndices (n : Fin 4) : List Nat :=
  match n.val with
  | 1 => [0]
  | 2 => [0, 1]
  | 3 => [0, 1, 2]
  | _ => []

def fixedWalk (n : Fin 4) : PredictionWalk :=
  ⟨fixedIndices n, fixedIndices n, true⟩

theorem fixed_forward : ∀ n : Fin 4, (fixedWalk n).forward = (dynamicWalk n).forward := by decide
theorem fixed_correction : ∀ n : Fin 4, (fixedWalk n).correction = (dynamicWalk n).correction := by decide

/-- The CUDA correction scans the original three slots in order,
skipping slots at or above the launch count. -/
theorem fixed_correction_fold (n : Fin 4) (step : F32 → Nat → F32) :
    ((fixedWalk n).correction).foldl step zero =
      (List.range 3).foldl (fun acc k => if k < n.val then step acc k else acc) zero := by
  simp only [fixed_correction, dynamicWalk, List.foldl_filter, decide_eq_true_eq]

def Width.hasFixedPredictions (w : Width) (n : Fin 4) : Bool :=
  match w with
  | .sampled10 => n.val == 2 || n.val == 3
  | .sampled14 | .sampled24 | .full => n.val == 1

def Width.defaultSoftcap : Width → Softcap
  | .full => .defaultBC
  | _ => .defaultABC

def Width.fixedParametersMatch (w : Width) (p : Parameters) : Bool :=
  w.defaultSoftcap.accepts p && GradScale.lmHead.accepts p

def Width.fixedLaunch (w : Width) (predictions : Fin 4) (p : Parameters) : Bool :=
  w.hasFixedPredictions predictions && w.fixedParametersMatch p

def Width.launchSoftcap (w : Width) (predictions : Fin 4) (p : Parameters) : Softcap :=
  if w.fixedLaunch predictions p then w.defaultSoftcap else .runtime

def Width.launchGradScale (w : Width) (predictions : Fin 4) (p : Parameters) : GradScale :=
  if w.fixedLaunch predictions p then .lmHead else .runtime

theorem Width.launch_softcap_accepted (w : Width) (predictions : Fin 4) (p : Parameters) :
    (w.launchSoftcap predictions p).accepts p = true := by
  unfold Width.launchSoftcap
  split
  · rename_i selected
    simp only [fixedLaunch, fixedParametersMatch, Bool.and_eq_true] at selected
    exact selected.2.1
  · rfl

theorem Width.launch_grad_scale_accepted (w : Width) (predictions : Fin 4) (p : Parameters) :
    (w.launchGradScale predictions p).accepts p = true := by
  unfold Width.launchGradScale
  split
  · rename_i selected
    simp only [fixedLaunch, fixedParametersMatch, Bool.and_eq_true] at selected
    exact selected.2.2
  · rfl

def lossFromScaled {w : Width} (p : Parameters) (r : Row w)
    (positions : List Nat) (scaled : Column w → F32) (lse : F32) : F32 :=
  let plus := fun value scale c => fma scale (sub lse (scaled c)) value
  let mtp := positions.foldl (fun acc k =>
    if live : r.row.val + k < r.batchRows then plus acc (r.weights k) (r.targets ⟨_, live⟩) else acc) zero
  match r.prefixTarget with
  | none => mtp
  | some c => plus mtp p.prefixWeight c

def loss {w : Width} (p : Parameters) (r : Row w)
    (positions : List Nat) (sigma : Column w → Half) (lse : F32) : F32 :=
  lossFromScaled p r positions (fun c => (ScaledLogit.ofHalf p (sigma c)).scaled) lse

private def addWeight {w : Width} (r : Row w) (reverseSeed : Bool) (acc : F32) (k : Nat) : F32 :=
  if reverseSeed && k == 0 then add (r.weights k) acc else add acc (r.weights k)

private theorem add_weight_equal {w : Width} (r : Row w) : addWeight r true = addWeight r false := by
  funext acc k
  unfold addWeight
  split <;> simp [add, FP.ScalarOpsLaws.add_comm]

def weightSum {w : Width} (p : Parameters) (r : Row w) (walk : PredictionWalk) : F32 :=
  let mtp := walk.forward.foldl (addWeight r walk.reverseSeed) zero
  if r.prefixTarget.isSome then add p.prefixWeight mtp else mtp

def mtpTarget {w : Width} (r : Row w) (k : Nat) : Option (Column w) :=
  if live : r.row.val + k < r.batchRows then some (r.targets ⟨_, live⟩) else none

def correction {w : Width} (p : Parameters) (r : Row w) (positions : List Nat) (c : Column w) : F32 :=
  let mtp := positions.foldl (fun acc k =>
    if mtpTarget r k == some c then add acc (r.weights k) else acc) zero
  if r.prefixTarget == some c then add p.prefixWeight mtp else mtp

def corrected {w : Width} (p : Parameters) (r : Row w) (positions : List Nat) (c : Column w) : Bool :=
  positions.any (fun k => mtpTarget r k == some c) ||
    (r.prefixTarget == some c && FP.lt ftz zero p.prefixWeight)

/-- The prefix is the final slot. Its sign test controls whether that slot
can write, while its weight still participates in corrections at a matching
MTP target. Padding positions have no target. -/
def correctionSlots {w : Width} (p : Parameters) (r : Row w)
    (positions : List Nat) : List (Option (Column w)) :=
  positions.map (mtpTarget r) ++ [if FP.lt ftz zero p.prefixWeight then r.prefixTarget else none]

def firstCorrectionSlot {w : Width} (p : Parameters) (r : Row w)
    (positions : List Nat) (c : Column w) : Option Nat :=
  (correctionSlots p r positions).findIdx? (· == some c)

theorem first_correction_slot {w : Width} (p : Parameters) (r : Row w) (positions : List Nat)
    (c : Column w) (slot : Nat) :
    firstCorrectionSlot p r positions c = some slot ↔
      ∃ live : slot < (correctionSlots p r positions).length,
        (correctionSlots p r positions)[slot] = some c ∧
          ∀ prior (earlier : prior < slot),
            (correctionSlots p r positions)[prior]'(Nat.lt_trans earlier live) ≠ some c := by
  simp [firstCorrectionSlot, List.findIdx?_eq_some_iff_getElem]

theorem correction_slot_coverage {w : Width} (p : Parameters) (r : Row w)
    (positions : List Nat) (c : Column w) :
    (firstCorrectionSlot p r positions c).isSome = corrected p r positions c := by
  unfold firstCorrectionSlot correctionSlots corrected
  simp only [List.findIdx?_isSome, List.any_append, List.any_map, Function.comp_def, List.any_cons,
    List.any_nil, Bool.or_false]
  cases FP.lt ftz zero p.prefixWeight <;> simp

def WritesCorrection {w : Width} (p : Parameters) (r : Row w) (positions : List Nat)
    (c : Column w) (slot : Nat) (lane : Lane) : Prop :=
  firstCorrectionSlot p r positions c = some slot ∧ columnOwner c = lane

/-- The correction belongs to a single slot and to the lane that issued
the base gradient store covering this column. -/
theorem correction_writer_unique {w : Width} (p : Parameters) (r : Row w) (positions : List Nat)
    (c : Column w) (slot other : Nat) (lane next : Lane)
    (a : WritesCorrection p r positions c slot lane)
    (b : WritesCorrection p r positions c other next) :
    slot = other ∧ lane = next :=
  ⟨Option.some.inj (a.1.symm.trans b.1), a.2.symm.trans b.2⟩

theorem correction_writer_owns_base {w : Width} (p : Parameters) (r : Row w) (positions : List Nat)
    (c : Column w) (slot chunk : Nat) (writer baseLane : Lane) (byte : Byte)
    (write : WritesCorrection p r positions c slot writer)
    (base : column chunk baseLane byte = c.val) : writer = baseLane :=
  write.2.symm.trans (column_owner_exact c chunk baseLane byte base)

private def gradientByte (factor sigma dz : F32) : UInt8 :=
  floatToE5m2 (mul (sub one sigma) (mul sigma (mul factor dz)))

def baseGradient {w : Width} (p : Parameters) (r : Row w) (walk : PredictionWalk)
    (factor : F32) (entry : ProbabilityEntry) : UInt8 :=
  gradientByte factor entry.sigma (mul (weightSum p r walk) entry.probability)

abbrev baseGradientAddress : Nat := 2048

/-- After the row sum, the base bytes reuse the start of the exponential table. -/
def baseGradientStores {w : Width} (p : Parameters) (r : Row w)
    (factor : F32) (read : UInt8 → ProbabilityEntry) : List CudaStores.Store :=
  writeRegion baseGradientAddress fun (lane : Fin 256) =>
    baseGradient p r (fixedWalk r.predictions) factor (read (UInt8.ofNat lane.val))

def compiledCacheStores (payload : UInt8 → F32 × UInt8) : List CudaStores.Store :=
  writeBytes probabilityAddress (fun (i : Fin 1024) =>
    lowByte (payload (UInt8.ofNat (i.val / 4))).1.raw (i.val % 4)) ++
  writeRegion baseGradientAddress (fun (lane : Fin 256) => (payload (UInt8.ofNat lane.val)).2)

theorem compiled_cache_stores_exact {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (before : CudaStores.Memory) (lse : F32) :
    compiledCacheStores (fun code =>
      let entry := (readSharedTable before code).logit.forRow op lse
      (entry.probability, baseGradient p r (fixedWalk r.predictions) (readGradientFactor before) entry)) =
      probabilityStores op lse before ++
        baseGradientStores p r (readGradientFactor before)
          (fun code => (readSharedTable before code).logit.forRow op lse) := by
  have bytes : (fun (i : Fin 1024) =>
      let entry := (readSharedTable before (UInt8.ofNat (i.val / 4))).logit.forRow op lse
      lowByte entry.probability.raw (i.val % 4)) = probabilityByte op lse before := by
    funext i
    have lane : i.val / 4 < 256 := by have h := i.isLt; omega
    simp only [probabilityByte, ScaledLogit.forRow, readSharedTable, TableField.start,
      UInt8.toNat_ofNat', Nat.mod_eq_of_lt lane]
  dsimp only [compiledCacheStores]
  rw [bytes]
  rfl

theorem base_gradient_publication {w : Width} (p : Parameters) (r : Row w)
    (factor : F32) (read : UInt8 → ProbabilityEntry) (order : List CudaStores.Store)
    (schedule : order.Perm (baseGradientStores p r factor read)) (prior : CudaStores.Memory) :
    (∀ code : UInt8, CudaStores.commitAll order prior (baseGradientAddress + code.toNat) =
      baseGradient p r (fixedWalk r.predictions) factor (read code)) ∧
    (∀ address, address < baseGradientAddress ∨ baseGradientAddress + 256 ≤ address →
      CudaStores.commitAll order prior address = prior address) := by
  obtain ⟨bytes, frame⟩ := region_exact baseGradientAddress
    (fun (lane : Fin 256) => baseGradient p r (fixedWalk r.predictions) factor
      (read (UInt8.ofNat lane.val))) order schedule prior
  refine ⟨fun code => ?_, frame⟩
  simpa only [UInt8.ofNat_toNat] using bytes ⟨code.toNat, UInt8.toNat_lt_size code⟩

theorem row_cache_stores_commute {w : Width} (op : Native) (lse : F32)
    (p : Parameters) (r : Row w) (factor : F32) (read : UInt8 → ProbabilityEntry)
    (prior : CudaStores.Memory) (order : List CudaStores.Store)
    (schedule : order.Perm (probabilityStores op lse prior ++ baseGradientStores p r factor read)) :
    CudaStores.commitAll order prior =
      CudaStores.commitAll (probabilityStores op lse prior ++ baseGradientStores p r factor read) prior := by
  apply (CudaStores.unique_stores_commute _ _ _ ?_ schedule.symm).symm
  simp only [List.map_append, List.nodup_append]
  refine ⟨write_bytes_unique _ _ probability_address_unique,
    write_bytes_unique _ _ (fun _ _ same => Fin.ext (Nat.add_left_cancel same)), ?_⟩
  intro a probability b base
  simp only [probabilityStores, writeBytes, List.map_map, List.mem_map] at probability
  simp only [baseGradientStores, writeRegion, writeBytes, List.map_map, List.mem_map] at base
  obtain ⟨i, _, rfl⟩ := probability
  obtain ⟨j, _, rfl⟩ := base
  have bound := i.isLt
  change probabilityAddress i ≠ 2048 + j.val
  unfold probabilityAddress
  omega

def gradientValue {w : Width} (p : Parameters) (r : Row w)
    (walk : PredictionWalk) (factor : F32) (entry : ProbabilityEntry) (c : Column w) (patch : Bool) : UInt8 :=
  if patch then
    gradientByte factor entry.sigma
      (sub (mul (weightSum p r walk) entry.probability) (correction p r walk.correction c))
  else baseGradient p r walk factor entry

def gradient {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (walk : PredictionWalk) (sigma : Column w → Half) (lse : F32) (c : Column w) : UInt8 :=
  gradientValue p r walk (gradientFactor op p) ((ScaledLogit.ofHalf p (sigma c)).forRow op lse)
    c (corrected p r walk.forward c)

structure GradientWrites where
  base : UInt8
  correction : Option (Nat × UInt8)

def gradientWrites {w : Width} (p : Parameters) (r : Row w)
    (walk : PredictionWalk) (factor : F32) (entry : Column w → ProbabilityEntry)
    (c : Column w) : GradientWrites :=
  ⟨baseGradient p r walk factor (entry c),
    (firstCorrectionSlot p r walk.forward c).map fun slot =>
      (slot, gradientValue p r walk factor (entry c) c true)⟩

/-- Observe the base store and any later correction issued by this lane. -/
def GradientWrites.result (writes : GradientWrites) : UInt8 :=
  writes.correction.elim writes.base Prod.snd

theorem gradient_writes_exact {w : Width} (p : Parameters) (r : Row w)
    (walk : PredictionWalk) (factor : F32) (entry : Column w → ProbabilityEntry) (c : Column w) :
    (gradientWrites p r walk factor entry c).result =
      gradientValue p r walk factor (entry c) c (corrected p r walk.forward c) := by
  rw [← correction_slot_coverage]
  cases h : firstCorrectionSlot p r walk.forward c <;>
    simp [gradientWrites, GradientWrites.result, gradientValue, h]

structure Output (w : Width) where
  loss : F32
  gradient : Column w → UInt8

abbrev Run := (w : Width) → Native → Parameters → Row w → Output w

def execute (convert : Convert) (walk : Fin 4 → PredictionWalk) (scan : Width → Scan) : Run := fun w op p r =>
  let sigma := sigmoidRow convert op p r.logits
  let lse := logSumExp (scan w) op p sigma
  let order := walk r.predictions
  ⟨loss p r order.forward sigma lse, gradient op p r order sigma lse⟩

def reference : Run := execute referencePair dynamicWalk linearScan
def paired : Run := execute packedPair dynamicWalk linearScan

private def tableLogSumExp {w : Width} (op : Native) (p : Parameters) (read : Column w → LogitEntry) : F32 :=
  finishLogSumExp op p (laneSumBy (tableScan w) (fun c => (read c).contribution))

private def tableOutputWrites {w : Width} (p : Parameters) (r : Row w)
    (factor lse : F32) (read : Column w → ProbabilityEntry) : F32 × (Column w → GradientWrites) :=
  let order := fixedWalk r.predictions
  ⟨lossFromScaled p r order.forward (fun c => mul (read c).sigma p.a) lse,
    gradientWrites p r order factor read⟩

private def observeOutputWrites {w : Width} (output : F32 × (Column w → GradientWrites)) : Output w :=
  ⟨output.1, fun c => (output.2 c).result⟩

private def tabledStores {w : Width} (op : Native) (p : Parameters) (r : Row w) :
    F32 × (Column w → GradientWrites) :=
  let stagedCodes := Vector.ofFn r.logits
  let table := logitTable op p
  let lse := tableLogSumExp op p (fun (c : Column w) => readLogitTable table stagedCodes[c.val])
  let probabilities := probabilityTable op lse table
  tableOutputWrites p r (gradientFactor op p) lse (fun c => readProbabilityTable probabilities stagedCodes[c.val])

/-- The first scan uses the exponential table. After reducing logsumexp,
the CTA publishes one probability for each byte before issuing gradients. -/
def tabled : Run := fun _ op p r => observeOutputWrites (tabledStores op p r)

theorem loss_gradient_equal : paired = reference :=
  congrArg (fun convert => execute convert dynamicWalk linearScan) pairs_equal

theorem fixed_equal : execute packedPair fixedWalk linearScan = paired := by
  funext w op p r
  simp only [execute, paired]
  congr 1
  · rw [fixed_forward]
  · funext c
    simp only [gradient, gradientValue, baseGradient, weightSum, fixed_forward, fixed_correction]
    have seed : addWeight r (fixedWalk r.predictions).reverseSeed =
        addWeight r (dynamicWalk r.predictions).reverseSeed := add_weight_equal r
    rw [seed]

theorem tabled_equal : tabled = paired := by
  rw [← fixed_equal, pairs_equal]
  funext w op p r
  simp only [tabled, tabledStores, observeOutputWrites, tableLogSumExp, tableOutputWrites, execute,
    table_scan_exact, Vector.getElem_ofFn, probability_lookup_exact, table_lookup_exact, gradient_writes_exact]
  rw [show sigmoidRow referencePair op p r.logits = fun c =>
    floatToHalf (sigmoid op p (decode (r.logits c))) from funext (reference_sigmoid op p r.logits)]
  rfl

def stageRow {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (tableOrder codeOrder : List CudaStores.Store) (prior : CudaStores.Memory) :
    F32 × CudaStores.Memory :=
  let memory := CudaStores.commitAll codeOrder (CudaStores.commitAll tableOrder prior)
  let lse := tableLogSumExp op p (fun (c : Column w) => readSharedTable memory (memory (codeAddress + c.val)))
  let cache := baseGradientStores p r (readSharedWord memory gradientFactorAddress)
    (fun code => (readSharedTable memory code).logit.forRow op lse)
  (lse, CudaStores.commitAll (probabilityStores op lse memory ++ cache) memory)

def stagedLaneSums (w : Width) (memory : CudaStores.Memory) : Lane → F32 :=
  laneSumBy (tableScan w)
    (fun (c : Column w) => (readSharedTable memory (memory (codeAddress + c.val))).contribution)

def stagedWarpLeaders (w : Width) (memory : CudaStores.Memory) : Fin 8 → F32 :=
  warpLeaders (stagedLaneSums w memory)

set_option maxRecDepth 4096 in
theorem staged_row_publication {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (tables codes : List CudaStores.Store) (prior : CudaStores.Memory) (code : UInt8) :
    let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
    let lse := logSumExpFromLeaders op p (stagedWarpLeaders w before)
    let entry := (readSharedTable before code).logit.forRow op lse
    let result := stageRow op p r tables codes prior
    result.1 = lse ∧ readSharedProbability result.2 code = entry ∧
      result.2 (baseGradientAddress + code.toNat) = baseGradient p r (fixedWalk r.predictions)
        (readGradientFactor before) entry := by
  let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
  let lse := logSumExpFromLeaders op p (stagedWarpLeaders w before)
  let probabilities := probabilityStores op lse before
  let mid := CudaStores.commitAll probabilities before
  let entries := fun code => (readSharedTable before code).logit.forRow op lse
  let base := baseGradientStores p r (readGradientFactor before) entries
  let ready := CudaStores.commitAll base mid
  have state : stageRow op p r tables codes prior = (lse, ready) := by
    simp only [stageRow, CudaStores.commit_append]
    rfl
  rw [state]
  have baseWrites := base_gradient_publication p r (readGradientFactor before) entries
    base (.refl _) mid
  have preserved (field : TableField) (limit : field.start code + 4 ≤ 2048) :
      readSharedWord ready (field.start code) = readSharedWord mid (field.start code) := by
    apply read_shared_word_congr
    intro byte
    exact baseWrites.2 _ (Or.inl (Nat.lt_of_lt_of_le (Nat.add_lt_add_left byte.isLt _) limit))
  have length : code.toNat < 256 := UInt8.toNat_lt_size code
  have readback : readSharedProbability ready code = readSharedProbability mid code := by
    unfold readSharedProbability
    rw [preserved .sigma (by change 8 * code.toNat + 4 ≤ 2048; omega),
      preserved .scaled (by change 8 * code.toNat + 4 + 4 ≤ 2048; omega)]
  have probability : readSharedProbability mid code = entries code :=
    probability_read_exact op lse before probabilities (.refl _) code
  change lse = lse ∧ readSharedProbability ready code = entries code ∧
    ready (baseGradientAddress + code.toNat) =
      baseGradient p r (fixedWalk r.predictions) (readGradientFactor before) (entries code)
  refine ⟨Eq.refl lse, ?_, ?_⟩
  · rw [readback, probability]
  · exact baseWrites.1 code

def executeStagedStores {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (tableOrder codeOrder : List CudaStores.Store) (prior : CudaStores.Memory) :
    F32 × (Column w → GradientWrites) :=
  let (lse, ready) := stageRow op p r tableOrder codeOrder prior
  let output := tableOutputWrites p r (readSharedWord ready gradientFactorAddress) lse
    (fun c => readSharedProbability ready (ready (codeAddress + c.val)))
  (output.1, fun c => ⟨ready (baseGradientAddress + (readPackedCode ready (codeAddress + c.val)).toNat),
    (output.2 c).correction⟩)

def executeStaged {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (tableOrder codeOrder : List CudaStores.Store) (prior : CudaStores.Memory) : Output w :=
  observeOutputWrites (executeStagedStores op p r tableOrder codeOrder prior)

theorem staged_loss_gradient_equal {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (tableOrder : List CudaStores.Store) (initialized : tableOrder.Perm (tableStores op p))
    (codeOrder : List CudaStores.Store) (staged : codeOrder.Perm (codeStores r))
    (prior : CudaStores.Memory) :
    executeStaged op p r tableOrder codeOrder prior = reference w op p r := by
  let memory := CudaStores.commitAll codeOrder (CudaStores.commitAll tableOrder prior)
  let table := logitTable op p
  let lse := tableLogSumExp op p (fun (c : Column w) => readLogitTable table (r.logits c))
  let writes := probabilityStores op lse memory
  let factorBefore := readSharedWord memory gradientFactorAddress
  let cache := baseGradientStores p r factorBefore
    (fun code => (readSharedTable memory code).logit.forRow op lse)
  let probabilities := CudaStores.commitAll writes memory
  let ready := CudaStores.commitAll cache probabilities
  have reads := funext (staged_table_read op p r tableOrder initialized codeOrder staged prior)
  have baseFactor : factorBefore = gradientFactor op p :=
    staged_gradient_factor op p r tableOrder initialized codeOrder staged prior
  have publication := base_gradient_publication p r factorBefore
    (fun code => (readSharedTable memory code).logit.forRow op lse) cache (.refl cache) probabilities
  have frame (address : Nat) (outside : 2304 ≤ address) : ready address = memory address :=
    (publication.2 address (Or.inr outside)).trans
      (probability_store_frame op lse memory writes (.refl writes) address (Or.inl (by omega)))
  have factor : readSharedWord ready gradientFactorAddress = gradientFactor op p := by
    rw [← baseFactor]
    apply read_shared_word_congr
    intro byte
    exact frame _ (by change 2304 ≤ 3072 + byte.val; omega)
  have codeFrame (c : Column w) : ready (codeAddress + c.val) = memory (codeAddress + c.val) :=
    frame _ (by change 2304 ≤ 3076 + c.val; omega)
  have probFrame (code : UInt8) : readSharedProbability ready code = readSharedProbability probabilities code := by
    have word (field : TableField) (limit : field.start code + 4 ≤ baseGradientAddress) :
        readSharedWord ready (field.start code) = readSharedWord probabilities (field.start code) := by
      apply read_shared_word_congr
      intro byte
      exact publication.2 _ (Or.inl (Nat.lt_of_lt_of_le (Nat.add_lt_add_left byte.isLt _) limit))
    have bound : code.toNat < 256 := UInt8.toNat_lt_size code
    unfold readSharedProbability
    rw [word .sigma (by change 8 * code.toNat + 4 ≤ 2048; omega),
      word .scaled (by change 8 * code.toNat + 4 + 4 ≤ 2048; omega)]
  have loaded (c : Column w) :
      readSharedProbability ready (ready (codeAddress + c.val)) =
        readProbabilityTable (probabilityTable op lse table) (r.logits c) := by
    rw [codeFrame, probFrame]
    have prob := probability_read_exact op lse memory writes (.refl writes) (memory (codeAddress + c.val))
    rw [prob, probability_lookup_exact, staged_table_read op p r tableOrder initialized codeOrder staged prior]
  have cached (c : Column w) :
      ready (baseGradientAddress + (ready (codeAddress + c.val)).toNat) =
        baseGradient p r (fixedWalk r.predictions) (gradientFactor op p)
          (readProbabilityTable (probabilityTable op lse table) (r.logits c)) := by
    rw [codeFrame]
    change CudaStores.commitAll cache probabilities _ = _
    rw [publication.1, baseFactor, probability_lookup_exact,
      staged_table_read op p r tableOrder initialized codeOrder staged prior]
  rw [← loss_gradient_equal, ← tabled_equal]
  simp only [executeStaged, executeStagedStores, stageRow, packed_shared_code_exact, tabled, tabledStores,
    Vector.getElem_ofFn]
  rw [reads, CudaStores.commit_append]
  change observeOutputWrites (_, fun c => ⟨ready (baseGradientAddress + (ready (codeAddress + c.val)).toNat), _⟩) = _
  rw [factor, funext loaded]
  apply congrArg observeOutputWrites
  refine Prod.ext rfl ?_
  funext c
  change (⟨ready (baseGradientAddress + (ready (codeAddress + c.val)).toNat), _⟩ : GradientWrites) = _
  rw [cached c]
  rfl

/-- Fixed prediction counts with matching launch constants use the
staged table. The other launches use the paired dynamic scan. -/
def specialized : Run := fun w op p r =>
  let mode := w.launchSoftcap r.predictions p
  if mode = .runtime then paired w op p r else
    let prepared := (w.launchGradScale r.predictions p).parameters (mode.parameters p)
    executeStaged op prepared r (tableStores op prepared) (codeStores r) (fun _ => 0)

theorem specialized_equal : specialized = reference := by
  funext w op p r
  unfold specialized
  dsimp only
  split
  · rw [loss_gradient_equal]
  · have exactParameters := Softcap.parameters_exact (w.launchSoftcap r.predictions p) p
      (w.launch_softcap_accepted r.predictions p)
    rw [exactParameters, GradScale.parameters_exact _ p (w.launch_grad_scale_accepted r.predictions p)]
    exact staged_loss_gradient_equal op p r _ (.refl _) _ (.refl _) _

#axiomsClean unpack_exact
#axiomsClean Softcap.parameters_exact
#axiomsClean GradScale.parameters_exact
#axiomsClean GradScale.lm_head_word
#axiomsClean Width.launch_softcap_accepted
#axiomsClean Width.launch_grad_scale_accepted
#axiomsClean unpack_halves_exact
#axiomsClean pairs_equal
#axiomsClean column_unique
#axiomsClean column_covered
#axiomsClean column_owner_exact
#axiomsClean reference_sigmoid
#axiomsClean table_lookup_exact
#axiomsClean packed_shared_code_exact
#axiomsClean probability_lookup_exact
#axiomsClean staged_code_exact
#axiomsClean published_table_words
#axiomsClean published_gradient_factor
#axiomsClean staged_gradient_factor
#axiomsClean staged_table_read
#axiomsClean probability_other_lanes_preserve_scaled
#axiomsClean probability_store_frame
#axiomsClean probability_read_exact
#axiomsClean probability_stores_commute
#axiomsClean base_gradient_publication
#axiomsClean row_cache_stores_commute
#axiomsClean gradient_store_order
#axiomsClean table_scan_exact
#axiomsClean loss_gradient_equal
#axiomsClean fixed_correction_fold
#axiomsClean first_correction_slot
#axiomsClean correction_slot_coverage
#axiomsClean correction_writer_unique
#axiomsClean correction_writer_owns_base
#axiomsClean gradient_writes_exact
#axiomsClean tabled_equal
#axiomsClean staged_loss_gradient_equal
#axiomsClean staged_row_publication
#axiomsClean specialized_equal
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCE
