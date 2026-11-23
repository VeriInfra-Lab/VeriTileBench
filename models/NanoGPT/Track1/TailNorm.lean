import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-! FP32 NormTwoOps for ATen's aligned, contiguous CUDA all-reduce.
The two input formats have the same four-value vectors, 512-thread
blocks, accumulator format, decreasing shuffle offsets and staging order.
The FP32 projection (CUDA sqrt) is applied to the resulting accumulator. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TailNorm
open VeriTile.Triton FP

abbrev F32 := Value .fp32
abbrev BF16 := Value .bf16
abbrev Bank := Nat → BF16

def zero : F32 := Value.zero .fp32
def one : F32 := Value.ofBits .fp32 0x3f800000
def widen (x : BF16) : F32 := FP.cast .fp32 Config.ieeeValue x
def narrow (x : F32) : BF16 := FP.cast .bf16 Config.ieeeValue x

def ceilDiv (n d : Nat) : Nat := (n + d - 1) / d

structure Device where
  multiprocessors : Nat
  maxThreadsPerSM : Nat

def Device.ctas (d : Device) (size : Nat) : Nat :=
  let valuesPerThread := ceilDiv size 512
  let targetGrid := d.multiprocessors * (d.maxThreadsPerSM / 512)
  if valuesPerThread ≥ 256 && targetGrid ≥ 1 then
    max (min targetGrid (ceilDiv valuesPerThread 16)) (ceilDiv valuesPerThread 256)
  else 1

def vectorHead (base scalarBytes : Nat) : Nat := base % (4 * scalarBytes) / scalarBytes

theorem aligned_vector_heads (baseBF16 baseFP32 : Nat) (hbf : baseBF16 % 16 = 0)
    (hfp : baseFP32 % 16 = 0) : vectorHead baseBF16 2 = 0 ∧ vectorHead baseFP32 4 = 0 := by
  unfold vectorHead
  omega

def add (a b : F32) : F32 := FP.add Config.ieeeValue a b

def squareStep (contract : Bool) (acc x : F32) : F32 :=
  if contract then FP.fma Config.ieeeValue x x acc else add acc (FP.mul Config.ieeeValue x x)

def threadAccumulator (contract : Bool) (size ctas block lane component : Nat) (read : Nat → F32) : F32 :=
  let step := 512 * ctas
  let initial := 512 * block + lane
  (List.range (ceilDiv (size / 4) step)).foldl (fun acc iteration =>
    let vector := initial + iteration * step
    if vector < size / 4 then squareStep contract acc (read (4 * vector + component)) else acc) zero

def threadSum (contract : Bool) (size ctas block lane : Nat) (read : Nat → F32) : F32 :=
  let acc := fun i => threadAccumulator contract size ctas block lane i read
  add (add (add (acc 0) (acc 1)) (acc 2)) (acc 3)

def combineBlockX (width : Nat) (read : Nat → F32) : F32 :=
  (((List.range width.log2).reverse).foldl (fun values shift lane =>
    let offset := 2 ^ shift
    if lane < offset then add (values lane) (values (lane + offset)) else values lane) read) 0

def combineThreads : (Nat → F32) → F32 := combineBlockX 512

theorem combine_threads_fixed (read : Nat → F32) :
    combineThreads read =
      (([256,128,64,32,16,8,4,2,1] : List Nat).foldl (fun values offset lane =>
        if lane < offset then add (values lane) (values (lane + offset)) else values lane) read) 0 := by
  rfl

/-- The last CTA reads the staging buffer in CTA-index order in each
lane, then applies the same block reduction to those lane accumulators. -/
def combineBlocks (ctas : Nat) (blockSum : Nat → F32) : F32 :=
  if ctas ≤ 1 then blockSum 0
  else combineThreads fun lane =>
    (List.range (ceilDiv ctas 512)).foldl (fun acc i =>
      let block := lane + 512 * i
      if block < ctas then add acc (blockSum block) else acc) zero

def squares (contract : Bool) (size ctas : Nat) (read : Nat → F32) : F32 :=
  combineBlocks ctas (fun block => combineThreads (fun lane => threadSum contract size ctas block lane read))

structure Math where
  device : Device
  contract : Bool
  sqrt : F32 → F32

def sourceNorm (a : Math) (size : Nat) (weights : Bank) : F32 :=
  let fp32 := fun byte => widen (weights (byte / 4))
  a.sqrt (squares a.contract size (a.device.ctas size) (fun i => fp32 (4 * i)))

def directNorm (a : Math) (size : Nat) (weights : Bank) : F32 :=
  let bf16 := fun byte => weights (byte / 2)
  a.sqrt (squares a.contract size (a.device.ctas size) (fun i => widen (bf16 (2 * i))))

theorem norm_equal (a : Math) (size : Nat) (weights : Bank) :
    directNorm a size weights = sourceNorm a size weights := by
  have leaves : (fun i => widen (weights ((2 * i) / 2))) =
      (fun i => widen (weights ((4 * i) / 4))) := by
    funext i
    congr 2 <;> omega
  unfold directNorm sourceNorm
  exact congrArg (fun read => a.sqrt (squares a.contract size (a.device.ctas size) read)) leaves

inductive Slot where
  | embedding | lmHead | mlp | qk | vo
  deriving DecidableEq

def Slot.id : Slot → Nat
  | .embedding => 0
  | .lmHead => 1
  | .mlp => 2
  | .qk => 3
  | .vo => 4

def Slot.elements : Slot → Nat
  | .embedding | .lmHead => 38633472
  | .mlp => 51904512
  | .qk | .vo => 9437184

def tileSize : Nat := 4096
def tiles : Nat := 36144

def Slot.start : Slot → Nat
  | .embedding => 0
  | .lmHead => 9432
  | .mlp => 18864
  | .qk => 31536
  | .vo => 33840

def decode (cta : Nat) : Slot × Nat :=
  if cta < 9432 then (.embedding, cta)
  else if cta < 18864 then (.lmHead, cta - 9432)
  else if cta < 31536 then (.mlp, cta - 18864)
  else if cta < 33840 then (.qk, cta - 31536)
  else (.vo, cta - 33840)

theorem decode_owner (slot : Slot) (tile : Nat) (live : tile < slot.elements / tileSize) :
    decode (slot.start + tile) = (slot, tile) := by
  cases slot <;> simp only [Slot.start, Slot.elements, tileSize] at *
  · simp [decode, show tile < 9432 by omega]
  · simp [decode, show ¬9432 + tile < 9432 by omega, show 9432 + tile < 18864 by omega]
  · simp [decode, show ¬18864 + tile < 9432 by omega, show ¬18864 + tile < 18864 by omega,
      show 18864 + tile < 31536 by omega]
  · simp [decode, show ¬31536 + tile < 9432 by omega, show ¬31536 + tile < 18864 by omega,
      show ¬31536 + tile < 31536 by omega, show 31536 + tile < 33840 by omega]
  · simp [decode, show ¬33840 + tile < 9432 by omega, show ¬33840 + tile < 18864 by omega,
      show ¬33840 + tile < 31536 by omega, show ¬33840 + tile < 33840 by omega]

theorem decode_live (cta : Fin tiles) :
    (decode cta.val).2 * tileSize + tileSize ≤ (decode cta.val).1.elements := by
  have h := cta.isLt
  unfold tiles at h
  by_cases h₁ : cta.val < 9432
  · simp [decode, h₁, Slot.elements, tileSize]; omega
  by_cases h₂ : cta.val < 18864
  · simp [decode, h₁, h₂, Slot.elements, tileSize]; omega
  by_cases h₃ : cta.val < 31536
  · simp [decode, h₁, h₂, h₃, Slot.elements, tileSize]; omega
  by_cases h₄ : cta.val < 33840
  · simp [decode, h₁, h₂, h₃, h₄, Slot.elements, tileSize]; omega
  simp [decode, h₁, h₂, h₃, h₄, Slot.elements, tileSize]
  omega

theorem full_tile (slot : Slot) : slot.elements % tileSize = 0 := by cases slot <;> decide

private theorem thread_bits : ∀ thread : Fin 256, ((thread.val <<< 3) &&& 2040) = thread.val * 8 := by
  decide +kernel

def ptxIndex (cta : Fin tiles) (thread : Fin 256) (plane : Fin 2) (word : Fin 8) : Nat :=
  (((decode cta.val).2 <<< 12) ||| ((thread.val <<< 3) &&& 2040)) + plane.val * 2048 + word.val

theorem ptx_index_exact (cta : Fin tiles) (thread : Fin 256) (plane : Fin 2) (word : Fin 8) :
    ptxIndex cta thread plane word =
      (decode cta.val).2 * tileSize + plane.val * 2048 + thread.val * 8 + word.val := by
  have bound : thread.val * 8 < 2 ^ 12 := by have := thread.isLt; omega
  rw [ptxIndex, thread_bits, ← Nat.shiftLeft_add_eq_or_of_lt bound, Nat.shiftLeft_eq]
  unfold tileSize
  omega

theorem ptx_index_live (cta : Fin tiles) (thread : Fin 256) (plane : Fin 2) (word : Fin 8) :
    ptxIndex cta thread plane word < (decode cta.val).1.elements := by
  have ht := thread.isLt
  have hp := plane.isLt
  have hw := word.isLt
  have live := decode_live cta
  rw [ptx_index_exact]
  unfold tileSize at *
  omega

theorem tile_thread_owner (i : Fin tileSize) :
    i.val / 2048 * 2048 + i.val % 2048 / 8 * 8 + i.val % 8 = i.val := by
  omega

def ratio (before after : F32) : F32 :=
  let quotient := FP.div Config.ieeeValue before after
  if FP.lt Config.ieeeValue zero after then quotient else one

/-! The baseline's CPU scalar is promoted from the FP32 tensor to a
Python double, then passed to ATen's FP32 opmath scalar. Represent the
intermediate by the sign, significand and exponent of that promoted
value. Binary64 has 29 more fraction bits and a wider exponent range. -/
def hostScalar (x : F32) : F32 :=
  let exp := x.raw / 8388608
  let fraction := x.raw % 8388608
  Value.ofBits .fp32 (exp * 8388608 + (fraction * 536870912) / 536870912)

theorem host_scalar_exact (x : F32) : hostScalar x = x := by
  have bits : x.raw / 8388608 * 8388608 + x.raw % 8388608 * 536870912 / 536870912 = x.raw := by
    omega
  calc
    hostScalar x = Value.ofBits .fp32 x.raw := congrArg (Value.ofBits .fp32) bits
    _ = x := by cases x; simp [Value.ofBits, Value.raw, BitVec.ofNat_toNat]

def scaled (gain : F32) (weight : BF16) : BF16 :=
  narrow (FP.mul Config.ieeeValue gain (widen weight))

structure Problem where
  direct : Bool
  arithmetic : Math
  fallbackNorm : Slot → Bank → F32
  before : Slot → Bank
  after : Slot → Bank

abbrev Update := Problem → Slot → Bank

def reference : Update := fun p slot i =>
  let norm := if p.direct then sourceNorm p.arithmetic slot.elements else p.fallbackNorm slot
  let gain := ratio (norm (p.before slot)) (norm (p.after slot))
  scaled (hostScalar gain) (p.after slot i)

def grouped : Update := fun p slot i =>
  if p.direct && i < slot.elements then
    let (bank, tile) := decode (slot.start + i / tileSize)
    let norm := directNorm p.arithmetic bank.elements
    let gain := ratio (norm (p.before bank)) (norm (p.after bank))
    scaled gain (p.after bank (tile * tileSize + i % tileSize))
  else reference p slot i

theorem grouped_equal (p : Problem) (slot : Slot) (i : Nat) :
    grouped p slot i = reference p slot i := by
  unfold grouped
  split
  · rename_i h
    have ready := (Bool.and_eq_true_iff.mp h).1
    have hi := of_decide_eq_true (Bool.and_eq_true_iff.mp h).2
    have live : i / tileSize < slot.elements / tileSize := by
      have aligned := full_tile slot
      unfold tileSize at *
      omega
    rw [decode_owner slot (i / tileSize) live]
    have offset : i / tileSize * tileSize + i % tileSize = i := by
      unfold tileSize
      omega
    simp [norm_equal, reference, ready, host_scalar_exact, offset]
  · rfl

theorem removed_cast_traffic :
    (38633472 * 2 + 51904512 + 9437184 * 2) * 16 = 2368733184 := by decide

#axiomsClean aligned_vector_heads
#axiomsClean norm_equal
#axiomsClean decode_owner
#axiomsClean decode_live
#axiomsClean ptx_index_exact
#axiomsClean ptx_index_live
#axiomsClean tile_thread_owner
#axiomsClean host_scalar_exact
#axiomsClean grouped_equal
end VeriTile.Bench.Models.NanoGPT.Track1.TailNorm
