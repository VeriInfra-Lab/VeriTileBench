import models.NanoGPT.Track1.HostFloat64
import models.NanoGPT.Track1.TailNorm

/-! The numerical validation report starts from the contiguous FP32
per-token loss returned by the model. It replays the ATen 2.10 CUDA mean,
the Python loop and CUDA scalar division, and the one-element NCCL 2.27.5
FP32 ring reduce. The environment supplies the ring order and each rank's
scalar payload. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationLoss
open VeriTile.Triton FP
open TailNorm (F32 add zero ceilDiv)

structure Batch where
  words : Array F32
  /-- Number of FP32 words by which the address is ahead of a vector boundary. -/
  alignment : Fin 4
  device : TailNorm.Device

structure MeanLayout where
  vectorized : Bool
  width : Nat
  ctas : Nat
  deriving BEq

def meanLayout (device : TailNorm.Device) (size : Nat) : MeanLayout :=
  let vectorized := size ≥ 128
  let columns := if vectorized then size / 4 else size
  let width := 2 ^ (min 512 columns).log2
  ⟨vectorized, width, if width = 512 then device.ctas size else 1⟩

private def sumStride (count first stride : Nat) (read : Nat → F32) (initial : F32) : F32 :=
  (List.range (ceilDiv (count - first) stride)).foldl (fun acc i => add acc (read (first + i * stride))) initial

private def sumComponents (component : Nat → F32) : F32 :=
  add (add (add (component 0) (component 1)) (component 2)) (component 3)

/-- FP32 `vt0=4`. The scalar path unrolls across successive strides. The
vector path keeps one accumulator per adjacent word of the aligned load.
Only CTA zero can add an unaligned head or a partial tail. -/
def meanThread (batch : Batch) (layout : MeanLayout) (cta lane : Nat) : F32 :=
  let stride := layout.width * layout.ctas
  let first := cta * layout.width + lane
  let read := fun index => batch.words[index]?.getD zero
  if layout.vectorized then
    let shift := batch.alignment.val
    let head := if shift = 0 then 0 else 4 - shift
    let aligned := batch.words.size - head
    let vectors := aligned / 4
    let tail := 4 * vectors + lane
    sumComponents fun component =>
      let headSum := if component = 0 && cta = 0 && shift > 0 && shift ≤ lane && lane < 4
        then add zero (read (lane - shift)) else zero
      let middle := sumStride vectors first stride (fun v => read (head + 4 * v + component)) headSum
      if component = 0 && cta = 0 && tail < aligned then add middle (read (head + tail))
      else middle
  else
    sumComponents fun component => sumStride batch.words.size (first + component * stride) (stride * 4) read zero

private def meanFactor (count : Nat) : F32 :=
  let cfg := Config.ieeeValue
  FP.div cfg TailNorm.one (FP.round .fp32 cfg (count : Rat))

def batchMean (batch : Batch) : F32 :=
  if batch.words.isEmpty then Value.qNaN .fp32 else
    let layout := meanLayout batch.device batch.words.size
    let sum := TailNorm.combineBlocks layout.ctas (fun cta =>
      TailNorm.combineBlockX layout.width (meanThread batch layout cta))
    FP.mul Config.ieeeValue sum (meanFactor batch.words.size)

theorem hopper_default_mean_geometry :
    meanLayout ⟨132, 2048⟩ 262144 = ⟨true, 512, 32⟩ := by rfl

def alignedAddress (cta : Fin 32) (lane : Fin 512) (component iteration : Fin 4) : Nat :=
  4 * (cta.val * 512 + lane.val + iteration.val * 16384) + component.val

theorem aligned_default_address (cta : Fin 32) (lane : Fin 512) (component iteration : Fin 4) :
    alignedAddress cta lane component iteration < 262144 := by
  have := cta.isLt
  have := lane.isLt
  have := component.isLt
  have := iteration.isLt
  unfold alignedAddress
  omega

theorem aligned_default_covers (i : Fin 262144) :
    ∃ (cta : Fin 32) (lane : Fin 512) (component iteration : Fin 4),
      alignedAddress cta lane component iteration = i.val := by
  have hi := i.isLt
  refine ⟨⟨i.val / 2048 % 32, by omega⟩, ⟨i.val / 4 % 512, by omega⟩,
    ⟨i.val % 4, by omega⟩, ⟨i.val / 65536, by omega⟩, ?_⟩
  dsimp [alignedAddress]
  omega

theorem aligned_default_unique {a b : Fin 32} {l k : Fin 512} {x y i j : Fin 4}
    (same : alignedAddress a l x i = alignedAddress b k y j) :
    a = b ∧ l = k ∧ x = y ∧ i = j := by
  simp only [Fin.ext_iff]
  unfold alignedAddress at same
  have := a.isLt
  have := b.isLt
  have := l.isLt
  have := k.isLt
  have := x.isLt
  have := y.isLt
  have := i.isLt
  have := j.isLt
  omega

/-- `val_tokens % val_batch_size == 0` precedes the integer quotient and
the calls that consume the validation loader. -/
def batchCount (tokens batchSize : Nat) : Option Nat :=
  if batchSize = 0 || tokens % batchSize != 0 then none else some (tokens / batchSize)

theorem default_batch_count : batchCount 10485760 2097152 = some 5 := by rfl

/-- The CUDA true-division kernel with a CPU scalar divides on the host,
narrows the reciprocal to FP32, then multiplies on the device. -/
def scaleBatches (sum : F32) (count : Nat) : F32 :=
  let cfg := Config.ieeeValue
  let inv := HostFloat64.roundPositive (1 / HostFloat64.roundPositive (count : Rat))
  FP.mul cfg sum (FP.round .fp32 cfg inv)

theorem batch_scaling_rounding :
    let sum := Value.ofBits .fp32 0x418332f3
    (scaleBatches sum 5).raw = 0x4051eb1f ∧
    (FP.div Config.ieeeValue sum (Value.ofBits .fp32 0x40a00000)).raw = 0x4051eb1e := by
  decide +kernel

structure Ring where
  order : List (Fin 8)
  perm : order.Perm (List.finRange 8)
  rootLast : order.getLast? = some 0

def eighth : F32 := Value.ofBits .fp32 0x3e000000

def rankStep (own incoming : F32) : F32 := FP.fma Config.ieeeValue own eighth incoming

theorem rank_hop_rounding :
    let contribution := Value.ofBits .fp32 4
    let incoming := Value.ofBits .fp32 1
    (rankStep contribution incoming).raw = 2 ∧
      (add (FP.mul Config.ieeeValue contribution eighth) incoming).raw = 1 := by
  decide +kernel

/-- NCCL uses `PreMulSum` for FP32 AVG. The rank after the root sends the
first scaled value. The pinned SM90 LL and Simple paths fuse subsequent
hops. LL128 also fuses the first word, which is the sole live word in the
validation collective. The ring ends with the local contribution at rank zero. -/
def rankAverage (ring : Ring) (get : Fin 8 → F32) : Option F32 :=
  match ring.order with
  | [] => none
  | first :: rest =>
    let cfg := Config.ieeeValue
    some (rest.foldl (fun incoming rank => rankStep (get rank) incoming) (FP.mul cfg (get first) eighth))

theorem rank_average_total (ring : Ring) (get : Fin 8 → F32) : (rankAverage ring get).isSome = true := by
  cases h : ring.order with
  | nil => have bad := ring.rootLast; simp [h] at bad
  | cons rank rest => simp [rankAverage, h]

structure Exchange where
  rank : Fin 8
  ring : Ring
  peers : Fin 8 → F32

/-- The root's own value comes from this execution. A non-root rank
participates in the collective without producing a reported loss. -/
def rootLoss (e : Exchange) (own : F32) : Option F32 :=
  if e.rank = 0 then rankAverage e.ring (fun rank => if rank = 0 then own else e.peers rank)
  else none

theorem root_uses_own (e : Exchange) (own : F32) (root : e.rank = 0) :
    rootLoss e own = rankAverage e.ring (fun rank => if rank = 0 then own else e.peers rank) := by
  simp [rootLoss, root]

#axiomsClean hopper_default_mean_geometry
#axiomsClean aligned_default_address
#axiomsClean aligned_default_covers
#axiomsClean aligned_default_unique
#axiomsClean batch_scaling_rounding
#axiomsClean rank_hop_rounding
#axiomsClean rank_average_total
#axiomsClean root_uses_own
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationLoss
