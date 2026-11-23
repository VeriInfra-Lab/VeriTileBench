import models.NanoGPT.Track1.RopeShift

/-! The QKV backward epilogue. The 128-channel path removes unused K
gradient loads. The 64-channel paths include the paired head routing
and the shorter value heads. All three preserve the ordered FP32
derivative, the BF16 round trip, and both E4M3 store layouts. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.RopeBackward
open RopeShift (Length Head Channel Arithmetic)
open VeriTile.Triton FP

def pairedChannel (d : Channel) : Channel :=
  ⟨d.val ^^^ 1, Nat.xor_lt_two_pow (n := 7) d.isLt (by decide)⟩

theorem rotating_partner {d : Channel} (h : d.val < 64) : (pairedChannel d).val < 64 :=
  Nat.xor_lt_two_pow (n := 6) h (by decide)

theorem xor_one_reverses_pair (d : Nat) :
    d ^^^ 1 = 2 * (d / 2) + (1 - d % 2) := by
  have half : (d ^^^ 1) / 2 = d / 2 := by simp [Nat.xor_div_two]
  have low : (d ^^^ 1) % 2 = 1 ↔ d % 2 ≠ 1 := by simp
  omega

/-- Reshaping to 64 adjacent pairs and reversing the last axis. -/
def reversedPairChannel (d : Channel) : Channel :=
  ⟨2 * (d.val / 2) + (1 - d.val % 2), by have := d.isLt; omega⟩

theorem reversed_pair_index : ∀ d : Channel, reversedPairChannel d = pairedChannel d := by
  intro d
  exact Fin.ext (xor_one_reverses_pair d.val).symm

/-- The TTGIR flip lowers through an XOR reduction over each BF16 pair. -/
theorem reversed_pair_bits (lo hi : BitVec 16) :
    (lo ^^^ (lo ^^^ hi), hi ^^^ (lo ^^^ hi)) = (hi, lo) := by
  have low : lo ^^^ (lo ^^^ hi) = hi := by
    rw [← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]
  have high : hi ^^^ (lo ^^^ hi) = lo := by
    rw [BitVec.xor_comm lo hi, ← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]
  rw [low, high]

def loadCurrent {α : Type} (zero : α) (read : Channel → α) (token : Nat) (d : Channel) : α :=
  if d.val < 64 ∨ token = 0 then read d else zero

def loadSuccessor {α : Type} (zero : α) (read : Channel → α) (d : Channel) : α :=
  if 64 ≤ d.val then read d else zero

/-- The successor's validity mask is common and is included in successor.
The original trace at the selected gradient retains both terms at token zero. -/
def selected {α : Type} (rotate add : α → α → α) (current successor : Channel → α)
    (token : Nat) (d : Channel) : α :=
  if 64 ≤ d.val then
    if token = 0 then add (current d) (successor d) else successor d
  else rotate (current d) (current (pairedChannel d))

theorem trimmed_loads_equal {α : Type} (zero : α) (rotate add : α → α → α)
    (current successor : Channel → α) (token : Nat) (d : Channel) :
    selected rotate add (loadCurrent zero current token) (loadSuccessor zero successor) token d =
      selected rotate add current successor token d := by
  by_cases h : 64 ≤ d.val
  · by_cases first : token = 0 <;> simp [selected, h, first, loadCurrent, loadSuccessor]
  · have rotating : d.val < 64 := by omega
    simp [selected, h, rotating, rotating_partner rotating, loadCurrent]

inductive Band where | query | key | value
  deriving DecidableEq

def Band.base : Band → Nat | .query => 0 | .key => 6 | .value => 12

structure Point (n : Length) where
  token : Fin n.tokens
  band : Band
  head : Head
  channel : Channel

abbrev BF16 := Value .bf16
abbrev F32 := Value .fp32
abbrev ChannelRow := Channel → F32

structure Inputs (n : Length) where
  qk : Fin n.tokens → Band → Head → Channel → BF16
  grad : Fin n.tokens → Band → Head → Channel → BF16
  factor1 : Fin n.tokens → Channel → BF16
  factor2 : Fin n.tokens → Channel → BF16
  scale : F32

def feature {n : Length} (p : Point n) : Nat :=
  (p.band.base + p.head.val) * 128 + p.channel.val

def rowAddress {n : Length} (p : Point n) : Nat := p.token.val * 2304 + feature p
def transposedAddress {n : Length} (p : Point n) : Nat := feature p * n.tokens + p.token.val

theorem feature_bound {n : Length} (p : Point n) : feature p < 2304 := by
  have hh := p.head.isLt
  have hc := p.channel.isLt
  cases p with
  | mk t band h d =>
    cases band <;> simp [feature, Band.base] <;> omega

theorem addresses_bounded {n : Length} (p : Point n) :
    rowAddress p < n.tokens * 2304 ∧ transposedAddress p < n.tokens * 2304 ∧
      n.tokens * 2304 < 2^31 := by
  have ht := p.token.isLt
  have hf := feature_bound p
  cases n <;> simp only [Length.tokens, rowAddress, transposedAddress] at * <;> omega

theorem token_tile_covers {n : Length} (t : Fin n.tokens) :
    t.val / 32 < n.tokens / 32 ∧ 32 * (t.val / 32) + t.val % 32 = t.val := by
  have ht := t.isLt
  cases n <;> simp only [Length.tokens] at * <;> omega

def keyReadBytes {n : Length} (trim : Bool) (t : Fin n.tokens) : Nat :=
  let current := if trim && t.val != 0 then 128 else 256
  let successor := if t.val + 1 < n.tokens then if trim then 128 else 256 else 0
  current + successor

theorem interior_key_reads {n : Length} (t : Fin n.tokens) (first : t.val ≠ 0)
    (room : t.val + 1 < n.tokens) :
    keyReadBytes false t = 512 ∧ keyReadBytes true t = 256 := by
  simp [keyReadBytes, first, room]

private theorem point_eq {n : Length} (p q : Point n)
    (tokenEq : p.token = q.token) (featureEq : feature p = feature q) : p = q := by
  have hp := p.head.isLt
  have hq := q.head.isLt
  have dp := p.channel.isLt
  have dq := q.channel.isLt
  have hchan : p.channel = q.channel := Fin.ext (by unfold feature at featureEq; omega)
  have headEq : p.band.base + p.head.val = q.band.base + q.head.val := by
    unfold feature at featureEq
    omega
  cases p with
  | mk t a h d =>
    cases q with
    | mk u b j e =>
      cases a <;> cases b <;> simp only [Band.base] at headEq <;>
        first
        | omega
        | have hd : h = j := Fin.ext (by omega)
          congr

theorem row_address_unique {n : Length} (p q : Point n)
    (h : rowAddress p = rowAddress q) : p = q := by
  have hp := feature_bound p
  have hq := feature_bound q
  unfold rowAddress at h
  exact point_eq p q (Fin.ext (by omega)) (by omega)

theorem transpose_address_unique {n : Length} (p q : Point n)
    (h : transposedAddress p = transposedAddress q) : p = q := by
  have hp := p.token.isLt
  have hq := q.token.isLt
  unfold transposedAddress at h
  cases n <;> simp only [Length.tokens] at hp hq h <;>
    exact point_eq p q (Fin.ext (by omega)) (by omega)

private def successor {n : Length} (a : Arithmetic) (r : Inputs n) (p : Point n) : ChannelRow :=
  fun d =>
    if room : p.token.val + 1 < n.tokens then
      FP.cast .fp32 a.config (r.grad ⟨p.token.val + 1, room⟩ .key p.head d)
    else Value.zero .fp32 false

private def current {n : Length} (a : Arithmetic) (r : Inputs n) (p : Point n) : ChannelRow :=
  fun d => FP.cast .fp32 a.config (r.grad p.token p.band p.head d)

private def rotaryVjp {n : Length} (contiguous : Bool) (a : Arithmetic) (r : Inputs n) (p : Point n)
    (d : Channel) (g flipped : F32) : F32 :=
  let f1 := FP.cast .fp32 a.config (r.factor1 p.token d)
  let d2 := if contiguous then reversedPairChannel d else pairedChannel d
  let f2 := FP.cast .fp32 a.config (r.factor2 p.token d2)
  FP.fma a.config f1 g (FP.mul a.config f2 flipped)

private theorem rotary_vjp_equal {n : Length} (a : Arithmetic) (r : Inputs n) (p : Point n) (d : Channel) :
    rotaryVjp true a r p d = rotaryVjp false a r p d := by
  funext g flipped
  simp [rotaryVjp, reversed_pair_index]

/-- The loaded successor at the last token is +0. For a stationary
channel the first token retains the addition of its two contributions. -/
def normalizedGradient {n : Length} (trim : Bool) (a : Arithmetic) (r : Inputs n)
    (p : Point n) : ChannelRow := fun d =>
  let incoming := current a r p
  if p.band = .key then
    let following := successor a r p
    let zero := Value.zero .fp32 false
    if trim then
      selected (rotaryVjp true a r p d) (FP.add a.config) (loadCurrent zero incoming p.token.val)
        (loadSuccessor zero following) p.token.val d
    else
      selected (rotaryVjp false a r p d) (FP.add a.config) incoming following p.token.val d
  else rotaryVjp trim a r p d (incoming d) (incoming (pairedChannel d))

theorem normalized_gradient_equal {n : Length} (a : Arithmetic) (r : Inputs n) (p : Point n) :
    normalizedGradient true a r p = normalizedGradient false a r p := by
  funext d
  unfold normalizedGradient
  simp only [rotary_vjp_equal]
  split
  · exact trimmed_loads_equal _ _ _ _ _ _ _
  · rfl

private def channel {lanes : Nat} (lane : Fin lanes) (reg : Fin 16) : Fin (16 * lanes) :=
  ⟨16 * lane.val + reg.val, by have := lane.isLt; have := reg.isLt; omega⟩

private def laneOf {lanes : Nat} (d : Fin (16 * lanes)) : Fin lanes :=
  ⟨d.val / 16, by have := d.isLt; omega⟩

private def xorLane {k : Nat} (a b : Fin (2 ^ k)) : Fin (2 ^ k) :=
  ⟨a.val ^^^ b.val, Nat.xor_lt_two_pow a.isLt b.isLt⟩

private def butterfly {k : Nat} (cfg : Config) :
    List (Fin (2 ^ k)) → (Fin (2 ^ k) → F32) → Fin (2 ^ k) → F32
  | [], input => input
  | delta :: rest, input =>
    butterfly cfg rest fun lane => FP.add cfg (input lane) (input (xorLane lane delta))

/-- Each reduction starts with channel 1, then contracts channel 0
and channels 2 through 15 into the accumulator. The lane count and
butterfly order come from the compiled head width. -/
private def laneDot {k : Nat} (cfg : Config) (steps : List (Fin (2 ^ k)))
    (l r : Fin (16 * 2 ^ k) → F32) (d : Fin (16 * 2 ^ k)) : F32 :=
  let laneSum := fun (lane : Fin (2 ^ k)) =>
    let a := channel lane 1
    let seed := FP.mul cfg (l a) (r a)
    ([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : List (Fin 16)).foldl
      (fun acc i => FP.fma cfg (l (channel lane i)) (r (channel lane i)) acc) seed
  butterfly cfg steps laneSum (laneOf d)

def inner (cfg : Config) (l r : ChannelRow) (d : Channel) : F32 :=
  laneDot (k := 3) cfg [4, 2, 1] l r d

private def epsilon : F32 := Value.ofBits .fp32 0x34000000

private def rmsVjp {D : Type} (a : Arithmetic) (width : F32)
    (dot : (D → F32) → (D → F32) → D → F32) (swapDot : Bool)
    (x incoming : D → F32) (d : D) : F32 :=
  let norm := fun j => a.rsqrtFtz (FP.add a.config (a.divFull (dot x x j) width) epsilon)
  let unit := fun j => FP.mul a.config (norm j) (x j)
  let correction := a.divFull (if swapDot then dot unit incoming d else dot incoming unit d) width
  FP.mul a.config (norm d) (FP.fma a.config (unit d).neg correction (incoming d))

/-- The RMS reduction is evaluated from the unmodified QK row. The
correction uses the selected gradient. Positive and negative zero and
non-finite values flow through the original operation order. -/
def qkDerivative {n : Length} (a : Arithmetic) (r : Inputs n) (p : Point n)
    (incoming : ChannelRow) : F32 :=
  let x := fun d => FP.cast .fp32 a.config (r.qk p.token p.band p.head d)
  rmsVjp a (Value.ofBits .fp32 0x43000000) (inner a.config) (decide (p.band = .key)) x incoming p.channel

/-- The FP8 instruction has a single shared target semantics. Its
input is the ordered FP32 chain below, with the BF16 round trip in QK. -/
structure Target extends Arithmetic where
  satE4M3 : F32 → UInt8

def packed {n : Length} (trim : Bool) (a : Target) (r : Inputs n) (p : Point n) : UInt8 :=
  let value :=
    if p.band = .value then current a.toArithmetic r p p.channel else
      let dx := qkDerivative a.toArithmetic r p (normalizedGradient trim a.toArithmetic r p)
      FP.cast .fp32 a.config (FP.cast .bf16 a.config dx)
  a.satE4M3 (a.divFull value r.scale)

def writes {n : Length} (trim : Bool) (a : Target) (r : Inputs n) (p : Point n) :
    (Nat × UInt8) × (Nat × UInt8) :=
  let byte := packed trim a r p
  ((rowAddress p, byte), (transposedAddress p, byte))

theorem both_layouts_equal {n : Length} (a : Target) (r : Inputs n) (p : Point n) :
    writes true a r p = writes false a r p := by
  simp only [writes, packed, normalized_gradient_equal]

namespace Narrow
open RopeShift (Plane)

inductive Shape where | paired | halfValue
  deriving DecidableEq

def Shape.valueWidth : Shape → Nat | .paired => 128 | .halfValue => 64
def Shape.width (s : Shape) : Band → Nat | .value => s.valueWidth | _ => 64
def Shape.features (s : Shape) : Nat := 768 + 6 * s.valueWidth
def Shape.qkPitch : Shape → Nat | .paired => 768 | .halfValue => 1152
def Shape.factorPitch : Shape → Nat | .paired => 128 | .halfValue => 64
def Shape.gradPitch : Shape → Nat | .paired => 192 | .halfValue => 384

structure Point (s : Shape) (n : Length) where
  token : Fin n.tokens
  band : Band
  head : Head
  channel : Fin (s.width band)

structure Inputs where
  qk : Nat → BF16
  gradQK : Plane → Nat → BF16
  gradV : Nat → BF16
  factor1 : Nat → BF16
  factor2 : Nat → BF16
  scale : F32

abbrev QkChannel := Fin 64

def qkAddress {n : Length} (s : Shape) (t : Fin n.tokens) (p : Plane) (h : Head)
    (d : QkChannel) : Nat :=
  t.val * s.qkPitch + (p.base + h.val) * 64 + d.val

/-- The original logical head selects the position and the head in the
packed gradient. The rotary table chooses its half using the head parity. -/
def gradAddress {n : Length} (s : Shape) (t : Fin n.tokens) (h : Head) (d : QkChannel) : Nat :=
  match s with
  | .paired => (2 * t.val + h.val / 3) * 192 + (h.val % 3) * 64 + d.val
  | .halfValue => t.val * 384 + h.val * 64 + d.val

def factorAddress {n : Length} (s : Shape) (t : Fin n.tokens) (h : Head) (d : QkChannel) : Nat :=
  t.val * s.factorPitch + (if s = .paired then h.val % 2 else 0) * 64 + d.val

def gradVAddress {n : Length} (s : Shape) (t : Fin n.tokens) (h : Head) (d : Fin s.valueWidth) : Nat :=
  (t.val * 6 + h.val) * s.valueWidth + d.val

/-- The paired PTX uses the high half of an unsigned multiply for the
division and isolates bit 6 of the shifted head for the table half. -/
theorem paired_head_lowering : ∀ h : Head,
    h.val * 1431655766 / 2 ^ 32 = h.val / 3 ∧
    h.val - (h.val * 1431655766 / 2 ^ 32) * 3 = h.val % 3 ∧
    (h.val <<< 6) &&& 64 = (h.val % 2) * 64 := by
  decide +kernel

def scalarPairIndex (d : QkChannel) : QkChannel :=
  ⟨d.val ^^^ 1, Nat.xor_lt_two_pow (n := 6) d.isLt (by decide)⟩

def vectorPairIndex (d : QkChannel) : QkChannel :=
  ⟨2 * (d.val / 2) + (1 - d.val % 2), by have := d.isLt; omega⟩

theorem paired_factor_address {n : Length} (s : Shape) (t : Fin n.tokens) (h : Head) (d : QkChannel) :
    factorAddress s t h (vectorPairIndex d) = factorAddress s t h (scalarPairIndex d) := by
  exact congrArg (factorAddress s t h) (Fin.ext (xor_one_reverses_pair d.val).symm)

theorem read_addresses_bounded {n : Length} (s : Shape) (t : Fin n.tokens) (p : Plane) (h : Head)
    (d : QkChannel) :
    qkAddress s t p h d < n.tokens * s.qkPitch ∧
    gradAddress s t h d < n.tokens * 384 ∧
    factorAddress s t h d < n.tokens * s.factorPitch := by
  have ht := t.isLt
  have hh := h.isLt
  have hd := d.isLt
  cases s <;> cases p <;>
    simp [qkAddress, gradAddress, factorAddress, Shape.qkPitch, Shape.factorPitch, Plane.base] <;> omega

theorem value_read_bounded {n : Length} (s : Shape) (t : Fin n.tokens) (h : Head)
    (d : Fin s.valueWidth) : gradVAddress s t h d < n.tokens * 6 * s.valueWidth := by
  have ht := t.isLt
  have hh := h.isLt
  have hd := d.isLt
  cases s <;> simp [gradVAddress, Shape.valueWidth] at * <;> omega

def feature {s : Shape} {n : Length} (p : Point s n) : Nat :=
  match p.band with
  | .query => p.head.val * 64 + p.channel.val
  | .key => (6 + p.head.val) * 64 + p.channel.val
  | .value => 768 + p.head.val * s.valueWidth + p.channel.val

def rowAddress {s : Shape} {n : Length} (p : Point s n) : Nat :=
  p.token.val * s.features + feature p

def transposedAddress {s : Shape} {n : Length} (p : Point s n) : Nat :=
  feature p * n.tokens + p.token.val

theorem feature_bound {s : Shape} {n : Length} (p : Point s n) : feature p < s.features := by
  rcases p with ⟨t, band, h, d⟩
  have hh := h.isLt
  have hd := d.isLt
  cases s <;> cases band <;> simp [feature, Shape.features, Shape.width, Shape.valueWidth] at * <;> omega

theorem addresses_bounded {s : Shape} {n : Length} (p : Point s n) :
    rowAddress p < n.tokens * s.features ∧ transposedAddress p < n.tokens * s.features ∧
    n.tokens * s.features < 2 ^ 31 := by
  have ht := p.token.isLt
  have hf := feature_bound p
  cases s <;> cases n <;>
    simp [rowAddress, transposedAddress, Shape.features, Shape.valueWidth, Length.tokens] at * <;> omega

private theorem coordinates_unique {s : Shape} {n : Length} (p q : Point s n)
    (ht : p.token = q.token) (hf : feature p = feature q) : p = q := by
  rcases p with ⟨t, pBand, h, d⟩
  rcases q with ⟨u, qBand, j, e⟩
  change t = u at ht
  subst u
  have hh := h.isLt
  have hj := j.isLt
  have hd := d.isLt
  have he := e.isLt
  cases s <;> cases pBand <;> cases qBand <;>
    simp [feature, Shape.width, Shape.valueWidth] at hf hd he <;>
    first
    | omega
    | have hk : h = j := Fin.ext (by omega)
      have hc : d = e := Fin.ext (by omega)
      rw [hk, hc]

theorem row_address_unique {s : Shape} {n : Length} (p q : Point s n) (h : rowAddress p = rowAddress q) :
    p = q := by
  have hp := feature_bound p
  have hq := feature_bound q
  cases s <;> simp [rowAddress, Shape.features, Shape.valueWidth] at h hp hq <;>
    exact coordinates_unique p q (Fin.ext (by omega)) (by omega)

theorem transpose_address_unique {s : Shape} {n : Length} (p q : Point s n)
    (h : transposedAddress p = transposedAddress q) : p = q := by
  have hp := p.token.isLt
  have hq := q.token.isLt
  cases n <;> simp [transposedAddress, Length.tokens] at h hp hq <;>
    exact coordinates_unique p q (Fin.ext (by omega)) (by omega)

private def qkDerivative {n : Length} (vector : Bool) (a : Arithmetic) (r : Inputs)
    (s : Shape) (t : Fin n.tokens) (p : Plane) (h : Head) (d : QkChannel) : F32 :=
  let widen := FP.cast .fp32 a.config
  let x := fun j => widen (r.qk (qkAddress s t p h j))
  let grad := fun j => widen (r.gradQK p (gradAddress s t h j))
  let rotate := fun j =>
    let f1 := widen (r.factor1 (factorAddress s t h j))
    let pair := if vector then vectorPairIndex j else scalarPairIndex j
    let f2 := widen (r.factor2 (factorAddress s t h pair))
    FP.fma a.config f1 (grad j) (FP.mul a.config f2 (grad (scalarPairIndex j)))
  rmsVjp a (Value.ofBits .fp32 0x42800000) (laneDot (k := 2) a.config [2, 1]) false x rotate d

private theorem derivatives_equal {n : Length} (a : Arithmetic) (r : Inputs)
    (s : Shape) (t : Fin n.tokens) (p : Plane) (h : Head) (d : QkChannel) :
    qkDerivative true a r s t p h d = qkDerivative false a r s t p h d := by
  simp [qkDerivative, paired_factor_address]

def packed {s : Shape} {n : Length} (vector : Bool) (a : Target) (r : Inputs) (p : Point s n) : UInt8 :=
  let rounded := fun p' d =>
    FP.cast .fp32 a.config (FP.cast .bf16 a.config (qkDerivative vector a.toArithmetic r s p.token p' p.head d))
  let value := match p with
    | ⟨_, .query, _, d⟩ => rounded .query d
    | ⟨_, .key, _, d⟩ => rounded .key d
    | ⟨t, .value, h, d⟩ =>
      FP.cast .fp32 a.config (r.gradV (gradVAddress s t h d))
  a.satE4M3 (a.divFull value r.scale)

def writes {s : Shape} {n : Length} (vector : Bool) (a : Target) (r : Inputs) (p : Point s n) :
    (Nat × UInt8) × (Nat × UInt8) :=
  let b := packed vector a r p
  ((rowAddress p, b), (transposedAddress p, b))

theorem both_layouts_equal {s : Shape} {n : Length} (a : Target) (r : Inputs) (p : Point s n) :
    writes true a r p = writes false a r p := by
  rcases p with ⟨t, band, h, d⟩
  cases band <;> simp [writes, packed, derivatives_equal]

#axiomsClean paired_factor_address
#axiomsClean paired_head_lowering
#axiomsClean read_addresses_bounded
#axiomsClean value_read_bounded
#axiomsClean addresses_bounded
#axiomsClean row_address_unique
#axiomsClean transpose_address_unique
#axiomsClean both_layouts_equal
end Narrow

#axiomsClean normalized_gradient_equal
#axiomsClean addresses_bounded
#axiomsClean token_tile_covers
#axiomsClean interior_key_reads
#axiomsClean row_address_unique
#axiomsClean transpose_address_unique
#axiomsClean both_layouts_equal
#axiomsClean rotating_partner
#axiomsClean reversed_pair_index
#axiomsClean xor_one_reverses_pair
#axiomsClean reversed_pair_bits
#axiomsClean trimmed_loads_equal
end VeriTile.Bench.Models.NanoGPT.Track1.RopeBackward
