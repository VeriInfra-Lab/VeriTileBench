import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Models.NanoGPT.Track1.Ngram
open VeriTile Triton.FP

abbrev D := 768
abbrev TokenTile := 4
abbrev ChannelTile := 256

def index (token channel : Nat) : Nat := token * D + channel

theorem index_token (token channel : Nat) (h : channel < D) :
    index token channel / D = token := by
  rw [index, Nat.mul_comm token D, Nat.mul_add_div (by decide), Nat.div_eq_of_lt h, Nat.add_zero]

theorem index_channel (token channel : Nat) (h : channel < D) :
    index token channel % D = channel :=
  Nat.mul_add_mod_of_lt h

/-- Owner in the Triton launch grid and the two offsets inside its block. -/
structure Owner (tokens : Nat) where
  blockToken : Fin ((tokens + TokenTile - 1) / TokenTile)
  blockChannel : Fin (D / ChannelTile)
  laneToken : Fin TokenTile
  laneChannel : Fin ChannelTile

def Owner.token {tokens : Nat} (i : Owner tokens) : Nat := i.blockToken * TokenTile + i.laneToken
def Owner.channel {tokens : Nat} (i : Owner tokens) : Nat := i.blockChannel * ChannelTile + i.laneChannel

def tileOwner {tokens : Nat} (t : Fin tokens) (c : Fin D) : Owner tokens where
  blockToken := ⟨t.val / TokenTile, by
    have ht := t.isLt
    change t.val / 4 < (tokens + 4 - 1) / 4
    omega⟩
  blockChannel := ⟨c.val / ChannelTile, by
    have hc := c.isLt
    change c.val < 768 at hc
    change c.val / 256 < 3
    omega⟩
  laneToken := ⟨t.val % TokenTile, Nat.mod_lt _ (by decide)⟩
  laneChannel := ⟨c.val % ChannelTile, Nat.mod_lt _ (by decide)⟩

theorem owner_reads {tokens : Nat} (t : Fin tokens) (c : Fin D) :
    (tileOwner t c).token = t.val ∧ (tileOwner t c).channel = c.val :=
  ⟨Nat.div_add_mod' t.val TokenTile, Nat.div_add_mod' c.val ChannelTile⟩

theorem unique_owner {tokens : Nat} (i j : Owner tokens)
    (ht : i.token = j.token) (hc : i.channel = j.channel) : i = j := by
  cases i with
  | mk ib ic il ix =>
    cases j with
    | mk jb jc jl jx =>
      have hil := il.isLt
      have hjl := jl.isLt
      have hix := ix.isLt
      have hjx := jx.isLt
      change il.val < 4 at hil
      change jl.val < 4 at hjl
      change ix.val < 256 at hix
      change jx.val < 256 at hjx
      change ib.val * 4 + il.val = jb.val * 4 + jl.val at ht
      change ic.val * 256 + ix.val = jc.val * 256 + jx.val at hc
      have hb : ib = jb := Fin.ext (by omega)
      have hd : ic = jc := Fin.ext (by omega)
      have he : il = jl := Fin.ext (by omega)
      have hf : ix = jx := Fin.ext (by omega)
      cases hb
      cases hd
      cases he
      cases hf
      rfl

/-- Addresses into the [T,D] and [2T,D] stores fit int32 at the largest
sequence packing used by the Track 1 training and validation batches. -/
theorem address_bounds (tokens token channel : Nat) (ht : token < tokens)
    (hc : channel < D) (hb : tokens ≤ 262144) :
    index token channel < tokens * D ∧ tokens * D + index token channel < 2^31 := by
  change channel < 768 at hc
  change token * 768 + channel < tokens * 768 ∧ tokens * 768 + (token * 768 + channel) < 2147483648
  omega

theorem store_offsets_injective {tokens : Nat} (t u : Fin tokens) (c d : Fin D)
    (h : index t.val c.val = index u.val d.val) : t = u ∧ c = d := by
  constructor
  · apply Fin.ext
    simpa only [index_token _ _ c.isLt, index_token _ _ d.isLt] using congrArg (· / D) h
  · apply Fin.ext
    simpa only [index_channel _ _ c.isLt, index_channel _ _ d.isLt] using congrArg (· % D) h

/-- The backward's two plane stores cannot alias, including between programs. -/
theorem backward_planes_disjoint {tokens : Nat} (t u : Fin tokens) (c d : Fin D) :
    index t.val c.val < tokens * D + index u.val d.val := by
  have ht := t.isLt
  have hc := c.isLt
  change c.val < 768 at hc
  change t.val * 768 + c.val < tokens * 768 + (u.val * 768 + d.val)
  omega

/-- Int32 token words read by the shared sign-pool hash. Multiplication
and xor use the low 32 bits, including on negative input words. -/
def signRows (read : Nat → UInt32) (mask : UInt32) (t : Nat) : Nat × Nat :=
  let cur := read t
  let prev := read (t - 1)
  let prior := read (t - 2)
  (if t < 1 then 0 else (((30011 * prev) ^^^ (48271 * cur)) &&& mask).toNat,
   if t < 2 then 0 else (((26801 * prior) ^^^ (39779 * prev) ^^^ (58699 * cur)) &&& mask).toNat)

private theorem mul_low (k word : UInt32) :
    (k * word).toNat % 2 ^ 13 = (k.toNat % 2 ^ 13 * word.toNat) % 2 ^ 13 := by
  rw [UInt32.toNat_mul, Nat.mod_mod_of_dvd _ (by decide : 2 ^ 13 ∣ 2 ^ 32), Nat.mod_mul_mod]

theorem reduced_bigram (cur prev : UInt32) :
    (((5435 * prev) ^^^ (7311 * cur)) &&& 8191) =
      (((30011 * prev) ^^^ (48271 * cur)) &&& 8191) := by
  apply UInt32.toNat.inj
  simp only [UInt32.toNat_and, show (8191 : UInt32).toNat = 2 ^ 13 - 1 by rfl,
    Nat.and_two_pow_sub_one_eq_mod, UInt32.toNat_xor, Nat.xor_mod_two_pow, mul_low]
  rfl

theorem reduced_trigram (cur prev prior : UInt32) :
    (((7011 * prev) ^^^ (1355 * cur) ^^^ (2225 * prior)) &&& 8191) =
      (((26801 * prior) ^^^ (39779 * prev) ^^^ (58699 * cur)) &&& 8191) := by
  apply UInt32.toNat.inj
  simp only [UInt32.toNat_and, show (8191 : UInt32).toNat = 2 ^ 13 - 1 by rfl,
    Nat.and_two_pow_sub_one_eq_mod, UInt32.toNat_xor, Nat.xor_mod_two_pow, mul_low]
  rw [Nat.xor_comm (_ ^^^ _) _, ← Nat.xor_assoc]
  rfl

theorem sign_rows_bound (read : Nat → UInt32) (token : Nat) :
    (signRows read 8191 token).1 < 8192 ∧ (signRows read 8191 token).2 < 8192 := by
  have low (word : UInt32) : (word &&& 8191).toNat < 8192 := by
    rw [UInt32.toNat_and]
    simpa only [show (8191 : UInt32).toNat = 2 ^ 13 - 1 by rfl, Nat.and_two_pow_sub_one_eq_mod] using
      Nat.mod_lt word.toNat (by decide : 0 < 2 ^ 13)
  constructor <;> simp only [signRows] <;> split <;> first | decide | apply low

def emittedSigns (read : Nat → UInt32) (t : Nat) : Nat × Nat :=
  let cur := read t
  let prev := if 0 < t then read (t - 1) else 0
  let prior := if 1 < t then read (t - 2) else 0
  (if 0 < t then (((5435 * prev) ^^^ (7311 * cur)) &&& 8191).toNat else 0,
   if 1 < t then (((7011 * prev) ^^^ (1355 * cur) ^^^ (2225 * prior)) &&& 8191).toNat else 0)

theorem emitted_signs_exact (read : Nat → UInt32) (t : Nat) :
    emittedSigns read t = signRows read 8191 t := by
  unfold emittedSigns signRows
  by_cases prev : 0 < t <;> by_cases prior : 1 < t <;>
    simp only [prev, prior, if_true, if_false,
      show (t < 1) ↔ ¬ 0 < t by omega, show (t < 2) ↔ ¬ 1 < t by omega,
      not_true_eq_false, not_false_eq_true, reduced_bigram, reduced_trigram] <;> omega

structure FullLane (tokens : Nat) where
  block : Fin ((tokens + TokenTile - 1) / TokenTile)
  thread : Fin 128
  tile : Fin (D / ChannelTile)
  piece : Fin 8

def FullLane.owner {tokens : Nat} (lane : FullLane tokens) : Owner tokens where
  blockToken := lane.block
  blockChannel := lane.tile
  laneToken := ⟨lane.thread.val / 32, by
    have h := lane.thread.isLt
    change lane.thread.val / 32 < 4
    omega⟩
  laneChannel := ⟨lane.thread.val % 32 * 8 + lane.piece.val, by
    have h := lane.piece.isLt
    change lane.thread.val % 32 * 8 + lane.piece.val < 256
    omega⟩

def FullLane.ofOwner {tokens : Nat} (owner : Owner tokens) : FullLane tokens where
  block := owner.blockToken
  thread := ⟨owner.laneToken.val * 32 + owner.laneChannel.val / 8, by
    have t := owner.laneToken.isLt
    have c := owner.laneChannel.isLt
    change owner.laneToken.val < 4 at t
    change owner.laneChannel.val < 256 at c
    omega⟩
  tile := owner.blockChannel
  piece := ⟨owner.laneChannel.val % 8, by omega⟩

theorem full_lane_owner {tokens : Nat} (owner : Owner tokens) : (FullLane.ofOwner owner).owner = owner := by
  have channel := owner.laneChannel.isLt
  change owner.laneChannel.val < 256 at channel
  have row : (FullLane.ofOwner owner).owner.laneToken = owner.laneToken := Fin.ext (by
    change (owner.laneToken.val * 32 + owner.laneChannel.val / 8) / 32 = owner.laneToken.val
    omega)
  have col : (FullLane.ofOwner owner).owner.laneChannel = owner.laneChannel := Fin.ext (by
    change (owner.laneToken.val * 32 + owner.laneChannel.val / 8) % 32 * 8 +
      owner.laneChannel.val % 8 = owner.laneChannel.val
    omega)
  exact unique_owner _ _
    (congrArg (fun i : Fin TokenTile => owner.blockToken.val * TokenTile + i.val) row)
    (congrArg (fun i : Fin ChannelTile => owner.blockChannel.val * ChannelTile + i.val) col)

theorem full_lane_live {tokens : Nat} (lane : FullLane tokens) (aligned : tokens % 16 = 0) :
    lane.owner.token < tokens ∧ lane.owner.channel < D := by
  have block := lane.block.isLt
  have thread := lane.thread.isLt
  have tile := lane.tile.isLt
  have piece := lane.piece.isLt
  change lane.block.val < (tokens + 4 - 1) / 4 at block
  change lane.tile.val < 3 at tile
  change lane.block.val * 4 + lane.thread.val / 32 < tokens ∧
    lane.tile.val * 256 + (lane.thread.val % 32 * 8 + lane.piece.val) < 768
  omega

private theorem full_owner_lane {tokens : Nat} (lane : FullLane tokens) : FullLane.ofOwner lane.owner = lane := by
  cases lane with
  | mk block thread tile piece =>
    have last := piece.isLt
    have hthread : (FullLane.ofOwner (FullLane.owner ⟨block, thread, tile, piece⟩)).thread = thread :=
      Fin.ext (by change thread.val / 32 * 32 + (thread.val % 32 * 8 + piece.val) / 8 = _; omega)
    have hpiece : (FullLane.ofOwner (FullLane.owner ⟨block, thread, tile, piece⟩)).piece = piece :=
      Fin.ext (by change (thread.val % 32 * 8 + piece.val) % 8 = _; omega)
    exact (congrArg (fun v => FullLane.mk block v tile
      (FullLane.ofOwner (FullLane.owner ⟨block, thread, tile, piece⟩)).piece) hthread).trans
      (congrArg (FullLane.mk block thread tile) hpiece)

theorem full_lane_unique {tokens : Nat} (a b : FullLane tokens)
    (row : a.owner.token = b.owner.token) (col : a.owner.channel = b.owner.channel) : a = b := by
  have same := congrArg FullLane.ofOwner (unique_owner _ _ row col)
  simpa only [full_owner_lane] using same

structure Inputs where
  cache : Nat → Value .bf16
  slots : Nat → Nat
  pool : Nat → Value .bf16
  tokens : Nat → UInt32
  signMask : UInt32
  grad : Nat → Value .bf16

def promote (x : Value .bf16) : Value .fp32 := cast .fp32 Config.ieeeValue x
def narrow (x : Value .fp32) : Value .bf16 := cast .bf16 Config.ieeeValue x

/-- The expression has the same two multiply operands and sum on each lane.
This includes separate operations and either legal FMA contraction. -/
inductive Contraction where
  | separate | leftFma | rightFma

inductive BackwardArithmetic where
  | widened | packed

def gradientValue (arithmetic : BackwardArithmetic) (g s : Value .bf16) : Value .bf16 :=
  match arithmetic with
  | .widened => narrow (mul Config.ieeeValue (promote g) (promote s))
  | .packed => mul Config.ieeeValue g s

def embedValue (contract : Contraction) (a b sa sb : Value .bf16) : Value .bf16 :=
  let c := Config.ieeeValue
  let x := promote a
  let y := promote b
  let sx := promote sa
  let sy := promote sb
  narrow (match contract with
  | .separate => add c (mul c x sx) (mul c y sy)
  | .leftFma => fma c x sx (mul c y sy)
  | .rightFma => fma c y sy (mul c x sx))

def lookup (s : Inputs) (tokens token channel : Nat) (contract : Contraction) : Value .bf16 :=
  let (bigram, trigram) := signRows s.tokens s.signMask token
  embedValue contract
    (s.cache (index (s.slots token) channel))
    (s.cache (index (s.slots (tokens + token)) channel))
    (s.pool (index bigram channel))
    (s.pool (index trigram channel))

def pullback (s : Inputs) (offset token channel : Nat) (arithmetic : BackwardArithmetic) :
    Value .bf16 × Value .bf16 :=
  let (bigram, trigram) := signRows s.tokens s.signMask token
  let g := s.grad offset
  (gradientValue arithmetic g (s.pool (index bigram channel)),
   gradientValue arithmetic g (s.pool (index trigram channel)))

def flatForward (s : Inputs) (tokens offset : Nat) (contract : Contraction) :=
  lookup s tokens (offset / D) (offset % D) contract
def tiledForward {tokens : Nat} (s : Inputs) (i : Owner tokens) (contract : Contraction) :=
  lookup s tokens i.token i.channel contract

def flatBackward (s : Inputs) (offset : Nat) (arithmetic : BackwardArithmetic) :=
  pullback s offset (offset / D) (offset % D) arithmetic
def tiledBackward {tokens : Nat} (s : Inputs) (i : Owner tokens) (arithmetic : BackwardArithmetic) :=
  pullback s (index i.token i.channel) i.token i.channel arithmetic

def fullForward {tokens : Nat} (s : Inputs) (i : FullLane tokens) (contract : Contraction) :=
  tiledForward s i.owner contract

def fullBackward {tokens : Nat} (s : Inputs) (i : FullLane tokens) (arithmetic : BackwardArithmetic) :=
  tiledBackward s i.owner arithmetic

theorem forward_equal {tokens : Nat} (s : Inputs) (t : Fin tokens) (c : Fin D)
    (contract : Contraction) :
    tiledForward s (tileOwner t c) contract = flatForward s tokens (index t.val c.val) contract := by
  obtain ⟨ht,hc⟩ := owner_reads t c
  simp only [tiledForward, flatForward, ht, hc, index_token _ _ c.isLt, index_channel _ _ c.isLt]

theorem backward_equal {tokens : Nat} (s : Inputs) (t : Fin tokens) (c : Fin D)
    (arithmetic : BackwardArithmetic) :
    tiledBackward s (tileOwner t c) arithmetic = flatBackward s (index t.val c.val) arithmetic := by
  obtain ⟨ht,hc⟩ := owner_reads t c
  simp only [tiledBackward, flatBackward, ht, hc, index_token _ _ c.isLt, index_channel _ _ c.isLt]

theorem full_forward_equal {tokens : Nat} (s : Inputs) (t : Fin tokens) (c : Fin D) (contract : Contraction) :
    fullForward s (FullLane.ofOwner (tileOwner t c)) contract =
      flatForward s tokens (index t.val c.val) contract := by
  rw [fullForward, full_lane_owner, forward_equal]

theorem full_backward_equal {tokens : Nat} (s : Inputs) (t : Fin tokens) (c : Fin D)
    (arithmetic : BackwardArithmetic) :
    fullBackward s (FullLane.ofOwner (tileOwner t c)) arithmetic =
      flatBackward s (index t.val c.val) arithmetic := by
  rw [fullBackward, full_lane_owner, backward_equal]

#axiomsClean owner_reads
#axiomsClean unique_owner
#axiomsClean address_bounds
#axiomsClean store_offsets_injective
#axiomsClean backward_planes_disjoint
#axiomsClean forward_equal
#axiomsClean backward_equal
#axiomsClean reduced_bigram
#axiomsClean reduced_trigram
#axiomsClean sign_rows_bound
#axiomsClean emitted_signs_exact
#axiomsClean full_lane_live
#axiomsClean full_lane_unique
#axiomsClean full_forward_equal
#axiomsClean full_backward_equal
end VeriTile.Bench.Models.NanoGPT.Track1.Ngram
