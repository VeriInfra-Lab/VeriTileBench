import models.NanoGPT.Track1.NgramTiles

/-! N-gram lane addressing and numerical operations for the compiled
flat and full-row SM90a launches. BF16 loads and stores index halfwords. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramEmbedPTX
open Ngram VeriTile Triton.FP

def flatToken (offset : Nat) : Nat :=
  (offset * 715827883 / 4294967296) / 128

theorem flat_token_exact (offset : Nat) (live : offset < 262144 * 768) :
    flatToken offset = offset / D := by
  unfold flatToken
  rw [Nat.div_div_eq_div_mul]
  change (offset * 715827883) / 549755813888 = offset / 768
  have decomposition := Nat.div_add_mod' offset 768
  have remainder := Nat.mod_lt offset (by decide : 0 < 768)
  apply Nat.div_eq_of_lt_le <;> omega

def fullToken (block thread : Nat) : Nat :=
  (block <<< 2) ||| ((thread >>> 5) &&& 3)

def laneChannel (thread : Nat) : Nat := (thread <<< 3) &&& 248

theorem full_token_exact {tokens : Nat} (lane : FullLane tokens) :
    fullToken lane.block lane.thread = lane.owner.token := by
  have thread := lane.thread.isLt
  have small : lane.thread.val / 32 < 2 ^ 2 := by omega
  simp only [fullToken, Nat.shiftRight_eq_div_pow, show (2 : Nat) ^ 5 = 32 by rfl]
  rw [show (3 : Nat) = 2 ^ 2 - 1 by rfl, Nat.and_two_pow_sub_one_of_lt_two_pow small,
    ← Nat.shiftLeft_add_eq_or_of_lt small, Nat.shiftLeft_eq]
  rfl

theorem lane_channel_exact (thread : Nat) : laneChannel thread = thread % 32 * 8 := by
  unfold laneChannel
  rw [show (248 : Nat) = (31 : Nat) <<< 3 by rfl, ← Nat.shiftLeft_and_distrib,
    show (31 : Nat) = 2 ^ 5 - 1 by rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.shiftLeft_eq]

def fullOffset {tokens : Nat} (lane : FullLane tokens) : Nat :=
  let token := fullToken lane.block lane.thread
  ((token * D) ||| laneChannel lane.thread) + lane.tile.val * ChannelTile + lane.piece.val

theorem full_offset_exact {tokens : Nat} (lane : FullLane tokens) :
    fullOffset lane = index lane.owner.token lane.owner.channel := by
  unfold fullOffset
  rw [full_token_exact, lane_channel_exact]
  let token := lane.owner.token
  have low : lane.thread.val % 32 * 8 < 2 ^ 8 := by omega
  have shift : token * D = (token * 3) <<< 8 := by
    simp only [Nat.shiftLeft_eq, D]
    omega
  change (token * D ||| lane.thread.val % 32 * 8) + lane.tile.val * ChannelTile + lane.piece.val = _
  rw [shift, ← Nat.shiftLeft_add_eq_or_of_lt low, ← shift]
  change token * 768 + lane.thread.val % 32 * 8 + lane.tile.val * 256 + lane.piece.val =
    token * 768 + (lane.tile.val * 256 + (lane.thread.val % 32 * 8 + lane.piece.val))
  ac_rfl

/-- Each supported output halfword has exactly one full-row writer. -/
theorem full_offset_owner {tokens : Nat} (aligned : tokens % 16 = 0)
    (offset : Fin (tokens * D)) :
    ∃ lane : FullLane tokens, fullOffset lane = offset.val ∧
      ∀ other : FullLane tokens, fullOffset other = offset.val → other = lane := by
  let t : Fin tokens := ⟨offset.val / D, (Nat.div_lt_iff_lt_mul (by decide)).mpr offset.isLt⟩
  let c : Fin D := ⟨offset.val % D, Nat.mod_lt _ (by decide)⟩
  let lane := FullLane.ofOwner (tileOwner t c)
  have decomposition : index t c = offset.val := Nat.div_add_mod' offset.val D
  have owner : lane.owner = tileOwner t c := full_lane_owner _
  obtain ⟨row, col⟩ := owner_reads t c
  have writes : fullOffset lane = offset.val := by
    rw [full_offset_exact, owner, row, col, decomposition]
  refine ⟨lane, writes, ?_⟩
  intro other equals
  have same := equals.trans writes.symm
  simp only [full_offset_exact] at same
  obtain ⟨_, otherChannel⟩ := full_lane_live other aligned
  obtain ⟨_, laneChannel⟩ := full_lane_live lane aligned
  apply full_lane_unique
  · simpa only [index_token _ _ otherChannel, index_token _ _ laneChannel] using congrArg (· / D) same
  · simpa only [index_channel _ _ otherChannel, index_channel _ _ laneChannel] using congrArg (· % D) same

theorem full_gradient_planes_disjoint {tokens : Nat} (aligned : tokens % 16 = 0)
    (a b : FullLane tokens) : fullOffset a < tokens * D + fullOffset b := by
  obtain ⟨ta, ca⟩ := full_lane_live a aligned
  obtain ⟨tb, cb⟩ := full_lane_live b aligned
  simpa only [full_offset_exact] using
    backward_planes_disjoint (⟨_, ta⟩ : Fin tokens) (⟨_, tb⟩ : Fin tokens) (⟨_, ca⟩ : Fin D) (⟨_, cb⟩ : Fin D)

/-- The cache uses signed 32-bit slots and a wide byte-address multiply.
The slot bound ensures the lowering uses the intended nonnegative row. -/
theorem signed_slot_address (slot channel : Nat) (signed : slot < 2147483648) :
    (Int32.ofNat slot).toInt * 1536 + (channel : Int) * 2 = (2 * index slot channel : Nat) := by
  rw [Int32.toInt_ofNat_of_lt signed]
  change (slot : Int) * 1536 + (channel : Int) * 2 = ((2 * (slot * 768 + channel) : Nat) : Int)
  omega

abbrev Write := Nat × Value .bf16

def referenceForward (s : Inputs) (tokens offset : Nat) : Write :=
  (offset, lookup s tokens (flatToken offset) (offset - flatToken offset * D) .leftFma)

def referenceBackward (s : Inputs) (tokens offset : Nat) : Write × Write :=
  let (bigram, trigram) := pullback s offset (flatToken offset) (offset - flatToken offset * D) .packed
  ((offset, bigram), (tokens * D + offset, trigram))

def emittedForward {tokens : Nat} (s : Inputs) (lane : FullLane tokens) : Write :=
  let token := fullToken lane.block lane.thread
  let c := lane.tile.val * ChannelTile + laneChannel lane.thread + lane.piece.val
  let (a, b) := emittedSigns s.tokens token
  (fullOffset lane, embedValue .leftFma
    (s.cache (index (s.slots token) c))
    (s.cache (index (s.slots (tokens + token)) c))
    (s.pool (index a c))
    (s.pool (index b c)))

def emittedBackward {tokens : Nat} (s : Inputs) (lane : FullLane tokens) : Write × Write :=
  let token := fullToken lane.block lane.thread
  let c := lane.tile.val * ChannelTile + laneChannel lane.thread + lane.piece.val
  let (a, b) := emittedSigns s.tokens token
  let offset := fullOffset lane
  let g := s.grad offset
  ((offset, gradientValue .packed g (s.pool (index a c))),
   (tokens * D + offset, gradientValue .packed g (s.pool (index b c))))

theorem emitted_forward {tokens : Nat} (s : Inputs) (lane : FullLane tokens) (mask : s.signMask = 8191) :
    emittedForward s lane = (index lane.owner.token lane.owner.channel, tiledForward s lane.owner .leftFma) := by
  simp only [emittedForward, full_token_exact, lane_channel_exact, emitted_signs_exact, full_offset_exact,
    tiledForward, lookup, mask]
  simp only [FullLane.owner, Owner.channel, Nat.add_assoc]

theorem emitted_backward {tokens : Nat} (s : Inputs) (lane : FullLane tokens) (mask : s.signMask = 8191) :
    let offset := index lane.owner.token lane.owner.channel
    let (a, b) := tiledBackward s lane.owner .packed
    emittedBackward s lane = ((offset, a), (tokens * D + offset, b)) := by
  simp only [emittedBackward, full_token_exact, lane_channel_exact, emitted_signs_exact, full_offset_exact,
    tiledBackward, pullback, mask]
  simp only [FullLane.owner, Owner.channel, Nat.add_assoc]

theorem stores_equal {tokens : Nat} (s : Inputs) (bounded : tokens ≤ 262144)
    (mask : s.signMask = 8191) (t : Fin tokens) (c : Fin D) :
    let offset := index t.val c.val
    let lane := FullLane.ofOwner (tileOwner t c)
    emittedForward s lane = referenceForward s tokens offset ∧
    emittedBackward s lane = referenceBackward s tokens offset := by
  intro offset lane
  have valid : offset < tokens * D := (address_bounds tokens t c t.isLt c.isLt bounded).1
  have fits : offset < 262144 * 768 := Nat.lt_of_lt_of_le valid (Nat.mul_le_mul_right D bounded)
  have row : flatToken offset = t.val :=
    (flat_token_exact offset fits).trans (index_token t c c.isLt)
  have col : offset - t.val * D = c.val := by
    change t.val * D + c.val - t.val * D = c.val
    exact Nat.add_sub_cancel_left _ _
  obtain ⟨ht, hc⟩ := owner_reads t c
  have owned : lane.owner = tileOwner t c := full_lane_owner _
  constructor
  · simp only [emitted_forward s lane mask, owned, referenceForward, row, col,
      tiledForward, ht, hc]
    rfl
  · simp only [emitted_backward s lane mask, owned, referenceBackward, row, col,
      tiledBackward, ht, hc]
    rfl

inductive InputBuffer
  | token | slot | cache | pool | gradient

def InputBuffer.words (tokens cacheRows : Nat) : InputBuffer → Nat
  | .token => tokens
  | .slot => 2 * tokens
  | .cache => cacheRows * D
  | .pool => 8192 * D
  | .gradient => tokens * D

def reads {tokens : Nat} (forward : Bool) (s : Inputs) (lane : FullLane tokens) :
    List (InputBuffer × Nat) :=
  let t := fullToken lane.block lane.thread
  let c := lane.tile.val * ChannelTile + laneChannel lane.thread + lane.piece.val
  let (b, g) := emittedSigns s.tokens t
  [some (.token, t),
   if 0 < t then some (.token, t - 1) else none,
   if 1 < t then some (.token, t - 2) else none,
   if forward then some (.slot, t) else none,
   if forward then some (.slot, tokens + t) else none,
   if forward then some (.cache, index (s.slots t) c) else none,
   if forward then some (.cache, index (s.slots (tokens + t)) c) else none,
   if forward then none else some (.gradient, fullOffset lane),
   some (.pool, index b c), some (.pool, index g c)].filterMap id

private theorem element_within (rows row channel : Nat) (hr : row < rows) (hc : channel < D) :
    index row channel < rows * D := by
  change channel < 768 at hc
  change row * 768 + channel < rows * 768
  omega

theorem reads_within {tokens cacheRows : Nat} (forward : Bool) (s : Inputs) (lane : FullLane tokens)
    (aligned : tokens % 16 = 0) (slots : ∀ i < 2 * tokens, s.slots i < cacheRows)
    (buffer : InputBuffer) (i : Nat) (issued : (buffer, i) ∈ reads forward s lane) :
    i < buffer.words tokens cacheRows := by
  obtain ⟨ht, hc⟩ := full_lane_live lane aligned
  let t := lane.owner.token
  let c := lane.owner.channel
  have signs := sign_rows_bound s.tokens t
  have readChannel : lane.tile.val * ChannelTile + laneChannel lane.thread + lane.piece.val = c := by
    rw [lane_channel_exact]
    exact Nat.add_assoc _ _ _
  have tokenFields : t < 2 * tokens ∧ tokens + t < 2 * tokens := by
    change t < tokens at ht
    omega
  have prev : t - 1 < tokens := by change t < tokens at ht; omega
  have prior : t - 2 < tokens := by change t < tokens at ht; omega
  simp only [reads, full_token_exact, readChannel, emitted_signs_exact, full_offset_exact,
    List.mem_filterMap, List.mem_cons, List.not_mem_nil, or_false, id_eq] at issued
  obtain ⟨entry, choices, chosen⟩ := issued
  rcases choices with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    (try (split at chosen <;> try contradiction)) <;>
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj chosen)
  · exact ht
  · exact prev
  · exact prior
  · exact tokenFields.1
  · exact tokenFields.2
  · exact element_within _ _ _ (slots _ tokenFields.1) hc
  · exact element_within _ _ _ (slots _ tokenFields.2) hc
  · exact element_within _ _ _ ht hc
  · exact element_within _ _ _ signs.1 hc
  · exact element_within _ _ _ signs.2 hc

#axiomsClean flat_token_exact
#axiomsClean full_offset_exact
#axiomsClean full_offset_owner
#axiomsClean full_gradient_planes_disjoint
#axiomsClean signed_slot_address
#axiomsClean stores_equal
#axiomsClean reads_within

end VeriTile.Bench.Models.NanoGPT.Track1.NgramEmbedPTX
