import models.NanoGPT.Track1.TailLerp
import models.NanoGPT.Track1.TailNorm

/-! The local shards are written in the upstream blend order. The calls at the
collective boundary deliver remote rank shards, which this program
concatenates before measuring the post-ship FP32 norms. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TailShip
open VeriTile.Triton FP Driver
open TailLerp (Shard)
open TailNorm (BF16 F32 Bank)

abbrev Weights := Shard → Bank

inductive Average where | first | second

structure Snapshot where
  whole : Weights
  shard : Weights
  average : Shard → Average → Nat → F32
  seeded : Shard → Average → Bool
  normMath : TailNorm.Math
  lerpContract : Bool

structure Action where
  label : Shard
  average : Average
  rate : Option F32

def actions : List Action :=
  let ema := some (Value.ofBits .fp32 tailShipEMABits)
  let bank := some (Value.ofBits .fp32 tailShipBankBits)
  let afterAverage := some (Value.ofBits .fp32 tailShipAfterAverageBits)
  [⟨.lmHead, .first, ema⟩, ⟨.embedding, .first, ema⟩,
   ⟨.vo, .first, none⟩, ⟨.mlp, .first, none⟩, ⟨.valueEmbedding, .first, none⟩,
   ⟨.qk, .second, bank⟩, ⟨.vo, .second, afterAverage⟩, ⟨.mlp, .second, afterAverage⟩]

def gatherOrder : List Shard := (actions.map Action.label).eraseDups

theorem gather_order : gatherOrder = [.lmHead, .embedding, .vo, .mlp, .valueEmbedding, .qk] := by
  decide

def stableLerp (contract : Bool) (weight start finish : F32) : F32 :=
  let cfg := Config.ieeeValue
  let delta := FP.sub cfg finish start
  if FP.lt cfg weight (Value.ofBits .fp32 0x3f000000) then
    if contract then FP.fma cfg weight delta start else FP.add cfg start (FP.mul cfg weight delta)
  else
    let rest := FP.sub cfg TailNorm.one weight
    if contract then FP.fma cfg delta.neg rest finish else FP.sub cfg finish (FP.mul cfg delta rest)

def install (weights : Weights) (label : Shard) (value : Bank) : Weights := fun bank =>
  if bank = label then value else weights bank

def applyAction (s : Snapshot) (weights : Weights) (act : Action) : Option Weights := do
  if !s.seeded act.label act.average then none else
    let new := fun i =>
      let avg := s.average act.label act.average i
      match act.rate with
      | none => TailNorm.narrow avg
      | some rate => TailNorm.narrow
          (stableLerp s.lerpContract rate (TailNorm.widen (weights act.label i)) avg)
    some (install weights act.label new)

def blendLocally (s : Snapshot) : Option Weights := actions.foldlM (applyAction s) s.shard

structure PeerShards where
  rank : Nat
  elementsPerRank : Nat
  read : Nat → Bank

def land (p : PeerShards) (own : Bank) : Bank := fun i =>
  let rank := i / p.elementsPerRank
  let offset := i % p.elementsPerRank
  if rank = p.rank then own offset else p.read rank offset

theorem own_land (p : PeerShards) (own : Bank) (i : Nat) (live : i < p.elementsPerRank) :
    land p own (p.rank * p.elementsPerRank + i) = own i := by
  have positive : 0 < p.elementsPerRank := by omega
  simp [land, Nat.add_comm (p.rank * p.elementsPerRank) i, Nat.add_mul_div_right _ _ positive,
    Nat.div_eq_of_lt live, Nat.mod_eq_of_lt live]

def scaledLabel : Shard → Option TailNorm.Slot
  | .embedding => some .embedding
  | .lmHead => some .lmHead
  | .mlp => some .mlp
  | .qk => some .qk
  | .vo => some .vo
  | .valueEmbedding => none

def label : TailNorm.Slot → Shard
  | .embedding => .embedding
  | .lmHead => .lmHead
  | .mlp => .mlp
  | .qk => .qk
  | .vo => .vo

abbrev Gather (State : Type) := Shard → Bank → State → Option (PeerShards × State)

def execute {State : Type} (restore : TailNorm.Update) (gather : Gather State)
    (s : Snapshot) (initial : State) : Option (Weights × State) := do
  let updated ← blendLocally s
  let (gathered, final) ← gatherOrder.foldlM (fun (assembled, state) bank => do
    let (payload, next) ← gather bank (updated bank) state
    pure (install assembled bank (land payload (updated bank)), next)) (s.whole, initial)
  let restored := restore ⟨true, s.normMath, fun _ _ => TailNorm.zero, s.whole ∘ label, gathered ∘ label⟩
  return ((fun bank => match scaledLabel bank with
    | none => gathered bank
    | some slot => restored slot), final)

theorem execution_equal {State : Type} (gather : Gather State) (s : Snapshot) (initial : State) :
    execute TailNorm.grouped gather s initial = execute TailNorm.reference gather s initial := by
  have eq : TailNorm.grouped = TailNorm.reference := by
    funext p slot i
    exact TailNorm.grouped_equal p slot i
  rw [eq]

#axiomsClean gather_order
#axiomsClean own_land
#axiomsClean execution_equal
end VeriTile.Bench.Models.NanoGPT.Track1.TailShip
