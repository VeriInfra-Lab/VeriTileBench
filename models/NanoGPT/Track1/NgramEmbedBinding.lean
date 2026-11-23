import models.NanoGPT.Track1.NgramEmbedArtifacts
import models.NanoGPT.Track1.NgramEmbedPTX
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The pinned Hopper forward contracts the bigram multiply with the add.
The backward multiplies packed BF16 lanes and publishes two gradient planes. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramEmbedBinding
open Ngram

inductive Layout where
  | flat | fullRows

def expected (layout : Layout) (forward : Bool) : String :=
  match layout, forward with
  | .flat, true => NgramEmbedArtifacts.referenceForward
  | .flat, false => NgramEmbedArtifacts.referenceBackward
  | .fullRows, true => NgramEmbedArtifacts.fullForward
  | .fullRows, false => NgramEmbedArtifacts.fullBackward

def recognizes (layout : Layout) (forward : Bool) (ptx : String) : Bool :=
  QKVTTIR.normalizePTX ptx == QKVTTIR.normalizePTX (expected layout forward)

def Supported (tokens : Nat) : Prop := 0 < tokens ∧ tokens ≤ 262144 ∧ tokens % 16 = 0

instance {tokens : Nat} : Decidable (Supported tokens) := by unfold Supported; infer_instance

def forward {tokens : Nat} (layout : Layout) (ptx : String) (s : Inputs)
    (row : Fin tokens) (channel : Fin D) : Option NgramEmbedPTX.Write :=
  if Supported tokens ∧ s.signMask = 8191 ∧ recognizes layout true ptx = true then
    some (match layout with
      | .flat => NgramEmbedPTX.referenceForward s tokens (index row.val channel.val)
      | .fullRows => NgramEmbedPTX.emittedForward s (FullLane.ofOwner (tileOwner row channel)))
  else none

def backward {tokens : Nat} (layout : Layout) (ptx : String) (s : Inputs)
    (row : Fin tokens) (channel : Fin D) : Option (NgramEmbedPTX.Write × NgramEmbedPTX.Write) :=
  if Supported tokens ∧ s.signMask = 8191 ∧ recognizes layout false ptx = true then
    some (match layout with
      | .flat => NgramEmbedPTX.referenceBackward s tokens (index row.val channel.val)
      | .fullRows => NgramEmbedPTX.emittedBackward s (FullLane.ofOwner (tileOwner row channel)))
  else none

structure Artifacts where
  flatForward : String
  flatBackward : String
  fullForward : String
  fullBackward : String

structure Certificate (a : Artifacts) : Type where
  flatForward : recognizes .flat true a.flatForward = true
  flatBackward : recognizes .flat false a.flatBackward = true
  fullForward : recognizes .fullRows true a.fullForward = true
  fullBackward : recognizes .fullRows false a.fullBackward = true

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : recognizes .flat true a.flatForward = true ∧ recognizes .flat false a.flatBackward = true ∧
      recognizes .fullRows true a.fullForward = true ∧ recognizes .fullRows false a.fullBackward = true then
    some ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩
  else none

theorem replay {a : Artifacts} (checked : Certificate a) {tokens : Nat} (s : Inputs)
    (shape : Supported tokens) (mask : s.signMask = 8191) (row : Fin tokens) (channel : Fin D) :
    forward .fullRows a.fullForward s row channel = forward .flat a.flatForward s row channel ∧
    backward .fullRows a.fullBackward s row channel = backward .flat a.flatBackward s row channel := by
  have output := NgramEmbedPTX.stores_equal s shape.2.1 mask row channel
  exact ⟨by simp only [forward, shape, mask, checked.fullForward, checked.flatForward, true_and, if_true, output.1],
    by simp only [backward, shape, mask, checked.fullBackward, checked.flatBackward, true_and, if_true, output.2]⟩

#axiomsClean replay
end VeriTile.Bench.Models.NanoGPT.Track1.NgramEmbedBinding
