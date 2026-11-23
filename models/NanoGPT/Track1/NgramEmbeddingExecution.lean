import models.NanoGPT.Track1.FPEquiv
import models.NanoGPT.Track1.NgramEmbedBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver.NgramEmbedding

open VeriTile.Triton FP Ngram

set_option maxRecDepth 4096

def lookup (a : NgramEmbedBinding.Artifacts) (fullRows : Bool) (tokens : Nat) (s : Inputs)
    (contract : Contraction) (row : Fin tokens) (col : Fin D) : Value .bf16 :=
  let source := candidateCalls.lookup fullRows tokens s contract row col
  match fullRows, contract with
  | true, .leftFma =>
    ((NgramEmbedBinding.forward .fullRows a.fullForward s row col).map Prod.snd).getD source
  | _, _ => source

def backward (a : NgramEmbedBinding.Artifacts) (fullRows : Bool) (tokens : Nat) (s : Inputs)
    (arithmetic : BackwardArithmetic) (row : Fin tokens) (col : Fin D) : Value .bf16 × Value .bf16 :=
  let source := candidateCalls.backward fullRows tokens s arithmetic row col
  match fullRows, arithmetic with
  | true, .packed =>
    ((NgramEmbedBinding.backward .fullRows a.fullBackward s row col).map
      (fun stores => (stores.1.2, stores.2.2))).getD source
  | _, _ => source

private theorem full_forward_value {tokens : Nat} (s : Inputs) (row : Fin tokens) (col : Fin D)
    (mask : s.signMask = 8191) :
    (NgramEmbedPTX.emittedForward s (FullLane.ofOwner (tileOwner row col))).2 =
      candidateCalls.lookup true tokens s .leftFma row col := by
  let lane := FullLane.ofOwner (tileOwner row col)
  change (NgramEmbedPTX.emittedForward s lane).2 = fullForward s lane .leftFma
  exact congrArg Prod.snd (NgramEmbedPTX.emitted_forward s lane mask)

private theorem full_backward_values {tokens : Nat} (s : Inputs) (row : Fin tokens) (col : Fin D)
    (mask : s.signMask = 8191) :
    let stored := NgramEmbedPTX.emittedBackward s (FullLane.ofOwner (tileOwner row col))
    (stored.1.2, stored.2.2) = candidateCalls.backward true tokens s .packed row col := by
  dsimp only
  let lane := FullLane.ofOwner (tileOwner row col)
  let words := tiledBackward s lane.owner .packed
  have out := congrArg (fun writes : NgramEmbedPTX.Write × NgramEmbedPTX.Write => (writes.1.2, writes.2.2))
    (NgramEmbedPTX.emitted_backward s lane mask)
  change _ = fullBackward s lane .packed
  change _ = (let (a, b) := words; (a, b)) at out
  have pair (w : Value .bf16 × Value .bf16) : (let (a, b) := w; (a, b)) = w := by
    rcases w with ⟨a, b⟩
    rfl
  exact out.trans (pair words)

theorem lookup_eq {a : NgramEmbedBinding.Artifacts} (checked : NgramEmbedBinding.Certificate a) :
    lookup a = candidateCalls.lookup := by
  funext fullRows tokens s contract row col
  cases fullRows <;> cases contract <;> try rfl
  simp only [lookup, NgramEmbedBinding.forward, checked.fullForward, and_true]
  split
  · rename_i supported
    exact full_forward_value s row col supported.2
  · rfl

theorem backward_eq {a : NgramEmbedBinding.Artifacts} (checked : NgramEmbedBinding.Certificate a) :
    backward a = candidateCalls.backward := by
  funext fullRows tokens s arithmetic row col
  cases fullRows <;> cases arithmetic <;> try rfl
  simp only [backward, NgramEmbedBinding.backward, checked.fullBackward, and_true]
  split
  · rename_i supported
    exact full_backward_values s row col supported.2
  · rfl

#axiomsClean lookup_eq
#axiomsClean backward_eq

end VeriTile.Bench.Models.NanoGPT.Track1.Driver.NgramEmbedding
