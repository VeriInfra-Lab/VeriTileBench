import models.NanoGPT.Track1.FPEquiv
import models.NanoGPT.Track1.YarnBinding
import models.NanoGPT.Track1.YarnStationaryBinding

/-! Compact YaRN decoders at the training and evaluation call sites.
The host table evolution keeps the original arithmetic and feeds the
matched decoder at every supported layout and forward or backward size. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver.YarnExecution

open RopeShift (Length)
open YarnTable (Layout Phase)

structure Decoders (p : Phase → Type) where
  forward : Length → p .forward
  backward : Length → Option (p .backward)

def lengths : List Length := [.t16k, .t32k, .t40k, .t48k, .validation]

def Decoders.matches {p : Phase → Type} (d : Decoders p) (check : (phase : Phase) → p phase → Bool) : Bool :=
  lengths.all fun n =>
    check .forward (d.forward n) && ((d.backward n).map (check .backward)).getD true

theorem Decoders.matches_eq_true {p : Phase → Type} (d : Decoders p)
    (check : (phase : Phase) → p phase → Bool)
    (valid : ∀ phase program, check phase program = true) :
    d.matches check = true := by
  apply List.all_eq_true.mpr
  intro n _
  rw [valid]
  cases program : d.backward n <;> simp [program, valid]

structure Compact (s : Layout) (phase : Phase) where
  ir : String
  ptx : String
  checked : YarnTable.Certificate s phase ir ptx

structure Stationary (phase : Phase) where
  ir : String
  ptx : String
  checked : YarnStationary.Certificate phase ir ptx

structure Programs where
  compact : (s : Layout) → Decoders (Compact s)
  stationary : Decoders Stationary

private def Compact.matches {s : Layout} {phase : Phase} (program : Compact s phase) : Bool :=
  YarnTable.matchesDecoder s phase program.ir program.ptx

private def Stationary.matches {phase : Phase} (program : Stationary phase) : Bool :=
  YarnStationary.matchesDecoder phase program.ir program.ptx

def Programs.compactTables (k : Programs) (a : YarnTable.Arithmetic) (s : Layout)
    (rows : Nat) (state : YarnTable.State) (steps : List YarnTable.Step) : YarnTable.State :=
  if (k.compact s).matches (fun _ p => p.matches) then YarnTable.observed a s rows state steps
  else YarnTable.reference a s rows state steps

theorem Programs.compactTables_eq (k : Programs) : k.compactTables = candidateCalls.yarnTables := by
  funext a s rows state steps
  have matched := (k.compact s).matches_eq_true (fun _ p => p.matches) (fun _ p => p.checked.matched)
  simp only [Programs.compactTables, matched, if_true, candidateCalls]

def Programs.stationaryTables (k : Programs) (a : YarnTable.Arithmetic) (rows : Nat)
    (state : YarnStationary.RotatingState) (steps : List YarnStationary.Step) : YarnStationary.Observation :=
  if k.stationary.matches (fun _ p => p.matches) then YarnStationary.candidate a rows state steps
  else YarnStationary.reference a rows state steps

theorem Programs.stationaryTables_eq (k : Programs) : k.stationaryTables = candidateCalls.wideYarnTables := by
  funext a rows state steps
  have matched := k.stationary.matches_eq_true (fun _ p => p.matches) (fun _ p => p.checked.matched)
  simp only [Programs.stationaryTables, matched, if_true, candidateCalls]

#axiomsClean Programs.compactTables_eq
#axiomsClean Programs.stationaryTables_eq

end VeriTile.Bench.Models.NanoGPT.Track1.Driver.YarnExecution
