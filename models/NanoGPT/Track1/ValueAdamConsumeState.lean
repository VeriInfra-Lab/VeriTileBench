import models.NanoGPT.Track1.ValueAdamNativeShard

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

inductive History where
  | shared
  | clean
  | indexed (previous : PriorMarked)

structure Workspace where
  gradient : HalfShard
  markers : Nat → Nat

def Workspace.zero : Workspace :=
  ⟨fun _ _ => 0, fun _ => 0⟩

def Workspace.indexed (p : PriorMarked) : Workspace :=
  ⟨claimedFrom Workspace.zero.gradient p.pull p.indices p.order eighth p.rows,
   scaleClaim p.pull p.indices (p.order.localAtomics 0) (p.order.peerAtomics 0)⟩

def History.represents (history : History) (bytes : Workspace) : Prop :=
  match history with
  | .shared => True
  | .clean => bytes = Workspace.zero
  | .indexed p => bytes = Workspace.indexed p

def clearSaved (previous : PriorMarked) (bytes : Workspace) (row : Nat) : Line :=
  (List.finRange previous.pull.ids.count).foldl
    (clearEntry fun i => decide ((scaleEntry previous.indices i).toInt = row))
    (bytes.gradient row, bytes.markers row)

def Workspace.beforeScatter (bytes : Workspace) : History → Workspace
  | .shared => Workspace.zero
  | .clean => bytes
  | .indexed previous => ⟨fun row => (clearSaved previous bytes row).1, fun row => (clearSaved previous bytes row).2⟩

theorem clear_saved_zero (p : PriorMarked) (row : Nat) :
    clearSaved p (Workspace.indexed p) row = (fun _ => 0, 0) := by
  change clearIndexedLine p row = _
  rw [← clear_raw_line_equal, clear_raw_line_zero]

theorem sparse_preparation_zero (before : Workspace) (history : History)
    (valid : history.represents before) :
    before.beforeScatter history = Workspace.zero := by
  cases history with
  | shared => rfl
  | clean => exact valid
  | indexed p =>
    change before = Workspace.indexed p at valid
    subst before
    simp [Workspace.beforeScatter, Workspace.zero, clear_saved_zero]

def Workspace.beforeDense (bytes : Workspace) : History → Workspace
  | .clean => bytes
  | _ => {bytes with gradient := Workspace.zero.gradient}

theorem dense_preparation_zero (before : Workspace) (history : History)
    (valid : history.represents before) :
    (before.beforeDense history).gradient = Workspace.zero.gradient := by
  cases history with
  | shared => rfl
  | clean => exact congrArg Workspace.gradient valid
  | indexed => rfl

def History.afterConsume (history : History) (sparse : Bool) : History :=
  if sparse then .clean
  else match history with
    | .clean => .clean
    | _ => .shared

def Workspace.consumed (before : Workspace) (history : History) (sparse : Bool) : Workspace :=
  let prepared := if sparse then before.beforeScatter history else before.beforeDense history
  {prepared with gradient := Workspace.zero.gradient}

/-- The unmarked scatter leaves the prepared marker plane alone. The
checked kernel clears each gradient word after taking its input, so a
later event can skip the zeroing only with both planes known clear. -/
theorem consume_keeps_history (history : History) (before : Workspace) (sparse : Bool)
    (valid : history.represents before) :
    (history.afterConsume sparse).represents (before.consumed history sparse) := by
  cases sparse
  · cases history with
    | shared => trivial
    | clean =>
      change before = Workspace.zero at valid
      subst before
      rfl
    | indexed => trivial
  · simp only [History.afterConsume, Workspace.consumed, if_true]
    rw [sparse_preparation_zero before history valid]
    rfl

#axiomsClean clear_saved_zero
#axiomsClean sparse_preparation_zero
#axiomsClean dense_preparation_zero
#axiomsClean consume_keeps_history

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
