import models.NanoGPT.Track1.TrainingCE

/-! Observation of the fixed-count gradient stores under arbitrary
interleaving of CUDA lanes. Each lane issues its base stores before its
corrections. Within either group, its columns have disjoint addresses. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCE
open CudaStores

private def laneColumns (w : Width) (lane : Lane) : List (Column w) :=
  (List.finRange w.columns).filter fun c => columnOwner c == lane

private def baseStore {w : Width} (writes : Column w → GradientWrites) (c : Column w) : Store :=
  ⟨c.val, (writes c).base⟩

private def patchStore {w : Width} (writes : Column w → GradientWrites) (c : Column w) : Option Store :=
  (writes c).correction.map fun (_, value) => ⟨c.val, value⟩

private def onLane (lane : Lane) (store : Store) : Bool :=
  store.address / 8 % 256 == lane.val

def LaneStoreOrder {w : Width} (writes : Column w → GradientWrites) (issued : List Store) : Prop :=
  ∀ lane : Lane, ∃ base patch : List Store,
    issued.filter (onLane lane) = base ++ patch ∧
    base.Perm ((laneColumns w lane).map (baseStore writes)) ∧
    patch.Perm ((laneColumns w lane).filterMap (patchStore writes))

/-- Stores to other addresses do not affect the byte observed at this
address, even when this address is written more than once. -/
private theorem commit_filtered (stores : List Store) (keep : Store → Bool) (address : Nat)
    (retained : ∀ s : Store, s.address = address → keep s = true)
    (prior filtered : Memory) (same : prior address = filtered address) :
    commitAll stores prior address = commitAll (stores.filter keep) filtered address := by
  induction stores generalizing prior filtered with
  | nil => exact same
  | cons s rest ih =>
    by_cases h : keep s = true
    · simp only [commitAll, List.filter_cons, h, ↓reduceIte]
      exact ih _ _ (by simp [commit, same])
    · have away : address ≠ s.address := fun eq => h (retained s eq.symm)
      simp only [commitAll, List.filter_cons, h]
      exact ih _ _ (by simp [commit, away, same])

private theorem column_filter {w : Width} (c : Column w) :
    (laneColumns w (columnOwner c)).filter (fun d => d.val == c.val) = [c] := by
  have unique : (laneColumns w (columnOwner c)).Nodup :=
    (List.nodup_finRange w.columns).filter (fun d => columnOwner d == columnOwner c)
  have present : c ∈ laneColumns w (columnOwner c) :=
    List.mem_filter.mpr ⟨List.mem_finRange c, by simp⟩
  generalize laneColumns w (columnOwner c) = columns at unique present ⊢
  induction columns with
  | nil => simp at present
  | cons d rest ih =>
    obtain ⟨absent, restUnique⟩ := List.nodup_cons.mp unique
    by_cases eq : d = c
    · subst d
      have filtered : rest.filter (fun d => d.val == c.val) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro d member matchValue
        exact absent (Fin.ext (beq_iff_eq.mp matchValue) ▸ member)
      simp [filtered]
    · have values : d.val ≠ c.val := fun same => eq (Fin.ext same)
      have member : c ∈ rest := (List.mem_cons.mp present).resolve_left (Ne.symm eq)
      simp [values, ih restUnique member]

private def atAddress (address : Nat) (store : Store) : Bool := store.address == address

private theorem base_at_column {w : Width} (writes : Column w → GradientWrites) (c : Column w) :
    ((laneColumns w (columnOwner c)).map (baseStore writes)).filter (atAddress c.val) =
      [baseStore writes c] := by
  rw [List.filter_map]
  change ((laneColumns w (columnOwner c)).filter (fun d => d.val == c.val)).map (baseStore writes) = _
  rw [column_filter]
  rfl

private theorem patch_at_column {w : Width} (writes : Column w → GradientWrites) (c : Column w) :
    ((laneColumns w (columnOwner c)).filterMap (patchStore writes)).filter (atAddress c.val) =
      (patchStore writes c).toList := by
  have filterPatch (columns : List (Column w)) :
      (columns.filterMap (patchStore writes)).filter (atAddress c.val) =
        (columns.filter (fun d => d.val == c.val)).filterMap (patchStore writes) := by
    induction columns with
    | nil => rfl
    | cons d rest ih =>
      cases h : (writes d).correction with
      | none => by_cases same : d.val = c.val <;> simp [patchStore, h, same, ih]
      | some pair =>
        rcases pair with ⟨slot, value⟩
        by_cases same : d.val = c.val <;> simp [patchStore, atAddress, h, same, ih]
  rw [filterPatch, column_filter]
  cases h : (writes c).correction <;> simp [patchStore, h]

/-- Any weave of the lane stores yields the same final byte at every
column. The only ordering requirement is local to each lane. -/
theorem interleaved_gradient_writes_exact {w : Width} (writes : Column w → GradientWrites)
    (issued : List Store) (ordered : LaneStoreOrder writes issued) (prior : Memory) (c : Column w) :
    commitAll issued prior c.val = (writes c).result := by
  obtain ⟨base, patch, projected, basePerm, patchPerm⟩ := ordered (columnOwner c)
  rw [commit_filtered issued (onLane (columnOwner c)) c.val
        (by intro s eq; simp [onLane, eq, columnOwner]) prior prior rfl, projected]
  rw [commit_filtered (base ++ patch) (atAddress c.val) c.val
        (by intro s eq; simp [atAddress, eq]) prior prior rfl, List.filter_append]
  have baseAt : (base.filter (atAddress c.val)).Perm [baseStore writes c] := by
    simpa only [base_at_column] using basePerm.filter (atAddress c.val)
  have patchAt : (patch.filter (atAddress c.val)).Perm (patchStore writes c).toList := by
    simpa only [patch_at_column] using patchPerm.filter (atAddress c.val)
  rw [baseAt.eq_singleton]
  cases h : (writes c).correction with
  | none =>
    have empty : patch.filter (atAddress c.val) = [] := by
      have : (patchStore writes c).toList = [] := by simp [patchStore, h]
      simpa [this] using patchAt
    simp [empty, commitAll, commit, baseStore, GradientWrites.result, h]
  | some pair =>
    rcases pair with ⟨slot, value⟩
    have atByte : patch.filter (atAddress c.val) = [⟨c.val, value⟩] := by
      apply List.Perm.eq_singleton
      simpa [patchStore, h] using patchAt
    simp [atByte, commitAll, commit, GradientWrites.result, h]

theorem committed_gradient_exact {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (walk : PredictionWalk) (sigma : Column w → Half) (lse : F32)
    (issued : List Store) (ordered : LaneStoreOrder
      (gradientWrites p r walk (gradientFactor op p)
        (fun c => (ScaledLogit.ofHalf p (sigma c)).forRow op lse)) issued)
    (prior : Memory) (c : Column w) :
    commitAll issued prior c.val = gradient op p r walk sigma lse c := by
  rw [interleaved_gradient_writes_exact _ _ ordered, gradient_writes_exact]
  rfl

theorem gradient_store_frame {w : Width} (writes : Column w → GradientWrites)
    (issued : List Store) (ordered : LaneStoreOrder writes issued) (prior : Memory)
    (address : Nat) (outside : w.columns ≤ address) :
    commitAll issued prior address = prior address := by
  apply preserve_unwritten
  intro store written atAddress
  let lane : Lane := ⟨store.address / 8 % 256, Nat.mod_lt _ (by decide)⟩
  obtain ⟨base, patch, projected, basePerm, patchPerm⟩ := ordered lane
  have inLane : store ∈ base ++ patch := by
    rw [← projected]
    exact List.mem_filter.mpr ⟨written, by simp [onLane, lane]⟩
  rcases List.mem_append.mp inLane with inBase | inPatch
  · obtain ⟨c, _, eq⟩ := List.mem_map.mp (basePerm.mem_iff.mp inBase)
    have bound := c.isLt
    have value : c.val = address := by rw [← atAddress, ← eq]; rfl
    omega
  · obtain ⟨c, _, eq⟩ := List.mem_filterMap.mp (patchPerm.mem_iff.mp inPatch)
    have bound := c.isLt
    have value : c.val = address := by
      cases h : (writes c).correction with
      | none => simp [patchStore, h] at eq
      | some pair =>
        have same : (⟨c.val, pair.2⟩ : Store) = store := by simpa [patchStore, h] using eq
        exact (congrArg Store.address same).trans atAddress
    omega

#axiomsClean interleaved_gradient_writes_exact
#axiomsClean committed_gradient_exact
#axiomsClean gradient_store_frame
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCE
