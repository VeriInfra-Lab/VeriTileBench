import models.NanoGPT.Track1.AnvilGramTMA

/-! Each CTA publishes its Gram tile after its writes complete.
The first Gram's last diagonal publisher writes its matrix's divisors. In later
rounds each CTA acquires the tiles read by its two operands and bias. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram.Cooperative

open AnvilEpilogue (Bank)
open TailNorm (F32)

inductive Arrivals (blocks : Nat) : List (Fin blocks) → List (Fin blocks) → Prop where
  | initial : Arrivals blocks [] (List.finRange blocks)
  | publish {published before after : List (Fin blocks)} (block : Fin blocks)
      (previous : Arrivals blocks published (before ++ block :: after)) :
      Arrivals blocks (block :: published) (before ++ after)

theorem arrival_count {blocks : Nat} {published pending : List (Fin blocks)}
    (h : Arrivals blocks published pending) : published.length + pending.length = blocks := by
  induction h with
  | initial => simp
  | publish _ _ ih => simp only [List.length_append, List.length_cons] at *; omega

theorem tracked_block {blocks : Nat} {published pending : List (Fin blocks)}
    (h : Arrivals blocks published pending) (block : Fin blocks) :
    block ∈ published ∨ block ∈ pending := by
  induction h with
  | initial => simp
  | publish id _ ih =>
    simp only [List.mem_append, List.mem_cons] at *
    tauto

def matrixCTAs (bank : Bank) : Nat :=
  Bank.side bank * (Bank.side bank + 1) / 2

theorem positive_matrix_grid (bank : Bank) : 0 < matrixCTAs bank := by cases bank <;> decide

theorem launch_factorization (bank : Bank) : compactCount bank = batches bank * matrixCTAs bank := rfl

def matrixPid (bank : Bank) (matrix : Fin (batches bank)) (tile : Fin (matrixCTAs bank)) :
    Fin (compactCount bank) :=
  ⟨matrix.val * matrixCTAs bank + tile.val, by
    have h := Nat.mul_le_mul_right (matrixCTAs bank) (Nat.succ_le_of_lt matrix.isLt)
    simp only [Nat.succ_mul] at h
    rw [launch_factorization]
    omega⟩

theorem matrix_counter_address (bank : Bank) (matrix : Fin (batches bank)) (tile : Fin (matrixCTAs bank)) :
    (matrixPid bank matrix tile).val / matrixCTAs bank = matrix.val := by
  change (matrix.val * matrixCTAs bank + tile.val) / matrixCTAs bank = matrix.val
  rw [Nat.mul_comm matrix.val, Nat.mul_add_div (positive_matrix_grid bank), Nat.div_eq_of_lt tile.isLt]
  simp

theorem matrix_tile_slot (bank : Bank) (matrix : Fin (batches bank)) (tile : Fin (matrixCTAs bank)) :
    (matrixPid bank matrix tile).val % matrixCTAs bank = tile.val := by
  change (matrix.val * matrixCTAs bank + tile.val) % matrixCTAs bank = tile.val
  rw [Nat.mul_comm matrix.val, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt tile.isLt]

theorem tile_reads_its_matrix (bank : Bank) (matrix : Fin (batches bank)) (tile : Fin (matrixCTAs bank)) :
    (compactTile (Bank.side bank) (matrixPid bank matrix tile).val).batch = matrix.val := by
  exact matrix_counter_address bank matrix tile

theorem last_ticket_iff {blocks : Nat} {published before after : List (Fin blocks)}
    (block : Fin blocks) (h : Arrivals blocks published (before ++ block :: after)) :
    published.length = blocks - 1 ↔ before ++ after = [] := by
  have total := arrival_count h
  simp only [List.length_append, List.length_cons] at total
  rw [← List.length_eq_zero_iff, List.length_append]
  omega

theorem last_publisher_sees_every_tile {blocks : Nat} {published before after : List (Fin blocks)}
    (block : Fin blocks) (h : Arrivals blocks published (before ++ block :: after))
    (last : published.length = blocks - 1) :
    ∀ tile : Fin blocks, tile ∈ block :: published := by
  let afterRelease := Arrivals.publish block h
  have finished := (last_ticket_iff block h).mp last
  intro tile
  simpa [finished] using tracked_block afterRelease tile

theorem no_second_final_publisher {blocks : Nat} {published before after : List (Fin blocks)}
    (block next : Fin blocks) (h : Arrivals blocks published (before ++ block :: after))
    (last : published.length = blocks - 1) (front rest : List (Fin blocks)) :
    before ++ after ≠ front ++ next :: rest := by
  rw [(last_ticket_iff block h).mp last]
  intro impossible
  have lengths := congrArg List.length impossible
  simp at lengths

def diagonalSlot (bank : Bank) (row : Nat) : Nat :=
  row * (2 * Bank.side bank + 1 - row) / 2

theorem diagonal_slot_bounded (bank : Bank) :
    ∀ row : Fin (Bank.side bank), diagonalSlot bank row.val < matrixCTAs bank := by
  cases bank <;> decide

def diagonalPid (bank : Bank) (matrix : Fin (batches bank)) (row : Fin (Bank.side bank)) :
    Fin (compactCount bank) :=
  matrixPid bank matrix ⟨diagonalSlot bank row.val, diagonal_slot_bounded bank row⟩

theorem diagonal_coordinates (bank : Bank) :
    ∀ row : Fin (Bank.side bank),
      (compactTile (Bank.side bank) (diagonalSlot bank row.val)).row = row.val ∧
      (compactTile (Bank.side bank) (diagonalSlot bank row.val)).col = row.val := by
  cases bank <;> decide

theorem diagonal_cta_owns_tile (bank : Bank) (matrix : Fin (batches bank)) (row : Fin (Bank.side bank)) :
    compactTile (Bank.side bank) (diagonalPid bank matrix row).val = ⟨matrix.val, row.val, row.val⟩ := by
  let tile : Fin (matrixCTAs bank) := ⟨diagonalSlot bank row.val, diagonal_slot_bounded bank row⟩
  have col := matrix_tile_slot bank matrix tile
  have batch := matrix_counter_address bank matrix tile
  obtain ⟨hr, hc⟩ := diagonal_coordinates bank row
  change (matrixPid bank matrix tile).val % matrixCTAs bank = tile.val at col
  simp only [compactTile] at hr hc ⊢
  change (matrixPid bank matrix tile).val / matrixCTAs bank = matrix.val at batch
  simp only [diagonalPid]
  change Tile.mk ((matrixPid bank matrix tile).val / matrixCTAs bank)
    (((if Bank.side bank == 2 then (0x48 : UInt64) else 0x5b2c76356346c688) >>>
      UInt64.ofNat (3 * ((matrixPid bank matrix tile).val % matrixCTAs bank))) &&& 7).toNat
    (((if Bank.side bank == 2 then (0x40 : UInt64) else 0x591b692449240000) >>>
      UInt64.ofNat (3 * ((matrixPid bank matrix tile).val % matrixCTAs bank))) &&& 7).toNat = _
  rw [batch, col]
  have hlt := tile.isLt
  simp only [matrixCTAs] at hlt
  rw [Nat.mod_eq_of_lt hlt] at hr hc
  exact congrArg₂ (Tile.mk matrix.val) hr hc

theorem last_publisher_has_diagonal {bank : Bank}
    (matrix : Fin (batches bank)) (row : Fin (Bank.side bank))
    {published before after : List (Fin (Bank.side bank))}
    (cta : Fin (Bank.side bank)) (h : Arrivals (Bank.side bank) published (before ++ cta :: after))
    (ticket : published.length = Bank.side bank - 1) :
    ∃ publisher : Fin (Bank.side bank), publisher ∈ cta :: published ∧
      compactTile (Bank.side bank) (diagonalPid bank matrix publisher).val =
        ⟨matrix.val, row.val, row.val⟩ :=
  ⟨row, last_publisher_sees_every_tile cta h ticket row, diagonal_cta_owns_tile bank matrix row⟩

theorem each_diagonal_word_is_published {bank : Bank} (matrix : Fin (batches bank))
    (index : Fin (outputWidth bank))
    {published before after : List (Fin (Bank.side bank))}
    (cta : Fin (Bank.side bank)) (h : Arrivals (Bank.side bank) published (before ++ cta :: after))
    (ticket : published.length = Bank.side bank - 1) :
    ∃ publisher : Fin (Bank.side bank), publisher ∈ cta :: published ∧
      compactTile (Bank.side bank) (diagonalPid bank matrix publisher).val = (Cell.mk matrix index index).tile := by
  let row : Fin (Bank.side bank) := ⟨index.val / 128, by
    have hw := index.isLt
    simp only [outputWidth] at hw
    omega⟩
  obtain ⟨owner, wasReleased, writes⟩ := last_publisher_has_diagonal matrix row cta h ticket
  exact ⟨owner, wasReleased, by simpa [Cell.tile, owningTile, row] using writes⟩

private def bitField (start width : Nat) : Nat := (2 ^ width - 1) <<< start

/-- The masks use the same lower-triangle order as the two TMA destinations.
For six rows the earlier columns form a shifted prefix, and the current
column contributes a contiguous field. -/
def incidentMask (bank : Bank) (row : Nat) : UInt32 :=
  UInt32.ofNat (if bank == .qk then 3 <<< row else
    let start := row * (13 - row) / 2
    (((bitField 0 (start - row) &&& 0xd221) <<< row) ||| bitField start (6 - row)))

private def localTile (bank : Bank) (slot : Fin (matrixCTAs bank)) : Tile :=
  compactTile (Bank.side bank) slot.val

theorem incident_mask_exact (bank : Bank) :
    ∀ (row : Fin (Bank.side bank)) (slot : Fin (matrixCTAs bank)),
      (incidentMask bank row.val).toNat.testBit slot.val =
        decide ((localTile bank slot).row = row.val ∨ (localTile bank slot).col = row.val) := by
  cases bank <;> decide

theorem local_coordinates_in_bounds (bank : Bank) :
    ∀ slot : Fin (matrixCTAs bank),
      (localTile bank slot).row < Bank.side bank ∧ (localTile bank slot).col < Bank.side bank := by
  cases bank <;> decide

theorem every_pair_has_a_tile (bank : Bank) :
    ∀ row col : Fin (Bank.side bank),
      ∃ slot : Fin (matrixCTAs bank),
        localTile bank slot = owningTile 0 row.val col.val := by
  cases bank <;> decide

def neededMask (bank : Bank) (target : Fin (matrixCTAs bank)) : UInt32 :=
  let tile := localTile bank target
  incidentMask bank tile.row ||| incidentMask bank tile.col

def publicationBit {bank : Bank} (slot : Fin (matrixCTAs bank)) : UInt32 :=
  UInt32.ofNat (2 ^ slot.val)

theorem publication_bit_exact (bank : Bank) :
    ∀ a b : Fin (matrixCTAs bank), (publicationBit (bank := bank) a).toNat.testBit b.val = decide (a = b) := by
  cases bank <;> decide

def publishedMask (bank : Bank) : List (Fin (matrixCTAs bank)) → UInt32
  | [] => 0
  | slot :: seen => publicationBit (bank := bank) slot ||| publishedMask bank seen

theorem observed_bit_exact (bank : Bank) (seen : List (Fin (matrixCTAs bank))) (slot : Fin (matrixCTAs bank)) :
    (publishedMask bank seen).toNat.testBit slot.val = decide (slot ∈ seen) := by
  induction seen with
  | nil => simp [publishedMask]
  | cons head tail ih =>
    simp only [publishedMask, UInt32.toNat_or, Nat.testBit_or, publication_bit_exact, ih,
      List.mem_cons, eq_comm]
    by_cases same : slot = head <;> simp [same]

theorem own_release_extends_observation (bank : Bank) (seen : List (Fin (matrixCTAs bank)))
    (slot : Fin (matrixCTAs bank)) :
    publishedMask bank seen ||| publicationBit (bank := bank) slot =
      publishedMask bank (slot :: seen) := by
  apply UInt32.toNat_inj.mp
  simp only [publishedMask, UInt32.toNat_or]
  exact Nat.or_comm _ _

theorem full_mask_releases_every_cta (bank : Bank) :
    ∀ target : Fin (matrixCTAs bank),
      publishedMask bank (List.finRange (matrixCTAs bank)) &&& neededMask bank target =
        neededMask bank target := by
  cases bank <;> decide

theorem mask_admits_needed {bank : Bank} {seen : List (Fin (matrixCTAs bank))}
    (target source : Fin (matrixCTAs bank))
    (covered : publishedMask bank seen &&& neededMask bank target = neededMask bank target)
    (needed : (neededMask bank target).toNat.testBit source.val = true) :
    source ∈ seen := by
  have bit := congrArg (fun bits : UInt32 => bits.toNat.testBit source.val) covered
  simp only [UInt32.toNat_and, Nat.testBit_and, needed, Bool.and_true, observed_bit_exact,
    decide_eq_true_eq] at bit
  exact bit

theorem row_operand_is_published {bank : Bank} {seen : List (Fin (matrixCTAs bank))}
    (target : Fin (matrixCTAs bank)) (row : Fin (Bank.side bank))
    (oneOperand : row.val = (localTile bank target).row ∨ row.val = (localTile bank target).col)
    (inner : Fin (Bank.side bank))
    (covered : publishedMask bank seen &&& neededMask bank target = neededMask bank target) :
    ∃ source ∈ seen, localTile bank source = owningTile 0 row.val inner.val := by
  obtain ⟨source, owns⟩ := every_pair_has_a_tile bank row inner
  have includes : (localTile bank source).row = row.val ∨ (localTile bank source).col = row.val := by
    rw [owns]
    simp only [owningTile]
    by_cases h : row.val ≤ inner.val
    · exact Or.inr (min_eq_left h)
    · exact Or.inl (max_eq_left (by omega))
  have bit := incident_mask_exact bank row source
  rw [decide_eq_true includes] at bit
  refine ⟨source, mask_admits_needed target source covered ?_, owns⟩
  simp only [neededMask, UInt32.toNat_or, Nat.testBit_or]
  rcases oneOperand with first | second
  · rw [← first, bit, Bool.true_or]
  · rw [← second, bit, Bool.or_true]

theorem bias_tile_is_published {bank : Bank} {seen : List (Fin (matrixCTAs bank))}
    (target : Fin (matrixCTAs bank))
    (covered : publishedMask bank seen &&& neededMask bank target = neededMask bank target) :
    target ∈ seen := by
  obtain ⟨bounded, _⟩ := local_coordinates_in_bounds bank target
  have bit := incident_mask_exact bank ⟨(localTile bank target).row, bounded⟩ target
  simp only [true_or, decide_true] at bit
  apply mask_admits_needed target target covered
  simp [neededMask, UInt32.toNat_or, bit]

/-- Round zero uses the diagonal ticket counter and each later round owns a
fresh zero mask. Graph replay clears all six slices during rail preparation. -/
def publicationSlot (bank : Bank) (round : Fin 6) (matrix : Fin (batches bank)) : Nat :=
  round.val * batches bank + matrix.val

theorem publication_slot_bounded (bank : Bank) (round : Fin 6) (matrix : Fin (batches bank)) :
    publicationSlot bank round matrix < 6 * batches bank := by
  have bound := Nat.mul_le_mul_right (batches bank) (Nat.succ_le_of_lt round.isLt)
  simp only [Nat.succ_mul] at bound
  unfold publicationSlot
  omega

theorem publication_slot_injective (bank : Bank) :
    ∀ (r s : Fin 6) (a b : Fin (batches bank)),
      publicationSlot bank r a = publicationSlot bank s b → r = s ∧ a = b := by
  cases bank <;> decide

theorem seed_lanes_cover_all_rounds (bank : Bank) : 6 * batches bank ≤ 64 := by cases bank <;> decide

theorem matrix_slot_has_same_coordinates (bank : Bank) (matrix : Fin (batches bank))
    (slot : Fin (matrixCTAs bank)) :
    compactTile (Bank.side bank) (matrixPid bank matrix slot).val =
      { localTile bank slot with batch := matrix.val } := by
  have hBatch : (matrixPid bank matrix slot).val / (Bank.side bank * (Bank.side bank + 1) / 2)
      = matrix.val := matrix_counter_address bank matrix slot
  have hSlot : (matrixPid bank matrix slot).val % (Bank.side bank * (Bank.side bank + 1) / 2)
      = slot.val := matrix_tile_slot bank matrix slot
  have hLocal : slot.val % (Bank.side bank * (Bank.side bank + 1) / 2) = slot.val :=
    Nat.mod_eq_of_lt slot.isLt
  simp only [compactTile, localTile, hBatch, hSlot, hLocal]

theorem matrix_operand_is_published {bank : Bank} {seen : List (Fin (matrixCTAs bank))}
    (matrix : Fin (batches bank)) (target : Fin (matrixCTAs bank)) (row : Fin (Bank.side bank))
    (oneOperand : row.val = (localTile bank target).row ∨ row.val = (localTile bank target).col)
    (inner : Fin (Bank.side bank))
    (covered : publishedMask bank seen &&& neededMask bank target = neededMask bank target) :
    ∃ source ∈ seen, compactTile (Bank.side bank) (matrixPid bank matrix source).val =
      owningTile matrix.val row.val inner.val := by
  obtain ⟨slot, published, owns⟩ := row_operand_is_published target row oneOperand inner covered
  exact ⟨slot, published, by rw [matrix_slot_has_same_coordinates, owns]; rfl⟩

theorem dependency_count (bank : Bank) :
    ∀ slot : Fin (matrixCTAs bank),
      ((List.finRange (matrixCTAs bank)).filter
          (fun bit => (neededMask bank slot).toNat.testBit bit.val)).length =
        if (localTile bank slot).row = (localTile bank slot).col then Bank.side bank
        else 2 * Bank.side bank - 1 := by
  cases bank <;> decide

/-- The first phase commits precisely the Gram image read by the second
phase. The runtime supplies the same ordered native WGMMA primitive. -/
def round {State : Type} (rt : Runtime State) (bank : Bank) (alpha beta : F32)
    (s : State) : Option State := do
  let (input, s) ← rt.read bank .gram s
  let s ← rt.commit bank .gram (TMA.run bank .gram input) s
  let op := Operation.polynomial alpha beta
  let (input, s) ← rt.read bank op s
  rt.commit bank op (TMA.run bank op input) s

theorem round_eq {State : Type} (rt : Runtime State) (bank : Bank) (alpha beta : F32)
    (s : State) :
    round rt bank alpha beta s =
      (launch rt reference bank .gram s).bind (launch rt reference bank (.polynomial alpha beta)) := by
  unfold round launch
  rw [TMA.run_eq, stores_equal]
  cases rt.read bank .gram s with
  | none => rfl
  | some entry =>
    obtain ⟨input, state⟩ := entry
    cases rt.commit bank .gram (reference bank .gram input) state <;> rfl

def execute : Round := fun _ rt bank c state => round rt bank c.alpha c.beta state

theorem execute_eq : execute = splitRound reference := by
  funext State rt bank c state
  exact round_eq rt bank c.alpha c.beta state

/-- The last diagonal publisher produces the original divisor words from the finished
Gram. The following launch applies the two divisions at their original BF16
rounding points. The divisor storage and the count are private to the driver. -/
def fusedInitial : Initial := fun _ rt bank normalize state => do
  let (input, state) ← rt.read bank .gram state
  let state ← rt.commit bank .gram (TMA.run bank .gram input) state
  let (input, state) ← rt.readNormalization bank state
  rt.commitNormalization bank (normalize bank input) state

theorem fused_initial_eq :
    fusedInitial = splitInitial reference := by
  funext State rt bank normalize state
  simp only [fusedInitial, splitInitial, launch, TMA.run_eq, stores_equal]
  cases rt.read bank .gram state with
  | none => rfl
  | some entry =>
    obtain ⟨input, state⟩ := entry
    cases rt.commit bank .gram (reference bank .gram input) state <;> rfl

theorem initial_launches_removed :
    (2 - 1) * [Bank.qk, .vo, .mlp].length = 3 := by decide

theorem total_launches_removed :
    ((2 - 1) + (coefficients.length - 1)) * [Bank.qk, .vo, .mlp].length = 18 := by decide

#axiomsClean fused_initial_eq
#axiomsClean initial_launches_removed
#axiomsClean total_launches_removed


def seedsCounter (bank : Bank) (cta lane : Nat) : Bool := cta == 0 && lane < 6 * batches bank

theorem seed_owner (bank : Bank) (cta lane : Nat) :
    seedsCounter bank cta lane = true ↔ cta = 0 ∧ lane < 6 * batches bank := by
  simp [seedsCounter]

theorem grid_fits_full_hopper (bank : Bank) : compactCount bank ≤ 132 := by
  cases bank <;> decide

theorem initial_publications_removed :
    ([Bank.qk, .vo, .mlp].map fun bank => compactCount bank - batches bank * Bank.side bank).sum = 81 := by
  decide

#axiomsClean each_diagonal_word_is_published
#axiomsClean initial_publications_removed
#axiomsClean seed_owner
#axiomsClean last_publisher_sees_every_tile
#axiomsClean no_second_final_publisher
#axiomsClean own_release_extends_observation
#axiomsClean full_mask_releases_every_cta
#axiomsClean incident_mask_exact
#axiomsClean seed_lanes_cover_all_rounds
#axiomsClean publication_slot_bounded
#axiomsClean publication_slot_injective
#axiomsClean matrix_operand_is_published
#axiomsClean bias_tile_is_published
#axiomsClean dependency_count
#axiomsClean execute_eq
#axiomsClean grid_fits_full_hopper
end VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram.Cooperative
