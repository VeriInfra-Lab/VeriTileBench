import VeriTile.Meta.StatementAudit
import models.NanoGPT.Track1.NgramSlots
import models.NanoGPT.Track1.TrainingCE

namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphStage

/-! The graph is already captured and reads six independent static buffers.
The inputs to each copy are observed as raw words, including the bit patterns
of the two FP32 weight buffers. -/

inductive Shape where
  | initial | middle | bulk | taper | extension

def shapes : List Shape := [.initial, .middle, .bulk, .taper, .extension]

def Shape.tokens : Shape → Nat
  | .initial | .extension => 16384
  | .middle => 32768
  | .bulk => 49152
  | .taper => 40960

def Shape.ends : Shape → Nat
  | .initial | .extension => 64
  | .middle => 96
  | .bulk | .taper => 128

def Shape.mtp : Shape → Nat
  | .initial => 3
  | .middle => 2
  | .bulk | .taper | .extension => 1

def Shape.blocks (s : Shape) : Nat := s.tokens / 256

inductive Field where
  | inputs | targets | ends | slots | mtp | prefix
  deriving DecidableEq

def extent (s : Shape) : Field → Nat
  | .inputs | .targets => s.tokens
  | .slots => 2 * s.tokens
  | .ends => s.ends
  | .mtp => s.mtp
  | .prefix => 1

def Field.tile : Field → Nat
  | .inputs | .targets => 256
  | .slots => 512
  | .ends => 128
  | .mtp => 4
  | .prefix => 1

def Field.firstOnly : Field → Bool
  | .inputs | .targets | .slots => false
  | .ends | .mtp | .prefix => true

def valid (s : Shape) (f : Field) (pid lane : Nat) : Prop :=
  pid < s.blocks ∧ lane < f.tile ∧ (f.firstOnly → pid = 0) ∧ pid * f.tile + lane < extent s f

theorem unique_owner (s : Shape) (f : Field) (word : Fin (extent s f)) :
    ∃ pid lane, valid s f pid lane ∧
      pid * f.tile + lane = word.val ∧
      ∀ a b, valid s f a b → a * f.tile + b = word.val → a = pid ∧ b = lane := by
  let pid := word.val / f.tile
  let lane := word.val % f.tile
  refine ⟨pid, lane, ?_, ?_, ?_⟩
  · have h := word.isLt
    cases f <;> cases s <;>
      simp [pid, lane, valid, Field.tile, Field.firstOnly, extent, Shape.blocks,
        Shape.tokens, Shape.ends, Shape.mtp] at h ⊢ <;> omega
  · exact Nat.div_add_mod' _ _
  · intro a b inside eq
    have hb := inside.2.1
    have nz : 0 < f.tile := by cases f <;> decide
    have hdiv : (a * f.tile + b) / f.tile = a := by
      rw [Nat.mul_comm a f.tile, Nat.mul_add_div nz, Nat.div_eq_of_lt hb, Nat.add_zero]
    have hmod : (a * f.tile + b) % f.tile = b := by
      rw [Nat.mul_comm a f.tile, Nat.mul_add_mod, Nat.mod_eq_of_lt hb]
    simp [pid, lane, ← eq, hdiv, hmod]


abbrev StaticInputs (s : Shape) := (field : Field) → Fin (extent s field) → UInt64

def sourceWord (s : Shape) (input : StaticInputs s) (f : Field) (pid lane : Nat) : Option UInt64 :=
  let _ : Decidable (valid s f pid lane) := by unfold valid; infer_instance
  if h : valid s f pid lane then some (input f ⟨pid * f.tile + lane, h.2.2.2⟩)
  else none

/-- The launched store reads the same source address as the corresponding
word in the eager device copy. All fields use raw bits. -/
theorem replay_copy_exact (s : Shape) (input : StaticInputs s) (f : Field)
    (word : Fin (extent s f)) :
    ∃ pid lane, valid s f pid lane ∧ pid * f.tile + lane = word.val ∧
    sourceWord s input f pid lane = some (input f word) := by
  obtain ⟨pid, lane, live, address, _⟩ := unique_owner s f word
  refine ⟨pid, lane, live, address, ?_⟩
  simp only [sourceWord, dif_pos live, Option.some.injEq]
  exact congrArg (input f) (Fin.ext address)

abbrev Copy := (s : Shape) → StaticInputs s → StaticInputs s

def eager : Copy := fun _ input => input

/-- Observe a destination at the only block and lane that writes it. -/
def fused : Copy := fun s input f word =>
  (sourceWord s input f (word.val / f.tile) (word.val % f.tile)).getD 0

theorem copy_equal : fused = eager := by
  funext s input f word
  obtain ⟨pid, lane, live, address, unique⟩ := unique_owner s f word
  obtain ⟨a, b, ha, hab, read⟩ := replay_copy_exact s input f word
  obtain ⟨ha', hb'⟩ := unique a b ha hab
  have atOwner : sourceWord s input f pid lane = some (input f word) := by
    simpa only [ha', hb'] using read
  have hLane := live.2.1
  have hTile : 0 < f.tile := by cases f <;> decide
  have hDiv : word.val / f.tile = pid := by
    rw [← address, Nat.mul_comm pid f.tile, Nat.mul_add_div hTile,
      Nat.div_eq_of_lt hLane, Nat.add_zero]
  have hMod : word.val % f.tile = lane := by
    rw [← address, Nat.mul_comm pid f.tile, Nat.mul_add_mod, Nat.mod_eq_of_lt hLane]
  simp [fused, eager, hDiv, hMod, atOwner]

def queryBlocks (s : Shape) : Nat := extent s .slots / 128

def queryCTA (s : Shape) (index : Nat) : Nat := s.blocks + index / 128

def queryLane (index : Nat) : Nat := index % 128

/-- One lookup CTA and lane own both slot stores for an input row. The copy
CTAs precede all lookup CTAs in the one-dimensional launch. -/
theorem query_ownership (s : Shape) (word : Fin (extent s .slots)) :
    s.blocks ≤ queryCTA s word.val ∧ queryCTA s word.val < s.blocks + queryBlocks s ∧
    queryLane word.val < 128 ∧
    (queryCTA s word.val - s.blocks) * 128 + queryLane word.val = word.val ∧
    ∀ pid lane, s.blocks ≤ pid → lane < 128 →
      (pid - s.blocks) * 128 + lane = word.val → pid = queryCTA s word.val ∧ lane = queryLane word.val := by
  refine ⟨by simp [queryCTA], ?_, Nat.mod_lt _ (by decide), ?_, ?_⟩
  · have fits := word.isLt
    cases s <;> simp [queryCTA, queryBlocks, extent, Shape.blocks, Shape.tokens] at fits ⊢ <;> omega
  · simp only [queryCTA, queryLane, Nat.add_sub_cancel_left]
    exact Nat.div_add_mod' _ _
  · intro pid lane afterCopies validLane address
    unfold queryCTA queryLane
    omega

structure Observation (s : Shape) where
  graph : StaticInputs s
  pendingSlots : Fin (extent s .slots) → Nat

def replaceSlots {s : Shape} (fields : StaticInputs s) (slots : Fin (extent s .slots) → Nat) :
    StaticInputs s
  | .slots, i => UInt64.ofNat (slots i)
  | .inputs, i => fields .inputs i
  | .targets, i => fields .targets i
  | .ends, i => fields .ends i
  | .mtp, i => fields .mtp i
  | .prefix, i => fields .prefix i

abbrev StageWithLookup :=
  (s : Shape) → StaticInputs s → NgramSlots.Want → (Fin (extent s .slots) → Int) → Observation s

def separateLookup : StageWithLookup := fun s fields want rows =>
  let slots := fun i => NgramSlots.native false want (rows i)
  ⟨eager s (replaceSlots fields slots), slots⟩

def fusedLookup : StageWithLookup := fun s fields want rows =>
  let slots := fun i => NgramSlots.bucketed true want (rows i)
  ⟨replaceSlots (fused s fields) slots, slots⟩

theorem lookup_stage_equal : fusedLookup = separateLookup := by
  funext s fields want rows
  simp [fusedLookup, separateLookup, copy_equal, eager, NgramSlots.bucketed_exact, NgramSlots.native]

theorem fused_pending_i32 (s : Shape) (fields : StaticInputs s) (want : NgramSlots.Want)
    (rows : Fin (extent s .slots) → Int) (i : Fin (extent s .slots)) (fits : want.size < 2 ^ 31) :
    let slot := (fusedLookup s fields want rows).pendingSlots i
    (Int32.ofNat slot).toInt = slot := by
  rw [lookup_stage_equal]
  exact Int32.toInt_ofNat_of_lt (Nat.lt_of_le_of_lt (NgramSlots.native_within want (rows i)) fits)

structure Kernels where
  copy : Copy
  withLookup : StageWithLookup

def staged : Kernels := ⟨eager, separateLookup⟩
def combined : Kernels := ⟨fused, fusedLookup⟩

theorem kernels_equal : combined = staged := by
  simp [combined, staged, copy_equal, lookup_stage_equal]

structure CapturedInputs where
  shape : Shape
  fields : StaticInputs shape

#axiomsClean unique_owner
#axiomsClean replay_copy_exact
#axiomsClean copy_equal
#axiomsClean query_ownership
#axiomsClean lookup_stage_equal
#axiomsClean fused_pending_i32
end VeriTile.Bench.Models.NanoGPT.Track1.GraphStage
