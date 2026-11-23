import models.NanoGPT.Track1.ValueAdamReadScale

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

def readOR (raw : Nat → Half) : List Nat → Half
  | [] => 0
  | i :: rest => raw i ||| readOR raw rest

theorem read_or_zero (raw : Nat → Half) (inputs : List Nat) :
    readOR raw inputs = 0 ↔ ∀ i ∈ inputs, raw i = 0 := by
  induction inputs with
  | nil => simp [readOR]
  | cons word rest ih =>
    simp only [readOR, UInt16.or_eq_zero_iff, ih, List.mem_cons]
    constructor
    · rintro ⟨head, tail⟩ i (rfl | member)
      · exact head
      · exact tail i member
    · intro values
      exact ⟨values word (.inl rfl), fun i member => values i (.inr member)⟩

inductive ClearWrite where
  | always
  | ifDirty (sources : List Nat)
  deriving DecidableEq, Inhabited

/-- A vector store may skip one lane precisely when the untouched word is
already positive zero. The predicate must inspect the word it may skip. -/
def ClearWrite.covers (write : ClearWrite) (index : Nat) : Prop :=
  match write with
  | .always => True
  | .ifDirty words => index ∈ words

instance (write : ClearWrite) (index : Nat) : Decidable (write.covers index) := by
  cases write <;> unfold ClearWrite.covers <;> infer_instance

def ClearWrite.apply (write : ClearWrite) (raw : Nat → Half) (index : Nat) : Half :=
  match write with
  | .always => 0
  | .ifDirty words => if readOR raw words = 0 then raw index else 0

theorem ClearWrite.zeros (write : ClearWrite) (raw : Nat → Half) (index : Nat)
    (covered : write.covers index) : write.apply raw index = 0 := by
  cases write with
  | always => rfl
  | ifDirty sources =>
    change (if readOR raw sources = 0 then raw index else 0) = 0
    split
    · rename_i zero
      exact (read_or_zero raw sources).mp zero index covered
    · rfl

/-- Unsigned halfwords include both signs of zero, so skipping a negative
zero would fail the same operational postcondition. -/
theorem negative_zero_is_cleared :
    ClearWrite.apply (.ifDirty [0]) (fun _ => 0x8000) 0 = 0 := by
  exact ClearWrite.zeros _ _ _ (by simp [ClearWrite.covers])

def ternaryOrLUT (a b c : Bool) : Bool :=
  Nat.testBit 0xfe (4 * a.toNat + 2 * b.toNat + c.toNat)

theorem ternary_or_lut (a b c : Bool) :
    ternaryOrLUT a b c = (a || b || c) := by
  cases a <;> cases b <;> cases c <;> decide +kernel

def referenceWorkspace (previous : PriorMarked) (index : Fin (25152 * 768)) : Half :=
  (clearIndexedLine previous (index.val / 768)).1
    ⟨index.val % 768, Nat.mod_lt _ (by decide)⟩

/-- On the next value Adam event the raw workspace can start from the
same positive-zero contents as the indexed clear. This theorem allows a
different proved gradient predicate for each of the emitted stores. -/
theorem consumed_workspace_eq (previous : PriorMarked) (raw : Nat → Half)
    (writes : Fin (25152 * 768) → ClearWrite)
    (covered : ∀ i, (writes i).covers i.val) :
    (fun i => (writes i).apply raw i.val) = referenceWorkspace previous := by
  funext index
  rw [ClearWrite.zeros _ _ _ (covered index)]
  unfold referenceWorkspace
  rw [← clear_raw_line_equal, clear_raw_line_zero]

#axiomsClean read_or_zero
#axiomsClean ClearWrite.zeros
#axiomsClean negative_zero_is_cleared
#axiomsClean ternary_or_lut
#axiomsClean consumed_workspace_eq

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
