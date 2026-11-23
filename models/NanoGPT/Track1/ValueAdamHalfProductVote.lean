import models.NanoGPT.Track1.ValueAdamReadScale

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise.GuardedHalf

structure HalfPair (α : Type) where
  low : α
  high : α

structure Lane (α : Type) where
  first : HalfPair α
  second : HalfPair α
  third : HalfPair α
  fourth : HalfPair α

def HalfPair.map (f : α → β) (pair : HalfPair α) : HalfPair β :=
  ⟨f pair.low, f pair.high⟩

def Lane.map (f : α → β) (lane : Lane α) : Lane β :=
  ⟨lane.first.map f, lane.second.map f, lane.third.map f, lane.fourth.map f⟩

def Lane.words (lane : Lane α) : List α :=
  [lane.first.low, lane.first.high, lane.second.low, lane.second.high, lane.third.low,
    lane.third.high, lane.fourth.low, lane.fourth.high]

theorem Lane.words_map (f : α → β) (lane : Lane α) :
    (lane.map f).words = lane.words.map f := rfl

/-- The classifier tracks which packed halves are NaN. The operand selected
by max on two numeric values belongs to the numeric class in either order. -/
private def HalfPair.classes (isNaN : Half → Bool) (pair : HalfPair Half) : Bool × Bool :=
  (isNaN pair.low, isNaN pair.high)

private def maxNaNClasses (a b : Bool × Bool) : Bool × Bool :=
  (a.1 || b.1, a.2 || b.2)

/-- The three packed max.NaN operations feed set.nan, followed by a nonzero
test on the two lane masks. The result is the ballot input of this lane. -/
def Lane.ballotInput (isNaN : Half → Bool) (lane : Lane Half) : Bool :=
  let lowWords := maxNaNClasses (lane.first.classes isNaN) (lane.second.classes isNaN)
  let highWords := maxNaNClasses (lane.third.classes isNaN) (lane.fourth.classes isNaN)
  let flags := maxNaNClasses lowWords highWords
  flags.1 || flags.2

theorem Lane.ballot_input_exact (isNaN : Half → Bool) (lane : Lane Half) :
    lane.ballotInput isNaN = lane.words.any isNaN := by
  simp [Lane.ballotInput, Lane.words, HalfPair.classes, maxNaNClasses,
    Bool.or_left_comm, Bool.or_comm]

/-- With the full member mask, all 32 lanes see the same vote and that vote
is true exactly when at least one source half in the warp is NaN. -/
def warpVote (isNaN : Half → Bool) (warp : Fin 32 → Lane Half) (_lane : Fin 32) : Bool :=
  (List.finRange 32).any fun lane => (warp lane).ballotInput isNaN

theorem warp_vote_exact (isNaN : Half → Bool) (warp : Fin 32 → Lane Half) (lane : Fin 32) :
    warpVote isNaN warp lane = ((List.finRange 32).flatMap (Lane.words ∘ warp)).any isNaN := by
  simp [warpVote, Lane.ballot_input_exact, Function.comp_def]

theorem warp_vote_uniform (isNaN : Half → Bool) (warp : Fin 32 → Lane Half)
    (first second : Fin 32) :
    warpVote isNaN warp first = warpVote isNaN warp second := rfl

#axiomsClean Lane.ballot_input_exact
#axiomsClean warp_vote_exact
#axiomsClean warp_vote_uniform

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise.GuardedHalf
