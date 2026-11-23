import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.Domain
import Mathlib.Tactic.IntervalCases

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
/-- Check the eight 1024-word blocks starting at BF16 word 0x4000. -/
theorem checked4000 (hi : Nat) (ordinal : hi / 8 = 2) : blockPasses hi = true := by
  have lower : 16 ≤ hi := by omega
  have upper : hi < 24 := by omega
  interval_cases hi <;> decide +kernel

end VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic
