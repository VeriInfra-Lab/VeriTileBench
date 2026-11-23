import models.NanoGPT.Track1.ValueAdamHalfProductArithmetic

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

check_binary16_interval fromCode16384 16384
check_binary16_interval fromCode17408 17408
check_binary16_interval fromCode18432 18432
check_binary16_interval fromCode19456 19456
check_binary16_interval fromCode20480 20480
check_binary16_interval fromCode21504 21504
check_binary16_interval fromCode22528 22528
check_binary16_interval fromCode23552 23552
check_binary16_interval fromCode24576 24576
check_binary16_interval fromCode25600 25600
check_binary16_interval fromCode26624 26624
check_binary16_interval fromCode27648 27648
check_binary16_interval fromCode28672 28672
check_binary16_interval fromCode29696 29696
check_binary16_interval fromCode30720 30720
check_binary16_interval fromCode31744 31744

private theorem checkBlock16384 : HalfProductSlice 16384 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode16384 fromCode17408) (HalfProductSlice.append fromCode18432 fromCode19456))

private theorem checkBlock20480 : HalfProductSlice 20480 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode20480 fromCode21504) (HalfProductSlice.append fromCode22528 fromCode23552))

private theorem checkBlock24576 : HalfProductSlice 24576 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode24576 fromCode25600) (HalfProductSlice.append fromCode26624 fromCode27648))

private theorem checkBlock28672 : HalfProductSlice 28672 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode28672 fromCode29696) (HalfProductSlice.append fromCode30720 fromCode31744))

theorem halfProductCodes1 : HalfProductSlice 16384 16384 :=
  (HalfProductSlice.append (HalfProductSlice.append checkBlock16384 checkBlock20480) (HalfProductSlice.append checkBlock24576 checkBlock28672))

#axiomsClean halfProductCodes1

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
