import models.NanoGPT.Track1.ValueAdamHalfProductArithmetic

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

check_binary16_interval fromCode0 0
check_binary16_interval fromCode1024 1024
check_binary16_interval fromCode2048 2048
check_binary16_interval fromCode3072 3072
check_binary16_interval fromCode4096 4096
check_binary16_interval fromCode5120 5120
check_binary16_interval fromCode6144 6144
check_binary16_interval fromCode7168 7168
check_binary16_interval fromCode8192 8192
check_binary16_interval fromCode9216 9216
check_binary16_interval fromCode10240 10240
check_binary16_interval fromCode11264 11264
check_binary16_interval fromCode12288 12288
check_binary16_interval fromCode13312 13312
check_binary16_interval fromCode14336 14336
check_binary16_interval fromCode15360 15360

private theorem checkBlock0 : HalfProductSlice 0 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode0 fromCode1024) (HalfProductSlice.append fromCode2048 fromCode3072))

private theorem checkBlock4096 : HalfProductSlice 4096 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode4096 fromCode5120) (HalfProductSlice.append fromCode6144 fromCode7168))

private theorem checkBlock8192 : HalfProductSlice 8192 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode8192 fromCode9216) (HalfProductSlice.append fromCode10240 fromCode11264))

private theorem checkBlock12288 : HalfProductSlice 12288 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode12288 fromCode13312) (HalfProductSlice.append fromCode14336 fromCode15360))

theorem halfProductCodes0 : HalfProductSlice 0 16384 :=
  (HalfProductSlice.append (HalfProductSlice.append checkBlock0 checkBlock4096) (HalfProductSlice.append checkBlock8192 checkBlock12288))

#axiomsClean halfProductCodes0

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
