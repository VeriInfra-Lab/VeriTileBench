import models.NanoGPT.Track1.ValueAdamHalfProductArithmetic

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

check_binary16_interval fromCode32768 32768
check_binary16_interval fromCode33792 33792
check_binary16_interval fromCode34816 34816
check_binary16_interval fromCode35840 35840
check_binary16_interval fromCode36864 36864
check_binary16_interval fromCode37888 37888
check_binary16_interval fromCode38912 38912
check_binary16_interval fromCode39936 39936
check_binary16_interval fromCode40960 40960
check_binary16_interval fromCode41984 41984
check_binary16_interval fromCode43008 43008
check_binary16_interval fromCode44032 44032
check_binary16_interval fromCode45056 45056
check_binary16_interval fromCode46080 46080
check_binary16_interval fromCode47104 47104
check_binary16_interval fromCode48128 48128

private theorem checkBlock32768 : HalfProductSlice 32768 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode32768 fromCode33792) (HalfProductSlice.append fromCode34816 fromCode35840))

private theorem checkBlock36864 : HalfProductSlice 36864 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode36864 fromCode37888) (HalfProductSlice.append fromCode38912 fromCode39936))

private theorem checkBlock40960 : HalfProductSlice 40960 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode40960 fromCode41984) (HalfProductSlice.append fromCode43008 fromCode44032))

private theorem checkBlock45056 : HalfProductSlice 45056 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode45056 fromCode46080) (HalfProductSlice.append fromCode47104 fromCode48128))

theorem halfProductCodes2 : HalfProductSlice 32768 16384 :=
  (HalfProductSlice.append (HalfProductSlice.append checkBlock32768 checkBlock36864) (HalfProductSlice.append checkBlock40960 checkBlock45056))

#axiomsClean halfProductCodes2

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
