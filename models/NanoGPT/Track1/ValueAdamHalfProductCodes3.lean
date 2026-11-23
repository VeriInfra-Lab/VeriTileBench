import models.NanoGPT.Track1.ValueAdamHalfProductArithmetic

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

check_binary16_interval fromCode49152 49152
check_binary16_interval fromCode50176 50176
check_binary16_interval fromCode51200 51200
check_binary16_interval fromCode52224 52224
check_binary16_interval fromCode53248 53248
check_binary16_interval fromCode54272 54272
check_binary16_interval fromCode55296 55296
check_binary16_interval fromCode56320 56320
check_binary16_interval fromCode57344 57344
check_binary16_interval fromCode58368 58368
check_binary16_interval fromCode59392 59392
check_binary16_interval fromCode60416 60416
check_binary16_interval fromCode61440 61440
check_binary16_interval fromCode62464 62464
check_binary16_interval fromCode63488 63488
check_binary16_interval fromCode64512 64512

private theorem checkBlock49152 : HalfProductSlice 49152 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode49152 fromCode50176) (HalfProductSlice.append fromCode51200 fromCode52224))

private theorem checkBlock53248 : HalfProductSlice 53248 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode53248 fromCode54272) (HalfProductSlice.append fromCode55296 fromCode56320))

private theorem checkBlock57344 : HalfProductSlice 57344 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode57344 fromCode58368) (HalfProductSlice.append fromCode59392 fromCode60416))

private theorem checkBlock61440 : HalfProductSlice 61440 4096 :=
  (HalfProductSlice.append (HalfProductSlice.append fromCode61440 fromCode62464) (HalfProductSlice.append fromCode63488 fromCode64512))

theorem halfProductCodes3 : HalfProductSlice 49152 16384 :=
  (HalfProductSlice.append (HalfProductSlice.append checkBlock49152 checkBlock53248) (HalfProductSlice.append checkBlock57344 checkBlock61440))

#axiomsClean halfProductCodes3

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
