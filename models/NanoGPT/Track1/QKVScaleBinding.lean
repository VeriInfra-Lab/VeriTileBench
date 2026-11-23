import models.NanoGPT.Track1.QKVScale
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Exact graph and instruction recognition for the scale kernel in the
offline sm90 build. DWARF and source locations do not affect execution. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVScale
open QKVLaunch

private def ptxLine (line : String) : String :=
  let body := ((line.splitOn "//").headD "").replace "\t" " "
  String.intercalate " " ((body.splitOn " ").filter (!·.isEmpty))

private def normalizePTX (text : String) : List String :=
  let lines := (text.splitOn "\n").map ptxLine
  (lines.takeWhile (!·.startsWith ".section .debug_")).filter
    (fun l => !l.isEmpty && !l.startsWith ".loc " && !l.startsWith ".file ")

private def ttirFragment : String := r#"
module {
tt.func public @_qkv_scale_from_partials(%partial_amax: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scales: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant 0.00223214296 : f32
%cst_0 = arith.constant 9.99999996E-13 : f32
%batch = tt.get_program_id x : i32
%parts = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%partials = arith.muli %batch, 32 : i32
%partials_1 = tt.addptr %partial_amax, %partials : !tt.ptr<f32>, i32
%partials_2 = tt.splat %partials_1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
%partials_3 = tt.addptr %partials_2, %parts : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
%partials_4 = tt.load %partials_3 : tensor<32x!tt.ptr<f32>>
%amax = "tt.reduce"(%partials_4) <{axis = 0 : i32}> ({
^bb0(%arg2: f32, %arg3: f32):
%amax_6 = arith.maximumf %arg2, %arg3 : f32
tt.reduce.return %amax_6 : f32
}) : (tensor<32xf32>) -> f32
%floored = arith.cmpf olt, %amax, %cst_0 : f32
%floored_5 = arith.select %floored, %cst_0, %amax : f32
%0 = tt.addptr %scales, %batch : !tt.ptr<f32>, i32
%1 = arith.mulf %floored_5, %cst : f32
tt.store %0, %1 : !tt.ptr<f32>
tt.return
}
}
"#

private def ptxFragment : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _qkv_scale_from_partials(
.param .u64 .ptr .global .align 1 _qkv_scale_from_partials_param_0,
.param .u64 .ptr .global .align 1 _qkv_scale_from_partials_param_1,
.param .u64 .ptr .global .align 1 _qkv_scale_from_partials_param_2,
.param .u64 .ptr .global .align 1 _qkv_scale_from_partials_param_3
)
.reqntid 32
{
.reg .pred %p<3>;
.reg .b32 %r<18>;
.reg .b64 %rd<6>;
$L__func_begin0:
ld.param.b64 %rd3, [_qkv_scale_from_partials_param_0];
ld.param.b64 %rd4, [_qkv_scale_from_partials_param_1];
$L__tmp0:
mov.u32 %r3, %ctaid.x;
mov.u32 %r4, %tid.x;
and.b32 %r5, %r4, 31;
shl.b32 %r6, %r3, 5;
mad.wide.s32 %rd5, %r6, 4, %rd3;
mad.wide.u32 %rd1, %r5, 4, %rd5;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
shfl.sync.bfly.b32 %r7, %r1, 16, 31, -1;
$L__tmp1:
max.NaN.f32 %r8, %r1, %r7;
$L__tmp2:
shfl.sync.bfly.b32 %r9, %r8, 8, 31, -1;
$L__tmp3:
max.NaN.f32 %r10, %r8, %r9;
$L__tmp4:
shfl.sync.bfly.b32 %r11, %r10, 4, 31, -1;
$L__tmp5:
max.NaN.f32 %r12, %r10, %r11;
$L__tmp6:
shfl.sync.bfly.b32 %r13, %r12, 2, 31, -1;
$L__tmp7:
max.NaN.f32 %r14, %r12, %r13;
$L__tmp8:
shfl.sync.bfly.b32 %r15, %r14, 1, 31, -1;
$L__tmp9:
max.NaN.f32 %r16, %r14, %r15;
$L__tmp10:
setp.lt.f32 %p2, %r16, 0f2B8CBCCC;
selp.f32 %r17, 0f2B8CBCCC, %r16, %p2;
mad.wide.u32 %rd2, %r3, 4, %rd4;
mul.f32 %r2, %r17, 0f3B124925;
setp.eq.b32 %p1, %r5, 0;
@%p1 st.global.b32 [ %rd2 + 0 ], { %r2 };
ret;
$L__tmp11:
$L__func_end0:
}
"#

private def fragmentLines (s : String) : List String :=
  (s.splitOn "\n").filter (!·.isEmpty)

def recognizes (ttir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ttir == some (fragmentLines ttirFragment) &&
  normalizePTX ptx == fragmentLines ptxFragment

/-- Interpretation of the recognized kernel's global scale store. -/
def decodeStore (ttir ptx : String) (scratch : Nat → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) : Option (Nat × Scalar) :=
  if recognizes ttir ptx then some (fusedStore scratch w batch) else none

theorem decoded_store_equal (ttir ptx : String) (h : recognizes ttir ptx = true)
    (scratch : Nat → Scalar) (w : Width) (batch : Fin (Layout.ofWidth w).batches) :
    decodeStore ttir ptx scratch w batch = some (referenceStore scratch w batch) := by
  simp [decodeStore, h, store_equal]

structure Certificate (ttir ptx : String) : Type where
  stores : ∀ (scratch : Nat → Scalar) (w : Width) (batch : Fin (Layout.ofWidth w).batches),
    decodeStore ttir ptx scratch w batch = some (referenceStore scratch w batch)

def certify (ttir ptx : String) : Option (Certificate ttir ptx) :=
  if h : recognizes ttir ptx = true then some ⟨decoded_store_equal ttir ptx h⟩ else none

#axiomsClean decoded_store_equal
#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.QKVScale
