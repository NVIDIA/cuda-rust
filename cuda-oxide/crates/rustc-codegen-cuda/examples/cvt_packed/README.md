# cvt_packed

Behavior example for the packed conversion functions in
`cuda_device::convert` (sm_80+).

The first kernel packs the same `(lo = 1.0065, hi = -1.0065)` pair with all
five generated packers (`cvt_f16x2_f32`, `cvt_rz_f16x2_f32`,
`cvt_rn_relu_f16x2_f32`, `cvt_rn_relu_bf16x2_f32`, `cvt_rz_bf16x2_f32`).
The input sits on opposite sides of the nearest and toward-zero results in
both f16 and bf16, so the host checks the exact packed bits rather than a
tolerance that could let the wrong rounding mode pass. A second launch
checks that the ReLU variants canonicalize NaN inputs.

The second kernel round-trips: it packs with the generated
`cvt_f16x2_f32` / `cvt_rz_bf16x2_f32`, then unpacks the words with the
hand-written unpackers `cvt_f32x2_f16x2`, `cvt_f32_f16x2_{lo,hi}`,
`cvt_f32x2_bf16x2`, and `cvt_f32_bf16x2_{lo,hi}`, writing the widened `f32`
values out. Widening is exact in both formats, so the host again compares
exact bits:

- f16 rn: `1.0065 -> 0x3C07 -> 1.0068359375`
- bf16 rz: `1.0065 -> 0x3F80 -> 1.0`

The f16 unpackers are the only place an example runs the runtime
u16-to-f16 transmute + fpext path on the device.

Run with:

```bash
cargo oxide run cvt_packed
```
