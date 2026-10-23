/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! SM100 packed FP32 register arithmetic, distinct from integer addresses.

/// Two FP32 lanes occupying one 64-bit register, without memory-vector alignment.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct PackedF32x2(u64);

/// Pack scalar lanes without changing NaN payloads or signed zero.
#[must_use]
#[inline(always)]
pub fn pack_f32x2(lo: f32, hi: f32) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::pack_f32x2(lo, hi))
}

/// Recover the low lane without arithmetic conversion.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_lo(value: PackedF32x2) -> f32 {
    cuda_device::f32x2::unpack_f32x2_lo(value.0)
}

/// Recover the high lane without arithmetic conversion.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_hi(value: PackedF32x2) -> f32 {
    cuda_device::f32x2::unpack_f32x2_hi(value.0)
}

/// Add matching FP32 lanes with round-to-nearest-even on SM100+.
///
/// ```compile_fail
/// ctm_rs::f32x2::add_f32x2(0_u64, 0_u64);
/// ```
#[must_use]
#[inline(always)]
pub fn add_f32x2(a: PackedF32x2, b: PackedF32x2) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::add_f32x2(a.0, b.0))
}

/// Multiply matching FP32 lanes with round-to-nearest-even on SM100+.
#[must_use]
#[inline(always)]
pub fn mul_f32x2(a: PackedF32x2, b: PackedF32x2) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::mul_f32x2(a.0, b.0))
}

/// Fused multiply-add each FP32 lane, rounding once per lane, on SM100+.
#[must_use]
#[inline(always)]
pub fn fma_f32x2(a: PackedF32x2, b: PackedF32x2, c: PackedF32x2) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::fma_f32x2(a.0, b.0, c.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Swapped lanes, numeric casts and NaN canonicalization are observable.
    #[test]
    fn packed_lanes_preserve_storage_bits() {
        let patterns = [
            0_u32,
            0x8000_0000,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7f81_2345,
        ];
        for low in patterns {
            for high in patterns {
                let packed = pack_f32x2(f32::from_bits(low), f32::from_bits(high));
                assert_eq!(unpack_f32x2_lo(packed).to_bits(), low);
                assert_eq!(unpack_f32x2_hi(packed).to_bits(), high);
            }
        }
        assert_eq!(core::mem::size_of::<PackedF32x2>(), 8);
        assert_eq!(
            core::mem::align_of::<PackedF32x2>(),
            core::mem::align_of::<u64>()
        );
    }
}
