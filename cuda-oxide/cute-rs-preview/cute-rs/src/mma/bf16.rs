/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Typed BF16 m16n8k16 operand partitions and FP32 accumulation.
//!
//! Static thread/value layouts determine every operand and result coordinate.
//! Load callbacks can describe ordinary matrices or virtual im2col views.
//! CTM supplies only the final instruction and packed physical loads.
use crate::markers::{C, Layout};
use crate::numeric::BFloat16;
use crate::static_layout::StaticLayout;
use core::marker::PhantomData;

type Threads = (C<4>, C<8>);
/// A's `(lane, scalar value)` map into a row-major 16-by-16 logical tile.
pub type ALayout = Layout<(Threads, ((C<2>, C<2>), C<2>)), ((C<2>, C<16>), ((C<1>, C<128>), C<8>))>;
/// B's `(lane, scalar value)` map into the stored 8-by-16 `(N,K)` tile.
pub type BLayout = Layout<(Threads, (C<2>, C<2>)), ((C<2>, C<16>), (C<1>, C<8>))>;
/// C's `(lane, register)` map into a row-major 16-by-8 logical tile.
pub type CLayout = Layout<(Threads, (C<2>, C<2>)), ((C<2>, C<8>), (C<1>, C<64>))>;

/// This lane's BF16 A registers, whose layout is established by its partition.
#[derive(Clone, Copy)]
pub struct FragmentA(ctm_rs::mma::Bf16A);
/// This lane's BF16 B registers, distinct from an A operand.
#[derive(Clone, Copy)]
pub struct FragmentB(ctm_rs::mma::Bf16B);
/// This lane's FP32 accumulator in the atom's C layout.
#[derive(Clone, Copy)]
pub struct FragmentC(ctm_rs::mma::Accumulator16x8);
impl FragmentC {
    /// Positive FP32 zero in every register, for starting one warp product.
    pub const ZERO: Self = Self(ctm_rs::mma::Accumulator16x8::zero());
}

/// One lane's handle to the BF16 16-by-8-by-16 matrix operation.
#[derive(Clone, Copy)]
pub struct Bf16Mma16x8x16 {
    lane: usize,
}
impl Bf16Mma16x8x16 {
    /// Select a lane in the 32-thread atom. Device use must pass the actual lane.
    #[inline(always)]
    pub fn get_slice(lane: usize) -> Self {
        assert!(lane < 32);
        Self { lane }
    }
    /// Partition A's sixteen rows and sixteen reduction coordinates.
    #[inline(always)]
    pub fn partition_a(self) -> APartition {
        APartition { lane: self.lane }
    }
    /// Partition B's eight stored output rows and sixteen reduction coordinates.
    #[inline(always)]
    pub fn partition_b(self) -> BPartition {
        BPartition { lane: self.lane }
    }
    /// Partition the sixteen-by-eight result tile.
    #[inline(always)]
    pub fn partition_c(self) -> CPartition {
        CPartition { lane: self.lane }
    }
    /// Accumulate one BF16 product, with no conversion or reassociation.
    ///
    /// # Safety
    /// All 32 lanes must execute together on SM80 or newer, using their actual
    /// lane handles and A/B/C fragments of the same logical warp product.
    #[must_use]
    #[inline(always)]
    pub unsafe fn mma_atom_call(self, c: FragmentC, a: FragmentA, b: FragmentB) -> FragmentC {
        FragmentC(unsafe { ctm_rs::mma::mma_bf16(c.0, a.0, b.0) })
    }
}
/// A lane's immutable partition of the A operand.
#[derive(Clone, Copy)]
pub struct APartition {
    lane: usize,
}
/// A lane's immutable partition of the B operand.
#[derive(Clone, Copy)]
pub struct BPartition {
    lane: usize,
}
/// A lane's partition of the accumulator coordinates.
#[derive(Clone, Copy)]
pub struct CPartition {
    lane: usize,
}

/// Borrowed storage proof for one/two-row BF16 decode with complete K/N tiles.
///
/// Construction checks bounds, even row pitches and four-byte alignment once.
/// The borrow retains operand storage; it does not establish GPU readiness.
pub struct PackedDecode<'a> {
    // Raw carriers admit null as a bit pattern, so Option<Self> has an ordinary
    // discriminant supported by the device enum ABI. Only the checked borrowed
    // constructor creates this private payload; the phantom retains both loans.
    input: *const BFloat16,
    weight: *const BFloat16,
    rows: usize,
    input_stride: usize,
    weight_stride: usize,
    borrow: PhantomData<&'a [BFloat16]>,
}
impl<'a> PackedDecode<'a> {
    /// Validate the storage needed by packed decode operand loads.
    #[must_use]
    #[inline(always)]
    pub fn new(
        input: &'a [BFloat16],
        weight: &'a [BFloat16],
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) -> Option<Self> {
        if Self::supported(
            input,
            weight,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        ) {
            Some(Self {
                input: input.as_ptr(),
                weight: weight.as_ptr(),
                rows,
                input_stride,
                weight_stride,
                borrow: PhantomData,
            })
        } else {
            None
        }
    }

    /// Borrow a validated packed view inside the successful validation branch.
    ///
    /// Returns whether validation succeeded. The callback runs exactly once on
    /// success and never on failure; its borrowed proof cannot leave the call.
    /// Construction and use stay in the same branch, so device code need not
    /// merge an optional aggregate containing the operand descriptors.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn with_validated(
        input: &'a [BFloat16],
        weight: &'a [BFloat16],
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
        use_proof: impl FnOnce(&Self),
    ) -> bool {
        if Self::supported(
            input,
            weight,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        ) {
            let source = Self {
                input: input.as_ptr(),
                weight: weight.as_ptr(),
                rows,
                input_stride,
                weight_stride,
                borrow: PhantomData,
            };
            use_proof(&source);
            true
        } else {
            false
        }
    }

    #[inline(always)]
    fn supported(
        input: &[BFloat16],
        weight: &[BFloat16],
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) -> bool {
        matches!(rows, 1 | 2)
            && width != 0
            && width.is_multiple_of(16)
            && outputs != 0
            && outputs.is_multiple_of(8)
            && input_stride >= width
            && weight_stride >= width
            && input_stride.is_multiple_of(2)
            && weight_stride.is_multiple_of(2)
            && (input.as_ptr() as usize).is_multiple_of(4)
            && (weight.as_ptr() as usize).is_multiple_of(4)
            && rows <= input.len() / input_stride
            && outputs <= weight.len() / weight_stride
    }
}

#[inline(always)]
fn coordinate<L: StaticLayout, const COLS: usize>(lane: usize, value: usize) -> (usize, usize) {
    let offset = L::crd2idx(lane + 32 * value);
    (offset / COLS, offset % COLS)
}
impl APartition {
    /// Load this lane's eight BF16 elements, preserving low/high pair order.
    /// The callback receives local `(M,K)` coordinates in register order.
    #[inline(always)]
    pub fn load_with(self, mut read: impl FnMut(usize, usize) -> BFloat16) -> FragmentA {
        let mut pair = |i| {
            let (r, c) = coordinate::<ALayout, 16>(self.lane, i);
            let low = read(r, c).to_bits();
            let (r, c) = coordinate::<ALayout, 16>(self.lane, i + 1);
            u32::from(low) | (u32::from(read(r, c).to_bits()) << 16)
        };
        FragmentA(ctm_rs::mma::Bf16A::from_registers([
            pair(0),
            pair(2),
            pair(4),
            pair(6),
        ]))
    }

    /// Load a complete decode K fragment, repeating valid row addresses and
    /// zero-masking absent logical rows without divergent loads.
    ///
    /// # Safety
    /// `k` must be a multiple of 16 selecting a complete K tile within the
    /// width passed to `PackedDecode::new` or `PackedDecode::with_validated`.
    /// Operand data must be ready to read.
    #[inline(always)]
    pub unsafe fn load_decode_packed(self, source: &PackedDecode<'_>, k: usize) -> FragmentA {
        let (row, low_k) = coordinate::<ALayout, 16>(self.lane, 0);
        let (_, high_k) = coordinate::<ALayout, 16>(self.lane, 4);
        let physical_row = row & (source.rows - 1);
        let mask = 0u32.wrapping_sub((row < source.rows) as u32);
        let base = source.input;
        let low = unsafe {
            ctm_rs::memory::load_aligned_u32(
                base.add(physical_row * source.input_stride + k + low_k)
                    .cast(),
            )
        };
        let high = unsafe {
            ctm_rs::memory::load_aligned_u32(
                base.add(physical_row * source.input_stride + k + high_k)
                    .cast(),
            )
        };
        FragmentA(ctm_rs::mma::Bf16A::from_registers([
            low & mask,
            0,
            high & mask,
            0,
        ]))
    }
}
impl BPartition {
    /// Load this lane's four BF16 elements from local stored `(N,K)` coordinates.
    #[inline(always)]
    pub fn load_with(self, mut read: impl FnMut(usize, usize) -> BFloat16) -> FragmentB {
        let mut pair = |i| {
            let (r, c) = coordinate::<BLayout, 16>(self.lane, i);
            let low = read(r, c).to_bits();
            let (r, c) = coordinate::<BLayout, 16>(self.lane, i + 1);
            u32::from(low) | (u32::from(read(r, c).to_bits()) << 16)
        };
        FragmentB(ctm_rs::mma::Bf16B::from_registers([pair(0), pair(2)]))
    }

    /// Load one complete eight-output decode weight tile with packed reads.
    ///
    /// # Safety
    /// `n` and `k` must select complete N=8 and K=16 tiles, aligned to those
    /// dimensions and within the extents passed to `PackedDecode::new` or
    /// `PackedDecode::with_validated`.
    /// Operand data must be ready to read.
    #[inline(always)]
    pub unsafe fn load_decode_packed(
        self,
        source: &PackedDecode<'_>,
        n: usize,
        k: usize,
    ) -> FragmentB {
        let (row, low_k) = coordinate::<BLayout, 16>(self.lane, 0);
        let (_, high_k) = coordinate::<BLayout, 16>(self.lane, 2);
        let base = source.weight;
        let low = unsafe {
            ctm_rs::memory::load_aligned_u32(
                base.add((n + row) * source.weight_stride + k + low_k)
                    .cast(),
            )
        };
        let high = unsafe {
            ctm_rs::memory::load_aligned_u32(
                base.add((n + row) * source.weight_stride + k + high_k)
                    .cast(),
            )
        };
        FragmentB(ctm_rs::mma::Bf16B::from_registers([low, high]))
    }
}
impl CPartition {
    /// Visit each accumulator once in register order with its local `(M,N)`.
    /// The callback applies the explicit epilogue and output predication.
    #[inline(always)]
    pub fn store_with(self, accumulator: FragmentC, mut write: impl FnMut(usize, usize, f32)) {
        let values = accumulator.0.registers();
        let mut i = 0;
        while i < 4 {
            let (r, c) = coordinate::<CLayout, 8>(self.lane, i);
            write(r, c, values[i]);
            i += 1;
        }
    }

    /// Visit only initialized decode result rows, in the original register order.
    #[inline(always)]
    pub fn store_decode_with(
        self,
        accumulator: FragmentC,
        source: &PackedDecode<'_>,
        mut write: impl FnMut(usize, usize, f32),
    ) {
        let (row, _) = coordinate::<CLayout, 8>(self.lane, 0);
        if row < source.rows {
            let values = accumulator.0.registers();
            let mut i = 0;
            while i < 2 {
                let (r, c) = coordinate::<CLayout, 8>(self.lane, i);
                write(r, c, values[i]);
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Swapping pair halves or the upper A row band changes these PTX fixtures.
    #[test]
    fn operand_storage_preserves_bf16_bits_and_register_order() {
        let a = Bf16Mma16x8x16::get_slice(0)
            .partition_a()
            .load_with(|r, k| BFloat16::from_bits(0x8000 + (r * 16 + k) as u16));
        assert_eq!(
            a.0.registers(),
            [0x8001_8000, 0x8081_8080, 0x8009_8008, 0x8089_8088]
        );
        let b = Bf16Mma16x8x16::get_slice(31)
            .partition_b()
            .load_with(|n, k| BFloat16::from_bits(0x8000 + (n * 16 + k) as u16));
        assert_eq!(b.0.registers(), [0x8077_8076, 0x807f_807e]);
    }

    /// Result mapping must preserve every FP32 payload until the explicit epilogue.
    #[test]
    fn result_partition_preserves_values_including_special_bits() {
        let bits = [0x8000_0000, 0x7fc1_2345, 0x0000_0001, 0xff80_0000];
        let acc = FragmentC(ctm_rs::mma::Accumulator16x8::from_registers(
            bits.map(f32::from_bits),
        ));
        let mut result = [(0, 0, 0); 4];
        let mut index = 0;
        Bf16Mma16x8x16::get_slice(31)
            .partition_c()
            .store_with(acc, |r, c, value| {
                result[index] = (r, c, value.to_bits());
                index += 1;
            });
        assert_eq!(
            result,
            [
                (7, 6, bits[0]),
                (7, 7, bits[1]),
                (15, 6, bits[2]),
                (15, 7, bits[3])
            ]
        );
    }
}
