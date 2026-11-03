/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Typed dense FP16 descriptors using the canonical instruction encoders.
use super::tmem::{TmemF32, TmemFormat};
use crate::{
    groups::{CtaGroup, CtaGroup2},
    memory::*,
};
use core::{cell::UnsafeCell, marker::PhantomData};
use cuda_device::tcgen05::{
    Tcgen05InstructionDescriptor, Tcgen05SmemDescriptor, Tcgen05SwizzleMode,
};

/// Restricts MMA specifications, storage formats, operand roles and K slices.
mod sealed {
    /// Implemented only for supported dense FP16 instruction configurations.
    pub trait Spec {}
    /// Marks physical FP16 layouts accepted as shared MMA inputs.
    pub trait Input {}
    /// Marks the distinct A and B instruction operand roles.
    pub trait Role {}
    /// Admits one compile-time K16 slice index within a K64 stage.
    pub trait Step<const N: u32> {}
}
/// The shared operand's K coordinate is contiguous before swizzling.
///
/// This descriptor marker is distinct from cute-rs's `KMajor<Mode>`
/// (`cute_rs::block_scaled::KMajor`), a block-scaled operand layout that also
/// carries row and K extents.
pub enum KMajor {}
/// Left matrix input; cannot be substituted for [`OperandB`].
///
/// cute-rs reexports this marker as `cute_rs::sm100::OperandA` (also at its
/// crate root), so shared tiles there and instruction operands here share one
/// role vocabulary.
pub enum OperandA {}
/// Right matrix input; cannot be substituted for [`OperandA`].
///
/// cute-rs reexports this marker as `cute_rs::sm100::OperandB` (also at its
/// crate root), so shared tiles there and instruction operands here share one
/// role vocabulary.
pub enum OperandB {}
impl sealed::Role for OperandA {}
impl sealed::Role for OperandB {}
/// Sealed distinction between left and right shared MMA operands.
pub trait OperandRole: sealed::Role {}
impl OperandRole for OperandA {}
impl OperandRole for OperandB {}
/// Supported SM100 group-two FP16 MMA configuration.
///
/// Implemented only for `G = CtaGroup2`, `M = 256`, `N = 192` or `160`, and
/// both majors [`KMajor`]. M/N count the collective output, while A and B
/// occupy `(128, 64)` and `(N / 2, 64)` FP16 elements per CTA. Each issue
/// selects one K16 slice of those K64 shared stages with [`SmemOperand::k16`].
///
/// ```
/// use ctm_rs::{groups::CtaGroup2, tcgen05::descriptors::*};
/// type Tile = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
/// let descriptor = InstructionDescriptor::<Tile>::new();
/// ```
pub struct DenseSharedF16<G, const M: u32, const N: u32, AMajor, BMajor>(
    #[doc = "Records CTA group and operand-major choices without runtime data."]
    PhantomData<(G, AMajor, BMajor)>,
);
/// A supported shared FP16 tile usable by the descriptor encoder.
pub trait MmaInputFormat: PhysicalFormat<Element = f16> + sealed::Input {}
/// Register shared FP16 formats that the descriptor encoder can represent.
macro_rules! input { ($($f:ty),*) => { $(impl sealed::Input for $f {} impl MmaInputFormat for $f {})* }; }
input!(F16Tile128, F16TileB96, F16TileB80);
/// One supported combination of instruction bits and typed operand formats.
pub trait MmaSpec: sealed::Spec {
    /// CTA group used by every operation touching the accumulator.
    type Group: CtaGroup;
    /// Physical shared layout of one CTA’s A operand.
    type AFormat: MmaInputFormat;
    /// Physical shared layout of one CTA’s B operand.
    type BFormat: MmaInputFormat;
    /// FP32 tensor-memory shape of the collective result.
    type DstFormat: TmemFormat;
    /// Canonical packed hardware instruction bits for this configuration.
    const DESCRIPTOR: Tcgen05InstructionDescriptor;
}
/// Define a supported collective N shape and its local B storage format.
macro_rules! spec {
    ($n:literal,$b:ty) => {
        impl sealed::Spec for DenseSharedF16<CtaGroup2, 256, $n, KMajor, KMajor> {}
        impl MmaSpec for DenseSharedF16<CtaGroup2, 256, $n, KMajor, KMajor> {
            type Group = CtaGroup2;
            type AFormat = F16Tile128;
            type BFormat = $b;
            type DstFormat = TmemF32<256, $n>;
            const DESCRIPTOR: Tcgen05InstructionDescriptor =
                Tcgen05InstructionDescriptor::builder().m(256).n($n).build();
        }
    };
}
spec!(192, F16TileB96);
spec!(160, F16TileB80);

/// Instruction bits for one supported [`DenseSharedF16`] configuration.
pub struct InstructionDescriptor<S: MmaSpec> {
    /// Canonical packed instruction descriptor for S.
    raw: Tcgen05InstructionDescriptor,
    /// Preserves the group, shape and operand-format specification in the type.
    spec: PhantomData<S>,
}
impl<S: MmaSpec> InstructionDescriptor<S> {
    /// Use the static instruction descriptor selected by S.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            raw: S::DESCRIPTOR,
            spec: PhantomData,
        }
    }
    /// Return the instruction’s packed 32-bit descriptor.
    #[inline(always)]
    pub const fn raw(&self) -> u32 {
        self.raw.raw()
    }
}
impl<S: MmaSpec> Default for InstructionDescriptor<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a shared address cannot be represented by this operand descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorError {
    /// The address violates descriptor or physical-format alignment.
    Misaligned,
    /// The shared byte offset exceeds the descriptor address field.
    AddressOutOfRange,
}
#[inline(always)]
/// Pack a 16-byte-aligned shared offset into a no-swizzle K-major descriptor (`Tcgen05SwizzleMode::None`).
/// Encodes `address >> 4`, `leading_dim_bytes >> 4`, and `stride_bytes >> 4` with the canonical
/// bit-46 descriptor flag for `tcgen05.mma` and `tcgen05.cp`.
#[must_use]
pub const fn encode_no_swizzle_smem_desc(
    address: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
) -> u64 {
    Tcgen05SmemDescriptor::builder()
        .address(address)
        .leading_dim_bytes(leading_dim_bytes)
        .stride_bytes(stride_bytes)
        .swizzle(Tcgen05SwizzleMode::None)
        .build()
        .raw()
}

#[inline(always)]
/// Encode a group-one dense FP16 instruction descriptor (`Tcgen05InstructionDescriptor`).
/// Configures `M = m` and `N = n` with FP16 inputs and FP32 TMEM accumulation.
#[must_use]
pub const fn encode_cg1_f16_idesc(m: u32, n: u32) -> u32 {
    Tcgen05InstructionDescriptor::builder()
        .m(m)
        .n(n)
        .build()
        .raw()
}

#[inline(always)]
/// Encode a group-one NVFP4 (`mxf4nvf4.block_scale.block16`, `E2M1 × E2M1` with `E4M3` scales)
/// instruction descriptor (`Tcgen05InstructionDescriptor`) for tile dimensions `(m, n)`.
#[must_use]
pub const fn encode_cg1_mxf4nvf4_block16_idesc(m: u32, n: u32) -> u32 {
    0x0000_0480u32 | ((n >> 3) << 17) | ((m >> 4) << 24)
}

#[inline(always)]
/// Pack a 16-byte-aligned shared offset into the fixed B128 K-major descriptor.
fn encode_smem_offset(offset: ByteOffset) -> Result<Tcgen05SmemDescriptor, DescriptorError> {
    if !offset.bytes().is_multiple_of(16) {
        return Err(DescriptorError::Misaligned);
    }
    if offset.bytes() > 0x3fff0 {
        return Err(DescriptorError::AddressOutOfRange);
    }
    Ok(Tcgen05SmemDescriptor::builder()
        .address(offset.bytes() as u64)
        .leading_dim_bytes(16)
        .stride_bytes(1024)
        .swizzle(Tcgen05SwizzleMode::Swizzle128B)
        .build())
}
/// Sealed proof that a K16 slice index is in 0..4.
pub trait ValidKStep<const N: u32>: sealed::Step<N> {}
/// Admit only the four K16 slice indices in a K64 shared stage.
macro_rules! step { ($($n:literal),*) => { $(impl sealed::Step<$n> for () {} impl ValidKStep<$n> for () {})* }; }
step!(0, 1, 2, 3);

/// A role-specific descriptor borrowing physical shared storage.
pub struct SmemOperand<'s, R: OperandRole, F: MmaInputFormat> {
    /// Original K64-stage descriptor, retained across repeated slice selection.
    stage: Tcgen05SmemDescriptor,
    /// Descriptor for the currently selected K16 slice.
    selected: Tcgen05SmemDescriptor,
    /// Keeps the shared allocation borrowed through asynchronous operand use.
    storage: PhantomData<&'s UnsafeCell<f16>>,
    /// Records A/B role and physical format without runtime data.
    marker: PhantomData<(R, F)>,
}
impl<'s, R: OperandRole, F: MmaInputFormat> SmemOperand<'s, R, F> {
    /// Borrow a K64 stage after checking its shared address and base alignment.
    /// Does not publish writes or wait for copies; the MMA issuer coordinates those.
    #[inline(always)]
    pub fn from_region(region: &SharedRegion<'s, F>) -> Result<Self, DescriptorError> {
        let offset = unsafe {
            cuda_device::shared::cvta_generic_to_shared_offset(region.address().as_raw().cast())
        };
        if offset > 0x3ffff {
            return Err(DescriptorError::AddressOutOfRange);
        }
        if offset % F::BASE_ALIGNMENT as u64 != 0 {
            return Err(DescriptorError::Misaligned);
        }
        let stage = encode_smem_offset(ByteOffset::new(offset as usize))?;
        Ok(Self {
            stage,
            selected: stage,
            storage: PhantomData,
            marker: PhantomData,
        })
    }
    /// Select one K16 slice relative to the original K64 stage, even on a slice.
    #[inline(always)]
    pub fn k16<const STEP: u32>(&self) -> Self
    where
        (): ValidKStep<STEP>,
    {
        // A 1024-aligned, representable stage leaves room for all four slices
        // within the address field. Only that field changes, in 16-byte units.
        Self {
            stage: self.stage,
            selected: Tcgen05SmemDescriptor::from_raw(self.stage.raw() + STEP as u64 * 2),
            storage: PhantomData,
            marker: PhantomData,
        }
    }
    #[inline(always)]
    /// Return the selected slice descriptor in the hardware instruction encoding.
    pub(crate) fn raw(&self) -> u64 {
        self.selected.raw()
    }
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::groups::CtaGroup2;
    /// Supported instruction and shared-address descriptors must match known hardware bit encodings.
    #[test]
    fn instruction_and_smem_descriptors_match_canonical_bits() {
        /// Supported group-two M256/N192 specification used to distinguish incompatible MMA configurations.
        type L = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
        /// Supported group-two M256/N160 specification used to distinguish incompatible MMA configurations.
        type R = DenseSharedF16<CtaGroup2, 256, 160, KMajor, KMajor>;
        assert_eq!(InstructionDescriptor::<L>::new().raw(), 0x1030_0010);
        assert_eq!(InstructionDescriptor::<R>::new().raw(), 0x1028_0010);
        assert_eq!(
            encode_smem_offset(ByteOffset::new(0)).unwrap().raw(),
            0x4000_4040_0001_0000
        );
        assert_eq!(
            encode_smem_offset(ByteOffset::new(0x1000)).unwrap().raw(),
            0x4000_4040_0001_0100
        );
        for step in 0..4 {
            assert_eq!(
                encode_smem_offset(ByteOffset::new(0x1000 + step * 32))
                    .unwrap()
                    .raw(),
                0x4000_4040_0001_0100 + step as u64 * 2
            );
        }
        assert!(encode_smem_offset(ByteOffset::new(15)).is_err());
        assert!(encode_smem_offset(ByteOffset::new(0x40000)).is_err());
        assert!(encode_smem_offset(ByteOffset::new(usize::MAX)).is_err());
        assert!(encode_smem_offset(ByteOffset::new(0x3fff0)).is_ok());
    }

    /// Selecting a K16 slice from another slice must still use the original K64 stage base.
    #[test]
    fn k16_selection_is_relative_to_original_stage() {
        let stage = encode_smem_offset(ByteOffset::new(0x1000)).unwrap();
        let a: SmemOperand<'_, OperandA, F16Tile128> = SmemOperand {
            stage,
            selected: stage,
            storage: PhantomData,
            marker: PhantomData,
        };
        assert_eq!(a.k16::<3>().k16::<1>().raw(), 0x4000_4040_0001_0102);
    }
}
