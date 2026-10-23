/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit MMA issue, collector selection and TMEM register readback.
use super::{
    descriptors::*,
    tmem::{TmemOperand, TmemReadF32},
};
use crate::groups::CtaGroup2;
use cuda_device::cusimd::CuSimd;

/// Explicit A-collector reuse selectors; kernel code must order compatible issues.
pub mod collector_a {
    /// Read A without retaining it for a subsequent issue.
    pub enum Discard {}
    /// Use retained A for its final issue and release the collector.
    pub enum LastUse {}
    /// Read A and retain it for subsequent collector uses.
    pub enum Fill {}
    /// Reuse retained A while keeping the collector available for later issues.
    pub enum Use {}
}
/// Restricts MMA issue to supported specification and collector-selector pairs.
mod sealed {
    /// Dispatch a typed MMA call to a literal hardware collector selector.
    pub trait MmaDispatch {
        /// Issue shared/shared FP16 MMA; caller supplies election, live operands and collector ordering.
        unsafe fn issue(dst: u32, a: u64, b: u64, descriptor: u32, enable_d: bool);
    }
}
/// Implement one collector selector using the existing group-two MMA intrinsic.
macro_rules! dispatch {
    ($usage:ty,$selector:literal) => {
        impl<S: MmaSpec> sealed::MmaDispatch for (S, $usage) {
            #[inline(always)]
            unsafe fn issue(dst: u32, a: u64, b: u64, descriptor: u32, enable_d: bool) {
                unsafe {
                    cuda_device::tcgen05::tcgen05_mma_shared::<0, 2, $selector>(
                        dst, a, b, descriptor, enable_d,
                    )
                }
            }
        }
    };
}
dispatch!(collector_a::Discard, 0);
dispatch!(collector_a::LastUse, 1);
dispatch!(collector_a::Fill, 2);
dispatch!(collector_a::Use, 3);

/// Issue one MMA. The collector selector does not infer a sequence or pipeline.
///
/// cute-rs exposes the `enable_d` flag (PTX `enable-input-d`) as `accumulate`,
/// CuTeDSL's `Field.ACCUMULATE`.
///
/// # Safety
/// The elected issuing thread must satisfy group-2 participation and collector
/// sequencing rules. Shared operands must be published and remain live until
/// completion; the accumulator must be initialized when `enable_d` is true.
/// Complete dependent asynchronous work explicitly before reading/reusing it.
#[inline(always)]
pub unsafe fn mma<S, U>(
    dst: &impl TmemOperand<Group = S::Group, Format = S::DstFormat>,
    a: &SmemOperand<'_, OperandA, S::AFormat>,
    b: &SmemOperand<'_, OperandB, S::BFormat>,
    descriptor: &InstructionDescriptor<S>,
    enable_d: bool,
) where
    S: MmaSpec,
    (S, U): sealed::MmaDispatch,
{
    unsafe {
        <(S, U) as sealed::MmaDispatch>::issue(
            dst.raw_address(),
            a.raw(),
            b.raw(),
            descriptor.raw(),
            enable_d,
        )
    }
}

/// Registers returned by a pending TMEM load; decode only after an explicit wait.
#[repr(transparent)]
pub struct TmemRaw32(
    #[doc = "Pending raw registers, consumed only after the caller’s explicit load wait."]
    CuSimd<u32, 32>,
);

/// Issue one warp-collective TMEM load; no wait is inserted.
///
/// # Safety
/// All lanes must issue uniformly with valid addresses and completed producers.
/// Keep the allocation live and call [`super::tmem::load_wait`] before
/// consuming the registers.
#[inline(always)]
pub unsafe fn load_32x32b_x32(
    src: &impl TmemOperand<Group = CtaGroup2, Format = TmemReadF32<32, 32>>,
) -> TmemRaw32 {
    TmemRaw32(unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x32_raw(src.raw_address()) })
}
impl TmemRaw32 {
    /// Interpret completed load registers in their original order.
    ///
    /// # Safety
    /// The issuing thread must execute [`super::tmem::load_wait`] after this load and before
    /// decoding. The wrapper does not model the hardware scoreboard.
    #[inline(always)]
    pub unsafe fn decode_f32(self) -> CuSimd<f32, 32> {
        // Destructure by value. Borrowing the carrier for get::<I>() forces
        // an addressable 128-byte local before downstream inlining/SROA.
        /// Bitcast each completed register by value, avoiding an addressable local carrier.
        macro_rules! decode { ($($r:ident),*) => {{
            let [$($r),*] = self.0.to_array();
            CuSimd::new([$(f32::from_bits($r)),*])
        }}; }
        decode!(
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18,
            r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31
        )
    }
}
