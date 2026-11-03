/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Borrowed TMEM views and explicit collective allocation instructions.
//!
//! Raw register transfers require PTX 8.6 and a tcgen05-capable architecture
//! profile (`sm_100a`, `sm_101a`, `sm_103a` or `sm_110a` in the intrinsic catalog).
//! The current CTM consumer validates `sm_100a`; these are unavailable on SM120a.
use crate::groups::{CtaGroup, CtaGroup1};
use core::marker::PhantomData;
pub use cuda_device::tcgen05::{TmemAddress as TmemAddr, TmemDeallocated, TmemReady, TmemUninit};

/// A typed handle to TMEM with explicit collective allocation and release.
///
/// The handle contains only its shared address-slot pointer and zero-sized
/// state/group markers. It does not elect threads, insert synchronization,
/// relinquish allocation permits, or deallocate on drop. Callers coordinate
/// every GPU participant and explicitly release the allocation after all uses
/// and asynchronous accesses have completed.
///
/// A ready handle can lend typed views, which prevent consuming that local
/// handle while a view remains in use. Other threads' handles and asynchronous
/// hardware operations still require caller-managed lifetime coordination.
#[repr(transparent)]
pub struct TmemGuard<State, const N_COLS: u32, Group: CtaGroup = CtaGroup1> {
    /// Live shared-memory slot containing the allocation’s packed TMEM address.
    smem_ptr: *mut u32,
    /// Records lifecycle state and CTA group without owning or freeing hardware storage.
    _state: PhantomData<(State, Group)>,
}

// Safety: Constructors and collective operations require callers to coordinate
// access to the shared slot and its allocation across all participating threads.
unsafe impl<S, const N: u32, G: CtaGroup> Send for TmemGuard<S, N, G> {}

/// Restricts allocation sizes, view formats and instruction operands.
mod sealed {
    /// Admits only supported power-of-two TMEM allocation widths.
    pub trait Columns<const N: u32> {}
    /// Marks supported accumulator and readback view shapes.
    pub trait Format {}
    /// Marks views that borrow a live allocation guard.
    pub trait Operand {}
}
/// Sealed proof that the allocation width is one of 32, 64, 128, 256 or 512.
pub trait ValidColumns<const N: u32>: sealed::Columns<N> {}
/// Admit the instruction’s 32, 64, 128, 256 and 512 column allocation sizes.
macro_rules! columns { ($($n:literal),*) => { $(impl sealed::Columns<$n> for () {} impl ValidColumns<$n> for () {})* }; }
columns!(32, 64, 128, 256, 512);

/// Offset in TMEM columns from the allocation's base, not a byte offset.
/// Range checks occur when creating a view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnOffset(#[doc = "Column displacement checked when forming a typed view."] u32);
impl ColumnOffset {
    /// Wrap a TMEM column displacement; view construction checks the range.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}
/// Group-two FP32 accumulator: collective `M = 256`, `N = 192` or `160`.
/// Each CTA holds 128 rows and N columns.
pub struct TmemF32<const M: u32, const N: u32>;
/// Local FP32 readback window; only `(ROWS, COLS) = (32, 32)` is supported.
pub struct TmemReadF32<const ROWS: u32, const COLS: u32>;
/// Sealed row/column shape used to validate a borrowed TMEM view.
pub trait TmemFormat: sealed::Format {
    /// Number of consecutive TMEM columns used by the view.
    const COLUMNS: u32;
    /// Number of rows used in this CTA’s 128-row tensor-memory partition.
    const LOCAL_ROWS: u32;
}
/// Marks view shapes accepted as MMA accumulators.
pub trait MmaTmemFormat: TmemFormat {}
/// Define an FP32 accumulator with 128 local rows and a collective M256 shape.
macro_rules! mma_format {
    ($n:literal) => {
        impl sealed::Format for TmemF32<256, $n> {}
        impl TmemFormat for TmemF32<256, $n> {
            const COLUMNS: u32 = $n;
            const LOCAL_ROWS: u32 = 128;
        }
        impl MmaTmemFormat for TmemF32<256, $n> {}
    };
}
mma_format!(192);
mma_format!(160);
impl sealed::Format for TmemReadF32<32, 32> {}
impl TmemFormat for TmemReadF32<32, 32> {
    const COLUMNS: u32 = 32;
    const LOCAL_ROWS: u32 = 32;
}

/// Why a selected row/column window cannot be packed into this allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmemRangeError {
    /// The requested column interval exceeds the allocation or overflows.
    Columns,
    /// The local row is not a 32-row boundary or the window exceeds 128 rows.
    Rows,
    /// Adding the selected window would overflow the packed instruction address.
    AddressOverflow,
}

#[inline(always)]
/// Check local row/column bounds and add their packed fields to a TMEM base address.
fn pack_address<const N: u32>(
    base: u32,
    row: u32,
    column: ColumnOffset,
    width: u32,
    rows: u32,
) -> Result<u32, TmemRangeError> {
    let end = column.0.checked_add(width).ok_or(TmemRangeError::Columns)?;
    if end > N {
        return Err(TmemRangeError::Columns);
    }
    if !row.is_multiple_of(32) || row.checked_add(rows).ok_or(TmemRangeError::Rows)? > 128 {
        return Err(TmemRangeError::Rows);
    }
    if (base & 0xffff)
        .checked_add(end)
        .ok_or(TmemRangeError::AddressOverflow)?
        > 0x10000
    {
        return Err(TmemRangeError::AddressOverflow);
    }
    base.checked_add((row << 16) | column.0)
        .ok_or(TmemRangeError::AddressOverflow)
}

/// A local borrow of a live allocation; other GPU participants still require coordination.
pub struct TmemView<'g, F: TmemFormat, const N: u32, G: CtaGroup> {
    /// Packed TMEM address after checking the selected row and column window.
    address: u32,
    /// Prevents consuming the local allocation handle while this view is borrowed.
    guard: PhantomData<&'g TmemGuard<TmemReady, N, G>>,
    /// Carries the view’s row and column extents without runtime storage.
    format: PhantomData<F>,
}
/// Sealed instruction operand that borrows live tensor memory.
pub trait TmemOperand: sealed::Operand {
    /// CTA group that owns the underlying tensor-memory allocation.
    type Group: CtaGroup;
    /// Validated row/column shape addressed by the operand.
    type Format: TmemFormat;
    /// Escape to the packed instruction address.
    ///
    /// # Safety
    /// The address may only be used while this view and its guard remain live;
    /// callers must also enforce cross-thread lifetime and completion rules.
    unsafe fn raw_address(&self) -> u32;
}
impl<F: TmemFormat, const N: u32, G: CtaGroup> sealed::Operand for TmemView<'_, F, N, G> {}
impl<F: TmemFormat, const N: u32, G: CtaGroup> TmemOperand for TmemView<'_, F, N, G> {
    type Group = G;
    type Format = F;
    #[inline(always)]
    unsafe fn raw_address(&self) -> u32 {
        self.address
    }
}

impl<const N: u32, G: CtaGroup> TmemGuard<TmemUninit, N, G>
where
    (): ValidColumns<N>,
{
    /// Adopt the shared-memory address slot without allocating or synchronizing.
    ///
    /// # Safety
    /// `slot` must be aligned live shared storage through allocation and release.
    #[inline(always)]
    pub unsafe fn from_raw_slot(slot: *mut u32) -> Self {
        Self {
            smem_ptr: slot,
            _state: PhantomData,
        }
    }
    /// Issue allocation without election, fences or barriers.
    ///
    /// # Safety
    /// One complete warp per required CTA must issue uniformly. All tcgen05
    /// operations must use the same group mode. Synchronize before other users
    /// adopt the resulting allocation or read its shared address slot.
    #[inline(always)]
    pub unsafe fn alloc_collective(self) -> TmemGuard<TmemReady, N, G> {
        unsafe { G::alloc(self.smem_ptr, N) };
        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
}
impl<const N: u32, G: CtaGroup> TmemGuard<TmemReady, N, G>
where
    (): ValidColumns<N>,
{
    /// Read the allocated TMEM address from the synchronized shared slot.
    #[inline(always)]
    pub fn address(&self) -> TmemAddr {
        unsafe { TmemAddr::from_raw(self.raw_address()) }
    }

    /// Read the raw TMEM address for low-level instruction interoperability.
    #[inline(always)]
    pub fn raw_address(&self) -> u32 {
        unsafe { *self.smem_ptr }
    }

    /// Return the number of columns in this allocation.
    #[inline(always)]
    pub const fn n_cols(&self) -> u32 {
        N
    }

    /// Adopt another participant's allocation after caller-provided synchronization.
    ///
    /// # Safety
    /// The shared slot and its group-`G` allocation of `N` columns must remain
    /// valid until all views and asynchronous accesses complete. This handle
    /// does not establish global ownership or permission to deallocate.
    #[inline(always)]
    pub unsafe fn adopt_ready(slot: *mut u32) -> Self {
        Self {
            smem_ptr: slot,
            _state: PhantomData,
        }
    }
    /// Issue deallocation without election, fences or barriers.
    ///
    /// # Safety
    /// All users and asynchronous operations must be finished. The required
    /// complete warps must participate uniformly, exactly once per allocation.
    #[inline(always)]
    pub unsafe fn dealloc_collective(self) -> TmemGuard<TmemDeallocated, N, G> {
        unsafe { G::dealloc(self.raw_address(), N) };
        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
    /// Borrow all 128 local rows and `F::COLUMNS` columns for MMA.
    /// `column` counts TMEM columns. The entire view must fit in this allocation.
    #[inline(always)]
    pub fn view<F: MmaTmemFormat>(
        &self,
        column: ColumnOffset,
    ) -> Result<TmemView<'_, F, N, G>, TmemRangeError> {
        Ok(TmemView {
            address: pack_address::<N>(self.raw_address(), 0, column, F::COLUMNS, F::LOCAL_ROWS)?,
            guard: PhantomData,
            format: PhantomData,
        })
    }
    /// Borrow a local 32-row × 32-column window without loading it.
    /// `local_row` must be 0, 32, 64 or 96; the column range must fit in the
    /// allocation. Wait for the producer before issuing a load from this view.
    ///
    /// ```no_run
    /// use ctm_rs::{groups::CtaGroup2, tcgen05::tmem::*};
    /// # fn inspect(guard: &TmemGuard<TmemReady, 512, CtaGroup2>) -> Result<(), TmemRangeError> {
    /// let window = guard.read_32x32(32, ColumnOffset::new(192))?;
    /// // Local rows 32..64, columns 192..224; no load has been issued.
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn read_32x32(
        &self,
        local_row: u32,
        column: ColumnOffset,
    ) -> Result<TmemView<'_, TmemReadF32<32, 32>, N, G>, TmemRangeError> {
        Ok(TmemView {
            address: pack_address::<N>(self.raw_address(), local_row, column, 32, 32)?,
            guard: PhantomData,
            format: PhantomData,
        })
    }
}

/// Issue an SM100 `32x32b.x2` TMEM load into 2 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address, after producer completion and the required after-thread-sync fence.
/// All tcgen05 instructions must agree on CTA-group mode. True register
/// dependencies may consume the result without an immediate wait. Call
/// [`load_wait`] before conflicting memory operations, memory anti-dependencies
/// or a consumer-release handoff, with the required before-thread-sync fence.
/// Keep the allocation live until all accesses complete. See the
/// [PTX register-dependency rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread).
#[inline(always)]
pub unsafe fn load_32x32b_x2(address: u32) -> crate::memory::CuSimd<u32, 2> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x2_raw(address) }
}

/// Issue an SM100 `32x32b.x2` TMEM store from 2 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address; coordinate conflicting accesses explicitly. All tcgen05 instructions
/// must agree on CTA-group mode. Call [`store_wait`] before conflicting memory
/// operations or completion handoffs, with the required before-thread-sync fence.
/// Keep the allocation live until every participant's accesses complete.
#[inline(always)]
pub unsafe fn store_32x32b_x2(address: u32, data: crate::memory::CuSimd<u32, 2>) {
    unsafe { cuda_device::tcgen05::tcgen05_st_32x32b_x2_raw(address, data) }
}

/// Issue an SM100 `32x32b.x4` TMEM load into 4 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address, after producer completion and the required after-thread-sync fence.
/// All tcgen05 instructions must agree on CTA-group mode. True register
/// dependencies may consume the result without an immediate wait. Call
/// [`load_wait`] before conflicting memory operations, memory anti-dependencies
/// or a consumer-release handoff, with the required before-thread-sync fence.
/// Keep the allocation live until all accesses complete. See the
/// [PTX register-dependency rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread).
#[inline(always)]
pub unsafe fn load_32x32b_x4(address: u32) -> crate::memory::CuSimd<u32, 4> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x4_raw(address) }
}

/// Issue an SM100 `32x32b.x4` TMEM store from 4 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address; coordinate conflicting accesses explicitly. All tcgen05 instructions
/// must agree on CTA-group mode. Call [`store_wait`] before conflicting memory
/// operations or completion handoffs, with the required before-thread-sync fence.
/// Keep the allocation live until every participant's accesses complete.
#[inline(always)]
pub unsafe fn store_32x32b_x4(address: u32, data: crate::memory::CuSimd<u32, 4>) {
    unsafe { cuda_device::tcgen05::tcgen05_st_32x32b_x4_raw(address, data) }
}

/// Issue an SM100 `32x32b.x8` TMEM load into 8 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address, after producer completion and the required after-thread-sync fence.
/// All tcgen05 instructions must agree on CTA-group mode. True register
/// dependencies may consume the result without an immediate wait. Call
/// [`load_wait`] before conflicting memory operations, memory anti-dependencies
/// or a consumer-release handoff, with the required before-thread-sync fence.
/// Keep the allocation live until all accesses complete. See the
/// [PTX register-dependency rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread).
#[inline(always)]
pub unsafe fn load_32x32b_x8(address: u32) -> crate::memory::CuSimd<u32, 8> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x8_raw(address) }
}

/// Issue an SM100 `32x32b.x8` TMEM store from 8 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address; coordinate conflicting accesses explicitly. All tcgen05 instructions
/// must agree on CTA-group mode. Call [`store_wait`] before conflicting memory
/// operations or completion handoffs, with the required before-thread-sync fence.
/// Keep the allocation live until every participant's accesses complete.
#[inline(always)]
pub unsafe fn store_32x32b_x8(address: u32, data: crate::memory::CuSimd<u32, 8>) {
    unsafe { cuda_device::tcgen05::tcgen05_st_32x32b_x8_raw(address, data) }
}

/// Issue an SM100 `32x32b.x16` TMEM load into 16 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address, after producer completion and the required after-thread-sync fence.
/// All tcgen05 instructions must agree on CTA-group mode. True register
/// dependencies may consume the result without an immediate wait. Call
/// [`load_wait`] before conflicting memory operations, memory anti-dependencies
/// or a consumer-release handoff, with the required before-thread-sync fence.
/// Keep the allocation live until all accesses complete. See the
/// [PTX register-dependency rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread).
#[inline(always)]
pub unsafe fn load_32x32b_x16(address: u32) -> crate::memory::CuSimd<u32, 16> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x16_raw(address) }
}

/// Issue an SM100 `32x32b.x16` TMEM store from 16 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address; coordinate conflicting accesses explicitly. All tcgen05 instructions
/// must agree on CTA-group mode. Call [`store_wait`] before conflicting memory
/// operations or completion handoffs, with the required before-thread-sync fence.
/// Keep the allocation live until every participant's accesses complete.
#[inline(always)]
pub unsafe fn store_32x32b_x16(address: u32, data: crate::memory::CuSimd<u32, 16>) {
    unsafe { cuda_device::tcgen05::tcgen05_st_32x32b_x16_raw(address, data) }
}

/// Issue an SM100 `32x32b.x32` TMEM load into 32 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address, after producer completion and the required after-thread-sync fence.
/// All tcgen05 instructions must agree on CTA-group mode. True register
/// dependencies may consume the result without an immediate wait. Call
/// [`load_wait`] before conflicting memory operations, memory anti-dependencies
/// or a consumer-release handoff, with the required before-thread-sync fence.
/// Keep the allocation live until all accesses complete. See the
/// [PTX register-dependency rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread).
#[inline(always)]
pub unsafe fn load_32x32b_x32(address: u32) -> crate::memory::CuSimd<u32, 32> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x32_raw(address) }
}

/// Issue an SM100 `32x32b.x32` TMEM store from 32 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address; coordinate conflicting accesses explicitly. All tcgen05 instructions
/// must agree on CTA-group mode. Call [`store_wait`] before conflicting memory
/// operations or completion handoffs, with the required before-thread-sync fence.
/// Keep the allocation live until every participant's accesses complete.
#[inline(always)]
pub unsafe fn store_32x32b_x32(address: u32, data: crate::memory::CuSimd<u32, 32>) {
    unsafe { cuda_device::tcgen05::tcgen05_st_32x32b_x32_raw(address, data) }
}

/// Issue an SM100 `32x32b.x64` TMEM load into 64 raw registers per lane.
/// This uses an encoded instruction address; no bounds checks or waits are inserted.
///
/// # Safety
/// The encoded address must cover a live allocation's selected tile and be
/// row-group aligned. All 32 warp lanes must execute convergently with the same
/// address, after producer completion and the required after-thread-sync fence.
/// All tcgen05 instructions must agree on CTA-group mode. Call [`load_wait`]
/// before conflicting memory operations or a consumer-release handoff.
#[inline(always)]
pub unsafe fn load_32x32b_x64(address: u32) -> crate::memory::CuSimd<u32, 64> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x64_raw(address) }
}

/// Complete the executing thread's previously issued SM100 TMEM loads.
/// No CTA barrier, producer wait or memory/proxy fence is inserted.
///
/// # Safety
/// All 32 warp lanes must execute convergently. The loads must target live,
/// valid TMEM allocations with correctly ordered producers. All tcgen05
/// operations must use one CTA-group mode.
///
/// ```compile_fail,E0133
/// ctm_rs::tcgen05::tmem::load_wait();
/// ```
#[inline(always)]
pub unsafe fn load_wait() {
    cuda_device::tcgen05::tcgen05_load_wait()
}

/// Complete the executing thread's previously issued SM100 TMEM stores.
/// No CTA barrier, consumer synchronization or memory/proxy fence is inserted.
///
/// # Safety
/// All 32 warp lanes must execute convergently. The stores must target live,
/// valid TMEM allocations with no conflicting accesses. All tcgen05 operations
/// must use one CTA-group mode.
///
/// ```compile_fail,E0133
/// ctm_rs::tcgen05::tmem::store_wait();
/// ```
#[inline(always)]
pub unsafe fn store_wait() {
    cuda_device::tcgen05::tcgen05_store_wait()
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;
    /// Valid TMEM windows must preserve allocation bits while rejecting row/column overflow.
    #[test]
    fn checked_tmem_ranges_preserve_allocation_bits() {
        assert_eq!(
            pack_address::<512>(0x0200_0000, 0, ColumnOffset::new(0), 192, 128),
            Ok(0x0200_0000)
        );
        assert_eq!(
            pack_address::<512>(0x0200_0000, 0, ColumnOffset::new(192), 160, 128),
            Ok(0x0200_00c0)
        );
        for row in [0, 32, 64, 96] {
            assert_eq!(
                pack_address::<512>(0x0200_0000, row, ColumnOffset::new(192), 32, 32),
                Ok(0x0200_00c0 | row << 16)
            );
        }
        assert!(pack_address::<512>(0, 128, ColumnOffset::new(0), 32, 32).is_err());
        assert!(pack_address::<512>(0, 1, ColumnOffset::new(0), 32, 32).is_err());
        assert!(pack_address::<256>(0, 0, ColumnOffset::new(192), 160, 128).is_err());
        assert!(pack_address::<512>(0, 0, ColumnOffset::new(u32::MAX), 32, 32).is_err());
        assert!(pack_address::<512>(0x0000_fff0, 0, ColumnOffset::new(32), 32, 32).is_err());
        assert!(pack_address::<512>(0xffff_0000, 32, ColumnOffset::new(0), 32, 32).is_err());
    }
}
