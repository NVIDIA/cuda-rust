/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed tensor maps with canonical opaque descriptor storage.
use super::memory::PhysicalFormat;
use super::{
    cluster::CtaMask,
    groups::CtaGroup2,
    memory::{ClusterSharedAddress, SharedAddress, SharedRegion},
};
use cuda_device::TmaDescriptor;
use cuda_device::barrier::Barrier;

/// Signed element coordinates; descriptor-defined out-of-bounds behavior is retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coordinates2D {
    /// Coordinate in the descriptor’s contiguous dimension, measured in storage elements.
    inner: i32,
    /// Coordinate in the descriptor’s outer dimension, normally the global row.
    outer: i32,
}
impl Coordinates2D {
    /// Create descriptor coordinates in inner-then-outer element order.
    pub const fn new(inner: i32, outer: i32) -> Self {
        Self { inner, outer }
    }
    /// Return the signed coordinate in the contiguous element dimension.
    pub const fn inner(self) -> i32 {
        self.inner
    }
    /// Return the signed coordinate in the outer, normally row, dimension.
    pub const fn outer(self) -> i32 {
        self.outer
    }
}

/// Issue one ordinary TMA load into this CTA's shared storage.
///
/// This is the CuTeDSL primitives name for the unicast `cp.async.bulk.tensor`
/// load; the rank is fixed by [`TensorMap`]. No issuer election, fence,
/// barrier arrival or completion wait is inserted.
///
/// # Safety
/// The caller must select the issuer explicitly and publish an initialized
/// completion barrier with the expected byte count. Keep the map, global
/// allocation, destination and barrier live until completion. Coordinate all
/// accesses to the destination and wait explicitly before reading its contents.
#[inline(always)]
pub unsafe fn cp_async_bulk_tensor_shared_cta_global<T, F: PhysicalFormat<Element = T>>(
    dst: &SharedRegion<'_, F>,
    map: &TensorMap<T, F>,
    coords: Coordinates2D,
    completion: SharedAddress<'_, Barrier>,
) {
    unsafe {
        cuda_device::tma::cp_async_bulk_tensor_2d_g2s_cta(
            dst.address().as_raw().cast(),
            map.as_descriptor(),
            coords.inner,
            coords.outer,
            completion.as_raw(),
        )
    }
}

/// Issue one group-2 multicast TMA load, using the default cache hint (zero).
///
/// This is the CuTeDSL primitives name for the cluster-multicast
/// `cp.async.bulk.tensor` load; the `CtaMask<CtaGroup2>` argument fixes the
/// `.multicast::cluster` and `cta_group::2` qualifiers.
///
/// # Safety
/// The elected issuer must satisfy group-2 TMA rules. Every mask bit must name
/// a participating CTA with matching destination storage. The remote completion
/// barrier must be initialized with the expected byte count. Keep the map,
/// allocation, destinations and barriers live until completion; wait explicitly.
#[inline(always)]
pub unsafe fn cp_async_bulk_tensor_shared_cluster_global<T, F: PhysicalFormat<Element = T>>(
    dst: &SharedRegion<'_, F>,
    map: &TensorMap<T, F>,
    coords: Coordinates2D,
    completion: ClusterSharedAddress<'_, Barrier>,
    mask: CtaMask<CtaGroup2>,
) {
    unsafe {
        cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2(
            dst.address().as_raw().cast(),
            map.as_descriptor(),
            coords.inner,
            coords.outer,
            completion.as_raw(),
            mask.bits(),
        )
    }
}
/// Issue one shared-to-global tensor store without a fence, commit or wait.
///
/// This is the CuTeDSL primitives name for the shared-to-global
/// `cp.async.bulk.tensor` store.
///
/// # Safety
/// Publish shared writes to the async proxy before issuing. Source and global
/// storage must remain live through the appropriate explicit completion waits.
#[inline(always)]
pub unsafe fn cp_async_bulk_tensor_global_shared_cta<T, F: PhysicalFormat<Element = T>>(
    src: &SharedRegion<'_, F>,
    map: &TensorMap<T, F>,
    coords: Coordinates2D,
) {
    unsafe {
        cuda_device::tma::cp_async_bulk_tensor_2d_s2g(
            src.address().as_raw().cast(),
            map.as_descriptor(),
            coords.inner,
            coords.outer,
        )
    }
}
/// Prefetch TMA descriptor metadata into the tensor-map cache (PTX `prefetch.tensormap`).
///
/// `descriptor` is either a typed [`TensorMap`] reference or a raw
/// `*const TmaDescriptor` encoded by another tensor abstraction. The raw form
/// does not interpret element or shared-layout types, which permits
/// interoperability with layout-typed descriptors such as CuTe tensor maps
/// without adopting a CTM physical format.
///
/// # Safety
/// `descriptor` must name a valid, 64-byte-aligned, 128-byte CUDA tensor map in
/// device-accessible storage that remains live through the prefetch.
#[inline(always)]
pub unsafe fn prefetch_tensormap<D: TmaDescriptorRef>(descriptor: D) {
    unsafe {
        cuda_device::tma::prefetch_tma_descriptor(sealed::Descriptor::descriptor_ptr(descriptor))
    }
}
/// Sealed proof that a value names a 128-byte TMA descriptor [`prefetch_tensormap`] may read.
pub trait TmaDescriptorRef: sealed::Descriptor {}
impl<T, F: PhysicalFormat<Element = T>> sealed::Descriptor for &TensorMap<T, F> {
    #[inline(always)]
    fn descriptor_ptr(self) -> *const TmaDescriptor {
        self.as_descriptor()
    }
}
impl<T, F: PhysicalFormat<Element = T>> TmaDescriptorRef for &TensorMap<T, F> {}
impl sealed::Descriptor for *const TmaDescriptor {
    #[inline(always)]
    fn descriptor_ptr(self) -> *const TmaDescriptor {
        self
    }
}
impl TmaDescriptorRef for *const TmaDescriptor {}
/// Commit this thread’s pending bulk copies as one group; does not wait.
#[inline(always)]
pub fn cp_async_bulk_commit_group() {
    cuda_device::tma::cp_async_bulk_commit_group()
}
/// Limits bulk-copy wait counts and prefetch operands to the forms the instructions accept.
mod sealed {
    /// Select a literal maximum number of committed groups that may remain pending.
    pub trait Count<const N: u32> {
        /// Wait until at most N groups still read their shared source buffers.
        fn wait();
    }
    /// Resolve a descriptor reference to the raw descriptor bytes the instruction reads.
    pub trait Descriptor {
        /// Return the address of the 128-byte descriptor.
        fn descriptor_ptr(self) -> *const cuda_device::TmaDescriptor;
    }
}
/// Sealed proof that a bulk-copy wait immediate is in 0..=7.
pub trait ValidWaitCount<const N: u32>: sealed::Count<N> {}
/// Generate the supported immediate bulk-copy read waits.
macro_rules! waits { ($($n:literal),*) => { $(impl sealed::Count<$n> for () { #[inline(always)] fn wait() { cuda_device::tma::cp_async_bulk_wait_group_read($n) } } impl ValidWaitCount<$n> for () {})* }; }
waits!(0, 1, 2, 3, 4, 5, 6, 7);
/// Wait until at most `N` committed bulk groups still read their shared sources.
/// This is not proof that their global writes are visible to other threads.
#[inline(always)]
pub fn cp_async_bulk_wait_group_read<const N: u32>()
where
    (): ValidWaitCount<N>,
{
    <() as sealed::Count<N>>::wait()
}
use core::marker::PhantomData;

/// Host-encoded tensor map matching one reviewed physical shared format.
///
/// A copy of this descriptor does not extend the global allocation's lifetime.
#[repr(transparent)]
pub struct TensorMap<T, F: PhysicalFormat<Element = T>> {
    /// Opaque CUDA descriptor bytes; copying them does not retain the tensor allocation.
    raw: TmaDescriptor,
    /// Records the matching element type and physical shared format.
    marker: PhantomData<fn() -> (T, F)>,
}
impl<T, F: PhysicalFormat<Element = T>> Copy for TensorMap<T, F> {}
impl<T, F: PhysicalFormat<Element = T>> Clone for TensorMap<T, F> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, F: PhysicalFormat<Element = T>> TensorMap<T, F> {
    /// Adopt a descriptor encoded outside the typed host encoder.
    ///
    /// # Safety
    /// Bytes must describe element type `T`, rank two and the tile/swizzle
    /// specified by `F`. Underlying global storage must remain live through
    /// every asynchronous operation using this map.
    #[inline(always)]
    pub const unsafe fn from_encoded(raw: TmaDescriptor) -> Self {
        Self {
            raw,
            marker: PhantomData,
        }
    }
    /// Borrow canonical opaque descriptor storage for instruction interoperability.
    #[inline(always)]
    pub const fn as_descriptor(&self) -> &TmaDescriptor {
        &self.raw
    }
}
