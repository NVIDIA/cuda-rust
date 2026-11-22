/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit cluster membership, shared-address mapping and remote arrival.
use super::{
    groups::CtaGroup,
    memory::{ClusterSharedAddress, SharedAddress},
};
use core::marker::PhantomData;
use cuda_device::barrier::Barrier;

/// Generate forwarding functions for cluster special registers.
macro_rules! coordinate {
    ($($name:ident),* $(,)?) => {$(
        #[doc = concat!("Read the cluster coordinate through [`cuda_device::cluster::", stringify!($name), "`].")]
        #[allow(non_snake_case)]
        #[inline(always)]
        pub fn $name() -> u32 {
            cuda_device::cluster::$name()
        }
    )*};
}
coordinate!(
    cluster_ctaidX,
    cluster_ctaidY,
    cluster_ctaidZ,
    cluster_nctaidX,
    cluster_nctaidY,
    cluster_nctaidZ,
    cluster_idx,
    cluster_idx_x,
    cluster_idx_y,
    cluster_idx_z,
    num_clusters,
    cluster_grid_dim_x,
    cluster_grid_dim_y,
    cluster_grid_dim_z,
    cluster_size,
);

/// Read this CTA's linear rank in its cluster, the value PTX `%cluster_ctarank` holds.
///
/// Forwards [`cuda_device::cluster::block_rank`] under the CuTeDSL primitives name.
#[inline(always)]
pub fn cluster_ctarank() -> u32 {
    cuda_device::cluster::block_rank()
}

/// Arrive at the cluster barrier with release ordering.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute the matching wait.
#[inline(always)]
pub unsafe fn barrier_cluster_arrive() {
    unsafe { cuda_device::cluster::barrier_cluster_arrive() }
}

/// Arrive at the cluster barrier in aligned mode with release ordering.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute the matching wait. Every non-exited thread in the warp must
/// execute with identical control flow.
#[inline(always)]
pub unsafe fn barrier_cluster_arrive_aligned() {
    unsafe { cuda_device::cluster::barrier_cluster_arrive_aligned() }
}

/// Arrive at the cluster barrier without a release guarantee.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute the matching wait. Publish earlier memory accesses with the
/// required cluster-scope fence first.
#[inline(always)]
pub unsafe fn barrier_cluster_arrive_relaxed() {
    unsafe { cuda_device::cluster::barrier_cluster_arrive_relaxed() }
}

/// Arrive at the cluster barrier in aligned mode without a release guarantee.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute the matching wait. Every non-exited thread in the warp must
/// execute with identical control flow. Publish earlier memory accesses with
/// the required cluster-scope fence first.
#[inline(always)]
pub unsafe fn barrier_cluster_arrive_relaxed_aligned() {
    unsafe { cuda_device::cluster::barrier_cluster_arrive_relaxed_aligned() }
}

/// Wait at the cluster barrier with acquire ordering.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute this matching wait.
#[inline(always)]
pub unsafe fn barrier_cluster_wait() {
    unsafe { cuda_device::cluster::barrier_cluster_wait() }
}

/// Wait at the cluster barrier in aligned mode with acquire ordering.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute this matching wait. Every non-exited thread in the warp must
/// execute with identical control flow.
#[inline(always)]
pub unsafe fn barrier_cluster_wait_aligned() {
    unsafe { cuda_device::cluster::barrier_cluster_wait_aligned() }
}

/// A nonempty multicast mask. Group mode does not restrict cluster membership.
pub struct CtaMask<G: CtaGroup> {
    /// Nonzero set of destination CTA ranks, one bit per rank.
    bits: u16,
    /// Instruction group selector; does not limit which ranks the mask can name.
    group: PhantomData<G>,
}
impl<G: CtaGroup> Copy for CtaMask<G> {}
impl<G: CtaGroup> Clone for CtaMask<G> {
    fn clone(&self) -> Self {
        *self
    }
}
/// Why a CTA multicast mask cannot be issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskError {
    /// No destination CTA is selected.
    Empty,
}
impl<G: CtaGroup> CtaMask<G> {
    /// Adopt a nonzero set of destination-rank bits; group mode does not cap membership.
    #[inline(always)]
    pub const fn try_from_bits(bits: u16) -> Result<Self, MaskError> {
        if bits == 0 {
            Err(MaskError::Empty)
        } else {
            Ok(Self {
                bits,
                group: PhantomData,
            })
        }
    }
    /// Return the mask bits consumed by the multicast instruction.
    #[inline(always)]
    pub const fn bits(self) -> u16 {
        self.bits
    }
}
/// Map local shared storage to the same offset in a cluster peer (PTX `mapa`).
///
/// # Safety
/// `rank` must name an active CTA in this cluster with matching live storage.
/// Remote initialization, readiness and lifetime must be coordinated explicitly.
#[inline(always)]
pub unsafe fn mapa<'s, T>(address: SharedAddress<'s, T>, rank: u32) -> ClusterSharedAddress<'s, T> {
    unsafe {
        ClusterSharedAddress::from_raw(cuda_device::cluster::map_shared_rank_mut(
            address.as_raw(),
            rank,
        ))
    }
}
/// Issue one remote arrival without waiting or changing a software phase.
///
/// # Safety
/// The remote barrier must be initialized and expecting this arrival. All
/// participants must agree on the phase and keep its storage live.
#[inline(always)]
pub unsafe fn arrive_remote(address: ClusterSharedAddress<'_, Barrier>) {
    unsafe {
        let offset = cuda_device::shared::cvta_generic_to_cluster_shared_offset(
            address.as_raw().cast::<u8>(),
        );
        cuda_device::barrier::mbarrier_arrive_cluster(offset)
    }
}

/// Arrive at a remote barrier without publishing preceding memory accesses.
///
/// The barrier count and execution dependency retain cluster scope. Unlike
/// [`arrive_remote`], this operation has no release memory ordering.
///
/// # Safety
/// The remote barrier must be initialized and expecting this arrival. All
/// participants must agree on the phase and keep its storage live. The caller
/// must supply the proxy or TMEM fences required by its data-handoff protocol;
/// this relaxed arrival alone does not make preceding data accesses visible.
#[inline(always)]
pub unsafe fn arrive_remote_relaxed(address: ClusterSharedAddress<'_, Barrier>) {
    unsafe {
        let offset = cuda_device::shared::cvta_generic_to_cluster_shared_offset(
            address.as_raw().cast::<u8>(),
        );
        cuda_device::barrier::mbarrier_arrive_cluster_relaxed(offset)
    }
}
