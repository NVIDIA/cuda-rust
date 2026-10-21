/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Two-CTA persistent FP16 GEMM expressed entirely with instruction primitives.
//! The schedule matches fp16_gemm_256x352_cute; election, phases and completion
//! are visible here rather than inserted by a layout/pipeline dialect.
use ctm_rs::mbarrier::BarrierAddressExt;
use ctm_rs::memory::DynamicSharedArray;
use ctm_rs::tcgen05::tmem::{TmemGuard, TmemReady, TmemUninit, TmemView};
use ctm_rs::{
    cluster::{self, CtaMask},
    groups::CtaGroup2,
    mbarrier::{self, Barrier, GeneralBarrier, ManagedBarrier, Ready, TmaBarrier, Uninit},
    memory::{F16Tile128, F16TileB80, F16TileB96, F16TileC, PhysicalFormat, SharedRegion},
    tcgen05::{
        self,
        descriptors::*,
        mma::{self, collector_a},
        tmem::{ColumnOffset, TmemF32},
    },
    thread,
    tma::{self, Coordinates2D, TensorMap},
    warp,
};
use ctm_rs::{cluster_launch, cuda_module, kernel, launch_bounds, launch_contract};

/// Per-CTA 128x64 FP16 A tile with the shared swizzle used by its tensor map.
pub type ASmem = F16Tile128;
/// Per-CTA 96x64 FP16 B tile feeding the 192-column MMA partition.
pub type B0Smem = F16TileB96;
/// Per-CTA 80x64 FP16 B tile feeding the 160-column MMA partition.
pub type B1Smem = F16TileB80;
/// Per-CTA 128x32 FP16 output tile with the B64 shared swizzle.
pub type CSmem = F16TileC;
/// Two-CTA FP16 MMA shape for the 256x192 left output partition.
type Left = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
/// Two-CTA FP16 MMA shape for the 256x160 right output partition.
type Right = DenseSharedF16<CtaGroup2, 256, 160, KMajor, KMajor>;
/// Ready two-CTA tensor-memory allocation retaining all 512 allocated columns.
type Accumulator = TmemGuard<TmemReady, 512, CtaGroup2>;
/// Borrowed FP32 tensor-memory view of the 256x192 left accumulator partition.
type LeftView<'a> = TmemView<'a, TmemF32<256, 192>, 512, CtaGroup2>;
/// Borrowed FP32 tensor-memory view of the 256x160 right accumulator partition.
type RightView<'a> = TmemView<'a, TmemF32<256, 160>, 512, CtaGroup2>;

/// Eight FP16 values stored as one naturally aligned 16-byte shared-memory chunk.
#[repr(C, align(16))]
struct F16x8(
    #[doc = "FP16 lanes in increasing column order within the contiguous chunk."] [f16; 8],
);
/// Compile-time check that the chunk carrier has exactly 16-byte size and alignment.
const _: () = assert!(core::mem::size_of::<F16x8>() == 16 && core::mem::align_of::<F16x8>() == 16);

/// Threads required in each CTA by the kernel launch contract.
pub const THREADS: u32 = 192;
/// Number of A/B shared-memory stages overlapped by TMA and MMA.
const STAGES: usize = 5;
/// Shared bytes occupied by one CTA's A stage.
const A_BYTES: usize = ASmem::FOOTPRINT_BYTES;
/// Shared bytes occupied by both B partitions in one CTA stage.
const B_BYTES: usize = B0Smem::FOOTPRINT_BYTES + B1Smem::FOOTPRINT_BYTES;
/// Byte offset of the staged B ring after all staged A tiles.
const B_OFFSET: usize = A_BYTES * STAGES;
/// Byte offset of the two shared output stages after the A/B rings.
const C_OFFSET: usize = B_OFFSET + B_BYTES * STAGES;
/// Shared bytes occupied by one FP16 output stage.
const C_STAGE_BYTES: usize = CSmem::FOOTPRINT_BYTES;
/// Byte offset of load, accumulator, and deallocation barriers after output storage.
const BARRIER_OFFSET: usize = C_OFFSET + 2 * C_STAGE_BYTES;
/// Total dynamic shared-memory allocation in bytes required per CTA.
pub const DYNAMIC_SMEM_BYTES: usize = BARRIER_OFFSET + 128;
/// Combined A/B input bytes delivered by both CTAs for one K64 stage.
const TX_BYTES: u32 = (2 * (A_BYTES + B_BYTES)) as u32;
/// Compile-time checks for the complete shared-memory footprint and two-CTA transaction bytes.
const _: () = assert!(DYNAMIC_SMEM_BYTES == 211_072 && TX_BYTES == 77_824);

// A violated physical contract must abort the kernel, never let one lane leave
// a collective while its peers wait. All valid launch layouts satisfy these.
/// Trap on a physical-contract error so one lane cannot leave its collective peers waiting.
#[inline(always)]
fn checked<T, E>(value: Result<T, E>) -> T {
    match value {
        Ok(value) => value,
        Err(_) => ctm_rs::debug::trap(),
    }
}

/// Borrow a typed shared tile; the caller provides live, correctly aligned storage of at least its footprint.
#[inline(always)]
unsafe fn region<'s, F: PhysicalFormat>(base: *mut u8) -> SharedRegion<'s, F> {
    unsafe { SharedRegion::<F>::from_raw_parts_or_trap(base.cast(), F::FOOTPRINT_BYTES) }
}
/// Initialize a live barrier with the requested arrivals; the caller owns initialization and publication ordering.
#[inline(always)]
unsafe fn init<K>(pointer: *mut Barrier, arrivals: u32) {
    unsafe {
        ManagedBarrier::<Uninit, K>::from_static(pointer).init_instruction(arrivals);
    }
}
/// Retry cluster parity probes on a live initialized barrier until the requested phase completes.
#[inline(always)]
unsafe fn wait<K>(barrier: &ManagedBarrier<Ready, K>, phase: u32) {
    unsafe { while !mbarrier::try_wait_parity_cluster(barrier, phase) {} }
}
/// Return the checked two-bit participation mask naming both CTAs in the cluster.
#[inline(always)]
fn both_ctas() -> CtaMask<CtaGroup2> {
    checked(CtaMask::try_from_bits(3))
}

/// Issue one K16 step for both N partitions with fill/last-use A collectors; operands must be ready and all issuer lanes participate.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn issue_pair<const K: u32>(
    left: &LeftView<'_>,
    right: &RightView<'_>,
    a: &SmemOperand<'_, OperandA, ASmem>,
    b0: &SmemOperand<'_, OperandB, B0Smem>,
    b1: &SmemOperand<'_, OperandB, B1Smem>,
    left_desc: &InstructionDescriptor<Left>,
    right_desc: &InstructionDescriptor<Right>,
    enable: bool,
) where
    (): ValidKStep<K>,
{
    unsafe {
        let a = a.k16::<K>();
        let b0 = b0.k16::<K>();
        let b1 = b1.k16::<K>();
        if warp::is_elected_sync(u32::MAX) {
            mma::mma::<Left, collector_a::Fill>(left, &a, &b0, left_desc, enable);
        }
        if warp::is_elected_sync(u32::MAX) {
            mma::mma::<Right, collector_a::LastUse>(right, &a, &b1, right_desc, enable);
        }
    }
}

/// Issue all four K16 steps for a ready shared stage; stage must index the live five-slot allocation.
#[inline(always)]
unsafe fn issue_stage(
    base: *mut u8,
    stage: usize,
    accumulate: bool,
    left: &LeftView<'_>,
    right: &RightView<'_>,
    left_desc: &InstructionDescriptor<Left>,
    right_desc: &InstructionDescriptor<Right>,
) {
    unsafe {
        let a = region::<ASmem>(base.add(stage * A_BYTES));
        let b0 = region::<B0Smem>(base.add(B_OFFSET + stage * B_BYTES));
        let b1 = region::<B1Smem>(base.add(B_OFFSET + stage * B_BYTES + B0Smem::FOOTPRINT_BYTES));
        let a = checked(SmemOperand::<OperandA, ASmem>::from_region(&a));
        let b0 = checked(SmemOperand::<OperandB, B0Smem>::from_region(&b0));
        let b1 = checked(SmemOperand::<OperandB, B1Smem>::from_region(&b1));
        issue_pair::<0>(left, right, &a, &b0, &b1, left_desc, right_desc, accumulate);
        issue_pair::<1>(left, right, &a, &b0, &b1, left_desc, right_desc, true);
        issue_pair::<2>(left, right, &a, &b0, &b1, left_desc, right_desc, true);
        issue_pair::<3>(left, right, &a, &b0, &b1, left_desc, right_desc, true);
    }
}

// Each swizzled 8-half chunk is contiguous and 16-byte aligned.
/// Add bias and store eight FP16 lanes; output must be 16-byte aligned, tid below 128, and COL a valid eight-lane chunk.
#[inline(always)]
unsafe fn store_chunk<const COL: usize>(output: *mut f16, values: [f32; 32], tid: u32, bias: f32) {
    unsafe {
        output
            .add(crate::physical::output_offset(tid, COL as u32))
            .cast::<F16x8>()
            .write(F16x8([
                (values[COL] + bias) as f16,
                (values[COL + 1] + bias) as f16,
                (values[COL + 2] + bias) as f16,
                (values[COL + 3] + bias) as f16,
                (values[COL + 4] + bias) as f16,
                (values[COL + 5] + bias) as f16,
                (values[COL + 6] + bias) as f16,
                (values[COL + 7] + bias) as f16,
            ]));
    }
}

/// Read a ready TMEM subtile, wait for its load, and store biased FP16 chunks into live aligned shared output.
#[inline(always)]
unsafe fn epilogue(acc: &Accumulator, output: *mut f16, tid: u32, subtile: u32, bias: f32) {
    unsafe {
        let read = checked(acc.read_32x32((tid / 32) * 32, ColumnOffset::new(subtile * 32)));
        let pending = mma::load_32x32b_x32(&read);
        mma::load_wait();
        // By-value, constant-index extraction keeps the hardware result order.
        let values = pending.decode_f32().to_array();
        store_chunk::<0>(output, values, tid, bias);
        store_chunk::<8>(output, values, tid, bias);
        store_chunk::<16>(output, values, tid, bias);
        store_chunk::<24>(output, values, tid, bias);
    }
}

/// Two-CTA SM100 GEMM entry point and its explicit launch/storage contract.
#[cuda_module]
pub mod kernels {

    use super::*;

    /// Tensor allocations and all four map descriptors stay live through completion.
    #[launch_bounds(192)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(domain = 1, block = (192, 1, 1), dynamic_shared = 211072,
                      dynamic_shared_alignment = 1024, min_compute_capability = (10, 0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp16_gemm_256x352(
        #[grid_constant] a: &TensorMap<f16, ASmem>,
        #[grid_constant] b0: &TensorMap<f16, B0Smem>,
        #[grid_constant] b1: &TensorMap<f16, B1Smem>,
        #[grid_constant] c: &TensorMap<f16, CSmem>,
        bias: *const f16,
        m: u64,
        n: u64,
        k: u64,
        has_bias: u64,
    ) {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        let tid = thread::threadIdx_x();
        let warp_id = tid / 32;
        let rank = cluster::block_rank();
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        unsafe {
            let full = smem.add(BARRIER_OFFSET).cast::<Barrier>();
            let empty = full.add(STAGES);
            let acc_empty_ptr = empty.add(STAGES);
            let acc_full_ptr = acc_empty_ptr.add(1);
            let dealloc_ptr = acc_full_ptr.add(1);
            let token = dealloc_ptr.add(1).cast::<u32>();
            if warp_id == 4 {
                tma::prefetch(a);
                tma::prefetch(b0);
                tma::prefetch(b1);
                tma::prefetch(c);
            }
            if warp_id == 0 && warp::is_elected_sync(u32::MAX) {
                init::<GeneralBarrier>(dealloc_ptr, 32);
                init::<GeneralBarrier>(acc_empty_ptr, 8);
                init::<GeneralBarrier>(acc_full_ptr, 1);
                let mut stage = 0;
                while stage < STAGES {
                    init::<TmaBarrier>(full.add(stage), 1);
                    init::<GeneralBarrier>(empty.add(stage), 1);
                    stage += 1;
                }
            }
            mbarrier::fence_mbarrier_init_release_cluster();
            cluster::barrier_cluster_arrive_relaxed();
            let mtiles = m / 256;
            let total_tiles = mtiles * n.div_ceil(352);
            let mut tile = thread::blockIdx_x() / 2;
            let tile_stride = thread::gridDim_x() / 2;
            let ktiles = k.div_ceil(64);
            cluster::barrier_cluster_wait();
            let acc_empty = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(acc_empty_ptr);
            let acc_full = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(acc_full_ptr);

            if warp_id == 5 {
                let mut stage = 0usize;
                let mut phase = 1u32;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let mut kt = 0;
                    while kt < ktiles {
                        let reusable =
                            ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(stage));
                        wait(&reusable, phase);
                        let loaded =
                            ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(stage));
                        if warp::is_elected_sync(u32::MAX) {
                            if rank == 0 {
                                mbarrier::arrive_expect_tx_cluster(&loaded, TX_BYTES);
                            }
                            let completion = cluster::map_shared(loaded.shared_address(), 0);
                            let mask = checked(CtaMask::<CtaGroup2>::try_from_bits(1 << rank));
                            let a_tile = region::<ASmem>(smem.add(stage * A_BYTES));
                            let b0_tile = region::<B0Smem>(smem.add(B_OFFSET + stage * B_BYTES));
                            let b1_tile = region::<B1Smem>(
                                smem.add(B_OFFSET + stage * B_BYTES + B0Smem::FOOTPRINT_BYTES),
                            );
                            tma::load_2d_cg2(
                                &a_tile,
                                a,
                                Coordinates2D::new((kt * 64) as i32, row as i32),
                                completion,
                                mask,
                            );
                            tma::load_2d_cg2(
                                &b0_tile,
                                b0,
                                Coordinates2D::new((kt * 64) as i32, (col + rank * 96) as i32),
                                completion,
                                mask,
                            );
                            tma::load_2d_cg2(
                                &b1_tile,
                                b1,
                                Coordinates2D::new(
                                    (kt * 64) as i32,
                                    (col + 192 + rank * 80) as i32,
                                ),
                                completion,
                                mask,
                            );
                        }
                        stage += 1;
                        if stage == STAGES {
                            stage = 0;
                            phase ^= 1;
                        }
                        kt += 1;
                    }
                    tile += tile_stride;
                }
                // Reach the last producer slot/phase even for short K and ring wraps.
                let mut tail = 0;
                while tail < STAGES - 1 {
                    stage += 1;
                    if stage == STAGES {
                        stage = 0;
                        phase ^= 1;
                    }
                    tail += 1;
                }
                if warp::is_elected_sync(u32::MAX) {
                    wait(
                        &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(stage)),
                        phase,
                    );
                }
            } else if warp_id == 4 {
                mbarrier::barrier_cta_sync(2, 160);
                let acc = Accumulator::adopt_ready(token);
                let mut stage = 0usize;
                let mut phase = 0u32;
                let mut acc_phase = 1u32;
                while tile < total_tiles {
                    if rank == 0 {
                        wait(&acc_empty, acc_phase);
                        tcgen05::fence_after_thread_sync();
                        let left = checked(acc.view::<TmemF32<256, 192>>(ColumnOffset::new(0)));
                        let right = checked(acc.view::<TmemF32<256, 160>>(ColumnOffset::new(192)));
                        let left_desc = InstructionDescriptor::<Left>::new();
                        let right_desc = InstructionDescriptor::<Right>::new();
                        let mut accumulate = false;
                        let mut kt = 0;
                        while kt < ktiles {
                            let loaded =
                                ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(stage));
                            wait(&loaded, phase);
                            tcgen05::fence_after_thread_sync();
                            issue_stage(
                                smem,
                                stage,
                                accumulate,
                                &left,
                                &right,
                                &left_desc,
                                &right_desc,
                            );
                            if warp::is_elected_sync(u32::MAX) {
                                let reusable = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                                    empty.add(stage),
                                );
                                tcgen05::commit_multicast_cg2(
                                    reusable.shared_address(),
                                    both_ctas(),
                                );
                            }
                            accumulate = true;
                            stage += 1;
                            if stage == STAGES {
                                stage = 0;
                                phase ^= 1;
                            }
                            kt += 1;
                        }
                        if warp::is_elected_sync(u32::MAX) {
                            tcgen05::commit_multicast_cg2(acc_full.shared_address(), both_ctas());
                        }
                    }
                    acc_phase ^= 1;
                    tile += tile_stride;
                }
                if rank == 0 && warp::is_elected_sync(u32::MAX) {
                    wait(&acc_empty, acc_phase);
                    tcgen05::fence_after_thread_sync();
                }
            } else {
                if warp_id == 0 {
                    let owner = TmemGuard::<TmemUninit, 512, CtaGroup2>::from_raw_slot(token);
                    owner.alloc_collective();
                    tcgen05::relinquish_alloc_permit_cg2();
                }
                mbarrier::barrier_cta_sync(2, 160);
                let acc = Accumulator::adopt_ready(token);
                let mut phase = 0u32;
                let mut epi_stage = 0usize;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let row_bias = if has_bias != 0 {
                        *bias.add((row + tid) as usize) as f32
                    } else {
                        0.0
                    };
                    wait(&acc_full, phase);
                    tcgen05::fence_after_thread_sync();
                    let mut subtile = 0;
                    while subtile < 11 {
                        epi_stage ^= 1;
                        let output = smem.add(C_OFFSET + epi_stage * C_STAGE_BYTES).cast::<f16>();
                        let dst = region::<CSmem>(output.cast::<u8>());
                        epilogue(&acc, output, tid, subtile, row_bias);
                        mbarrier::fence_proxy_async_shared_cta();
                        mbarrier::barrier_cta_sync(1, 128);
                        if tid == 0 {
                            tma::store_2d(
                                &dst,
                                c,
                                Coordinates2D::new((col + subtile * 32) as i32, row as i32),
                            );
                            tma::bulk_commit_group();
                            tma::bulk_wait_group_read::<1>();
                        }
                        mbarrier::barrier_cta_sync(1, 128);
                        subtile += 1;
                    }
                    tcgen05::fence_before_thread_sync();
                    if warp::is_elected_sync(u32::MAX) {
                        cluster::arrive_remote(cluster::map_shared(acc_empty.shared_address(), 0));
                    }
                    phase ^= 1;
                    tile += tile_stride;
                }
                if warp_id == 0 {
                    if tid == 0 {
                        tma::bulk_wait_group_read::<0>();
                    }
                    mbarrier::barrier_cta_sync(3, 32);
                    let dealloc = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(dealloc_ptr);
                    cluster::arrive_remote(cluster::map_shared(dealloc.shared_address(), rank ^ 1));
                    wait(&dealloc, 0);
                    acc.dealloc_collective();
                }
            }
        }
    }
}
