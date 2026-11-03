// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use ctm_rs::cuda_module;

// Keep this boundary present: concrete shared pointers must adapt to a
// generic Rust helper ABI without changing pointee, mutability or pointer kind.
#[inline(never)]
fn generic_shared_identity(pointer: *mut u8) -> *mut u8 {
    pointer
}

#[cuda_module]
mod probes {
    use ctm_rs::{
        async_copy, barrier,
        cluster::{self, CtaMask},
        convert, fence, float,
        groups::CtaGroup2,
        kernel, launch_bounds, launch_contract,
        mbarrier::{self, Barrier, BarrierAddressExt, ManagedBarrier, Ready, TmaBarrier, Uninit},
        memory::{
            DisjointSlice, DynamicSharedArray, F16Tile128, F16TileB80, F16TileB96, F16TileC,
            SharedRegion,
        },
        tcgen05::{
            self,
            descriptors::*,
            mma::{self, collector_a},
            tmem::{self, ColumnOffset, TmemGuard, TmemReady, TmemUninit},
        },
        thread,
        tma::{self, TensorMap},
        warp,
    };
    use cuda_device::{SharedArray, cluster_launch};

    static mut SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
    static mut BAR: Barrier = Barrier::UNINIT;
    static mut A: SharedArray<f16, 8192, 1024> = SharedArray::UNINIT;
    static mut B0: SharedArray<f16, 6144, 1024> = SharedArray::UNINIT;
    static mut B1: SharedArray<f16, 5120, 1024> = SharedArray::UNINIT;
    static mut C: SharedArray<f16, 4096, 512> = SharedArray::UNINIT;

    #[kernel]
    pub fn primitives_grid_constant(#[grid_constant] map: &TensorMap<f16, F16Tile128>) {
        unsafe { tma::prefetch_tensormap(map) };
    }

    // These instruction probes are compile-only. Their straight-line issue
    // sequence intentionally does not implement a launchable collective kernel.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub fn primitives_lifecycle() {
        unsafe {
            let slot = SharedArray::as_raw_mut_ptr(&raw mut SLOT);
            let guard = TmemGuard::<TmemUninit, 512, CtaGroup2>::from_raw_slot(slot);
            let guard = guard.alloc_collective();
            tcgen05::relinquish_alloc_permit_cg2();
            let _dead = guard.dealloc_collective();
            let bar =
                ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR).init_instruction(1);
            let _dead = bar.invalidate_instruction();
        }
    }

    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub fn primitives_tma(
        #[grid_constant] a_map: &TensorMap<f16, F16Tile128>,
        #[grid_constant] b0_map: &TensorMap<f16, F16TileB96>,
        #[grid_constant] b1_map: &TensorMap<f16, F16TileB80>,
        #[grid_constant] c_map: &TensorMap<f16, F16TileC>,
        out: *mut u64,
    ) {
        unsafe {
            let Ok(a) = SharedRegion::<F16Tile128>::from_array(&*(&raw const A)) else {
                return;
            };
            let Ok(b0) = SharedRegion::<F16TileB96>::from_array(&*(&raw const B0)) else {
                return;
            };
            let Ok(b1) = SharedRegion::<F16TileB80>::from_array(&*(&raw const B1)) else {
                return;
            };
            let Ok(c) = SharedRegion::<F16TileC>::from_array(&*(&raw const C)) else {
                return;
            };
            let bar = ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(&raw mut BAR);
            let completion = cluster::mapa(bar.shared_address(), 0);
            let Ok(mask) = CtaMask::<CtaGroup2>::try_from_bits(0b11) else {
                return;
            };
            let coords = tma::Coordinates2D::new(-32, 4096);
            tma::cp_async_bulk_tensor_shared_cluster_global(&a, a_map, coords, completion, mask);
            tma::cp_async_bulk_tensor_shared_cluster_global(&b0, b0_map, coords, completion, mask);
            tma::cp_async_bulk_tensor_shared_cluster_global(&b1, b1_map, coords, completion, mask);
            tma::cp_async_bulk_tensor_global_shared_cta(
                &c,
                c_map,
                tma::Coordinates2D::new(64, 128),
            );
            tma::cp_async_bulk_commit_group();
            tma::cp_async_bulk_wait_group_read::<0>();
            tma::cp_async_bulk_wait_group_read::<7>();
            out.write(mbarrier::arrive(&bar).raw());
            out.add(1)
                .write(mbarrier::arrive_expect_tx(&bar, 16384).raw());
            out.add(2).write(mbarrier::try_wait_parity(&bar, 1) as u64);
            out.add(3)
                .write(mbarrier::arrive_expect_tx_cluster(&bar, 77824).raw());
            out.add(4)
                .write(mbarrier::try_wait_parity_cluster(&bar, 0) as u64);
            cluster::arrive_remote(cluster::mapa(bar.shared_address(), 1));
            tcgen05::commit_multicast_cg2(bar.shared_address(), mask);
        }
    }

    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub fn primitives_sync(out: *mut u64) {
        unsafe {
            let shared = DynamicSharedArray::<u8, 1024>::get_raw();
            out.add(3)
                .write(super::generic_shared_identity(shared) as u64);
            fence::fence_mbarrier_init();
            fence::fence_proxy_async_shared_cta();
            tcgen05::fence_before_thread_sync();
            tcgen05::fence_after_thread_sync();
            barrier::barrier_cta_sync(3, 128);
            barrier::barrier_cta_sync_aligned(3, 128);
            barrier::barrier_cta_arrive(3, 128);
            barrier::barrier_cta_arrive_aligned(3, 128);
            cluster::barrier_cluster_arrive();
            cluster::barrier_cluster_wait();
            cluster::barrier_cluster_arrive_aligned();
            cluster::barrier_cluster_wait_aligned();
            cluster::barrier_cluster_arrive_relaxed();
            cluster::barrier_cluster_arrive_relaxed_aligned();
            let (lane, elected) = warp::elect_sync(u32::MAX);
            out.write(lane as u64);
            out.add(1).write(elected as u64);
            out.add(2).write(
                (warp::warp_id()
                    + thread::blockIdx_x()
                    + cluster::cluster_ctarank()
                    + cluster::cluster_size()) as u64,
            );
        }
    }

    // Distinct constants expose operand permutations and exact math modifiers
    // in the exported IR; these probes are never launched.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn primitives_attention(mut out: DisjointSlice<f32>, packed: *mut u32) {
        use ctm_rs::mma::{Accumulator16x8, F16A, F16B, mma_f16};
        unsafe {
            async_copy::cp_async_commit_group();
            async_copy::cp_async_wait_group::<0>();
            async_copy::cp_async_wait_group::<7>();
            let out = out.as_mut_ptr();
            out.write(float::ex2_approx_f32(1.25));
            out.add(1).write(float::ex2_approx_ftz_f32(2.5));
            out.add(2).write(float::rcp_approx_ftz_f32(4.0));
            out.add(3).write(warp::shuffle_xor_f32(8.0, 2));
            packed.write(convert::cvt_f16x2_f32(1.25, 2.5));
            let c = Accumulator16x8::from_registers([1.0, 2.0, 3.0, 4.0]);
            let a = F16A::from_registers([0x3c003c00, 0x40004000, 0x42004200, 0x44004400]);
            let b = F16B::from_registers([0x46004600, 0x48004800]);
            let registers = mma_f16(c, a, b).registers();
            out.add(4).write(registers[0]);
            out.add(5).write(registers[1]);
            out.add(6).write(registers[2]);
            out.add(7).write(registers[3]);
        }
    }

    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub fn primitives_mma(out: *mut f32) {
        type Left = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
        type Right = DenseSharedF16<CtaGroup2, 256, 160, KMajor, KMajor>;
        unsafe {
            let Ok(a) = SharedRegion::<F16Tile128>::from_array(&*(&raw const A)) else {
                return;
            };
            let Ok(b0) = SharedRegion::<F16TileB96>::from_array(&*(&raw const B0)) else {
                return;
            };
            let Ok(b1) = SharedRegion::<F16TileB80>::from_array(&*(&raw const B1)) else {
                return;
            };
            let Ok(a) = SmemOperand::<OperandA, F16Tile128>::from_region(&a) else {
                return;
            };
            let Ok(b0) = SmemOperand::<OperandB, F16TileB96>::from_region(&b0) else {
                return;
            };
            let Ok(b1) = SmemOperand::<OperandB, F16TileB80>::from_region(&b1) else {
                return;
            };
            let guard = TmemGuard::<TmemReady, 512, CtaGroup2>::adopt_ready(
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
            );
            let Ok(left) = guard.view::<<Left as MmaSpec>::DstFormat>(ColumnOffset::new(0)) else {
                return;
            };
            let Ok(right) = guard.view::<<Right as MmaSpec>::DstFormat>(ColumnOffset::new(192))
            else {
                return;
            };
            let l = InstructionDescriptor::<Left>::new();
            let r = InstructionDescriptor::<Right>::new();
            mma::mma::<Left, collector_a::Fill>(&left, &a.k16::<0>(), &b0.k16::<0>(), &l, false);
            mma::mma::<Left, collector_a::Use>(&left, &a.k16::<1>(), &b0.k16::<1>(), &l, true);
            mma::mma::<Left, collector_a::LastUse>(&left, &a.k16::<2>(), &b0.k16::<2>(), &l, true);
            mma::mma::<Left, collector_a::Discard>(&left, &a.k16::<3>(), &b0.k16::<3>(), &l, true);
            mma::mma::<Right, collector_a::Fill>(&right, &a.k16::<0>(), &b1.k16::<0>(), &r, false);
            mma::mma::<Right, collector_a::Use>(&right, &a.k16::<1>(), &b1.k16::<1>(), &r, true);
            mma::mma::<Right, collector_a::LastUse>(
                &right,
                &a.k16::<2>(),
                &b1.k16::<2>(),
                &r,
                true,
            );
            mma::mma::<Right, collector_a::Discard>(
                &right,
                &a.k16::<3>(),
                &b1.k16::<3>(),
                &r,
                true,
            );
            let Ok(read) = guard.read_32x32(96, ColumnOffset::new(192)) else {
                return;
            };
            let raw = mma::load_32x32b_x32(&read);
            tmem::load_wait();
            let regs = raw.decode_f32();
            macro_rules! store { ($($i:literal),*) => { $(out.add($i).write(regs.get::<$i>());)* }; }
            store!(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31
            );
            let c = SharedArray::as_raw_mut_ptr(&raw mut C);
            c.write(regs.get::<0>() as f16);
        }
    }
}
