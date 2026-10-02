/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

//! Block-scaled MXFP4 GEMM through cute layouts.
//!
//! C(MxN, f16) = A(MxK, e2m1) x B(KxN, e2m1), with one ue8m0 scale per
//! 32-element K-block per row of A and per column of B. Four FP4 values share
//! one 16-bit carrier, and one typed 16x8x64 block-scaled MMA consumes two
//! scale groups per operation.
//!
//! A typed three-stage TMA pipeline overlaps lookahead loads with the existing
//! MMA mapping and tiled FP16 TMA epilogue:
//!
//! ```text
//! host: packed A/B + canonical SFA/SFB
//!             |
//!             v
//!       four typed TMA-load descriptors + one TMA-store descriptor
//!             |
//!             v
//! resident CTAs: blockIdx.x, then += gridDim.x
//!             |
//!             v
//! for each scheduled 128x128 output tile {
//!   warp 8, lane 0: fill the three-slot ring until EMPTY blocks
//!                     each stage is 17,408 bytes from four TMA loads
//!   warps 0..7: wait FULL, then consume two packed K64 steps
//!                accumulators.accumulate_k64(...)
//!   typed epilogue: AccC -> f16x2 -> stmatrix -> shared
//!   warp 0, lane 0: two 128x64 TMA stores -> global C
//! }
//! producer/consumer slot + phase survive between output tiles
//! ```
//!
//! `TmaDesc<T, Layout>` makes the host encoder and device copy name the same
//! placement. The byte TMA view and f16 consumer overlay encode the same B64
//! swizzle; the scale view keeps the hardware's canonical atom unchanged.
//!
//! B stages N-major (columns of B, k contiguous), so both operands share
//! one tile type and both copies land through the same 64-byte swizzle.
//!
//! Why B needs no transposed source view: storing B as N x K gives both
//! operands K-contiguous packed rows. Each f16 carrier holds four consecutive
//! nibbles, low to high, and the backend selects the matching fragment loads
//! and same-format block-scaled MMA.
//!
//! Data and scales are chosen so every f32 intermediate is exact and the
//! comparison is bitwise. The accelerated operation only exists on
//! sm_120/121; other GPUs report the requirement and skip execution. After
//! correctness, CUDA events measure 31 single launches following 100 warmups.
//!
//! Build and run with:
//!   cargo oxide run blockscale_gemm_cute --arch sm_120a

use core::{marker::PhantomData, mem::size_of};
use cuda_core::{CudaContext, LaunchConfig1D};
use cuda_device::barrier::Barrier;
use cuda_device::{
    DynamicSharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};
use cute_rs::tma::{TmaEncodeOptions, make_tma_desc_2d_with_options};
use cute_rs::{
    C, Composed, Consumer, L, Mkl, Mxf4AccumulatorTile2x8, Mxf4E2M1, Mxfp4TiledMma, Nkl,
    PipelineState, Producer, RowMajor, SharedScaleAtom, SharedTensor, Sm1xxBlockScaleKMajor,
    Sm120Epilogue128x128, Sm120EpilogueHalfLayout, Sm120ScaleAtom, SmemTile,
    StaticPersistentTileScheduler, Swizzle, T2, TmaDesc, TmaLoadPipeline, TmaStorePipeline,
    copy_tma_2d, copy_tma_s2g_2d,
};

const M: usize = 2048;
const N: usize = 2048;
const K: usize = 1024; // logical e2m1 elements; four packed into each f16 carrier
const KB: usize = K / 32; // K-blocks: one ue8m0 scale per 32 elements
const COMPUTE_WARPS: usize = 8; // 4 along M, 2 along N
const PRODUCER_WARP: usize = COMPUTE_WARPS;
const THREADS: usize = (COMPUTE_WARPS + 1) * 32;
const BLOCK_M: usize = 128;
const BLOCK_N: usize = 128;
const BATCHES: usize = 1;
type TileScheduler = StaticPersistentTileScheduler<16, 16, BATCHES>;

// TMA moves one 128-row x 64-byte packed-FP4 stage per operand. The consumer
// overlays those exact bytes as 128x32 f16 carriers for ldmatrix.
type TileP = L<T2<C<128>, C<32>>, T2<C<32>, C<1>>>;
type SmemP = Composed<Swizzle<2, 3, 3>, 0, TileP>;
type ByteSmem = Composed<Swizzle<2, 4, 3>, 0, RowMajor<128, 64>>;
type ScaleTmaSmem = RowMajor<1, 256>;
type SharedA = SharedTensor<Mxf4E2M1, f16, SmemP, Mkl>;
type SharedB = SharedTensor<Mxf4E2M1, f16, SmemP, Nkl>;

const K_STAGES: usize = KB / 4;
const DATA_STAGE_BYTES: usize = BLOCK_M * 64;
const SCALE_STAGE_BYTES: usize = Sm120ScaleAtom::BYTES;
const TMA_TX_BYTES: u32 = (2 * DATA_STAGE_BYTES + 2 * SCALE_STAGE_BYTES) as u32;
const PIPELINE_STAGES: usize = 3;
type MainloopPipeline = TmaLoadPipeline<PIPELINE_STAGES, 8, TMA_TX_BYTES>;
type EpiloguePipeline = TmaStorePipeline<1>;

// One 1024-byte-aligned dynamic allocation, partitioned without padding
// surprises. Each operand owns a three-slot ring, followed by the six 8-byte
// full/empty barriers, alignment padding, and the 32 KiB FP16 epilogue tile.
const SMEM_A_OFFSET: usize = 0;
const SMEM_B_OFFSET: usize = SMEM_A_OFFSET + PIPELINE_STAGES * DATA_STAGE_BYTES;
const SMEM_SFA_OFFSET: usize = SMEM_B_OFFSET + PIPELINE_STAGES * DATA_STAGE_BYTES;
const SMEM_SFB_OFFSET: usize = SMEM_SFA_OFFSET + PIPELINE_STAGES * SCALE_STAGE_BYTES;
const PIPELINE_OFFSET: usize = SMEM_SFB_OFFSET + PIPELINE_STAGES * SCALE_STAGE_BYTES;
const MAINLOOP_SMEM_BYTES: usize = PIPELINE_OFFSET + MainloopPipeline::STORAGE_BYTES;
const EPILOGUE_OFFSET: usize =
    MAINLOOP_SMEM_BYTES.next_multiple_of(Sm120Epilogue128x128::ALIGNMENT);
const DYNAMIC_SMEM_BYTES: usize = EPILOGUE_OFFSET + Sm120Epilogue128x128::BYTES;
const _: () = {
    assert!(M / BLOCK_M == 16);
    assert!(N / BLOCK_N == 16);
    assert!(TileScheduler::TOTAL_TILES == 256);
    assert!(TMA_TX_BYTES == 17_408);
    assert!(SMEM_A_OFFSET == 0);
    assert!(SMEM_B_OFFSET == 24_576);
    assert!(SMEM_SFA_OFFSET == 49_152);
    assert!(SMEM_SFB_OFFSET == 50_688);
    assert!(PIPELINE_OFFSET == 52_224);
    assert!(SMEM_A_OFFSET.is_multiple_of(1024));
    assert!(SMEM_B_OFFSET.is_multiple_of(1024));
    assert!(SMEM_SFA_OFFSET.is_multiple_of(512));
    assert!(SMEM_SFB_OFFSET.is_multiple_of(512));
    assert!(PIPELINE_OFFSET.is_multiple_of(MainloopPipeline::STORAGE_ALIGN));
    assert!(MAINLOOP_SMEM_BYTES == 52_272);
    assert!(EPILOGUE_OFFSET == 53_248);
    assert!(EPILOGUE_OFFSET.is_multiple_of(Sm120Epilogue128x128::ALIGNMENT));
    assert!(DYNAMIC_SMEM_BYTES == 86_016);
};

#[cuda_module]
mod kernels {
    use super::*;

    /// Four typed input tensor maps fill each 17,408-byte mainloop stage. The
    /// output tensor map carries the complete FP16 epilogue tile to global C.
    #[launch_bounds(288)]
    #[launch_contract(
        domain = 1,
        block = (288, 1, 1),
        dynamic_shared = 86016,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (12, 0),
    )]
    #[kernel]
    pub fn blockscale_gemm(
        a_tma: *const TmaDesc<u8, ByteSmem>,
        b_tma: *const TmaDesc<u8, ByteSmem>,
        sfa_tma: *const TmaDesc<u16, ScaleTmaSmem>,
        sfb_tma: *const TmaDesc<u16, ScaleTmaSmem>,
        out_tma: *const TmaDesc<f16, Sm120EpilogueHalfLayout>,
    ) {
        let tid = thread::threadIdx_x();

        // All aliases are fixed offsets into one externally sized allocation.
        // The launch contract and host LaunchConfig both reserve exactly the
        // same byte count and request the same 1024-byte base alignment.
        let dynamic_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = unsafe { dynamic_smem.add(SMEM_A_OFFSET) };
        let smem_b_ptr = unsafe { dynamic_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_ptr = unsafe { dynamic_smem.add(SMEM_SFA_OFFSET) };
        let smem_sfb_ptr = unsafe { dynamic_smem.add(SMEM_SFB_OFFSET) };
        let pipeline_base = unsafe { dynamic_smem.add(PIPELINE_OFFSET).cast::<Barrier>() };
        let pipeline = unsafe { MainloopPipeline::from_raw_base(pipeline_base) };
        let epilogue = unsafe {
            Sm120Epilogue128x128::from_raw(dynamic_smem.add(EPILOGUE_OFFSET).cast::<f16>())
        };
        unsafe { pipeline.init(0) };

        let warp_id = warp::warp_id() as usize;
        let lane = tid % 32;
        if warp_id < COMPUTE_WARPS {
            // CuTe's (4,2) MMA-atom layout. The 128-M/N permutations give
            // every compute warp two M atoms and four separated N16 pairs.
            let wm = warp_id % 4;
            let wn = warp_id / 4;
            let tiled_mma = Mxfp4TiledMma::<SmemP>::get_slice(lane);
            let warp_epilogue = unsafe { epilogue.get_slice(warp_id, lane as u32) };
            let epilogue_pipeline = EpiloguePipeline::new();
            let mut scheduler = TileScheduler::new_1d();
            // Slot and phase continue across output tiles. With eight K stages
            // and a three-slot ring, resetting this state per tile would
            // restart at the wrong barrier and can deadlock.
            let mut consumer_state = PipelineState::<Consumer, PIPELINE_STAGES>::new();
            while scheduler.has_work() {
                let work_tile = scheduler.current_tile();
                let (_, m_tile, n_tile, batch) = work_tile.coordinates();
                let mut accumulators = Mxf4AccumulatorTile2x8::zero();

                // Only the logical K coordinate resets per output tile.
                let mut stage = 0usize;
                while stage < K_STAGES {
                    unsafe { pipeline.consumer_wait(&consumer_state) };

                    let consumer_slot = consumer_state.slot();
                    let a_stage_ptr = unsafe { smem_a_ptr.add(consumer_slot * DATA_STAGE_BYTES) };
                    let b_stage_ptr = unsafe { smem_b_ptr.add(consumer_slot * DATA_STAGE_BYTES) };
                    let sfa_stage_ptr =
                        unsafe { smem_sfa_ptr.add(consumer_slot * SCALE_STAGE_BYTES) };
                    let sfb_stage_ptr =
                        unsafe { smem_sfb_ptr.add(consumer_slot * SCALE_STAGE_BYTES) };

                    // SAFETY: whole warps in uniform control flow; windows lie
                    // inside the completed TMA stage. The byte and f16 layouts
                    // encode the same B64 physical swizzle; scales preserve the
                    // SM120 atom.
                    unsafe {
                        let s_a = SharedA::from_raw_parts(a_stage_ptr.cast::<f16>(), BLOCK_M * 32);
                        let s_b = SharedB::from_raw_parts(b_stage_ptr.cast::<f16>(), BLOCK_N * 32);
                        let s_sfa = SharedScaleAtom::<Mkl>::from_raw_parts(
                            sfa_stage_ptr.cast::<u32>(),
                            Sm120ScaleAtom::WORDS,
                        );
                        let s_sfb = SharedScaleAtom::<Nkl>::from_raw_parts(
                            sfb_stage_ptr.cast::<u32>(),
                            Sm120ScaleAtom::WORDS,
                        );
                        let scale_tile = tiled_mma.load_scale_atom_128(&s_sfa, &s_sfb, wm, wn);
                        // Two packed K=64 steps per stage: window column kk
                        // selects one.
                        let mut kk = 0usize;
                        while kk < 2 {
                            // SAFETY: the fixed loop visits exactly the two K64
                            // halves of this K128 stage.
                            let scales = scale_tile.pairs_at_unchecked(kk);
                            let a_fragments = tiled_mma.load_a_128(&s_a, wm, kk);
                            let b_fragments = tiled_mma.get_b_tile_k64(&s_b, wn, kk);
                            accumulators.accumulate_k64(
                                &tiled_mma,
                                a_fragments,
                                b_fragments,
                                scales,
                            );
                            kk += 1;
                        }
                    }
                    // Every compute warp calls uniformly; the pipeline elects
                    // one lane-zero arrival, giving EMPTY eight arrivals.
                    unsafe { pipeline.consumer_release(&consumer_state) };
                    consumer_state.advance();
                    stage += 1;
                }

                // A single shared epilogue buffer is reused by every logical
                // tile assigned to this persistent CTA. Lane zero waits for
                // the previous TMA store's shared reads; the counted barrier
                // releases only the eight compute warps after that wait.
                if tid == 0 {
                    epilogue_pipeline.producer_acquire();
                }
                unsafe {
                    epilogue.sync_reusable();

                    warp_epilogue.store_tile(accumulators);

                    // This semantic boundary publishes every writer's
                    // generic-proxy stores before the counted hand-off to
                    // the one TMA issuer.
                    epilogue.sync_ready_for_tma();
                }

                if tid == 0 {
                    let c_outer_m = batch * (M / BLOCK_M) + m_tile;
                    let c_outer_n = n_tile * 2;
                    unsafe {
                        let left = epilogue.tma_half::<0>();
                        copy_tma_s2g_2d(out_tma, (c_outer_m, c_outer_n), left);
                        let right = epilogue.tma_half::<1>();
                        copy_tma_s2g_2d(out_tma, (c_outer_m, c_outer_n + 1), right);
                    }
                    epilogue_pipeline.producer_commit();
                }
                scheduler.advance();
            }
            if tid == 0 {
                epilogue_pipeline.producer_tail();
            }
        } else {
            // Lane zero is the single TMA issuer; the other producer-warp
            // lanes remain inactive after the uniform pipeline setup.
            if warp_id == PRODUCER_WARP && lane == 0 {
                let mut scheduler = TileScheduler::new_1d();
                // Keep the circular cursor live across all scheduled output
                // tiles; only the descriptor's logical K coordinate resets.
                let mut producer_state = PipelineState::<Producer, PIPELINE_STAGES>::new();
                while scheduler.has_work() {
                    let work_tile = scheduler.current_tile();
                    let (_, m_tile, n_tile, batch) = work_tile.coordinates();
                    let a_outer = batch * (M / BLOCK_M) + m_tile;
                    let b_outer = batch * (N / BLOCK_N) + n_tile;
                    let mut stage = 0usize;
                    while stage < K_STAGES {
                        let producer_slot = producer_state.slot();
                        let a_stage_ptr =
                            unsafe { smem_a_ptr.add(producer_slot * DATA_STAGE_BYTES) };
                        let b_stage_ptr =
                            unsafe { smem_b_ptr.add(producer_slot * DATA_STAGE_BYTES) };
                        let sfa_stage_ptr =
                            unsafe { smem_sfa_ptr.add(producer_slot * SCALE_STAGE_BYTES) };
                        let sfb_stage_ptr =
                            unsafe { smem_sfb_ptr.add(producer_slot * SCALE_STAGE_BYTES) };

                        unsafe {
                            pipeline.producer_acquire(&producer_state);
                            let full_barrier = pipeline.producer_expect_tx(&producer_state);
                            let mut a_tile = SmemTile::<u8, ByteSmem> {
                                base: a_stage_ptr,
                                capacity: DATA_STAGE_BYTES,
                                layout: PhantomData,
                            };
                            let mut b_tile = SmemTile::<u8, ByteSmem> {
                                base: b_stage_ptr,
                                capacity: DATA_STAGE_BYTES,
                                layout: PhantomData,
                            };
                            let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem> {
                                base: sfa_stage_ptr.cast::<u16>(),
                                capacity: SCALE_STAGE_BYTES / size_of::<u16>(),
                                layout: PhantomData,
                            };
                            let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem> {
                                base: sfb_stage_ptr.cast::<u16>(),
                                capacity: SCALE_STAGE_BYTES / size_of::<u16>(),
                                layout: PhantomData,
                            };
                            copy_tma_2d(a_tma, (a_outer, stage), &mut a_tile, full_barrier);
                            copy_tma_2d(b_tma, (b_outer, stage), &mut b_tile, full_barrier);
                            copy_tma_2d(
                                sfa_tma,
                                (a_outer * K_STAGES + stage, 0),
                                &mut sfa_tile,
                                full_barrier,
                            );
                            copy_tma_2d(
                                sfb_tma,
                                (b_outer * K_STAGES + stage, 0),
                                &mut sfb_tile,
                                full_barrier,
                            );
                        }
                        producer_state.advance();
                        stage += 1;
                    }
                    scheduler.advance();
                }
                // Drain only once, after the final scheduled tile. Per-tile
                // tails would destroy overlap between consecutive work tiles.
                unsafe { pipeline.producer_tail(producer_state) };
            }
        }
    }
}

/// The eight non-negative e2m1 values, indexed by nibble 0x0..=0x7;
/// nibbles 0x8..=0xF are their negations (0x8 is -0.0).
const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1_decode(nibble: u8) -> f32 {
    let magnitude = E2M1_VALUES[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Small deterministic hash used only to build the host fixture.
///
/// Mixing both coordinates prevents tile-sized row/column permutations from
/// hiding behind a short periodic input pattern.
fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

/// Deterministic data covering all 16 nibbles (zeros, negatives, both -0).
fn a_nibble(m: usize, k: usize) -> u8 {
    (mix32((m as u32).wrapping_mul(0x9e37_79b9) ^ k as u32 ^ 0xa511_e9b3) & 0xf) as u8
}

fn b_nibble(k: usize, n: usize) -> u8 {
    (mix32((n as u32).wrapping_mul(0x85eb_ca6b) ^ k as u32 ^ 0x63d8_3595) & 0xf) as u8
}

/// ue8m0 scale bytes: 2^(byte-127). 126..=128 keeps every product a small
/// binary fraction that `f32` represents exactly.
fn sfa_byte(m: usize, kb: usize) -> u8 {
    126 + (mix32((m as u32).wrapping_mul(0xc2b2_ae35) ^ kb as u32) % 3) as u8
}

fn sfb_byte(n: usize, kb: usize) -> u8 {
    126 + (mix32((n as u32).wrapping_mul(0x27d4_eb2f) ^ kb as u32 ^ 0x1656_67b1) % 3) as u8
}

fn ue8m0_decode(byte: u8) -> f32 {
    2f32.powi(byte as i32 - 127)
}

/// Pack four increasing-K FP4 nibbles into each 16-bit transport carrier.
fn pack_mxf4_view(nibbles: &[u8]) -> Vec<f16> {
    nibbles
        .as_chunks::<4>()
        .0
        .iter()
        .map(|q| {
            f16::from_bits(
                (q[0] as u16 & 0xf)
                    | ((q[1] as u16 & 0xf) << 4)
                    | ((q[2] as u16 & 0xf) << 8)
                    | ((q[3] as u16 & 0xf) << 12),
            )
        })
        .collect()
}

/// Reorder dense `[row][K/32]` scales into SM120's 128-row x four-group
/// atoms, then expose the exact bytes as the u16 carrier used by TMA.
fn pack_canonical_scales(dense: &[u8], rows: usize) -> Vec<u16> {
    let layout = Sm1xxBlockScaleKMajor::<32>::new(rows, K);
    let mut packed = vec![0u8; layout.storage_len(1)];
    for row in 0..rows {
        for group in 0..KB {
            packed[layout.offset(0, row, group)] = dense[row * KB + group];
        }
    }
    packed
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn percentile(sorted: &[f32], numerator: usize, denominator: usize) -> f32 {
    assert!(!sorted.is_empty());
    assert!(denominator > 0 && numerator <= denominator);
    let scaled = (sorted.len() - 1)
        .checked_mul(numerator)
        .expect("percentile index overflow");
    sorted[(scaled + denominator / 2) / denominator]
}

fn main() {
    println!("=== Block-scaled MXFP4 GEMM with cute layouts ===");
    println!("C({M}x{N}) = A({M}x{K} e2m1) x B({K}x{N} e2m1), ue8m0 scales per 32-element block\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    // Accelerated MXFP4 block-scaled MMA exists on SM120/121 targets.
    let (major, minor) = ctx.compute_capability().expect("query compute capability");
    println!("GPU Compute Capability: sm_{major}{minor}");
    if major != 12 {
        println!("skipping execution: mxf4 block-scale MMA requires sm_120/sm_121");
        return;
    }

    // Host-side packed FP4 payloads; B is stored transposed as N x K.
    let a_nibbles: Vec<u8> = (0..M * K).map(|i| a_nibble(i / K, i % K)).collect();
    let b_nibbles: Vec<u8> = (0..N * K).map(|i| b_nibble(i % K, i / K)).collect();
    let sfa: Vec<u8> = (0..M * KB).map(|i| sfa_byte(i / KB, i % KB)).collect();
    let sfb: Vec<u8> = (0..N * KB).map(|i| sfb_byte(i / KB, i % KB)).collect();

    // Exact f32 accumulator reference: per K-block, an exact 32-term dot times the two
    // power-of-two scales. Every partial sum is an exact binary fraction well
    // inside f32's exact range, so accumulation order cannot matter.
    println!("computing exact host reference ({M}x{N}x{K})...");
    let mut expected = vec![0.0f32; M * N];
    for m in 0..M {
        let a_row = &a_nibbles[m * K..(m + 1) * K];
        let sfa_row = &sfa[m * KB..(m + 1) * KB];
        for n in 0..N {
            let b_row = &b_nibbles[n * K..(n + 1) * K];
            let sfb_row = &sfb[n * KB..(n + 1) * KB];
            let mut total = 0.0f32;
            for kb in 0..KB {
                let mut block = 0.0f32;
                for j in 0..32 {
                    let k = kb * 32 + j;
                    block += e2m1_decode(a_row[k]) * e2m1_decode(b_row[k]);
                }
                total += ue8m0_decode(sfa_row[kb]) * ue8m0_decode(sfb_row[kb]) * block;
            }
            expected[m * N + n] = total;
        }
    }
    let expected_f16: Vec<f16> = expected.iter().copied().map(|value| value as f16).collect();

    let stream = ctx.default_stream();
    // SAFETY: this executable is the sole owner of the embedded device module.
    let module = unsafe { kernels::load(&ctx) }.expect("Failed to load embedded CUDA module");

    let a_packed = pack_mxf4_view(&a_nibbles);
    let b_packed = pack_mxf4_view(&b_nibbles);
    let sfa_packed = pack_canonical_scales(&sfa, M);
    let sfb_packed = pack_canonical_scales(&sfb, N);
    let a_dev = cuda_core::DeviceBuffer::from_host(&stream, &a_packed).unwrap();
    let b_dev = cuda_core::DeviceBuffer::from_host(&stream, &b_packed).unwrap();
    let sfa_dev = cuda_core::DeviceBuffer::from_host(&stream, &sfa_packed).unwrap();
    let sfb_dev = cuda_core::DeviceBuffer::from_host(&stream, &sfb_packed).unwrap();

    // The descriptor types are the host/kernel agreement: A/B land in the
    // B64 byte layout, while each flattened scale row is one canonical atom.
    let a_desc: TmaDesc<u8, ByteSmem> = make_tma_desc_2d_with_options(
        a_dev.cu_deviceptr() as *mut core::ffi::c_void,
        M as u64,
        (K / 2) as u64,
        (K / 2) as u64,
        TmaEncodeOptions::L2_128B,
    )
    .expect("A TMA descriptor encoding failed");
    let b_desc: TmaDesc<u8, ByteSmem> = make_tma_desc_2d_with_options(
        b_dev.cu_deviceptr() as *mut core::ffi::c_void,
        N as u64,
        (K / 2) as u64,
        (K / 2) as u64,
        TmaEncodeOptions::L2_128B,
    )
    .expect("B TMA descriptor encoding failed");
    let sfa_atom_rows = (M / BLOCK_M) * K_STAGES;
    let sfb_atom_rows = (N / BLOCK_N) * K_STAGES;
    let sfa_desc: TmaDesc<u16, ScaleTmaSmem> = make_tma_desc_2d_with_options(
        sfa_dev.cu_deviceptr() as *mut core::ffi::c_void,
        sfa_atom_rows as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )
    .expect("SFA TMA descriptor encoding failed");
    let sfb_desc: TmaDesc<u16, ScaleTmaSmem> = make_tma_desc_2d_with_options(
        sfb_dev.cu_deviceptr() as *mut core::ffi::c_void,
        sfb_atom_rows as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )
    .expect("SFB TMA descriptor encoding failed");
    let a_desc_dev = cuda_core::DeviceBuffer::from_host(&stream, &a_desc.bytes).unwrap();
    let b_desc_dev = cuda_core::DeviceBuffer::from_host(&stream, &b_desc.bytes).unwrap();
    let sfa_desc_dev = cuda_core::DeviceBuffer::from_host(&stream, &sfa_desc.bytes).unwrap();
    let sfb_desc_dev = cuda_core::DeviceBuffer::from_host(&stream, &sfb_desc.bytes).unwrap();
    let a_desc_ptr = a_desc_dev.cu_deviceptr() as *const TmaDesc<u8, ByteSmem>;
    let b_desc_ptr = b_desc_dev.cu_deviceptr() as *const TmaDesc<u8, ByteSmem>;
    let sfa_desc_ptr = sfa_desc_dev.cu_deviceptr() as *const TmaDesc<u16, ScaleTmaSmem>;
    let sfb_desc_ptr = sfb_desc_dev.cu_deviceptr() as *const TmaDesc<u16, ScaleTmaSmem>;

    let out = cuda_core::DeviceBuffer::<f16>::zeroed(&stream, M * N).unwrap();
    let out_desc: TmaDesc<f16, Sm120EpilogueHalfLayout> = make_tma_desc_2d_with_options(
        out.cu_deviceptr() as *mut core::ffi::c_void,
        M as u64,
        N as u64,
        N as u64,
        TmaEncodeOptions::L2_128B,
    )
    .expect("C TMA descriptor encoding failed");
    let out_desc_dev = cuda_core::DeviceBuffer::from_host(&stream, &out_desc.bytes).unwrap();
    let out_desc_ptr = out_desc_dev.cu_deviceptr() as *const TmaDesc<f16, Sm120EpilogueHalfLayout>;
    // Ask the driver how many copies of this CTA fit on each SM. The
    // persistent scheduler needs that number to choose its physical grid.
    let probe = module
        .prepare_blockscale_gemm(LaunchConfig1D::new(
            1,
            THREADS as u32,
            DYNAMIC_SMEM_BYTES as u32,
        ))
        .expect("launch contract validation failed");
    let active_ctas = probe
        .function()
        .max_active_blocks_per_multiprocessor(THREADS as u32, DYNAMIC_SMEM_BYTES as u32)
        .expect("query active CTAs per SM");
    let sm_count = ctx.multiprocessor_count().expect("query SM count");
    let persistent_ctas = TileScheduler::resident_grid_size(sm_count, active_ctas)
        .expect("persistent grid must have nonzero u32 capacity");
    drop(probe);

    let config = LaunchConfig1D::new(persistent_ctas, THREADS as u32, DYNAMIC_SMEM_BYTES as u32);
    let launch = module
        .prepare_blockscale_gemm(config)
        .expect("persistent launch contract validation failed");
    // SAFETY: 288-thread blocks contain eight compute warps and one producer
    // warp. The grid-stride scheduler partitions all 256 output tiles into
    // disjoint CTA streams, and the warp/lane mapping partitions each tile.
    module
        .blockscale_gemm(
            &stream,
            &launch,
            a_desc_ptr,
            b_desc_ptr,
            sfa_desc_ptr,
            sfb_desc_ptr,
            out_desc_ptr,
        )
        .expect("kernel launch failed");
    let got = out.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    let mut nonzero = 0;
    for m in 0..M {
        for n in 0..N {
            if expected[m * N + n] != 0.0 {
                nonzero += 1;
            }
            if got[m * N + n].to_bits() != expected_f16[m * N + n].to_bits() {
                if errors < 5 {
                    eprintln!(
                        "  C({m},{n}): expected {} (0x{:04x}), got {} (0x{:04x})",
                        expected_f16[m * N + n],
                        expected_f16[m * N + n].to_bits(),
                        got[m * N + n],
                        got[m * N + n].to_bits(),
                    );
                }
                errors += 1;
            }
        }
    }
    if errors != 0 {
        println!("\n✗ FAILED: {errors} mismatches");
        std::process::exit(1);
    }
    println!(
        "Numeric check: all {} FP16 output bit patterns exact ({nonzero} nonzero FP32 accumulators), {} K-blocks",
        M * N,
        KB,
    );

    // Single-kernel device timing: every sample places exactly one launch
    // between CUDA events. The throughput count is 2*M*N*K FLOPs per launch.
    const WARMUP: usize = 100;
    const SAMPLES: usize = 31;
    for _ in 0..WARMUP {
        module
            .blockscale_gemm(
                &stream,
                &launch,
                a_desc_ptr,
                b_desc_ptr,
                sfa_desc_ptr,
                sfb_desc_ptr,
                out_desc_ptr,
            )
            .expect("warmup launch failed");
    }
    stream.synchronize().expect("warmup sync failed");

    let mut times_ms = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = ctx
            .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("create start event");
        let end = ctx
            .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("create end event");

        start.record(&stream).expect("record start event");
        module
            .blockscale_gemm(
                &stream,
                &launch,
                a_desc_ptr,
                b_desc_ptr,
                sfa_desc_ptr,
                sfb_desc_ptr,
                out_desc_ptr,
            )
            .expect("timed launch failed");
        end.record(&stream).expect("record end event");
        times_ms.push(start.elapsed_ms(&end).expect("event timing"));
    }
    times_ms.sort_by(f32::total_cmp);
    let median_ms = percentile(&times_ms, 1, 2);
    let p10_ms = percentile(&times_ms, 1, 10);
    let p90_ms = percentile(&times_ms, 9, 10);
    let tflops = (2.0 * (M * N * K) as f64) / (f64::from(median_ms) * 1e-3) / 1e12;
    println!(
        "Single-kernel device time: median={median_ms:.6} ms, p10={p10_ms:.6} ms, \
         p90={p90_ms:.6} ms ({SAMPLES} samples, {WARMUP} warmups); {tflops:.2} TFLOP/s"
    );

    println!("\n✓ SUCCESS: block-scaled MXFP4 GEMM is bit-exact through CuTe semantics");
}
