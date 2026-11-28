// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// 16-byte aligned 4-word (`128`-bit) register packet used by the epilogue to emit
/// a single vectorized `st.global.v4.b32` (`STG.E.128`) store of 8 packed `bf16` values.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct U32x4(
    /// First pair of `bf16` output elements (`cols + 0..1`).
    u32,
    /// Second pair of `bf16` output elements (`cols + 2..3`).
    u32,
    /// Third pair of `bf16` output elements (`cols + 4..5`).
    u32,
    /// Fourth pair of `bf16` output elements (`cols + 6..7`).
    u32,
);

/// Number of pipeline stages in the contiguous shared-memory `A`/`B` ring buffer (`7` stages).
/// Holds `7 × 16 KiB` of `A` tiles followed by `7 × 16 KiB` of `B` tiles (`224 KiB` total per CTA).
const NUM_SMEM_STAGES: usize = 7;

/// Byte capacity of one per-CTA `128 × 64` `f16` operand stage (`16,384` bytes = `16 KiB`).
/// Used to advance `SMEM_BUF` stage pointers for both `A` and `B` operands.
const SMEM_STAGE_BYTES: usize = 16_384;

/// Starting byte offset (`114,688` bytes = `7 × 16 KiB`) of the 7-stage `B` ring in `SMEM_BUF`.
/// Placed immediately after the 7 contiguous `A` stages (`0..114,688`).
const B_SMEM_OFFSET: usize = NUM_SMEM_STAGES * SMEM_STAGE_BYTES;

/// Byte size of one per-CTA `128 × 64` `f16` `A` tile transferred by 2D TMA (`16,384` bytes).
/// Across the 2-CTA cluster (`cta_group::2`), the pair loads a `256 × 64` `A` slice per K-step.
const A_TILE_BYTES: u32 = 128 * 64 * 2;

/// Byte size of one `32 × 64` `f16` `B` sub-panel (`4,096` bytes) used at `4096³` and on `N192` tail tiles.
/// Allows 2D TMA box transfers in multiples of `32` N-columns (`3 × 32 = 96` per CTA for `N192`).
const B_PANEL32_BYTES: usize = 32 * 64 * 2;

/// Byte size of one per-CTA `128 × 64` `f16` `B` tile for full `N = 256` output tiles (`16,384` bytes).
/// Across the 2-CTA cluster, both CTAs together receive `256 × 64` `f16` elements (`32 KiB`).
const B_TILE_BYTES_256: u32 = 128 * 64 * 2;

/// Byte size of one per-CTA `96 × 64` `f16` `B` tile for `N = 192` wave-tail tiles (`12,288` bytes).
/// Across the 2-CTA cluster, both CTAs together receive `192 × 64` `f16` elements (`24 KiB`).
const B_TILE_BYTES_192: u32 = 96 * 64 * 2;

/// Stride byte offset (`1,024` bytes) to the next core matrix down in a 128B-swizzled `tcgen05` SMEM layout.
/// Encoded into bits `32..45` of the 64-bit `tcgen05` shared-memory descriptor.
const SBO_BYTES: u32 = 1_024;

/// Leading byte offset (`16` bytes) to the next core matrix right in a 128B-swizzled `tcgen05` SMEM layout.
/// Encoded into bits `16..29` of the 64-bit `tcgen05` shared-memory descriptor.
const LBO_BYTES: u32 = 16;

/// `tcgen05` shared-memory descriptor swizzle code for 128-byte XOR swizzling (`2`).
/// Placed in bits `61..63` of the 64-bit `tcgen05` shared-memory descriptor.
const SWIZZLE_128B: u8 = 2;

/// Warp index (`4`) dedicated to the asynchronous 2-CTA TMA producer and CLC work-stealing loop.
/// Runs ahead of `MMA_WARP` across the 7-stage `SMEM_BUF` ring and 4-slot `TILE_INFO` mailbox.
const TMA_WARP: u32 = 4;

/// Warp index (`5`) dedicated to issuing `tcgen05.mma.cta_group::2` instructions on the leader CTA (`my_rank == 0`).
/// Consumes `SMEM_BUF` stages produced by `TMA_WARP` and commits accumulator stages to `Warps 0..3`.
const MMA_WARP: u32 = 5;

/// Number of `256`-column Tensor Memory (`TMEM`) accumulator stages (`2`: `TMEM_0` at `0..256`, `TMEM_1` at `256..512`).
/// Enables `MMA_WARP` to accumulate Tile `t + 1` into `TMEM_1` while `Warps 0..3` drain `TMEM_0`.
const NUM_ACCUM_STAGES: u32 = 2;

/// Number of `TMEM` columns per accumulator stage (`256` columns out of the 512-column CTA allocation).
/// Added to `tmem_addr` when selecting between stage `0` (`+0`) and stage `1` (`+256`).
const ACCUM_STAGE_COLS: u32 = 256;

/// Two-CTA cluster multicast mask (`0b11`) targeting both CTA rank `0` and CTA rank `1` in the pair.
/// Passed to `tcgen05_commit_multicast_cg2` so a single leader MMA commit arrives on both CTAs' mbarriers.
const CTA_MASK_PAIR: u16 = 0b11;

/// Bitmask (`0xFEFF_FFF8`) that clears the cluster-CTA rank bit (`bit 24`) on a shared-memory barrier address.
/// Used by `cp.async.bulk.tensor.2d...multicast::cluster.cta_group::2` to route TMA tx-completions to the leader CTA.
const PEER_BIT_MASK: u32 = 0xFEFF_FFF8;

/// Number of 2-CTA cluster pairs resident per wave on an NVIDIA B200 GPU (`148 SMs / 2 = 74` pairs).
/// Drives the wave-tail tile conversion math so the final GPU wave fills all `74` CTA pairs.
const B200_CLUSTER_PAIRS: u32 = 74;

/// Tile kind tag (`0`) for a full `M256 × N256` cooperative output tile (`128` N-columns per CTA).
/// Packed into bits `30..31` of `TILE_INFO[t_slot * 4 + 1]`.
const TILE_KIND_N256: u32 = 0;

/// Tile kind tag (`1`) for a wave-tail `M256 × N192` cooperative output tile (`96` N-columns per CTA).
/// Used at `4096³` where converting `3 × 256` triplets into `4 × 192` tiles fills all `74` CTA pairs.
const TILE_KIND_N192: u32 = 1;

/// Tile kind tag (`2`) for a wave-tail `M256 × N224` cooperative output tile (`112` N-columns per CTA).
/// Used at `8192³` and `16384³` where converting `7 × 256` heptets into `8 × 224` tiles fills all `74` CTA pairs.
const TILE_KIND_N224: u32 = 2;

/// Number of B200 Streaming Multiprocessors (`148` SMs = `2 × 74`).
/// Drives 1-CTA wave-tail tile balancing in `gemm_sol_short_k_m128n256` (`592 = 4 × 148` work items).
const B200_SMS: u32 = 148;

/// Number of shared-memory TMA pipeline stages (`4`) in `gemm_sol_short_k_m128n256`.
/// Reserves `4 × 48 KiB = 192 KiB` (`16 KiB A + 32 KiB B`), leaving `36 KiB` free for `18 KiB` `SMEM_EPI`.
const SHORT_SMEM_STAGES: usize = 4;

/// Per-stage `B` tile size (`256 × 64 × 2 = 32,768` bytes = `32 KiB`) in `gemm_sol_short_k_m128n256`.
/// Holds up to `4 × 64` (`N256`) or `3 × 64` (`N192`) `SWIZZLE_128B` `f16` columns per `K = 64` stage.
const SHORT_B_STAGE_BYTES: usize = 256 * 64 * 2;

/// Byte offset (`4 × 16,384 = 65,536`) where the 4-stage `B` ring begins inside `gemm_sol_short_k_m128n256`.
/// Follows immediately after the `4 × 16 KiB` `A` stages.
const SHORT_B_SMEM_OFFSET: usize = SHORT_SMEM_STAGES * SMEM_STAGE_BYTES;

/// Total TMA shared-memory ring size (`65,536 + 131,072 = 196,608` bytes = `192 KiB`) in `gemm_sol_short_k_m128n256`.
/// Combined with `18 KiB` `SMEM_EPI` (`210 KiB` total), fits cleanly inside `228 KiB` per B200 SM.
const SHORT_SMEM_TOTAL_BYTES: usize =
    SHORT_B_SMEM_OFFSET + SHORT_SMEM_STAGES * SHORT_B_STAGE_BYTES;

/// Padded row stride in `U32x4` (`16`-byte) packets (`9` packets = `8` data + `1` skew = `64` BF16s).
/// Shifts each row by `4` SMEM banks (`16` bytes) so `STS.128` and `LDS.128` have `0` bank conflicts.
const EPI_ROW_PKTS: usize = 9;

/// Per-warp buffer size in `U32x4` packets (`32` rows x `9` packets = `288` packets = `4,608` bytes).
/// Gives each of `Warps 0..3` a disjoint `4.5 KiB` bank-conflict-free `64`-column staging slice.
const EPI_WARP_PKTS: usize = 32 * EPI_ROW_PKTS;

/// Total `U32x4` capacity (`4 x 288 = 1,152` packets = `18,432` bytes = `18 KiB`) of `SMEM_EPI`.
/// Converts `32x32b.x64` zero-shuffle register tiles into `128`-byte (`1` full L2 cache line/row) coalesced bursts.
const EPI_TOTAL_PKTS: usize = 4 * EPI_WARP_PKTS;

/// Device kernel module containing the `M256×N256` (`4096³`/`8192³`) and `M512×N256` (`16384³`)
/// Blackwell speed-of-light (`sm_100a`) GEMM entries.
///
/// ```text
/// 2-CTA CLUSTER (`cta_group::2`) 6-WARP PIPELINE ON B200 (`sm_100a`)
/// ────────────────────────────────────────────────────────────────────────────────────────────
/// Warp 4 (TMA + CLC):  [ 7-Stage Multicast TMA Ring (224 KiB SMEM) + Overlapped CLC Steal   ]
/// Warp 5 (Leader MMA): [ Tile 0 -> TMEM_0 ] [ Tile 1 -> TMEM_1 ] [ Tile 2 -> TMEM_0 ] ...
/// Warps 0..3 (Epilog):                      [ ld(TMEM_0) -> Free TMEM_0 -> Butterfly -> U32x4]
/// ```
#[cuda_module]
mod kernels {
    use super::*;

    /// Cooperative 2-CTA (`cta_group::2`) Blackwell GEMM kernel specialized for the `M256×N256`
    /// resource envelope (`4096³` and `8192³`).
    ///
    /// # Pipeline Architecture
    /// - **Warp 4 (TMA Producer & CLC Scheduler)**: Streams `128 × 64` `A` and `B` tiles across a
    ///   7-stage shared-memory ring (`224 KiB`), issues overlapped `clc_try_cancel_multicast` 4 K-steps
    ///   before the end of each tile, and publishes tile descriptors into a 4-slot `TILE_INFO` mailbox.
    /// - **Warp 5 (Leader `tcgen05` MMA Consumer)**: On `my_rank == 0`, issues 4 `tcgen05.mma.cta_group::2`
    ///   instructions per K-stage into alternating 256-column `TMEM` halves (`TMEM_0` and `TMEM_1`),
    ///   selecting `M256×N256`, `M256×N192`, or `M256×N224` descriptors to balance wave-tail occupancy.
    /// - **Warps 0..3 (Register-Direct Vector Epilogue)**: Unloads FP32 accumulators via
    ///   `tcgen05.ld.16x256b.x8`, immediately releases the `TMEM` stage for the next tile, transposes
    ///   adjacent lane pairs via two warp-butterfly shuffles (`xor 1`, `xor 2`), and writes 128-bit
    ///   vectorized `U32x4` (`STG.E.128`) stores to global memory.
    ///
    /// # Safety
    /// Must be launched on an `sm_100a`-compatible GPU with a `2 × 1 × 1` cluster dimension, `192`
    /// threads per CTA (`6` warps), valid 2D `SWIZZLE_128B` TMA descriptors `a_tma` and `b_tma`,
    /// and an output buffer `out` of at least `M * N / 2` packed `u32` (`bf16x2`) elements.
    #[expect(
        clippy::collapsible_if,
        reason = "keep warp-uniform branch separate from lane-0 elect to avoid MIR short-circuit join block and preserve 65-register PTX"
    )]
    #[kernel(unchecked_indexing)]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_sol_clc_multicast_4_stage_pipeline(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        _tiles_n: u32,
    ) {
        // SAFETY: All shared-memory arrays, hardware barriers, TMA transfers, TMEM allocations,
        // and global output stores below obey the 2-CTA cluster synchronization protocol.
        unsafe {
            /// Contiguous `224 KiB` (`7 × 16 KiB` `A` + `7 × 16 KiB` `B`), 128-byte aligned SMEM ring.
            /// Written by `TMA_WARP` via 2D TMA multicast and consumed by `MMA_WARP` via `tcgen05.mma`.
            static mut SMEM_BUF: SharedArray<u8, 229376, 128> = SharedArray::UNINIT;

            /// Single 32-bit shared-memory slot holding the base `TMEM` column address allocated by `Warp 0`.
            /// Broadcast to all warps after the initial `cluster_sync()`.
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            /// Four-slot (`4 × 4` `u32`) tile mailbox passing `(tile_m, packed_col, has_work, pad)`
            /// from `TMA_WARP` to `MMA_WARP` and `Warps 0..3`.
            static mut TILE_INFO: SharedArray<u32, 16, 4> = SharedArray::UNINIT;

            /// Per-stage `mbarrier` array (`7` active stages) signaled when 2-CTA TMA loads complete.
            /// Waited on by `MMA_WARP` before issuing `tcgen05.mma` for that stage.
            static mut TMA_BAR: SharedArray<Barrier, 8, 8> = SharedArray::UNINIT;

            /// Per-stage `mbarrier` array (`7` active stages) signaled by `tcgen05_commit_multicast_cg2`
            /// when `MMA_WARP` finishes consuming a shared-memory stage.
            static mut MMA_BAR: SharedArray<Barrier, 8, 8> = SharedArray::UNINIT;

            /// Accumulator-full barrier for `TMEM` stage `0` (columns `0..256`), signaled after the K-loop.
            /// Waited on by `Warps 0..3` before unloading `TMEM_0`.
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;

            /// Accumulator-full barrier for `TMEM` stage `1` (columns `256..512`), signaled after the K-loop.
            /// Waited on by `Warps 0..3` before unloading `TMEM_1`.
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;

            /// Accumulator-empty barrier for `TMEM` stage `0` (arrival count `8`: 4 epilogue warps × 2 CTAs).
            /// Signaled by `lane_id == 0` of each epilogue warp immediately after the final `tcgen05.wait::ld`.
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;

            /// Accumulator-empty barrier for `TMEM` stage `1` (arrival count `8`: 4 epilogue warps × 2 CTAs).
            /// Signaled by `lane_id == 0` of each epilogue warp immediately after the final `tcgen05.wait::ld`.
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;

            /// Four-slot `mbarrier` array synchronizing the 4-slot `TILE_INFO` descriptor mailbox.
            /// Signaled by `TMA_WARP` once `TILE_INFO[t_slot * 4 .. +3]` is written.
            static mut TILE_READY: SharedArray<Barrier, 4, 8> = SharedArray::UNINIT;

            /// 16-byte (`2 × u64`) response buffer written by `clusterlaunchcontrol.try_cancel`.
            /// Read by `TMA_WARP` after `CLC_BAR` completes to decode the next stolen CTA ID.
            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;

            /// Hardware barrier tracking completion of the 16-byte multicast CLC `try_cancel` response.
            /// Waited on by `TMA_WARP` before querying `clc_query_is_canceled`.
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            let num_stages: usize = NUM_SMEM_STAGES;
            let b_smem_offset: usize = B_SMEM_OFFSET;
            let swizzle_g: u32 = if tiles_m == 16 { 2 } else { 8 };

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let my_rank = cluster::cluster_ctaidX();
            let self_mask: u16 = 1u16 << my_rank;

            // Phase 0: Initialize all shared-memory mbarriers on thread 0 and allocate 512 TMEM columns on Warp 0.
            if tid == 0 {
                let mut s = 0usize;
                while s < NUM_SMEM_STAGES {
                    mbarrier_init(&raw mut TMA_BAR[s], 1);
                    mbarrier_init(&raw mut MMA_BAR[s], 1);
                    s += 1;
                }
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 8);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 8);
                let mut ts = 0usize;
                while ts < 4 {
                    mbarrier_init(&raw mut TILE_READY[ts], 1);
                    ts += 1;
                }
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            cluster::cluster_sync();
            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR[0], 512);
            }
            cluster::cluster_sync();
            let tmem_addr = TMEM_ADDR[0];
            let elect_one_cta = my_rank == 0;

            let idesc_256 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N256)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_192 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 192 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_224 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 224 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;

            // Phase 1: Warp 4 (lane 0) — 7-Stage Multicast TMA Producer + Wave-Tail Balancing + Overlapped CLC.
            if warp_id == TMA_WARP {
                if lane_id == 0 {
                    let mut stage: usize = 0;
                    let mut mma_parity: u32 = 1;
                    let mut tile_seq: usize = 0;
                    let mut clc_iter: u32 = 0;

                    let cluster_base_id = thread::blockIdx_x() - my_rank;
                    let mut raw_tile_idx = cluster_base_id / 2;
                    let wide_tiles_n = _tiles_n / 2;
                    let wide_total = tiles_m * wide_tiles_n;
                    #[expect(
                        clippy::manual_div_ceil,
                        reason = "keep branchless (wide_total + 73) / 74 division to avoid extra remainder/select predicate in Warp 4"
                    )]
                    let num_waves = (wide_total + (B200_CLUSTER_PAIRS - 1)) / B200_CLUSTER_PAIRS;
                    let work_total = num_waves * B200_CLUSTER_PAIRS;
                    let num_convert = work_total - wide_total;
                    let use_n224 = tiles_m >= 32;
                    let full_tiles = if use_n224 {
                        wide_total - 7 * num_convert
                    } else {
                        wide_total - 3 * num_convert
                    };
                    let head_cols = if use_n224 {
                        wide_tiles_n - 7
                    } else if tiles_m == 16 {
                        wide_tiles_n - 9
                    } else {
                        wide_tiles_n - 3
                    };
                    let head_tiles = tiles_m * head_cols;
                    let resp_ptr = &raw mut CLC_RESPONSE[0];
                    let aliased_tma_bar0 = core::ptr::with_exposed_provenance_mut::<Barrier>(
                        (&raw mut TMA_BAR[0]).expose_provenance() & (PEER_BIT_MASK as usize),
                    );

                    loop {
                        let clc_parity = clc_iter & 1;
                        if raw_tile_idx < work_total {
                            // Map linear `raw_tile_idx` to serpentine L2 N-band coordinates or wave-tail N192/N224 tiles.
                            let (tile_m, col_bf16_base, tile_kind) = if raw_tile_idx < head_tiles {
                                let group_tiles = swizzle_g * tiles_m;
                                let group = raw_tile_idx / group_tiles;
                                let in_group = raw_tile_idx % group_tiles;
                                let n_start = group * swizzle_g;
                                let band_w = if swizzle_g < head_cols - n_start {
                                    swizzle_g
                                } else {
                                    head_cols - n_start
                                };
                                let raw_m = in_group / band_w;
                                let tm = if (group & 1) != 0 {
                                    tiles_m - 1 - raw_m
                                } else {
                                    raw_m
                                };
                                let tn = n_start + in_group % band_w;
                                (tm, tn * 256, TILE_KIND_N256)
                            } else if use_n224 {
                                if raw_tile_idx < full_tiles {
                                    let rem = raw_tile_idx - head_tiles;
                                    let t = rem / 7;
                                    let sub = rem % 7;
                                    (t, (head_cols + sub) * 256, TILE_KIND_N256)
                                } else {
                                    let tail = raw_tile_idx - full_tiles;
                                    let t = (full_tiles - head_tiles) / 7 + (tail / 8);
                                    let sub = tail & 7;
                                    (t, head_cols * 256 + sub * 224, TILE_KIND_N224)
                                }
                            } else if raw_tile_idx < full_tiles {
                                let rem = raw_tile_idx - head_tiles;
                                let t = rem / 3;
                                let sub = rem % 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn = head_cols + band * 3 + sub;
                                (tm, tn * 256, TILE_KIND_N256)
                            } else {
                                let tail = raw_tile_idx - full_tiles;
                                let t = (full_tiles - head_tiles) / 3 + (tail / 4);
                                let sub = tail & 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn_base = head_cols + band * 3;
                                (tm, tn_base * 256 + sub * 192, TILE_KIND_N192)
                            };

                            let packed_col = (col_bf16_base >> 5) | (tile_kind << 30);

                            // Publish tile coordinates to `MMA_WARP` and `Warps 0..3` via the 4-slot mailbox.
                            let t_slot = tile_seq & 3;
                            TILE_INFO[t_slot * 4] = tile_m;
                            TILE_INFO[t_slot * 4 + 1] = packed_col;
                            TILE_INFO[t_slot * 4 + 2] = 1;
                            mbarrier_arrive(&raw const TILE_READY[t_slot]);
                            tile_seq += 1;

                            let m_offset = (tile_m * 256 + my_rank * 128) as i32;
                            let b_n_offset = if tile_kind == TILE_KIND_N192 {
                                (col_bf16_base + my_rank * 96) as i32
                            } else if tile_kind == TILE_KIND_N224 {
                                (col_bf16_base + my_rank * 112) as i32
                            } else {
                                (col_bf16_base + my_rank * 128) as i32
                            };
                            let tx_bytes = if tile_kind == TILE_KIND_N192 {
                                (A_TILE_BYTES + B_TILE_BYTES_192) * 2
                            } else {
                                (A_TILE_BYTES + B_TILE_BYTES_256) * 2
                            };

                            let clc_prefetch_k = k_iters.saturating_sub(4);
                            let mut k_idx: u32 = 0;
                            #[unroll(4)]
                            while k_idx < k_iters {
                                let smem_a_ptr = &raw mut SMEM_BUF[stage * SMEM_STAGE_BYTES];
                                let smem_b_ptr =
                                    &raw mut SMEM_BUF[b_smem_offset + stage * SMEM_STAGE_BYTES];
                                let tma_bar_const = &raw const TMA_BAR[stage];
                                let aliased_bar = aliased_tma_bar0.add(stage);
                                let mma_bar_const = &raw const MMA_BAR[stage];

                                while !mbarrier_try_wait_parity_cluster(mma_bar_const, mma_parity)
                                {}

                                if elect_one_cta {
                                    mbarrier_arrive_expect_tx(tma_bar_const, 1, tx_bytes);
                                }
                                let k_base = (k_idx * 64) as i32;
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_a_ptr,
                                    a_tma,
                                    k_base,
                                    m_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr,
                                    b_tma,
                                    k_base,
                                    b_n_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                if tile_kind == TILE_KIND_N192 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES * 2),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                } else if tiles_m == 16 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES * 2),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES * 3),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 96,
                                        aliased_bar,
                                        self_mask,
                                    );
                                }
                                // Prefetch the next CLC work-stealing response 4 K-stages before the tile ends (or on step 0 for K <= 256).
                                if k_idx == clc_prefetch_k {
                                    mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                                    if elect_one_cta {
                                        clc_try_cancel_multicast(
                                            resp_ptr.cast::<u8>(),
                                            &raw mut CLC_BAR,
                                        );
                                    }
                                }

                                stage += 1;
                                if stage == num_stages {
                                    stage = 0;
                                    mma_parity ^= 1;
                                }
                                k_idx += 1;
                            }
                        } else {
                            mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                            if elect_one_cta {
                                clc_try_cancel_multicast(resp_ptr.cast::<u8>(), &raw mut CLC_BAR);
                            }
                        }

                        while !mbarrier_try_wait_parity_cluster(&raw const CLC_BAR, clc_parity) {}
                        clc_iter += 1;

                        let resp_lo = CLC_RESPONSE[0];
                        let resp_hi = CLC_RESPONSE[1];
                        let is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                        if is_canceled == 0 {
                            let t_slot = tile_seq & 3;
                            TILE_INFO[t_slot * 4 + 2] = 0;
                            mbarrier_arrive(&raw const TILE_READY[t_slot]);
                            break;
                        }

                        let first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                        raw_tile_idx = first_stolen / 2;
                    }
                }
            }

            // Phase 2: Warp 5 (leader CTA, lane 0) — 5th-Gen Tensor Core (`tcgen05.mma.cta_group::2`) Consumer.
            if warp_id == MMA_WARP {
                if elect_one_cta && lane_id == 0 {
                    let mut tile_iter: u32 = 0;
                    let mut stage: usize = 0;
                    let mut tma_parity: u32 = 0;
                    let smem_a_base0 = cvta_generic_to_shared_offset(&raw const SMEM_BUF[0]);
                    let smem_b_base0 = smem_a_base0 + b_smem_offset as u64;
                    let a_desc0 =
                        build_smem_descriptor(smem_a_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);
                    let b_desc0 =
                        build_smem_descriptor(smem_b_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);

                    loop {
                        let t_slot = (tile_iter & 3) as usize;
                        let t_parity = (tile_iter >> 2) & 1;
                        while !mbarrier_try_wait_parity(&raw const TILE_READY[t_slot], t_parity) {}

                        let has_work = TILE_INFO[t_slot * 4 + 2];
                        if has_work == 0 {
                            break;
                        }
                        let packed_col = TILE_INFO[t_slot * 4 + 1];
                        let tile_kind = packed_col >> 30;
                        let idesc = if tile_kind == TILE_KIND_N192 {
                            idesc_192
                        } else if tile_kind == TILE_KIND_N224 {
                            idesc_224
                        } else {
                            idesc_256
                        };

                        let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                        let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                        // Wait until `Warps 0..3` on both CTAs have finished unloading `TMEM_{accum_stage}` from 2 tiles ago.
                        if tile_iter >= NUM_ACCUM_STAGES {
                            let empty_parity =
                                ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                            if accum_stage == 0 {
                                while !mbarrier_try_wait_parity_cluster(
                                    &raw const ACCUM_EMPTY0,
                                    empty_parity,
                                ) {}
                            } else {
                                while !mbarrier_try_wait_parity_cluster(
                                    &raw const ACCUM_EMPTY1,
                                    empty_parity,
                                ) {}
                            }
                            tcgen05_fence_after_thread_sync();
                        }

                        let mut k_idx: u32 = 0;
                        #[unroll(4)]
                        while k_idx < k_iters {
                            let stage_desc_offset = (stage as u64) << 10;
                            let a_desc_s = a_desc0 + stage_desc_offset;
                            let b_desc_s = b_desc0 + stage_desc_offset;
                            let tma_bar_const = &raw const TMA_BAR[stage];
                            let mma_bar_mut = &raw mut MMA_BAR[stage];

                            while !mbarrier_try_wait_parity_cluster(tma_bar_const, tma_parity) {}
                            tcgen05_fence_after_thread_sync();

                            let tmem_dst = tmem_addr + tmem_stage_offset;
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s,
                                b_desc_s,
                                idesc,
                                k_idx > 0,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 2,
                                b_desc_s + 2,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 4,
                                b_desc_s + 4,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 6,
                                b_desc_s + 6,
                                idesc,
                                true,
                            );

                            tcgen05_commit_multicast_cg2(
                                mma_bar_mut.cast::<u64>(),
                                CTA_MASK_PAIR,
                            );

                            stage += 1;
                            if stage == num_stages {
                                stage = 0;
                                tma_parity ^= 1;
                            }
                            k_idx += 1;
                        }

                        if accum_stage == 0 {
                            tcgen05_commit_multicast_cg2(
                                (&raw mut ACCUM_FULL0).cast::<u64>(),
                                CTA_MASK_PAIR,
                            );
                        } else {
                            tcgen05_commit_multicast_cg2(
                                (&raw mut ACCUM_FULL1).cast::<u64>(),
                                CTA_MASK_PAIR,
                            );
                        }

                        tile_iter += 1;
                    }
                }

                tcgen05_relinquish_alloc_permit_cg2();
            }

            // Phase 3: Warps 0..3 — Register-Direct TMEM Unload, Early TMEM Release, Butterfly Shuffle & 128-Bit Store.
            if warp_id < 4 {
                let mut epi_tile_iter: u32 = 0;

                let leader_accum_empty0_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0).cast::<u8>(),
                );
                let leader_accum_empty1_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0).cast::<u8>(),
                );

                let is_odd_lane = (lane_id & 1) != 0;
                let is_bit1_lane = (lane_id & 2) != 0;
                let lane_row_in_16 = ((lane_id / 4) + (lane_id & 1) * 8) as usize;
                let lane_col_u32 = ((lane_id & 2) * 2) as usize;

                loop {
                    let t_slot = (epi_tile_iter & 3) as usize;
                    let t_parity = (epi_tile_iter >> 2) & 1;
                    while !mbarrier_try_wait_parity(&raw const TILE_READY[t_slot], t_parity) {}

                    let has_work = TILE_INFO[t_slot * 4 + 2];
                    if has_work == 0 {
                        break;
                    }

                    let tile_m = TILE_INFO[t_slot * 4];
                    let packed_col = TILE_INFO[t_slot * 4 + 1];
                    let tile_kind = packed_col >> 30;
                    let col_u32_start = ((packed_col & 0x3FFF_FFFF) * 16) as usize;

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity_cluster(
                            &raw const ACCUM_FULL0,
                            full_parity,
                        ) {}
                    } else {
                        while !mbarrier_try_wait_parity_cluster(
                            &raw const ACCUM_FULL1,
                            full_parity,
                        ) {}
                    }
                    tcgen05_fence_after_thread_sync();

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 256 + my_rank * 128 + warp_id * 32) as usize;
                    let tile_col_base = col_u32_start + lane_col_u32;
                    let col_chunks: u32 = if tile_kind == TILE_KIND_N192 { 3 } else { 4 };

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;
                        let global_row =
                            tile_row_base + (tmem_row_block as usize * 16) + lane_row_in_16;
                        let row_out_base = global_row * n_u32 + tile_col_base;

                        let mut col_chunk = 0u32;
                        #[unroll(never)]
                        while col_chunk < col_chunks {
                            let col_offset = col_chunk * 64;

                            let regs = tcgen05_ld_16x256b_x8_pure(
                                tmem_addr + tmem_stage_offset + (tmem_row << 16) + col_offset,
                            );
                            tcgen05_load_wait();

                            let col_u32_base = (col_chunk as usize) * 32;
                            let max_pairs: usize =
                                if tile_kind == TILE_KIND_N224 && col_chunk == 3 {
                                    2
                                } else {
                                    4
                                };
                            let mut pair = 0usize;
                            #[unroll]
                            while pair < 4 {
                                if pair < max_pairs {
                                    let base = pair * 8;
                                    let p0_lo = cvt_bf16x2_f32(regs[base], regs[base + 1]);
                                    let p0_hi = cvt_bf16x2_f32(regs[base + 2], regs[base + 3]);
                                    let send0 = if is_odd_lane { p0_lo } else { p0_hi };
                                    let recv0 = warp::shuffle_xor_sync(u32::MAX, send0, 1);
                                    let lo0 = if is_odd_lane { recv0 } else { p0_lo };
                                    let hi0 = if is_odd_lane { p0_hi } else { recv0 };

                                    let p1_lo = cvt_bf16x2_f32(regs[base + 4], regs[base + 5]);
                                    let p1_hi = cvt_bf16x2_f32(regs[base + 6], regs[base + 7]);
                                    let send1 = if is_odd_lane { p1_lo } else { p1_hi };
                                    let recv1 = warp::shuffle_xor_sync(u32::MAX, send1, 1);
                                    let lo1 = if is_odd_lane { recv1 } else { p1_lo };
                                    let hi1 = if is_odd_lane { p1_hi } else { recv1 };

                                    let send_a = if is_bit1_lane { lo0 } else { lo1 };
                                    let send_b = if is_bit1_lane { hi0 } else { hi1 };
                                    let recv_a = warp::shuffle_xor_sync(u32::MAX, send_a, 2);
                                    let recv_b = warp::shuffle_xor_sync(u32::MAX, send_b, 2);

                                    let w0 = if is_bit1_lane { recv_a } else { lo0 };
                                    let w1 = if is_bit1_lane { recv_b } else { hi0 };
                                    let w2 = if is_bit1_lane { lo1 } else { recv_a };
                                    let w3 = if is_bit1_lane { hi1 } else { recv_b };

                                    *out.as_mut_ptr()
                                        .add(row_out_base + col_u32_base + pair * 8)
                                        .cast::<U32x4>() = U32x4(w0, w1, w2, w3);
                                }
                                pair += 1;
                            }

                            col_chunk += 1;
                        }
                        tmem_row_block += 1;
                    }

                    tcgen05_fence_before_thread_sync();
                    if lane_id == 0 {
                        if elect_one_cta {
                            if accum_stage == 0 {
                                mbarrier_arrive(&raw const ACCUM_EMPTY0);
                            } else {
                                mbarrier_arrive(&raw const ACCUM_EMPTY1);
                            }
                        } else if accum_stage == 0 {
                            mbarrier_arrive_cluster(leader_accum_empty0_addr);
                        } else {
                            mbarrier_arrive_cluster(leader_accum_empty1_addr);
                        }
                    }

                    epi_tile_iter += 1;
                }
            }

            cluster::cluster_sync();

            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }
            cluster::cluster_sync();
        }
    }

    /// Cooperative 2-CTA (`cta_group::2`) Blackwell GEMM kernel registered as the `M512×N256`
    /// `KernelFamily` entry (`16384³`, preferred when `tiles_m >= 64`).
    ///
    /// Uses the same 7-stage `224 KiB` multicast TMA ring, 2-stage `TMEM` ping-pong, `M256×N224`
    /// wave-tail balancing, overlapped CLC work-stealing, and register-direct `U32x4` epilogue
    /// as [`gemm_sol_clc_multicast_4_stage_pipeline`].
    ///
    /// # Safety
    /// Must be launched on an `sm_100a`-compatible GPU with a `2 × 1 × 1` cluster dimension, `192`
    /// threads per CTA (`6` warps), valid 2D `SWIZZLE_128B` TMA descriptors `a_tma` and `b_tma`,
    /// and an output buffer `out` of at least `M * N / 2` packed `u32` (`bf16x2`) elements.
    #[expect(
        clippy::collapsible_if,
        reason = "keep warp-uniform branch separate from lane-0 elect to avoid MIR short-circuit join block and preserve 65-register PTX"
    )]
    #[kernel(unchecked_indexing)]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_sol_clc_multicast_4_stage_pipeline_large(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        _tiles_n: u32,
    ) {
        // SAFETY: All shared-memory arrays, hardware barriers, TMA transfers, TMEM allocations,
        // and global output stores below obey the 2-CTA cluster synchronization protocol.
        unsafe {
            /// Contiguous `224 KiB` (`7 × 16 KiB` `A` + `7 × 16 KiB` `B`), 128-byte aligned SMEM ring.
            /// Written by `TMA_WARP` via 2D TMA multicast and consumed by `MMA_WARP` via `tcgen05.mma`.
            static mut SMEM_BUF: SharedArray<u8, 229376, 128> = SharedArray::UNINIT;

            /// Single 32-bit shared-memory slot holding the base `TMEM` column address allocated by `Warp 0`.
            /// Broadcast to all warps after the initial `cluster_sync()`.
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            /// Four-slot (`4 × 4` `u32`) tile mailbox passing `(tile_m, packed_col, has_work, pad)`
            /// from `TMA_WARP` to `MMA_WARP` and `Warps 0..3`.
            static mut TILE_INFO: SharedArray<u32, 16, 4> = SharedArray::UNINIT;

            /// Per-stage `mbarrier` array (`7` active stages) signaled when 2-CTA TMA loads complete.
            /// Waited on by `MMA_WARP` before issuing `tcgen05.mma` for that stage.
            static mut TMA_BAR: SharedArray<Barrier, 8, 8> = SharedArray::UNINIT;

            /// Per-stage `mbarrier` array (`7` active stages) signaled by `tcgen05_commit_multicast_cg2`
            /// when `MMA_WARP` finishes consuming a shared-memory stage.
            static mut MMA_BAR: SharedArray<Barrier, 8, 8> = SharedArray::UNINIT;

            /// Accumulator-full barrier for `TMEM` stage `0` (columns `0..256`), signaled after the K-loop.
            /// Waited on by `Warps 0..3` before unloading `TMEM_0`.
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;

            /// Accumulator-full barrier for `TMEM` stage `1` (columns `256..512`), signaled after the K-loop.
            /// Waited on by `Warps 0..3` before unloading `TMEM_1`.
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;

            /// Accumulator-empty barrier for `TMEM` stage `0` (arrival count `8`: 4 epilogue warps × 2 CTAs).
            /// Signaled by `lane_id == 0` of each epilogue warp immediately after the final `tcgen05.wait::ld`.
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;

            /// Accumulator-empty barrier for `TMEM` stage `1` (arrival count `8`: 4 epilogue warps × 2 CTAs).
            /// Signaled by `lane_id == 0` of each epilogue warp immediately after the final `tcgen05.wait::ld`.
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;

            /// Four-slot `mbarrier` array synchronizing the 4-slot `TILE_INFO` descriptor mailbox.
            /// Signaled by `TMA_WARP` once `TILE_INFO[t_slot * 4 .. +3]` is written.
            static mut TILE_READY: SharedArray<Barrier, 4, 8> = SharedArray::UNINIT;

            /// 16-byte (`2 × u64`) response buffer written by `clusterlaunchcontrol.try_cancel`.
            /// Read by `TMA_WARP` after `CLC_BAR` completes to decode the next stolen CTA ID.
            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;

            /// Hardware barrier tracking completion of the 16-byte multicast CLC `try_cancel` response.
            /// Waited on by `TMA_WARP` before querying `clc_query_is_canceled`.
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            let num_stages: usize = NUM_SMEM_STAGES;
            let b_smem_offset: usize = B_SMEM_OFFSET;
            let swizzle_g: u32 = if tiles_m == 16 { 2 } else { 8 };

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let my_rank = cluster::cluster_ctaidX();
            let self_mask: u16 = 1u16 << my_rank;

            // Phase 0: Initialize all shared-memory mbarriers on thread 0 and allocate 512 TMEM columns on Warp 0.
            if tid == 0 {
                let mut s = 0usize;
                while s < NUM_SMEM_STAGES {
                    mbarrier_init(&raw mut TMA_BAR[s], 1);
                    mbarrier_init(&raw mut MMA_BAR[s], 1);
                    s += 1;
                }
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 8);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 8);
                let mut ts = 0usize;
                while ts < 4 {
                    mbarrier_init(&raw mut TILE_READY[ts], 1);
                    ts += 1;
                }
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            cluster::cluster_sync();
            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR[0], 512);
            }
            cluster::cluster_sync();
            let tmem_addr = TMEM_ADDR[0];
            let elect_one_cta = my_rank == 0;

            let idesc_256 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N256)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_192 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 192 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_224 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 224 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;

            // Phase 1: Warp 4 (lane 0) — 7-Stage Multicast TMA Producer + Wave-Tail Balancing + Overlapped CLC.
            if warp_id == TMA_WARP {
                if lane_id == 0 {
                    let mut stage: usize = 0;
                    let mut mma_parity: u32 = 1;
                    let mut tile_seq: usize = 0;
                    let mut clc_iter: u32 = 0;

                    let cluster_base_id = thread::blockIdx_x() - my_rank;
                    let mut raw_tile_idx = cluster_base_id / 2;
                    let wide_tiles_n = _tiles_n / 2;
                    let wide_total = tiles_m * wide_tiles_n;
                    #[expect(
                        clippy::manual_div_ceil,
                        reason = "keep branchless (wide_total + 73) / 74 division to avoid extra remainder/select predicate in Warp 4"
                    )]
                    let num_waves = (wide_total + (B200_CLUSTER_PAIRS - 1)) / B200_CLUSTER_PAIRS;
                    let work_total = num_waves * B200_CLUSTER_PAIRS;
                    let num_convert = work_total - wide_total;
                    let use_n224 = tiles_m >= 32;
                    let full_tiles = if use_n224 {
                        wide_total - 7 * num_convert
                    } else {
                        wide_total - 3 * num_convert
                    };
                    let head_cols = if use_n224 {
                        wide_tiles_n - 7
                    } else if tiles_m == 16 {
                        wide_tiles_n - 9
                    } else {
                        wide_tiles_n - 3
                    };
                    let head_tiles = tiles_m * head_cols;
                    let resp_ptr = &raw mut CLC_RESPONSE[0];
                    let aliased_tma_bar0 = core::ptr::with_exposed_provenance_mut::<Barrier>(
                        (&raw mut TMA_BAR[0]).expose_provenance() & (PEER_BIT_MASK as usize),
                    );

                    loop {
                        let clc_parity = clc_iter & 1;
                        if raw_tile_idx < work_total {
                            // Map linear `raw_tile_idx` to serpentine L2 N-band coordinates or wave-tail N192/N224 tiles.
                            let (tile_m, col_bf16_base, tile_kind) = if raw_tile_idx < head_tiles {
                                let group_tiles = swizzle_g * tiles_m;
                                let group = raw_tile_idx / group_tiles;
                                let in_group = raw_tile_idx % group_tiles;
                                let n_start = group * swizzle_g;
                                let band_w = if swizzle_g < head_cols - n_start {
                                    swizzle_g
                                } else {
                                    head_cols - n_start
                                };
                                let raw_m = in_group / band_w;
                                let tm = if (group & 1) != 0 {
                                    tiles_m - 1 - raw_m
                                } else {
                                    raw_m
                                };
                                let tn = n_start + in_group % band_w;
                                (tm, tn * 256, TILE_KIND_N256)
                            } else if use_n224 {
                                if raw_tile_idx < full_tiles {
                                    let rem = raw_tile_idx - head_tiles;
                                    let t = rem / 7;
                                    let sub = rem % 7;
                                    (t, (head_cols + sub) * 256, TILE_KIND_N256)
                                } else {
                                    let tail = raw_tile_idx - full_tiles;
                                    let t = (full_tiles - head_tiles) / 7 + (tail / 8);
                                    let sub = tail & 7;
                                    (t, head_cols * 256 + sub * 224, TILE_KIND_N224)
                                }
                            } else if raw_tile_idx < full_tiles {
                                let rem = raw_tile_idx - head_tiles;
                                let t = rem / 3;
                                let sub = rem % 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn = head_cols + band * 3 + sub;
                                (tm, tn * 256, TILE_KIND_N256)
                            } else {
                                let tail = raw_tile_idx - full_tiles;
                                let t = (full_tiles - head_tiles) / 3 + (tail / 4);
                                let sub = tail & 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn_base = head_cols + band * 3;
                                (tm, tn_base * 256 + sub * 192, TILE_KIND_N192)
                            };

                            let packed_col = (col_bf16_base >> 5) | (tile_kind << 30);

                            // Publish tile coordinates to `MMA_WARP` and `Warps 0..3` via the 4-slot mailbox.
                            let t_slot = tile_seq & 3;
                            TILE_INFO[t_slot * 4] = tile_m;
                            TILE_INFO[t_slot * 4 + 1] = packed_col;
                            TILE_INFO[t_slot * 4 + 2] = 1;
                            mbarrier_arrive(&raw const TILE_READY[t_slot]);
                            tile_seq += 1;

                            let m_offset = (tile_m * 256 + my_rank * 128) as i32;
                            let b_n_offset = if tile_kind == TILE_KIND_N192 {
                                (col_bf16_base + my_rank * 96) as i32
                            } else if tile_kind == TILE_KIND_N224 {
                                (col_bf16_base + my_rank * 112) as i32
                            } else {
                                (col_bf16_base + my_rank * 128) as i32
                            };
                            let tx_bytes = if tile_kind == TILE_KIND_N192 {
                                (A_TILE_BYTES + B_TILE_BYTES_192) * 2
                            } else {
                                (A_TILE_BYTES + B_TILE_BYTES_256) * 2
                            };

                            let clc_prefetch_k = k_iters.saturating_sub(4);
                            let mut k_idx: u32 = 0;
                            #[unroll(4)]
                            while k_idx < k_iters {
                                let smem_a_ptr = &raw mut SMEM_BUF[stage * SMEM_STAGE_BYTES];
                                let smem_b_ptr =
                                    &raw mut SMEM_BUF[b_smem_offset + stage * SMEM_STAGE_BYTES];
                                let tma_bar_const = &raw const TMA_BAR[stage];
                                let aliased_bar = aliased_tma_bar0.add(stage);
                                let mma_bar_const = &raw const MMA_BAR[stage];

                                while !mbarrier_try_wait_parity_cluster(mma_bar_const, mma_parity)
                                {}

                                if elect_one_cta {
                                    mbarrier_arrive_expect_tx(tma_bar_const, 1, tx_bytes);
                                }
                                let k_base = (k_idx * 64) as i32;
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_a_ptr,
                                    a_tma,
                                    k_base,
                                    m_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr,
                                    b_tma,
                                    k_base,
                                    b_n_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                if tile_kind == TILE_KIND_N192 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES * 2),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                } else if tiles_m == 16 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES * 2),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES * 3),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 96,
                                        aliased_bar,
                                        self_mask,
                                    );
                                }
                                // Prefetch the next CLC work-stealing response 4 K-stages before the tile ends (or on step 0 for K <= 256).
                                if k_idx == clc_prefetch_k {
                                    mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                                    if elect_one_cta {
                                        clc_try_cancel_multicast(
                                            resp_ptr.cast::<u8>(),
                                            &raw mut CLC_BAR,
                                        );
                                    }
                                }

                                stage += 1;
                                if stage == num_stages {
                                    stage = 0;
                                    mma_parity ^= 1;
                                }
                                k_idx += 1;
                            }
                        } else {
                            mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                            if elect_one_cta {
                                clc_try_cancel_multicast(resp_ptr.cast::<u8>(), &raw mut CLC_BAR);
                            }
                        }

                        while !mbarrier_try_wait_parity_cluster(&raw const CLC_BAR, clc_parity) {}
                        clc_iter += 1;

                        let resp_lo = CLC_RESPONSE[0];
                        let resp_hi = CLC_RESPONSE[1];
                        let is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                        if is_canceled == 0 {
                            let t_slot = tile_seq & 3;
                            TILE_INFO[t_slot * 4 + 2] = 0;
                            mbarrier_arrive(&raw const TILE_READY[t_slot]);
                            break;
                        }

                        let first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                        raw_tile_idx = first_stolen / 2;
                    }
                }
            }

            // Phase 2: Warp 5 (leader CTA, lane 0) — 5th-Gen Tensor Core (`tcgen05.mma.cta_group::2`) Consumer.
            if warp_id == MMA_WARP {
                if elect_one_cta && lane_id == 0 {
                    let mut tile_iter: u32 = 0;
                    let mut stage: usize = 0;
                    let mut tma_parity: u32 = 0;
                    let smem_a_base0 = cvta_generic_to_shared_offset(&raw const SMEM_BUF[0]);
                    let smem_b_base0 = smem_a_base0 + b_smem_offset as u64;
                    let a_desc0 =
                        build_smem_descriptor(smem_a_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);
                    let b_desc0 =
                        build_smem_descriptor(smem_b_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);

                    loop {
                        let t_slot = (tile_iter & 3) as usize;
                        let t_parity = (tile_iter >> 2) & 1;
                        while !mbarrier_try_wait_parity(&raw const TILE_READY[t_slot], t_parity) {}

                        let has_work = TILE_INFO[t_slot * 4 + 2];
                        if has_work == 0 {
                            break;
                        }
                        let packed_col = TILE_INFO[t_slot * 4 + 1];
                        let tile_kind = packed_col >> 30;
                        let idesc = if tile_kind == TILE_KIND_N192 {
                            idesc_192
                        } else if tile_kind == TILE_KIND_N224 {
                            idesc_224
                        } else {
                            idesc_256
                        };

                        let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                        let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                        // Wait until `Warps 0..3` on both CTAs have finished unloading `TMEM_{accum_stage}` from 2 tiles ago.
                        if tile_iter >= NUM_ACCUM_STAGES {
                            let empty_parity =
                                ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                            if accum_stage == 0 {
                                while !mbarrier_try_wait_parity_cluster(
                                    &raw const ACCUM_EMPTY0,
                                    empty_parity,
                                ) {}
                            } else {
                                while !mbarrier_try_wait_parity_cluster(
                                    &raw const ACCUM_EMPTY1,
                                    empty_parity,
                                ) {}
                            }
                            tcgen05_fence_after_thread_sync();
                        }

                        let mut k_idx: u32 = 0;
                        #[unroll(4)]
                        while k_idx < k_iters {
                            let stage_desc_offset = (stage as u64) << 10;
                            let a_desc_s = a_desc0 + stage_desc_offset;
                            let b_desc_s = b_desc0 + stage_desc_offset;
                            let tma_bar_const = &raw const TMA_BAR[stage];
                            let mma_bar_mut = &raw mut MMA_BAR[stage];

                            while !mbarrier_try_wait_parity_cluster(tma_bar_const, tma_parity) {}
                            tcgen05_fence_after_thread_sync();

                            let tmem_dst = tmem_addr + tmem_stage_offset;
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s,
                                b_desc_s,
                                idesc,
                                k_idx > 0,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 2,
                                b_desc_s + 2,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 4,
                                b_desc_s + 4,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 6,
                                b_desc_s + 6,
                                idesc,
                                true,
                            );

                            tcgen05_commit_multicast_cg2(
                                mma_bar_mut.cast::<u64>(),
                                CTA_MASK_PAIR,
                            );

                            stage += 1;
                            if stage == num_stages {
                                stage = 0;
                                tma_parity ^= 1;
                            }
                            k_idx += 1;
                        }

                        if accum_stage == 0 {
                            tcgen05_commit_multicast_cg2(
                                (&raw mut ACCUM_FULL0).cast::<u64>(),
                                CTA_MASK_PAIR,
                            );
                        } else {
                            tcgen05_commit_multicast_cg2(
                                (&raw mut ACCUM_FULL1).cast::<u64>(),
                                CTA_MASK_PAIR,
                            );
                        }

                        tile_iter += 1;
                    }
                }

                tcgen05_relinquish_alloc_permit_cg2();
            }

            // Phase 3: Warps 0..3 — Register-Direct TMEM Unload, Early TMEM Release, Butterfly Shuffle & 128-Bit Store.
            if warp_id < 4 {
                let mut epi_tile_iter: u32 = 0;

                let leader_accum_empty0_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0).cast::<u8>(),
                );
                let leader_accum_empty1_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0).cast::<u8>(),
                );

                let is_odd_lane = (lane_id & 1) != 0;
                let is_bit1_lane = (lane_id & 2) != 0;
                let lane_row_in_16 = ((lane_id / 4) + (lane_id & 1) * 8) as usize;
                let lane_col_u32 = ((lane_id & 2) * 2) as usize;

                loop {
                    let t_slot = (epi_tile_iter & 3) as usize;
                    let t_parity = (epi_tile_iter >> 2) & 1;
                    while !mbarrier_try_wait_parity(&raw const TILE_READY[t_slot], t_parity) {}

                    let has_work = TILE_INFO[t_slot * 4 + 2];
                    if has_work == 0 {
                        break;
                    }

                    let tile_m = TILE_INFO[t_slot * 4];
                    let packed_col = TILE_INFO[t_slot * 4 + 1];
                    let tile_kind = packed_col >> 30;
                    let col_u32_start = ((packed_col & 0x3FFF_FFFF) * 16) as usize;

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity_cluster(
                            &raw const ACCUM_FULL0,
                            full_parity,
                        ) {}
                    } else {
                        while !mbarrier_try_wait_parity_cluster(
                            &raw const ACCUM_FULL1,
                            full_parity,
                        ) {}
                    }
                    tcgen05_fence_after_thread_sync();

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 256 + my_rank * 128 + warp_id * 32) as usize;
                    let tile_col_base = col_u32_start + lane_col_u32;
                    let col_chunks: u32 = if tile_kind == TILE_KIND_N192 { 3 } else { 4 };

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;
                        let global_row =
                            tile_row_base + (tmem_row_block as usize * 16) + lane_row_in_16;
                        let row_out_base = global_row * n_u32 + tile_col_base;

                        let mut col_chunk = 0u32;
                        #[unroll(never)]
                        while col_chunk < col_chunks {
                            let col_offset = col_chunk * 64;

                            let regs = tcgen05_ld_16x256b_x8_pure(
                                tmem_addr + tmem_stage_offset + (tmem_row << 16) + col_offset,
                            );
                            tcgen05_load_wait();

                            let col_u32_base = (col_chunk as usize) * 32;
                            let max_pairs: usize =
                                if tile_kind == TILE_KIND_N224 && col_chunk == 3 {
                                    2
                                } else {
                                    4
                                };
                            let mut pair = 0usize;
                            #[unroll]
                            while pair < 4 {
                                if pair < max_pairs {
                                    let base = pair * 8;
                                    let p0_lo = cvt_bf16x2_f32(regs[base], regs[base + 1]);
                                    let p0_hi = cvt_bf16x2_f32(regs[base + 2], regs[base + 3]);
                                    let send0 = if is_odd_lane { p0_lo } else { p0_hi };
                                    let recv0 = warp::shuffle_xor_sync(u32::MAX, send0, 1);
                                    let lo0 = if is_odd_lane { recv0 } else { p0_lo };
                                    let hi0 = if is_odd_lane { p0_hi } else { recv0 };

                                    let p1_lo = cvt_bf16x2_f32(regs[base + 4], regs[base + 5]);
                                    let p1_hi = cvt_bf16x2_f32(regs[base + 6], regs[base + 7]);
                                    let send1 = if is_odd_lane { p1_lo } else { p1_hi };
                                    let recv1 = warp::shuffle_xor_sync(u32::MAX, send1, 1);
                                    let lo1 = if is_odd_lane { recv1 } else { p1_lo };
                                    let hi1 = if is_odd_lane { p1_hi } else { recv1 };

                                    let send_a = if is_bit1_lane { lo0 } else { lo1 };
                                    let send_b = if is_bit1_lane { hi0 } else { hi1 };
                                    let recv_a = warp::shuffle_xor_sync(u32::MAX, send_a, 2);
                                    let recv_b = warp::shuffle_xor_sync(u32::MAX, send_b, 2);

                                    let w0 = if is_bit1_lane { recv_a } else { lo0 };
                                    let w1 = if is_bit1_lane { recv_b } else { hi0 };
                                    let w2 = if is_bit1_lane { lo1 } else { recv_a };
                                    let w3 = if is_bit1_lane { hi1 } else { recv_b };

                                    *out.as_mut_ptr()
                                        .add(row_out_base + col_u32_base + pair * 8)
                                        .cast::<U32x4>() = U32x4(w0, w1, w2, w3);
                                }
                                pair += 1;
                            }

                            col_chunk += 1;
                        }
                        tmem_row_block += 1;
                    }

                    tcgen05_fence_before_thread_sync();
                    if lane_id == 0 {
                        if elect_one_cta {
                            if accum_stage == 0 {
                                mbarrier_arrive(&raw const ACCUM_EMPTY0);
                            } else {
                                mbarrier_arrive(&raw const ACCUM_EMPTY1);
                            }
                        } else if accum_stage == 0 {
                            mbarrier_arrive_cluster(leader_accum_empty0_addr);
                        } else {
                            mbarrier_arrive_cluster(leader_accum_empty1_addr);
                        }
                    }

                    epi_tile_iter += 1;
                }
            }

            cluster::cluster_sync();

            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }
            cluster::cluster_sync();
        }
    }

    /// Persistent 1-CTA (`cta_group::1`) Blackwell GEMM kernel specialized for epilogue-bound
    /// short-`K` shapes (`M128xN256` + `M128xN192` wave-tail balancing, `K <= 512`, `grid = 148` CTAs).
    ///
    /// Eliminates cross-SM `cluster_sync()`, `CLC` GigaThread round-trips, and `16x256b` warp butterfly
    /// shuffles by combining a `4`-stage TMA ring (`192 KiB`), a `2`-stage `TMEM` ring (`2 x 256` cols),
    /// zero-shuffle `tcgen05_ld_32x32b_x64_raw` loads, and per-warp `18 KiB` `SMEM_EPI` (`32 x 9` `U32x4`)
    /// into `128`-byte (`1` full L2 cache line per row) coalesced global `STG.E.128` bursts:
    ///
    /// ```text
    /// [148 Persistent CTAs] ──► [4-Stage TMA (192 KiB)] ──► [2-Stage TMEM (2x256)] ──► [32x32b.x64 -> SMEM_EPI -> 128B STG.E.128]
    /// ```
    ///
    /// # Safety
    ///
    /// Requires `sm_100a` (`grid = (148, 1, 1)`, `block = (192, 1, 1)`), `128B`-swizzled TMA descriptors
    /// (`A`: `64x128`, `B`: `64x64`), `output` sized to `M * N / 2` `u32` words, and `M >= 256`, `N >= 256`,
    /// `K >= 64` aligned to `(256, 256, 64)`.
    #[kernel(unchecked_indexing)]
    #[cuda_device::launch_bounds(192)]
    pub unsafe fn gemm_sol_short_k_m128n256(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut [u32],
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
    ) {
        /// Contiguous `192 KiB` (`4 x 48 KiB`), 128-byte-aligned shared-memory TMA ring (`A[0..4]` + `B[0..4]`).
        /// Leaves `36 KiB` of the `228 KiB` SMEM budget free for `18 KiB` `SMEM_EPI`.
        static mut SMEM_BUF: SharedArray<u8, SHORT_SMEM_TOTAL_BYTES, 128> = SharedArray::UNINIT;

        /// Per-warp `18 KiB` (`4 x 32 x 9` `U32x4` packets) bank-conflict-free epilogue staging buffer.
        /// Synchronized intra-warp via `warp::sync_mask(u32::MAX)` to emit `128B` coalesced `STG.E.128`.
        static mut SMEM_EPI: SharedArray<U32x4, EPI_TOTAL_PKTS, 16> = SharedArray::UNINIT;

        /// Single-word shared-memory slot holding the `512`-column `TMEM` base handle (`2 x 256` cols).
        /// Written by `Warp 0` before `sync_threads()` and read by `MMA_WARP` and `Warps 0..3`.
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

        /// Per-stage TMA completion `mbarrier` array (`4` stages).
        /// Signaled when `A + B` (`49,152`B for `N256`, `40,960`B for `N192`) arrive in `SMEM_BUF`.
        static mut TMA_BAR: SharedArray<Barrier, SHORT_SMEM_STAGES, 8> = SharedArray::UNINIT;

        /// Per-stage UMMA consumption `mbarrier` array (`4` stages).
        /// Signaled by `tcgen05_commit_shared_cluster` when a stage can be refilled.
        static mut MMA_BAR: SharedArray<Barrier, SHORT_SMEM_STAGES, 8> = SharedArray::UNINIT;

        /// 2-stage `256`-column `TMEM` accumulator completion `mbarrier` ring (`0, 256`).
        /// Signaled by `MMA_WARP` and awaited intra-CTA by `Warps 0..3`.
        static mut ACCUM_FULL: SharedArray<Barrier, 2, 8> = SharedArray::UNINIT;

        /// 2-stage `256`-column `TMEM` accumulator release `mbarrier` ring (`4` epilogue warp leaders).
        /// Signaled once `Warps 0..3` finish unloading `TMEM` into registers.
        static mut ACCUM_EMPTY: SharedArray<Barrier, 2, 8> = SharedArray::UNINIT;

        unsafe {
            let tid = thread::threadIdx_x();
            let warp_id = tid / 32;
            let lane_id = tid % 32;

            if tid == 0 {
                let mut s = 0usize;
                #[unroll]
                while s < SHORT_SMEM_STAGES {
                    mbarrier_init(&raw mut TMA_BAR[s], 1);
                    mbarrier_init(&raw mut MMA_BAR[s], 1);
                    s += 1;
                }
                let mut a = 0usize;
                #[unroll]
                while a < 2 {
                    mbarrier_init(&raw mut ACCUM_FULL[a], 1);
                    mbarrier_init(&raw mut ACCUM_EMPTY[a], 4);
                    a += 1;
                }
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR[0], 512);
                tcgen05_relinquish_alloc_permit();
            }
            thread::sync_threads();

            let tmem_addr = TMEM_ADDR[0];
            let k_iters = (k / 64) as u32;
            let tiles_m128 = tiles_m << 1;
            let wide_tiles_n = tiles_n >> 1;
            let wide_total = tiles_m128 * wide_tiles_n;
            #[expect(
                clippy::manual_div_ceil,
                reason = "keep branchless wave division across all warps"
            )]
            let num_waves = (wide_total + (B200_SMS - 1)) / B200_SMS;
            let work_total = num_waves * B200_SMS;
            let num_convert = work_total - wide_total;
            let use_tail_n192 = num_convert > 0 && wide_tiles_n >= 9;
            let (head_cols, full_tiles, total_work) = if use_tail_n192 {
                (wide_tiles_n - 9, wide_total - 3 * num_convert, work_total)
            } else {
                (wide_tiles_n, wide_total, wide_total)
            };
            let head_tiles = tiles_m128 * head_cols;
            let cta_id = thread::blockIdx_x();
            let grid_ctas = thread::gridDim_x();
            let group_tiles = tiles_m128 << 2;

            if warp_id == TMA_WARP {
                if lane_id == 0 {
                    let mut stage: usize = 0;
                    let mut mma_parity: u32 = 1;
                    let mut raw_tile_idx = cta_id;
                    while raw_tile_idx < total_work {
                        let (tm, col_bf16_base, is_n192) = if raw_tile_idx < head_tiles {
                            let (group, in_group) = if tiles_m128 == 32 {
                                (raw_tile_idx >> 7, raw_tile_idx & 127)
                            } else {
                                let g = raw_tile_idx / group_tiles;
                                (g, raw_tile_idx - g * group_tiles)
                            };
                            let n_start = group << 2;
                            let rem_n = head_cols - n_start;
                            let (raw_m, in_band) = if rem_n >= 4 {
                                (in_group >> 2, in_group & 3)
                            } else {
                                (in_group / rem_n, in_group % rem_n)
                            };
                            let m_idx = if (group & 1) != 0 {
                                tiles_m128 - 1 - raw_m
                            } else {
                                raw_m
                            };
                            (m_idx, (n_start + in_band) * 256, false)
                        } else if raw_tile_idx < full_tiles {
                            let rem = raw_tile_idx - head_tiles;
                            let t = rem / 3;
                            let sub = rem - t * 3;
                            let (band, m_idx) = if tiles_m128 == 32 {
                                (t >> 5, t & 31)
                            } else {
                                let b = t / tiles_m128;
                                (b, t - b * tiles_m128)
                            };
                            (m_idx, (head_cols + band * 3 + sub) * 256, false)
                        } else {
                            let tail = raw_tile_idx - full_tiles;
                            let t = (full_tiles - head_tiles) / 3 + (tail >> 2);
                            let sub = tail & 3;
                            let (band, m_idx) = if tiles_m128 == 32 {
                                (t >> 5, t & 31)
                            } else {
                                let b = t / tiles_m128;
                                (b, t - b * tiles_m128)
                            };
                            (m_idx, (head_cols + band * 3) * 256 + sub * 192, true)
                        };

                        let m_offset = (tm * 128) as i32;
                        let n_offset = col_bf16_base as i32;
                        let tx_bytes: u32 = if is_n192 { 16_384 + 24_576 } else { 16_384 + 32_768 };

                        let mut k_idx: u32 = 0;
                        #[unroll(3)]
                        while k_idx < k_iters {
                            let smem_a_ptr = &raw mut SMEM_BUF[stage * SMEM_STAGE_BYTES];
                            let smem_b_ptr = &raw mut SMEM_BUF
                                [SHORT_B_SMEM_OFFSET + stage * SHORT_B_STAGE_BYTES];
                            let tma_bar_const = &raw const TMA_BAR[stage];
                            let tma_bar_mut = &raw mut TMA_BAR[stage];
                            let mma_bar_const = &raw const MMA_BAR[stage];

                            while !mbarrier_try_wait_parity(mma_bar_const, mma_parity) {}

                            mbarrier_arrive_expect_tx(tma_bar_const, 1, tx_bytes);
                            let k_base = (k_idx * 64) as i32;
                            cp_async_bulk_tensor_2d_g2s_cta(
                                smem_a_ptr,
                                a_tma,
                                k_base,
                                m_offset,
                                tma_bar_mut,
                            );
                            cp_async_bulk_tensor_2d_g2s_cta(
                                smem_b_ptr,
                                b_tma,
                                k_base,
                                n_offset,
                                tma_bar_mut,
                            );
                            cp_async_bulk_tensor_2d_g2s_cta(
                                smem_b_ptr.add(8_192),
                                b_tma,
                                k_base,
                                n_offset + 64,
                                tma_bar_mut,
                            );
                            cp_async_bulk_tensor_2d_g2s_cta(
                                smem_b_ptr.add(16_384),
                                b_tma,
                                k_base,
                                n_offset + 128,
                                tma_bar_mut,
                            );
                            if !is_n192 {
                                cp_async_bulk_tensor_2d_g2s_cta(
                                    smem_b_ptr.add(24_576),
                                    b_tma,
                                    k_base,
                                    n_offset + 192,
                                    tma_bar_mut,
                                );
                            }

                            stage += 1;
                            if stage == SHORT_SMEM_STAGES {
                                stage = 0;
                                mma_parity ^= 1;
                            }
                            k_idx += 1;
                        }
                        raw_tile_idx += grid_ctas;
                    }
                }
            } else if warp_id == MMA_WARP {
                if lane_id == 0 {
                    let idesc_256 = Tcgen05InstructionDescriptor::builder()
                        .shape(Tcgen05MmaShape::M128_N256)
                        .element_type(Tcgen05ElementType::F16)
                        .accumulator_type(Tcgen05AccumulatorType::F32)
                        .build()
                        .raw();
                    let idesc_192 = Tcgen05InstructionDescriptor::builder()
                        .shape(Tcgen05MmaShape { m: 128, n: 192 })
                        .element_type(Tcgen05ElementType::F16)
                        .accumulator_type(Tcgen05AccumulatorType::F32)
                        .build()
                        .raw();

                    let mut tile_iter: u32 = 0;
                    let mut stage: usize = 0;
                    let mut tma_parity: u32 = 0;
                    let smem_a_base0 = cvta_generic_to_shared_offset(&raw const SMEM_BUF[0]);
                    let smem_b_base0 = smem_a_base0 + SHORT_B_SMEM_OFFSET as u64;
                    let a_desc0 =
                        build_smem_descriptor(smem_a_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);
                    let b_desc0 =
                        build_smem_descriptor(smem_b_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);

                    let mut raw_tile_idx = cta_id;
                    while raw_tile_idx < total_work {
                        let idesc = if raw_tile_idx < full_tiles {
                            idesc_256
                        } else {
                            idesc_192
                        };
                        let accum_stage = (tile_iter & 1) as usize;
                        let tmem_stage_offset = (accum_stage as u32) * 256;

                        if tile_iter >= 2 {
                            let empty_parity = ((tile_iter - 2) >> 1) & 1;
                            while !mbarrier_try_wait_parity(
                                &raw const ACCUM_EMPTY[accum_stage],
                                empty_parity,
                            ) {}
                            tcgen05_fence_after_thread_sync();
                        }

                        let mut k_idx: u32 = 0;
                        #[unroll(3)]
                        while k_idx < k_iters {
                            let a_desc_s = a_desc0 + ((stage as u64) << 10);
                            let b_desc_s = b_desc0 + ((stage as u64) << 11);
                            let tma_bar_const = &raw const TMA_BAR[stage];
                            let mma_bar_mut = &raw mut MMA_BAR[stage];

                            while !mbarrier_try_wait_parity(tma_bar_const, tma_parity) {}
                            tcgen05_fence_after_thread_sync();

                            let mut j: u64 = 0;
                            #[unroll]
                            while j < 4 {
                                let k_desc_offset = j * 2;
                                let enable_accum = k_idx > 0 || j > 0;
                                tcgen05_mma_f16(
                                    tmem_addr + tmem_stage_offset,
                                    a_desc_s + k_desc_offset,
                                    b_desc_s + k_desc_offset,
                                    idesc,
                                    enable_accum,
                                );
                                j += 1;
                            }

                            tcgen05_commit_shared_cluster(mma_bar_mut.cast::<u64>());

                            stage += 1;
                            if stage == SHORT_SMEM_STAGES {
                                stage = 0;
                                tma_parity ^= 1;
                            }
                            k_idx += 1;
                        }

                        tcgen05_commit_shared_cluster(
                            (&raw mut ACCUM_FULL[accum_stage]).cast::<u64>(),
                        );
                        tile_iter += 1;
                        raw_tile_idx += grid_ctas;
                    }
                }
            } else {
                let mut epi_tile_iter: u32 = 0;
                let w_base = (warp_id as usize) * EPI_WARP_PKTS;
                let l_base = (tid as usize) * EPI_ROW_PKTS;
                let sub_r = (lane_id >> 3) as usize;
                let sub_c = (lane_id & 7) as usize;
                let sub_c_words = sub_c * 4;
                let t_row = warp_id * 32;
                let n_u32 = (n / 2) as usize;
                let out_ptr = output.as_mut_ptr();

                let mut raw_tile_idx = cta_id;
                while raw_tile_idx < total_work {
                    let (tm, col_bf16_base, chunks) = if raw_tile_idx < head_tiles {
                        let (group, in_group) = if tiles_m128 == 32 {
                            (raw_tile_idx >> 7, raw_tile_idx & 127)
                        } else {
                            let g = raw_tile_idx / group_tiles;
                            (g, raw_tile_idx - g * group_tiles)
                        };
                        let n_start = group << 2;
                        let rem_n = head_cols - n_start;
                        let (raw_m, in_band) = if rem_n >= 4 {
                            (in_group >> 2, in_group & 3)
                        } else {
                            (in_group / rem_n, in_group % rem_n)
                        };
                        let m_idx = if (group & 1) != 0 {
                            tiles_m128 - 1 - raw_m
                        } else {
                            raw_m
                        };
                        (m_idx, (n_start + in_band) * 256, 4usize)
                    } else if raw_tile_idx < full_tiles {
                        let rem = raw_tile_idx - head_tiles;
                        let t = rem / 3;
                        let sub = rem - t * 3;
                        let (band, m_idx) = if tiles_m128 == 32 {
                            (t >> 5, t & 31)
                        } else {
                            let b = t / tiles_m128;
                            (b, t - b * tiles_m128)
                        };
                        (m_idx, (head_cols + band * 3 + sub) * 256, 4usize)
                    } else {
                        let tail = raw_tile_idx - full_tiles;
                        let t = (full_tiles - head_tiles) / 3 + (tail >> 2);
                        let sub = tail & 3;
                        let (band, m_idx) = if tiles_m128 == 32 {
                            (t >> 5, t & 31)
                        } else {
                            let b = t / tiles_m128;
                            (b, t - b * tiles_m128)
                        };
                        (m_idx, (head_cols + band * 3) * 256 + sub * 192, 3usize)
                    };

                    let accum_stage = (epi_tile_iter & 1) as usize;
                    let tmem_stage_offset = (accum_stage as u32) * 256;
                    let full_parity = (epi_tile_iter >> 1) & 1;
                    while !mbarrier_try_wait_parity(&raw const ACCUM_FULL[accum_stage], full_parity)
                    {}
                    tcgen05_fence_after_thread_sync();

                    let base_row = (tm * 128 + t_row) as usize;
                    let tile_col_u32 = (col_bf16_base / 2) as usize;
                    let tmem_base = tmem_addr + tmem_stage_offset + (t_row << 16);

                    let mut chunk = 0usize;
                    #[unroll]
                    while chunk < 4 {
                        if chunk < chunks {
                            let raw = tcgen05_ld_32x32b_x64_raw(tmem_base + (chunk as u32) * 64);
                            tcgen05_load_wait();
                            if chunk + 1 == chunks {
                                tcgen05_fence_before_thread_sync();
                                if lane_id == 0 {
                                    mbarrier_arrive(&raw const ACCUM_EMPTY[accum_stage]);
                                }
                            }

                            let mut p = 0usize;
                            #[unroll]
                            while p < 8 {
                                let b = p * 8;
                                SMEM_EPI[l_base + p] = U32x4(
                                    cvt_bf16x2_f32(
                                        f32::from_bits(raw[b]),
                                        f32::from_bits(raw[b + 1]),
                                    ),
                                    cvt_bf16x2_f32(
                                        f32::from_bits(raw[b + 2]),
                                        f32::from_bits(raw[b + 3]),
                                    ),
                                    cvt_bf16x2_f32(
                                        f32::from_bits(raw[b + 4]),
                                        f32::from_bits(raw[b + 5]),
                                    ),
                                    cvt_bf16x2_f32(
                                        f32::from_bits(raw[b + 6]),
                                        f32::from_bits(raw[b + 7]),
                                    ),
                                );
                                p += 1;
                            }

                            warp::sync_mask(u32::MAX);

                            let col_u32_base = tile_col_u32 + chunk * 32 + sub_c_words;
                            let mut r_step = 0usize;
                            #[unroll]
                            while r_step < 8 {
                                let r = r_step * 4 + sub_r;
                                let pkt = SMEM_EPI[w_base + r * EPI_ROW_PKTS + sub_c];
                                *out_ptr
                                    .add((base_row + r) * n_u32 + col_u32_base)
                                    .cast::<U32x4>() = pkt;
                                r_step += 1;
                            }

                            warp::sync_mask(u32::MAX);
                        }
                        chunk += 1;
                    }

                    epi_tile_iter += 1;
                    raw_tile_idx += grid_ctas;
                }
            }

            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
        }
    }
}
