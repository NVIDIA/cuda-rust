/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Routed FP16 ReLU-squared MoE with FP32 weighted accumulation.
//!
//! Requires Tile IR bytecode 13.4 or newer (explicit in-bounds TMA loads).
//!
//! Entries form one ordered pipeline: prepare, optionally schedule and pack,
//! FC1, then FC2. Keep buffers alive through completion on the same stream,
//! or establish equivalent inter-stream dependencies.
//!
//! # Safety contracts for raw-pointer entries
//!
//! Allocations are contiguous, correctly aligned, mutually nonaliasing, and on
//! the launch device. T, H, I, E, R are positive; NR=T*R; E <= 256. H and I are
//! multiples of 256. BM, BN, BK are supported positive MMA tile sizes; H and I
//! are divisible by BN and BK. PR is a power of two >= NR; PG is a power of
//! two >= ceil(NR/BM). CAP >= ceil(NR/BM)+E. All sizes, padded indices and
//! intermediate integer products must fit i32. FK=BM*BK, FN=BM*BN.
//!
//! Required element capacities: `input[T,H]`, `up[E,I,H]`, `down[E,H,I]`,
//! `expert_ids[NR]` (each in `[0,E)`), `route_weights[NR]`, `map[E,NR]`, `counts[E]`,
//! `tasks[CAP,2]`, `total[1]`, `output[T,H]`. With PACKED, `packed[CAP*BM,H]` and
//! `activation[CAP*BM,I]`; otherwise `activation[NR,I]`. A route identifies
//! token=floor(route/R); repeated expert IDs are allowed.
//!
//! prepare_small requires NR <= 2*BM and a nonempty 1-D grid. prepare_large
//! and schedule require grid (E,1,1). Other entries use a nonempty 1-D persistent
//! grid. NO=T*H. Consume only tasks from the matching preparation. Do not reuse
//! scratch concurrently across invocations. pack_input requires F=BM*256.
//! The host example checks sizes and retains ownership of every allocation.

#[cutile::module]
pub mod kernels {
    use cutile::core::*;
    unsafe fn pointers_i32<const N: i32>(
        base: *mut i32,
        indices: Tile<i32, { [N] }>,
    ) -> PointerTile<*mut i32, { [N] }> {
        let p: PointerTile<*mut i32, { [] }> = pointer_to_tile(base);
        let p: PointerTile<*mut i32, { [1] }> = p.reshape(shape![1]);
        let p: PointerTile<*mut i32, { [N] }> = p.broadcast(shape![N]);
        let offsets: Tile<i64, { [N] }> = exti(indices);
        addptr_tile(p, offsets)
    }
    unsafe fn gather_i32<const N: i32, const L: i32>(
        base: *mut i32,
        indices: Tile<i32, { [N] }>,
        mask: Tile<bool, { [N] }>,
        padding: i32,
    ) -> Tile<i32, { [N] }> {
        let ptr: PointerTile<*mut i32, { [N] }> = unsafe { pointers_i32(base, indices) };
        let (v, _t): (Tile<i32, { [N] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(padding),
                None,
                Latency::<L>,
            )
        };
        v
    }
    unsafe fn scatter_i32<const N: i32>(
        base: *mut i32,
        indices: Tile<i32, { [N] }>,
        values: Tile<i32, { [N] }>,
        mask: Tile<bool, { [N] }>,
    ) {
        let ptr: PointerTile<*mut i32, { [N] }> = unsafe { pointers_i32(base, indices) };
        unsafe {
            store_ptr_tko(
                ptr,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
    unsafe fn pointers_i64<const N: i32>(
        base: *mut i64,
        indices: Tile<i32, { [N] }>,
    ) -> PointerTile<*mut i64, { [N] }> {
        let p: PointerTile<*mut i64, { [] }> = pointer_to_tile(base);
        let p: PointerTile<*mut i64, { [1] }> = p.reshape(shape![1]);
        let p: PointerTile<*mut i64, { [N] }> = p.broadcast(shape![N]);
        let offsets: Tile<i64, { [N] }> = exti(indices);
        addptr_tile(p, offsets)
    }
    unsafe fn gather_i64<const N: i32, const L: i32>(
        base: *mut i64,
        indices: Tile<i32, { [N] }>,
        mask: Tile<bool, { [N] }>,
        padding: i64,
    ) -> Tile<i64, { [N] }> {
        let ptr: PointerTile<*mut i64, { [N] }> = unsafe { pointers_i64(base, indices) };
        let (v, _t): (Tile<i64, { [N] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(padding),
                None,
                Latency::<L>,
            )
        };
        v
    }
    unsafe fn scatter_i64<const N: i32>(
        base: *mut i64,
        indices: Tile<i32, { [N] }>,
        values: Tile<i64, { [N] }>,
        mask: Tile<bool, { [N] }>,
    ) {
        let ptr: PointerTile<*mut i64, { [N] }> = unsafe { pointers_i64(base, indices) };
        unsafe {
            store_ptr_tko(
                ptr,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
    unsafe fn pointers_f16<const N: i32>(
        base: *mut f16,
        indices: Tile<i32, { [N] }>,
    ) -> PointerTile<*mut f16, { [N] }> {
        let p: PointerTile<*mut f16, { [] }> = pointer_to_tile(base);
        let p: PointerTile<*mut f16, { [1] }> = p.reshape(shape![1]);
        let p: PointerTile<*mut f16, { [N] }> = p.broadcast(shape![N]);
        let offsets: Tile<i64, { [N] }> = exti(indices);
        addptr_tile(p, offsets)
    }
    unsafe fn gather_f16<const N: i32, const L: i32>(
        base: *mut f16,
        indices: Tile<i32, { [N] }>,
        mask: Tile<bool, { [N] }>,
        padding: f16,
    ) -> Tile<f16, { [N] }> {
        let ptr: PointerTile<*mut f16, { [N] }> = unsafe { pointers_f16(base, indices) };
        let (v, _t): (Tile<f16, { [N] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(padding),
                None,
                Latency::<L>,
            )
        };
        v
    }
    unsafe fn scatter_f16<const N: i32>(
        base: *mut f16,
        indices: Tile<i32, { [N] }>,
        values: Tile<f16, { [N] }>,
        mask: Tile<bool, { [N] }>,
    ) {
        let ptr: PointerTile<*mut f16, { [N] }> = unsafe { pointers_f16(base, indices) };
        unsafe {
            store_ptr_tko(
                ptr,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
    unsafe fn pointers_f32<const N: i32>(
        base: *mut f32,
        indices: Tile<i32, { [N] }>,
    ) -> PointerTile<*mut f32, { [N] }> {
        let p: PointerTile<*mut f32, { [] }> = pointer_to_tile(base);
        let p: PointerTile<*mut f32, { [1] }> = p.reshape(shape![1]);
        let p: PointerTile<*mut f32, { [N] }> = p.broadcast(shape![N]);
        let offsets: Tile<i64, { [N] }> = exti(indices);
        addptr_tile(p, offsets)
    }
    unsafe fn gather_f32<const N: i32, const L: i32>(
        base: *mut f32,
        indices: Tile<i32, { [N] }>,
        mask: Tile<bool, { [N] }>,
        padding: f32,
    ) -> Tile<f32, { [N] }> {
        let ptr: PointerTile<*mut f32, { [N] }> = unsafe { pointers_f32(base, indices) };
        let (v, _t): (Tile<f32, { [N] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(padding),
                None,
                Latency::<L>,
            )
        };
        v
    }
    unsafe fn scatter_f32<const N: i32>(
        base: *mut f32,
        indices: Tile<i32, { [N] }>,
        values: Tile<f32, { [N] }>,
        mask: Tile<bool, { [N] }>,
    ) {
        let ptr: PointerTile<*mut f32, { [N] }> = unsafe { pointers_f32(base, indices) };
        unsafe {
            store_ptr_tko(
                ptr,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
    unsafe fn read_i32(base: *mut i32, index: i32) -> i32 {
        let idx: Tile<i32, { [1] }> = broadcast_scalar(index, shape![1]);
        let mask: Tile<bool, { [1] }> = constant(true, shape![1]);
        let v: Tile<i32, { [1] }> = unsafe { gather_i32::<1, 0>(base, idx, mask, 0i32) };
        let scalar: Tile<i32, { [] }> = v.reshape(shape![]);
        tile_to_scalar(scalar)
    }
    unsafe fn write_i32(base: *mut i32, index: i32, value: i32) {
        let idx: Tile<i32, { [1] }> = broadcast_scalar(index, shape![1]);
        let val: Tile<i32, { [1] }> = broadcast_scalar(value, shape![1]);
        let mask: Tile<bool, { [1] }> = constant(true, shape![1]);
        unsafe {
            scatter_i32(base, idx, val, mask);
        }
    }
    unsafe fn view(base: *mut f16, rows: i32, cols: i32) -> Tensor<f16, { [-1, -1] }> {
        let p: PointerTile<*mut f16, { [] }> = pointer_to_tile(base);
        let shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[rows, cols],
        };
        let strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> {
            dims: &[cols, 1i32],
        };
        let token: Token = new_token_unordered();
        unsafe { make_tensor_view(p, shape, strides, token) }
    }
    unsafe fn zero_output<const NO: i32>(o: *mut f32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let nb: (i32, i32, i32) = get_num_tile_blocks();
        let ids: Tile<i32, { [1024] }> = iota(shape![1024]);
        let zero: Tile<f32, { [1024] }> = constant(0.0f32, shape![1024]);
        let mut block = pid.0;
        while block * 1024i32 < NO {
            let idx: Tile<i32, { [1024] }> = broadcast_scalar(block * 1024i32, shape![1024]) + ids;
            let mask: Tile<bool, { [1024] }> = lt_tile(idx, broadcast_scalar(NO, shape![1024]));
            unsafe {
                scatter_f32(o, idx, zero, mask);
            }
            block = block + nb.0;
        }
    }
    /// One stage of the routed MoE pipeline.
    ///
    /// # Safety
    /// The module's buffer, shape, ordering and grid contracts apply.
    #[cutile::entry()]
    pub unsafe fn prepare_small<const NR: i32, const PR: i32, const BM: i32, const NO: i32>(
        experts: *mut i64,
        map: *mut i32,
        counts: *mut i32,
        tasks: *mut i32,
        total: *mut i32,
        o: *mut f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        if pid.0 == 0i32 {
            let indices: Tile<i32, { [PR] }> = iota(shape![PR]);
            let valid: Tile<bool, { [PR] }> = lt_tile(indices, broadcast_scalar(NR, shape![PR]));
            let ids64: Tile<i64, { [PR] }> =
                unsafe { gather_i64::<PR, 0>(experts, indices, valid, 0i64) };
            let ids: Tile<i32, { [PR] }> = trunci(ids64, overflow::NoSignedWrap);
            let idr: Tile<i32, { [PR, PR] }> = ids.reshape(shape![PR, 1]).broadcast(shape![PR, PR]);
            let idc: Tile<i32, { [PR, PR] }> = ids.reshape(shape![1, PR]).broadcast(shape![PR, PR]);
            let vc: Tile<bool, { [PR, PR] }> =
                valid.reshape(shape![1, PR]).broadcast(shape![PR, PR]);
            let same: Tile<bool, { [PR, PR] }> = eq_tile(idr, idc) & vc;
            let ir: Tile<i32, { [PR, PR] }> =
                indices.reshape(shape![PR, 1]).broadcast(shape![PR, PR]);
            let ic: Tile<i32, { [PR, PR] }> =
                indices.reshape(shape![1, PR]).broadcast(shape![PR, PR]);
            let before: Tile<bool, { [PR, PR] }> = lt_tile(ic, ir);
            let ones: Tile<i32, { [PR, PR] }> = constant(1i32, shape![PR, PR]);
            let zeros: Tile<i32, { [PR, PR] }> = constant(0i32, shape![PR, PR]);
            let rv: Tile<i32, { [PR, PR] }> = select(same & before, ones, zeros);
            let ranks: Tile<i32, { [PR] }> = reduce_sum(rv, 1i32);
            let ranks: Tile<i32, { [PR] }> = ranks.reshape(shape![PR]);
            let nv: Tile<i32, { [PR, PR] }> = select(same, ones, zeros);
            let n: Tile<i32, { [PR] }> = reduce_sum(nv, 1i32);
            let n: Tile<i32, { [PR] }> = n.reshape(shape![PR]);
            let z: Tile<i32, { [PR] }> = constant(0i32, shape![PR]);
            let leader: Tile<bool, { [PR] }> = valid & eq_tile(ranks, z);
            let groups: Tile<i32, { [PR] }> = select(
                leader,
                (n + broadcast_scalar(BM - 1i32, shape![PR])) / broadcast_scalar(BM, shape![PR]),
                z,
            );
            let scan: Tile<i32, { [PR] }> = scan_sum(groups, 0i32, reverse::Forward, 0i32);
            let starts: Tile<i32, { [PR] }> = scan - groups;
            unsafe {
                scatter_i32(
                    map,
                    ids * broadcast_scalar(NR, shape![PR]) + ranks,
                    indices,
                    valid,
                );
            }
            unsafe {
                scatter_i32(counts, ids, n, leader);
            }
            let task_indices: Tile<i32, { [PR] }> = starts * broadcast_scalar(2i32, shape![PR]);
            unsafe {
                scatter_i32(tasks, task_indices, ids, leader);
            }
            unsafe {
                scatter_i32(
                    tasks,
                    task_indices + broadcast_scalar(1i32, shape![PR]),
                    z,
                    leader,
                );
            }
            let extra: Tile<bool, { [PR] }> = leader & gt_tile(n, broadcast_scalar(BM, shape![PR]));
            unsafe {
                scatter_i32(
                    tasks,
                    task_indices + broadcast_scalar(2i32, shape![PR]),
                    ids,
                    extra,
                );
            }
            let one: Tile<i32, { [PR] }> = constant(1i32, shape![PR]);
            unsafe {
                scatter_i32(
                    tasks,
                    task_indices + broadcast_scalar(3i32, shape![PR]),
                    one,
                    extra,
                );
            }
            let sum: Tile<i32, { [] }> = reduce_sum(groups, 0i32);
            let sum: i32 = tile_to_scalar(sum);
            unsafe {
                write_i32(total, 0i32, sum);
            }
        }
        unsafe {
            zero_output::<NO>(o);
        }
    }
    /// One stage of the routed MoE pipeline.
    ///
    /// # Safety
    /// The module's buffer, shape, ordering and grid contracts apply.
    #[cutile::entry()]
    pub unsafe fn prepare_large<const NR: i32, const PR: i32, const NO: i32>(
        experts: *mut i64,
        map: *mut i32,
        counts: *mut i32,
        o: *mut f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let indices: Tile<i32, { [PR] }> = iota(shape![PR]);
        let valid: Tile<bool, { [PR] }> = lt_tile(indices, broadcast_scalar(NR, shape![PR]));
        let ids64: Tile<i64, { [PR] }> =
            unsafe { gather_i64::<PR, 0>(experts, indices, valid, 0i64) };
        let ids: Tile<i32, { [PR] }> = trunci(ids64, overflow::NoSignedWrap);
        let mask: Tile<bool, { [PR] }> = valid & eq_tile(ids, broadcast_scalar(pid.0, shape![PR]));
        let zero: Tile<i32, { [PR] }> = constant(0i32, shape![PR]);
        let one: Tile<i32, { [PR] }> = constant(1i32, shape![PR]);
        let values: Tile<i32, { [PR] }> = select(mask, one, zero);
        let ranks: Tile<i32, { [PR] }> = scan_sum(values, 0i32, reverse::Forward, 0i32) - one;
        unsafe {
            scatter_i32(
                map,
                broadcast_scalar(pid.0 * NR, shape![PR]) + ranks,
                indices,
                mask,
            );
        }
        let count: Tile<i32, { [] }> = reduce_sum(values, 0i32);
        let n: i32 = tile_to_scalar(count);
        unsafe {
            write_i32(counts, pid.0, n);
            zero_output::<NO>(o);
        }
    }
    /// One stage of the routed MoE pipeline.
    ///
    /// # Safety
    /// The module's buffer, shape, ordering and grid contracts apply.
    #[cutile::entry()]
    pub unsafe fn schedule<const EXPERTS: i32, const PG: i32, const BM: i32>(
        counts: *mut i32,
        tasks: *mut i32,
        total: *mut i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let indices: Tile<i32, { [256] }> = iota(shape![256]);
        let mask: Tile<bool, { [256] }> = lt_tile(indices, broadcast_scalar(EXPERTS, shape![256]));
        let n: Tile<i32, { [256] }> = unsafe { gather_i32::<256, 0>(counts, indices, mask, 0i32) };
        let groups: Tile<i32, { [256] }> =
            (n + broadcast_scalar(BM - 1i32, shape![256])) / broadcast_scalar(BM, shape![256]);
        let starts: Tile<i32, { [256] }> = scan_sum(groups, 0i32, reverse::Forward, 0i32) - groups;
        let this: Tile<bool, { [256] }> = eq_tile(indices, broadcast_scalar(pid.0, shape![256]));
        let zero: Tile<i32, { [256] }> = constant(0i32, shape![256]);
        let start: Tile<i32, { [] }> = reduce_sum(select(this, starts, zero), 0i32);
        let start: i32 = tile_to_scalar(start);
        let count: i32 = unsafe { read_i32(counts, pid.0) };
        let ii: Tile<i32, { [PG] }> = iota(shape![PG]);
        let valid: Tile<bool, { [PG] }> =
            lt_tile(ii, broadcast_scalar((count + BM - 1i32) / BM, shape![PG]));
        let index: Tile<i32, { [PG] }> =
            (broadcast_scalar(start, shape![PG]) + ii) * broadcast_scalar(2i32, shape![PG]);
        unsafe {
            scatter_i32(tasks, index, broadcast_scalar(pid.0, shape![PG]), valid);
            scatter_i32(tasks, index + broadcast_scalar(1i32, shape![PG]), ii, valid);
        }
        if pid.0 == 0i32 {
            let sum: Tile<i32, { [] }> = reduce_sum(groups, 0i32);
            let sum: i32 = tile_to_scalar(sum);
            unsafe {
                write_i32(total, 0i32, sum);
            }
        }
    }
    unsafe fn row_gather<const M: i32, const K: i32, const F: i32, const L: i32>(
        base: *mut f16,
        row: Tile<i32, { [M] }>,
        valid: Tile<bool, { [M] }>,
        stride: i32,
        column: i32,
    ) -> Tile<f16, { [M, K] }> {
        let rr: Tile<i32, { [M, K] }> = row.reshape(shape![M, 1]).broadcast(shape![M, K]);
        let cc: Tile<i32, { [K] }> = broadcast_scalar(column, shape![K]) + iota(shape![K]);
        let cc: Tile<i32, { [M, K] }> = cc.reshape(shape![1, K]).broadcast(shape![M, K]);
        let index: Tile<i32, { [M, K] }> = rr * broadcast_scalar(stride, shape![M, K]) + cc;
        let index: Tile<i32, { [F] }> = index.reshape(shape![F]);
        let mask: Tile<bool, { [M, K] }> = valid.reshape(shape![M, 1]).broadcast(shape![M, K]);
        let mask: Tile<bool, { [F] }> = mask.reshape(shape![F]);
        let zero: f16 = convert_scalar(0.0f32);
        let values: Tile<f16, { [F] }> = unsafe { gather_f16::<F, L>(base, index, mask, zero) };
        values.reshape(shape![M, K])
    }
    unsafe fn row_scatter<const M: i32, const N: i32, const F: i32>(
        base: *mut f16,
        row: Tile<i32, { [M] }>,
        valid: Tile<bool, { [M] }>,
        stride: i32,
        column: i32,
        values: Tile<f16, { [M, N] }>,
    ) {
        let rr: Tile<i32, { [M, N] }> = row.reshape(shape![M, 1]).broadcast(shape![M, N]);
        let cc: Tile<i32, { [N] }> = broadcast_scalar(column, shape![N]) + iota(shape![N]);
        let cc: Tile<i32, { [M, N] }> = cc.reshape(shape![1, N]).broadcast(shape![M, N]);
        let index: Tile<i32, { [M, N] }> = rr * broadcast_scalar(stride, shape![M, N]) + cc;
        let index: Tile<i32, { [F] }> = index.reshape(shape![F]);
        let mask: Tile<bool, { [M, N] }> = valid.reshape(shape![M, 1]).broadcast(shape![M, N]);
        let mask: Tile<bool, { [F] }> = mask.reshape(shape![F]);
        let values: Tile<f16, { [F] }> = values.reshape(shape![F]);
        unsafe {
            scatter_f16(base, index, values, mask);
        }
    }
    unsafe fn row_add<const M: i32, const N: i32, const F: i32>(
        base: *mut f32,
        row: Tile<i32, { [M] }>,
        valid: Tile<bool, { [M] }>,
        stride: i32,
        column: i32,
        values: Tile<f32, { [M, N] }>,
    ) {
        let rr: Tile<i32, { [M, N] }> = row.reshape(shape![M, 1]).broadcast(shape![M, N]);
        let cc: Tile<i32, { [N] }> = broadcast_scalar(column, shape![N]) + iota(shape![N]);
        let cc: Tile<i32, { [M, N] }> = cc.reshape(shape![1, N]).broadcast(shape![M, N]);
        let index: Tile<i32, { [M, N] }> = rr * broadcast_scalar(stride, shape![M, N]) + cc;
        let index: Tile<i32, { [F] }> = index.reshape(shape![F]);
        let mask: Tile<bool, { [M, N] }> = valid.reshape(shape![M, 1]).broadcast(shape![M, N]);
        let mask: Tile<bool, { [F] }> = mask.reshape(shape![F]);
        let values: Tile<f32, { [F] }> = values.reshape(shape![F]);
        let ptr: PointerTile<*mut f32, { [F] }> = unsafe { pointers_f32(base, index) };
        let (_old, _token): (Tile<f32, { [F] }>, Token) = unsafe {
            atomic_rmw_tko(
                ptr,
                values,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(mask),
                None,
            )
        };
    }
    unsafe fn tma_load<const M: i32, const K: i32, const L: i32>(
        src: &Tensor<f16, { [-1, -1] }>,
        row: i32,
        col: i32,
    ) -> Tile<f16, { [M, K] }> {
        let token: Token = src.token();
        let part = unsafe {
            make_partition_view(src, shape![M, K], padding::None, dim_map::Identity, token)
        };
        let (result, _token): (Tile<f16, { [M, K] }>, Token) = unsafe {
            load_view_raw(
                &part,
                [row, col],
                Some(token),
                ordering::Weak,
                scope::TileBlock,
                tma::Enabled,
                Some(L),
                [true, true],
                shape![M, K],
            )
        };
        result
    }
    unsafe fn tma_store<const M: i32, const N: i32>(
        src: &mut Tensor<f16, { [-1, -1] }>,
        row: i32,
        col: i32,
        values: Tile<f16, { [M, N] }>,
    ) {
        let mut part = src.partition_mut(shape![M, N]);
        unsafe {
            store_view_tko_mut(
                &mut part,
                values,
                [row, col],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
        }
    }
    /// One stage of the routed MoE pipeline.
    ///
    /// # Safety
    /// The module's buffer, shape, ordering and grid contracts apply.
    #[cutile::entry()]
    pub unsafe fn pack_input<
        const H: i32,
        const R: i32,
        const BM: i32,
        const NR: i32,
        const CAP: i32,
        const F: i32,
    >(
        x: *mut f16,
        map: *mut i32,
        counts: *mut i32,
        tasks: *mut i32,
        total: *mut i32,
        packed: *mut f16,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let nb: (i32, i32, i32) = get_num_tile_blocks();
        let nt: i32 = unsafe { read_i32(total, 0i32) };
        let mut output: Tensor<f16, { [-1, -1] }> = unsafe { view(packed, CAP * BM, H) };
        let mut job = pid.0;
        while job < nt * (H / 256i32) {
            let task = job / (H / 256i32);
            let column = job % (H / 256i32);
            let expert: i32 = unsafe { read_i32(tasks, task * 2i32) };
            let group: i32 = unsafe { read_i32(tasks, task * 2i32 + 1i32) };
            let count: i32 = unsafe { read_i32(counts, expert) };
            let rows: Tile<i32, { [BM] }> =
                broadcast_scalar(group * BM, shape![BM]) + iota(shape![BM]);
            let valid: Tile<bool, { [BM] }> = lt_tile(rows, broadcast_scalar(count, shape![BM]));
            let indices: Tile<i32, { [BM] }> = broadcast_scalar(expert * NR, shape![BM]) + rows;
            let routes: Tile<i32, { [BM] }> =
                unsafe { gather_i32::<BM, 0>(map, indices, valid, 0i32) };
            let routes: Tile<i32, { [BM] }> = routes.reshape(shape![BM]);
            let tokens: Tile<i32, { [BM] }> = routes / broadcast_scalar(R, shape![BM]);
            let values: Tile<f16, { [BM, 256] }> =
                unsafe { row_gather::<BM, 256, F, 0>(x, tokens, valid, H, column * 256i32) };
            unsafe {
                tma_store::<BM, 256>(&mut output, task, column, values);
            }
            job = job + nb.0;
        }
    }
    /// One stage of the routed MoE pipeline.
    ///
    /// # Safety
    /// The module's buffer, shape, ordering and grid contracts apply.
    #[cutile::entry()]
    pub unsafe fn fc1<
        const H: i32,
        const I: i32,
        const EXPERTS: i32,
        const R: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const NR: i32,
        const CAP: i32,
        const PACKED: bool,
        const FK: i32,
        const FN: i32,
    >(
        source: *mut f16,
        up: *mut f16,
        map: *mut i32,
        counts: *mut i32,
        tasks: *mut i32,
        total: *mut i32,
        a: *mut f16,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let nb: (i32, i32, i32) = get_num_tile_blocks();
        let nt: i32 = unsafe { read_i32(total, 0i32) };
        let src: Tensor<f16, { [-1, -1] }> =
            unsafe { view(source, if PACKED { CAP * BM } else { NR / R }, H) };
        let w: Tensor<f16, { [-1, -1] }> = unsafe { view(up, EXPERTS * I, H) };
        let mut out: Tensor<f16, { [-1, -1] }> =
            unsafe { view(a, if PACKED { CAP * BM } else { NR }, I) };
        let gather_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(source);
        let gather_base: PointerTile<*mut f16, { [1, 1] }> = gather_base.reshape(shape![1, 1]);
        let gather_base: PointerTile<*mut f16, { [BM, BK] }> =
            gather_base.broadcast(shape![BM, BK]);
        let inner: Tile<i32, { [BK] }> = iota(shape![BK]);
        let gather_zero_tile: Tile<f16, { [] }> = constant(f16::ZERO, shape![]);
        let gather_zero: f16 = tile_to_scalar(gather_zero_tile);
        let gather_stride32: Tile<i32, { [] }> = scalar_to_tile(H);
        let gather_stride64: Tile<i64, { [] }> = exti(gather_stride32);
        let gather_stride: Tile<i64, { [BM, BK] }> = gather_stride64
            .reshape(shape![1, 1])
            .broadcast(shape![BM, BK]);
        let gather_height32: Tile<i32, { [] }> = scalar_to_tile(NR / R);
        let gather_height64: Tile<i64, { [] }> = exti(gather_height32);
        let gather_height: Tile<i64, { [BM, BK] }> = gather_height64
            .reshape(shape![1, 1])
            .broadcast(shape![BM, BK]);
        let gather_height_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_height);
        let gather_stride_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_stride);
        let mut job = pid.0;
        while job < nt * (I / BN) {
            let task = job / (I / BN);
            let column = job % (I / BN);
            let expert: i32 = unsafe { read_i32(tasks, task * 2i32) };
            let group: i32 = unsafe { read_i32(tasks, task * 2i32 + 1i32) };
            let count: i32 = unsafe { read_i32(counts, expert) };
            let rows: Tile<i32, { [BM] }> =
                broadcast_scalar(group * BM, shape![BM]) + iota(shape![BM]);
            let valid: Tile<bool, { [BM] }> = lt_tile(rows, broadcast_scalar(count, shape![BM]));
            let indices: Tile<i32, { [BM] }> = broadcast_scalar(expert * NR, shape![BM]) + rows;
            let routes: Tile<i32, { [BM] }> =
                unsafe { gather_i32::<BM, 0>(map, indices, valid, 0i32) };
            let routes: Tile<i32, { [BM] }> = routes.reshape(shape![BM]);
            let tokens: Tile<i32, { [BM] }> = divi(
                routes,
                broadcast_scalar(R, shape![BM]),
                rounding::NegativeInf,
            );
            let gather_rows: Tile<i32, { [BM] }> = tokens.reshape(shape![BM]);
            let gather_rows64: Tile<i64, { [BM] }> = exti(gather_rows);
            let gather_rows64: Tile<i64, { [BM, BK] }> = gather_rows64
                .reshape(shape![BM, 1])
                .broadcast(shape![BM, BK]);
            let gather_rows_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_rows64);
            let gather_rowmask: Tile<bool, { [BM, BK] }> =
                lt_tile(gather_rows_unsigned, gather_height_unsigned);
            let gather_valid: Tile<bool, { [BM, BK] }> =
                valid.reshape(shape![BM, 1]).broadcast(shape![BM, BK]);
            let gather_offset: Tile<i64, { [BM, BK] }> = gather_rows64 * gather_stride;
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            for part in 0i32..H / BK {
                let values: Tile<f16, { [BM, BK] }> = if PACKED {
                    unsafe { tma_load::<BM, BK, 1>(&src, task, part) }
                } else {
                    {
                        let gather_cols: Tile<i32, { [BK] }> =
                            inner + broadcast_scalar(part * BK, shape![BK]);
                        let gather_cols64: Tile<i64, { [BK] }> = exti(gather_cols);
                        let gather_cols64: Tile<i64, { [BM, BK] }> = gather_cols64
                            .reshape(shape![1, BK])
                            .broadcast(shape![BM, BK]);
                        let gather_cols_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_cols64);
                        let gather_colmask: Tile<bool, { [BM, BK] }> =
                            lt_tile(gather_cols_unsigned, gather_stride_unsigned);
                        let gather_mask: Tile<bool, { [BM, BK] }> =
                            gather_rowmask & gather_colmask & gather_valid;
                        let gather_ptr: PointerTile<*mut f16, { [BM, BK] }> =
                            addptr_tile(gather_base, gather_offset + gather_cols64);
                        let (gather_value, _token): (Tile<f16, { [BM, BK] }>, Token) = unsafe {
                            load_ptr_tko(
                                gather_ptr,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(gather_mask),
                                Some(gather_zero),
                                None,
                                Latency::<1>,
                            )
                        };
                        let result: Tile<f16, { [BM, BK] }> = gather_value;
                        result
                    }
                };
                let weights: Tile<f16, { [BN, BK] }> =
                    unsafe { tma_load::<BN, BK, 8>(&w, expert * (I / BN) + column, part) };
                let weights: Tile<f16, { [BK, BN] }> = permute(weights, const_array![1, 0]);
                acc = mma(values, weights, acc);
            }
            let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            let relu: Tile<f32, { [BM, BN] }> = maxf(acc, zero, nan::Disabled, ftz::Disabled);
            let square: Tile<f32, { [BM, BN] }> = relu * relu;
            let result: Tile<f16, { [BM, BN] }> = convert_tile(square);
            if PACKED {
                unsafe {
                    tma_store::<BM, BN>(&mut out, task, column, result);
                }
            } else {
                unsafe {
                    row_scatter::<BM, BN, FN>(a, routes, valid, I, column * BN, result);
                }
            }
            job = job + nb.0;
        }
    }
    /// One stage of the routed MoE pipeline.
    ///
    /// # Safety
    /// The module's buffer, shape, ordering and grid contracts apply.
    #[cutile::entry()]
    pub unsafe fn fc2<
        const H: i32,
        const I: i32,
        const EXPERTS: i32,
        const R: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const NR: i32,
        const CAP: i32,
        const PACKED: bool,
        const FK: i32,
        const FN: i32,
    >(
        a: *mut f16,
        down: *mut f16,
        map: *mut i32,
        counts: *mut i32,
        tasks: *mut i32,
        total: *mut i32,
        weights: *mut f32,
        o: *mut f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let nb: (i32, i32, i32) = get_num_tile_blocks();
        let nt: i32 = unsafe { read_i32(total, 0i32) };
        let src: Tensor<f16, { [-1, -1] }> =
            unsafe { view(a, if PACKED { CAP * BM } else { NR }, I) };
        let w: Tensor<f16, { [-1, -1] }> = unsafe { view(down, EXPERTS * H, I) };
        let gather_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(a);
        let gather_base: PointerTile<*mut f16, { [1, 1] }> = gather_base.reshape(shape![1, 1]);
        let gather_base: PointerTile<*mut f16, { [BM, BK] }> =
            gather_base.broadcast(shape![BM, BK]);
        let inner: Tile<i32, { [BK] }> = iota(shape![BK]);
        let gather_zero_tile: Tile<f16, { [] }> = constant(f16::ZERO, shape![]);
        let gather_zero: f16 = tile_to_scalar(gather_zero_tile);
        let gather_stride32: Tile<i32, { [] }> = scalar_to_tile(I);
        let gather_stride64: Tile<i64, { [] }> = exti(gather_stride32);
        let gather_stride: Tile<i64, { [BM, BK] }> = gather_stride64
            .reshape(shape![1, 1])
            .broadcast(shape![BM, BK]);
        let gather_height32: Tile<i32, { [] }> = scalar_to_tile(NR);
        let gather_height64: Tile<i64, { [] }> = exti(gather_height32);
        let gather_height: Tile<i64, { [BM, BK] }> = gather_height64
            .reshape(shape![1, 1])
            .broadcast(shape![BM, BK]);
        let gather_height_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_height);
        let gather_stride_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_stride);
        let mut job = pid.0;
        while job < nt * (H / BN) {
            let task = job / (H / BN);
            let column = job % (H / BN);
            let expert: i32 = unsafe { read_i32(tasks, task * 2i32) };
            let group: i32 = unsafe { read_i32(tasks, task * 2i32 + 1i32) };
            let count: i32 = unsafe { read_i32(counts, expert) };
            let rows: Tile<i32, { [BM] }> =
                broadcast_scalar(group * BM, shape![BM]) + iota(shape![BM]);
            let valid: Tile<bool, { [BM] }> = lt_tile(rows, broadcast_scalar(count, shape![BM]));
            let indices: Tile<i32, { [BM] }> = broadcast_scalar(expert * NR, shape![BM]) + rows;
            let routes: Tile<i32, { [BM] }> =
                unsafe { gather_i32::<BM, 0>(map, indices, valid, 0i32) };
            let gather_rows: Tile<i32, { [BM] }> = routes.reshape(shape![BM]);
            let gather_rows64: Tile<i64, { [BM] }> = exti(gather_rows);
            let gather_rows64: Tile<i64, { [BM, BK] }> = gather_rows64
                .reshape(shape![BM, 1])
                .broadcast(shape![BM, BK]);
            let gather_rows_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_rows64);
            let gather_rowmask: Tile<bool, { [BM, BK] }> =
                lt_tile(gather_rows_unsigned, gather_height_unsigned);
            let gather_valid: Tile<bool, { [BM, BK] }> =
                valid.reshape(shape![BM, 1]).broadcast(shape![BM, BK]);
            let gather_offset: Tile<i64, { [BM, BK] }> = gather_rows64 * gather_stride;
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            for part in 0i32..I / BK {
                let values: Tile<f16, { [BM, BK] }> = if PACKED {
                    unsafe { tma_load::<BM, BK, 1>(&src, task, part) }
                } else {
                    {
                        let gather_cols: Tile<i32, { [BK] }> =
                            inner + broadcast_scalar(part * BK, shape![BK]);
                        let gather_cols64: Tile<i64, { [BK] }> = exti(gather_cols);
                        let gather_cols64: Tile<i64, { [BM, BK] }> = gather_cols64
                            .reshape(shape![1, BK])
                            .broadcast(shape![BM, BK]);
                        let gather_cols_unsigned: Tile<u64, { [BM, BK] }> = bitcast(gather_cols64);
                        let gather_colmask: Tile<bool, { [BM, BK] }> =
                            lt_tile(gather_cols_unsigned, gather_stride_unsigned);
                        let gather_mask: Tile<bool, { [BM, BK] }> =
                            gather_rowmask & gather_colmask & gather_valid;
                        let gather_ptr: PointerTile<*mut f16, { [BM, BK] }> =
                            addptr_tile(gather_base, gather_offset + gather_cols64);
                        let (gather_value, _token): (Tile<f16, { [BM, BK] }>, Token) = unsafe {
                            load_ptr_tko(
                                gather_ptr,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(gather_mask),
                                Some(gather_zero),
                                None,
                                Latency::<1>,
                            )
                        };
                        let result: Tile<f16, { [BM, BK] }> = gather_value;
                        result
                    }
                };
                let wt: Tile<f16, { [BN, BK] }> =
                    unsafe { tma_load::<BN, BK, 8>(&w, expert * (H / BN) + column, part) };
                let wt: Tile<f16, { [BK, BN] }> = permute(wt, const_array![1, 0]);
                acc = mma(values, wt, acc);
            }
            let scale: Tile<f32, { [BM] }> =
                unsafe { gather_f32::<BM, 0>(weights, routes, valid, 0.0f32) };
            let scale: Tile<f32, { [BM, BN] }> =
                scale.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
            let result: Tile<f32, { [BM, BN] }> = acc * scale;
            let routes: Tile<i32, { [BM] }> = routes.reshape(shape![BM]);
            let tokens: Tile<i32, { [BM] }> = divi(
                routes,
                broadcast_scalar(R, shape![BM]),
                rounding::NegativeInf,
            );
            unsafe {
                row_add::<BM, BN, FN>(o, tokens, valid, H, column * BN, result);
            }
            job = job + nb.0;
        }
    }
}
