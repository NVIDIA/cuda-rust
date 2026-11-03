/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// These kernels pin the behaviour of the deprecated annotation path, which
// must keep working unchanged until it is removed.
#![allow(deprecated)]

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod opt_hints_module {
    use cutile::core::*;

    #[cutile::entry]
    fn load_ptr_latency_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);
        let (loaded, _tok): (Tile<f32, S>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<4>,
            )
        };
        output.store(loaded);
    }

    #[cutile::entry]
    fn load_ptr_const_latency_kernel<const S: [i32; 1], const L: i32>(output: &mut Tensor<f32, S>) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);
        let (loaded, _tok): (Tile<f32, S>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<L>,
            )
        };
        output.store(loaded);
    }

    #[cutile::entry]
    fn store_ptr_latency_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);
        let vals: Tile<f32, S> = constant(1.0f32, output.shape());
        let _tok: Token = unsafe {
            store_ptr_tko(
                ptrs,
                vals,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<2>,
            )
        };
        output.store(vals);
    }

    #[cutile::entry]
    fn option_bindings_for_optional_operands_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
    ) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        let scope_opt: Option<scope::Device> = Some(scope::Device);
        let mask_opt: Option<Tile<bool, S>> = None;
        let padding_opt: Option<f32> = None;
        let token_none: Option<Token> = None;
        let (loaded, _tok): (Tile<f32, S>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Relaxed,
                scope_opt,
                mask_opt,
                padding_opt,
                token_none,
                Latency::<0>,
            )
        };

        let token: Token = new_token_unordered();
        let token_some: Option<Token> = Some(token);
        let scope_none: Option<scope::TileBlock> = None;
        let _store_tok: Token = unsafe {
            store_ptr_tko(
                ptrs,
                loaded,
                ordering::Weak,
                scope_none,
                None,
                token_some,
                Latency::<0>,
            )
        };

        output.store(loaded);
    }

    #[cutile::entry]
    fn load_view_latency_kernel<const S: [i32; 1]>(input: &Tensor<f32, S>) {
        let token: Token = new_token_unordered();
        let shape = input.shape();
        // SAFETY: this read-only input has no preceding writes in the kernel.
        let partition: Partition<f32, S> =
            unsafe { make_partition_view(input, shape, padding::None, dim_map::Identity, token) };
        let idx: [i32; 1] = [0i32];
        let _tile: Tile<f32, S> = load_view_tko(
            &partition,
            idx,
            ordering::Weak,
            scope::TileBlock,
            Some(8),
            tma::Enabled,
        );
    }

    #[cutile::entry]
    fn store_view_disallow_tma_kernel<const S: [i32; 1]>(y: &mut Tensor<f32, S>) {
        let shape = y.shape();
        let token: Token = get_tensor_token(y);
        let mut partition: PartitionMut<f32, S> =
            unsafe { make_partition_view_mut(y, shape, padding::None, token) };
        let tile: Tile<f32, S> = constant(1.0f32, shape);
        let idx: [i32; 1] = [0i32];
        unsafe {
            store_view_tko_mut(
                &mut partition,
                tile,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Disabled,
            );
        }
    }

    #[cutile::entry(optimization_hints = (
        sm_120 = (
            occupancy = 4,
            num_cta_in_cga = 2,
            num_worker_warps_per_cta = 4,
        ),
    ))]
    fn entry_hints_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    #[cutile::entry(optimization_hints = (
        sm_120 = (num_worker_warps_per_cta = 1,),
    ))]
    fn worker_warps_value_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    /// Latency as a const generic — specialized at launch time.
    #[cutile::entry]
    fn load_view_const_latency_kernel<const S: [i32; 1], const L: i32>(input: &Tensor<f32, S>) {
        let token: Token = new_token_unordered();
        let shape = input.shape();
        // SAFETY: this read-only input has no preceding writes in the kernel.
        let partition: Partition<f32, S> =
            unsafe { make_partition_view(input, shape, padding::None, dim_map::Identity, token) };
        let idx: [i32; 1] = [0i32];
        let _tile: Tile<f32, S> = load_view_tko(
            &partition,
            idx,
            ordering::Weak,
            scope::TileBlock,
            Some(L),
            tma::Enabled,
        );
    }

    /// Safe checked load with a literal latency hint.
    #[cutile::entry]
    fn safe_load_pipelined_kernel<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let part = x.partition(shape!(S));
        let tile = part.load_pipelined::<6>([0]);
        z.store(tile);
    }

    /// Safe checked load with the latency hint from a kernel const generic.
    #[cutile::entry]
    fn safe_load_pipelined_const_kernel<const S: [i32; 1], const L: i32>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let part = x.partition(shape!(S));
        let tile = part.load_pipelined::<L>([0]);
        z.store(tile);
    }

    /// Two loads with different latency hints in one kernel: each must keep
    /// its own hint (not last-wins).
    #[cutile::entry]
    fn safe_load_two_latencies_kernel<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let part_x = x.partition(shape!(S));
        let part_y = y.partition(shape!(S));
        let tile_x = part_x.load_pipelined::<3>([0]);
        let tile_y = part_y.load_pipelined::<9>([0]);
        z.store(tile_x + tile_y);
    }

    /// Safe bounded load with a latency hint: the proof path (`with_bounds`
    /// + `coord`) and the pipelining knob compose.
    #[cutile::entry]
    fn safe_bounded_load_pipelined_kernel<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = Dim::new(x.shape()[0] / BM);
        let n = Dim::new(x.shape()[1] / BN);
        let part = x.partition(shape![BM, BN]).with_bounds((m, n));
        for i in m {
            for j in n {
                let tile = part.load_pipelined::<7>(coord((i, j)));
                z.store(tile);
            }
        }
    }

    /// The two-statement spelling: `partition_mut` bound to a name, then
    /// `with_bounds` on the binding. Semantically identical to the chained
    /// form, and it once failed typeck at the subsequent `.store(..)` — this
    /// kernel keeps that path compiling, with both bindings folding against
    /// the declared `[1, N]` shape.
    #[cutile::entry]
    fn two_stmt_with_bounds<const N: i32, const B: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / B);
        let out_part = out.partition_mut(shape![1, B]);
        let mut out_part = out_part.with_bounds((Dim::new(1), cols));
        for j in cols {
            let t: Tile<f32, { [1, B] }> = constant(0.0, shape![1, B]);
            out_part.store(t, coord((0i32, j)));
        }
    }

    /// The shared-extent pattern (Theorem 2): one `Dim`, derived from `x`'s
    /// contraction extent, bound to an axis of *both* partitions — the GEMM
    /// spelling. The declared precondition states the extents are equal, so
    /// the cross-tensor binding discharges from that fact plus the numerator's
    /// own divisibility check: zero in-kernel asserts.
    #[cutile::entry(
        preconditions = (
            dim(x, 1) == dim(y, 0),
        )
    )]
    fn shared_dim_binding<const BM: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BK] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = Dim::new(x.shape()[0] / BM);
        let k = Dim::new(x.shape()[1] / BK);
        let n = Dim::new(y.shape()[1] / BM);
        let xp = x.partition(shape![BM, BK]).with_bounds((m, k));
        let yp = y.partition(shape![BK, BM]).with_bounds((k, n));
        for i in m {
            for j in k {
                let tx = xp.load(coord((i, j)));
                z.store(tx);
            }
        }
        for j in k {
            for l in n {
                let ty = yp.load(coord((j, l)));
                z.store(ty.transpose());
            }
        }
    }

    /// Fail-closed twin of `shared_dim_binding`: same kernel, no declared
    /// equality. The cross-tensor binding must NOT hoist — an unproven extent
    /// equality is strictly stronger than the tile-count equality the binding
    /// obliges, so hoisting it unasked would reject launches the device assert
    /// accepts. It stays a device assert until the kernel opts in.
    #[cutile::entry]
    fn shared_dim_binding_undeclared<const BM: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BK] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = Dim::new(x.shape()[0] / BM);
        let k = Dim::new(x.shape()[1] / BK);
        let n = Dim::new(y.shape()[1] / BM);
        let xp = x.partition(shape![BM, BK]).with_bounds((m, k));
        let yp = y.partition(shape![BK, BM]).with_bounds((k, n));
        for i in m {
            for j in k {
                let tx = xp.load(coord((i, j)));
                z.store(tx);
            }
        }
        for j in k {
            for l in n {
                let ty = yp.load(coord((j, l)));
                z.store(ty.transpose());
            }
        }
    }

    /// Declared divisibility, the payoff of stating binding obligations over
    /// the extent atom: with `dim(x, k) % 64 == 0` declared (and so verified by
    /// the launcher before the kernel runs), both binding obligations are
    /// entailed at JIT. No in-kernel assert, and no launch check either -- the
    /// declared facts already carry the verification.
    #[cutile::entry(
        preconditions = (
            dim(x, 0) % 64 == 0,
            dim(x, 1) % 64 == 0,
        )
    )]
    fn declared_divisibility_binding<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = Dim::new(x.shape()[0] / BM);
        let n = Dim::new(x.shape()[1] / BN);
        let part = x.partition(shape![BM, BN]).with_bounds((m, n));
        for i in m {
            for j in n {
                let tile = part.load(coord((i, j)));
                z.store(tile);
            }
        }
    }

    /// SOUNDNESS PROBE (immutable / read side): the same unvalidated contract on
    /// `Tensor::partition`, which has been safe on `main` all along. The declared
    /// row bound is absurd (999); nothing checks it against the view's tile count.
    #[cutile::entry]
    fn with_bounds_bogus_dim_load<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let bogus_rows = Dim::new(999);
        let n = Dim::new(x.shape()[1] / BN);
        let part = x.partition(shape![BM, BN]).with_bounds((bogus_rows, n));
        for i in bogus_rows {
            for j in n {
                let tile = part.load(coord((i, j)));
                z.store(tile);
            }
        }
    }

    /// SOUNDNESS PROBE: `with_bounds` is an *unvalidated* contract — the `Dim`
    /// value is never checked against the view's real tile count. The declared
    /// row bound here is absurd (999) for a single-row view, yet the branded
    /// store discharges and no check is emitted. This is safe Rust: it must be
    /// rejected, or the bound must become a verified `Launch` obligation.
    #[cutile::entry]
    fn with_bounds_bogus_dim<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let bogus_rows = Dim::new(999);
        let tile_shape = shape![1, BLOCK_SIZE];
        let mut out_part = out
            .partition_mut(tile_shape)
            .with_bounds((bogus_rows, cols));
        for i in bogus_rows {
            for j in cols {
                let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
                out_part.store(tile, coord((i, j)));
            }
        }
    }

    /// Mutable mirror of `with_bounds`: `partition_mut` → `with_bounds` →
    /// branded-coord `store`. The row is re-based (constant `0`) so only the
    /// branded column index `j` varies — the raw kernel's tight store shape.
    #[cutile::entry]
    fn bounded_mut_store_smoke<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let num_tiles = N / BLOCK_SIZE;
        let cols = Dim::new(num_tiles);
        let tile_shape = shape![1, BLOCK_SIZE];
        let mut out_part = out
            .partition_mut(tile_shape)
            .with_bounds((Dim::new(1), cols));
        for j in cols {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            out_part.store(tile, coord((0i32, j)));
        }
    }

    /// The launch-hoist rungs with a *genuinely* dynamic extent: the row count
    /// is declared `-1`, so no compile-time source can decide either the
    /// binding `tiles(out, 0) == 1` or the constant row-`0` access. Both leave
    /// the kernel for the host, in the *view* frame — a `&mut Tensor` is
    /// slabbed, so the checks are over the per-CTA slab extent, not the root.
    ///
    /// Contrast `bounded_mut_store_smoke`, whose `[1, N]` signature pins the row
    /// extent: there everything discharges at compile time and no launch check
    /// is produced at all.
    #[cutile::entry]
    fn hoisted_dynamic_row_store<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [-1, N] }>,
    ) {
        let num_tiles = N / BLOCK_SIZE;
        let cols = Dim::new(num_tiles);
        let tile_shape = shape![1, BLOCK_SIZE];
        let mut out_part = out
            .partition_mut(tile_shape)
            .with_bounds((Dim::new(1), cols));
        for j in cols {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            out_part.store(tile, coord((0i32, j)));
        }
    }

    /// Raw (unchecked) twin of `bounded_mut_store_smoke`: the identical store
    /// loop via `partition_mut` + an unchecked `store`, with no bounds check at
    /// all. This is the register baseline the hoisted safe store must match —
    /// the two kernels differ only in the store's safety, so any register delta
    /// is exactly the cost of the store's bounds check.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn unchecked_mut_store_smoke<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let num_tiles = N / BLOCK_SIZE;
        let tile_shape = shape![1, BLOCK_SIZE];
        let mut out_part = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            out_part.store(tile, [0i32, j]);
        }
    }

    /// A safe bounded store whose row is a tile-block id (`get_tile_block_id().0`)
    /// into a dynamic-row `&mut Tensor` — the owned-axis (per-CTA row) pattern.
    /// The row-axis access `TileBlockId(0) < num_partitions(out, 0)` discharges
    /// at JIT via the universal hardware axiom `TileBlockId(0) < NumTileBlocks(0)`
    /// (the direct partition's launch grid is verified equal to its partition
    /// grid), so NO in-kernel bounds check is emitted for the row axis.
    #[cutile::entry]
    fn blockid_row_store<const N: i32, const BLOCK_SIZE: i32>(out: &mut Tensor<f32, { [-1, N] }>) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let rows = Dim::new(get_num_tile_blocks().0);
        let tile_shape = shape![1, BLOCK_SIZE];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let mut out_part = out.partition_mut(tile_shape).with_bounds((rows, cols));
        for j in cols {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            out_part.store(tile, coord((row, j)));
        }
    }

    /// Negative: a block-id whose *component* does not match the axis it indexes
    /// must be rejected. Here grid component 1 (`pid.1`) is used as the row (axis
    /// 0): `TileBlockId(1) < NumTileBlocks(1)` does not bound `num_partitions(out,
    /// 0)`, so the discharge must not fire — the correspondence gate is `k == axis`.
    #[cutile::entry]
    fn blockid_wrong_axis_store<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [-1, N] }>,
    ) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let rows = Dim::new(get_num_tile_blocks().0);
        let tile_shape = shape![1, BLOCK_SIZE];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wrong = pid.1; // component 1 used as the axis-0 row
        let mut out_part = out.partition_mut(tile_shape).with_bounds((rows, cols));
        for j in cols {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            out_part.store(tile, coord((wrong, j)));
        }
    }

    /// Raw (unchecked) twin of `blockid_row_store` — same loop, unchecked store.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn unchecked_blockid_row_store<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [-1, N] }>,
    ) {
        let num_tiles = N / BLOCK_SIZE;
        let tile_shape = shape![1, BLOCK_SIZE];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let mut out_part = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            out_part.store(tile, [row, j]);
        }
    }

    /// A checked load whose index (`2 * tile-block id`) is not provably in
    /// bounds — the block-id axiom gives `pid.0 < gridDim.0` but says nothing
    /// about `2 * pid.0`, so a bounds check must remain in the kernel. Compiles
    /// normally: the residual check is emitted in-kernel.
    #[cutile::entry]
    fn residual_in_kernel_check<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part = x.partition(shape![BM, BN]);
        let pid = get_tile_block_id();
        let tile: Tile<f32, { [BM, BN] }> = part.load([pid.0 * 2i32, pid.1]);
        z.store(tile);
    }

    /// Identical body under `deny_in_kernel_checks`: the residual bounds check
    /// makes this a hard JIT error naming the axis that could not leave the
    /// kernel. (Contrast `unchecked_accesses`, which would drop the check.)
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn residual_in_kernel_check_denied<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part = x.partition(shape![BM, BN]);
        let pid = get_tile_block_id();
        let tile: Tile<f32, { [BM, BN] }> = part.load([pid.0 * 2i32, pid.1]);
        z.store(tile);
    }

    /// DENY-COVERAGE PROBE: `deny_in_kernel_checks` must cover *every*
    /// compiler-emitted safety check, not just the partition-access one. Here
    /// the view extent is dynamic, so the `with_bounds` binding obligation
    /// cannot fold at JIT and lands as an in-kernel assert — a residual check
    /// that costs device registers, which is exactly what the attribute
    /// forbids.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn with_bounds_binding_check_denied<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [-1, -1] }>,
    ) {
        // A special-register-derived bound has no symbolic form at any stage
        // below Device: it cannot fold and cannot hoist, so it must be
        // rejected under the flag.
        let rows = Dim::new(get_num_tile_blocks().0);
        let cols = Dim::new(get_num_tile_blocks().1);
        let tile_shape = shape![BM, BN];
        let mut out_part = out.partition_mut(tile_shape).with_bounds((rows, cols));
        for i in rows {
            for j in cols {
                let tile: Tile<f32, { [BM, BN] }> = constant(0.0, tile_shape);
                out_part.store(tile, coord((i, j)));
            }
        }
    }

    /// DENY-COVERAGE: the mapped sub-range family. Runtime range bounds cannot
    /// fold at compile time and are not launch-known, so the range assert is a
    /// residual in-kernel check — which the flag must reject, exactly as it
    /// rejects the access and binding families.
    #[cutile::entry(deny_in_kernel_checks = true)]
    unsafe fn subrange_check_denied<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        start_tile: i32,
        n_tiles: i32,
    ) {
        for index in z.iter_indices_within([(start_tile, n_tiles), (0i32, -1i32)]) {
            let tile: Tile<f32, { [BM, BN] }> = constant(0.0, shape![BM, BN]);
            z.store(tile, index);
        }
    }

    /// POSITIVE CONTROL: every check discharges at compile time, so the flag
    /// must ACCEPT this. Without a case like it, a bug that made the flag
    /// always reject would pass the whole suite.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn deny_accepts_fully_discharged<const N: i32, const BLOCK: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let tile_shape = shape![1, BLOCK];
        let mut p = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles(&p, 1) {
            let tile: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
            p.store(tile, [0i32, j]);
        }
    }

    /// POSITIVE CONTROL: nothing here discharges at compile time — the foreign
    /// access into `y` needs a fact no precondition declares. It compiles under
    /// the flag only because that fact is derived and checked on the host
    /// (measured: 3 launch checks), so this pins the flag and launch-hoisting
    /// working together. If the hoist regressed, this kernel stops building.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn deny_accepts_relocated_cross_tensor<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(shape![BM, BK]);
        let py = y.partition(shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0, shape![BM, BN]);
        for k in 0i32..num_tiles(&px, 1) {
            let tx = px.load([0i32, k]);
            let ty = py.load([k, 0i32]);
            acc = mma(tx, ty, acc);
        }
        z.store(acc);
    }

    /// Non-denied twin of `with_bounds_binding_check_denied`: two bindings that
    /// can fold nowhere and hoist nowhere (special-register-derived), so both
    /// need the runtime tile count -- and must share one index-space query.
    #[cutile::entry]
    fn with_bounds_binding_check_unchecked<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [-1, -1] }>,
    ) {
        let rows = Dim::new(get_num_tile_blocks().0);
        let cols = Dim::new(get_num_tile_blocks().1);
        let tile_shape = shape![BM, BN];
        let mut out_part = out.partition_mut(tile_shape).with_bounds((rows, cols));
        for i in rows {
            for j in cols {
                let tile: Tile<f32, { [BM, BN] }> = constant(0.0, tile_shape);
                out_part.store(tile, coord((i, j)));
            }
        }
    }
}

use opt_hints_module::__module_ast_self;

fn compile_kernel(name: &str, strides: &[(&str, &[i32])], options: &CompileOptions) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "opt_hints_module",
        name,
        &[128.to_string()],
        strides,
        &[],
        &[],
        None,
        options,
    )
    .expect("Failed to compile")
}

// The re-based row axis of a mutable partition is dynamic (extent set per-block
// at launch), so the constant `0` row coordinate cannot be discharged by the
// constant rung, which needs a static axis extent. Launch-time check hoisting
// resolves it: the safety condition reduces to `extent(out, 0) > 0`, a
// launch-known predicate, so the check is evacuated to a host `validate_launch`
// and NO in-kernel bounds assert is emitted — the store lowers with zero
// bounds-check registers.
#[test]
fn bounded_mut_store_lowers() {
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "bounded_mut_store_smoke",
            &["256".to_string(), "64".to_string()],
            &[("out", &[256, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("Failed to compile bounded_mut_store_smoke");
        println!("{mlir}");
        assert!(
            mlir.contains("store_view_tko"),
            "expected store_view_tko in MLIR:\n{mlir}"
        );
        // The row bounds check was hoisted to launch, not emitted in-kernel.
        assert!(
            !mlir.contains("out of bounds"),
            "row bounds check must be hoisted to launch, not emitted in-kernel:\n{mlir}"
        );
    });
}

/// Compiles a kernel from `opt_hints_module` all the way to a cubin and returns
/// its device register count via `cuobjdump --dump-resource-usage`. This is pure
/// compilation — `tileiras` + `cuobjdump`, no kernel execution — so it runs
/// anywhere those tools are present, no GPU launch required.
fn kernel_register_count(
    function_name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> u32 {
    use cutile::compile_api::KernelCompiler;
    let generics: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    let artifacts = KernelCompiler::new(__module_ast_self, "opt_hints_module", function_name)
        .target("sm_120")
        .generics(generics)
        .strides(strides)
        .compile()
        .expect("compile to Tile IR");
    // compile_tile_ir_module returns the cubin BYTES since the on-disk cache
    // landed (#193); cuobjdump wants a file, so write them to a temp path.
    let cubin = cutile_compiler::cuda_tile_runtime_utils::compile_tile_ir_module(
        artifacts.module(),
        "sm_120",
    )
    .expect("compile Tile IR to cubin");
    let cubin_path = std::env::temp_dir().join(format!(
        "cutile_regcount_{}_{}.cubin",
        function_name,
        std::process::id()
    ));
    std::fs::write(&cubin_path, &cubin).expect("write cubin for cuobjdump");
    let output = std::process::Command::new("cuobjdump")
        .arg("--dump-resource-usage")
        .arg(&cubin_path)
        .output()
        .expect("run cuobjdump");
    let _ = std::fs::remove_file(&cubin_path);
    let text = String::from_utf8_lossy(&output.stdout);
    // cuobjdump prints e.g. `REG:40 STACK:0 SHARED:0 ...` per function; take the
    // max across functions (the entry is the register-heaviest).
    text.split_whitespace()
        .filter_map(|tok| tok.strip_prefix("REG:").and_then(|n| n.parse::<u32>().ok()))
        .max()
        .unwrap_or_else(|| {
            panic!("no register count in cuobjdump output for {function_name}:\n{text}")
        })
}

// The safe bounded store hoists its bounds check to launch, so the safe kernel
// must reach the raw unchecked store's register count — the whole point of
// launch-time check hoisting. Asserts the invariant (safe <= raw), not a brittle
// golden number, so it survives future codegen changes.
#[test]
fn bounded_store_matches_raw_register_count() {
    // Requires the CUDA offline toolchain (tileiras + cuobjdump). Skip cleanly
    // where it is absent rather than fail (cf. LLVM lit `REQUIRES:`).
    if std::process::Command::new("cuobjdump")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping bounded_store_matches_raw_register_count: cuobjdump not available");
        return;
    }
    common::with_test_stack(|| {
        let generics = ["256", "64"];
        let strides: &[(&str, &[i32])] = &[("out", &[256, 1])];
        let safe = kernel_register_count("bounded_mut_store_smoke", &generics, strides);
        let raw = kernel_register_count("unchecked_mut_store_smoke", &generics, strides);
        eprintln!("register counts: safe={safe}, raw={raw}");
        assert!(
            safe <= raw,
            "hoisted safe store must not exceed the raw store's registers: safe={safe}, raw={raw}"
        );
    });
}

// A per-CTA row store indexed by a tile-block id must NOT be discharged: the
// launch grid counts CTA *slabs* (`div_ceil(root_shape, slab_shape)`, what
// `validate_grids` checks) while the access needs the tile count *within* the
// kernel-visible view. Those are unrelated, so `TileBlockId(0) < NumTileBlocks(0)`
// does not bound this access. A rung that discharged it (by proving the axiom
// against itself) admitted out-of-bounds stores and was reverted; until the real
// fix lands — form the actual proposition and verify `NumTileBlocks(axis) ==
// num_partitions(view, axis)` as a `Launch` obligation — this must be rejected.
#[test]
fn blockid_row_store_is_rejected() {
    common::with_test_stack(|| {
        let err = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "blockid_row_store",
            &["256".to_string(), "64".to_string()],
            &[("out", &[256, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect_err("a tile-block-id row index must not be discharged (unsound)");
        assert!(
            err.to_string()
                .contains("must come from iterating the matching dimension"),
            "unexpected block-id row error: {err}"
        );
    });
}

// Soundness: a block-id whose component does not match the indexed axis must
// NOT discharge — the gate is `k == axis`, so this fails to compile (rejected).
#[test]
fn blockid_wrong_axis_store_is_rejected() {
    common::with_test_stack(|| {
        let err = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "blockid_wrong_axis_store",
            &["256".to_string(), "64".to_string()],
            &[("out", &[256, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect_err("a block-id with a mismatched component must be rejected");
        assert!(
            err.to_string()
                .contains("must come from iterating the matching dimension"),
            "unexpected wrong-axis block-id error: {err}"
        );
    });
}

// The block-id owned-axis store must not cost registers vs the raw unchecked
// twin: the row check discharges at JIT (zero code), the column check hoists.

// The row bounds check the compiler hoisted to launch must actually reject a
// degenerate (zero-extent) tensor when evaluated at launch — soundness of the
// evacuated check. Uses the *real* emitted check (not a synthesized one) and is
// pure compilation (no launch).
//
// The row extent is declared `-1`, so this is a check with something to decide:
// no compile-time source knows the extent, and an empty tensor really would put
// the constant row-`0` store out of bounds.
#[test]
fn hoisted_check_rejects_zero_extent_at_launch() {
    common::with_test_stack(|| {
        use cutile::compile_api::KernelCompiler;
        use cutile::tile_kernel::validate_launch_checks;
        let artifacts = KernelCompiler::new(
            __module_ast_self,
            "opt_hints_module",
            "hoisted_dynamic_row_store",
        )
        .target("sm_120")
        .generics(vec!["256".to_string(), "64".to_string()])
        .strides(&[("out", &[256, 1])])
        .compile()
        .expect("compile hoisted_dynamic_row_store");
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "hoisted_dynamic_row_store",
            &["256".to_string(), "64".to_string()],
            &[("out", &[256, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("compile hoisted_dynamic_row_store");
        let asserts = mlir
            .lines()
            .filter(|l| l.trim_start().starts_with("assert "))
            .count();
        assert_eq!(asserts, 0, "everything should leave the kernel:\n{mlir}");
        let checks = artifacts.launch_checks();
        // Everything left the kernel: the row binding (`tiles == 1`, a range
        // pair) and the row-`0` access (non-empty), all in the view frame.
        assert!(
            !checks.is_empty(),
            "expected hoisted launch checks, got none"
        );
        assert!(
            format!("{checks:?}").contains("ViewExtent"),
            "a &mut param's extent checks must be in the view (slab) frame: {checks:?}"
        );
        // The frame matters: the root shape says 256 rows, but the checks are
        // against the kernel-visible slab. An empty slab must be rejected even
        // though the root is non-empty...
        let roots = [vec![256i32, 256]];
        assert!(
            validate_launch_checks(checks, &roots, &[vec![0i32, 256]], (1, 1, 1)).is_err(),
            "hoisted checks must reject a zero-extent view at launch"
        );
        // ...a one-row slab (what the kernel's `with_bounds` claims) accepted...
        assert!(
            validate_launch_checks(checks, &roots, &[vec![1i32, 256]], (1, 1, 1)).is_ok(),
            "hoisted checks must accept the declared one-row view"
        );
        // ...and a two-row slab rejected: the binding claimed exactly one tile.
        assert!(
            validate_launch_checks(checks, &roots, &[vec![2i32, 256]], (1, 1, 1)).is_err(),
            "hoisted checks must reject a view with more tiles than the binding declares"
        );
    });
}

// A signature that pins the extent must be believed. `bounded_mut_store_smoke`
// declares `out: &mut Tensor<f32, {[1, N]}>`, so `tiles(out, 0) == 1` and the
// constant row-`0` store is safe outright — there is nothing for the host to
// decide, and emitting a launch check would be checking a compile-time constant.
//
// The two checks reach that extent by different routes: the `with_bounds`
// binding reads the declared parameter shape, the access reads the partition
// view type — which erases a `&mut Tensor` param's shape to `?`. When only the
// binding consulted the signature, the access fell a rung and manufactured a
// launch check for an extent the signature had already fixed. Both now read the
// same source, so this asserts the disagreement stays fixed.
#[test]
fn a_pinned_extent_produces_no_launch_check() {
    common::with_test_stack(|| {
        use cutile::compile_api::KernelCompiler;
        let artifacts = KernelCompiler::new(
            __module_ast_self,
            "opt_hints_module",
            "bounded_mut_store_smoke",
        )
        .target("sm_120")
        .generics(vec!["256".to_string(), "64".to_string()])
        .strides(&[("out", &[256, 1])])
        .compile()
        .expect("compile bounded_mut_store_smoke");
        let checks = artifacts.launch_checks();
        assert!(
            checks.is_empty(),
            "a signature-pinned extent should discharge at compile time, but a launch \
             check was emitted: {checks:?}"
        );
    });
}

// `deny_in_kernel_checks`: a kernel whose safety check cannot be discharged at
// compile time or hoisted to launch (so it would remain in the kernel) is a hard
// JIT error under the flag, while the same kernel compiles without it. This is
// the safe-strict counterpart to `unchecked_accesses` (the unsafe waiver).
#[test]
fn deny_in_kernel_checks_rejects_a_residual_check() {
    common::with_test_stack(|| {
        // Without the flag: the residual check is emitted in-kernel; compiles.
        common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "residual_in_kernel_check",
            &["64".to_string(), "64".to_string()],
            &[("z", &[64, 1]), ("x", &[64, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("residual_in_kernel_check should compile (check stays in kernel)");
        // With deny_in_kernel_checks: the same residual check is a hard error
        // that names the offending axis.
        let err = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "residual_in_kernel_check_denied",
            &["64".to_string(), "64".to_string()],
            &[("z", &[64, 1]), ("x", &[64, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect_err("deny_in_kernel_checks must reject a residual in-kernel check");
        assert!(
            err.to_string().contains("deny_in_kernel_checks"),
            "expected a deny_in_kernel_checks diagnostic, got: {err}"
        );
    });
}

// Coverage of the sub-range family. Every compiler-synthesized assert routes
// through `deny_residual_check`, and the flag is only worth trusting if it
// covers all of them — a partial guarantee would let an author ship exactly
// the register cost it claims to forbid.
#[test]
fn deny_in_kernel_checks_covers_the_subrange_check() {
    common::with_test_stack(|| {
        let err = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "subrange_check_denied",
            &[
                "32".to_string(),
                "32".to_string(),
                "4".to_string(),
                "4".to_string(),
            ],
            &[("z", &[256, 1])],
            &[],
            &[],
            Some((16, 1, 1)),
            &CompileOptions::default(),
        )
        .expect_err("a runtime sub-range bound leaves a residual check the flag must reject");
        assert!(
            err.to_string().contains("deny_in_kernel_checks")
                && err.to_string().contains("sub-range"),
            "expected a sub-range deny diagnostic, got: {err}"
        );
    });
}

// The flag must ACCEPT a kernel whose checks all leave. Without this, a bug
// making it always reject would satisfy every other deny test here, since they
// all assert failure.
#[test]
fn deny_in_kernel_checks_accepts_a_fully_discharged_kernel() {
    common::with_test_stack(|| {
        common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "deny_accepts_fully_discharged",
            &["256".to_string(), "64".to_string()],
            &[("out", &[256, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("a kernel whose checks all discharge must compile under the flag");
    });
}

// The same acceptance, earned at launch rather than at compile time: the
// foreign access into `y` has no declaring precondition, so this compiles only
// because the fact it needs is derived and checked on the host.
#[test]
fn deny_in_kernel_checks_accepts_a_check_relocated_to_launch() {
    common::with_test_stack(|| {
        common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "deny_accepts_relocated_cross_tensor",
            &["32".to_string(), "32".to_string(), "32".to_string()],
            &[("z", &[32, 1]), ("x", &[32, 1]), ("y", &[32, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("a cross-tensor check that relocates to launch must satisfy the flag");
    });
}

// `deny_in_kernel_checks` promises that NO safety check remains in the kernel.
// That promise has to hold for every compiler-emitted check, not just the
// partition-access one: a residual `with_bounds` binding assert costs the same
// device registers and pins the same operands live. Accepting this kernel would
// make the attribute a partial guarantee, which is worse than none — a kernel
// author who trusts it would ship the very register cost it claims to forbid.
#[test]
fn deny_in_kernel_checks_covers_the_with_bounds_binding_check() {
    common::with_test_stack(|| {
        let err = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "with_bounds_binding_check_denied",
            &["64".to_string(), "64".to_string()],
            &[("out", &[64, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect_err(
            "deny_in_kernel_checks must reject a residual with_bounds binding check, \
             just as it rejects a residual partition-access check",
        );
        assert!(
            err.to_string().contains("deny_in_kernel_checks"),
            "expected a deny_in_kernel_checks diagnostic, got: {err}"
        );
    });
}

// A partition's index-space shape is one query answering every axis at once.
// Emitting it per axis is pure waste: a rank-n `with_bounds` whose bindings all
// need the runtime path would materialize n identical ops, each keeping its
// results live. One op, n results.
#[test]
fn with_bounds_queries_the_index_space_once_per_partition() {
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "with_bounds_binding_check_unchecked",
            &["64".to_string(), "64".to_string()],
            &[("out", &[64, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("Failed to compile");
        let queries = mlir.matches("get_index_space_shape").count();
        assert_eq!(
            queries, 1,
            "a rank-2 `with_bounds` should query the index space once, got {queries}:\n{mlir}"
        );
    });
}

#[test]
fn load_ptr_latency_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "load_ptr_latency_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 4"),
            "Expected latency=4 in load_ptr_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn load_ptr_const_latency_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "load_ptr_const_latency_kernel",
            &["128".to_string(), "5".to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("Failed to compile");
        assert!(
            mlir.contains("latency = 5"),
            "Expected latency=5 from Latency::<L> with L=5.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn store_ptr_latency_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "store_ptr_latency_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 2"),
            "Expected latency=2 in store_ptr_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn option_bindings_for_optional_operands_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "option_bindings_for_optional_operands_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("load_ptr_tko relaxed device"),
            "Expected Option-bound device scope on load_ptr_tko.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("store_ptr_tko weak") && mlir.contains("token=%"),
            "Expected Option-bound token operand on store_ptr_tko.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn load_view_latency_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "load_view_latency_kernel",
            &[("input", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 8"),
            "Expected latency=8 in load_view_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn store_view_disallow_tma_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "store_view_disallow_tma_kernel",
            &[("y", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("allow_tma = false"),
            "Expected allow_tma=false in store_view_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn entry_level_occupancy_hints_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "entry_hints_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("occupancy = 4"),
            "Expected occupancy=4 in entry optimization_hints.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("num_cta_in_cga = 2"),
            "Expected num_cta_in_cga=2 in entry optimization_hints.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("num_worker_warps_per_cta = 4"),
            "Expected num_worker_warps_per_cta=4 in entry optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn compile_options_override_entry_hints() {
    common::with_test_stack(|| {
        let options = CompileOptions::default()
            .occupancy(8)
            .num_cta_in_cga(4)
            .num_worker_warps_per_cta(8);
        let mlir = compile_kernel("entry_hints_kernel", &[("output", &[1])], &options);
        println!("{mlir}");
        assert!(
            mlir.contains("occupancy = 8"),
            "Expected runtime occupancy=8 to override entry-level occupancy=4.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("num_cta_in_cga = 4"),
            "Expected runtime num_cta_in_cga=4 to override entry-level num_cta_in_cga=2.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("num_worker_warps_per_cta = 8"),
            "Expected runtime num_worker_warps_per_cta=8 to override entry-level value 4.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn worker_warps_values_are_forwarded_to_backend() {
    common::with_test_stack(|| {
        let entry_mlir = compile_kernel(
            "worker_warps_value_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        assert!(entry_mlir.contains("num_worker_warps_per_cta = 1"));

        let options_mlir = compile_kernel(
            "entry_hints_kernel",
            &[("output", &[1])],
            &CompileOptions::default().num_worker_warps_per_cta(32),
        );
        assert!(options_mlir.contains("num_worker_warps_per_cta = 32"));
    });
}

#[test]
fn different_compile_options_produce_different_mlir() {
    common::with_test_stack(|| {
        let mlir_a = compile_kernel(
            "entry_hints_kernel",
            &[("output", &[1])],
            &CompileOptions::default().occupancy(2),
        );
        let mlir_b = compile_kernel(
            "entry_hints_kernel",
            &[("output", &[1])],
            &CompileOptions::default().occupancy(16),
        );
        assert!(
            mlir_a.contains("occupancy = 2"),
            "First compilation should have occupancy=2.\nMLIR:\n{mlir_a}"
        );
        assert!(
            mlir_b.contains("occupancy = 16"),
            "Second compilation should have occupancy=16.\nMLIR:\n{mlir_b}"
        );
        assert_ne!(
            mlir_a, mlir_b,
            "Different CompileOptions should produce different MLIR"
        );
    });
}

#[test]
fn load_view_const_latency_in_mlir() {
    // Latency as a const generic: L=5 should appear as `latency = 5` in MLIR.
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "load_view_const_latency_kernel",
            &[128.to_string(), 5.to_string()], // S=128, L=5
            &[("input", &[1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("Failed to compile");
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 5"),
            "Expected latency=5 from const generic L=5.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn safe_load_pipelined_latency_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "safe_load_pipelined_kernel",
            &[("z", &[1]), ("x", &[1])],
            &CompileOptions::default(),
        );
        assert!(
            mlir.contains("latency = 6"),
            "Expected latency=6 from Partition::load_pipelined.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn safe_load_pipelined_const_latency_in_mlir() {
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "safe_load_pipelined_const_kernel",
            &[128.to_string(), 5.to_string()], // S=128, L=5
            &[("z", &[1]), ("x", &[1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("Failed to compile");
        assert!(
            mlir.contains("latency = 5"),
            "Expected latency=5 from load_pipelined::<L> with L=5.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn safe_bounded_load_pipelined_latency_in_mlir() {
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "safe_bounded_load_pipelined_kernel",
            &[16.to_string(), 16.to_string()], // BM=16, BN=16
            &[("z", &[16, 1]), ("x", &[128, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("Failed to compile");
        assert!(
            mlir.contains("latency = 7"),
            "Expected latency=7 from BoundedPartition::load_pipelined.\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("partition access out of bounds"),
            "Bounded pipelined loads must keep the discharge proof.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn two_pipelined_loads_keep_distinct_latencies_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "safe_load_two_latencies_kernel",
            &[("z", &[1]), ("x", &[1]), ("y", &[1])],
            &CompileOptions::default(),
        );
        assert!(
            mlir.contains("latency = 3") && mlir.contains("latency = 9"),
            "each load must keep its own latency hint (not last-wins).\nMLIR:\n{mlir}"
        );
    });
}

// SOUNDNESS SPEC (currently failing — documents a live hole in the safe surface).
//
// `with_bounds` records which `Dim` brands which axis but never verifies the
// `Dim`'s *value* against the view's real tile count, so a bogus bound is
// silently trusted and the branded store discharges. This is reachable from
// safe Rust (both `Tensor::partition` and `partition_mut` are safe), so safe
// code can produce out-of-bounds accesses. `LAUNCH_PRECONDITION_HOISTING.md`
// names this ("an unvalidated host-contract") and proposes synthesizing a
// precondition from `with_bounds`; that synthesis is not implemented.
//
// The fix: emit `declared_dim <= num_partitions(view, axis)` as a verified
// `Launch` obligation. Then this kernel either fails at JIT or is caught at
// launch. Un-ignore when it lands.
#[test]
fn with_bounds_bogus_dim_must_not_silently_discharge() {
    common::with_test_stack(|| {
        let result = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "with_bounds_bogus_dim",
            &["256".to_string(), "64".to_string()],
            &[("out", &[256, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        );
        match result {
            Err(_) => {} // rejected at JIT: acceptable.
            Ok(mlir) => assert!(
                mlir.contains("does not equal this partition's tile count")
                    || mlir.contains("out of bounds"),
                "a bogus with_bounds Dim discharged with no check and no launch \
                 obligation — safe code can write out of bounds:\n{mlir}"
            ),
        }
    });
}

// SOUNDNESS SPEC (read side): a bogus declared bound must be *caught*, at the
// earliest stage able to decide it. On `main` this kernel silently discharged —
// a safe-code out-of-bounds READ. The first fix trapped it with a device
// assert. The constant launch rung now decides it before the kernel exists:
// `Dim::new(999)` on a dynamic immutable axis becomes the host range check
// `998·BM < dim(x, 0) ≤ 999·BM`, which every real launch of this kernel fails —
// strictly earlier detection than the device trap, zero registers.
#[test]
fn with_bounds_bogus_dim_load_must_not_silently_discharge() {
    common::with_test_stack(|| {
        use cutile::compile_api::KernelCompiler;
        use cutile::tile_kernel::validate_launch_checks;
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "with_bounds_bogus_dim_load",
            &["16".to_string(), "64".to_string()],
            &[("z", &[64, 1]), ("x", &[64, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("compiles; the bogus bound is deferred to a launch check");
        let asserts = mlir
            .lines()
            .filter(|l| l.trim_start().starts_with("assert "))
            .count();
        assert_eq!(
            asserts, 0,
            "the bogus bound should hoist, not trap in-kernel:\n{mlir}"
        );

        let artifacts = KernelCompiler::new(
            __module_ast_self,
            "opt_hints_module",
            "with_bounds_bogus_dim_load",
        )
        .target("sm_120")
        .generics(vec!["16".to_string(), "64".to_string()])
        .strides(&[("z", &[64, 1]), ("x", &[64, 1])])
        .compile()
        .expect("compile with_bounds_bogus_dim_load");
        let checks = artifacts.launch_checks();
        assert!(
            !checks.is_empty(),
            "the bogus bound must leave a launch obligation behind: {checks:?}"
        );
        // Any plausible tensor fails the 999-tile range check at launch.
        assert!(
            validate_launch_checks(
                checks,
                &[vec![16, 64], vec![128, 128]],
                &[vec![16, 64], vec![128, 128]],
                (1, 1, 1)
            )
            .is_err(),
            "a 128-row tensor must be rejected against a declared 999-tile bound: {checks:?}"
        );
        // The declared bound is satisfiable by exactly one extent range: with
        // BM = 16, rows in (15968, 15984] — and 64 must divide the columns.
        assert!(
            validate_launch_checks(
                checks,
                &[vec![16, 64], vec![15984, 128]],
                &[vec![16, 64], vec![15984, 128]],
                (1, 1, 1)
            )
            .is_ok(),
            "the one extent range that satisfies the declared bound must be accepted: {checks:?}"
        );
    });
}

// The binding's Launch rung. `safe_bounded_load_pipelined_kernel` binds
// `Dim::new(x.shape()[k] / B)` to a partition of `x` tiled by `B`. That bound is
// `floor(e/B)` while the partition has `ceil(e/B)` tiles, so the binding holds
// exactly when `B` divides `e` -- a predicate over the extent atom itself. Both
// operands are launch-known, so it must leave the kernel entirely and be decided
// once by the host, including rejecting an extent that violates it.
//
// Stating it as divisibility rather than naming the division as an opaque atom
// is what keeps it discharge-able: the goal mentions only `Dim`, the same
// vocabulary preconditions are written in.
#[test]
fn a_divisible_binding_leaves_the_kernel_for_the_host() {
    common::with_test_stack(|| {
        use cutile::compile_api::KernelCompiler;
        use cutile::tile_kernel::validate_launch_checks;
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "safe_bounded_load_pipelined_kernel",
            &["64".to_string(), "64".to_string()],
            &[("z", &[64, 1]), ("x", &[64, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("Failed to compile");
        let asserts = mlir
            .lines()
            .filter(|l| l.trim_start().starts_with("assert "))
            .count();
        assert_eq!(asserts, 0, "both bindings should leave the kernel:\n{mlir}");

        let artifacts = KernelCompiler::new(
            __module_ast_self,
            "opt_hints_module",
            "safe_bounded_load_pipelined_kernel",
        )
        .target("sm_120")
        .generics(vec!["64".to_string(), "64".to_string()])
        .strides(&[("z", &[64, 1]), ("x", &[64, 1])])
        .compile()
        .expect("compile safe_bounded_load_pipelined_kernel");
        let checks = artifacts.launch_checks();
        assert_eq!(
            checks.len(),
            2,
            "expected one host check per axis: {checks:?}"
        );
        // x is 128x128: 64 divides both extents.
        assert!(
            validate_launch_checks(
                checks,
                &[vec![64i32, 64], vec![128i32, 128]],
                &[vec![64i32, 64], vec![128i32, 128]],
                (1, 1, 1)
            )
            .is_ok(),
            "a divisible shape must be accepted: {checks:?}"
        );
        // x is 100x128: 64 does not divide 100, so `floor` undercounts the tiles
        // and an index from the loop could reach past the last full tile.
        assert!(
            validate_launch_checks(
                checks,
                &[vec![64i32, 64], vec![100i32, 128]],
                &[vec![64i32, 64], vec![100i32, 128]],
                (1, 1, 1)
            )
            .is_err(),
            "a non-divisible extent must be rejected at launch: {checks:?}"
        );
    });
}

// The two-statement `with_bounds` spelling (bind the partition, then bound it)
// is the natural way to write the pattern and once died in typeck at the
// subsequent store. It must compile, and with a declared [1, N] shape both
// bindings must fold: zero in-kernel asserts.
#[test]
fn two_statement_with_bounds_compiles_and_folds() {
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "two_stmt_with_bounds",
            &["256".to_string(), "64".to_string()],
            &[("out", &[256, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("the two-statement with_bounds form must compile");
        let asserts = mlir
            .lines()
            .filter(|l| l.trim_start().starts_with("assert "))
            .count();
        assert_eq!(asserts, 0, "both bindings should fold at JIT:\n{mlir}");
    });
}

// Theorem 2, end to end: a shared `Dim` bound across two tensors. With the
// extent equality declared (and so verified by the launcher), the cross-tensor
// binding discharges from the declared fact plus the numerator's divisibility
// check -- the same host compare the sibling binding emits, deduplicated. The
// undeclared twin must keep its device assert: hoisting an unproven extent
// equality would be stricter than the binding's real obligation.
#[test]
fn a_shared_dim_binding_discharges_from_the_declared_equality() {
    common::with_test_stack(|| {
        use cutile::compile_api::KernelCompiler;
        use cutile::tile_kernel::validate_launch_checks;
        let count_asserts = |name: &str| {
            common::compile_to_ir(
                __module_ast_self,
                "opt_hints_module",
                name,
                &["32".to_string(), "64".to_string()],
                &[("z", &[64, 1]), ("x", &[128, 1]), ("y", &[128, 1])],
                &[],
                &[],
                None,
                &CompileOptions::default(),
            )
            .map(|m| {
                m.lines()
                    .filter(|l| l.trim_start().starts_with("assert "))
                    .count()
            })
            .expect("compile")
        };
        assert_eq!(
            count_asserts("shared_dim_binding"),
            0,
            "with the equality declared, every binding should leave the kernel"
        );
        assert_eq!(
            count_asserts("shared_dim_binding_undeclared"),
            1,
            "without the declared equality the cross-tensor binding must stay a device assert"
        );

        let artifacts =
            KernelCompiler::new(__module_ast_self, "opt_hints_module", "shared_dim_binding")
                .target("sm_120")
                .generics(vec!["32".to_string(), "64".to_string()])
                .strides(&[("z", &[64, 1]), ("x", &[128, 1]), ("y", &[128, 1])])
                .compile()
                .expect("compile shared_dim_binding");
        let checks = artifacts.launch_checks();
        // m|BM on x0, k|BK on x1 (shared by both bindings, deduplicated),
        // n|BM on y1: exactly three host compares.
        assert_eq!(
            checks.len(),
            3,
            "expected three deduplicated checks: {checks:?}"
        );
        // x = [128, 128], y = [128, 128]: 32 | 128 and 64 | 128.
        assert!(
            validate_launch_checks(
                checks,
                &[vec![32, 64], vec![128, 128], vec![128, 128]],
                &[vec![32, 64], vec![128, 128], vec![128, 128]],
                (1, 1, 1)
            )
            .is_ok(),
            "divisible extents must be accepted: {checks:?}"
        );
        // x = [128, 100]: 64 does not divide the shared contraction extent.
        assert!(
            validate_launch_checks(
                checks,
                &[vec![32, 64], vec![128, 100], vec![128, 128]],
                &[vec![32, 64], vec![128, 100], vec![128, 128]],
                (1, 1, 1)
            )
            .is_err(),
            "a non-divisible shared extent must be rejected at launch: {checks:?}"
        );
    });
}

// The full trust chain for a declared divisibility fact, in one place:
//
//   declared  ->  entailed at JIT (no assert, no launch check)
//             ->  verified by the generated launcher (bad shape rejected)
//
// The first half is what the reduction bought: the binding obligation
// `divisible_by(dim(x, k), 64)` is exact set membership against the declared
// fact, so it resolves at `Jit` and nothing is emitted anywhere. The second
// half is why believing the declaration is sound: the launcher rejects any
// launch whose shapes violate it, before the kernel exists.
#[test]
fn a_declared_divisibility_discharges_at_jit() {
    common::with_test_stack(|| {
        use cutile::compile_api::KernelCompiler;
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "opt_hints_module",
            "declared_divisibility_binding",
            &["64".to_string(), "64".to_string()],
            &[("z", &[64, 1]), ("x", &[128, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("compile declared_divisibility_binding");
        let asserts = mlir
            .lines()
            .filter(|l| l.trim_start().starts_with("assert "))
            .count();
        assert_eq!(
            asserts, 0,
            "declared facts should entail both bindings:\n{mlir}"
        );
        let artifacts = KernelCompiler::new(
            __module_ast_self,
            "opt_hints_module",
            "declared_divisibility_binding",
        )
        .target("sm_120")
        .generics(vec!["64".to_string(), "64".to_string()])
        .strides(&[("z", &[64, 1]), ("x", &[128, 1])])
        .compile()
        .expect("compile declared_divisibility_binding");
        assert!(
            artifacts.launch_checks().is_empty(),
            "an entailed obligation must not also emit a launch check: {:?}",
            artifacts.launch_checks()
        );
    });
}
