/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scatter/reduce kernels over the typed aggregation capabilities.
//!
//! Each kernel writes through an aggregation parameter (`AddTensor`,
//! `MaxTensor`, or `OrTensor`), so overlapping destinations are the intended
//! case: repeated indices, duplicate bins, and shared segments compose
//! atomically. Kernels are generic over the element where the added value
//! comes from an input tensor; the element's sealed trait bounds the set
//! (`AtomicAdd` admits the floats and 32/64-bit integers, `AtomicOrd` and
//! `AtomicBitwise` the 32/64-bit integers), and for the comparisons the
//! element also selects the signed or unsigned Tile IR mode. Common
//! contract, stated once:
//!
//! - **Indices** are `u32` row/bin ids loaded from memory and must lie in
//!   range for the destination; each access is bounds-checked on device.
//! - **Initialization** is the caller's: seed the destination with the
//!   operation's identity for a pure reduction (`0`/`0.0` for add and OR,
//!   the signedness-correct extreme for max), or any other seed for seeded
//!   accumulation.
//! - **Aliasing**: the destination may not also be bound as an ordinary
//!   input or output of the same launch, nor as a different aggregation
//!   operation; the launcher rejects both before launch.
//! - **Float order**: atomic float addition fixes no order, so float results
//!   may differ between runs; integer kernels are exact.
//!
//! Unsupported elements fail at the type level; there is no float maximum
//! and no atomic reduction for 8- or 16-bit integers:
//!
//! ```compile_fail
//! #[cutile::module]
//! mod rejected {
//!     use cutile::core::*;
//!     #[cutile::entry()]
//!     fn max_f32(out: &MaxTensor<f32, { [-1] }>) {}
//! }
//! ```

#[cutile::module]
pub mod scatter_add_module {
    use cutile::core::*;

    /// `out[indices[i]] += values[i]` for every element `i`, atomically.
    /// Repeated indices accumulate. Launch with grid `(values.len(), 1, 1)`.
    #[cutile::entry()]
    fn scatter_add<E: atomic::AtomicAdd>(
        indices: &Tensor<u32, { [-1] }>,
        values: &Tensor<E, { [-1] }>,
        out: &AddTensor<E, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let id_tile: Tile<u32, { [1] }> = indices.partition(const_shape![1]).load([pid.0]);
        let idc: Tile<i32, { [1] }> = bitcast(id_tile);
        let idx: i32 = tile_to_scalar(idc.reshape(const_shape![]));
        let v: Tile<E, { [1] }> = values.partition(const_shape![1]).load([pid.0]);
        let acc: AddPartition<E, { [1] }> = out.partition(const_shape![1]);
        let _t: Token = acc.atomic_add([idx], v);
    }
}

#[cutile::module]
pub mod embedding_backward_scatter_add_module {
    use cutile::core::*;

    /// Accumulates row gradients into an embedding-table gradient:
    /// `table_grad[token_ids[i], :] += grads[i, :]`, atomically, so repeated
    /// token ids compose. `D` is the embedding width; grid `(rows, 1, 1)`.
    #[cutile::entry()]
    fn embedding_backward_scatter_add<E: atomic::AtomicAdd, const D: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        grads: &Tensor<E, { [-1, D] }>,
        table_grad: &AddTensor<E, { [-1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let id_tile: Tile<u32, { [1] }> = token_ids.partition(const_shape![1]).load([pid.0]);
        let idc: Tile<i32, { [1] }> = bitcast(id_tile);
        let row: i32 = tile_to_scalar(idc.reshape(const_shape![]));
        let g: Tile<E, { [1, D] }> = grads.partition(const_shape![1, D]).load([pid.0, 0i32]);
        let acc: AddPartition<E, { [1, D] }> = table_grad.partition(const_shape![1, D]);
        let _t: Token = acc.atomic_add([row, 0i32], g);
    }
}

#[cutile::module]
pub mod segment_sum_module {
    use cutile::core::*;

    /// `out[segment_ids[i]] += values[i]`, atomically. An empty segment keeps
    /// its seed. Launch with grid `(values.len(), 1, 1)`.
    #[cutile::entry()]
    fn segment_sum<E: atomic::AtomicAdd>(
        segment_ids: &Tensor<u32, { [-1] }>,
        values: &Tensor<E, { [-1] }>,
        out: &AddTensor<E, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seg_tile: Tile<u32, { [1] }> = segment_ids.partition(const_shape![1]).load([pid.0]);
        let segc: Tile<i32, { [1] }> = bitcast(seg_tile);
        let seg: i32 = tile_to_scalar(segc.reshape(const_shape![]));
        let v: Tile<E, { [1] }> = values.partition(const_shape![1]).load([pid.0]);
        let acc: AddPartition<E, { [1] }> = out.partition(const_shape![1]);
        let _t: Token = acc.atomic_add([seg], v);
    }
}

#[cutile::module]
pub mod histogram_module {
    use cutile::core::*;

    /// `bins[bin_ids[i]] += 1` for every element, atomically. The increment
    /// is `E::ONE` materialized on-device, so no weights buffer is read;
    /// `weighted_histogram` is the variant that reads a weight per element.
    /// Launch with grid `(bin_ids.len(), 1, 1)`.
    #[cutile::entry()]
    fn histogram<E: atomic::AtomicAdd>(
        bin_ids: &Tensor<u32, { [-1] }>,
        bins: &AddTensor<E, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bin_tile: Tile<u32, { [1] }> = bin_ids.partition(const_shape![1]).load([pid.0]);
        let binc: Tile<i32, { [1] }> = bitcast(bin_tile);
        let bin: i32 = tile_to_scalar(binc.reshape(const_shape![]));
        let one: Tile<E, { [1] }> = constant(E::ONE, const_shape![1]);
        let acc: AddPartition<E, { [1] }> = bins.partition(const_shape![1]);
        let _t: Token = acc.atomic_add([bin], one);
    }
}

#[cutile::module]
pub mod weighted_histogram_module {
    use cutile::core::*;

    /// `bins[bin_ids[i]] += weights[i]`, atomically. The element-generic
    /// histogram. Launch with grid `(bin_ids.len(), 1, 1)`.
    #[cutile::entry()]
    fn weighted_histogram<E: atomic::AtomicAdd>(
        bin_ids: &Tensor<u32, { [-1] }>,
        weights: &Tensor<E, { [-1] }>,
        bins: &AddTensor<E, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bin_tile: Tile<u32, { [1] }> = bin_ids.partition(const_shape![1]).load([pid.0]);
        let binc: Tile<i32, { [1] }> = bitcast(bin_tile);
        let bin: i32 = tile_to_scalar(binc.reshape(const_shape![]));
        let w: Tile<E, { [1] }> = weights.partition(const_shape![1]).load([pid.0]);
        let acc: AddPartition<E, { [1] }> = bins.partition(const_shape![1]);
        let _t: Token = acc.atomic_add([bin], w);
    }
}

#[cutile::module]
pub mod max_reduce_module {
    use cutile::core::*;

    /// Folds each row of `B` values into `out[0]` by atomic maximum. The
    /// element selects the comparison for both the intra-tile reduction and
    /// the atomic: signed `Max` for `i32`/`i64`, unsigned `Umax` for
    /// `u32`/`u64`. Seed `out[0]` with the signedness-correct minimum
    /// (`i32::MIN`, or `0` for unsigned) for a pure maximum.
    ///
    /// `values` is viewed as `[rows, B]`; `B` is inferred at launch. Grid
    /// `(rows, 1, 1)`.
    #[cutile::entry()]
    fn max_reduce<E: atomic::AtomicOrd, const B: i32>(
        values: &Tensor<E, { [-1, B] }>,
        out: &MaxTensor<E, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row: Tile<E, { [1, B] }> = values.partition(const_shape![1, B]).load([pid.0, 0i32]);
        let m: Tile<E, { [1] }> = reduce_max(row, 1i32);
        let acc: MaxPartition<E, { [1] }> = out.partition(const_shape![1]);
        let _t: Token = acc.atomic_max([0i32], m);
    }
}

#[cutile::module]
pub mod bitmask_or_module {
    use cutile::core::*;

    /// `out[dest_ids[i]] |= masks[i]`, atomically. OR is idempotent, so
    /// repeated identical masks are harmless. Launch with grid
    /// `(masks.len(), 1, 1)`.
    #[cutile::entry()]
    fn bitmask_or<E: atomic::AtomicBitwise>(
        dest_ids: &Tensor<u32, { [-1] }>,
        masks: &Tensor<E, { [-1] }>,
        out: &OrTensor<E, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let id_tile: Tile<u32, { [1] }> = dest_ids.partition(const_shape![1]).load([pid.0]);
        let idc: Tile<i32, { [1] }> = bitcast(id_tile);
        let dest: i32 = tile_to_scalar(idc.reshape(const_shape![]));
        let mask: Tile<E, { [1] }> = masks.partition(const_shape![1]).load([pid.0]);
        let acc: OrPartition<E, { [1] }> = out.partition(const_shape![1]);
        let _t: Token = acc.atomic_or([dest], mask);
    }
}

pub use bitmask_or_module::bitmask_or;
pub use embedding_backward_scatter_add_module::embedding_backward_scatter_add;
pub use histogram_module::histogram;
pub use max_reduce_module::max_reduce;
pub use scatter_add_module::scatter_add;
pub use segment_sum_module::segment_sum;
pub use weighted_histogram_module::weighted_histogram;
