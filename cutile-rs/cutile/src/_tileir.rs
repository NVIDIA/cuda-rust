/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// Raw 1:1 Tile IR op surface.
///
/// Every item here mirrors an op in the Tile IR specification
/// (<https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html>)
/// under its `Ops.td` name, declared `unsafe` wholesale: the contract is
/// bit-for-bit Tile IR semantics with the caller owning every
/// precondition. New Tile IR features land here first, mechanically;
/// their safe/idiomatic exposure in [`crate::core`] follows separately.
/// See `.internal/tasks/in_progress/dsl_improvements/op_cleanup_v0.0.2/OP_CLEANUP.md`
/// Phase 2.
#[cutile_macro::module(tile_rust_crate = true)]
pub mod tileir {
    use crate::_core::core::*;

    /// View-based atomic reduction: element-wise
    /// `view[index] := mode(view[index], value)` without returning old
    /// values (use `atomic_rmw_tko` when old values are needed). `mode`
    /// selects the op (`atomic::*`): integer modes for `i32`/`i64`, `AddF`
    /// for `f16`/`bf16`/`f32`/`f64`. `memory_ordering` must be `Relaxed`;
    /// `memory_scope` ⊆ {TileBlock, Device}. Returns the completion token.
    // TODO (hme): document safety
    #[cuda_tile::op(name="cuda_tile.atomic_red_view_tko", since="V13_3", params=["view", "value", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn atomic_red_view_tko<
        E: ElementType,
        const D: [i32; N],
        M: atomic::Mode,
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        view: &mut PartitionMut<E, D>,
        value: Tile<E, D>,
        index: [i32; N],
        mode: M,
        memory_ordering: O,
        memory_scope: Sc,
    ) -> Token {
        unreachable!()
    }

    // ── GatherScatterView ────────────────────────────────────────────────────

    /// Sparse-indexed view into a tensor.
    ///
    /// One dimension (`SPARSE_DIM`) uses a 1D tile of integer indices for
    /// gather/scatter; all other dimensions are traversed densely.
    /// Created by [`make_gather_scatter_view`]; loaded with
    /// [`load_gather_scatter_view_tko`].
    // TODO (hme): document safety
    #[cuda_tile::ty(name="!cuda_tile.gather_scatter_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct GatherScatterView<
        'a,
        E: ElementType,
        const TILE_SHAPE: [i32; N],
        const SPARSE_DIM: i32,
    > {
        _marker: ::core::marker::PhantomData<(&'a (), E)>,
    }

    /// Build a `GatherScatterView` from a tensor.
    ///
    /// `_tile` encodes the output tile shape (a `Shape<TILE_SHAPE>` witness).
    /// `SPARSE_DIM` selects which tensor dimension uses sparse indexing.
    /// `_padding` selects out-of-bounds fill; nonzero padding modes require
    /// floating-point elements. Requires Tile IR 13.3.
    ///
    /// # Safety
    /// The tensor must describe valid live storage for every subsequent
    /// unmasked access. Consumers must respect access bounds, aliasing and
    /// token ordering; constructing a view does not establish ownership.
    #[cuda_tile::op(name = "cuda_tile.make_gather_scatter_view", since = "V13_3")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_gather_scatter_view<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        const SPARSE_DIM: i32,
        P: padding::Mode,
    >(
        tensor_view: &'a Tensor<E, TENSOR_SHAPE>,
        _tile: Shape<TILE_SHAPE>,
        _padding: P,
    ) -> GatherScatterView<'a, E, TILE_SHAPE, SPARSE_DIM> {
        unreachable!()
    }

    /// Load a tile from a `GatherScatterView`.
    ///
    /// `sparse_index`: 1-D tile of integer indices into the sparse dimension.
    /// `dense_index`: scalar index for the dense dimension.
    /// Returns `(Tile<E, TILE_SHAPE>, Token)` with a fresh completion token.
    /// This convenience form is two-dimensional; use `load_view_raw` for
    /// other ranks, explicit hints, or optional input tokens.
    ///
    /// # Safety
    /// Every unpadded address must be valid and readable. The input token
    /// must order conflicting prior accesses; sparse indices must not
    /// introduce a data race with concurrent writes.
    #[cuda_tile::op(name = "cuda_tile.load_gather_scatter_view_tko", since = "V13_3")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn load_gather_scatter_view_tko<
        E: ElementType,
        const TILE_SHAPE: [i32; N],
        const SPARSE_DIM: i32,
        const SPARSE_IDX_SIZE: i32,
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        view: &GatherScatterView<'_, E, TILE_SHAPE, SPARSE_DIM>,
        sparse_index: Tile<i32, { [SPARSE_IDX_SIZE] }>,
        dense_index: i32,
        token: Token,
        memory_ordering: O,
        memory_scope: Sc,
    ) -> (Tile<E, TILE_SHAPE>, Token) {
        unreachable!()
    }

    // ── StridedView ──────────────────────────────────────────────────────────

    /// Strided view into a tensor with configurable per-dimension traversal.
    ///
    /// `TRAVERSAL_STRIDES` controls how far to step per tile along each
    /// dimension; `DIM_MAP` maps tile axes to tensor axes. Created by
    /// [`make_strided_view`]; loaded with [`load_strided_view_tko`].
    // TODO (hme): document safety
    #[cuda_tile::ty(name="!cuda_tile.strided_view",
                    type_params=["tile"],
                    type_params_optional=["traversal_strides", "dim_map", "padding_value"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct StridedView<
        'a,
        E: ElementType,
        const TILE_SHAPE: [i32; N],
        const TRAVERSAL_STRIDES: [i32; N],
        const DIM_MAP: [i32; N],
    > {
        _marker: ::core::marker::PhantomData<(&'a (), E)>,
    }

    /// Build a `StridedView` from a tensor.
    ///
    /// `TRAVERSAL_STRIDES` and `DIM_MAP` are encoded as const generic
    /// parameters on the return type; pass them via a turbofish or return-type
    /// annotation. `_tile` is a shape witness for `TILE_SHAPE`.
    ///
    /// # Safety
    /// The tensor must describe valid live storage. Consumers must uphold
    /// bounds and token ordering, including overlaps between traversals.
    #[cuda_tile::op(name = "cuda_tile.make_strided_view", since = "V13_3")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_strided_view<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        const TRAVERSAL_STRIDES: [i32; N],
        const DIM_MAP: [i32; N],
        P: padding::Mode,
    >(
        tensor_view: &'a Tensor<E, TENSOR_SHAPE>,
        _tile: Shape<TILE_SHAPE>,
        _padding: P,
    ) -> StridedView<'a, E, TILE_SHAPE, TRAVERSAL_STRIDES, DIM_MAP> {
        unreachable!()
    }

    /// Load a tile from a `StridedView`.
    ///
    /// `index` is an N-element array of scalar indices, one per tile
    /// dimension. Returns `(Tile<E, TILE_SHAPE>, Token)`.
    ///
    /// # Safety
    /// Every unpadded address must be valid and readable, and the input
    /// token must order conflicting prior accesses to the same storage.
    #[cuda_tile::op(name = "cuda_tile.load_strided_view_tko", since = "V13_3")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn load_strided_view_tko<
        E: ElementType,
        const TILE_SHAPE: [i32; N],
        const TRAVERSAL_STRIDES: [i32; N],
        const DIM_MAP: [i32; N],
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        view: &StridedView<'_, E, TILE_SHAPE, TRAVERSAL_STRIDES, DIM_MAP>,
        index: [i32; N],
        token: Token,
        memory_ordering: O,
        memory_scope: Sc,
    ) -> (Tile<E, TILE_SHAPE>, Token) {
        unreachable!()
    }
}
