/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVFP4 mixture of experts kernels by Emmet Zhang et al.
 * Adapted from their NVFP4 MoE implementation; the device kernel code is unchanged.
 *
 * Run with: cargo run --release -p cutile-examples --example nvfp4_moe
 *
 * Pipeline: device routing -> input quantization -> FC1 + ReLU-squared +
 * requantization -> FC2 -> route-weighted BF16 output. Both routing kernels
 * are exercised across power-of-two batch sizes from 1 through 256. CUDA
 * events measure graph replay latency after warmup, with output validation
 * outside the timed region.
 *
 * The small synthetic fixture uses H=I=1024, eight allocated experts, top-k=4,
 * and four active experts. Constant packed weights and deterministic routes
 * admit an analytical reference. Performance will be optimized in a future
 * version.
 */

#![allow(unused_unsafe)]

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cuda_core::{Context, Stream};
use cutile::api;
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    pub const ROUTING_PAIRWISE_CAPACITY: i32 = 16;

    // Small batches: all routes fit in one 16-entry tile, but an expert may need
    // multiple TILE_N groups.
    // Pairwise routing pipeline:
    //   1. Compare all route pairs to compute each route's expert-local count and rank.
    //   2. Mark the first route for each expert and round its count up to TILE_N.
    //   3. Compute each expert's output offset from the padded counts in expert-ID order.
    //   4. Scatter token indices and reverse mappings using output_offset + local_rank.
    //   5. Emit expert and valid-lane metadata at each TILE_N group boundary.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn routing_group_pairwise_kernel(
        group_tokens: &Tensor<i32, { [-1, -1] }>, // [max_groups, TN]
        group_experts: &Tensor<i32, { [-1] }>,    // [max_groups]
        group_limits: &Tensor<i32, { [-1] }>,     // [max_groups]
        expanded_to_permuted: &Tensor<i32, { [-1, -1] }>, // [num_tokens, top_k]
        num_groups: &Tensor<i32, { [-1] }>,       // [1]
        expert_ids: &Tensor<i32, { [-1, -1] }>,   // [num_tokens, top_k]
        top_k: i32,
        routed_size: i32, // routed_size = num_tokens * top_k
        tile_n: i32,
        tile_n_log2: i32,
    ) {
        let routed_indices: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            iota(const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let routed_limit: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            broadcast_scalar(routed_size, const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let routed_mask: Tile<bool, { [ROUTING_PAIRWISE_CAPACITY] }> =
            cmpi(routed_indices, routed_limit, predicate::LessThan);
        let token_indices =
            routed_indices / broadcast_scalar(top_k, const_shape![ROUTING_PAIRWISE_CAPACITY]);

        // Load expert_ids mapping each (token, top-k slot) to an expert.
        let expert_ids_base: PointerTile<*const i32, { [] }> = pointer_to_tile(expert_ids.as_ptr());
        let expert_ids_1d: PointerTile<*const i32, { [1] }> =
            expert_ids_base.reshape(const_shape![1]);
        let expert_ids_broadcast: PointerTile<*const i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            expert_ids_1d.broadcast(const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let expert_ids_ptrs: PointerTile<*const i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            expert_ids_broadcast.offset_tile(routed_indices);
        let (routed_experts, _): (Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }>, Token) = unsafe {
            load_ptr_tko(
                expert_ids_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(routed_mask),
                Some(0i32),
                None::<Token>,
                Latency::<1>,
            )
        };

        let previous_indices = routed_indices
            .reshape(const_shape![ROUTING_PAIRWISE_CAPACITY, 1])
            .broadcast(const_shape![
                ROUTING_PAIRWISE_CAPACITY,
                ROUTING_PAIRWISE_CAPACITY
            ]);
        let current_indices = routed_indices
            .reshape(const_shape![1, ROUTING_PAIRWISE_CAPACITY])
            .broadcast(const_shape![
                ROUTING_PAIRWISE_CAPACITY,
                ROUTING_PAIRWISE_CAPACITY
            ]);
        let previous_experts = routed_experts
            .reshape(const_shape![ROUTING_PAIRWISE_CAPACITY, 1])
            .broadcast(const_shape![
                ROUTING_PAIRWISE_CAPACITY,
                ROUTING_PAIRWISE_CAPACITY
            ]);
        let current_experts = routed_experts
            .reshape(const_shape![1, ROUTING_PAIRWISE_CAPACITY])
            .broadcast(const_shape![
                ROUTING_PAIRWISE_CAPACITY,
                ROUTING_PAIRWISE_CAPACITY
            ]);
        let previous_mask = routed_mask
            .reshape(const_shape![ROUTING_PAIRWISE_CAPACITY, 1])
            .broadcast(const_shape![
                ROUTING_PAIRWISE_CAPACITY,
                ROUTING_PAIRWISE_CAPACITY
            ]);
        let same_expert = cmpi(previous_experts, current_experts, predicate::Equal);
        let one_matrix: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY, ROUTING_PAIRWISE_CAPACITY] }> =
            constant(
                1i32,
                const_shape![ROUTING_PAIRWISE_CAPACITY, ROUTING_PAIRWISE_CAPACITY],
            );
        let zero_matrix: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY, ROUTING_PAIRWISE_CAPACITY] }> =
            constant(
                0i32,
                const_shape![ROUTING_PAIRWISE_CAPACITY, ROUTING_PAIRWISE_CAPACITY],
            );
        // entry (i, j) is 1 when route i is valid and targets the same expert as route j.
        let count_values = select(same_expert & previous_mask, one_matrix, zero_matrix);
        // route_counts[j] is the total number of valid routes assigned to the same expert as route j.
        let route_counts: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            reduce_sum(count_values, 0i32);
        // entry (i, j) is 1 when route i is valid, targets the same expert as route j, and i < j.
        let rank_values = select(
            same_expert
                & previous_mask
                & cmpi(previous_indices, current_indices, predicate::LessThan),
            one_matrix,
            zero_matrix,
        );
        // selected_ranks[j] is the zero-based rank of route j among routes assigned to the same expert.
        let selected_ranks: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            reduce_sum(rank_values, 0i32);

        let zero_routes: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            constant(0i32, const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let one_routes: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            constant(1i32, const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let tile_n_values: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            broadcast_scalar(tile_n, const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let tile_n_log2_values: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            broadcast_scalar(tile_n_log2, const_shape![ROUTING_PAIRWISE_CAPACITY]);
        // Mark the first valid route assigned to each expert.
        let first_routes = routed_mask & cmpi(selected_ranks, zero_routes, predicate::Equal);
        let padded_counts = shri(
            route_counts + tile_n_values - one_routes,
            tile_n_log2_values,
        ) * tile_n_values;
        let first_padded_counts = select(first_routes, padded_counts, zero_routes);
        let total_padded: Tile<i32, { [] }> = reduce_sum(first_padded_counts, 0i32);
        let tile_n_log2_scalar: Tile<i32, { [] }> = scalar_to_tile(tile_n_log2);
        let total_groups: Tile<i32, { [] }> = shri(total_padded, tile_n_log2_scalar);
        let num_groups_ptr: PointerTile<*mut i32, { [1] }> =
            pointer_to_tile(cast_mut(num_groups.as_ptr())).reshape(const_shape![1]);
        let _: Token = unsafe {
            store_ptr_tko(
                num_groups_ptr,
                total_groups.reshape(const_shape![1]),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None::<Token>,
                Latency::<1>,
            )
        };

        let first_route_matrix = first_routes
            .reshape(const_shape![ROUTING_PAIRWISE_CAPACITY, 1])
            .broadcast(const_shape![
                ROUTING_PAIRWISE_CAPACITY,
                ROUTING_PAIRWISE_CAPACITY
            ]);
        let padded_count_matrix = padded_counts
            .reshape(const_shape![ROUTING_PAIRWISE_CAPACITY, 1])
            .broadcast(const_shape![
                ROUTING_PAIRWISE_CAPACITY,
                ROUTING_PAIRWISE_CAPACITY
            ]);
        // offset_values[i, j] is route i's padded expert size when i is the first route of a
        // lower-ID expert than route j.
        let offset_values = select(
            first_route_matrix & cmpi(previous_experts, current_experts, predicate::LessThan),
            padded_count_matrix,
            zero_matrix,
        );
        // selected_offsets[j] is the padded-buffer start offset of the expert assigned to route j.
        let selected_offsets: Tile<i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            reduce_sum(offset_values, 0i32);
        // Map each route to its flattened position in the expert-grouped, TILE_N-padded token buffer.
        // Example with TILE_N = 4: expert 0 has offset 0 and local ranks [0, 1, 2],
        // giving positions [0, 1, 2]; expert 1 has offset 4 and ranks [0, 1],
        // giving positions [4, 5].
        let permuted_indices = selected_offsets + selected_ranks;

        let group_tokens_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(group_tokens.as_ptr()));
        let group_tokens_1d: PointerTile<*mut i32, { [1] }> =
            group_tokens_base.reshape(const_shape![1]);
        let group_tokens_broadcast: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            group_tokens_1d.broadcast(const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let group_token_ptrs: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            group_tokens_broadcast.offset_tile(permuted_indices);
        let _: Token = unsafe {
            store_ptr_tko(
                group_token_ptrs,
                token_indices,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(routed_mask),
                None::<Token>,
                Latency::<1>,
            )
        };
        let expanded_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(expanded_to_permuted.as_ptr()));
        let expanded_1d: PointerTile<*mut i32, { [1] }> = expanded_base.reshape(const_shape![1]);
        let expanded_broadcast: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            expanded_1d.broadcast(const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let expanded_ptrs: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            expanded_broadcast.offset_tile(routed_indices);
        let _: Token = unsafe {
            store_ptr_tko(
                expanded_ptrs,
                permuted_indices,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(routed_mask),
                None::<Token>,
                Latency::<1>,
            )
        };

        let tile_n_mask = tile_n_values - one_routes;
        let group_starts = routed_mask
            & cmpi(
                andi(selected_ranks, tile_n_mask),
                zero_routes,
                predicate::Equal,
            );
        let output_groups = shri(permuted_indices, tile_n_log2_values);
        let group_valid_lanes = mini(route_counts - selected_ranks, tile_n_values);
        let group_experts_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(group_experts.as_ptr()));
        let group_experts_1d: PointerTile<*mut i32, { [1] }> =
            group_experts_base.reshape(const_shape![1]);
        let group_expert_output_broadcast: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            group_experts_1d.broadcast(const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let group_expert_ptrs: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            group_expert_output_broadcast.offset_tile(output_groups);
        let _: Token = unsafe {
            store_ptr_tko(
                group_expert_ptrs,
                routed_experts,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(group_starts),
                None::<Token>,
                Latency::<1>,
            )
        };

        let group_limits_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(group_limits.as_ptr()));
        let group_limits_1d: PointerTile<*mut i32, { [1] }> =
            group_limits_base.reshape(const_shape![1]);
        let group_limit_output_broadcast: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            group_limits_1d.broadcast(const_shape![ROUTING_PAIRWISE_CAPACITY]);
        let group_limit_ptrs: PointerTile<*mut i32, { [ROUTING_PAIRWISE_CAPACITY] }> =
            group_limit_output_broadcast.offset_tile(output_groups);
        let _: Token = unsafe {
            store_ptr_tko(
                group_limit_ptrs,
                group_valid_lanes,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(group_starts),
                None::<Token>,
                Latency::<1>,
            )
        };
    }

    // Larger batches: build per-expert histograms and offsets, then scatter the
    // routes into one or more TILE_N groups per expert.
    // Histogram routing pipeline:
    //   1. Zero per-expert counts, then atomically count routes and assign each route an expert-local rank.
    //   2. Round each expert's count up to TILE_N and prefix-scan the padded counts to build expert offsets.
    //   3. Derive the total number of groups from the final padded offset.
    //   4. Map each route to expert_offset + local_rank.
    //   5. Scatter token indices and reverse mappings into the expert-grouped layout.
    //   6. Emit expert and valid-lane metadata at each TILE_N group boundary.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn routing_group_histogram_kernel<
        const ROUTED_POW2: i32,
        const EXPERTS_POW2: i32,
    >(
        group_tokens: &Tensor<i32, { [-1, -1] }>, // [max_groups, TN]
        group_experts: &Tensor<i32, { [-1] }>,    // [max_groups]
        group_limits: &Tensor<i32, { [-1] }>,     // [max_groups]
        expanded_to_permuted: &Tensor<i32, { [-1, -1] }>, // [num_tokens, top_k]
        expert_counts: &Tensor<i32, { [-1] }>,    // [num_experts]
        expert_offsets: &Tensor<i32, { [-1] }>,   // [num_experts + 1]
        num_groups: &Tensor<i32, { [-1] }>,       // [1]
        expert_ids: &Tensor<i32, { [-1, -1] }>,   // [num_tokens, top_k]
        top_k: i32,
        num_experts: i32,
        routed_size: i32,
        tile_n: i32,
        tile_n_log2: i32,
    ) {
        let routed_indices: Tile<i32, { [ROUTED_POW2] }> = iota(const_shape![ROUTED_POW2]);
        let routed_limit: Tile<i32, { [ROUTED_POW2] }> =
            broadcast_scalar(routed_size, const_shape![ROUTED_POW2]);
        let routed_mask: Tile<bool, { [ROUTED_POW2] }> =
            cmpi(routed_indices, routed_limit, predicate::LessThan);
        let token_indices = routed_indices / broadcast_scalar(top_k, const_shape![ROUTED_POW2]);

        let expert_ids_base: PointerTile<*const i32, { [] }> = pointer_to_tile(expert_ids.as_ptr());
        let expert_ids_1d: PointerTile<*const i32, { [1] }> =
            expert_ids_base.reshape(const_shape![1]);
        let expert_ids_broadcast: PointerTile<*const i32, { [ROUTED_POW2] }> =
            expert_ids_1d.broadcast(const_shape![ROUTED_POW2]);
        let expert_ids_ptrs: PointerTile<*const i32, { [ROUTED_POW2] }> =
            expert_ids_broadcast.offset_tile(routed_indices);
        let (routed_experts, _): (Tile<i32, { [ROUTED_POW2] }>, Token) = unsafe {
            load_ptr_tko(
                expert_ids_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(routed_mask),
                Some(0i32),
                None::<Token>,
                Latency::<1>,
            )
        };

        let experts: Tile<i32, { [EXPERTS_POW2] }> = iota(const_shape![EXPERTS_POW2]);
        let expert_limit: Tile<i32, { [EXPERTS_POW2] }> =
            broadcast_scalar(num_experts, const_shape![EXPERTS_POW2]);
        let valid_experts: Tile<bool, { [EXPERTS_POW2] }> =
            cmpi(experts, expert_limit, predicate::LessThan);

        // expert_counts[e] is the number of valid routes assigned to expert e.
        let expert_counts_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(expert_counts.as_ptr()));
        let expert_counts_1d: PointerTile<*mut i32, { [1] }> =
            expert_counts_base.reshape(const_shape![1]);
        let expert_counts_broadcast: PointerTile<*mut i32, { [EXPERTS_POW2] }> =
            expert_counts_1d.broadcast(const_shape![EXPERTS_POW2]);
        let expert_counts_ptrs: PointerTile<*mut i32, { [EXPERTS_POW2] }> =
            expert_counts_broadcast.offset_tile(experts);
        let zero_counts: Tile<i32, { [EXPERTS_POW2] }> = constant(0i32, const_shape![EXPERTS_POW2]);
        let zero_counts_token = unsafe {
            store_ptr_tko(
                expert_counts_ptrs,
                zero_counts,
                ordering::Relaxed,
                Some(scope::TileBlock),
                Some(valid_experts),
                None::<Token>,
                Latency::<1>,
            )
        };

        let routed_count_base: PointerTile<*mut i32, { [1] }> =
            expert_counts_base.reshape(const_shape![1]);
        let routed_count_broadcast: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            routed_count_base.broadcast(const_shape![ROUTED_POW2]);
        let routed_count_ptrs: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            routed_count_broadcast.offset_tile(routed_experts);
        let increments: Tile<i32, { [ROUTED_POW2] }> = constant(1i32, const_shape![ROUTED_POW2]);
        let (selected_ranks, count_token): (Tile<i32, { [ROUTED_POW2] }>, Token) = unsafe {
            atomic_rmw_tko(
                routed_count_ptrs,
                increments,
                atomic::Add,
                ordering::Relaxed,
                scope::TileBlock,
                Some(routed_mask),
                Some(zero_counts_token),
            )
        };
        // Load expert_counts from global memory after the zero-initialization store and atomic stores.
        let (counts, counts_token): (Tile<i32, { [EXPERTS_POW2] }>, Token) = unsafe {
            load_ptr_tko(
                expert_counts_ptrs,
                ordering::Relaxed,
                Some(scope::TileBlock),
                Some(valid_experts),
                Some(0i32),
                Some(count_token),
                Latency::<1>,
            )
        };

        let tile_n_values: Tile<i32, { [EXPERTS_POW2] }> =
            broadcast_scalar(tile_n, const_shape![EXPERTS_POW2]);
        let tile_n_log2_values: Tile<i32, { [EXPERTS_POW2] }> =
            broadcast_scalar(tile_n_log2, const_shape![EXPERTS_POW2]);
        let one_counts: Tile<i32, { [EXPERTS_POW2] }> = constant(1i32, const_shape![EXPERTS_POW2]);
        let rounded_groups: Tile<i32, { [EXPERTS_POW2] }> =
            shri(counts + tile_n_values - one_counts, tile_n_log2_values);
        let padded_counts: Tile<i32, { [EXPERTS_POW2] }> = rounded_groups * tile_n_values;
        let padded_ends: Tile<i32, { [EXPERTS_POW2] }> =
            scan_sum(padded_counts, 0i32, reverse::Forward, 0i32);
        // padded_offsets[e] is expert e's starting position in the TILE_N-padded grouped-token buffer.
        let padded_offsets: Tile<i32, { [EXPERTS_POW2] }> = padded_ends - padded_counts;

        let expert_offsets_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(expert_offsets.as_ptr()));
        let expert_offsets_1d: PointerTile<*mut i32, { [1] }> =
            expert_offsets_base.reshape(const_shape![1]);
        let expert_offsets_broadcast: PointerTile<*mut i32, { [EXPERTS_POW2] }> =
            expert_offsets_1d.broadcast(const_shape![EXPERTS_POW2]);
        let expert_offsets_ptrs: PointerTile<*mut i32, { [EXPERTS_POW2] }> =
            expert_offsets_broadcast.offset_tile(experts);
        let offsets_token = unsafe {
            store_ptr_tko(
                expert_offsets_ptrs,
                padded_offsets,
                ordering::Relaxed,
                Some(scope::TileBlock),
                Some(valid_experts),
                Some(counts_token),
                Latency::<1>,
            )
        };

        // Append total_padded as the sentinel expert_offsets[num_experts], marking the end of the
        // last expert's padded range. This allows every expert range to be represented uniformly as
        // [expert_offsets[e], expert_offsets[e + 1]).
        let last_expert: Tile<i32, { [] }> = scalar_to_tile(num_experts - 1);
        let total_padded: Tile<i32, { [1] }> = extract(padded_ends, [last_expert]);
        let final_offset: Tile<i32, { [1] }> = broadcast_scalar(num_experts, const_shape![1]);
        let final_offset_base: PointerTile<*mut i32, { [1] }> =
            expert_offsets_base.reshape(const_shape![1]);
        let final_offset_ptr: PointerTile<*mut i32, { [1] }> =
            final_offset_base.offset_tile(final_offset);
        let _: Token = unsafe {
            store_ptr_tko(
                final_offset_ptr,
                total_padded,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None::<Token>,
                Latency::<1>,
            )
        };

        let tile_n_log2_one: Tile<i32, { [1] }> = broadcast_scalar(tile_n_log2, const_shape![1]);
        let total_groups = shri(total_padded, tile_n_log2_one);
        let num_groups_ptr: PointerTile<*mut i32, { [1] }> =
            pointer_to_tile(cast_mut(num_groups.as_ptr())).reshape(const_shape![1]);
        let _: Token = unsafe {
            store_ptr_tko(
                num_groups_ptr,
                total_groups,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None::<Token>,
                Latency::<1>,
            )
        };

        let selected_offset_base: PointerTile<*mut i32, { [1] }> =
            expert_offsets_base.reshape(const_shape![1]);
        let selected_offset_broadcast: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            selected_offset_base.broadcast(const_shape![ROUTED_POW2]);
        let selected_offset_ptrs: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            selected_offset_broadcast.offset_tile(routed_experts);
        let (selected_offsets, selected_offsets_token): (Tile<i32, { [ROUTED_POW2] }>, Token) = unsafe {
            load_ptr_tko(
                selected_offset_ptrs,
                ordering::Relaxed,
                Some(scope::TileBlock),
                Some(routed_mask),
                Some(0i32),
                Some(offsets_token),
                Latency::<1>,
            )
        };

        let permuted_indices = selected_offsets + selected_ranks;

        let (selected_counts, _): (Tile<i32, { [ROUTED_POW2] }>, Token) = unsafe {
            load_ptr_tko(
                routed_count_ptrs,
                ordering::Relaxed,
                Some(scope::TileBlock),
                Some(routed_mask),
                Some(0i32),
                Some(selected_offsets_token),
                Latency::<1>,
            )
        };
        let routed_tile_n: Tile<i32, { [ROUTED_POW2] }> =
            broadcast_scalar(tile_n, const_shape![ROUTED_POW2]);
        let routed_tile_n_log2: Tile<i32, { [ROUTED_POW2] }> =
            broadcast_scalar(tile_n_log2, const_shape![ROUTED_POW2]);
        let routed_tile_n_mask = routed_tile_n - constant(1i32, const_shape![ROUTED_POW2]);
        let zero_routed: Tile<i32, { [ROUTED_POW2] }> = constant(0i32, const_shape![ROUTED_POW2]);
        let group_starts = routed_mask
            & cmpi(
                andi(selected_ranks, routed_tile_n_mask),
                zero_routed,
                predicate::Equal,
            );
        let output_groups = shri(permuted_indices, routed_tile_n_log2);
        let group_valid_lanes = mini(selected_counts - selected_ranks, routed_tile_n);

        let group_tokens_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(group_tokens.as_ptr()));
        let selected_group_tokens_1d: PointerTile<*mut i32, { [1] }> =
            group_tokens_base.reshape(const_shape![1]);
        let selected_group_tokens_broadcast: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            selected_group_tokens_1d.broadcast(const_shape![ROUTED_POW2]);
        let selected_group_token_ptrs: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            selected_group_tokens_broadcast.offset_tile(permuted_indices);
        let _: Token = unsafe {
            store_ptr_tko(
                selected_group_token_ptrs,
                token_indices,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(routed_mask),
                None::<Token>,
                Latency::<1>,
            )
        };

        let expanded_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(expanded_to_permuted.as_ptr()));
        let expanded_1d: PointerTile<*mut i32, { [1] }> = expanded_base.reshape(const_shape![1]);
        let expanded_broadcast: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            expanded_1d.broadcast(const_shape![ROUTED_POW2]);
        let expanded_ptrs: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            expanded_broadcast.offset_tile(routed_indices);
        let _: Token = unsafe {
            store_ptr_tko(
                expanded_ptrs,
                permuted_indices,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(routed_mask),
                None::<Token>,
                Latency::<1>,
            )
        };

        let group_experts_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(group_experts.as_ptr()));
        let group_experts_1d: PointerTile<*mut i32, { [1] }> =
            group_experts_base.reshape(const_shape![1]);
        let group_limits_base: PointerTile<*mut i32, { [] }> =
            pointer_to_tile(cast_mut(group_limits.as_ptr()));
        let group_limits_1d: PointerTile<*mut i32, { [1] }> =
            group_limits_base.reshape(const_shape![1]);

        let group_expert_output_broadcast: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            group_experts_1d.broadcast(const_shape![ROUTED_POW2]);
        let group_expert_ptrs: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            group_expert_output_broadcast.offset_tile(output_groups);
        let _: Token = unsafe {
            store_ptr_tko(
                group_expert_ptrs,
                routed_experts,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(group_starts),
                None::<Token>,
                Latency::<1>,
            )
        };

        let group_limit_output_broadcast: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            group_limits_1d.broadcast(const_shape![ROUTED_POW2]);
        let group_limit_ptrs: PointerTile<*mut i32, { [ROUTED_POW2] }> =
            group_limit_output_broadcast.offset_tile(output_groups);
        let _: Token = unsafe {
            store_ptr_tko(
                group_limit_ptrs,
                group_valid_lanes,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(group_starts),
                None::<Token>,
                Latency::<1>,
            )
        };
    }

    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn nvfp4_quantize_kernel<
        const TILE_COLS: i32,
        const PACKED_COLS: i32,      // TILE_COLS / 2
        const SCALE_GROUPS: i32,     // TILE_COLS / 16
        const VALUE_SHAPE: [i32; 2], // [TILE_COLS / 16, 16]
        const SCALE_SHAPE: [i32; 1], // TILE_COLS / 16
    >(
        packed_out: &Tensor<f4e2m1fnx2, { [-1, -1] }>, // [num_tokens, hidden_size / 2]
        scales_out: &Tensor<f8e4m3fn, { [-1, -1] }>,   // [num_tokens, hidden_size / 16]
        input: &Tensor<f16, { [-1, -1] }>,             // [num_tokens, hidden_size]
        input_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let col_tile = pid.1;
        let input_part = input.partition(const_shape![1, TILE_COLS]);
        let values_f16: Tile<f16, { [1, TILE_COLS] }> = input_part.load([row, col_tile]);
        let value_shape: Shape<VALUE_SHAPE> = Shape::<VALUE_SHAPE> {
            dims: &[SCALE_GROUPS, 16],
        };
        let scale_shape: Shape<SCALE_SHAPE> = Shape::<SCALE_SHAPE> {
            dims: &[SCALE_GROUPS],
        };
        let values: Tile<f32, VALUE_SHAPE> = convert_tile(values_f16.reshape(value_shape));
        let amax_reduced: Tile<f32, { [SCALE_GROUPS] }> = reduce_max(absf(values), 1i32);
        let amax: Tile<f32, SCALE_SHAPE> = amax_reduced.reshape(scale_shape);
        let six: Tile<f32, SCALE_SHAPE> = constant(6.0f32, scale_shape);
        let input_scale_tile: Tile<f32, SCALE_SHAPE> = broadcast_scalar(input_scale, scale_shape);
        let scales_unclamped = amax * (input_scale_tile / six);
        let e4m3_max: Tile<f32, SCALE_SHAPE> = constant(448.0f32, scale_shape);
        let scales_f32 = minf(scales_unclamped, e4m3_max, nan::Disabled, ftz::Disabled);
        let scales: Tile<f8e4m3fn, SCALE_SHAPE> = convert_tile(scales_f32);
        let scale_values: Tile<f32, SCALE_SHAPE> = convert_tile(scales);
        let zero_scale: Tile<f32, SCALE_SHAPE> = constant(0.0f32, scale_shape);
        let nonzero = cmpf(
            amax,
            zero_scale,
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let one_scale: Tile<f32, SCALE_SHAPE> = constant(1.0f32, scale_shape);
        let safe_scale = select(nonzero, scale_values, one_scale);
        let inv_scale = input_scale_tile / safe_scale;
        let quantized_f32 = values
            * inv_scale
                .reshape(const_shape![SCALE_GROUPS, 1])
                .broadcast(value_shape);
        let quantized: Tile<f4e2m1fn, { [TILE_COLS] }> =
            convert_tile(quantized_f32.reshape(const_shape![TILE_COLS]));
        let packed: Tile<f4e2m1fnx2, { [1, PACKED_COLS] }> =
            quantized.pack(const_shape![1, PACKED_COLS]);

        let mut packed_part: PartitionMut<f4e2m1fnx2, { [1, PACKED_COLS] }> =
            unsafe { packed_out.partition_full_mut(const_shape![1, PACKED_COLS]) };
        let mut scales_part: PartitionMut<f8e4m3fn, { [1, SCALE_GROUPS] }> =
            unsafe { scales_out.partition_full_mut(const_shape![1, SCALE_GROUPS]) };
        unsafe {
            packed_part.store(packed, [row, col_tile]);
            scales_part.store(
                scales.reshape(const_shape![1, SCALE_GROUPS]),
                [row, col_tile],
            );
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn grouped_fc1_kernel<
        const TM: i32,
        const TM_GROUPS: i32, // TM / 32
        const TN: i32,
        const TILE_K: i32,
        const TILE_K_CHUNKS: i32,   // TILE_K / 64
        const TILE_K_PACKED: i32,   // TILE_K / 2
        const TILE_K_SCALES: i32,   // TILE_K / 16
        const PACKED_TM: i32,       // TILE_M / 2
        const TM_SCALE_GROUPS: i32, // TILE_M / 16
        const X_LATENCY: i32,
        const W_LATENCY: i32,
        const REQUANT_SHAPE: [i32; 2],       // [TN, TILE_M / 16]
        const REQUANT_BLOCK_SHAPE: [i32; 3], // [TN, TILE_M / 16, 16]
    >(
        inter_out: &Tensor<f4e2m1fnx2, { [-1, -1, -1] }>,
        inter_scales: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        hidden: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        hidden_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        hidden_row_stride: i32,
        hidden_scale_row_stride: i32,
        group_tokens: &Tensor<i32, { [-1, -1] }>,
        group_experts: &Tensor<i32, { [-1] }>,
        group_limits: &Tensor<i32, { [-1] }>,
        num_groups: &Tensor<i32, { [-1] }>,
        w_up: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        w_up_scales: &Tensor<f8e4m3fn, { [-1, -1, 2, 256] }>,
        w_up_global: &Tensor<f32, { [-1] }>,
        input_scale: f32,
        intermediate_size: i32,
        k_tiles: i32,
    ) {
        let pid = get_tile_block_id();
        let output_tile = pid.0;
        let group_start = pid.1;
        let group_step = get_num_tile_blocks().1;
        let active_groups: i32 = tile_to_scalar(
            num_groups
                .partition(const_shape![1])
                .load([0i32])
                .reshape(const_shape![]),
        );

        let group_token_part = group_tokens.partition(const_shape![1, TN]);
        let group_expert_part = group_experts.partition(const_shape![1]);
        let group_limit_part = group_limits.partition(const_shape![1]);
        let w_part = w_up.partition(const_shape![TM, TILE_K_PACKED]);
        let w_scale_part = w_up_scales.partition(const_shape![1, TILE_K_CHUNKS, 2, 256]);
        let global_part = w_up_global.partition(const_shape![1]);
        let mut inter_part: PartitionMut<f4e2m1fnx2, { [1, TN, PACKED_TM] }> =
            unsafe { inter_out.partition_full_mut(const_shape![1, TN, PACKED_TM]) };
        let mut inter_scale_part: PartitionMut<f8e4m3fn, { [1, TM_SCALE_GROUPS, TN] }> =
            unsafe { inter_scales.partition_full_mut(const_shape![1, TM_SCALE_GROUPS, TN]) };

        for group in (group_start..active_groups).step_by(group_step as usize) {
            let expert = tile_to_scalar(group_expert_part.load([group]).reshape(const_shape![]));

            let physical_row = expert * intermediate_size + output_tile * TM;
            let scale_row_tile = physical_row / 128;
            let scale_row_offset = physical_row % 128;

            let group_limit =
                tile_to_scalar(group_limit_part.load([group]).reshape(const_shape![]));
            let group_limit_tile: Tile<i32, { [TN, 1] }> =
                broadcast_scalar(group_limit, const_shape![TN, 1]);

            let lane_indices_1d: Tile<i32, { [TN] }> = iota(const_shape![TN]);
            let lane_indices: Tile<i32, { [TN, 1] }> = lane_indices_1d.reshape(const_shape![TN, 1]);
            let valid_lanes: Tile<bool, { [TN, 1] }> =
                cmpi(lane_indices, group_limit_tile, predicate::LessThan);

            let tokens: Tile<i32, { [TN, 1] }> = group_token_part
                .load([group, 0i32])
                .reshape(const_shape![TN, 1]);
            let zero_tokens: Tile<i32, { [TN, 1] }> = constant(0i32, const_shape![TN, 1]);
            let safe_tokens: Tile<i32, { [TN, 1] }> = select(valid_lanes, tokens, zero_tokens);

            let mut acc: Tile<f32, { [TM, TN] }> = constant(0.0f32, const_shape![TM, TN]);

            for k_tile in 0i32..k_tiles {
                // x
                let packed_col_indices: Tile<i32, { [TILE_K_PACKED] }> =
                    iota(const_shape![TILE_K_PACKED]);
                let packed_tile_offset: Tile<i32, { [TILE_K_PACKED] }> =
                    broadcast_scalar(k_tile * TILE_K_PACKED, const_shape![TILE_K_PACKED]);
                let packed_cols = packed_col_indices + packed_tile_offset;
                let hidden_row_stride_tile: Tile<i32, { [TN, TILE_K_PACKED] }> =
                    broadcast_scalar(hidden_row_stride, const_shape![TN, TILE_K_PACKED]);
                let packed_offsets: Tile<i32, { [TN, TILE_K_PACKED] }> =
                    safe_tokens.broadcast(const_shape![TN, TILE_K_PACKED]) * hidden_row_stride_tile
                        + packed_cols
                            .reshape(const_shape![1, TILE_K_PACKED])
                            .broadcast(const_shape![TN, TILE_K_PACKED]);
                let packed_mask: Tile<bool, { [TN, TILE_K_PACKED] }> =
                    valid_lanes.broadcast(const_shape![TN, TILE_K_PACKED]);

                let hidden_base: PointerTile<*const f4e2m1fnx2, { [] }> =
                    pointer_to_tile(hidden.as_ptr());
                let hidden_2d: PointerTile<*const f4e2m1fnx2, { [1, 1] }> =
                    hidden_base.reshape(const_shape![1, 1]);
                let hidden_broadcast: PointerTile<*const f4e2m1fnx2, { [TN, TILE_K_PACKED] }> =
                    hidden_2d.broadcast(const_shape![TN, TILE_K_PACKED]);
                let hidden_ptrs: PointerTile<*const f4e2m1fnx2, { [TN, TILE_K_PACKED] }> =
                    hidden_broadcast.offset_tile(packed_offsets);

                // `load_ptr_tko` takes `Latency<const u32>`, while kernel const generics are i32.
                // Rust cannot cast a generic const inside another const argument, so instantiate
                // every autotuned latency explicitly for this load and the scale load below.
                let (x_bytes, _): (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) =
                    if X_LATENCY == 0 {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<0>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 2 {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<2>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 3 {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<3>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 4 {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<4>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 6 {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<6>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 7 {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<7>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 8 {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<8>,
                            )
                        };
                        loaded
                    } else {
                        let loaded: (Tile<f4e2m1fnx2, { [TN, TILE_K_PACKED] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(packed_mask),
                                None::<f4e2m1fnx2>,
                                None::<Token>,
                                Latency::<1>,
                            )
                        };
                        loaded
                    };
                let x_fp4: Tile<f4e2m1fn, { [TILE_K, TN] }> =
                    x_bytes.unpack(const_shape![TN, TILE_K]).transpose();

                // x_scales
                let scale_col_indices: Tile<i32, { [TILE_K_SCALES] }> =
                    iota(const_shape![TILE_K_SCALES]);
                let scale_tile_offset: Tile<i32, { [TILE_K_SCALES] }> =
                    broadcast_scalar(k_tile * TILE_K_SCALES, const_shape![TILE_K_SCALES]);
                let scale_cols = scale_col_indices + scale_tile_offset;
                let hidden_scale_row_stride_tile: Tile<i32, { [TN, TILE_K_SCALES] }> =
                    broadcast_scalar(hidden_scale_row_stride, const_shape![TN, TILE_K_SCALES]);
                let scale_offsets: Tile<i32, { [TN, TILE_K_SCALES] }> = safe_tokens
                    .broadcast(const_shape![TN, TILE_K_SCALES])
                    * hidden_scale_row_stride_tile
                    + scale_cols
                        .reshape(const_shape![1, TILE_K_SCALES])
                        .broadcast(const_shape![TN, TILE_K_SCALES]);
                let scale_mask: Tile<bool, { [TN, TILE_K_SCALES] }> =
                    valid_lanes.broadcast(const_shape![TN, TILE_K_SCALES]);

                let hidden_scale_base: PointerTile<*const f8e4m3fn, { [] }> =
                    pointer_to_tile(hidden_scales.as_ptr());
                let hidden_scale_2d: PointerTile<*const f8e4m3fn, { [1, 1] }> =
                    hidden_scale_base.reshape(const_shape![1, 1]);
                let hidden_scale_broadcast: PointerTile<*const f8e4m3fn, { [TN, TILE_K_SCALES] }> =
                    hidden_scale_2d.broadcast(const_shape![TN, TILE_K_SCALES]);
                let hidden_scale_ptrs: PointerTile<*const f8e4m3fn, { [TN, TILE_K_SCALES] }> =
                    hidden_scale_broadcast.offset_tile(scale_offsets);

                let (x_scale_values, _): (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) =
                    if X_LATENCY == 0 {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<0>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 2 {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<2>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 3 {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<3>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 4 {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<4>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 6 {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<6>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 7 {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<7>,
                            )
                        };
                        loaded
                    } else if X_LATENCY == 8 {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<8>,
                            )
                        };
                        loaded
                    } else {
                        let loaded: (Tile<f8e4m3fn, { [TN, TILE_K_SCALES] }>, Token) = unsafe {
                            load_ptr_tko(
                                hidden_scale_ptrs,
                                ordering::Weak,
                                None::<scope::TileBlock>,
                                Some(scale_mask),
                                None::<f8e4m3fn>,
                                None::<Token>,
                                Latency::<1>,
                            )
                        };
                        loaded
                    };
                let x_scales: Tile<f8e4m3fn, { [TILE_K_SCALES, TN] }> = x_scale_values.transpose();

                // w
                // A profile latency of 0 means no explicit latency hint. We don't support 'None' yet.
                let w_bytes: Tile<f4e2m1fnx2, { [TM, TILE_K_PACKED] }> = if W_LATENCY == 0 {
                    let loaded: Tile<f4e2m1fnx2, { [TM, TILE_K_PACKED] }> =
                        w_part.load([physical_row / TM, k_tile]);
                    loaded
                } else {
                    let loaded: Tile<f4e2m1fnx2, { [TM, TILE_K_PACKED] }> = load_view_tko(
                        &w_part,
                        [physical_row / TM, k_tile],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(W_LATENCY),
                        tma::Enabled,
                    );
                    loaded
                };
                let w_fp4: Tile<f4e2m1fn, { [TM, TILE_K] }> =
                    w_bytes.unpack(const_shape![TM, TILE_K]);

                // w_scales
                let scales_physical: Tile<f8e4m3fn, { [1, TILE_K_CHUNKS, 2, 256] }> =
                    if W_LATENCY == 0 {
                        let loaded: Tile<f8e4m3fn, { [1, TILE_K_CHUNKS, 2, 256] }> =
                            w_scale_part.load([scale_row_tile, k_tile, 0i32, 0i32]);
                        loaded
                    } else {
                        let loaded: Tile<f8e4m3fn, { [1, TILE_K_CHUNKS, 2, 256] }> = load_view_tko(
                            &w_scale_part,
                            [scale_row_tile, k_tile, 0i32, 0i32],
                            ordering::Weak,
                            scope::TileBlock,
                            Some(W_LATENCY),
                            tma::Enabled,
                        );
                        loaded
                    };
                let scales_4d: Tile<f8e4m3fn, { [TILE_K_CHUNKS, 32, 4, 4] }> =
                    scales_physical.reshape(const_shape![TILE_K_CHUNKS, 32, 4, 4]);
                let scales_permuted: Tile<f8e4m3fn, { [4, 32, TILE_K_CHUNKS, 4] }> =
                    permute(scales_4d, const_array![2, 1, 0, 3]);
                let scales_logical: Tile<f8e4m3fn, { [128, TILE_K_SCALES] }> =
                    scales_permuted.reshape(const_shape![128, TILE_K_SCALES]);
                let row: Tile<i32, { [] }> = scalar_to_tile(scale_row_offset / TM);
                let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
                let w_scales: Tile<f8e4m3fn, { [TM, TILE_K_SCALES] }> =
                    extract(scales_logical, [row, zero]);
                acc = mmaf_scaled(w_fp4, x_fp4, acc, w_scales, x_scales);
            }

            // Weight is shuffled.
            let physical: Tile<f32, { [TN, TM_GROUPS, 4, 8] }> =
                acc.transpose().reshape(const_shape![TN, TM_GROUPS, 4, 8]);
            let logical_4d: Tile<f32, { [TN, TM_GROUPS, 8, 4] }> =
                permute(physical, const_array![0, 1, 3, 2]);
            let logical: Tile<f32, { [TN, TM] }> = logical_4d.reshape(const_shape![TN, TM]);

            let global: f32 = tile_to_scalar(global_part.load([expert]).reshape(const_shape![]));
            let global = global / input_scale;
            let global_tile: Tile<f32, { [TN, TM] }> =
                broadcast_scalar(global, const_shape![TN, TM]);
            let scaled: Tile<f32, { [TN, TM] }> = logical * global_tile;
            let zero_activated: Tile<f32, { [TN, TM] }> = constant(0.0f32, const_shape![TN, TM]);
            let relu: Tile<f32, { [TN, TM] }> =
                maxf(scaled, zero_activated, nan::Disabled, ftz::Enabled);
            let activated: Tile<f32, { [TN, TM] }> = relu * relu;
            let valid_output: Tile<bool, { [TN, TM] }> =
                valid_lanes.broadcast(const_shape![TN, TM]);
            let activated: Tile<f32, { [TN, TM] }> =
                select(valid_output, activated, zero_activated);
            let requant_shape: Shape<REQUANT_SHAPE> = Shape::<REQUANT_SHAPE> {
                dims: &[TN, TM_SCALE_GROUPS],
            };
            let requant_block_shape: Shape<REQUANT_BLOCK_SHAPE> = Shape::<REQUANT_BLOCK_SHAPE> {
                dims: &[TN, TM_SCALE_GROUPS, 16],
            };
            let blocks: Tile<f32, REQUANT_BLOCK_SHAPE> = activated.reshape(requant_block_shape);
            let amax_reduced: Tile<f32, { [TN, TM_SCALE_GROUPS] }> = reduce_max(absf(blocks), 2i32);
            let amax: Tile<f32, REQUANT_SHAPE> = amax_reduced.reshape(requant_shape);
            let reciprocal_six: Tile<f32, REQUANT_SHAPE> =
                broadcast_scalar(1.0f32 / 6.0f32, requant_shape);
            let scales_f32: Tile<f32, REQUANT_SHAPE> = amax * reciprocal_six;
            let scales: Tile<f8e4m3fn, REQUANT_SHAPE> = convert_tile(scales_f32);
            let decoded: Tile<f32, REQUANT_SHAPE> = convert_tile(scales);
            let zero_scale: Tile<f32, REQUANT_SHAPE> = constant(0.0f32, requant_shape);
            let nonzero = cmpf(
                amax,
                zero_scale,
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            );
            let one_scale: Tile<f32, REQUANT_SHAPE> = constant(1.0f32, requant_shape);
            let safe_scale: Tile<f32, REQUANT_SHAPE> = select(nonzero, decoded, one_scale);
            let quantized_f32 = true_div(
                blocks,
                safe_scale
                    .reshape(const_shape![TN, TM_SCALE_GROUPS, 1])
                    .broadcast(requant_block_shape),
            );
            let quantized: Tile<f4e2m1fn, { [TN, TM] }> =
                convert_tile(quantized_f32.reshape(const_shape![TN, TM]));
            let packed = quantized.pack(const_shape![TN, PACKED_TM]);
            let scales_transposed: Tile<f8e4m3fn, { [TM_SCALE_GROUPS, TN] }> = scales
                .reshape(const_shape![TN, TM_SCALE_GROUPS])
                .transpose();
            unsafe {
                inter_part.store(
                    packed.reshape(const_shape![1, TN, PACKED_TM]),
                    [group, 0i32, output_tile],
                );
                inter_scale_part.store(
                    scales_transposed.reshape(const_shape![1, TM_SCALE_GROUPS, TN]),
                    [group, output_tile, 0i32],
                );
            }
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn grouped_fc2_kernel<
        const TM: i32,
        const TM_GROUPS: i32, // TM / 32
        const TN: i32,
        const TILE_K: i32,
        const TILE_K_CHUNKS: i32,       // TILE_K / 64
        const TILE_K_PACKED: i32,       // TILE_K / 2
        const TILE_K_SCALES: i32,       // TILE_K / 16
        const TILE_K_SCALE_VALUES: i32, // TILE_K / 64 * 512
        const X_LATENCY: i32,
        const W_LATENCY: i32,
        const S_LATENCY: i32,
    >(
        grouped_out: &Tensor<bf16, { [-1, -1, -1] }>,
        inter: &Tensor<f4e2m1fnx2, { [-1, -1, -1] }>,
        inter_scales: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        group_experts: &Tensor<i32, { [-1] }>,
        num_groups: &Tensor<i32, { [-1] }>,
        w_down: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        w_down_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w_down_global: &Tensor<f32, { [-1] }>,
        hidden_size: i32,
        k_tiles: i32,
    ) {
        let output_tile = get_tile_block_id().0;
        let group_start = get_tile_block_id().1;
        let group_step = get_num_tile_blocks().1;
        let active_groups: i32 = tile_to_scalar(
            num_groups
                .partition(const_shape![1])
                .load([0i32])
                .reshape(const_shape![]),
        );
        let inter_part = inter.partition(const_shape![1, TN, TILE_K_PACKED]);
        let inter_scale_part = inter_scales.partition(const_shape![1, TILE_K_SCALES, TN]);
        let group_expert_part = group_experts.partition(const_shape![1]);
        let w_part = w_down.partition(const_shape![TM, TILE_K_PACKED]);
        let w_scale_part = w_down_scales.partition(const_shape![1, TILE_K_SCALE_VALUES]);
        let global_part = w_down_global.partition(const_shape![1]);
        let mut out_part: PartitionMut<bf16, { [1, TN, TM] }> =
            unsafe { grouped_out.partition_full_mut(const_shape![1, TN, TM]) };

        for group in (group_start..active_groups).step_by(group_step as usize) {
            let expert = tile_to_scalar(group_expert_part.load([group]).reshape(const_shape![]));
            let global = tile_to_scalar(global_part.load([expert]).reshape(const_shape![]));
            let physical_row = expert * hidden_size + output_tile * TM;
            let scale_row_tile = physical_row / 128;
            let scale_row_offset = physical_row % 128;
            let mut acc: Tile<f32, { [TM, TN] }> = constant(0.0f32, const_shape![TM, TN]);

            for k_tile in 0i32..k_tiles {
                // x
                let x_bytes: Tile<f4e2m1fnx2, { [1, TN, TILE_K_PACKED] }> = if X_LATENCY == 0 {
                    let loaded: Tile<f4e2m1fnx2, { [1, TN, TILE_K_PACKED] }> =
                        inter_part.load([group, 0i32, k_tile]);
                    loaded
                } else {
                    let loaded: Tile<f4e2m1fnx2, { [1, TN, TILE_K_PACKED] }> = load_view_tko(
                        &inter_part,
                        [group, 0i32, k_tile],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(X_LATENCY),
                        tma::Enabled,
                    );
                    loaded
                };
                let x_fp4: Tile<f4e2m1fn, { [TILE_K, TN] }> = x_bytes
                    .reshape(const_shape![TN, TILE_K_PACKED])
                    .unpack(const_shape![TN, TILE_K])
                    .transpose();

                // x_scales
                let x_scale_values: Tile<f8e4m3fn, { [1, TILE_K_SCALES, TN] }> = if X_LATENCY == 0 {
                    let loaded: Tile<f8e4m3fn, { [1, TILE_K_SCALES, TN] }> =
                        inter_scale_part.load([group, k_tile, 0i32]);
                    loaded
                } else {
                    let loaded: Tile<f8e4m3fn, { [1, TILE_K_SCALES, TN] }> = load_view_tko(
                        &inter_scale_part,
                        [group, k_tile, 0i32],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(X_LATENCY),
                        tma::Enabled,
                    );
                    loaded
                };
                let x_scales: Tile<f8e4m3fn, { [TILE_K_SCALES, TN] }> =
                    x_scale_values.reshape(const_shape![TILE_K_SCALES, TN]);

                // w_bytes
                let w_bytes: Tile<f4e2m1fnx2, { [TM, TILE_K_PACKED] }> = if W_LATENCY == 0 {
                    let loaded: Tile<f4e2m1fnx2, { [TM, TILE_K_PACKED] }> =
                        w_part.load([physical_row / TM, k_tile]);
                    loaded
                } else {
                    let loaded: Tile<f4e2m1fnx2, { [TM, TILE_K_PACKED] }> = load_view_tko(
                        &w_part,
                        [physical_row / TM, k_tile],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(W_LATENCY),
                        tma::Enabled,
                    );
                    loaded
                };
                let w_fp4: Tile<f4e2m1fn, { [TM, TILE_K] }> =
                    w_bytes.unpack(const_shape![TM, TILE_K]);

                // w_scales
                let scales_physical: Tile<f8e4m3fn, { [1, TILE_K_SCALE_VALUES] }> =
                    if S_LATENCY == 0 {
                        let loaded: Tile<f8e4m3fn, { [1, TILE_K_SCALE_VALUES] }> =
                            w_scale_part.load([scale_row_tile, k_tile]);
                        loaded
                    } else {
                        let loaded: Tile<f8e4m3fn, { [1, TILE_K_SCALE_VALUES] }> = load_view_tko(
                            &w_scale_part,
                            [scale_row_tile, k_tile],
                            ordering::Weak,
                            scope::TileBlock,
                            Some(S_LATENCY),
                            tma::Enabled,
                        );
                        loaded
                    };
                let scales_4d: Tile<f8e4m3fn, { [TILE_K_CHUNKS, 32, 4, 4] }> =
                    scales_physical.reshape(const_shape![TILE_K_CHUNKS, 32, 4, 4]);
                let scales_permuted: Tile<f8e4m3fn, { [4, 32, TILE_K_CHUNKS, 4] }> =
                    permute(scales_4d, const_array![2, 1, 0, 3]);
                let scales_logical: Tile<f8e4m3fn, { [128, TILE_K_SCALES] }> =
                    scales_permuted.reshape(const_shape![128, TILE_K_SCALES]);
                let row: Tile<i32, { [] }> = scalar_to_tile(scale_row_offset / TM);
                let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
                let w_scales: Tile<f8e4m3fn, { [TM, TILE_K_SCALES] }> =
                    extract(scales_logical, [row, zero]);

                acc = mmaf_scaled(w_fp4, x_fp4, acc, w_scales, x_scales);
            }

            acc = acc * broadcast_scalar(global, const_shape![TM, TN]);
            let physical: Tile<f32, { [TN, TM_GROUPS, 4, 8] }> =
                acc.transpose().reshape(const_shape![TN, TM_GROUPS, 4, 8]);
            let logical_4d: Tile<f32, { [TN, TM_GROUPS, 8, 4] }> =
                permute(physical, const_array![0, 1, 3, 2]);
            let logical: Tile<f32, { [1, TN, TM] }> = logical_4d.reshape(const_shape![1, TN, TM]);
            let output: Tile<bf16, { [1, TN, TM] }> = convert_tile(logical);
            unsafe { out_part.store(output, [group, 0i32, output_tile]) };
        }
    }

    // Remove unchecked_accesses will cause performance regression.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn finalize_kernel<const TILE_HIDDEN: i32>(
        out: &Tensor<bf16, { [-1, -1] }>, // [num_tokens, hidden_size]
        grouped: &Tensor<bf16, { [-1, -1, -1] }>, // [max_groups, TN, hidden_size]
        expanded_to_permuted: &Tensor<i32, { [-1, -1] }>, // [num_tokens, top_k]
        routing_weights: &Tensor<bf16, { [-1, -1] }>, // [num_tokens, top_k]
        top_k: i32,
        tile_n: i32,
        tile_n_log2: i32,
    ) {
        let pid = get_tile_block_id();
        let token = pid.0;
        let output_tile = pid.1;
        let grouped_part = grouped.partition(const_shape![1, 1, TILE_HIDDEN]);
        let expanded_part = expanded_to_permuted.partition(const_shape![1, 1]);
        let routing_part = routing_weights.partition(const_shape![1, 1]);
        let mut acc: Tile<f32, { [1, TILE_HIDDEN] }> =
            constant(0.0f32, const_shape![1, TILE_HIDDEN]);

        for slot in 0i32..top_k {
            // expanded_to_permuted saves (token, slot) -> permuted
            // permuted = groupID * TN + laneID
            let permuted: i32 =
                tile_to_scalar(expanded_part.load([token, slot]).reshape(const_shape![]));
            let group = permuted >> tile_n_log2;
            let lane = permuted & (tile_n - 1);

            let routing_weight: Tile<f32, { [1, 1] }> =
                convert_tile(routing_part.load([token, slot]));
            let routing_weight = tile_to_scalar(routing_weight.reshape(const_shape![]));
            let partial: Tile<f32, { [1, 1, TILE_HIDDEN] }> =
                convert_tile(grouped_part.load([group, lane, output_tile]));
            acc = acc
                + partial.reshape(const_shape![1, TILE_HIDDEN])
                    * broadcast_scalar(routing_weight, const_shape![1, TILE_HIDDEN]);
        }

        let mut out_part: PartitionMut<bf16, { [1, TILE_HIDDEN] }> =
            unsafe { out.partition_full_mut(const_shape![1, TILE_HIDDEN]) };
        let output: Tile<bf16, { [1, TILE_HIDDEN] }> = convert_tile(acc);
        unsafe { out_part.store(output, [token, output_tile]) };
    }
}

use kernels::{
    finalize_kernel, grouped_fc1_kernel, grouped_fc2_kernel, nvfp4_quantize_kernel,
    routing_group_histogram_kernel, routing_group_pairwise_kernel,
};

// Kernel config
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 1024;
const NUM_EXPERTS: usize = 8;
const MAX_NUM_TOKENS: usize = 256;
const BATCH_SIZES: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, MAX_NUM_TOKENS];
const TOP_K: usize = 4;
const TILE_M: usize = 128;
const TILE_K: usize = 512;
const QUANTIZE_TILE: usize = 1024;
const TILE_N: usize = 16;
const FINALIZE_TILE: usize = 256;
const ROUTING_PAIRWISE_MAX_ROUTED: usize = kernels::ROUTING_PAIRWISE_CAPACITY as usize;
const TIMING_WARMUPS: usize = 50;
const TIMING_REPLAYS: usize = 1000;
const TIMING_SAMPLES: usize = 20;

// For simple precision test
const ROUTING_PATTERN: [f32; TOP_K] = [0.125, 0.25, 0.25, 0.375];
const UP_FACTORS: [f32; TOP_K] = [0.5, 1.0, 2.0, 1.0];
const DOWN_FACTORS: [f32; TOP_K] = [1.0, 2.0, 0.5, 1.0];

const INPUT_SCALE: f32 = 1.0;

fn max_num_groups(routed_size: usize, tile_n: usize) -> usize {
    let filled_experts = NUM_EXPERTS.min(routed_size);
    filled_experts + (routed_size - filled_experts) / tile_n
}

fn hidden_value(token: usize) -> f32 {
    let magnitude = 0.375 * [1.0, 2.0, 4.0][token % 3];
    if token.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

fn expert_id(token: usize, slot: usize) -> usize {
    (token + slot) % TOP_K
}

fn routing_weight(token: usize, slot: usize) -> f32 {
    ROUTING_PATTERN[(token + slot) % TOP_K]
}

fn expected_output(token: usize) -> bf16 {
    let hidden = hidden_value(token);
    let mut total = 0.0f32;
    for slot in 0..TOP_K {
        let expert = expert_id(token, slot);
        let pre_activation = hidden * UP_FACTORS[expert];
        let relu = pre_activation.max(0.0);
        let activated = relu * relu;
        let fc2 = activated * DOWN_FACTORS[expert];
        total += routing_weight(token, slot) * fc2;
    }
    bf16::from_f32(total)
}

struct MoeTensors {
    hidden: Arc<Tensor<f16>>,
    expert_ids: Arc<Tensor<i32>>,
    routing_weights: Arc<Tensor<bf16>>,
    group_tokens: Arc<Tensor<i32>>,
    group_experts: Arc<Tensor<i32>>,
    group_limits: Arc<Tensor<i32>>,
    expanded_to_permuted: Arc<Tensor<i32>>,
    expert_counts: Arc<Tensor<i32>>,
    expert_offsets: Arc<Tensor<i32>>,
    num_groups: Arc<Tensor<i32>>,
    quantized_hidden: Arc<Tensor<f4e2m1fnx2>>,
    quantized_hidden_scales: Arc<Tensor<f8e4m3fn>>,
    grouped_inter: Arc<Tensor<f4e2m1fnx2>>,
    grouped_inter_scales: Arc<Tensor<f8e4m3fn>>,
    grouped_out: Arc<Tensor<bf16>>,
    w_up: Arc<Tensor<f4e2m1fnx2>>,
    w_up_scales: Arc<Tensor<f8e4m3fn>>,
    w_up_global: Arc<Tensor<f32>>,
    w_down: Arc<Tensor<f4e2m1fnx2>>,
    w_down_scales: Arc<Tensor<f8e4m3fn>>,
    w_down_global: Arc<Tensor<f32>>,
    output: Arc<Tensor<bf16>>,
    num_tokens: usize,
}

fn allocate_tensors(
    num_tokens: usize,
    stream: &Arc<Stream>,
) -> Result<MoeTensors, Box<dyn std::error::Error>> {
    let routed_size = num_tokens * TOP_K;
    let max_groups = max_num_groups(routed_size, TILE_N);
    let hidden_host = (0..num_tokens)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(hidden_value(token)), HIDDEN))
        .collect::<Vec<_>>();
    let hidden: Arc<Tensor<f16>> = api::copy_host_vec_to_device(&Arc::new(hidden_host))
        .reshape(&[num_tokens, HIDDEN])
        .sync_on(stream)?
        .into();
    let expert_ids_host = (0..num_tokens)
        .flat_map(|token| (0..TOP_K).map(move |slot| expert_id(token, slot) as i32))
        .collect::<Vec<_>>();
    let expert_ids: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(expert_ids_host))
        .reshape(&[num_tokens, TOP_K])
        .sync_on(stream)?
        .into();
    let routing_host = (0..num_tokens)
        .flat_map(|token| (0..TOP_K).map(move |slot| bf16::from_f32(routing_weight(token, slot))))
        .collect::<Vec<_>>();
    let routing_weights: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(routing_host))
        .reshape(&[num_tokens, TOP_K])
        .sync_on(stream)?
        .into();

    let fp4_one = f4e2m1fnx2::from_nibbles(0x2, 0x2);
    let scale_one = f8e4m3fn(0x38);
    let w_up: Arc<Tensor<f4e2m1fnx2>> =
        api::full(fp4_one, &[NUM_EXPERTS * INTERMEDIATE, HIDDEN / 2])
            .sync_on(stream)?
            .into();
    let w_up_scales: Arc<Tensor<f8e4m3fn>> = api::full(
        scale_one,
        &[NUM_EXPERTS * INTERMEDIATE / 128, HIDDEN / 64, 2, 256],
    )
    .sync_on(stream)?
    .into();
    let mut w_up_global_host = vec![1.0f32 / HIDDEN as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_up_global_host[expert] = UP_FACTORS[expert] / HIDDEN as f32;
    }
    let w_up_global: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(w_up_global_host))
        .reshape(&[NUM_EXPERTS])
        .sync_on(stream)?
        .into();

    let w_down: Arc<Tensor<f4e2m1fnx2>> =
        api::full(fp4_one, &[NUM_EXPERTS * HIDDEN, INTERMEDIATE / 2])
            .sync_on(stream)?
            .into();
    let w_down_scales: Arc<Tensor<f8e4m3fn>> = api::full(
        scale_one,
        &[NUM_EXPERTS * HIDDEN / 128, INTERMEDIATE / 64 * 512],
    )
    .sync_on(stream)?
    .into();
    let mut w_down_global_host = vec![1.0f32 / INTERMEDIATE as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_down_global_host[expert] = DOWN_FACTORS[expert] / INTERMEDIATE as f32;
    }
    let w_down_global: Arc<Tensor<f32>> =
        api::copy_host_vec_to_device(&Arc::new(w_down_global_host))
            .reshape(&[NUM_EXPERTS])
            .sync_on(stream)?
            .into();

    let group_tokens = api::zeros::<i32>(&[max_groups, TILE_N])
        .sync_on(stream)?
        .into();
    let group_experts = api::zeros::<i32>(&[max_groups]).sync_on(stream)?.into();
    let group_limits = api::zeros::<i32>(&[max_groups]).sync_on(stream)?.into();
    let expanded_to_permuted = api::zeros::<i32>(&[num_tokens, TOP_K])
        .sync_on(stream)?
        .into();
    let expert_counts = api::zeros::<i32>(&[NUM_EXPERTS]).sync_on(stream)?.into();
    let expert_offsets = api::zeros::<i32>(&[NUM_EXPERTS + 1])
        .sync_on(stream)?
        .into();
    let num_groups = api::zeros::<i32>(&[1]).sync_on(stream)?.into();
    let quantized_hidden = api::zeros::<f4e2m1fnx2>(&[num_tokens, HIDDEN / 2])
        .sync_on(stream)?
        .into();
    let quantized_hidden_scales = api::zeros::<f8e4m3fn>(&[num_tokens, HIDDEN / 16])
        .sync_on(stream)?
        .into();
    let grouped_inter = api::zeros::<f4e2m1fnx2>(&[max_groups, TILE_N, INTERMEDIATE / 2])
        .sync_on(stream)?
        .into();
    let grouped_inter_scales = api::zeros::<f8e4m3fn>(&[max_groups, INTERMEDIATE / 16, TILE_N])
        .sync_on(stream)?
        .into();
    let grouped_out = api::zeros::<bf16>(&[max_groups, TILE_N, HIDDEN])
        .sync_on(stream)?
        .into();
    let output = api::zeros::<bf16>(&[num_tokens, HIDDEN])
        .sync_on(stream)?
        .into();

    Ok(MoeTensors {
        hidden,
        expert_ids,
        routing_weights,
        group_tokens,
        group_experts,
        group_limits,
        expanded_to_permuted,
        expert_counts,
        expert_offsets,
        num_groups,
        quantized_hidden,
        quantized_hidden_scales,
        grouped_inter,
        grouped_inter_scales,
        grouped_out,
        w_up,
        w_up_scales,
        w_up_global,
        w_down,
        w_down_scales,
        w_down_global,
        output,
        num_tokens,
    })
}

// Launch once on a stream to prepare the kernels, then capture the complete CUDA graph.
#[derive(Clone, Copy)]
enum LaunchTarget<'a> {
    Stream(&'a Arc<Stream>),
    Graph(&'a cuda_async::cuda_graph::Scope),
}

fn launch(
    target: LaunchTarget<'_>,
    op: impl GraphNode,
) -> Result<(), cuda_async::error::DeviceError> {
    match target {
        LaunchTarget::Stream(stream) => {
            let _ = op.sync_on(stream)?;
        }
        LaunchTarget::Graph(scope) => {
            let _ = scope.record(op)?;
        }
    }
    Ok(())
}

fn run_routing(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let routed_size = tensors.num_tokens * TOP_K;
    if routed_size <= ROUTING_PAIRWISE_MAX_ROUTED {
        let op = unsafe {
            routing_group_pairwise_kernel(
                tensors.group_tokens.clone(),
                tensors.group_experts.clone(),
                tensors.group_limits.clone(),
                tensors.expanded_to_permuted.clone(),
                tensors.num_groups.clone(),
                tensors.expert_ids.clone(),
                TOP_K as i32,
                routed_size as i32,
                TILE_N as i32,
                TILE_N.trailing_zeros() as i32,
            )
        }
        .grid((1, 1, 1));
        launch(target, op)?;
    } else {
        let experts_pow2 = NUM_EXPERTS.next_power_of_two();
        let op = unsafe {
            routing_group_histogram_kernel(
                tensors.group_tokens.clone(),
                tensors.group_experts.clone(),
                tensors.group_limits.clone(),
                tensors.expanded_to_permuted.clone(),
                tensors.expert_counts.clone(),
                tensors.expert_offsets.clone(),
                tensors.num_groups.clone(),
                tensors.expert_ids.clone(),
                TOP_K as i32,
                NUM_EXPERTS as i32,
                routed_size as i32,
                TILE_N as i32,
                TILE_N.trailing_zeros() as i32,
            )
        }
        .generics(vec![
            (MAX_NUM_TOKENS * TOP_K).to_string(),
            experts_pow2.to_string(),
        ])
        .grid((1, 1, 1));
        launch(target, op)?;
    }
    Ok(())
}

fn run_quantize(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let op = unsafe {
        nvfp4_quantize_kernel(
            tensors.quantized_hidden.clone(),
            tensors.quantized_hidden_scales.clone(),
            tensors.hidden.clone(),
            INPUT_SCALE,
        )
    }
    .generics(vec![
        QUANTIZE_TILE.to_string(),
        (QUANTIZE_TILE / 2).to_string(),
        (QUANTIZE_TILE / 16).to_string(),
        (QUANTIZE_TILE / 16).to_string(),
        "16".to_string(),
        (QUANTIZE_TILE / 16).to_string(),
    ])
    .grid((
        tensors.num_tokens as u32,
        (HIDDEN / QUANTIZE_TILE) as u32,
        1,
    ));
    launch(target, op)
}

fn run_fc1(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let num_output_tiles = INTERMEDIATE / TILE_M;
    let worker_groups = max_num_groups(tensors.num_tokens * TOP_K, TILE_N);
    let op = unsafe {
        grouped_fc1_kernel(
            tensors.grouped_inter.clone(),
            tensors.grouped_inter_scales.clone(),
            tensors.quantized_hidden.clone(),
            tensors.quantized_hidden_scales.clone(),
            (HIDDEN / 2) as i32,
            (HIDDEN / 16) as i32,
            tensors.group_tokens.clone(),
            tensors.group_experts.clone(),
            tensors.group_limits.clone(),
            tensors.num_groups.clone(),
            tensors.w_up.clone(),
            tensors.w_up_scales.clone(),
            tensors.w_up_global.clone(),
            INPUT_SCALE,
            INTERMEDIATE as i32,
            (HIDDEN / TILE_K) as i32,
        )
    }
    .generics(vec![
        TILE_M.to_string(),
        (TILE_M / 32).to_string(),
        TILE_N.to_string(),
        TILE_K.to_string(),
        (TILE_K / 64).to_string(),
        (TILE_K / 2).to_string(),
        (TILE_K / 16).to_string(),
        (TILE_M / 2).to_string(),
        (TILE_M / 16).to_string(),
        "1".to_string(),
        "1".to_string(),
        TILE_N.to_string(),
        (TILE_M / 16).to_string(),
        TILE_N.to_string(),
        (TILE_M / 16).to_string(),
        "16".to_string(),
    ])
    .grid((num_output_tiles as u32, worker_groups as u32, 1));
    launch(target, op)
}

fn run_fc2(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let num_output_tiles = HIDDEN / TILE_M;
    let group_workers = max_num_groups(tensors.num_tokens * TOP_K, TILE_N);
    let op = unsafe {
        grouped_fc2_kernel(
            tensors.grouped_out.clone(),
            tensors.grouped_inter.clone(),
            tensors.grouped_inter_scales.clone(),
            tensors.group_experts.clone(),
            tensors.num_groups.clone(),
            tensors.w_down.clone(),
            tensors.w_down_scales.clone(),
            tensors.w_down_global.clone(),
            HIDDEN as i32,
            (INTERMEDIATE / TILE_K) as i32,
        )
    }
    .generics(vec![
        TILE_M.to_string(),
        (TILE_M / 32).to_string(),
        TILE_N.to_string(),
        TILE_K.to_string(),
        (TILE_K / 64).to_string(),
        (TILE_K / 2).to_string(),
        (TILE_K / 16).to_string(),
        (TILE_K / 64 * 512).to_string(),
        "1".to_string(),
        "1".to_string(),
        "1".to_string(),
    ])
    .grid((num_output_tiles as u32, group_workers as u32, 1));
    launch(target, op)
}

fn run_finalize(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let op = unsafe {
        finalize_kernel(
            tensors.output.clone(),
            tensors.grouped_out.clone(),
            tensors.expanded_to_permuted.clone(),
            tensors.routing_weights.clone(),
            TOP_K as i32,
            TILE_N as i32,
            TILE_N.trailing_zeros() as i32,
        )
    }
    .generics(vec![FINALIZE_TILE.to_string()])
    .grid((
        tensors.num_tokens as u32,
        (HIDDEN / FINALIZE_TILE) as u32,
        1,
    ));
    launch(target, op)
}

fn run_pipeline(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    run_routing(target, tensors)?;
    run_quantize(target, tensors)?;
    run_fc1(target, tensors)?;
    run_fc2(target, tensors)?;
    run_finalize(target, tensors)
}

fn check_output(
    tensors: &MoeTensors,
    stream: &Arc<Stream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = tensors.output.clone().to_host_vec().sync_on(stream)?;
    let mut max_abs_error = 0.0f32;
    let mut max_reference = 1.0f32;
    for (index, actual) in output.iter().enumerate() {
        let token = index / HIDDEN;
        let actual = actual.to_f32();
        if !actual.is_finite() {
            return Err(format!(
                "output check failed: non-finite output at index {index}: {actual}"
            )
            .into());
        }
        let reference = expected_output(token).to_f32();
        max_abs_error = max_abs_error.max((actual - reference).abs());
        max_reference = max_reference.max(reference.abs());
    }
    let normalized_error = max_abs_error / max_reference;
    if normalized_error > 0.01 {
        return Err(format!(
            "output check failed: normalized max error {normalized_error:.6}, max abs error {max_abs_error:.6}, max |reference| {max_reference:.6}"
        )
        .into());
    }
    println!(
        "Output check passed: normalized max error={normalized_error:.6}, max_abs_error={max_abs_error:.6}"
    );
    Ok(())
}

/// Mean time per graph replay in microseconds, excluding warmup, compilation,
/// allocation, and output checking. The events use the graph's launch stream.
fn time_graph(
    graph: &CudaGraph<()>,
    stream: &Arc<Stream>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start = stream.context().new_event()?;
    let stop = stream.context().new_event()?;
    let timing = (|| -> Result<f64, Box<dyn std::error::Error>> {
        // SAFETY: the graph retains its buffers and remains alive throughout
        // these same-stream launches. The stream is drained below on success
        // and error before the caller can release the graph or its tensors.
        for _ in 0..TIMING_WARMUPS {
            unsafe { graph.launch().async_on(stream)? };
        }
        start.record(stream)?;
        for _ in 0..TIMING_REPLAYS {
            unsafe { graph.launch().async_on(stream)? };
        }
        stop.record(stream)?;
        stop.synchronize()?;
        let latency_us = f64::from(start.elapsed_time(&stop)?) * 1000.0 / TIMING_REPLAYS as f64;
        if !latency_us.is_finite() || latency_us <= 0.0 {
            return Err("invalid CUDA event timing".into());
        }
        Ok(latency_us)
    })();
    // SAFETY: all work submitted here uses this live stream. Drain it even
    // when an enqueue or event operation fails after earlier submissions.
    unsafe { stream.synchronize()? };
    timing
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cutile_examples::requirements::BLOCK_SCALED_MMA.check("nvfp4_moe", 0)? {
        return Ok(());
    }

    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    println!("NVFP4 MoE batch-size benchmark");
    println!("H={HIDDEN}, I={INTERMEDIATE}, experts={NUM_EXPERTS}, top-k={TOP_K}, tile_n={TILE_N}");
    println!(
        "CUDA graph timing: {TIMING_SAMPLES} samples, {TIMING_REPLAYS} replays/sample, \
         {TIMING_WARMUPS} warmups/sample"
    );

    // Batches through four use pairwise routing; larger batches use histogram
    // routing. Sweep in order so a failure identifies the first failing size.
    let mut latencies = Vec::with_capacity(BATCH_SIZES.len());
    for num_tokens in BATCH_SIZES {
        println!("Batch {num_tokens}:");
        let tensors = allocate_tensors(num_tokens, &stream)?;
        // SAFETY: the fixture and launch geometry keep every unchecked access
        // within its allocation. Each stage owns its writes, shares read-only
        // inputs, and completes before the next stage consumes its outputs.
        // Tensors stay alive through graph replay and the final synchronization.
        run_pipeline(LaunchTarget::Stream(&stream), &tensors)?;
        println!("Batch {num_tokens}, direct launches:");
        check_output(&tensors, &stream)?;

        let graph = CudaGraph::scope(&stream, |scope| {
            run_pipeline(LaunchTarget::Graph(scope), &tensors)
        })?;
        graph.launch().sync_on(&stream)?;
        println!("Batch {num_tokens}, graph replay:");
        check_output(&tensors, &stream)?;

        let mut samples_us = Vec::with_capacity(TIMING_SAMPLES);
        for _ in 0..TIMING_SAMPLES {
            samples_us.push(time_graph(&graph, &stream)?);
        }
        check_output(&tensors, &stream)?;
        let mean_us = samples_us.iter().sum::<f64>() / TIMING_SAMPLES as f64;
        println!("Batch {num_tokens}: mean graph latency = {mean_us:.3} us");
        latencies.push((num_tokens, mean_us));
    }
    println!("\nBatch size | Mean graph latency (us)");
    for (batch, mean_us) in latencies {
        println!("{batch:>10} | {mean_us:>23.3}");
    }
    println!("NVFP4 MoE passed.");
    Ok(())
}
