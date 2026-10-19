/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::CatalogFile;
use crate::render::common::rust_header;
use crate::render::families::{
    clc_intrinsics, cluster_memory, debug_controls, elect_intrinsics, execution_controls,
    extended_minmax, integer_minmaxes, mbarrier_extended, movmatrix, scalar_arithmetics,
    scalar_conversions, scalar_maths, stmatrices, tcgen05_intrinsics, tma_intrinsics,
    wgmma_controls,
};

mod cluster_tensor;
mod mma;
mod scalar_packed;
mod warp_sync;

pub(super) use cluster_tensor::*;
pub(super) use mma::*;
pub(super) use scalar_packed::*;
pub(super) use warp_sync::*;

pub(super) fn render_dialect_mod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "mod active_mask;\nmod cluster_barrier;\nmod cp_async;\nmod dotprod;\nmod ldmatrix;\nmod mbarrier_basic;\nmod movmatrix;\nmod packed_alu;\nmod packed_atomic;\nmod packed_conversion;\nmod prmt;\nmod redux;\nmod register_mma;\nmod sparse_mma;\nmod sreg;\nmod sync;\nmod vote;\nmod warp_barrier;\nmod warp_match;\nmod warp_shuffle;\n\npub use active_mask::*;\npub use cluster_barrier::*;\npub use cp_async::*;\npub use dotprod::*;\npub use ldmatrix::*;\npub use mbarrier_basic::*;\npub use movmatrix::*;\npub use packed_alu::*;\npub use packed_atomic::*;\npub use packed_conversion::*;\npub use prmt::*;\npub use redux::*;\npub use register_mma::*;\npub use sparse_mma::*;\npub use sreg::*;\npub use sync::*;\npub use vote::*;\npub use warp_barrier::*;\npub use warp_match::*;\npub use warp_shuffle::*;\n\nuse pliron::context::Context;\n\npub(super) fn register(ctx: &mut Context) {\n    active_mask::register(ctx);\n    cluster_barrier::register(ctx);\n    cp_async::register(ctx);\n    dotprod::register(ctx);\n    ldmatrix::register(ctx);\n    mbarrier_basic::register(ctx);\n    movmatrix::register(ctx);\n    packed_alu::register(ctx);\n    packed_atomic::register(ctx);\n    packed_conversion::register(ctx);\n    prmt::register(ctx);\n    redux::register(ctx);\n    register_mma::register(ctx);\n    sparse_mma::register(ctx);\n    sreg::register(ctx);\n    sync::register(ctx);\n    vote::register(ctx);\n    warp_barrier::register(ctx);\n    warp_match::register(ctx);\n    warp_shuffle::register(ctx);\n}\n",
    );
    if scalar_conversions(catalog).next().is_some() {
        output = output
            .replace("mod redux;", "mod redux;\nmod scalar_conversion;")
            .replace(
                "pub use redux::*;",
                "pub use redux::*;\npub use scalar_conversion::*;",
            )
            .replace(
                "    redux::register(ctx);",
                "    redux::register(ctx);\n    scalar_conversion::register(ctx);",
            );
    }
    if scalar_arithmetics(catalog).next().is_some() {
        output = output
            .replace("mod redux;", "mod redux;\nmod scalar_arithmetic;")
            .replace(
                "pub use redux::*;",
                "pub use redux::*;\npub use scalar_arithmetic::*;",
            )
            .replace(
                "    redux::register(ctx);",
                "    redux::register(ctx);\n    scalar_arithmetic::register(ctx);",
            );
    }
    if scalar_maths(catalog).next().is_some() {
        output = output
            .replace("mod redux;", "mod redux;\nmod scalar_math;")
            .replace(
                "pub use redux::*;",
                "pub use redux::*;\npub use scalar_math::*;",
            )
            .replace(
                "    redux::register(ctx);",
                "    redux::register(ctx);\n    scalar_math::register(ctx);",
            );
    }
    if integer_minmaxes(catalog).next().is_some() {
        output = output
            .replace("mod ldmatrix;", "mod integer_minmax;\nmod ldmatrix;")
            .replace(
                "pub use ldmatrix::*;",
                "pub use integer_minmax::*;\npub use ldmatrix::*;",
            )
            .replace(
                "    ldmatrix::register(ctx);",
                "    integer_minmax::register(ctx);\n    ldmatrix::register(ctx);",
            );
    }
    if extended_minmax(catalog).next().is_some() {
        output = output
            .replace("mod dotprod;", "mod dotprod;\nmod extended_minmax;")
            .replace(
                "pub use dotprod::*;",
                "pub use dotprod::*;\npub use extended_minmax::*;",
            )
            .replace(
                "    dotprod::register(ctx);",
                "    dotprod::register(ctx);\n    extended_minmax::register(ctx);",
            );
    }
    if debug_controls(catalog).next().is_some() {
        output = output
            .replace("mod dotprod;", "mod debug_control;\nmod dotprod;")
            .replace(
                "pub use dotprod::*;",
                "pub use debug_control::*;\npub use dotprod::*;",
            )
            .replace(
                "    dotprod::register(ctx);",
                "    debug_control::register(ctx);\n    dotprod::register(ctx);",
            );
    }
    if cluster_memory(catalog).next().is_some() {
        output = output
            .replace(
                "mod cluster_barrier;",
                "mod cluster_barrier;\nmod cluster_memory;",
            )
            .replace(
                "pub use cluster_barrier::*;",
                "pub use cluster_barrier::*;\npub use cluster_memory::*;",
            )
            .replace(
                "    cluster_barrier::register(ctx);",
                "    cluster_barrier::register(ctx);\n    cluster_memory::register(ctx);",
            );
    }
    if stmatrices(catalog).next().is_some() {
        output = output
            .replace("mod sync;", "mod stmatrix;\nmod sync;")
            .replace("pub use sync::*;", "pub use stmatrix::*;\npub use sync::*;")
            .replace(
                "    sync::register(ctx);",
                "    stmatrix::register(ctx);\n    sync::register(ctx);",
            );
    }
    if movmatrix(catalog).next().is_none() {
        output = output.replace("mod movmatrix;\n", "");
        output = output.replace("pub use movmatrix::*;\n", "");
        output = output.replace("    movmatrix::register(ctx);\n", "");
    }
    if clc_intrinsics(catalog).next().is_some() {
        output = output
            .replace("mod cluster_barrier;", "mod clc;\nmod cluster_barrier;")
            .replace(
                "pub use cluster_barrier::*;",
                "pub use clc::*;\npub use cluster_barrier::*;",
            )
            .replace(
                "    cluster_barrier::register(ctx);",
                "    clc::register(ctx);\n    cluster_barrier::register(ctx);",
            );
    }
    if elect_intrinsics(catalog).next().is_some() {
        output = output
            .replace("mod dotprod;", "mod dotprod;\nmod elect;")
            .replace(
                "pub use dotprod::*;",
                "pub use dotprod::*;\npub use elect::*;",
            )
            .replace(
                "    dotprod::register(ctx);",
                "    dotprod::register(ctx);\n    elect::register(ctx);",
            );
    }
    if mbarrier_extended(catalog).next().is_some() {
        output = output.replace(
            "mod mbarrier_basic;\n",
            "mod mbarrier_basic;\nmod mbarrier_extended;\n",
        );
        output = output.replace(
            "pub use mbarrier_basic::*;\n",
            "pub use mbarrier_basic::*;\npub use mbarrier_extended::*;\n",
        );
        output = output.replace(
            "    mbarrier_basic::register(ctx);\n",
            "    mbarrier_basic::register(ctx);\n    mbarrier_extended::register(ctx);\n",
        );
    }
    if wgmma_controls(catalog).next().is_some() {
        output = output
            .replace("mod warp_shuffle;", "mod warp_shuffle;\nmod wgmma_control;")
            .replace(
                "pub use warp_shuffle::*;",
                "pub use warp_shuffle::*;\npub use wgmma_control::*;",
            )
            .replace(
                "    warp_shuffle::register(ctx);",
                "    warp_shuffle::register(ctx);\n    wgmma_control::register(ctx);",
            );
    }
    if execution_controls(catalog).next().is_some() {
        output = output
            .replace("mod dotprod;", "mod dotprod;\nmod execution_control;")
            .replace(
                "pub use dotprod::*;",
                "pub use dotprod::*;\npub use execution_control::*;",
            )
            .replace(
                "    dotprod::register(ctx);",
                "    dotprod::register(ctx);\n    execution_control::register(ctx);",
            );
    }
    if tma_intrinsics(catalog).next().is_some() {
        output = output
            .replace("mod sync;", "mod sync;\nmod tma;")
            .replace("pub use sync::*;", "pub use sync::*;\npub use tma::*;")
            .replace(
                "    sync::register(ctx);",
                "    sync::register(ctx);\n    tma::register(ctx);",
            );
    }
    if tcgen05_intrinsics(catalog).next().is_some() {
        output = output
            .replace("mod sync;", "mod sync;\nmod tcgen05;")
            .replace("pub use sync::*;", "pub use sync::*;\npub use tcgen05::*;")
            .replace(
                "    sync::register(ctx);",
                "    sync::register(ctx);\n    tcgen05::register(ctx);",
            );
    }
    output
}
