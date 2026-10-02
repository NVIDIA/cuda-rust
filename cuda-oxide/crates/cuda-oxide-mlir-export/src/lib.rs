/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA Oxide mappings for textual MLIR export.
//!
//! The generic exporter builds typed MLIR text. This crate maps CUDA Oxide
//! operations to the types and operations accepted by the pinned CUTLASS compiler.

mod cute;
mod cute_gemm;
mod cute_gemv;
mod cute_sm100;
mod grid_constant;
mod mir_core;
mod mir_memory;
mod nvvm_cluster;
mod nvvm_sregs;
mod nvvm_tcgen05;
mod packs;
mod profile;

pub use cute::register_cute_elementwise_pack;
pub(crate) use cute_gemm::register_cute_gemm_pack;
pub(crate) use cute_gemv::register_cute_cutlass_profile_pack;
pub(crate) use cute_sm100::register_cute_sm100_pack;
pub use mir_core::register_mir_core_pack;
pub use nvvm_cluster::register_nvvm_cluster_pack;
pub use nvvm_sregs::register_nvvm_sreg_pack;
pub use nvvm_tcgen05::register_nvvm_tcgen05_pack;
pub use packs::register_builtin_pack;
pub use profile::{CutlassFullCuteMlir22, ExportManifest, MlirConsumerProfile, ProfileError};
