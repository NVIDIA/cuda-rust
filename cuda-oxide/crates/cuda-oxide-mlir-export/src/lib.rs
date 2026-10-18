/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuda-oxide mappings for textual MLIR export.
//!
//! This crate maps cuda-oxide operations to the types and operations accepted
//! by the pinned CUTLASS compiler. Upstream pliron-mlir-interop renders the
//! resulting target graph with its generic MLIR printer.
//!
//! Use a consumer profile to export a prepared Pliron context and module:
//!
//! ```no_run
//! use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
//! # use pliron::{builtin::ops::ModuleOp, context::Context};
//! # fn export(ctx: &Context, module: &ModuleOp) -> Result<(), cuda_oxide_mlir_export::ProfileError> {
//! let profile = CutlassFullCuteMlir22::new("sm_120a")?;
//! let target = profile.translate_module(ctx, module)?;
//! let text = cuda_oxide_mlir_export::render_module(&target)?;
//! # Ok(())
//! # }
//! ```
//!
//! Both concrete profiles verify the source and wrap device functions in the
//! `gpu.module` required to produce a cubin. Registering mapping packs alone
//! does not perform those steps. Exporting text needs no GPU or CUTLASS library.

mod cute;
mod grid_constant;
mod interop;
mod mir;
mod nvvm;
mod packs;
mod primitives;
mod profile;
mod translation;

pub use interop::{INTEROP_REVISION, render_module};
pub use translation::*;

pub use cute::register_cute_elementwise_pack;
pub(crate) use cute::{
    register_cute_attention_pack, register_cute_cutlass_profile_pack, register_cute_gemm_pack,
    register_cute_sm100_pack,
};
pub use mir::core::register_mir_core_pack;
pub use nvvm::{
    attention::register_nvvm_attention_pack, cluster::register_nvvm_cluster_pack,
    numeric::register_nvvm_numeric_pack, sm120::register_nvvm_sm120_pack,
    sregs::register_nvvm_sreg_pack, sync_restrict::register_nvvm_sync_restrict_pack,
    tcgen05::register_nvvm_tcgen05_pack,
};
pub use packs::register_builtin_pack;
pub use primitives::{CutlassPrimitivesMlir22, verify_primitives_source};
pub use profile::{CutlassFullCuteMlir22, ExportManifest, MlirConsumerProfile, ProfileError};
