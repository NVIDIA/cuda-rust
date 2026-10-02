/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use pliron::{builtin::ops::ModuleOp, context::Context, op::Op};
use pliron_mlir_export::{
    MlirAttribute, MlirBlock, MlirBlockId, MlirModule, MlirOperation, MlirRegion,
    TranslationConfig, TranslationError, TranslationRegistry,
    translate_module as translate_pliron_module,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    register_builtin_pack, register_cute_cutlass_profile_pack, register_cute_gemm_pack,
    register_cute_sm100_pack, register_mir_core_pack, register_nvvm_cluster_pack,
    register_nvvm_sreg_pack, register_nvvm_tcgen05_pack,
};

/// Everything that makes one textual MLIR contract reproducible.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportManifest {
    pub profile: String,
    pub exporter_version: String,
    pub pliron_revision: String,
    pub consumer_revision: String,
    pub llvm_major: u32,
    pub gpu_arch: String,
    pub pointer_model: String,
    pub index_bits: u32,
}

impl ExportManifest {
    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error(transparent)]
    Registry(#[from] TranslationError),
    #[error("CuTe source validation failed before MLIR translation: {0}")]
    CuteSemantics(String),
    #[error("could not build CUTLASS's executable GPU module envelope: {0}")]
    ExecutableEnvelope(String),
    #[error("unsupported GPU architecture `{0}` for this MLIR profile")]
    UnsupportedArchitecture(String),
}

/// A pinned target contract plus the mapping packs that implement it.
pub trait MlirConsumerProfile {
    fn name(&self) -> &'static str;
    fn manifest(&self) -> ExportManifest;
    fn build_registry(&self) -> Result<TranslationRegistry, ProfileError>;

    /// Check profile-specific source rules before any target text is built.
    fn verify_source_module(&self, ctx: &Context, module: &ModuleOp) -> Result<(), ProfileError>;

    fn translation_config(&self) -> TranslationConfig {
        TranslationConfig::new(self.name())
    }

    /// Validate the shared high-level module, then translate it.
    fn translate_module(
        &self,
        ctx: &Context,
        module: &ModuleOp,
    ) -> Result<MlirModule, ProfileError> {
        self.verify_source_module(ctx, module)?;
        let registry = self.build_registry()?;
        Ok(translate_pliron_module(
            ctx,
            module,
            &registry,
            &self.translation_config(),
        )?)
    }
}

/// CuTe mappings supported by the pinned CUTLASS 4.7 compiler.
///
/// Each CUTLASS release needs its own profile until its syntax, generated code,
/// and runtime behavior have passed validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutlassFullCuteMlir22 {
    gpu_arch: String,
}

impl CutlassFullCuteMlir22 {
    pub const CONSUMER_REVISION: &'static str = "cutlass-compiler-v4.7.0";
    pub const PLIRON_REVISION: &'static str = "06d99a3df938394583715aacf5eadb65b4791c09";

    pub fn new(gpu_arch: impl Into<String>) -> Result<Self, ProfileError> {
        let gpu_arch = gpu_arch.into();
        if !matches!(gpu_arch.as_str(), "sm_100a" | "sm_120a") {
            return Err(ProfileError::UnsupportedArchitecture(gpu_arch));
        }
        Ok(Self { gpu_arch })
    }
}

impl MlirConsumerProfile for CutlassFullCuteMlir22 {
    fn name(&self) -> &'static str {
        "cutlass-full-cute-mlir22"
    }

    fn manifest(&self) -> ExportManifest {
        ExportManifest {
            profile: self.name().into(),
            exporter_version: env!("CARGO_PKG_VERSION").into(),
            pliron_revision: Self::PLIRON_REVISION.into(),
            consumer_revision: Self::CONSUMER_REVISION.into(),
            llvm_major: 22,
            gpu_arch: self.gpu_arch.clone(),
            pointer_model: "opaque".into(),
            index_bits: 64,
        }
    }

    fn build_registry(&self) -> Result<TranslationRegistry, ProfileError> {
        let mut registry = TranslationRegistry::new();
        register_builtin_pack(&mut registry)?;
        register_mir_core_pack(&mut registry)?;
        register_nvvm_sreg_pack(&mut registry)?;
        register_nvvm_tcgen05_pack(&mut registry)?;
        register_nvvm_cluster_pack(&mut registry)?;
        register_cute_cutlass_profile_pack(&mut registry)?;
        register_cute_gemm_pack(&mut registry)?;
        register_cute_sm100_pack(&mut registry)?;
        registry.seal();
        Ok(registry)
    }

    fn verify_source_module(&self, ctx: &Context, module: &ModuleOp) -> Result<(), ProfileError> {
        // Both backends check the same CuTe operation graph and tensor origins
        // before generating code. Verification leaves the module unchanged.
        dialect_cute::verify::verify_cute_semantics(ctx, module.get_operation())
            .map_err(|error| ProfileError::CuteSemantics(error.to_string()))
    }

    fn translate_module(
        &self,
        ctx: &Context,
        module: &ModuleOp,
    ) -> Result<MlirModule, ProfileError> {
        self.verify_source_module(ctx, module)?;
        let registry = self.build_registry()?;
        let mut target =
            translate_pliron_module(ctx, module, &registry, &self.translation_config())?;
        add_cutlass_executable_envelope(&mut target).map_err(ProfileError::ExecutableEnvelope)?;
        Ok(target)
    }
}

/// CUTLASS's binary pass only serializes `gpu.module` operations. Keeping device
/// functions at the top level is syntactically valid but makes the compiler
/// return success without producing PTX or a cubin, so the profile always
/// builds the executable envelope itself.
fn add_cutlass_executable_envelope(module: &mut MlirModule) -> Result<(), String> {
    if module.root.name != "builtin.module" {
        return Err(format!(
            "expected builtin.module root, got {}",
            module.root.name
        ));
    }
    if module.root.regions.len() != 1 || module.root.regions[0].blocks.len() != 1 {
        return Err("builtin.module must contain exactly one top-level block".into());
    }

    let gpu_block_id = next_block_id(&module.root)?;
    let root_block = &mut module.root.regions[0].blocks[0];
    let device_operations = std::mem::take(&mut root_block.operations);

    let mut gpu_module = MlirOperation::new("gpu.module")?;
    gpu_module
        .properties
        .insert("sym_name".into(), MlirAttribute::String("kernels".into()));
    gpu_module.regions.push(MlirRegion {
        blocks: vec![MlirBlock {
            id: gpu_block_id,
            arguments: vec![],
            operations: device_operations,
        }],
    });
    gpu_module.location = module.root.location.clone();

    // The source module name is useful before translation, while CUTLASS's runtime
    // contract names the serialized device module `kernels`.
    module.root.properties.remove("sym_name");
    module
        .root
        .attributes
        .insert("gpu.container_module".into(), MlirAttribute::Unit);
    root_block.operations.push(gpu_module);
    Ok(())
}

fn next_block_id(operation: &MlirOperation) -> Result<MlirBlockId, String> {
    fn visit(operation: &MlirOperation, maximum: &mut u64) {
        for region in &operation.regions {
            for block in &region.blocks {
                *maximum = (*maximum).max(block.id.0);
                for operation in &block.operations {
                    visit(operation, maximum);
                }
            }
        }
    }

    let mut maximum = 0;
    visit(operation, &mut maximum);
    maximum
        .checked_add(1)
        .map(MlirBlockId)
        .ok_or_else(|| "MLIR block id space is exhausted".into())
}

/// Keep the operation-mapping goldens focused on their mapping pack. The
/// executable envelope has its own structural tests below and in the
/// production-shaped integration fixture.
#[cfg(test)]
pub(crate) fn render_mapping_module_without_cutlass_envelope(
    module: &MlirModule,
    source_name: &str,
) -> String {
    let mut unwrapped = module.clone();
    let gpu_module = &module.root.regions[0].blocks[0].operations[0];
    assert_eq!(gpu_module.name, "gpu.module");
    unwrapped.root.regions[0].blocks[0].operations =
        gpu_module.regions[0].blocks[0].operations.clone();
    unwrapped.root.attributes.remove("gpu.container_module");
    unwrapped
        .root
        .properties
        .insert("sym_name".into(), MlirAttribute::String(source_name.into()));
    pliron_mlir_export::render_module(&unwrapped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::identifier::Identifier;

    #[test]
    fn profile_is_exact_and_machine_readable() {
        let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
        let json = profile.manifest().to_pretty_json().unwrap();

        assert!(json.contains(CutlassFullCuteMlir22::CONSUMER_REVISION));
        assert!(json.contains("\"llvm_major\": 22"));
        assert!(profile.build_registry().unwrap().is_sealed());
    }

    #[test]
    fn profile_rejects_an_unproved_architecture() {
        assert!(matches!(
            CutlassFullCuteMlir22::new("sm_90a"),
            Err(ProfileError::UnsupportedArchitecture(arch)) if arch == "sm_90a"
        ));
    }

    #[test]
    fn profile_verifies_the_live_module_through_an_immutable_context() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("immutable_verify").unwrap());

        let immutable_ctx = &ctx;
        CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .verify_source_module(immutable_ctx, &module)
            .unwrap();
    }
}
