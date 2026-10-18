/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pinned consumer contracts, source validation, and executable GPU module packaging.

use crate::{
    MlirAttribute, MlirBlock, MlirBlockId, MlirModule, MlirOperation, MlirRegion,
    TranslationConfig, TranslationError, TranslationRegistry,
    translate_module as translate_pliron_module,
};
use pliron::{
    builtin::ops::ModuleOp, context::Context, linked_list::ContainsLinkedList, op::Op,
    operation::Operation,
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
    /// Stable consumer-profile identity used to distinguish mapping contracts.
    pub profile: String,
    /// Version of the cuda-oxide MLIR exporter that produced the artifact.
    pub exporter_version: String,
    /// Exact Pliron revision whose source IR contract was translated.
    pub pliron_revision: String,
    /// Pinned upstream printer revision and local generic extension version.
    pub interop_revision: String,
    /// Pinned CUTLASS compiler release expected to accept this target MLIR.
    pub consumer_revision: String,
    /// LLVM/MLIR major version defining the target dialect syntax.
    pub llvm_major: u32,
    /// Validated GPU architecture used for source capability checks and compilation.
    pub gpu_arch: String,
    /// Target pointer representation, currently opaque LLVM pointers.
    pub pointer_model: String,
    /// Bit width used for target index values.
    pub index_bits: u32,
}

impl ExportManifest {
    /// Serialize the complete reproducibility manifest as indented JSON.
    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Failure to validate, register, or package a source module for the selected consumer.
#[derive(Debug, Error)]
pub enum ProfileError {
    /// A mapping registration or generic translation failure.
    #[error(transparent)]
    Registry(
        #[doc = "Underlying registration or generic translation diagnostic."]
        #[from]
        TranslationError,
    ),
    /// The shared high-level CuTe verifier rejected the source module.
    #[error("CuTe source validation failed before MLIR translation: {0}")]
    CuteSemantics(#[doc = "Diagnostic from the shared CuTe source verifier."] String),
    /// The selected profile rejected a source instruction or metadata contract.
    #[error("primitives source validation failed: {0}")]
    SourceValidation(#[doc = "Reason the source violates the selected profile's contract."] String),
    /// The translated module could not be packaged for CUTLASS binary emission.
    #[error("could not build CUTLASS's executable GPU module envelope: {0}")]
    ExecutableEnvelope(
        #[doc = "Reason a valid executable module envelope could not be constructed."] String,
    ),
    /// The requested target architecture has no reviewed profile.
    #[error("unsupported GPU architecture `{0}` for this MLIR profile")]
    UnsupportedArchitecture(
        #[doc = "Unrecognized or unreviewed target architecture spelling."] String,
    ),
}

/// A pinned target contract plus the mapping packs that implement it.
pub trait MlirConsumerProfile {
    /// Return the stable profile identity used in diagnostics and cache manifests.
    fn name(&self) -> &'static str;
    /// Describe the exact exporter, source IR, consumer, and target contract.
    fn manifest(&self) -> ExportManifest;
    /// Register the profile's reviewed mappings exactly once and seal the resulting registry.
    fn build_registry(&self) -> Result<TranslationRegistry, ProfileError>;

    /// Check profile-specific source rules before any target text is built.
    fn verify_source_module(&self, ctx: &Context, module: &ModuleOp) -> Result<(), ProfileError>;

    /// Configure generic translation diagnostics with this profile's identity.
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
    /// Validated GPU architecture used for source capability checks and compilation.
    gpu_arch: String,
}

impl CutlassFullCuteMlir22 {
    /// Exact CUTLASS compiler release defining this profile's accepted dialects.
    pub const CONSUMER_REVISION: &'static str = "cutlass-compiler-v4.7.0";
    /// Exact Pliron revision defining this profile's source IR contract.
    pub const PLIRON_REVISION: &'static str = "435536c8611af34c48c8f73d78e77894a05d103d";

    /// Accept only sm_100a or sm_120a, whose mappings have been validated for this consumer.
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
            interop_revision: crate::INTEROP_REVISION.into(),
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
        crate::register_nvvm_sm120_pack(&mut registry)?;
        crate::register_nvvm_sync_restrict_pack(&mut registry)?;
        crate::nvvm::numeric::register_nvvm_profile_numeric_pack(&mut registry, &self.gpu_arch)?;
        register_cute_cutlass_profile_pack(&mut registry)?;
        register_cute_gemm_pack(&mut registry)?;
        register_cute_sm100_pack(&mut registry)?;
        crate::register_cute_attention_pack(&mut registry)?;
        crate::cute::register_cute_sm100_attention_pack(&mut registry)?;
        registry.seal();
        Ok(registry)
    }

    fn verify_source_module(&self, ctx: &Context, module: &ModuleOp) -> Result<(), ProfileError> {
        // Both backends check the same CuTe operation graph and tensor origins
        // before generating code. Verification leaves the module unchanged.
        dialect_cute::verify::verify_cute_semantics(ctx, module.get_operation())
            .map_err(|error| ProfileError::CuteSemantics(error.to_string()))?;
        verify_nvvm_architecture(ctx, module, &self.gpu_arch)
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

/// Raw NVVM intrinsics bypass CuTe atom selection, so both profiles must check
/// their architecture requirements before translating any operation.
pub(crate) fn verify_nvvm_architecture(
    ctx: &Context,
    module: &ModuleOp,
    gpu_arch: &str,
) -> Result<(), ProfileError> {
    let mut pending = vec![module.get_operation()];
    while let Some(ptr) = pending.pop() {
        let id = Operation::get_opid(ptr, ctx);
        if gpu_arch == "sm_120a" && id.to_string().starts_with("nvvm.tcgen05") {
            return Err(ProfileError::SourceValidation(
                "TCGen05 is unsupported on sm_120a".into(),
            ));
        }
        if gpu_arch != "sm_120a"
            && id == dialect_nvvm::ops::RegisterMmaOp::get_opid_static()
            && dialect_nvvm::ops::RegisterMmaOp::new(ptr)
                .get_attr_nvvm_register_mma_kind(ctx)
                .as_deref()
                == Some(&dialect_nvvm::ops::RegisterMmaKindAttr::Mxf4)
        {
            return Err(ProfileError::SourceValidation(
                "MXF4 register MMA requires sm_120a".into(),
            ));
        }
        for region in ptr.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    Ok(())
}

/// CUTLASS's binary pass only serializes `gpu.module` operations. Keeping device
/// functions at the top level is syntactically valid but makes the compiler
/// return success without producing PTX or a cubin, so the profile always
/// builds the executable envelope itself.
pub(crate) fn add_cutlass_executable_envelope(module: &mut MlirModule) -> Result<(), String> {
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
    let mut device_operations = std::mem::take(&mut root_block.operations);
    crate::mir::shared::hoist_shared_globals(&mut device_operations)?;

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

/// Choose a fresh block identifier beyond every block already present in the module.
fn next_block_id(operation: &MlirOperation) -> Result<MlirBlockId, String> {
    /// Walk nested regions to update the largest existing block identifier.
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
    crate::render_module(&unwrapped).unwrap()
}

/// Compare mapping goldens modulo SSA and block spelling, retaining each binding's identity.
#[cfg(test)]
pub(crate) fn normalize_mlir_identifiers(text: &str) -> String {
    use std::collections::HashMap;

    let mut values = HashMap::new();
    let mut blocks = HashMap::new();
    let mut normalized = String::new();
    let mut characters = text.chars().peekable();
    let mut quoted = false;
    let mut escaped = false;
    while let Some(character) = characters.next() {
        if quoted {
            normalized.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
        } else if character == '"' {
            quoted = true;
            normalized.push(character);
        } else if matches!(character, '%' | '^') {
            let mut name = String::new();
            while let Some(next) = characters.peek().copied() {
                if !next.is_ascii_alphanumeric() && !matches!(next, '_' | '.' | '$' | '-') {
                    break;
                }
                name.push(characters.next().unwrap());
            }
            if name.is_empty() {
                normalized.push(character);
                continue;
            }
            let names = if character == '%' {
                &mut values
            } else {
                &mut blocks
            };
            let next = names.len();
            let index = names.entry(name).or_insert(next);
            normalized.push(character);
            normalized.push_str(&index.to_string());
        } else {
            normalized.push(character);
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    //! Profile manifest, architecture rejection, and immutable-source verification checks.

    use super::*;
    use pliron::identifier::Identifier;

    /// Alpha-renaming must not erase changed operands, block references, or string contents.
    #[test]
    fn golden_normalization_preserves_binding_identity_and_strings() {
        let original = "%x = op(%a, %a) [^x, ^next] {note = \"%x ^next\"}";
        let renamed = "%r = op(%input, %input) [^entry, ^exit] {note = \"%x ^next\"}";
        assert_eq!(
            normalize_mlir_identifiers(original),
            normalize_mlir_identifiers(renamed)
        );
        for changed in [
            "%r = op(%input, %other) [^entry, ^exit] {note = \"%x ^next\"}",
            "%r = op(%input, %input) [^entry, ^entry] {note = \"%x ^next\"}",
            "%r = op(%input, %input) [^entry, ^exit] {note = \"%r ^exit\"}",
        ] {
            assert_ne!(
                normalize_mlir_identifiers(original),
                normalize_mlir_identifiers(changed)
            );
        }
    }

    /// Check manifest JSON records the pinned consumer/LLVM contract and the registry is sealed.
    #[test]
    fn profile_is_exact_and_machine_readable() {
        let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
        let json = profile.manifest().to_pretty_json().unwrap();

        assert!(json.contains(CutlassFullCuteMlir22::CONSUMER_REVISION));
        assert!(json.contains(CutlassFullCuteMlir22::PLIRON_REVISION));
        assert!(json.contains(crate::INTEROP_REVISION));
        assert!(json.contains("\"llvm_major\": 22"));
        assert!(profile.build_registry().unwrap().is_sealed());
    }

    /// Reject a target without a reviewed CUTLASS mapping profile.
    #[test]
    fn profile_rejects_an_unproved_architecture() {
        assert!(matches!(
            CutlassFullCuteMlir22::new("sm_90a"),
            Err(ProfileError::UnsupportedArchitecture(arch)) if arch == "sm_90a"
        ));
    }

    /// Check whole-module verification accepts immutable access without rewriting source IR.
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
