/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generated-intrinsic metadata collected before MIR lowering erases it.

use crate::error::PipelineError;
use crate::generated_intrinsic_targets::{
    GENERATED_INTRINSIC_MARKER_ATTR, GeneratedIntrinsicBackend, GeneratedIntrinsicTarget,
    GeneratedIntrinsicVariant, GeneratedTargetContract, GeneratedTargetRequirement,
    GeneratedTargetSelectorBinding, GeneratedTcgen05MmaTargetSelector,
    generated_intrinsic_operation_matches, generated_intrinsic_target_by_marker,
    generated_intrinsic_target_is_direct_dialect_candidate, generated_intrinsic_targets_by_op_name,
};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::printable::Printable;
use std::collections::BTreeMap;

type GeneratedTargetsByMarker = BTreeMap<&'static str, &'static GeneratedIntrinsicTarget>;
type GeneratedResolvedTargetKey = (&'static str, Option<(&'static str, &'static str)>);
type GeneratedResolvedTargets = BTreeMap<GeneratedResolvedTargetKey, GeneratedResolvedTarget>;

/// Whether generated dialect operations must carry their Rust-source ABI marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GeneratedMarkerPolicy {
    /// The rustc frontend must preserve the exact source ABI marker.
    Required,
    /// Direct dialect frontends may select the unique catalog variant structurally.
    Optional,
}

/// Exact generated-intrinsic requirements found in typed, pre-lowering IR.
///
/// `targets` keeps one entry per ABI marker. Selector-dependent calls are also
/// retained separately by marker and exact selector tuple.
#[derive(Debug, Clone)]
pub(crate) struct GeneratedModuleRequirements {
    pub(crate) targets: Vec<&'static GeneratedIntrinsicTarget>,
    resolved_targets: Vec<GeneratedResolvedTarget>,
    backend: GeneratedIntrinsicBackend,
}

/// One exact target contract retained from a typed intrinsic call.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedResolvedTarget {
    pub(crate) target: &'static GeneratedIntrinsicTarget,
    pub(crate) selector: Option<GeneratedTargetSelectorBinding>,
    contract: Option<&'static GeneratedTargetContract>,
}

/// The target requirement selected for one retained call variant.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GeneratedResolvedRequirement {
    Target(GeneratedTargetRequirement),
    Contract(&'static GeneratedTargetContract),
}

impl Default for GeneratedModuleRequirements {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            resolved_targets: Vec::new(),
            backend: GeneratedIntrinsicBackend::LlvmNvptx,
        }
    }
}

impl GeneratedModuleRequirements {
    pub(crate) fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn for_backend(mut self, backend: GeneratedIntrinsicBackend) -> Self {
        self.backend = backend;
        for resolved in &mut self.resolved_targets {
            resolved.contract = resolved.selector.and_then(|selector| {
                resolved.target.target_contract_for_backend_selector(
                    backend,
                    selector.name,
                    selector.value,
                )
            });
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn requirement(
        &self,
        target: &GeneratedIntrinsicTarget,
    ) -> GeneratedTargetRequirement {
        target.requirement_for_backend(self.backend)
    }

    pub(crate) fn resolved_targets(&self) -> &[GeneratedResolvedTarget] {
        &self.resolved_targets
    }

    pub(crate) fn resolved_requirement(
        &self,
        resolved: &GeneratedResolvedTarget,
    ) -> Option<GeneratedResolvedRequirement> {
        match resolved.selector {
            Some(_) => resolved
                .contract
                .map(GeneratedResolvedRequirement::Contract),
            None => Some(GeneratedResolvedRequirement::Target(
                resolved.target.requirement_for_backend(self.backend),
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_targets(targets: Vec<&'static GeneratedIntrinsicTarget>) -> Self {
        let resolved_targets = targets
            .iter()
            .copied()
            .map(|target| GeneratedResolvedTarget {
                target,
                selector: None,
                contract: None,
            })
            .collect();
        Self {
            targets,
            resolved_targets,
            ..Self::default()
        }
    }
}

/// Collect generated-intrinsic requirements before lowering erases typed
/// operations and their compiler-only source ABI markers.
#[cfg(test)]
pub(crate) fn collect_generated_intrinsic_requirements(
    ctx: &Context,
    root: Ptr<Operation>,
    marker_policy: GeneratedMarkerPolicy,
) -> Result<GeneratedModuleRequirements, PipelineError> {
    collect_generated_intrinsic_requirements_for_backend(
        ctx,
        root,
        marker_policy,
        GeneratedIntrinsicBackend::LlvmNvptx,
    )
}

/// Collect requirements for the backend selected before MIR lowering.
pub(crate) fn collect_generated_intrinsic_requirements_for_backend(
    ctx: &Context,
    root: Ptr<Operation>,
    marker_policy: GeneratedMarkerPolicy,
    backend: GeneratedIntrinsicBackend,
) -> Result<GeneratedModuleRequirements, PipelineError> {
    use pliron::identifier::Identifier;
    let marker_key = Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).map_err(|error| {
        PipelineError::Verification {
            name: "generated intrinsic target requirements".to_string(),
            message: format!("invalid generated-intrinsic marker key: {error}"),
            operation: None,
        }
    })?;
    let mut targets = BTreeMap::new();
    let mut resolved_targets = BTreeMap::new();

    fn visit(
        ctx: &Context,
        op_ptr: Ptr<Operation>,
        marker_key: &pliron::identifier::Identifier,
        marker_policy: GeneratedMarkerPolicy,
        backend: GeneratedIntrinsicBackend,
        targets: &mut GeneratedTargetsByMarker,
        resolved_targets: &mut GeneratedResolvedTargets,
    ) -> Result<(), PipelineError> {
        use pliron::builtin::attributes::StringAttr;

        let op_name = Operation::get_opid(op_ptr, ctx).to_string();
        let op_ref = op_ptr.deref(ctx);
        let marker_attribute_exists = op_ref.attributes.0.contains_key(marker_key);
        let marker = op_ref
            .attributes
            .get::<StringAttr>(marker_key)
            .cloned()
            .map(String::from);
        let candidates = generated_intrinsic_targets_by_op_name(&op_name).collect::<Vec<_>>();

        match marker {
            Some(marker) => {
                let target = generated_intrinsic_target_by_marker(&marker).ok_or_else(|| {
                    generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "operation `{op_name}` carries unknown generated-intrinsic marker `{marker}`"
                        ),
                    )
                })?;
                if !candidates.contains(&target) {
                    return Err(generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "generated-intrinsic marker `{marker}` belongs to `{}`, not `{op_name}`",
                            target.dialect_op
                        ),
                    ));
                }
                if !generated_intrinsic_operation_matches(ctx, target, op_ptr) {
                    return Err(generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "generated-intrinsic marker `{marker}` does not match the exact variant attributes on `{op_name}`"
                        ),
                    ));
                }
                retain_generated_target(ctx, op_ptr, backend, target, targets, resolved_targets)?;
            }
            None if marker_attribute_exists => {
                return Err(generated_requirement_error(
                    ctx,
                    op_ptr,
                    format!(
                        "operation `{op_name}` has a non-string `{GENERATED_INTRINSIC_MARKER_ATTR}` attribute"
                    ),
                ));
            }
            None if !candidates.is_empty() && marker_policy == GeneratedMarkerPolicy::Required => {
                return Err(generated_requirement_error(
                    ctx,
                    op_ptr,
                    format!(
                        "generated intrinsic operation `{op_name}` is missing its exact ABI marker"
                    ),
                ));
            }
            None if !candidates.is_empty() => {
                let matching = candidates
                    .into_iter()
                    .filter(|target| generated_intrinsic_target_is_direct_dialect_candidate(target))
                    .filter(|target| generated_intrinsic_operation_matches(ctx, target, op_ptr))
                    .collect::<Vec<_>>();
                let [target] = matching.as_slice() else {
                    return Err(generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "direct dialect operation `{op_name}` matches {} generated catalog variants; expected exactly one",
                            matching.len()
                        ),
                    ));
                };
                retain_generated_target(ctx, op_ptr, backend, target, targets, resolved_targets)?;
            }
            None => {}
        }

        for region in op_ref.regions() {
            let region_ref = region.deref(ctx);
            for block in region_ref.iter(ctx) {
                let block_ref = block.deref(ctx);
                for child_op in block_ref.iter(ctx) {
                    visit(
                        ctx,
                        child_op,
                        marker_key,
                        marker_policy,
                        backend,
                        targets,
                        resolved_targets,
                    )?;
                }
            }
        }
        Ok(())
    }

    visit(
        ctx,
        root,
        &marker_key,
        marker_policy,
        backend,
        &mut targets,
        &mut resolved_targets,
    )?;
    Ok(GeneratedModuleRequirements {
        targets: targets.into_values().collect(),
        resolved_targets: resolved_targets.into_values().collect(),
        backend,
    })
}

fn retain_generated_target(
    ctx: &Context,
    op: Ptr<Operation>,
    backend: GeneratedIntrinsicBackend,
    target: &'static GeneratedIntrinsicTarget,
    targets: &mut GeneratedTargetsByMarker,
    resolved_targets: &mut GeneratedResolvedTargets,
) -> Result<(), PipelineError> {
    let selector = generated_target_selector(ctx, target, op)?;
    let contract = match selector {
        Some(selector) => Some(
            target
                .target_contract_for_backend_selector(
                    backend,
                    selector.name,
                    selector.value,
                )
                .ok_or_else(|| {
                    generated_requirement_error(
                        ctx,
                        op,
                        format!(
                            "generated intrinsic `{}` (`{}`) has no unique {:?} target contract for {}={}",
                            target.id,
                            target.marker,
                            backend,
                            selector.name,
                            selector.value
                        ),
                    )
                })?,
        ),
        None => None,
    };
    targets.entry(target.marker).or_insert(target);
    resolved_targets
        .entry((
            target.marker,
            selector.map(|selector| (selector.name, selector.value)),
        ))
        .or_insert(GeneratedResolvedTarget {
            target,
            selector,
            contract,
        });
    Ok(())
}

fn generated_target_selector(
    ctx: &Context,
    target: &GeneratedIntrinsicTarget,
    operation: Ptr<Operation>,
) -> Result<Option<GeneratedTargetSelectorBinding>, PipelineError> {
    let GeneratedIntrinsicVariant::Tcgen05Mma {
        target_selector: GeneratedTcgen05MmaTargetSelector::Kind,
        ..
    } = target.variant
    else {
        return Ok(None);
    };

    use dialect_nvvm::ops::Tcgen05MmaKindAttr;
    let kind_key =
        pliron::identifier::Identifier::try_from("nvvm_tcgen05_mma_kind").map_err(|error| {
            generated_requirement_error(
                ctx,
                operation,
                format!("invalid tcgen05 MMA kind attribute key: {error}"),
            )
        })?;
    let kind = operation
        .deref(ctx)
        .attributes
        .get::<Tcgen05MmaKindAttr>(&kind_key)
        .cloned();
    let value = match kind {
        Some(Tcgen05MmaKindAttr::F16) => "f16",
        Some(Tcgen05MmaKindAttr::Tf32) => "tf32",
        Some(Tcgen05MmaKindAttr::F8f6f4) => "f8f6f4",
        Some(Tcgen05MmaKindAttr::I8) => "i8",
        None => {
            return Err(generated_requirement_error(
                ctx,
                operation,
                format!(
                    "generated intrinsic `{}` (`{}`) is missing its `kind` target selector",
                    target.id, target.marker
                ),
            ));
        }
    };
    Ok(Some(GeneratedTargetSelectorBinding {
        name: "kind",
        value,
    }))
}

fn generated_requirement_error(
    ctx: &Context,
    op: Ptr<Operation>,
    message: String,
) -> PipelineError {
    PipelineError::Verification {
        name: "generated intrinsic target requirements".to_string(),
        message,
        operation: Some(op.deref(ctx).disp(ctx).to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated_intrinsic_targets::{
        GENERATED_INTRINSIC_MARKER_ATTR, GeneratedHardwareAlternative, GeneratedHardwareTarget,
        GeneratedPtxVersion, GeneratedTargetAlternative, GeneratedTcgen05MmaForm,
        generated_intrinsic_targets,
    };
    use dialect_nvvm::ops::{
        ScalarConversionOp, ScalarConversionRoundingAttr, ScalarConversionSaturationAttr,
        Tcgen05MmaBBufferAttr, Tcgen05MmaBUsageAttr, Tcgen05MmaCollectorAAttr,
        Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr, Tcgen05MmaKindAttr, Tcgen05MmaOp,
    };
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::identifier::Identifier;
    use pliron::op::Op;

    fn register_dialects(ctx: &mut Context) {
        dialect_mir::register(ctx);
        dialect_nvvm::register(ctx);
    }

    fn generated_tid_x_op(ctx: &mut Context, marker: Option<&str>) -> Ptr<Operation> {
        let result_type = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle();
        let op = Operation::new(
            ctx,
            dialect_nvvm::ops::ReadPtxSregTidXOp::get_concrete_op_info(),
            vec![result_type],
            vec![],
            vec![],
            0,
        );
        if let Some(marker) = marker {
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
        }
        op
    }

    struct ScalarConversionCase {
        rounding: ScalarConversionRoundingAttr,
        saturation: ScalarConversionSaturationAttr,
        marker: &'static str,
        id: &'static str,
        minimum_ptx: u16,
        minimum_sm: u16,
    }

    fn scalar_conversion_cases() -> [ScalarConversionCase; 10] {
        use ScalarConversionRoundingAttr::{NearestAway, NearestEven, TowardZero};
        use ScalarConversionSaturationAttr::{None, Relu, ReluSatfinite, Satfinite};

        [
            ScalarConversionCase {
                rounding: NearestAway,
                saturation: None,
                marker: "v1:i0368",
                id: "cvt_rna_tf32_f32",
                minimum_ptx: 70,
                minimum_sm: 80,
            },
            ScalarConversionCase {
                rounding: NearestAway,
                saturation: Satfinite,
                marker: "v1:i0369",
                id: "cvt_rna_satfinite_tf32_f32",
                minimum_ptx: 81,
                minimum_sm: 80,
            },
            ScalarConversionCase {
                rounding: NearestEven,
                saturation: None,
                marker: "v1:i0370",
                id: "cvt_rn_tf32_f32",
                minimum_ptx: 78,
                minimum_sm: 90,
            },
            ScalarConversionCase {
                rounding: NearestEven,
                saturation: Relu,
                marker: "v1:i0371",
                id: "cvt_rn_relu_tf32_f32",
                minimum_ptx: 78,
                minimum_sm: 90,
            },
            ScalarConversionCase {
                rounding: NearestEven,
                saturation: Satfinite,
                marker: "v1:i0372",
                id: "cvt_rn_satfinite_tf32_f32",
                minimum_ptx: 86,
                minimum_sm: 100,
            },
            ScalarConversionCase {
                rounding: NearestEven,
                saturation: ReluSatfinite,
                marker: "v1:i0373",
                id: "cvt_rn_relu_satfinite_tf32_f32",
                minimum_ptx: 86,
                minimum_sm: 100,
            },
            ScalarConversionCase {
                rounding: TowardZero,
                saturation: None,
                marker: "v1:i0374",
                id: "cvt_rz_tf32_f32",
                minimum_ptx: 78,
                minimum_sm: 90,
            },
            ScalarConversionCase {
                rounding: TowardZero,
                saturation: Relu,
                marker: "v1:i0375",
                id: "cvt_rz_relu_tf32_f32",
                minimum_ptx: 78,
                minimum_sm: 90,
            },
            ScalarConversionCase {
                rounding: TowardZero,
                saturation: Satfinite,
                marker: "v1:i0376",
                id: "cvt_rz_satfinite_tf32_f32",
                minimum_ptx: 86,
                minimum_sm: 100,
            },
            ScalarConversionCase {
                rounding: TowardZero,
                saturation: ReluSatfinite,
                marker: "v1:i0377",
                id: "cvt_rz_relu_satfinite_tf32_f32",
                minimum_ptx: 86,
                minimum_sm: 100,
            },
        ]
    }

    fn scalar_conversion_op(
        ctx: &mut Context,
        rounding: ScalarConversionRoundingAttr,
        saturation: ScalarConversionSaturationAttr,
        marker: Option<&str>,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let block = BasicBlock::new(ctx, None, vec![f32_ty.into()]);
        let value = block.deref(ctx).get_argument(0);
        let op = ScalarConversionOp::build(ctx, value, rounding, saturation);
        if let Some(marker) = marker {
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
        }
        op
    }

    fn generated_test_module(ctx: &mut Context, ops: &[Ptr<Operation>]) -> Ptr<Operation> {
        let module = pliron::builtin::ops::ModuleOp::new(ctx, "test".try_into().unwrap());
        let module_op = module.get_operation();
        for op in ops {
            crate::lower::append_to_module(ctx, module_op, *op);
        }
        module_op
    }

    fn tcgen05_mma_shared_op(
        ctx: &mut Context,
        kind: Option<Tcgen05MmaKindAttr>,
        marker: Option<&str>,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            Tcgen05MmaOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        let mma = Tcgen05MmaOp::new(op);
        mma.set_attr_nvvm_tcgen05_mma_form(ctx, Tcgen05MmaFormAttr::Shared);
        if let Some(kind) = kind {
            mma.set_attr_nvvm_tcgen05_mma_kind(ctx, kind);
        }
        mma.set_attr_nvvm_tcgen05_mma_cta_group(ctx, Tcgen05MmaCtaGroupAttr::Cg1);
        mma.set_attr_nvvm_tcgen05_mma_collector_a(ctx, Tcgen05MmaCollectorAAttr::Discard);
        if let Some(marker) = marker {
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
        }
        op
    }

    fn tcgen05_mma_ws_tensor_op(ctx: &mut Context) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            Tcgen05MmaOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        let mma = Tcgen05MmaOp::new(op);
        mma.set_attr_nvvm_tcgen05_mma_form(ctx, Tcgen05MmaFormAttr::WsTensor);
        mma.set_attr_nvvm_tcgen05_mma_kind(ctx, Tcgen05MmaKindAttr::F8f6f4);
        mma.set_attr_nvvm_tcgen05_mma_b_buffer(ctx, Tcgen05MmaBBufferAttr::B0);
        mma.set_attr_nvvm_tcgen05_mma_b_usage(ctx, Tcgen05MmaBUsageAttr::Discard);
        op
    }

    #[test]
    fn required_markers_are_recursive_and_deduplicated() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let first = generated_tid_x_op(&mut ctx, Some("v1:i0001"));
        let second = generated_tid_x_op(&mut ctx, Some("v1:i0001"));
        let module = generated_test_module(&mut ctx, &[first, second]);

        let requirements =
            collect_generated_intrinsic_requirements(&ctx, module, GeneratedMarkerPolicy::Required)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0001");
        assert_eq!(requirements.targets[0].id, "thread_idx_x");
    }

    #[test]
    fn rust_source_requires_a_marker_but_direct_dialect_input_can_derive_it() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = generated_tid_x_op(&mut ctx, None);

        let error =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(error.contains("missing its exact ABI marker"), "{error}");

        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();
        assert_eq!(requirements.targets[0].marker, "v1:i0001");
    }

    #[test]
    fn marker_validation_rejects_non_string_unknown_and_wrong_operation_ids() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        let non_string = generated_tid_x_op(&mut ctx, None);
        let i32_type = IntegerType::get(&ctx, 32, Signedness::Unsigned).to_handle();
        non_string.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            TypeAttr::new(i32_type),
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            non_string,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("has a non-string"), "{error}");

        let unknown = generated_tid_x_op(&mut ctx, Some("v1:i9999"));
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            unknown,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown generated-intrinsic marker `v1:i9999`"));

        let mismatch = generated_tid_x_op(&mut ctx, Some("v1:i0002"));
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            mismatch,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("marker `v1:i0002` belongs to `nvvm.read_ptx_sreg_ctaid_x`"));
        assert!(error.contains("not `nvvm.read_ptx_sreg_tid_x`"));
    }

    #[test]
    fn direct_dialect_input_selects_exact_ldmatrix_variant() {
        use dialect_mir::types::MirPtrType;
        use dialect_nvvm::ops::{
            LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp,
            LdmatrixShapeAttr, LdmatrixStateSpaceAttr,
        };
        use pliron::basic_block::BasicBlock;

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let pointer_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
        let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into()]);
        let pointer = block.deref(&ctx).get_argument(0);
        let op = LdmatrixOp::build(
            &mut ctx,
            pointer,
            LdmatrixShapeAttr::M8n8,
            LdmatrixMultiplicityAttr::X2,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B16,
            LdmatrixStateSpaceAttr::Shared,
        );

        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].id, "ldmatrix_m8n8_x2_b16");

        op.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0013".to_string()),
        );
        let error =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );
    }

    #[test]
    fn blackwell_ldmatrix_variants_select_exact_markers_and_targets() {
        use crate::generated_intrinsic_targets::{
            GeneratedHardwareAlternative, GeneratedHardwareTarget,
        };
        use dialect_mir::types::MirPtrType;
        use dialect_nvvm::ops::{
            LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp,
            LdmatrixShapeAttr, LdmatrixStateSpaceAttr,
        };

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
        let pointer_ty = MirPtrType::get_shared(&mut ctx, u8_ty.into(), false);
        let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into()]);
        let pointer = block.deref(&ctx).get_argument(0);
        let cases = [
            (
                "v1:i0378",
                "ldmatrix_m16n16_x1_trans_b8",
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8,
            ),
            (
                "v1:i0379",
                "ldmatrix_m16n16_x1_trans_b8x16_b4x16_p64",
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B4x16P64,
            ),
            (
                "v1:i0380",
                "ldmatrix_m16n16_x1_trans_b8x16_b6x16_p32",
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B6x16P32,
            ),
            (
                "v1:i0381",
                "ldmatrix_m16n16_x2_trans_b8",
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8,
            ),
            (
                "v1:i0382",
                "ldmatrix_m16n16_x2_trans_b8x16_b4x16_p64",
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B4x16P64,
            ),
            (
                "v1:i0383",
                "ldmatrix_m16n16_x2_trans_b8x16_b6x16_p32",
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B6x16P32,
            ),
            (
                "v1:i0384",
                "ldmatrix_m8n16_x1_b8x16_b4x16_p64",
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
            ),
            (
                "v1:i0385",
                "ldmatrix_m8n16_x1_b8x16_b6x16_p32",
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
            ),
            (
                "v1:i0386",
                "ldmatrix_m8n16_x2_b8x16_b4x16_p64",
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
            ),
            (
                "v1:i0387",
                "ldmatrix_m8n16_x2_b8x16_b6x16_p32",
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
            ),
            (
                "v1:i0388",
                "ldmatrix_m8n16_x4_b8x16_b4x16_p64",
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X4,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
            ),
            (
                "v1:i0389",
                "ldmatrix_m8n16_x4_b8x16_b6x16_p32",
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X4,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
            ),
        ];
        let expected_hardware = [
            GeneratedHardwareAlternative::ExactArchitecture(100),
            GeneratedHardwareAlternative::FamilyTarget(100),
            GeneratedHardwareAlternative::ExactArchitecture(103),
            GeneratedHardwareAlternative::FamilyTarget(103),
            GeneratedHardwareAlternative::ExactArchitecture(110),
            GeneratedHardwareAlternative::FamilyTarget(110),
            GeneratedHardwareAlternative::ExactArchitecture(120),
            GeneratedHardwareAlternative::FamilyTarget(120),
            GeneratedHardwareAlternative::ExactArchitecture(121),
            GeneratedHardwareAlternative::FamilyTarget(121),
        ];

        for (marker, id, shape, multiplicity, layout, element) in cases {
            let op = LdmatrixOp::build(
                &mut ctx,
                pointer,
                shape,
                multiplicity,
                layout,
                element,
                LdmatrixStateSpaceAttr::Shared,
            );
            let structural =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                    .unwrap();
            assert_eq!(structural.targets.len(), 1, "{id}");
            assert_eq!(structural.targets[0].id, id);

            op.deref_mut(&ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
            let marked =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(marked.targets[0].marker, marker);
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let requirement = marked
                    .clone()
                    .for_backend(backend)
                    .requirement(marked.targets[0]);
                assert_eq!(requirement.minimum_ptx.encoded(), 86, "{id} {backend:?}");
                assert!(
                    matches!(
                        requirement.hardware,
                        GeneratedHardwareTarget::AnyOf(alternatives)
                            if alternatives == expected_hardware
                    ),
                    "{id} {backend:?}"
                );
            }
        }
    }

    #[test]
    fn blackwell_ldmatrix_rejects_wrong_marker_and_unsupported_shape() {
        use dialect_mir::types::MirPtrType;
        use dialect_nvvm::ops::{
            LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp,
            LdmatrixShapeAttr, LdmatrixStateSpaceAttr,
        };

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
        let pointer_ty = MirPtrType::get_shared(&mut ctx, u8_ty.into(), false);
        let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into()]);
        let pointer = block.deref(&ctx).get_argument(0);

        let wrong_marker = LdmatrixOp::build(
            &mut ctx,
            pointer,
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8,
            LdmatrixStateSpaceAttr::Shared,
        );
        wrong_marker.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0379".to_string()),
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            wrong_marker,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );

        let unsupported = LdmatrixOp::build(
            &mut ctx,
            pointer,
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X4,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8,
            LdmatrixStateSpaceAttr::Shared,
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            unsupported,
            GeneratedMarkerPolicy::Optional,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("matches 0 generated catalog variants"),
            "{error}"
        );
    }

    #[test]
    fn classic_ldmatrix_compatibility_ops_keep_exact_target_requirements() {
        use crate::generated_intrinsic_targets::{
            GeneratedHardwareAlternative, GeneratedHardwareTarget,
        };
        use dialect_mir::types::MirPtrType;
        use dialect_nvvm::ops::{
            LdmatrixX1Op, LdmatrixX1TransOp, LdmatrixX2Op, LdmatrixX2TransOp, LdmatrixX4Op,
            LdmatrixX4TransOp,
        };
        use pliron::basic_block::BasicBlock;

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let pointer_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
        let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into()]);
        let pointer = block.deref(&ctx).get_argument(0);
        let cases = [
            (
                LdmatrixX1Op::get_concrete_op_info(),
                1,
                "ldmatrix_m8n8_x1_b16",
            ),
            (
                LdmatrixX1TransOp::get_concrete_op_info(),
                1,
                "ldmatrix_m8n8_x1_trans_b16",
            ),
            (
                LdmatrixX2Op::get_concrete_op_info(),
                2,
                "ldmatrix_m8n8_x2_b16",
            ),
            (
                LdmatrixX2TransOp::get_concrete_op_info(),
                2,
                "ldmatrix_m8n8_x2_trans_b16",
            ),
            (
                LdmatrixX4Op::get_concrete_op_info(),
                4,
                "ldmatrix_m8n8_x4_b16",
            ),
            (
                LdmatrixX4TransOp::get_concrete_op_info(),
                4,
                "ldmatrix_m8n8_x4_trans_b16",
            ),
        ];

        for (op_info, result_count, expected_id) in cases {
            let op = Operation::new(
                &mut ctx,
                op_info,
                vec![u32_ty.into(); result_count],
                vec![pointer],
                vec![],
                0,
            );
            let requirements =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].id, expected_id);
            let requirement = requirements.requirement(requirements.targets[0]);
            assert_eq!(requirement.minimum_ptx.encoded(), 65);
            assert!(matches!(
                requirement.hardware,
                GeneratedHardwareTarget::AnyOf(alternatives)
                    if alternatives == [GeneratedHardwareAlternative::MinimumSm(75)]
            ));

            let error =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("missing its exact ABI marker"), "{error}");
        }
    }

    #[test]
    fn register_mma_markers_and_attributes_select_one_exact_variant() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::types::FP32Type;

        fn register_mma(
            ctx: &mut Context,
            element: RegisterMmaElementAttr,
            set_operation: bool,
            marker: Option<&str>,
        ) -> Ptr<Operation> {
            let f32_ty = FP32Type::get(ctx);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| f32_ty.into())
                .chain((0..6).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..10)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![f32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(ctx, RegisterMmaShapeAttr::M16n8k16);
            if set_operation {
                mma.set_attr_nvvm_register_mma_operation(ctx, RegisterMmaOperationAttr::Multiply);
            }
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::F32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, element.clone());
            mma.set_attr_nvvm_register_mma_b_element(ctx, element);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::NotApplicable);
            if let Some(marker) = marker {
                operation.deref_mut(ctx).attributes.set(
                    Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                    StringAttr::new(marker.to_string()),
                );
            }
            operation
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        let bf16 = register_mma(&mut ctx, RegisterMmaElementAttr::Bf16, false, None);
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, bf16, GeneratedMarkerPolicy::Optional)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0105");

        bf16.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0106".to_string()),
        );
        let error =
            collect_generated_intrinsic_requirements(&ctx, bf16, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );

        let f16 = register_mma(
            &mut ctx,
            RegisterMmaElementAttr::F16,
            true,
            Some("v1:i0106"),
        );
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, f16, GeneratedMarkerPolicy::Required)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0106");

        let crossed = register_mma(&mut ctx, RegisterMmaElementAttr::F16, true, None);
        RegisterMmaOp::new(crossed)
            .set_attr_nvvm_register_mma_b_element(&ctx, RegisterMmaElementAttr::Bf16);
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            crossed,
            GeneratedMarkerPolicy::Optional,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("matches 0 generated catalog variants"),
            "{error}"
        );
    }

    #[test]
    fn integer_register_mma_markers_require_exact_variant_attributes() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;

        fn register_mma(
            ctx: &mut Context,
            shape: RegisterMmaShapeAttr,
            a_element: RegisterMmaElementAttr,
            b_element: RegisterMmaElementAttr,
            overflow: RegisterMmaOverflowAttr,
            marker: &str,
        ) -> Ptr<Operation> {
            let (a_count, b_count) = match &shape {
                RegisterMmaShapeAttr::M16n8k16 => (2, 1),
                RegisterMmaShapeAttr::M16n8k32 => (4, 2),
                _ => panic!("unsupported integer MMA shape"),
            };
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| i32_ty.into())
                .chain((0..a_count + b_count).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..4 + a_count + b_count)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(ctx, shape);
            mma.set_attr_nvvm_register_mma_operation(ctx, RegisterMmaOperationAttr::Multiply);
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, a_element);
            mma.set_attr_nvvm_register_mma_b_element(ctx, b_element);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, overflow);
            operation.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
            operation
        }

        fn require_marker(ctx: &Context, op: Ptr<Operation>, marker: &str, id: &str) {
            let requirements =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, marker);
            assert_eq!(requirements.targets[0].id, id);
        }

        fn reject_marker(ctx: &Context, op: Ptr<Operation>) {
            let error =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("does not match the exact variant attributes"),
                "{error}"
            );
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        let k16_satfinite = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k16,
            RegisterMmaElementAttr::S8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Satfinite,
            "v1:i0118",
        );
        require_marker(
            &ctx,
            k16_satfinite,
            "v1:i0118",
            "mma_m16n8k16_s32_s8_u8_satfinite",
        );

        let k32_wrapping = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k32,
            RegisterMmaElementAttr::U8,
            RegisterMmaElementAttr::S8,
            RegisterMmaOverflowAttr::Wrapping,
            "v1:i0116",
        );
        require_marker(&ctx, k32_wrapping, "v1:i0116", "mma_m16n8k32_s32_u8_s8");

        let wrong_signedness = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k16,
            RegisterMmaElementAttr::U8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Satfinite,
            "v1:i0118",
        );
        reject_marker(&ctx, wrong_signedness);

        let wrong_overflow = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k16,
            RegisterMmaElementAttr::S8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Wrapping,
            "v1:i0118",
        );
        reject_marker(&ctx, wrong_overflow);

        let wrong_shape = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k32,
            RegisterMmaElementAttr::S8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Satfinite,
            "v1:i0118",
        );
        reject_marker(&ctx, wrong_shape);
    }

    #[test]
    fn b1_register_mma_markers_require_exact_operation_and_shape() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;

        fn register_mma(
            ctx: &mut Context,
            shape: RegisterMmaShapeAttr,
            operation: Option<RegisterMmaOperationAttr>,
            marker: &str,
        ) -> Ptr<Operation> {
            let (accumulator_count, a_count, b_count, result_count) = match shape {
                RegisterMmaShapeAttr::M8n8k128 => (2, 1, 1, 2),
                RegisterMmaShapeAttr::M16n8k128 => (4, 2, 1, 4),
                RegisterMmaShapeAttr::M16n8k256 => (4, 4, 2, 4),
                _ => panic!("unsupported B1 MMA shape"),
            };
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..accumulator_count)
                .map(|_| i32_ty.into())
                .chain((0..a_count + b_count).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..accumulator_count + a_count + b_count)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let op = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); result_count],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(op);
            mma.set_attr_nvvm_register_mma_shape(ctx, shape);
            if let Some(operation) = operation {
                mma.set_attr_nvvm_register_mma_operation(ctx, operation);
            }
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, RegisterMmaElementAttr::B1);
            mma.set_attr_nvvm_register_mma_b_element(ctx, RegisterMmaElementAttr::B1);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::Wrapping);
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
            op
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        for (shape, operation, marker, id) in [
            (
                RegisterMmaShapeAttr::M8n8k128,
                RegisterMmaOperationAttr::XorPopc,
                "v1:i0157",
                "mma_m8n8k128_s32_b1_xor_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k128,
                RegisterMmaOperationAttr::XorPopc,
                "v1:i0158",
                "mma_m16n8k128_s32_b1_xor_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k256,
                RegisterMmaOperationAttr::XorPopc,
                "v1:i0159",
                "mma_m16n8k256_s32_b1_xor_popc",
            ),
            (
                RegisterMmaShapeAttr::M8n8k128,
                RegisterMmaOperationAttr::AndPopc,
                "v1:i0160",
                "mma_m8n8k128_s32_b1_and_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k128,
                RegisterMmaOperationAttr::AndPopc,
                "v1:i0161",
                "mma_m16n8k128_s32_b1_and_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k256,
                RegisterMmaOperationAttr::AndPopc,
                "v1:i0162",
                "mma_m16n8k256_s32_b1_and_popc",
            ),
        ] {
            let op = register_mma(&mut ctx, shape, Some(operation), marker);
            let requirements =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, marker);
            assert_eq!(requirements.targets[0].id, id);
        }

        for op in [
            register_mma(
                &mut ctx,
                RegisterMmaShapeAttr::M8n8k128,
                Some(RegisterMmaOperationAttr::AndPopc),
                "v1:i0157",
            ),
            register_mma(&mut ctx, RegisterMmaShapeAttr::M8n8k128, None, "v1:i0157"),
        ] {
            let error =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("does not match the exact variant attributes"),
                "{error}"
            );
        }
    }

    #[test]
    fn sparse_mma_markers_require_exact_variant_attributes() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
            SparseMmaAccumulatorAttr, SparseMmaElementAttr, SparseMmaLayoutAttr,
            SparseMmaMetadataAttr, SparseMmaOp, SparseMmaOverflowAttr, SparseMmaSelectorAttr,
            SparseMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;

        fn sparse_mma(
            ctx: &mut Context,
            a_element: SparseMmaElementAttr,
            b_element: SparseMmaElementAttr,
            overflow: SparseMmaOverflowAttr,
            metadata: SparseMmaMetadataAttr,
            marker: Option<&str>,
        ) -> Ptr<Operation> {
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| i32_ty.into())
                .chain((0..6).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..10)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(ctx, SparseMmaShapeAttr::M16n8k32);
            mma.set_attr_nvvm_sparse_mma_accumulator(ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(ctx, a_element);
            mma.set_attr_nvvm_sparse_mma_b_element(ctx, b_element);
            mma.set_attr_nvvm_sparse_mma_a_layout(ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(ctx, overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(ctx, metadata);
            mma.set_attr_nvvm_sparse_mma_selector(ctx, SparseMmaSelectorAttr::ImmediateZeroOrOne);
            if let Some(marker) = marker {
                operation.deref_mut(ctx).attributes.set(
                    Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                    StringAttr::new(marker.to_string()),
                );
            }
            operation
        }

        fn dense_mma_with_sparse_marker(ctx: &mut Context, marker: &str) -> Ptr<Operation> {
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| i32_ty.into())
                .chain((0..6).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..10)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(ctx, RegisterMmaShapeAttr::M16n8k32);
            mma.set_attr_nvvm_register_mma_operation(ctx, RegisterMmaOperationAttr::Multiply);
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, RegisterMmaElementAttr::S8);
            mma.set_attr_nvvm_register_mma_b_element(ctx, RegisterMmaElementAttr::S8);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::Wrapping);
            operation.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
            operation
        }

        fn require_marker(ctx: &Context, op: Ptr<Operation>, marker: &str, id: &str) {
            let requirements =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, marker);
            assert_eq!(requirements.targets[0].id, id);
        }

        fn reject_marker(ctx: &Context, op: Ptr<Operation>) {
            let error =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("does not match the exact variant attributes"),
                "{error}"
            );
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        for (a_element, b_element, overflow, metadata, marker, id) in [
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0163",
                "mma_sp_m16n8k32_s32_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0164",
                "mma_sp_m16n8k32_s32_s8_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0165",
                "mma_sp_m16n8k32_s32_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0166",
                "mma_sp_m16n8k32_s32_u8_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0167",
                "mma_sp_m16n8k32_s32_s8_satfinite",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0168",
                "mma_sp_m16n8k32_s32_s8_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0169",
                "mma_sp_m16n8k32_s32_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0170",
                "mma_sp_m16n8k32_s32_u8_s8_satfinite",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0171",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0172",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0173",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0174",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0175",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8_satfinite",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0176",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0177",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0178",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8_s8_satfinite",
            ),
        ] {
            let op = sparse_mma(
                &mut ctx,
                a_element,
                b_element,
                overflow,
                metadata,
                Some(marker),
            );
            require_marker(&ctx, op, marker, id);
        }

        let structural = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
            SparseMmaMetadataAttr::Standard,
            None,
        );
        let requirements = collect_generated_intrinsic_requirements(
            &ctx,
            structural,
            GeneratedMarkerPolicy::Optional,
        )
        .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0170");

        let wrong_element = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0163"),
        );
        reject_marker(&ctx, wrong_element);

        let wrong_overflow = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0163"),
        );
        reject_marker(&ctx, wrong_overflow);

        let standard_with_ordered_marker = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0171"),
        );
        reject_marker(&ctx, standard_with_ordered_marker);

        let ordered_with_standard_marker = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Ordered,
            Some("v1:i0163"),
        );
        reject_marker(&ctx, ordered_with_standard_marker);

        let wrong_layout = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0163"),
        );
        SparseMmaOp::new(wrong_layout)
            .set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Row);
        reject_marker(&ctx, wrong_layout);

        let sparse_with_dense_marker = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0116"),
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            sparse_with_dense_marker,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("belongs to `nvvm.register_mma`"), "{error}");
        let dense_with_sparse_marker = dense_mma_with_sparse_marker(&mut ctx, "v1:i0163");
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            dense_with_sparse_marker,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("belongs to `nvvm.sparse_mma`"), "{error}");
    }

    #[test]
    fn sparse_mma_targets_require_metadata_specific_ptx_and_ampere() {
        use crate::generated_intrinsic_targets::{
            GeneratedHardwareAlternative, GeneratedHardwareTarget,
        };

        for (marker, minimum_ptx) in [
            ("v1:i0163", 71),
            ("v1:i0164", 71),
            ("v1:i0165", 71),
            ("v1:i0166", 71),
            ("v1:i0167", 71),
            ("v1:i0168", 71),
            ("v1:i0169", 71),
            ("v1:i0170", 71),
            ("v1:i0171", 85),
            ("v1:i0172", 85),
            ("v1:i0173", 85),
            ("v1:i0174", 85),
            ("v1:i0175", 85),
            ("v1:i0176", 85),
            ("v1:i0177", 85),
            ("v1:i0178", 85),
        ] {
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let requirement = target.requirement_for_backend(backend);
                assert_eq!(requirement.minimum_ptx.encoded(), minimum_ptx, "{marker}");
                assert_eq!(
                    requirement.hardware,
                    GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(80)]),
                    "{marker}"
                );
            }
        }
    }

    #[test]
    fn integer_register_mma_targets_require_ampere_on_both_backends() {
        use crate::generated_intrinsic_targets::{
            GeneratedHardwareAlternative, GeneratedHardwareTarget,
        };

        for marker in [
            "v1:i0108", "v1:i0110", "v1:i0111", "v1:i0112", "v1:i0113", "v1:i0114", "v1:i0115",
            "v1:i0116", "v1:i0117", "v1:i0118", "v1:i0119", "v1:i0120", "v1:i0121", "v1:i0122",
            "v1:i0123", "v1:i0124",
        ] {
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let requirement = target.requirement_for_backend(backend);
                assert_eq!(requirement.minimum_ptx.encoded(), 70, "{marker}");
                assert_eq!(
                    requirement.hardware,
                    GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(80)]),
                    "{marker}"
                );
            }
        }
    }

    #[test]
    fn scalar_conversion_markers_and_attributes_select_exact_targets() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        for case in scalar_conversion_cases() {
            let marked = scalar_conversion_op(
                &mut ctx,
                case.rounding.clone(),
                case.saturation.clone(),
                Some(case.marker),
            );
            let requirements = collect_generated_intrinsic_requirements(
                &ctx,
                marked,
                GeneratedMarkerPolicy::Required,
            )
            .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, case.marker);
            assert_eq!(requirements.targets[0].id, case.id);

            let structural = scalar_conversion_op(&mut ctx, case.rounding, case.saturation, None);
            let requirements = collect_generated_intrinsic_requirements(
                &ctx,
                structural,
                GeneratedMarkerPolicy::Optional,
            )
            .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, case.marker);
            assert_eq!(requirements.targets[0].id, case.id);
        }
    }

    #[test]
    fn scalar_conversion_rejects_wrong_same_op_marker_and_invalid_attributes() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        let wrong_marker = scalar_conversion_op(
            &mut ctx,
            ScalarConversionRoundingAttr::NearestAway,
            ScalarConversionSaturationAttr::None,
            Some("v1:i0369"),
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            wrong_marker,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );

        for saturation in [
            ScalarConversionSaturationAttr::Relu,
            ScalarConversionSaturationAttr::ReluSatfinite,
        ] {
            let invalid = scalar_conversion_op(
                &mut ctx,
                ScalarConversionRoundingAttr::NearestAway,
                saturation,
                None,
            );
            let error = collect_generated_intrinsic_requirements(
                &ctx,
                invalid,
                GeneratedMarkerPolicy::Optional,
            )
            .unwrap_err()
            .to_string();
            assert!(
                error.contains("matches 0 generated catalog variants"),
                "{error}"
            );
        }
    }

    #[test]
    fn extended_minmax_target_matcher_is_exact() {
        use dialect_nvvm::ops::{
            ExtendedMinMaxFormatAttr, ExtendedMinMaxNanAttr, ExtendedMinMaxOp,
            ExtendedMinMaxOperationAttr, ExtendedMinMaxSubnormalAttr, ExtendedMinMaxXorSignAbsAttr,
        };

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let f32_ty = FP32Type::get(&ctx);
        let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), f32_ty.into()]);
        let a = block.deref(&ctx).get_argument(0);
        let b = block.deref(&ctx).get_argument(1);

        let exact = ExtendedMinMaxOp::build(
            &mut ctx,
            a,
            b,
            ExtendedMinMaxFormatAttr::F32,
            ExtendedMinMaxOperationAttr::Min,
            ExtendedMinMaxSubnormalAttr::Preserve,
            ExtendedMinMaxNanAttr::Number,
            ExtendedMinMaxXorSignAbsAttr::Enabled,
        );
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, exact, GeneratedMarkerPolicy::Optional)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0562");
        assert_eq!(requirements.targets[0].id, "min_xorsign_abs_f32");

        let adjacent = ExtendedMinMaxOp::build(
            &mut ctx,
            a,
            b,
            ExtendedMinMaxFormatAttr::F32,
            ExtendedMinMaxOperationAttr::Min,
            ExtendedMinMaxSubnormalAttr::Preserve,
            ExtendedMinMaxNanAttr::Number,
            ExtendedMinMaxXorSignAbsAttr::Disabled,
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            adjacent,
            GeneratedMarkerPolicy::Optional,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("matches 0 generated catalog variants"),
            "{error}"
        );

        let wrong_marker = ExtendedMinMaxOp::build(
            &mut ctx,
            a,
            b,
            ExtendedMinMaxFormatAttr::F32,
            ExtendedMinMaxOperationAttr::Min,
            ExtendedMinMaxSubnormalAttr::Preserve,
            ExtendedMinMaxNanAttr::Number,
            ExtendedMinMaxXorSignAbsAttr::Enabled,
        );
        wrong_marker.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0559".to_string()),
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            wrong_marker,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );
    }

    #[test]
    fn scalar_conversion_collector_preserves_all_backend_floor_groups() {
        use crate::generated_intrinsic_targets::{
            GeneratedHardwareAlternative, GeneratedHardwareTarget,
        };

        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        for case in scalar_conversion_cases() {
            let op =
                scalar_conversion_op(&mut ctx, case.rounding, case.saturation, Some(case.marker));
            let requirements =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            let target = requirements.targets[0];
            assert_eq!(target.backend_requirements.len(), 2, "{}", case.marker);

            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let backend_requirements = requirements.clone().for_backend(backend);
                let requirement = backend_requirements.requirement(target);
                assert_eq!(
                    requirement.minimum_ptx.encoded(),
                    case.minimum_ptx,
                    "{} {backend:?}",
                    case.marker
                );
                assert!(
                    matches!(
                        requirement.hardware,
                        GeneratedHardwareTarget::AnyOf(alternatives)
                            if alternatives
                                == [GeneratedHardwareAlternative::MinimumSm(case.minimum_sm)]
                    ),
                    "{} {backend:?}",
                    case.marker
                );
            }
        }
    }

    #[test]
    fn direct_cluster_barrier_arrive_selects_only_i0277() {
        use dialect_nvvm::ops::{ClusterBarrierModeAttr, ClusterBarrierOp};

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = ClusterBarrierOp::build(&mut ctx, ClusterBarrierModeAttr::Arrive);
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();

        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0277");
        assert_eq!(requirements.targets[0].id, "barrier_cluster_arrive");
        assert_eq!(
            requirements
                .requirement(requirements.targets[0])
                .minimum_ptx
                .encoded(),
            78
        );
    }

    #[test]
    fn direct_cluster_barrier_relaxed_selects_only_i0279() {
        use dialect_nvvm::ops::{ClusterBarrierModeAttr, ClusterBarrierOp};

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = ClusterBarrierOp::build(&mut ctx, ClusterBarrierModeAttr::ArriveRelaxed);
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();

        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0279");
        assert_eq!(requirements.targets[0].id, "barrier_cluster_arrive_relaxed");
        assert_eq!(
            requirements
                .requirement(requirements.targets[0])
                .minimum_ptx
                .encoded(),
            80
        );
    }

    #[test]
    fn cluster_barrier_marker_rejects_a_different_mode() {
        use dialect_nvvm::ops::{ClusterBarrierModeAttr, ClusterBarrierOp};

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = ClusterBarrierOp::build(&mut ctx, ClusterBarrierModeAttr::Arrive);
        op.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0279".to_string()),
        );

        let error =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );
    }

    #[test]
    fn tcgen05_mma_kind_contracts_are_retained_per_marker_and_kind() {
        let target = generated_intrinsic_targets()
            .find(|target| target.id == "tcgen05_mma_shared")
            .expect("generated tcgen05 shared MMA target");
        assert_eq!(target.marker, "v1:i0763");

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let f16 =
            tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::F16), Some(target.marker));
        let f16_requirements = collect_generated_intrinsic_requirements_for_backend(
            &ctx,
            f16,
            GeneratedMarkerPolicy::Required,
            GeneratedIntrinsicBackend::LlvmNvptx,
        )
        .unwrap();
        assert!(crate::target::generated_target_satisfied(
            &"sm_103a".parse().unwrap(),
            &f16_requirements
        ));
        assert!(crate::target::generated_target_satisfied(
            &"sm_100f".parse().unwrap(),
            &f16_requirements
        ));

        let i8 = tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::I8), Some(target.marker));
        let i8_requirements = collect_generated_intrinsic_requirements_for_backend(
            &ctx,
            i8,
            GeneratedMarkerPolicy::Required,
            GeneratedIntrinsicBackend::LlvmNvptx,
        )
        .unwrap();
        assert!(!crate::target::generated_target_satisfied(
            &"sm_103a".parse().unwrap(),
            &i8_requirements
        ));
        assert!(!crate::target::generated_target_satisfied(
            &"sm_100f".parse().unwrap(),
            &i8_requirements
        ));
        assert!(crate::target::generated_target_satisfied(
            &"sm_101a".parse().unwrap(),
            &i8_requirements
        ));
        let switched = i8_requirements
            .clone()
            .for_backend(GeneratedIntrinsicBackend::LibNvvm);
        assert!(!crate::target::generated_target_satisfied(
            &"sm_101a".parse().unwrap(),
            &switched
        ));

        let f16 =
            tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::F16), Some(target.marker));
        let i8 = tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::I8), Some(target.marker));
        let module = generated_test_module(&mut ctx, &[f16, i8]);
        let both = collect_generated_intrinsic_requirements_for_backend(
            &ctx,
            module,
            GeneratedMarkerPolicy::Required,
            GeneratedIntrinsicBackend::LlvmNvptx,
        )
        .unwrap();
        assert_eq!(both.targets.len(), 1);
        assert_eq!(both.resolved_targets().len(), 2);
        assert_eq!(
            both.resolved_targets()
                .iter()
                .map(|resolved| resolved.selector.unwrap().value)
                .collect::<Vec<_>>(),
            ["f16", "i8"]
        );
        assert!(!crate::target::generated_target_satisfied(
            &"sm_103a".parse().unwrap(),
            &both
        ));
        assert!(!crate::target::generated_target_satisfied(
            &"sm_100f".parse().unwrap(),
            &both
        ));
        assert!(crate::target::generated_target_satisfied(
            &"sm_100a".parse().unwrap(),
            &both
        ));

        let f16 =
            tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::F16), Some(target.marker));
        let libnvvm_f16 = collect_generated_intrinsic_requirements_for_backend(
            &ctx,
            f16,
            GeneratedMarkerPolicy::Required,
            GeneratedIntrinsicBackend::LibNvvm,
        )
        .unwrap();
        assert!(!crate::target::generated_target_satisfied(
            &"sm_101a".parse().unwrap(),
            &libnvvm_f16
        ));
        assert!(crate::target::generated_target_satisfied(
            &"sm_103a".parse().unwrap(),
            &libnvvm_f16
        ));

        let i8 = tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::I8), Some(target.marker));
        let libnvvm_i8 = collect_generated_intrinsic_requirements_for_backend(
            &ctx,
            i8,
            GeneratedMarkerPolicy::Required,
            GeneratedIntrinsicBackend::LibNvvm,
        )
        .unwrap();
        assert!(!crate::target::generated_target_satisfied(
            &"sm_101a".parse().unwrap(),
            &libnvvm_i8
        ));
        assert!(crate::target::generated_target_satisfied(
            &"sm_110a".parse().unwrap(),
            &libnvvm_i8
        ));
    }

    #[test]
    fn markerless_tcgen05_mma_ignores_compatibility_aliases() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = tcgen05_mma_ws_tensor_op(&mut ctx);
        let requirements = collect_generated_intrinsic_requirements_for_backend(
            &ctx,
            op,
            GeneratedMarkerPolicy::Optional,
            GeneratedIntrinsicBackend::LlvmNvptx,
        )
        .unwrap();

        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].id, "tcgen05_mma_ws_tensor");
        assert_eq!(requirements.resolved_targets().len(), 1);
        assert_eq!(
            requirements.resolved_targets()[0].selector.unwrap().value,
            "f8f6f4"
        );
    }

    #[test]
    fn tcgen05_mma_selector_contract_failures_are_closed() {
        static ALTERNATIVES: &[GeneratedTargetAlternative] = &[GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(100),
        }];
        static F16_CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
            selectors: &[GeneratedTargetSelectorBinding {
                name: "kind",
                value: "f16",
            }],
            alternatives: ALTERNATIVES,
        }];
        static DUPLICATE_F16_CONTRACTS: &[GeneratedTargetContract] = &[
            GeneratedTargetContract {
                selectors: &[GeneratedTargetSelectorBinding {
                    name: "kind",
                    value: "f16",
                }],
                alternatives: ALTERNATIVES,
            },
            GeneratedTargetContract {
                selectors: &[GeneratedTargetSelectorBinding {
                    name: "kind",
                    value: "f16",
                }],
                alternatives: ALTERNATIVES,
            },
        ];
        static MISSING_I8_TARGET: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
            marker: "test:tcgen_mma_missing_i8",
            id: "tcgen_mma_missing_i8",
            abi_id: "test",
            dialect_op: "nvvm.tcgen05_mma",
            variant: GeneratedIntrinsicVariant::Tcgen05Mma {
                form: GeneratedTcgen05MmaForm::Shared,
                target_selector: GeneratedTcgen05MmaTargetSelector::Kind,
                compatibility_alias: false,
            },
            requirement: GeneratedTargetRequirement {
                minimum_ptx: GeneratedPtxVersion::from_encoded(86),
                hardware: GeneratedHardwareTarget::TargetMatrix {
                    contracts: F16_CONTRACTS,
                },
            },
            backend_requirements: &[],
            selections: &[],
            llvm: None,
        };
        static DUPLICATE_F16_TARGET: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
            marker: "test:tcgen_mma_duplicate_f16",
            id: "tcgen_mma_duplicate_f16",
            abi_id: "test",
            dialect_op: "nvvm.tcgen05_mma",
            variant: GeneratedIntrinsicVariant::Tcgen05Mma {
                form: GeneratedTcgen05MmaForm::Shared,
                target_selector: GeneratedTcgen05MmaTargetSelector::Kind,
                compatibility_alias: false,
            },
            requirement: GeneratedTargetRequirement {
                minimum_ptx: GeneratedPtxVersion::from_encoded(86),
                hardware: GeneratedHardwareTarget::TargetMatrix {
                    contracts: DUPLICATE_F16_CONTRACTS,
                },
            },
            backend_requirements: &[],
            selections: &[],
            llvm: None,
        };

        fn retain_error(
            ctx: &Context,
            op: Ptr<Operation>,
            target: &'static GeneratedIntrinsicTarget,
        ) -> String {
            let mut targets = std::collections::BTreeMap::new();
            let mut resolved = std::collections::BTreeMap::new();
            retain_generated_target(
                ctx,
                op,
                GeneratedIntrinsicBackend::LlvmNvptx,
                target,
                &mut targets,
                &mut resolved,
            )
            .unwrap_err()
            .to_string()
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let missing = tcgen05_mma_shared_op(&mut ctx, None, None);
        let error = retain_error(&ctx, missing, &MISSING_I8_TARGET);
        assert!(
            error.contains("missing its `kind` target selector"),
            "{error}"
        );

        let unknown = tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::I8), None);
        let error = retain_error(&ctx, unknown, &MISSING_I8_TARGET);
        assert!(
            error.contains("no unique LlvmNvptx target contract for kind=i8"),
            "{error}"
        );

        let duplicate = tcgen05_mma_shared_op(&mut ctx, Some(Tcgen05MmaKindAttr::F16), None);
        let error = retain_error(&ctx, duplicate, &DUPLICATE_F16_TARGET);
        assert!(
            error.contains("no unique LlvmNvptx target contract for kind=f16"),
            "{error}"
        );
    }
}
