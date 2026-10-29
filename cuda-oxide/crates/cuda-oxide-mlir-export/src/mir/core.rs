/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The part of `dialect-mir` that maps directly to standard MLIR dialects.
//!
//! Rust signedness lives on the Pliron integer type. MLIR integers are
//! signless, so operations that care about signedness choose an explicit
//! signed or unsigned `arith` operation before that information disappears.

use crate::{
    AttributeTranslation, DropAttribute, FixedType, MlirAttribute, MlirBlockArgument,
    MlirFloatType, MlirOperation, MlirResult, MlirType, MlirValueUse, OperationInput,
    OperationTranslation, TranslationError, TranslationRegistry, TranslationSession,
};
use dialect_mir::{
    attributes::{
        InlineIntentAttr, MirCastKindAttr, MirFP16Attr, MirPointerKindAuthorityAttr,
        ReferenceParamValidityAttr, UnrollAttr,
    },
    ops::{
        MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCastOp, MirCondBranchOp, MirConstantOp,
        MirDivOp, MirEqOp, MirFloatConstantOp, MirFuncOp, MirGeOp, MirGotoOp, MirGtOp, MirLeOp,
        MirLtOp, MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirRemOp, MirReturnOp, MirShlOp, MirShrOp,
        MirSubOp, MirUndefOp,
    },
    types::{MirDisjointSliceType, MirFP16Type, MirSliceType},
};
use pliron::{
    attribute::AttrObj,
    builtin::{
        op_interfaces::{ATTR_KEY_SYM_NAME, OperandSegmentInterface, SymbolOpInterface},
        type_interfaces::{FloatTypeInterface, FunctionTypeInterface},
        types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed, type_cast, type_impls},
    utils::apfloat::Float,
};

/// Register standard-MLIR mappings for scalar MIR and arbitrary CFG.
pub fn register_mir_core_pack(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_type::<MirFP16Type>(FixedType(MlirType::Float(MlirFloatType::F16)))?;
    registry.register_attribute::<MirFP16Attr>(MirFp16AttributeTranslation)?;
    // A `mir.func`'s `#[inline]` request. The function translation folds the
    // keyword into the LLVM `passthrough` list, as `mir-lower` does natively.
    registry.register_attribute::<InlineIntentAttr>(InlineIntentTranslation)?;
    registry.register_attribute::<MirCastKindAttr>(DropAttribute)?;
    // Rust-boundary provenance metadata consumed by the dialect-mir verifier;
    // like the cast kind, it has no MLIR counterpart and is dropped at export.
    registry.register_attribute::<MirPointerKindAuthorityAttr>(DropAttribute)?;
    // These optional optimization hints use Rust argument indices. Drop them
    // until they can be remapped to the launch arguments after slice flattening.
    registry.register_attribute::<ReferenceParamValidityAttr>(DropAttribute)?;
    registry.register_attribute::<UnrollAttr>(DropAttribute)?;

    registry.register_operation::<MirFuncOp>(FunctionTranslation)?;
    registry.register_operation::<MirReturnOp>(RenameOperation("func.return"))?;
    registry.register_operation::<MirGotoOp>(RenameOperation("cf.br"))?;
    registry.register_operation::<MirCondBranchOp>(ConditionalBranchTranslation)?;

    registry.register_operation::<MirConstantOp>(ConstantTranslation::Integer)?;
    registry.register_operation::<MirFloatConstantOp>(ConstantTranslation::Float)?;
    registry.register_operation::<MirUndefOp>(RenameOperation("ub.poison"))?;

    registry.register_operation::<MirAddOp>(NumericBinaryTranslation::Add)?;
    registry.register_operation::<MirSubOp>(NumericBinaryTranslation::Sub)?;
    registry.register_operation::<MirMulOp>(NumericBinaryTranslation::Mul)?;
    registry.register_operation::<MirDivOp>(NumericBinaryTranslation::Div)?;
    registry.register_operation::<MirRemOp>(NumericBinaryTranslation::Rem)?;
    registry.register_operation::<MirBitAndOp>(IntegerBinaryTranslation("arith.andi"))?;
    registry.register_operation::<MirBitOrOp>(IntegerBinaryTranslation("arith.ori"))?;
    registry.register_operation::<MirBitXorOp>(IntegerBinaryTranslation("arith.xori"))?;
    registry.register_operation::<MirShlOp>(ShiftTranslation { right: false })?;
    registry.register_operation::<MirShrOp>(ShiftTranslation { right: true })?;
    registry.register_operation::<MirNegOp>(NegTranslation)?;
    registry.register_operation::<MirNotOp>(NotTranslation)?;

    registry.register_operation::<MirLtOp>(ComparisonTranslation::Lt)?;
    registry.register_operation::<MirLeOp>(ComparisonTranslation::Le)?;
    registry.register_operation::<MirGtOp>(ComparisonTranslation::Gt)?;
    registry.register_operation::<MirGeOp>(ComparisonTranslation::Ge)?;
    registry.register_operation::<MirEqOp>(ComparisonTranslation::Eq)?;
    registry.register_operation::<MirNeOp>(ComparisonTranslation::Ne)?;
    registry.register_operation::<MirCastOp>(CastTranslation)?;
    crate::mir::memory::register_mir_memory_pack(registry)?;
    crate::mir::checked::register(registry)?;
    crate::mir::shared::register(registry)?;
    Ok(())
}

/// Preserve the FP16 bit pattern when translating MIR half-precision constants.
struct MirFp16AttributeTranslation;

impl AttributeTranslation for MirFp16AttributeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let value = source
            .downcast_ref::<MirFP16Attr>()
            .ok_or_else(|| "expected mir.fp16_attr".to_owned())?;
        Ok(Some(MlirAttribute::Float {
            value: format!("0x{:04X}", value.0.to_bits()),
            ty: MlirType::Float(MlirFloatType::F16),
        }))
    }
}

/// Carry a `mir.func`'s `#[inline]` request as the LLVM attribute keyword that
/// `mir-lower` emits for it on the native path.
struct InlineIntentTranslation;

impl AttributeTranslation for InlineIntentTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let intent = source
            .downcast_ref::<InlineIntentAttr>()
            .ok_or_else(|| "expected mir.inline_intent".to_owned())?;
        let keyword = match intent {
            InlineIntentAttr::Hint => "inlinehint",
            InlineIntentAttr::Always => "alwaysinline",
            InlineIntentAttr::Never => "noinline",
        };
        Ok(Some(MlirAttribute::String(keyword.into())))
    }
}

/// Forward translated structure to a standard MLIR operation spelling.
struct RenameOperation(#[doc = "Target MLIR operation spelling."] &'static str);

impl OperationTranslation for RenameOperation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        Ok(vec![renamed(self.0, input)?])
    }
}

/// Map MIR functions, launch contracts, slice ABI, and enum scratch to executable MLIR.
struct FunctionTranslation;

impl OperationTranslation for FunctionTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let mut target = renamed("func.func", input)?;
        move_property(&mut target, ATTR_KEY_SYM_NAME.as_ref(), "sym_name")?;
        move_property(&mut target, "mir_func_type", "function_type")?;
        let grid_constants = crate::grid_constant::take_parameters(ctx, source, &mut target)?;
        lower_alwaysinline_contract(&mut target)?;
        lower_inline_intent_contract(&mut target)?;
        let mut is_kernel = false;
        if let Some(marker) = target.attributes.remove("gpu_kernel") {
            match marker {
                MlirAttribute::String(value) if value == "true" => {
                    is_kernel = true;
                    target
                        .attributes
                        .insert("cute.kernel".into(), MlirAttribute::Unit);
                    target
                        .attributes
                        .insert("gpu.kernel".into(), MlirAttribute::Unit);
                    move_exact_block_contract(&mut target)?;
                    move_launch_bounds_contract(&mut target)?;
                    move_cluster_contract(&mut target)?;
                    flatten_kernel_slice_abi(ctx, source, &mut target, session, &grid_constants)?;
                }
                other => {
                    return Err(format!(
                        "gpu_kernel must be the string `true`, got {other:?}"
                    ));
                }
            }
        }
        lower_helper_linkage(ctx, source, &mut target, is_kernel)?;
        crate::mir::checked::hoist_enum_scratch(&mut target)?;
        Ok(vec![target])
    }
}

/// Keep the same public roots as native LLVM export, internalizing only local definitions.
///
/// The original MIR symbol retains the reserved `#[device]` marker even when
/// target symbol legalization removes it. Kernel entries, device exports, and
/// declarations retain their external ABI. Internal linkage lets LLVM remove
/// zero-size arguments and results from ordinary Rust helper signatures before
/// libNVVM lowers calls to the PTX ABI, which cannot represent those values.
fn lower_helper_linkage(
    ctx: &Context,
    source: Ptr<Operation>,
    target: &mut MlirOperation,
    is_kernel: bool,
) -> Result<(), String> {
    for name in ["llvm.linkage", "sym_visibility"] {
        if target.attributes.contains_key(name) {
            return Err(format!(
                "mir.func `{name}` conflicts with source-derived function linkage"
            ));
        }
    }
    let function = MirFuncOp::wrap(ctx, source).ok_or("expected mir.func")?;
    let source_name = function.get_symbol_name(ctx);
    let is_definition = target
        .regions
        .iter()
        .any(|region| !region.blocks.is_empty());
    if is_definition
        && !is_kernel
        && !reserved_oxide_symbols::is_device_symbol(source_name.as_ref())
    {
        target.attributes.insert(
            "llvm.linkage".into(),
            MlirAttribute::dialect("#llvm.linkage<internal>")?,
        );
    }
    Ok(())
}

/// Lower Rust's `#[inline(always)]` marker at the CUTLASS boundary.
///
/// An arbitrary `alwaysinline = "true"` MLIR attribute survives conversion to
/// `llvm.func`, but LLVM translation does not interpret it. LLVM dialect's
/// `passthrough` spelling carries the explicit Rust request through CUTLASS,
/// including for pointer-bearing helper signatures.
fn lower_alwaysinline_contract(target: &mut MlirOperation) -> Result<(), String> {
    let Some(marker) = target.attributes.remove("alwaysinline") else {
        return Ok(());
    };
    match marker {
        MlirAttribute::String(value) if value == "true" => {
            let function_type = target
                .properties
                .get("function_type")
                .ok_or_else(|| "alwaysinline function is missing function_type".to_owned())?;
            let MlirAttribute::Type(_) = function_type else {
                return Err("alwaysinline function_type is not a type".into());
            };
            if target.attributes.contains_key("passthrough") {
                return Err("alwaysinline function already has a passthrough attribute".into());
            }
            target.attributes.insert(
                "passthrough".into(),
                MlirAttribute::Array(vec![MlirAttribute::String("alwaysinline".into())]),
            );
            Ok(())
        }
        other => Err(format!(
            "alwaysinline must be the string `true`, got {other:?}"
        )),
    }
}

/// Lower the typed `#[inline]` request recorded on `mir.func` as `inline_intent`.
///
/// `InlineIntentTranslation` has already turned the request into the keyword
/// `mir-lower` uses on the native path (`inlinehint`, `alwaysinline`,
/// `noinline`). Put it in the LLVM `passthrough` list so CUTLASS hands the
/// same request to LLVM, sharing the list with the older `alwaysinline`
/// marker when both are present.
fn lower_inline_intent_contract(target: &mut MlirOperation) -> Result<(), String> {
    let Some(keyword) = target
        .attributes
        .remove(dialect_mir::ops::function::INLINE_INTENT_ATTR_KEY)
    else {
        return Ok(());
    };
    let MlirAttribute::String(keyword) = keyword else {
        return Err(format!(
            "inline_intent must translate to an LLVM attribute keyword, got {keyword:?}"
        ));
    };
    let entry = MlirAttribute::String(keyword);
    match target.attributes.get_mut("passthrough") {
        None => {
            target
                .attributes
                .insert("passthrough".into(), MlirAttribute::Array(vec![entry]));
        }
        Some(MlirAttribute::Array(entries)) => {
            if !entries.contains(&entry) {
                entries.push(entry);
            }
        }
        Some(other) => {
            return Err(format!(
                "function passthrough attribute is not an array: {other:?}"
            ));
        }
    }
    Ok(())
}

/// Move cuda-oxide's three scalar block-contract attributes to the one array
/// attribute consumed by CUTLASS/NVVM. A partial shape is never a valid launch
/// contract, so fail instead of quietly dropping an axis.
fn move_exact_block_contract(target: &mut MlirOperation) -> Result<(), String> {
    let dimensions = ["reqntid_x", "reqntid_y", "reqntid_z"]
        .map(|name| take_positive_i32_attribute(target, name))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    match dimensions.as_slice() {
        [None, None, None] => Ok(()),
        [Some(x), Some(y), Some(z)] => {
            // ptxas rejects .maxntid and .reqntid on the same entry. The exact
            // shape is stronger, matching the native exporter's choice.
            target.attributes.remove("maxntid");
            target.attributes.insert(
                "nvvm.reqntid".into(),
                MlirAttribute::DenseI32Array(vec![*x, *y, *z]),
            );
            Ok(())
        }
        _ => Err("kernel reqntid_x/y/z must be present together".into()),
    }
}

/// Occupancy bounds are semantic NVVM attributes, not opaque source metadata.
/// In particular, minctasm determines the initial register budget needed by
/// setmaxnreg; leaving it unqualified lets ptxas discard register reallocation.
fn move_launch_bounds_contract(target: &mut MlirOperation) -> Result<(), String> {
    if let Some(maximum) = take_positive_i32_attribute(target, "maxntid")? {
        target.attributes.insert(
            "nvvm.maxntid".into(),
            MlirAttribute::DenseI32Array(vec![maximum, 1, 1]),
        );
    }
    if let Some(minimum) = take_positive_i32_attribute(target, "minctasm")? {
        target.attributes.insert(
            "nvvm.minctasm".into(),
            MlirAttribute::Integer {
                value: minimum as i128,
                ty: MlirType::Integer(32),
            },
        );
    }
    Ok(())
}

/// Preserve the compile-time cluster contract for collective two-CTA MMA.
fn move_cluster_contract(target: &mut MlirOperation) -> Result<(), String> {
    let dimensions = ["cluster_dim_x", "cluster_dim_y", "cluster_dim_z"]
        .map(|name| take_positive_i32_attribute(target, name))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    match dimensions.as_slice() {
        [None, None, None] => Ok(()),
        [Some(x), Some(y), Some(z)] => {
            target.attributes.insert(
                "nvvm.cluster_dim".into(),
                MlirAttribute::DenseI32Array(vec![*x, *y, *z]),
            );
            Ok(())
        }
        _ => Err("kernel cluster_dim_x/y/z must be present together".into()),
    }
}

/// Remove a named launch attribute and require a positive i32-representable value.
fn take_positive_i32_attribute(
    target: &mut MlirOperation,
    name: &str,
) -> Result<Option<i32>, String> {
    let Some(attribute) = target.attributes.remove(name) else {
        return Ok(None);
    };
    let MlirAttribute::Integer { value, .. } = attribute else {
        return Err(format!("{name} must be an integer attribute"));
    };
    let value = i32::try_from(value).map_err(|_| format!("{name} does not fit in i32"))?;
    if value <= 0 {
        return Err(format!("{name} must be positive, got {value}"));
    }
    Ok(Some(value))
}

/// Pass each Rust slice as two launch arguments: pointer, then length.
/// Rebuild the slice at kernel entry so the body can keep using its original value.
///
/// ```text
/// Rust arguments:   slice,        descriptor
/// Launch arguments: pointer, len, descriptor
/// Argument index:   0        1    2
/// ```
///
/// Grid-constant attributes must follow the descriptor to its new argument index.
fn flatten_kernel_slice_abi(
    ctx: &Context,
    source: Ptr<Operation>,
    target: &mut MlirOperation,
    session: &mut TranslationSession<'_>,
    grid_constants: &std::collections::BTreeMap<usize, crate::grid_constant::Parameter>,
) -> Result<(), String> {
    let source_function =
        MirFuncOp::wrap(ctx, source).ok_or_else(|| "expected mir.func".to_owned())?;
    let source_type = source_function.get_type(ctx);
    let source_type_ref = source_type.deref(ctx);
    let source_type = type_cast::<dyn FunctionTypeInterface>(&*source_type_ref)
        .ok_or_else(|| "mir.func type does not implement FunctionTypeInterface".to_owned())?;
    let source_inputs = source_type.arg_types();

    let function_type = target
        .properties
        .get("function_type")
        .ok_or_else(|| "func.func is missing function_type".to_owned())?;
    let MlirAttribute::Type(MlirType::Function {
        inputs: target_inputs,
        results: target_results,
    }) = function_type
    else {
        return Err("func.func function_type is not an MLIR function type".into());
    };
    if source_inputs.len() != target_inputs.len() {
        return Err(format!(
            "mir.func has {} source inputs but {} translated inputs",
            source_inputs.len(),
            target_inputs.len()
        ));
    }

    let Some(region) = target.regions.first_mut() else {
        return Err("kernel func.func has no body region".into());
    };
    let Some(entry) = region.blocks.first_mut() else {
        return Err("kernel func.func has no entry block".into());
    };
    if entry.arguments.len() != target_inputs.len() {
        return Err(format!(
            "kernel entry has {} arguments but function_type has {} inputs",
            entry.arguments.len(),
            target_inputs.len()
        ));
    }

    let pointer_type = MlirType::dialect("!llvm.ptr")?;
    let length_type = MlirType::Integer(64);
    let mut flattened_types = Vec::new();
    let mut flattened_arguments = Vec::new();
    let mut prologue = Vec::new();
    let mut argument_attributes = Vec::new();

    for (index, ((source_type, target_type), original_argument)) in source_inputs
        .into_iter()
        .zip(target_inputs.iter())
        .zip(entry.arguments.iter())
        .enumerate()
    {
        let source_type_ref = source_type.deref(ctx);
        let is_plain_slice = source_type_ref.downcast_ref::<MirSliceType>().is_some();
        let disjoint_slice = source_type_ref.downcast_ref::<MirDisjointSliceType>();
        if let Some(disjoint_slice) = disjoint_slice
            && !disjoint_slice.space_types().is_empty()
        {
            return Err(
                "kernel ABI flattening for disjoint slices with runtime index-space fields is not implemented"
                    .into(),
            );
        }
        if !is_plain_slice && disjoint_slice.is_none() {
            flattened_types.push(target_type.clone());
            flattened_arguments.push(original_argument.clone());
            argument_attributes.push(match grid_constants.get(&index) {
                Some(parameter) => parameter.attributes()?,
                None => MlirAttribute::Dictionary(Default::default()),
            });
            continue;
        }

        let expected = MlirType::dialect("!llvm.struct<(!llvm.ptr, i64)>")?;
        if target_type != &expected || original_argument.ty != expected {
            return Err(format!(
                "kernel slice argument translated to {target_type:?}; expected {expected:?}"
            ));
        }

        let pointer_id = session.fresh_value();
        let length_id = session.fresh_value();
        flattened_types.extend([pointer_type.clone(), length_type.clone()]);
        argument_attributes.extend([
            MlirAttribute::Dictionary(Default::default()),
            MlirAttribute::Dictionary(Default::default()),
        ]);
        flattened_arguments.extend([
            MlirBlockArgument {
                id: pointer_id,
                ty: pointer_type.clone(),
                location: original_argument.location.clone(),
            },
            MlirBlockArgument {
                id: length_id,
                ty: length_type.clone(),
                location: original_argument.location.clone(),
            },
        ]);

        let undef_id = session.fresh_value();
        let partial_id = session.fresh_value();
        let mut undef = MlirOperation::new("llvm.mlir.undef")?;
        undef.results.push(MlirResult {
            id: undef_id,
            ty: expected.clone(),
        });
        undef.location = original_argument.location.clone();

        let mut insert_pointer = MlirOperation::new("llvm.insertvalue")?;
        insert_pointer.results.push(MlirResult {
            id: partial_id,
            ty: expected.clone(),
        });
        insert_pointer.operands.extend([
            MlirValueUse {
                id: undef_id,
                ty: expected.clone(),
            },
            MlirValueUse {
                id: pointer_id,
                ty: pointer_type.clone(),
            },
        ]);
        insert_pointer
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![0]));
        insert_pointer.location = original_argument.location.clone();

        let mut insert_length = MlirOperation::new("llvm.insertvalue")?;
        insert_length.results.push(MlirResult {
            // The aggregate argument no longer defines this ID, so the final
            // insert can define it without changing any existing body use.
            id: original_argument.id,
            ty: expected.clone(),
        });
        insert_length.operands.extend([
            MlirValueUse {
                id: partial_id,
                ty: expected.clone(),
            },
            MlirValueUse {
                id: length_id,
                ty: length_type.clone(),
            },
        ]);
        insert_length
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![1]));
        insert_length.location = original_argument.location.clone();
        prologue.extend([undef, insert_pointer, insert_length]);
    }

    target.properties.insert(
        "function_type".into(),
        MlirAttribute::Type(MlirType::Function {
            inputs: flattened_types,
            results: target_results.clone(),
        }),
    );
    entry.arguments = flattened_arguments;
    prologue.append(&mut entry.operations);
    entry.operations = prologue;
    if !grid_constants.is_empty() {
        if target.properties.contains_key("arg_attrs")
            || target.attributes.contains_key("arg_attrs")
        {
            return Err("grid-constant kernel already has argument attributes".into());
        }
        target.properties.insert(
            "arg_attrs".into(),
            MlirAttribute::Array(argument_attributes),
        );
    }
    Ok(())
}

/// Rebuild conditional branch successor operand segments for cf.cond_br.
struct ConditionalBranchTranslation;

impl OperationTranslation for ConditionalBranchTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let source = MirCondBranchOp::new(source);
        let segment_sizes = source.get_operand_segment_sizes(ctx).0;
        if segment_sizes.len() != 3 || segment_sizes[0] != 1 {
            return Err(format!(
                "mir.cond_br expected [1, true_args, false_args] operand segments, got {segment_sizes:?}"
            ));
        }
        let mut target = renamed("cf.cond_br", input)?;
        target.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(
                segment_sizes
                    .into_iter()
                    .map(|size| i32::try_from(size).map_err(|_| "operand segment exceeds i32"))
                    .collect::<Result<_, _>>()?,
            ),
        );
        Ok(vec![target])
    }
}

/// Select integer-bit-preserving or floating-point constant translation.
enum ConstantTranslation {
    /// Preserve integer width and bits in a signless arith.constant.
    Integer,
    /// Preserve the floating-point constant's target format.
    Float,
}

impl OperationTranslation for ConstantTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let mut target = renamed("arith.constant", input)?;
        let value = match self {
            Self::Integer => target
                .attributes
                .remove("value")
                .ok_or_else(|| "mir.constant has no translated `value`".to_owned())?,
            Self::Float => {
                let keys = ["float_value_f16", "float_value", "float_value_f64"];
                let present = keys
                    .into_iter()
                    .filter_map(|key| target.attributes.remove(key).map(|value| (key, value)))
                    .collect::<Vec<_>>();
                if present.len() != 1 {
                    return Err(format!(
                        "mir.float_constant expected one value, found {}",
                        present.len()
                    ));
                }
                present.into_iter().next().unwrap().1
            }
        };
        target.properties.insert("value".into(), value);
        Ok(vec![target])
    }
}

/// Select integer or floating arithmetic while retaining source signedness where needed.
enum NumericBinaryTranslation {
    /// Emit addition in the source integer or floating-point domain.
    Add,
    /// Emit subtraction in the source integer or floating-point domain.
    Sub,
    /// Emit multiplication in the source integer or floating-point domain.
    Mul,
    /// Choose floating, signed-integer, or unsigned-integer division from the source type.
    Div,
    /// Choose floating, signed-integer, or unsigned-integer remainder from the source type.
    Rem,
}

impl OperationTranslation for NumericBinaryTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let scalar = source_operand_scalar(ctx, source, 0)?;
        let name = match (self, scalar) {
            (Self::Add, Scalar::Integer { .. }) => "arith.addi",
            (Self::Add, Scalar::Float { .. }) => "arith.addf",
            (Self::Sub, Scalar::Integer { .. }) => "arith.subi",
            (Self::Sub, Scalar::Float { .. }) => "arith.subf",
            (Self::Mul, Scalar::Integer { .. }) => "arith.muli",
            (Self::Mul, Scalar::Float { .. }) => "arith.mulf",
            (Self::Div, Scalar::Integer { signed: true, .. }) => "arith.divsi",
            (Self::Div, Scalar::Integer { signed: false, .. }) => "arith.divui",
            (Self::Div, Scalar::Float { .. }) => "arith.divf",
            (Self::Rem, Scalar::Integer { signed: true, .. }) => "arith.remsi",
            (Self::Rem, Scalar::Integer { signed: false, .. }) => "arith.remui",
            (Self::Rem, Scalar::Float { .. }) => "arith.remf",
        };
        Ok(vec![renamed(name, input)?])
    }
}

/// Map a bitwise integer operation to its explicit target spelling.
struct IntegerBinaryTranslation(#[doc = "Target bitwise integer operation spelling."] &'static str);

impl OperationTranslation for IntegerBinaryTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        source_integer(ctx, source, 0)?;
        Ok(vec![renamed(self.0, input)?])
    }
}

/// Match Rust shifts by casting and masking the count before selecting arithmetic or logical shift.
struct ShiftTranslation {
    /// Select right shift; otherwise emit a left shift.
    right: bool,
}

impl OperationTranslation for ShiftTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let integer = source_integer(ctx, source, 0)?;
        let name = if !self.right {
            "arith.shli"
        } else if integer.signedness() == Signedness::Signed {
            "arith.shrsi"
        } else {
            "arith.shrui"
        };
        let width = integer.width();
        let count = source_integer(ctx, source, 1)?;
        let count_width = count.width();
        if !width.is_power_of_two() || width > 128 {
            return Err(format!(
                "Rust shift requires a power-of-two integer width up to 128, got {width}"
            ));
        }
        // Rust permits a differently typed RHS, and release/wrapping shifts
        // mask it by width-1. MLIR/LLVM require equal types and otherwise
        // produce poison for oversized counts. Match the native MIR lowering.
        let ty = MlirType::Integer(width);
        let mut operations = vec![];
        if width != count_width {
            let id = session.fresh_value();
            let mut cast = MlirOperation::new(if width > count_width {
                "arith.extui"
            } else {
                "arith.trunci"
            })?;
            cast.operands.push(input.operands[1].clone());
            cast.results.push(MlirResult { id, ty: ty.clone() });
            cast.location = input.location.clone();
            operations.push(cast);
            input.operands[1] = MlirValueUse { id, ty: ty.clone() };
        }
        let mask_id = session.fresh_value();
        let mut mask = MlirOperation::new("arith.constant")?;
        mask.results.push(MlirResult {
            id: mask_id,
            ty: ty.clone(),
        });
        mask.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: i128::from(width - 1),
                ty: ty.clone(),
            },
        );
        mask.location = input.location.clone();
        operations.push(mask);
        let count_id = session.fresh_value();
        let mut masked = MlirOperation::new("arith.andi")?;
        masked.operands = vec![
            input.operands[1].clone(),
            MlirValueUse {
                id: mask_id,
                ty: ty.clone(),
            },
        ];
        masked.results.push(MlirResult {
            id: count_id,
            ty: ty.clone(),
        });
        masked.location = input.location.clone();
        operations.push(masked);
        input.operands[1] = MlirValueUse { id: count_id, ty };
        operations.push(renamed(name, input)?);
        Ok(operations)
    }
}

/// Negate using integer zero subtraction or floating-point negation as appropriate.
struct NegTranslation;

impl OperationTranslation for NegTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        match source_operand_scalar(ctx, source, 0)? {
            Scalar::Float { .. } => Ok(vec![renamed("arith.negf", input)?]),
            Scalar::Integer { .. } => {
                let result_type = input
                    .results
                    .first()
                    .ok_or_else(|| "mir.neg has no result".to_owned())?
                    .ty
                    .clone();
                let zero_id = session.fresh_value();
                let mut zero = MlirOperation::new("arith.constant")?;
                zero.results.push(crate::MlirResult {
                    id: zero_id,
                    ty: result_type.clone(),
                });
                zero.properties.insert(
                    "value".into(),
                    MlirAttribute::Integer {
                        value: 0,
                        ty: result_type.clone(),
                    },
                );
                zero.location = input.location.clone();

                let mut neg = renamed("arith.subi", input)?;
                neg.operands.insert(
                    0,
                    crate::MlirValueUse {
                        id: zero_id,
                        ty: result_type,
                    },
                );
                Ok(vec![zero, neg])
            }
        }
    }
}

/// Complement every bit of the source integer width.
struct NotTranslation;

impl OperationTranslation for NotTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let integer = source_integer(ctx, source, 0)?;
        if integer.width() > 128 {
            return Err(format!(
                "mir.not wider than 128 bits is not supported yet (got i{})",
                integer.width()
            ));
        }
        let result_type = input
            .results
            .first()
            .ok_or_else(|| "mir.not has no result".to_owned())?
            .ty
            .clone();
        let ones_id = session.fresh_value();
        let mut ones = MlirOperation::new("arith.constant")?;
        ones.results.push(crate::MlirResult {
            id: ones_id,
            ty: result_type.clone(),
        });
        ones.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: -1,
                ty: result_type.clone(),
            },
        );
        ones.location = input.location.clone();

        let mut not = renamed("arith.xori", input)?;
        not.operands.push(crate::MlirValueUse {
            id: ones_id,
            ty: result_type,
        });
        Ok(vec![ones, not])
    }
}

/// Choose ordered/unordered floating or signed/unsigned integer predicates from source types.
enum ComparisonTranslation {
    /// Compare less-than with source signedness or ordered floating semantics.
    Lt,
    /// Compare less-than-or-equal with source signedness or ordered floating semantics.
    Le,
    /// Compare greater-than with source signedness or ordered floating semantics.
    Gt,
    /// Compare greater-than-or-equal with source signedness or ordered floating semantics.
    Ge,
    /// Compare equality, false for unordered floating operands.
    Eq,
    /// Compare inequality, true for unordered floating operands.
    Ne,
}

impl OperationTranslation for ComparisonTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let scalar = source_operand_scalar(ctx, source, 0)?;
        let (name, predicate) = match scalar {
            Scalar::Integer { signed, .. } => {
                let predicate = match (self, signed) {
                    (Self::Eq, _) => 0,
                    (Self::Ne, _) => 1,
                    (Self::Lt, true) => 2,
                    (Self::Le, true) => 3,
                    (Self::Gt, true) => 4,
                    (Self::Ge, true) => 5,
                    (Self::Lt, false) => 6,
                    (Self::Le, false) => 7,
                    (Self::Gt, false) => 8,
                    (Self::Ge, false) => 9,
                };
                ("arith.cmpi", predicate)
            }
            Scalar::Float { .. } => {
                // Rust relational comparisons are ordered. `!=` is true for
                // unordered values, so it uses UNE rather than ONE.
                let predicate = match self {
                    Self::Eq => 1,
                    Self::Gt => 2,
                    Self::Ge => 3,
                    Self::Lt => 4,
                    Self::Le => 5,
                    Self::Ne => 13,
                };
                ("arith.cmpf", predicate)
            }
        };
        let mut target = renamed(name, input)?;
        target.properties.insert(
            "predicate".into(),
            MlirAttribute::Integer {
                value: predicate,
                ty: MlirType::Integer(64),
            },
        );
        Ok(vec![target])
    }
}

/// Translate scalar, bitwise, and pointer conversions without losing source address semantics.
struct CastTranslation;

impl OperationTranslation for CastTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let kind = MirCastOp::new(source)
            .get_attr_cast_kind(ctx)
            .ok_or_else(|| "mir.cast has no cast_kind".to_owned())?
            .clone();
        if kind == MirCastKindAttr::PointerCoercionUnsize {
            return super::memory::unsize_array_pointer(ctx, source, input, session);
        }
        if kind == MirCastKindAttr::Transmute
            && (super::storage::is_aggregate(ctx, source.deref(ctx).get_operand(0).get_type(ctx))
                || super::storage::is_aggregate(ctx, source.deref(ctx).get_result(0).get_type(ctx)))
        {
            return super::storage::transmute(ctx, source, input, session);
        }
        // Rust's pointer address (including ptr::addr's transmute) is generic,
        // unlike the explicit NVVM shared-offset instruction boundary.
        let pointer_transmute = kind == MirCastKindAttr::Transmute
            && source_pointer_address_space(ctx, source, true).is_ok();
        if kind == MirCastKindAttr::PointerExposeAddress || pointer_transmute {
            let space = source_pointer_address_space(ctx, source, true)?;
            let Scalar::Integer { width, .. } = source_result_scalar(ctx, source, 0)? else {
                return Err("pointer address requires an integer result".into());
            };
            if pointer_transmute && width != 64 {
                return Err(
                    "pointer transmute requires the 64-bit Rust address representation".into(),
                );
            }
            let mut ops = vec![];
            if space != 0 {
                let ty = MlirType::dialect("!llvm.ptr")?;
                let id = session.fresh_value();
                let mut cast = MlirOperation::new("llvm.addrspacecast")?;
                cast.operands = input.operands.clone();
                cast.results.push(MlirResult { id, ty: ty.clone() });
                cast.location = input.location.clone();
                ops.push(cast);
                input.operands = vec![MlirValueUse { id, ty }];
            }
            ops.push(renamed("llvm.ptrtoint", input)?);
            return Ok(ops);
        }
        let name = match kind {
            MirCastKindAttr::IntToInt => {
                let source_scalar = source_operand_scalar(ctx, source, 0)?;
                let destination_scalar = source_result_scalar(ctx, source, 0)?;
                let (
                    Scalar::Integer {
                        width: from,
                        signed,
                        ..
                    },
                    Scalar::Integer { width: to, .. },
                ) = (source_scalar, destination_scalar)
                else {
                    return Err("IntToInt cast does not have integer types".into());
                };
                if from < to {
                    if signed { "arith.extsi" } else { "arith.extui" }
                } else if from > to {
                    "arith.trunci"
                } else {
                    // Pliron signedness has disappeared from the target type;
                    // this remains a bit-for-bit identity operation.
                    "arith.bitcast"
                }
            }
            MirCastKindAttr::IntToFloat => match source_operand_scalar(ctx, source, 0)? {
                Scalar::Integer { signed: true, .. } => "arith.sitofp",
                Scalar::Integer { signed: false, .. } => "arith.uitofp",
                _ => return Err("IntToFloat cast does not have an integer source".into()),
            },
            MirCastKindAttr::FloatToInt => {
                return Ok(vec![saturating_float_to_int(ctx, source, input)?]);
            }
            MirCastKindAttr::FloatToFloat => {
                let source_scalar = source_operand_scalar(ctx, source, 0)?;
                let destination_scalar = source_result_scalar(ctx, source, 0)?;
                let (Scalar::Float { width: from }, Scalar::Float { width: to }) =
                    (source_scalar, destination_scalar)
                else {
                    return Err("FloatToFloat cast does not have float types".into());
                };
                if from < to {
                    "arith.extf"
                } else if from > to {
                    "arith.truncf"
                } else {
                    "arith.bitcast"
                }
            }
            MirCastKindAttr::Transmute => {
                let source_scalar = source_operand_scalar(ctx, source, 0)?;
                let destination_scalar = source_result_scalar(ctx, source, 0)?;
                match (source_scalar, destination_scalar) {
                    (Scalar::Integer { width: from, .. }, Scalar::Float { width: to })
                    | (Scalar::Float { width: from }, Scalar::Integer { width: to, .. })
                        if from == to =>
                    {
                        "arith.bitcast"
                    }
                    (from, to) => {
                        return Err(format!(
                            "scalar Transmute requires equal-width integer/float types, got {} bits to {} bits",
                            from.width(),
                            to.width()
                        ));
                    }
                }
            }
            MirCastKindAttr::PtrToPtr | MirCastKindAttr::PointerCoercionMutToConst => {
                let operand_type = source.deref(ctx).get_operand(0).get_type(ctx);
                if operand_type.deref(ctx).is::<MirSliceType>() {
                    let result_type = source.deref(ctx).get_result(0).get_type(ctx);
                    if result_type.deref(ctx).is::<MirSliceType>() {
                        // Pointer provenance and mutability do not change the
                        // canonical {generic data pointer, i64 length} layout.
                        return Ok(vec![renamed("builtin.unrealized_conversion_cast", input)?]);
                    }
                    if source_pointer_address_space(ctx, source, false)? != 0 {
                        return Err(
                            "slice-to-pointer cast requires a generic pointer result".into()
                        );
                    }
                    // Rust slice::as_ptr casts the fat pointer to its data
                    // pointer; extracting field zero must not load through it
                    // or accidentally reinterpret the length as an address.
                    let mut extract = renamed("llvm.extractvalue", input)?;
                    extract
                        .properties
                        .insert("position".into(), MlirAttribute::DenseI64Array(vec![0]));
                    return Ok(vec![extract]);
                }
                let source_address_space = source_pointer_address_space(ctx, source, true)?;
                let result_address_space = source_pointer_address_space(ctx, source, false)?;
                if source_address_space == result_address_space {
                    "builtin.unrealized_conversion_cast"
                } else {
                    "llvm.addrspacecast"
                }
            }
            MirCastKindAttr::PointerExposeAddress => {
                source_pointer_address_space(ctx, source, true)?;
                source_result_scalar(ctx, source, 0)?;
                "llvm.ptrtoint"
            }
            MirCastKindAttr::PointerWithExposedProvenance => {
                source_operand_scalar(ctx, source, 0)?;
                source_pointer_address_space(ctx, source, false)?;
                "llvm.inttoptr"
            }
            other => {
                return Err(format!(
                    "cast kind {other:?} needs a pointer or aggregate mapping pack"
                ));
            }
        };
        Ok(vec![renamed(name, input)?])
    }
}

/// Translate Rust's `as` float-to-integer cast, which saturates.
///
/// Out-of-range values clamp to the integer's MIN/MAX and NaN becomes 0.
/// `arith.fptosi`/`arith.fptoui` give poison for those inputs, so this calls
/// the LLVM `fptosi.sat`/`fptoui.sat` intrinsics, matching mir-lower.
fn saturating_float_to_int(
    ctx: &Context,
    source: Ptr<Operation>,
    input: OperationInput,
) -> Result<MlirOperation, String> {
    let Scalar::Float { width: float_width } = source_operand_scalar(ctx, source, 0)? else {
        return Err("FloatToInt cast does not have a floating-point source".into());
    };
    let Scalar::Integer { width, signed } = source_result_scalar(ctx, source, 0)? else {
        return Err("FloatToInt cast does not have an integer result".into());
    };
    if width > 64 {
        return Err(format!(
            "FloatToInt: integer destinations wider than 64 bits are not supported on the device, got i{width}"
        ));
    }
    let sign = if signed { "s" } else { "u" };
    let intrinsic = format!("llvm.fpto{sign}i.sat.i{width}.f{float_width}");
    let mut target = renamed("llvm.call_intrinsic", input)?;
    target
        .properties
        .insert("intrin".into(), MlirAttribute::String(intrinsic));
    target.properties.insert(
        "op_bundle_sizes".into(),
        MlirAttribute::DenseI32Array(vec![]),
    );
    target.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![1, 0]),
    );
    Ok(target)
}

/// Recover a source MIR pointer's numeric address space before opaque-pointer translation.
fn source_pointer_address_space(
    ctx: &Context,
    source: Ptr<Operation>,
    operand: bool,
) -> Result<u32, String> {
    let ty = if operand {
        source.deref(ctx).get_operand(0).get_type(ctx)
    } else {
        source.deref(ctx).get_result(0).get_type(ctx)
    };
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .map(|pointer| pointer.address_space())
        .ok_or_else(|| "expected a pointer type".to_owned())
}

/// Source width and signedness needed to choose scalar MLIR operations.
#[derive(Clone, Copy)]
enum Scalar {
    /// An integer's original bit width and signedness before MLIR erases signedness.
    Integer {
        #[doc = "Integer storage width in bits."]
        width: u32,
        #[doc = "Whether division, shifts, and comparisons use signed integer semantics."]
        signed: bool,
    },
    /// A source floating-point scalar with its original bit width.
    Float {
        #[doc = "Floating-point storage width in bits."]
        width: u32,
    },
}

impl Scalar {
    /// Return the source scalar's bit width before signedness is erased.
    fn width(self) -> u32 {
        match self {
            Self::Integer { width, .. } | Self::Float { width } => width,
        }
    }
}

/// Read scalar classification from a selected source operand.
fn source_operand_scalar(
    ctx: &Context,
    source: Ptr<Operation>,
    index: usize,
) -> Result<Scalar, String> {
    let ty = source.deref(ctx).get_operand(index).get_type(ctx);
    scalar(ctx, ty)
}

/// Read scalar classification from a selected source result.
fn source_result_scalar(
    ctx: &Context,
    source: Ptr<Operation>,
    index: usize,
) -> Result<Scalar, String> {
    let ty = source.deref(ctx).get_result(index).get_type(ctx);
    scalar(ctx, ty)
}

/// Classify reviewed integer and floating-point source types or reject unsupported scalars.
fn scalar(ctx: &Context, ty: TypeHandle) -> Result<Scalar, String> {
    let ty_ref = ty.deref(ctx);
    if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        return Ok(Scalar::Integer {
            width: integer.width(),
            // Under the shared integer contract, signless integers (notably bool) take
            // the unsigned path for operations where signedness matters.
            signed: integer.signedness() == Signedness::Signed,
        });
    }
    if !type_impls::<dyn FloatTypeInterface>(&*ty_ref) {
        return Err("expected a scalar integer or floating-point type".into());
    }
    let width = if ty_ref.is::<FP16Type>() || ty_ref.is::<MirFP16Type>() {
        16
    } else if ty_ref.is::<FP32Type>() {
        32
    } else if ty_ref.is::<FP64Type>() {
        64
    } else {
        return Err("unsupported floating-point type".into());
    };
    Ok(Scalar::Float { width })
}

/// Require an integer source value and retain its bit width and signedness.
fn source_integer(
    ctx: &Context,
    source: Ptr<Operation>,
    index: usize,
) -> Result<IntegerType, String> {
    let ty = source.deref(ctx).get_operand(index).get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<IntegerType>()
        .cloned()
        .ok_or_else(|| "expected an integer operand".to_owned())
}

/// Retain translated operation structure while changing its target operation name.
fn renamed(name: &str, input: OperationInput) -> Result<MlirOperation, String> {
    let mut target = MlirOperation::new(name)?;
    target.results = input.results;
    target.operands = input.operands;
    target.successors = input.successors;
    target.regions = input.regions;
    target.attributes = input.attributes;
    target.location = input.location;
    Ok(target)
}

/// Remove an optional source attribute and reattach it under the target property name.
fn move_property(
    operation: &mut MlirOperation,
    source_name: &str,
    target_name: &str,
) -> Result<(), String> {
    let value = operation
        .attributes
        .remove(source_name)
        .ok_or_else(|| format!("missing required `{source_name}` property"))?;
    operation.properties.insert(target_name.into(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Scalar, launch-contract, and control-flow mapping tests using ordinary MIR fixtures.

    use std::num::NonZero;

    /// Reject incomplete cluster shapes and dimensions that are zero, negative, or too large.
    #[test]
    fn cluster_contract_rejects_partial_and_nonpositive_dimensions() {
        use crate::{MlirAttribute, MlirOperation, MlirType};
        let mut kernel = MlirOperation::new("func.func").unwrap();
        kernel.attributes.insert(
            "cluster_dim_x".into(),
            MlirAttribute::Integer {
                value: 2,
                ty: MlirType::Integer(32),
            },
        );
        assert!(
            super::move_cluster_contract(&mut kernel)
                .unwrap_err()
                .contains("present together")
        );
        kernel.attributes.insert(
            "cluster_dim_x".into(),
            MlirAttribute::Integer {
                value: 0,
                ty: MlirType::Integer(32),
            },
        );
        assert!(
            super::move_cluster_contract(&mut kernel)
                .unwrap_err()
                .contains("positive")
        );
    }

    use crate::{
        CutlassFullCuteMlir22, MlirConsumerProfile,
        profile::render_mapping_module_without_cutlass_envelope,
    };
    use dialect_mir::{
        attributes::MirCastKindAttr,
        ops::{
            MirAddOp, MirCastOp, MirCondBranchOp, MirConstantOp, MirDivOp, MirFloatConstantOp,
            MirFuncOp, MirGotoOp, MirLtOp, MirReturnOp,
        },
        types::MirPtrType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{FPSingleAttr, IntegerAttr, StringAttr, TypeAttr},
            op_interfaces::{
                OperandSegmentInterface, SingleBlockRegionInterface, SymbolOpInterface,
            },
            ops::ModuleOp,
            types::{FP16Type, FP32Type, FP64Type, FunctionType, IntegerType, Signedness},
        },
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, TypedHandle},
        utils::apint::APInt,
        value::Value,
    };

    use dialect_mir::attributes::InlineIntentAttr;

    /// Create a source context with the MIR dialect needed by scalar translation tests.
    fn context() -> Context {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        ctx
    }

    /// Create a typed MIR function with an entry block and append it to the test module.
    fn append_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        inputs: Vec<TypeHandle>,
        results: Vec<TypeHandle>,
    ) -> (Ptr<Operation>, pliron::context::Ptr<pliron::region::Region>) {
        let function_type = FunctionType::get(ctx, inputs, results);
        let operation = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, operation, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, Identifier::try_from(name).unwrap());
        module.append_operation(ctx, operation, 0);
        let region = operation.deref(ctx).get_region(0);
        (operation, region)
    }

    /// Construct a source operation of the requested concrete kind and append it to the block.
    fn operation<O: Op>(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        results: Vec<TypeHandle>,
        operands: Vec<Value>,
    ) -> Ptr<Operation> {
        let operation =
            Operation::new(ctx, O::get_concrete_op_info(), results, operands, vec![], 0);
        operation.insert_at_back(block, ctx);
        operation
    }

    /// Export the scalar fixture and render only the operation-mapping module.
    fn translate(ctx: &mut Context, module: &ModuleOp) -> String {
        let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
        let target = profile.translate_module(ctx, module).unwrap();
        render_mapping_module_without_cutlass_envelope(
            &target,
            module.get_symbol_name(ctx).as_ref(),
        )
    }

    /// Obtain a source integer type with explicit width and signedness.
    fn integer(ctx: &mut Context, width: u32, signedness: Signedness) -> TypedHandle<IntegerType> {
        IntegerType::get(ctx, width, signedness)
    }

    /// Check Rust shift counts are converted and masked before MLIR receives equal-width operands.
    #[test]
    fn shifts_cast_and_mask_counts_before_emitting_equal_width_operands() {
        use dialect_mir::ops::{MirShlOp, MirShrOp};
        for (width, count_width, signed, right, name, cast) in [
            (64, 32, false, true, "arith.shrui", Some("arith.extui")),
            (16, 32, false, false, "arith.shli", Some("arith.trunci")),
            (32, 64, true, true, "arith.shrsi", Some("arith.trunci")),
            (32, 32, false, false, "arith.shli", None),
        ] {
            let mut ctx = context();
            let module = ModuleOp::new(&mut ctx, Identifier::try_from("shift").unwrap());
            let lhs = integer(
                &mut ctx,
                width,
                if signed {
                    Signedness::Signed
                } else {
                    Signedness::Unsigned
                },
            );
            let rhs = integer(&mut ctx, count_width, Signedness::Unsigned);
            let (_, region) = append_function(
                &mut ctx,
                &module,
                "shift",
                vec![lhs.into(), rhs.into()],
                vec![lhs.into()],
            );
            let entry = BasicBlock::new(&mut ctx, None, vec![lhs.into(), rhs.into()]);
            entry.insert_at_back(region, &ctx);
            let args = entry.deref(&ctx).arguments().collect();
            let shift = if right {
                operation::<MirShrOp>(&mut ctx, entry, vec![lhs.into()], args)
            } else {
                operation::<MirShlOp>(&mut ctx, entry, vec![lhs.into()], args)
            };
            let result = shift.deref(&ctx).get_result(0);
            operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![result]);
            let text = translate(&mut ctx, &module);
            let shift_line = text
                .lines()
                .find(|line| line.contains(&format!("\"{name}\"")))
                .unwrap();
            assert!(
                shift_line.contains(&format!(": (i{width}, i{width}) -> (i{width})")),
                "{text}"
            );
            assert!(
                text.contains(&format!("value = {} : i{width}", width - 1)),
                "{text}"
            );
            assert!(text.contains("\"arith.andi\""), "{text}");
            if let Some(cast) = cast {
                assert!(text.contains(&format!("\"{cast}\"")), "{text}");
            }
        }
    }

    /// Check signed and unsigned arithmetic select distinct target instructions after type signedness erasure.
    #[test]
    fn scalar_arithmetic_keeps_signedness_in_the_operation_names() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("core").unwrap());
        let u64_type = integer(&mut ctx, 64, Signedness::Unsigned);
        let i64_type = integer(&mut ctx, 64, Signedness::Signed);
        let i1_type = integer(&mut ctx, 1, Signedness::Signless);
        let f32_type = FP32Type::get(&ctx);
        let f64_type = FP64Type::get(&ctx);

        let (_, region) = append_function(
            &mut ctx,
            &module,
            "scalar",
            vec![
                u64_type.into(),
                u64_type.into(),
                i64_type.into(),
                i64_type.into(),
                f32_type.into(),
            ],
            vec![
                u64_type.into(),
                i64_type.into(),
                i1_type.into(),
                i1_type.into(),
                f32_type.into(),
                f64_type.into(),
            ],
        );
        let entry = BasicBlock::new(
            &mut ctx,
            None,
            vec![
                u64_type.into(),
                u64_type.into(),
                i64_type.into(),
                i64_type.into(),
                f32_type.into(),
            ],
        );
        entry.insert_at_back(region, &ctx);
        let arguments = entry.deref(&ctx).arguments().collect::<Vec<_>>();

        let four = operation::<MirConstantOp>(&mut ctx, entry, vec![u64_type.into()], vec![]);
        MirConstantOp::new(four).set_attr_value(
            &ctx,
            IntegerAttr::new(u64_type, APInt::from_u128(4, NonZero::new(64).unwrap())),
        );
        let four_value = four.deref(&ctx).get_result(0);

        let unsigned_div = operation::<MirDivOp>(
            &mut ctx,
            entry,
            vec![u64_type.into()],
            vec![arguments[0], four_value],
        );
        let signed_div = operation::<MirDivOp>(
            &mut ctx,
            entry,
            vec![i64_type.into()],
            vec![arguments[2], arguments[3]],
        );
        let unsigned_lt = operation::<MirLtOp>(
            &mut ctx,
            entry,
            vec![i1_type.into()],
            vec![arguments[0], arguments[1]],
        );
        let signed_lt = operation::<MirLtOp>(
            &mut ctx,
            entry,
            vec![i1_type.into()],
            vec![arguments[2], arguments[3]],
        );

        let one = operation::<MirFloatConstantOp>(&mut ctx, entry, vec![f32_type.into()], vec![]);
        MirFloatConstantOp::new(one).set_attr_float_value(&ctx, FPSingleAttr::from(1.0));
        let one_value = one.deref(&ctx).get_result(0);
        let sum = operation::<MirAddOp>(
            &mut ctx,
            entry,
            vec![f32_type.into()],
            vec![arguments[4], one_value],
        );
        let cast =
            operation::<MirCastOp>(&mut ctx, entry, vec![f64_type.into()], vec![arguments[4]]);
        MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::FloatToFloat);

        let returns = vec![
            unsigned_div.deref(&ctx).get_result(0),
            signed_div.deref(&ctx).get_result(0),
            unsigned_lt.deref(&ctx).get_result(0),
            signed_lt.deref(&ctx).get_result(0),
            sum.deref(&ctx).get_result(0),
            cast.deref(&ctx).get_result(0),
        ];
        operation::<MirReturnOp>(&mut ctx, entry, vec![], returns);

        let text = translate(&mut ctx, &module);
        let expected = r#""builtin.module"() <{sym_name = "core"}> ({
  ^bb0:
    "func.func"() <{function_type = (i64, i64, i64, i64, f32) -> (i64, i64, i1, i1, f32, f64), sym_name = "scalar"}> ({
      ^bb1(%v0: i64, %v1: i64, %v2: i64, %v3: i64, %v4: f32):
        %v5 = "arith.constant"() <{value = 4 : i64}> : () -> (i64)
        %v6 = "arith.divui"(%v0, %v5) : (i64, i64) -> (i64)
        %v7 = "arith.divsi"(%v2, %v3) : (i64, i64) -> (i64)
        %v8 = "arith.cmpi"(%v0, %v1) <{predicate = 6 : i64}> : (i64, i64) -> (i1)
        %v9 = "arith.cmpi"(%v2, %v3) <{predicate = 2 : i64}> : (i64, i64) -> (i1)
        %v10 = "arith.constant"() <{value = 0x3F800000 : f32}> : () -> (f32)
        %v11 = "arith.addf"(%v4, %v10) : (f32, f32) -> (f32)
        %v12 = "arith.extf"(%v4) : (f32) -> (f64)
        "func.return"(%v6, %v7, %v8, %v9, %v11, %v12) : (i64, i64, i1, i1, f32, f64) -> ()
    }) {llvm.linkage = #llvm.linkage<internal>} : () -> ()
}) : () -> ()
"#;
        assert_eq!(
            crate::profile::normalize_mlir_identifiers(&text),
            crate::profile::normalize_mlir_identifiers(expected)
        );
    }

    /// Check same-width scalar transmutation preserves bits rather than performing numeric conversion.
    #[test]
    fn equal_width_integer_float_transmute_maps_to_bitcast() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("transmute").unwrap());
        let u32_type = integer(&mut ctx, 32, Signedness::Unsigned);
        let f32_type = FP32Type::get(&ctx);
        let (_, region) = append_function(
            &mut ctx,
            &module,
            "bits_to_float",
            vec![u32_type.into()],
            vec![f32_type.into()],
        );
        let entry = BasicBlock::new(&mut ctx, None, vec![u32_type.into()]);
        entry.insert_at_back(region, &ctx);
        let input = entry.deref(&ctx).arguments().next().unwrap();
        let cast = operation::<MirCastOp>(&mut ctx, entry, vec![f32_type.into()], vec![input]);
        MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::Transmute);
        let cast_result = cast.deref(&ctx).get_result(0);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![cast_result]);

        let text = translate(&mut ctx, &module);
        assert!(
            text.contains(r#""arith.bitcast"(%v0) : (i32) -> (f32)"#),
            "{text}"
        );
    }

    /// Build `fn cast(x: from) -> to { x as to }` as a single FloatToInt cast.
    fn float_to_int_module(ctx: &mut Context, from: TypeHandle, to: TypeHandle) -> ModuleOp {
        let module = ModuleOp::new(ctx, Identifier::try_from("float_to_int").unwrap());
        let (_, region) = append_function(ctx, &module, "cast", vec![from], vec![to]);
        let entry = BasicBlock::new(ctx, None, vec![from]);
        entry.insert_at_back(region, ctx);
        let input = entry.deref(ctx).arguments().next().unwrap();
        let cast = operation::<MirCastOp>(ctx, entry, vec![to], vec![input]);
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::FloatToInt);
        let cast_result = cast.deref(ctx).get_result(0);
        operation::<MirReturnOp>(ctx, entry, vec![], vec![cast_result]);
        module
    }

    /// Rust `as` saturates (300.0 as u8 == 255, NaN as i32 == 0), so FloatToInt
    /// must use the `.sat` intrinsics; plain fptosi/fptoui are poison out of range.
    #[test]
    fn float_to_int_casts_saturate_like_rust() {
        for (float, mir_fp16, int, signedness, expected) in [
            (32, false, 32, Signedness::Signed, "llvm.fptosi.sat.i32.f32"),
            (32, false, 8, Signedness::Unsigned, "llvm.fptoui.sat.i8.f32"),
            (64, false, 64, Signedness::Signed, "llvm.fptosi.sat.i64.f64"),
            (
                16,
                false,
                16,
                Signedness::Unsigned,
                "llvm.fptoui.sat.i16.f16",
            ),
            (16, true, 16, Signedness::Signed, "llvm.fptosi.sat.i16.f16"),
        ] {
            let mut ctx = context();
            let from: TypeHandle = match (float, mir_fp16) {
                (16, true) => dialect_mir::types::MirFP16Type::get(&ctx).into(),
                (16, false) => FP16Type::get(&ctx).into(),
                (32, _) => FP32Type::get(&ctx).into(),
                _ => FP64Type::get(&ctx).into(),
            };
            let to = integer(&mut ctx, int, signedness).into();
            let module = float_to_int_module(&mut ctx, from, to);
            let text = translate(&mut ctx, &module);
            assert!(text.contains(r#""llvm.call_intrinsic"(%v0)"#), "{text}");
            assert!(text.contains(expected), "{expected} missing:\n{text}");
            assert!(
                !text.contains("arith.fptosi") && !text.contains("arith.fptoui"),
                "non-saturating conversion emitted:\n{text}"
            );
        }
    }

    /// Integer destinations wider than 64 bits have no device lowering, as in mir-lower.
    #[test]
    fn float_to_int_rejects_destinations_wider_than_64_bits() {
        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            let mut ctx = context();
            let from = FP64Type::get(&ctx).into();
            let to = integer(&mut ctx, 128, signedness).into();
            let module = float_to_int_module(&mut ctx, from, to);
            let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
            let error = format!("{:?}", profile.translate_module(&ctx, &module).unwrap_err());
            assert!(error.contains("wider than 64 bits"), "{error}");
        }
    }

    /// Check FP16 bit extraction and reconstruction use exact bit-preserving scalar operations.
    #[test]
    fn rust_f16_to_bits_and_from_bits_preserve_the_scalar_bit_pattern() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("half_bits").unwrap());
        let half: TypeHandle = dialect_mir::types::MirFP16Type::get(&ctx).into();
        let bits: TypeHandle = integer(&mut ctx, 16, Signedness::Unsigned).into();
        let (_, region) =
            append_function(&mut ctx, &module, "half_bits", vec![half], vec![bits, half]);
        let entry = BasicBlock::new(&mut ctx, None, vec![half]);
        entry.insert_at_back(region, &ctx);
        let input = entry.deref(&ctx).get_argument(0);
        let to_bits = operation::<MirCastOp>(&mut ctx, entry, vec![bits], vec![input]);
        MirCastOp::new(to_bits).set_attr_cast_kind(&ctx, MirCastKindAttr::Transmute);
        let bits_value = to_bits.deref(&ctx).get_result(0);
        let from_bits = operation::<MirCastOp>(&mut ctx, entry, vec![half], vec![bits_value]);
        MirCastOp::new(from_bits).set_attr_cast_kind(&ctx, MirCastKindAttr::Transmute);
        let half_value = from_bits.deref(&ctx).get_result(0);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![bits_value, half_value]);
        let text = translate(&mut ctx, &module);
        assert!(
            text.contains(r#""arith.bitcast"(%v0) : (f16) -> (i16)"#),
            "{text}"
        );
        assert!(
            text.contains(r#""arith.bitcast"(%v1) : (i16) -> (f16)"#),
            "{text}"
        );
    }

    /// Check slice metadata remains intact until the source operation requests only its pointer.
    #[test]
    // Reproduce the importer's Rust reference-to-raw-slice and as_ptr boundaries.
    #[allow(clippy::disallowed_methods)]
    fn slice_as_ptr_preserves_metadata_until_extracting_the_data_pointer() {
        use dialect_mir::{
            attributes::MirPointerKindAuthorityAttr,
            types::{MirFP16Type, MirPointerKind, MirSliceType},
        };
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("slice_pointer").unwrap());
        let half: TypeHandle = MirFP16Type::get(&ctx).into();
        let slice: TypeHandle =
            MirSliceType::get_with_kind(&mut ctx, half, MirPointerKind::SharedRef).into();
        let raw_slice: TypeHandle =
            MirSliceType::get_with_kind(&mut ctx, half, MirPointerKind::RawConst).into();
        let pointer: TypeHandle =
            MirPtrType::get_with_kind(&mut ctx, half, false, 0, MirPointerKind::RawConst).into();
        let (_, region) = append_function(
            &mut ctx,
            &module,
            "slice_pointer",
            vec![slice],
            vec![raw_slice, pointer],
        );
        let entry = BasicBlock::new(&mut ctx, None, vec![slice]);
        entry.insert_at_back(region, &ctx);
        let input = entry.deref(&ctx).get_argument(0);
        let raw = operation::<MirCastOp>(&mut ctx, entry, vec![raw_slice], vec![input]);
        MirCastOp::new(raw).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        MirCastOp::new(raw)
            .set_attr_pointer_kind_authority(&ctx, MirPointerKindAuthorityAttr::RawAddress);
        let raw_value = raw.deref(&ctx).get_result(0);
        let data = operation::<MirCastOp>(&mut ctx, entry, vec![pointer], vec![raw_value]);
        MirCastOp::new(data).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        MirCastOp::new(data)
            .set_attr_pointer_kind_authority(&ctx, MirPointerKindAuthorityAttr::RustCast);
        let pointer_value = data.deref(&ctx).get_result(0);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![raw_value, pointer_value]);
        let text = translate(&mut ctx, &module);
        assert!(text.contains(r#""builtin.unrealized_conversion_cast"(%v0) : (!llvm.struct<(!llvm.ptr, i64)>) -> (!llvm.struct<(!llvm.ptr, i64)>)"#), "{text}");
        assert!(text.contains(r#""llvm.extractvalue"(%v1) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, i64)>) -> (!llvm.ptr)"#), "{text}");
        assert!(!text.contains("llvm.load"), "{text}");
        assert!(!text.contains("llvm.ptrtoint"), "{text}");
    }

    /// Preserve native public roots together while internalizing only local definitions.
    #[test]
    fn helper_linkage_preserves_native_public_roots_and_declarations() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("linkage").unwrap());
        let cases = [
            ("helper".to_owned(), false, true, true),
            ("kernel".to_owned(), true, true, false),
            (
                reserved_oxide_symbols::device_symbol("exported"),
                false,
                true,
                false,
            ),
            (
                format!("{}legacy", reserved_oxide_symbols::LEGACY_DEVICE_PREFIX),
                false,
                true,
                false,
            ),
            ("external_declaration".to_owned(), false, false, false),
            (
                reserved_oxide_symbols::device_extern_symbol("external"),
                false,
                false,
                false,
            ),
        ];
        for (name, kernel, definition, _) in &cases {
            let (function, region) = append_function(&mut ctx, &module, name, vec![], vec![]);
            if *kernel {
                function.deref_mut(&ctx).attributes.set(
                    Identifier::try_from("gpu_kernel").unwrap(),
                    StringAttr::new("true".into()),
                );
            }
            if *definition {
                let entry = BasicBlock::new(&mut ctx, None, vec![]);
                entry.insert_at_back(region, &ctx);
                operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![]);
            }
        }
        let target = CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&ctx, &module)
            .unwrap();
        let functions =
            &target.root.regions[0].blocks[0].operations[0].regions[0].blocks[0].operations;
        assert_eq!(functions.len(), cases.len());
        for (function, (_, _, _, internal)) in functions.iter().zip(&cases) {
            assert_eq!(
                function.attributes.contains_key("llvm.linkage"),
                *internal,
                "{function:?}"
            );
        }
    }

    /// A zero-size value keeps its Rust type while its local helper gains LLVM linkage.
    #[test]
    fn helper_linkage_internalizes_zero_size_signatures() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("zst_linkage").unwrap());
        let unit: TypeHandle = dialect_mir::types::MirTupleType::get_with_layout(
            &mut ctx,
            vec![],
            vec![],
            vec![],
            0,
            1,
        )
        .into();
        let (_, region) = append_function(&mut ctx, &module, "zst_helper", vec![unit], vec![unit]);
        let entry = BasicBlock::new(&mut ctx, None, vec![unit]);
        entry.insert_at_back(region, &ctx);
        let value = entry.deref(&ctx).get_argument(0);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![value]);
        let text = translate(&mut ctx, &module);
        assert!(
            text.contains("llvm.linkage = #llvm.linkage<internal>"),
            "{text}"
        );
        assert!(
            text.contains("function_type = (!llvm.struct<()>) -> (!llvm.struct<()>)"),
            "{text}"
        );
    }

    /// Reject unmodeled linkage overrides and retain malformed kernel-marker errors.
    #[test]
    fn helper_linkage_rejects_conflicting_metadata() {
        for (name, value) in [("sym_visibility", "private"), ("gpu_kernel", "false")] {
            let mut ctx = context();
            let module = ModuleOp::new(&mut ctx, Identifier::try_from("bad_linkage").unwrap());
            let (function, region) = append_function(&mut ctx, &module, "helper", vec![], vec![]);
            function.deref_mut(&ctx).attributes.set(
                Identifier::try_from(name).unwrap(),
                StringAttr::new(value.into()),
            );
            let entry = BasicBlock::new(&mut ctx, None, vec![]);
            entry.insert_at_back(region, &ctx);
            operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![]);
            let result = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &module);
            assert!(result.is_err(), "accepted conflicting {name}");
        }
    }

    /// Check explicit inline(always) survives lowering for a scalar-only helper.
    #[test]
    fn alwaysinline_pointer_free_helper_uses_llvm_passthrough() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("inline").unwrap());
        let (function, region) = append_function(&mut ctx, &module, "helper", vec![], vec![]);
        function.deref_mut(&ctx).attributes.set(
            Identifier::try_from("alwaysinline").unwrap(),
            StringAttr::new("true".into()),
        );
        let entry = BasicBlock::new(&mut ctx, None, vec![]);
        entry.insert_at_back(region, &ctx);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![]);

        let text = translate(&mut ctx, &module);
        assert!(text.contains(r#"passthrough = ["alwaysinline"]"#), "{text}");
        assert!(!text.contains(r#"alwaysinline = "true""#), "{text}");
    }

    /// Check explicit inline(always) also survives lowering for a pointer-bearing helper.
    #[test]
    fn alwaysinline_pointer_helper_uses_llvm_passthrough() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("inline_ptr").unwrap());
        let f32_type: TypeHandle = FP32Type::get(&ctx).into();
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, f32_type, false).into();
        let (function, region) =
            append_function(&mut ctx, &module, "pointer_helper", vec![pointer], vec![]);
        function.deref_mut(&ctx).attributes.set(
            Identifier::try_from("alwaysinline").unwrap(),
            StringAttr::new("true".into()),
        );
        let entry = BasicBlock::new(&mut ctx, None, vec![pointer]);
        entry.insert_at_back(region, &ctx);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![]);

        let text = translate(&mut ctx, &module);
        assert!(text.contains(r#"passthrough = ["alwaysinline"]"#), "{text}");
        assert!(!text.contains(r#"alwaysinline = "true""#), "{text}");
    }

    /// Check each typed `#[inline]` request lowers to the LLVM passthrough keyword
    /// the native path uses, and that the request itself does not reach the text.
    #[test]
    fn inline_intent_uses_llvm_passthrough() {
        for (intent, keyword) in [
            (InlineIntentAttr::Hint, "inlinehint"),
            (InlineIntentAttr::Always, "alwaysinline"),
            (InlineIntentAttr::Never, "noinline"),
        ] {
            let mut ctx = context();
            let module = ModuleOp::new(&mut ctx, Identifier::try_from("inline_request").unwrap());
            let (function, region) = append_function(&mut ctx, &module, "helper", vec![], vec![]);
            MirFuncOp::wrap(&ctx, function)
                .unwrap()
                .set_inline_intent(&mut ctx, intent);
            let entry = BasicBlock::new(&mut ctx, None, vec![]);
            entry.insert_at_back(region, &ctx);
            operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![]);

            let text = translate(&mut ctx, &module);
            assert!(
                text.contains(&format!(r#"passthrough = ["{keyword}"]"#)),
                "{intent:?}: {text}"
            );
            assert!(!text.contains("inline_intent"), "{intent:?}: {text}");
        }
    }

    /// Check the typed request and the older string marker produce one keyword, not two.
    #[test]
    fn inline_intent_shares_the_passthrough_list_with_the_alwaysinline_marker() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("inline_both").unwrap());
        let (function, region) = append_function(&mut ctx, &module, "helper", vec![], vec![]);
        function.deref_mut(&ctx).attributes.set(
            Identifier::try_from("alwaysinline").unwrap(),
            StringAttr::new("true".into()),
        );
        MirFuncOp::wrap(&ctx, function)
            .unwrap()
            .set_inline_intent(&mut ctx, InlineIntentAttr::Always);
        let entry = BasicBlock::new(&mut ctx, None, vec![]);
        entry.insert_at_back(region, &ctx);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![]);

        let text = translate(&mut ctx, &module);
        assert!(text.contains(r#"passthrough = ["alwaysinline"]"#), "{text}");
        assert_eq!(text.matches("alwaysinline").count(), 1, "{text}");
    }

    /// Check an unstructured MIR branch graph retains successor edges and operand segment sizes.
    #[test]
    fn arbitrary_mir_cfg_maps_to_cf_with_explicit_segments() {
        let mut ctx = context();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("cfg").unwrap());
        let i1_type = integer(&mut ctx, 1, Signedness::Signless);
        let u64_type = integer(&mut ctx, 64, Signedness::Unsigned);
        let (_, region) = append_function(
            &mut ctx,
            &module,
            "choose",
            vec![i1_type.into(), u64_type.into()],
            vec![u64_type.into()],
        );
        let entry = BasicBlock::new(&mut ctx, None, vec![i1_type.into(), u64_type.into()]);
        let on_true = BasicBlock::new(&mut ctx, None, vec![u64_type.into()]);
        let on_false = BasicBlock::new(&mut ctx, None, vec![u64_type.into()]);
        entry.insert_at_back(region, &ctx);
        on_true.insert_at_back(region, &ctx);
        on_false.insert_at_back(region, &ctx);

        let entry_arguments = entry.deref(&ctx).arguments().collect::<Vec<_>>();
        let (operands, segments) = MirCondBranchOp::compute_segment_sizes(vec![
            vec![entry_arguments[0]],
            vec![entry_arguments[1]],
            vec![entry_arguments[1]],
        ]);
        let branch = Operation::new(
            &mut ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![on_true, on_false],
            0,
        );
        MirCondBranchOp::new(branch).set_operand_segment_sizes(&ctx, segments);
        branch.insert_at_back(entry, &ctx);

        let true_value = on_true.deref(&ctx).get_argument(0);
        operation::<MirReturnOp>(&mut ctx, on_true, vec![], vec![true_value]);
        let false_value = on_false.deref(&ctx).get_argument(0);
        let goto = Operation::new(
            &mut ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![false_value],
            vec![on_true],
            0,
        );
        goto.insert_at_back(on_false, &ctx);

        let text = translate(&mut ctx, &module);
        let expected = r#""builtin.module"() <{sym_name = "cfg"}> ({
  ^bb0:
    "func.func"() <{function_type = (i1, i64) -> (i64), sym_name = "choose"}> ({
      ^bb1(%v0: i1, %v1: i64):
        "cf.cond_br"(%v0, %v1, %v1) [^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i64, i64) -> ()
      ^bb2(%v2: i64):
        "func.return"(%v2) : (i64) -> ()
      ^bb3(%v3: i64):
        "cf.br"(%v3) [^bb2] : (i64) -> ()
    }) {llvm.linkage = #llvm.linkage<internal>} : () -> ()
}) : () -> ()
"#;
        assert_eq!(
            crate::profile::normalize_mlir_identifiers(&text),
            crate::profile::normalize_mlir_identifiers(expected)
        );
    }
}
