/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builtin scalar types, constants, and attributes shared by every export profile.

use crate::{
    AttributeTranslation, DropAttribute, FixedType, MlirAttribute, MlirFloatType, MlirOperation,
    MlirType, OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession, TypeTranslation,
};
use pliron::{
    attribute::AttrObj,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{
            BoolAttr, FPDoubleAttr, FPHalfAttr, FPSingleAttr, IdentifierAttr, IntegerAttr,
            OperandSegmentSizesAttr, StringAttr, TypeAttr, UnitAttr,
        },
        ops::{ConstantOp, ModuleOp},
        type_interfaces::FunctionTypeInterface,
        types::{FP16Type, FP32Type, FP64Type, FunctionType, IntegerType, UnitType},
    },
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed, type_cast},
    utils::apfloat::Float,
};

/// Register Pliron builtin items used by every CUDA mapping profile.
pub fn register_builtin_pack(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_operation::<ModuleOp>(BuiltinModuleTranslation)?;
    registry.register_operation::<ConstantOp>(BuiltinConstantTranslation)?;

    registry.register_type::<IntegerType>(IntegerTypeTranslation)?;
    registry.register_type::<FP16Type>(FixedType(MlirType::Float(MlirFloatType::F16)))?;
    registry.register_type::<FP32Type>(FixedType(MlirType::Float(MlirFloatType::F32)))?;
    registry.register_type::<FP64Type>(FixedType(MlirType::Float(MlirFloatType::F64)))?;
    registry.register_type::<FunctionType>(FunctionTypeTranslation)?;
    registry.register_type::<UnitType>(UnitTypeTranslation)?;

    registry.register_attribute::<IdentifierAttr>(IdentifierAttributeTranslation)?;
    registry.register_attribute::<StringAttr>(StringAttributeTranslation)?;
    registry.register_attribute::<BoolAttr>(BoolAttributeTranslation)?;
    registry.register_attribute::<IntegerAttr>(IntegerAttributeTranslation)?;
    registry.register_attribute::<FPHalfAttr>(FloatAttributeTranslation::F16)?;
    registry.register_attribute::<FPSingleAttr>(FloatAttributeTranslation::F32)?;
    registry.register_attribute::<FPDoubleAttr>(FloatAttributeTranslation::F64)?;
    registry.register_attribute::<UnitAttr>(UnitAttributeTranslation)?;
    registry.register_attribute::<TypeAttr>(TypeAttributeTranslation)?;
    // This describes how Pliron flattened a variadic operand list. Recipes
    // such as `mir.cond_br` rebuild the corresponding target property from
    // the source op instead of copying a Pliron-only attribute name.
    registry.register_attribute::<OperandSegmentSizesAttr>(DropAttribute)?;
    Ok(())
}

/// Translate a builtin module and preserve its nested regions and optional symbol name.
struct BuiltinModuleTranslation;

impl OperationTranslation for BuiltinModuleTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let mut target = MlirOperation::new("builtin.module")?;
        target.results = input.results;
        target.operands = input.operands;
        target.successors = input.successors;
        target.regions = input.regions;
        target.attributes = input.attributes;
        target.location = input.location;
        if let Some(name) = target
            .attributes
            .remove(pliron::builtin::op_interfaces::ATTR_KEY_SYM_NAME.as_ref())
        {
            target.properties.insert("sym_name".into(), name);
        }
        Ok(vec![target])
    }
}

// Scalar constants created by canonicalization and loop-unrolling passes use
// builtin.constant rather than the source-facing mir.constant operation.
/// Map canonicalized builtin scalar constants to arith.constant.
struct BuiltinConstantTranslation;
impl OperationTranslation for BuiltinConstantTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if input.results.len() != 1
            || !input.operands.is_empty()
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err(
                "builtin.constant requires one result and no operands or control flow".into(),
            );
        }
        let value = input
            .attributes
            .remove("builtin_constant_value")
            .ok_or("builtin.constant requires a typed value")?;
        let ty = match &value {
            MlirAttribute::Integer { ty, .. } | MlirAttribute::Float { ty, .. } => ty,
            _ => {
                return Err(
                    "builtin.constant currently supports scalar integer and floating values".into(),
                );
            }
        };
        if *ty != input.results[0].ty || !input.attributes.is_empty() {
            return Err(
                "builtin.constant value type or attributes violate the scalar contract".into(),
            );
        }
        let mut target = MlirOperation::new("arith.constant")?;
        target.results = input.results;
        target.location = input.location;
        target.properties.insert("value".into(), value);
        Ok(vec![target])
    }
}

/// Retain integer width while erasing source signedness from the MLIR type spelling.
struct IntegerTypeTranslation;

impl TypeTranslation for IntegerTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let integer = source_ref
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| "expected builtin.integer".to_owned())?;
        Ok(MlirType::Integer(integer.width()))
    }
}

/// Recursively translate function arguments and results through the active registry.
struct FunctionTypeTranslation;

impl TypeTranslation for FunctionTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let function = type_cast::<dyn FunctionTypeInterface>(&*source_ref)
            .ok_or_else(|| "expected builtin function type".to_owned())?;
        let inputs = function
            .arg_types()
            .into_iter()
            .map(|ty| translate_nested_type(ctx, registry, ty))
            .collect::<Result<_, _>>()?;
        let results = function
            .res_types()
            .into_iter()
            .map(|ty| translate_nested_type(ctx, registry, ty))
            .collect::<Result<_, _>>()?;
        Ok(MlirType::Function { inputs, results })
    }
}

/// Represent Rust unit as an empty aggregate rather than inventing a scalar value.
struct UnitTypeTranslation;

impl TypeTranslation for UnitTypeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        // Unit is only legal as an empty aggregate carrier. Mapping it to an
        // empty tuple keeps that fact explicit instead of inventing i1/i8.
        Ok(MlirType::Tuple(vec![]))
    }
}

/// Translate a nested source type and preserve any registry diagnostic.
fn translate_nested_type(
    ctx: &Context,
    registry: &TranslationRegistry,
    source: TypeHandle,
) -> Result<MlirType, String> {
    registry.translate_type(ctx, source)
}

/// Preserve the bytes of a builtin string attribute.
struct StringAttributeTranslation;

impl AttributeTranslation for StringAttributeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let value = source
            .downcast_ref::<StringAttr>()
            .ok_or_else(|| "expected builtin.string".to_owned())?;
        Ok(Some(MlirAttribute::String(value.as_str().into())))
    }
}

/// Represent a source identifier as the target string spelling.
struct IdentifierAttributeTranslation;

impl AttributeTranslation for IdentifierAttributeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let value = source
            .downcast_ref::<IdentifierAttr>()
            .ok_or_else(|| "expected builtin.identifier".to_owned())?;
        Ok(Some(MlirAttribute::String(value.as_ref().to_string())))
    }
}

/// Preserve a builtin boolean attribute's truth value.
struct BoolAttributeTranslation;

impl AttributeTranslation for BoolAttributeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let value = source
            .downcast_ref::<BoolAttr>()
            .ok_or_else(|| "expected builtin.bool".to_owned())?
            .clone();
        Ok(Some(MlirAttribute::Bool(value.into())))
    }
}

/// Preserve integer width and all bits in the target signless attribute.
struct IntegerAttributeTranslation;

impl AttributeTranslation for IntegerAttributeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: &AttrObj,
        registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let value = source
            .downcast_ref::<IntegerAttr>()
            .ok_or_else(|| "expected builtin.integer".to_owned())?;
        let source_type = Typed::get_type(value, ctx);
        let width = value.get_type().deref(ctx).width();
        if width > 128 {
            return Err(format!(
                "integer attributes wider than 128 bits are not supported yet (got i{width})"
            ));
        }
        Ok(Some(MlirAttribute::Integer {
            // MLIR integer types are signless. Rendering the same two's-
            // complement bits as an i128 preserves both signed and unsigned
            // Pliron constants, including u64::MAX -> -1 : i64.
            value: value.value().to_i128(),
            ty: registry.translate_type(ctx, source_type)?,
        }))
    }
}

/// Select the source floating-point width before preserving its value in MLIR.
enum FloatAttributeTranslation {
    /// Translate a half-precision floating-point attribute.
    F16,
    /// Translate a single-precision floating-point attribute.
    F32,
    /// Translate a double-precision floating-point attribute.
    F64,
}

impl AttributeTranslation for FloatAttributeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let (value, ty) = match self {
            Self::F16 => {
                let value = source
                    .downcast_ref::<FPHalfAttr>()
                    .ok_or_else(|| "expected builtin.half".to_owned())?;
                (
                    format!("0x{:04X}", value.0.to_bits()),
                    MlirType::Float(MlirFloatType::F16),
                )
            }
            Self::F32 => {
                let value = source
                    .downcast_ref::<FPSingleAttr>()
                    .ok_or_else(|| "expected builtin.single".to_owned())?;
                (
                    format!("0x{:08X}", value.0.to_bits()),
                    MlirType::Float(MlirFloatType::F32),
                )
            }
            Self::F64 => {
                let value = source
                    .downcast_ref::<FPDoubleAttr>()
                    .ok_or_else(|| "expected builtin.double".to_owned())?;
                (
                    format!("0x{:016X}", value.0.to_bits()),
                    MlirType::Float(MlirFloatType::F64),
                )
            }
        };
        Ok(Some(MlirAttribute::Float { value, ty }))
    }
}

/// Translate a source presence marker to an MLIR unit attribute.
struct UnitAttributeTranslation;

impl AttributeTranslation for UnitAttributeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        Ok(Some(MlirAttribute::Unit))
    }
}

/// Recursively translate the type carried by a source type attribute.
struct TypeAttributeTranslation;

impl AttributeTranslation for TypeAttributeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: &AttrObj,
        registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        let value = source
            .downcast_ref::<TypeAttr>()
            .ok_or_else(|| "expected builtin.type".to_owned())?;
        let ty = TypedAttrInterface::get_type(value, ctx);
        Ok(Some(MlirAttribute::Type(translate_nested_type(
            ctx, registry, ty,
        )?)))
    }
}
