/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use pliron::{
    attribute::AttrObj,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{
            BoolAttr, FPDoubleAttr, FPHalfAttr, FPSingleAttr, IdentifierAttr, IntegerAttr,
            OperandSegmentSizesAttr, StringAttr, TypeAttr, UnitAttr,
        },
        ops::ModuleOp,
        type_interfaces::FunctionTypeInterface,
        types::{FP16Type, FP32Type, FP64Type, FunctionType, IntegerType, UnitType},
    },
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed, type_cast},
    utils::apfloat::Float,
};
use pliron_mlir_export::{
    AttributeTranslation, DropAttribute, FixedType, MlirAttribute, MlirFloatType, MlirOperation,
    MlirType, OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession, TypeTranslation,
};

/// Register Pliron builtin items used by every CUDA mapping profile.
pub fn register_builtin_pack(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_operation::<ModuleOp>(BuiltinModuleTranslation)?;

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

fn translate_nested_type(
    ctx: &Context,
    registry: &TranslationRegistry,
    source: TypeHandle,
) -> Result<MlirType, String> {
    registry.translate_type(ctx, source)
}

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

enum FloatAttributeTranslation {
    F16,
    F32,
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
