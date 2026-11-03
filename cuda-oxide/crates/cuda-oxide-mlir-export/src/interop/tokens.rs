/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Adapt target types and attributes to upstream printer interfaces.
//!
//! Native Pliron builtins use their upstream implementations. Target dialect
//! tokens and typed scalar lexemes retain the spelling selected by the CUDA
//! mapping packs; no source-dialect lowering is inferred here.

use std::fmt;

use pliron::{
    attribute::AttrObj,
    builtin::{
        attributes::{BoolAttr, StringAttr, TypeAttr, UnitAttr, VecAttr},
        types::{BF16Type, FP16Type, FP32Type, FP64Type, FunctionType, IntegerType, Signedness},
    },
    combine::stream::position::SourcePosition,
    context::Context,
    derive::{attr_interface_impl, pliron_attr, pliron_type, type_interface_impl},
    location::{Location, Source},
    printable::{Printable, State},
    r#type::{Type, TypeHandle},
};
use pliron_mlir_interop::{
    MlirPrinter, ToMlirAttr, ToMlirType,
    printers::{mlir_dense_array, mlir_string_literal},
};

use crate::{MlirAttribute, MlirFloatType, MlirLocation, MlirType, TranslationError};

#[pliron_type(
    name = "cudaoxide_export.token",
    format = "$spelling",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
struct TokenType {
    spelling: String,
}

#[type_interface_impl]
impl ToMlirType for TokenType {
    fn to_mlir(
        &self,
        _: &Context,
        _: &State,
        f: &mut fmt::Formatter<'_>,
    ) -> pliron::result::Result<()> {
        validate_token(&self.spelling)?;
        write!(f, "{}", self.spelling)?;
        Ok(())
    }
}

#[pliron_attr(
    name = "cudaoxide_export.token",
    format = "$spelling",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct TokenAttr {
    spelling: String,
}

#[attr_interface_impl]
impl ToMlirAttr for TokenAttr {
    fn to_mlir(
        &self,
        _: &Context,
        _: &State,
        f: &mut fmt::Formatter<'_>,
    ) -> pliron::result::Result<()> {
        validate_token(&self.spelling)?;
        write!(f, "{}", self.spelling)?;
        Ok(())
    }
}

fn validate_token(spelling: &str) -> pliron::result::Result<()> {
    // Public mapping enums can bypass their checked constructors. Enforce the
    // same raw-token contract at printing, including recursively nested tokens
    // and fused-location metadata. Builtin strings use upstream escaping.
    if spelling.chars().any(char::is_control) {
        return pliron::input_err_noloc!("raw MLIR token contains a control character");
    }
    Ok(())
}

pub(super) fn printer_error(ctx: &Context, error: pliron::result::Error) -> TranslationError {
    TranslationError::InvalidTarget(format!("upstream MLIR printer: {}", error.disp(ctx)))
}

fn type_text(ctx: &Context, ty: &MlirType) -> Result<String, TranslationError> {
    let ty = materialize_type(ctx, ty)?;
    String::try_from(MlirPrinter::new(ctx, &ty)).map_err(|error| printer_error(ctx, error))
}

fn attribute_text(ctx: &Context, attr: &MlirAttribute) -> Result<String, TranslationError> {
    let attr = materialize_attribute(ctx, attr)?;
    String::try_from(MlirPrinter::new(ctx, &*attr)).map_err(|error| printer_error(ctx, error))
}

fn token_attr(spelling: String) -> AttrObj {
    Box::new(TokenAttr { spelling })
}

pub(super) fn dictionary_key(key: &str) -> String {
    let mut bytes = key.bytes();
    let bare = bytes
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        && bytes.all(|c| c.is_ascii_alphanumeric() || b"_.$".contains(&c));
    if bare {
        key.to_owned()
    } else {
        mlir_string_literal(key)
    }
}

pub(super) fn materialize_type(
    ctx: &Context,
    ty: &MlirType,
) -> Result<TypeHandle, TranslationError> {
    let spelling = match ty {
        MlirType::Integer(0) => {
            return Err(TranslationError::InvalidTarget(
                "integer type has zero width".into(),
            ));
        }
        MlirType::Integer(width) => {
            return Ok(IntegerType::get(ctx, *width, Signedness::Signless).into());
        }
        MlirType::Float(MlirFloatType::F16) => return Ok(FP16Type::get(ctx).into()),
        MlirType::Float(MlirFloatType::BF16) => return Ok(BF16Type::get(ctx).into()),
        MlirType::Float(MlirFloatType::F32) => return Ok(FP32Type::get(ctx).into()),
        MlirType::Float(MlirFloatType::F64) => return Ok(FP64Type::get(ctx).into()),
        MlirType::Function { inputs, results } => {
            let inputs = inputs
                .iter()
                .map(|ty| materialize_type(ctx, ty))
                .collect::<Result<_, _>>()?;
            let results = results
                .iter()
                .map(|ty| materialize_type(ctx, ty))
                .collect::<Result<_, _>>()?;
            return Ok(FunctionType::get(ctx, inputs, results).into());
        }
        MlirType::Index => "index".into(),
        MlirType::Float(MlirFloatType::Other(spelling)) | MlirType::Dialect(spelling) => {
            spelling.clone()
        }
        MlirType::Vector { shape, element } => {
            let dimensions = shape.iter().map(|n| format!("{n}x")).collect::<String>();
            format!("vector<{dimensions}{}>", type_text(ctx, element)?)
        }
        MlirType::Tuple(elements) => {
            let elements = elements
                .iter()
                .map(|ty| type_text(ctx, ty))
                .collect::<Result<Vec<_>, _>>()?;
            format!("tuple<{}>", elements.join(", "))
        }
    };
    Ok(Type::instantiate(TokenType { spelling }, ctx).into())
}

pub(super) fn materialize_attribute(
    ctx: &Context,
    attr: &MlirAttribute,
) -> Result<AttrObj, TranslationError> {
    Ok(match attr {
        MlirAttribute::Unit => Box::new(UnitAttr::new()),
        MlirAttribute::Bool(value) => Box::new(BoolAttr::new(*value)),
        MlirAttribute::String(value) => Box::new(StringAttr::new(value.clone())),
        MlirAttribute::Type(ty) => Box::new(TypeAttr::new(materialize_type(ctx, ty)?)),
        MlirAttribute::Array(values) => Box::new(VecAttr::new(
            values
                .iter()
                .map(|attr| materialize_attribute(ctx, attr))
                .collect::<Result<_, _>>()?,
        )),
        // Preserve exact signed decimal and floating bit-pattern spellings,
        // including index-typed integers that Pliron IntegerAttr cannot hold.
        MlirAttribute::Integer { value, ty } => {
            token_attr(format!("{value} : {}", type_text(ctx, ty)?))
        }
        MlirAttribute::Float { value, ty } => {
            token_attr(format!("{value} : {}", type_text(ctx, ty)?))
        }
        MlirAttribute::DenseI32Array(values) => token_attr(mlir_dense_array("i32", values)),
        MlirAttribute::DenseI64Array(values) => token_attr(mlir_dense_array("i64", values)),
        MlirAttribute::Dictionary(values) => {
            let values = values
                .iter()
                .map(|(key, value)| {
                    Ok(format!(
                        "{} = {}",
                        dictionary_key(key),
                        attribute_text(ctx, value)?
                    ))
                })
                .collect::<Result<Vec<_>, TranslationError>>()?;
            token_attr(format!("{{{}}}", values.join(", ")))
        }
        MlirAttribute::SymbolRef(symbol) => token_attr(format!("@{}", dictionary_key(symbol))),
        MlirAttribute::Dialect(spelling) => token_attr(spelling.clone()),
    })
}

pub(super) fn materialize_location(
    ctx: &Context,
    location: &MlirLocation,
) -> Result<Location, TranslationError> {
    Ok(match location {
        MlirLocation::Unknown => Location::Unknown,
        MlirLocation::FileLineCol { file, line, column } => Location::SrcPos {
            src: Source::new_from_file(ctx, file),
            pos: SourcePosition {
                line: (*line).try_into().map_err(|_| {
                    TranslationError::InvalidTarget(
                        "source line exceeds Pliron location range".into(),
                    )
                })?,
                column: (*column).try_into().map_err(|_| {
                    TranslationError::InvalidTarget(
                        "source column exceeds Pliron location range".into(),
                    )
                })?,
            },
        },
        MlirLocation::Fused {
            metadata,
            locations,
        } => Location::Fused {
            metadata: metadata
                .as_deref()
                .map(|attr| materialize_attribute(ctx, attr))
                .transpose()?,
            locations: locations
                .iter()
                .map(|loc| materialize_location(ctx, loc))
                .collect::<Result<_, _>>()?,
        },
        MlirLocation::Named { name, child } => Location::Named {
            name: name.clone(),
            child_loc: Box::new(materialize_location(ctx, child)?),
        },
        MlirLocation::CallSite { callee, caller } => Location::CallSite {
            callee: Box::new(materialize_location(ctx, callee)?),
            caller: Box::new(materialize_location(ctx, caller)?),
        },
    })
}
