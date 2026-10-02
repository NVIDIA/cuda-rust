/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Keep grid-constant values in read-only kernel parameter storage.

use std::collections::{BTreeMap, BTreeSet};

use dialect_mir::{ops::MirFuncOp, types::MirPtrType};
use pliron::{
    builtin::{attributes::TypeAttr, type_interfaces::FunctionTypeInterface},
    context::{Context, Ptr},
    operation::Operation,
    r#type::{Typed, type_cast},
};
use pliron_mlir_export::{MlirAttribute, MlirOperation, MlirType};
use reserved_oxide_symbols::{
    MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX, MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
};

pub(crate) struct Parameter {
    size: u64,
    alignment: u64,
}

impl Parameter {
    pub(crate) fn attributes(&self) -> Result<MlirAttribute, String> {
        // The byte array sets the launch argument's size. `llvm.align` keeps
        // Rust's alignment, which can exceed the byte array's alignment of 1.
        // The kernel body still accesses fields using the original Rust layout.
        // For a TMA descriptor: 128 bytes, aligned to 64 bytes.
        Ok(MlirAttribute::Dictionary(BTreeMap::from([
            (
                "llvm.byval".into(),
                MlirAttribute::Type(MlirType::dialect(format!(
                    "!llvm.array<{} x i8>",
                    self.size
                ))?),
            ),
            (
                "llvm.align".into(),
                MlirAttribute::Integer {
                    value: self.alignment.into(),
                    ty: MlirType::Integer(64),
                },
            ),
            ("nvvm.grid_constant".into(), MlirAttribute::Unit),
        ])))
    }
}

/// Validate grid-constant metadata and return parameters by Rust argument index.
///
/// These indices precede slice flattening. Missing or inconsistent metadata is
/// an error: the launch must carry the value's bytes, not an ordinary pointer.
pub(crate) fn take_parameters(
    ctx: &Context,
    source: Ptr<Operation>,
    target: &mut MlirOperation,
) -> Result<BTreeMap<usize, Parameter>, String> {
    let function = MirFuncOp::wrap(ctx, source).ok_or("expected mir.func")?;
    let function_type = function.get_type(ctx);
    let function_type = function_type.deref(ctx);
    let function_type = type_cast::<dyn FunctionTypeInterface>(&*function_type)
        .ok_or("mir.func does not have a function type")?;
    let inputs = function_type.arg_types();
    let mut indices = BTreeSet::new();
    for key in target.attributes.keys() {
        let Some((prefix, suffix)) = [
            MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
            MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX,
        ]
        .into_iter()
        .find_map(|prefix| key.strip_prefix(prefix).map(|suffix| (prefix, suffix))) else {
            continue;
        };
        let index = suffix
            .parse::<usize>()
            .map_err(|_| format!("malformed grid-constant attribute `{key}`"))?;
        if key != &format!("{prefix}{index}") || index >= inputs.len() {
            return Err(format!(
                "malformed or out-of-range grid-constant attribute `{key}`"
            ));
        }
        indices.insert(index);
    }
    if !indices.is_empty()
        && target.attributes.get("gpu_kernel") != Some(&MlirAttribute::String("true".into()))
    {
        return Err("grid-constant parameters require a kernel entry".into());
    }
    let mut parameters = BTreeMap::new();
    for index in indices {
        let pointee_key = format!("{MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX}{index}");
        let alignment_key = format!("{MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX}{index}");
        let (
            Some(MlirAttribute::Type(_)),
            Some(MlirAttribute::Integer {
                value: alignment,
                ty: MlirType::Integer(64),
            }),
        ) = (
            target.attributes.remove(&pointee_key),
            target.attributes.remove(&alignment_key),
        )
        else {
            return Err(format!(
                "parameter {index} has malformed or incomplete grid-constant metadata"
            ));
        };
        let alignment = u64::try_from(alignment)
            .map_err(|_| format!("parameter {index} grid-constant alignment does not fit u64"))?;
        if !alignment.is_power_of_two() {
            return Err(format!(
                "parameter {index} grid-constant alignment must be a nonzero power of two"
            ));
        }
        let pointer_type = inputs[index].deref(ctx);
        let pointer = pointer_type
            .downcast_ref::<MirPtrType>()
            .ok_or_else(|| format!("grid-constant parameter {index} is not a pointer"))?;
        if pointer.address_space() != 0 || pointer.is_mutable() {
            return Err(format!(
                "grid-constant parameter {index} must be an immutable generic pointer"
            ));
        }
        let operation = source.deref(ctx);
        let key = pointee_key
            .as_str()
            .try_into()
            .map_err(|_| "invalid grid-constant metadata key")?;
        let pointee = operation
            .attributes
            .get::<TypeAttr>(&key)
            .ok_or("grid-constant pointee is not a type")?
            .get_type(ctx);
        if pointee != pointer.pointee {
            return Err(format!(
                "grid-constant parameter {index} pointee metadata differs from its pointer type"
            ));
        }
        let size = crate::mir_memory::source_stored_size(ctx, pointee)
            .filter(|&size| size > 0)
            .ok_or_else(|| {
                format!("grid-constant parameter {index} requires a sized, nonzero pointee")
            })?;
        parameters.insert(index, Parameter { size, alignment });
    }
    Ok(parameters)
}
