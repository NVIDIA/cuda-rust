/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::export::DeviceExternDecl;
use llvm_export::export::DeviceExternType;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;

/// Inserts a function operation into the module's block.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn append_to_module(ctx: &Context, module_op_ptr: Ptr<Operation>, func_op_ptr: Ptr<Operation>) {
    let region = module_op_ptr.deref(ctx).get_region(0).deref(ctx);
    let block = region.iter(ctx).next().expect("Module should have a block");
    func_op_ptr.insert_at_back(block, ctx);
}

/// Lowers `dialect-mir` operations to the LLVM dialect.
///
/// Runs `mir-lower`'s `DialectConversion`-based pass, which converts each
/// `dialect-mir`/`dialect-nvvm` op to its LLVM dialect equivalent. The LLVM
/// dialect auto-registers when the `Context` is created, so no explicit
/// registration is needed here.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn lower_to_llvm(
    ctx: &mut Context,
    module_op_ptr: Ptr<Operation>,
    allow_fma_contraction: bool,
    intrinsic_backend: mir_lower::IntrinsicBackend,
    module_disambiguator: Option<u64>,
) -> Result<(), PipelineError> {
    mir_lower::register(ctx);

    match mir_lower::lower_mir_to_llvm_with_options(
        ctx,
        module_op_ptr,
        mir_lower::LoweringOptions {
            allow_fma_contraction,
            intrinsic_backend,
            module_disambiguator,
        },
    ) {
        Ok(()) => Ok(()),
        // Format with `ctx` so the failing op's location/span survives.
        Err(e) => Err(PipelineError::Lowering(e.disp(ctx).to_string())),
    }
}

/// Adds device extern function declarations to the LLVM dialect module.
///
/// Creates LLVM dialect `FuncOp` declarations (without bodies) for each
/// device extern function. These declarations ensure that calls to extern
/// functions pass verification; the matching `declare` statements with
/// attributes are emitted during LLVM IR export.
///
/// This runs before MIR-to-LLVM call lowering so the call converter can read
/// exact parameter address spaces. It is still idempotent with respect to any
/// LLVM declaration already present in the mixed module; inserting a second
/// `FuncOp` for the same symbol would fail module verification.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn add_device_extern_declarations(
    ctx: &mut Context,
    module_op_ptr: Ptr<Operation>,
    device_externs: &[DeviceExternDecl],
) -> Result<(), PipelineError> {
    use llvm_export::ops::FuncOp;
    use llvm_export::types::FuncType;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;
    use pliron::identifier::Identifier;
    use std::collections::HashMap;

    // Get the module's block pointer first (this is a Ptr, not a Ref, so no borrow issues)
    let block = {
        let region = module_op_ptr.deref(ctx).get_region(0).deref(ctx);
        region.iter(ctx).next().expect("Module should have a block")
    };

    let declared_symbols: HashMap<_, _> = block
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| {
            Operation::get_op::<FuncOp>(op, ctx)
                .map(|f| (f.get_symbol_name(ctx).to_string(), f.get_type(ctx)))
        })
        .collect();

    for decl in device_externs {
        let param_types: Vec<_> = decl
            .param_types
            .iter()
            .map(|ty| device_extern_type_to_pliron(ctx, ty, false))
            .collect::<Result<_, _>>()?;
        let return_type = device_extern_type_to_pliron(ctx, &decl.return_type, true)?;

        // Create function type (result, args, is_variadic)
        let func_type = FuncType::get(ctx, return_type, param_types, false);

        if let Some(existing_type) = declared_symbols.get(&decl.export_name) {
            let existing_ref = existing_type.deref(ctx);
            let existing = &*existing_ref;
            let expected_ref = func_type.deref(ctx);
            let expected = &*expected_ref;
            if existing.result_type() != expected.result_type()
                || existing.arg_types() != expected.arg_types()
                || existing.is_var_arg() != expected.is_var_arg()
            {
                return Err(PipelineError::Export(format!(
                    "device extern `@{}` conflicts with the call-site declaration: expected `{}`, found `{}`",
                    decl.export_name,
                    expected_ref.disp(ctx),
                    existing_ref.disp(ctx),
                )));
            }
            continue;
        }

        // Use the original export name (NOT the prefixed name).
        // The MIR sees calls to `cuda_oxide_device_extern_<hash>_foo`, but
        // mir-lower/convert/ops/call.rs strips the reserved prefix via
        // `reserved_oxide_symbols::device_extern_base_name`, so the LLVM IR
        // emits `call @foo(...)`. For that to resolve, we declare `@foo` here.
        let func_ident: Identifier = decl.export_name.clone().try_into().map_err(|_| {
            PipelineError::Export(format!(
                "device-extern symbol `{}` cannot be represented by the LLVM dialect",
                decl.export_name
            ))
        })?;

        // Create function declaration (no body = declaration)
        let func_op = FuncOp::new(ctx, func_ident, func_type);

        // Insert at the front of the module (declarations come before definitions)
        func_op.get_operation().insert_at_front(block, ctx);
    }

    Ok(())
}

/// Convert the structured device-extern ABI type to the opaque-pointer pliron
/// LLVM type used for verification and call lowering.
fn device_extern_type_to_pliron(
    ctx: &mut Context,
    ty: &DeviceExternType,
    allow_void: bool,
) -> Result<pliron::r#type::TypeHandle, PipelineError> {
    use llvm_export::types::{ArrayType, HalfType, PointerType, VoidType};
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};

    Ok(match ty {
        DeviceExternType::Void if allow_void => VoidType::get(ctx).into(),
        DeviceExternType::Void => {
            return Err(PipelineError::Export(
                "device-extern parameters and aggregate elements cannot be `void`".to_string(),
            ));
        }
        DeviceExternType::Integer(bits)
        | DeviceExternType::SignExtInteger(bits)
        | DeviceExternType::ZeroExtInteger(bits)
            if *bits > 0 =>
        {
            IntegerType::get(ctx, *bits, Signedness::Signless).into()
        }
        DeviceExternType::Integer(_)
        | DeviceExternType::SignExtInteger(_)
        | DeviceExternType::ZeroExtInteger(_) => {
            return Err(PipelineError::Export(
                "device-extern integer width must be non-zero".to_string(),
            ));
        }
        DeviceExternType::Float16 => HalfType::get(ctx).into(),
        DeviceExternType::Float32 => FP32Type::get(ctx).into(),
        DeviceExternType::Float64 => FP64Type::get(ctx).into(),
        DeviceExternType::Pointer {
            pointee,
            address_space,
        } => {
            if matches!(pointee.as_ref(), DeviceExternType::Void) {
                return Err(PipelineError::Export(
                    "device-extern pointer cannot have `void` as its pointee; use i8".to_string(),
                ));
            }
            PointerType::get(ctx, *address_space).into()
        }
        DeviceExternType::Array { element, len } => {
            let element = device_extern_type_to_pliron(ctx, element, false)?;
            ArrayType::get(ctx, element, *len).into()
        }
    })
}
