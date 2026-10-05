/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module-level export flow.

use std::fmt::Write;

use pliron::{
    builtin::{
        op_interfaces::{OneRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        type_interfaces::FunctionTypeInterface,
        types::{FP32Type, FP64Type, IntegerType},
    },
    context::Context,
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

use crate::{
    ops,
    ops::GlobalOpExt,
    types::{ArrayType, FuncType, HalfType, PointerType, VoidType},
};

use super::{
    ExportedModule,
    config::{DebugKind, ExportBackendConfig, NvvmIrDialect},
    externs::{DeviceExternDecl, DeviceExternType},
    metadata::{emit_nvvm_annotations, emit_nvvmir_version, needs_nvvm_annotations},
    state::{DebugSharedFunctionScope, GlobalSourceInfo, GlobalSymbolInfo, ModuleExportState},
};

fn validate_export_config(config: &dyn ExportBackendConfig) -> Result<(), String> {
    if config.nvvm_ir_dialect() == Some(NvvmIrDialect::LegacyLlvm7)
        && config.debug_kind() != DebugKind::Off
    {
        return Err(
            "legacy LLVM 7 NVVM IR does not yet support cuda-oxide debug metadata; disable device debug information"
                .to_string(),
        );
    }
    Ok(())
}

fn index_module_symbols(
    state: &mut ModuleExportState<'_>,
    module: &ModuleOp,
) -> Result<(), String> {
    let region = module.get_region(state.ctx).deref(state.ctx);
    let Some(block) = region.iter(state.ctx).next() else {
        return Ok(());
    };
    for operation in block.deref(state.ctx).iter(state.ctx) {
        if let Some(global) = Operation::get_op::<ops::GlobalOp>(operation, state.ctx) {
            let symbol = global.get_symbol_name(state.ctx).to_string();
            let value_type = global.get_type(state.ctx);
            let address_space = global.address_space(state.ctx);
            state.global_symbols.insert(
                symbol.clone(),
                GlobalSymbolInfo {
                    value_type,
                    address_space,
                },
            );
            if global.is_retained(state.ctx) {
                state.retained_globals.push(symbol.clone());
            }

            if let Some(source_key) = global.source_global_key(state.ctx) {
                let initializer_size = global
                    .initializer_hex(state.ctx)
                    .and_then(|hex| u64::try_from(hex.len() / 2).ok());
                let source_info = GlobalSourceInfo {
                    symbol: symbol.clone(),
                    value_type,
                    address_space,
                    initializer_size,
                };
                if let Some(existing) = state.global_sources.get(&source_key)
                    && (existing.symbol != source_info.symbol
                        || existing.value_type != source_info.value_type
                        || existing.address_space != source_info.address_space
                        || existing.initializer_size != source_info.initializer_size)
                {
                    return Err(format!(
                        "multiple LLVM globals map to rustc global key `{source_key}`: `@{}` and `@{symbol}`",
                        existing.symbol
                    ));
                }
                state.global_sources.insert(source_key, source_info);
            }
        } else if let Some(func) = Operation::get_op::<ops::FuncOp>(operation, state.ctx) {
            let raw_name = func.get_symbol_name(state.ctx);
            let raw_name_str: &str = raw_name.as_ref();
            let exported_name = if raw_name_str.starts_with("llvm_") {
                super::names::decode_intrinsic_identifier(raw_name_str)
            } else {
                super::names::strip_device_prefix(raw_name_str)
            };
            let function_type = func.get_type(state.ctx).into();
            if let Some(existing_name) = state
                .function_source_names
                .insert(exported_name.clone(), raw_name.to_string())
            {
                return Err(format!(
                    "multiple functions `{existing_name}` and `{raw_name}` normalize to `@{exported_name}`"
                ));
            }
            state
                .function_types
                .insert(exported_name.clone(), function_type);
            let grid_constants = state.grid_constant_parameters(&func)?;
            if !grid_constants.is_empty() {
                state
                    .function_grid_constants
                    .insert(exported_name.clone(), grid_constants);
            }
            if func.get_operation().deref(state.ctx).regions().count() != 0 {
                state.function_definitions.insert(exported_name);
            }
        }
    }
    Ok(())
}

/// Reserve the real owning `DISubprogram` before AS3 globals are emitted.
///
/// MIR lowering inserts generated globals at the front of the module, before
/// their function definitions. LLVM metadata may refer forward, but our text
/// exporter needs the numeric subprogram id while printing the global. This
/// prepass derives the source scope from the global identity, validates its raw
/// owner symbol against the indexed definition, and allocates that definition's
/// one cached subprogram early. The later function export reuses it.
fn prepare_debug_shared_function_scopes(
    state: &mut ModuleExportState<'_>,
    module: &ModuleOp,
) -> Result<(), String> {
    if !state.debug_kind.variables_enabled() {
        return Ok(());
    }

    let region = module.get_region(state.ctx).deref(state.ctx);
    let Some(block) = region.iter(state.ctx).next() else {
        return Ok(());
    };

    for operation in block.deref(state.ctx).iter(state.ctx) {
        let Some(global) = Operation::get_op::<ops::GlobalOp>(operation, state.ctx) else {
            continue;
        };
        if global.address_space(state.ctx) != crate::types::address_space::SHARED {
            continue;
        }
        let Some(info) = ops::debug_global_variable(state.ctx, global.get_operation()) else {
            continue;
        };
        if !info.is_function_local {
            continue;
        }
        let Some(raw_owner) = ops::debug_global_owner_function(state.ctx, global.get_operation())
        else {
            // A partial/malformed carrier must not fall back to namespace or CU
            // scope: that would make one per-block variable look module-global.
            continue;
        };
        let owner = super::function::exported_function_name(&raw_owner);
        if state
            .function_source_names
            .get(&owner)
            .is_none_or(|indexed| indexed != &raw_owner)
        {
            continue;
        }
        let Some((function_name, namespace)) = info.namespace.split_last() else {
            continue;
        };
        if namespace.is_empty() {
            continue;
        }
        let scope = DebugSharedFunctionScope {
            namespace: namespace.to_vec(),
            name: function_name.clone(),
        };
        if let Some(previous) = state
            .debug_shared_function_scopes
            .insert(owner.clone(), scope.clone())
            && previous != scope
        {
            return Err(format!(
                "conflicting source scopes for shared-static owner `@{owner}`: {previous:?} versus {scope:?}"
            ));
        }
    }

    for operation in block.deref(state.ctx).iter(state.ctx) {
        let Some(function) = Operation::get_op::<ops::FuncOp>(operation, state.ctx) else {
            continue;
        };
        if function.get_operation().deref(state.ctx).regions().count() == 0 {
            continue;
        }
        let raw_name = function.get_symbol_name(state.ctx);
        let owner = super::function::exported_function_name(raw_name.as_ref());
        if state.debug_shared_function_scopes.contains_key(&owner) {
            let loc = function.get_operation().deref(state.ctx).loc();
            let debug_name = ops::debug_function_name(state.ctx, function.get_operation())
                .unwrap_or_else(|| owner.clone());
            state.debug_subprogram_for_function(&debug_name, &owner, &loc);
        }
    }

    Ok(())
}

fn device_extern_type_matches_erased(
    ctx: &Context,
    expected: &DeviceExternType,
    actual: TypeHandle,
) -> bool {
    let actual = actual.deref(ctx);
    match expected {
        DeviceExternType::Void => actual.is::<VoidType>(),
        DeviceExternType::Integer(bits)
        | DeviceExternType::SignExtInteger(bits)
        | DeviceExternType::ZeroExtInteger(bits) => actual
            .downcast_ref::<IntegerType>()
            .is_some_and(|ty| ty.width() == *bits),
        DeviceExternType::Float16 => actual.is::<HalfType>(),
        DeviceExternType::Float32 => actual.is::<FP32Type>(),
        DeviceExternType::Float64 => actual.is::<FP64Type>(),
        DeviceExternType::Pointer { address_space, .. } => actual
            .downcast_ref::<PointerType>()
            .is_some_and(|ty| ty.address_space() == *address_space),
        DeviceExternType::Array { element, len } => {
            actual.downcast_ref::<ArrayType>().is_some_and(|ty| {
                ty.size() == *len && device_extern_type_matches_erased(ctx, element, ty.elem_type())
            })
        }
    }
}

fn validate_device_extern_function_shape(
    state: &ModuleExportState<'_>,
    decl: &DeviceExternDecl,
) -> Result<(), String> {
    let Some(function_type) = state.function_types.get(&decl.export_name).copied() else {
        return Ok(());
    };

    // A device extern describes the ordinary callable ABI. Erased pointer
    // shapes alone cannot establish compatibility with a kernel declaration
    // that transports the pointee's bytes by value. Suppressing that declaration
    // would otherwise discard both its byval storage and grid-constant metadata.
    if state
        .function_grid_constants
        .contains_key(&decl.export_name)
    {
        return Err(format!(
            "device extern `@{}` conflicts with a grid-constant kernel declaration; its launch ABI is not an ordinary device function ABI",
            decl.export_name
        ));
    }

    if state.function_definitions.contains(&decl.export_name) {
        return Err(format!(
            "device extern `@{}` conflicts with a function definition of the same exported name",
            decl.export_name
        ));
    }
    let source_name = state
        .function_source_names
        .get(&decl.export_name)
        .expect("indexed function type has a source name");
    if source_name != &decl.export_name {
        return Err(format!(
            "device extern `@{}` conflicts with function `{source_name}`, which normalizes to the same exported name",
            decl.export_name
        ));
    }

    let function_ref = function_type.deref(state.ctx);
    let function = function_ref.downcast_ref::<FuncType>().ok_or_else(|| {
        format!(
            "device extern `@{}` collides with a non-function LLVM type `{}`",
            decl.export_name,
            function_ref.disp(state.ctx)
        )
    })?;
    let args = function.arg_types();
    let shape_matches = !function.is_var_arg()
        && args.len() == decl.param_types.len()
        && device_extern_type_matches_erased(state.ctx, &decl.return_type, function.result_type())
        && decl.param_types.iter().zip(args).all(|(expected, actual)| {
            device_extern_type_matches_erased(state.ctx, expected, actual)
        });
    if !shape_matches {
        return Err(format!(
            "device extern `@{}` does not match the same-name LLVM declaration's parameter, result, or pointer address-space types",
            decl.export_name
        ));
    }
    Ok(())
}

/// Names of the module's externally consumed function definitions: kernel
/// entries plus `#[device]` exports. One shared list feeds both `@llvm.used`
/// emission and the internalization public-API list so the two root sets can
/// never drift apart. The union (rather than kernels-else-device-functions)
/// keeps a `#[device]` export visible even when the module also has kernels;
/// anonymous helpers carry neither marker and stay internalizable.
fn root_function_names<'a>(state: &'a ModuleExportState<'_>) -> Vec<&'a str> {
    let mut names: Vec<&str> = state
        .all_kernels
        .iter()
        .map(|kernel| kernel.name.as_str())
        .chain(state.device_functions.iter().map(String::as_str))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

fn emit_llvm_used(output: &mut String, state: &ModuleExportState<'_>) -> Result<(), String> {
    let function_names = root_function_names(state);
    let mut global_names = state
        .retained_globals
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    global_names.sort_unstable();
    global_names.dedup();
    if function_names.is_empty() && global_names.is_empty() {
        return Ok(());
    }

    let mut used_refs = Vec::with_capacity(function_names.len() + global_names.len());
    let element_type = if state.legacy_typed_pointers() {
        for name in function_names {
            let mut reference = String::from("i8* bitcast (");
            state.export_named_function_pointer_type(name, &mut reference)?;
            write!(&mut reference, " @{name} to i8*)").unwrap();
            used_refs.push(reference);
        }
        for name in global_names {
            let info = state
                .global_symbols
                .get(name)
                .ok_or_else(|| format!("retained global `@{name}` was not indexed"))?;
            let mut reference = String::from("i8* ");
            if info.address_space == 0 {
                reference.push_str("bitcast (");
                state.export_type(info.value_type, &mut reference)?;
                write!(&mut reference, "* @{name} to i8*)").unwrap();
            } else {
                reference.push_str("addrspacecast (");
                state.export_type(info.value_type, &mut reference)?;
                write!(
                    &mut reference,
                    " addrspace({})* @{name} to i8*)",
                    info.address_space
                )
                .unwrap();
            }
            used_refs.push(reference);
        }
        "i8*"
    } else {
        used_refs.extend(
            function_names
                .into_iter()
                .map(|name| format!("ptr @{name}")),
        );
        for name in global_names {
            let info = state
                .global_symbols
                .get(name)
                .ok_or_else(|| format!("retained global `@{name}` was not indexed"))?;
            if info.address_space == 0 {
                used_refs.push(format!("ptr @{name}"));
            } else {
                used_refs.push(format!(
                    "ptr addrspacecast (ptr addrspace({}) @{name} to ptr)",
                    info.address_space
                ));
            }
        }
        "ptr"
    };

    writeln!(output).unwrap();
    writeln!(
        output,
        "@llvm.used = appending global [{} x {element_type}] [{}], section \"llvm.metadata\"",
        used_refs.len(),
        used_refs.join(", ")
    )
    .unwrap();
    Ok(())
}

fn validate_device_extern_decl(
    decl: &DeviceExternDecl,
    legacy_typed_pointers: bool,
) -> Result<(), String> {
    if decl.export_name.is_empty()
        || !decl.export_name.chars().enumerate().all(|(index, ch)| {
            ch.is_ascii_alphabetic() || ch == '_' || ch == '$' || (index > 0 && ch.is_ascii_digit())
        })
    {
        return Err(format!(
            "device-extern symbol `{}` is not valid in the NVVM global-identifier subset `[A-Za-z$_][A-Za-z$_0-9]*`",
            decl.export_name
        ));
    }
    if decl.export_name.starts_with("llvm_") {
        return Err(format!(
            "device-extern symbol `{}` uses the `llvm_` prefix, which cuda-oxide reserves for LLVM intrinsics; rename the symbol or provide a wrapper",
            decl.export_name
        ));
    }
    if decl
        .param_types
        .iter()
        .any(|ty| matches!(ty, DeviceExternType::Void))
    {
        return Err(format!(
            "device extern `@{}` has a `void` parameter",
            decl.export_name
        ));
    }
    if matches!(decl.return_type, DeviceExternType::Array { .. })
        || decl
            .param_types
            .iter()
            .any(|ty| matches!(ty, DeviceExternType::Array { .. }))
    {
        return Err(format!(
            "device extern `@{}` passes an array by value, which is not supported yet; pass a pointer to the array instead",
            decl.export_name
        ));
    }
    if legacy_typed_pointers
        && (decl.return_type.contains_float16()
            || decl
                .param_types
                .iter()
                .any(DeviceExternType::contains_float16))
    {
        return Err(format!(
            "device extern `@{}` uses `half`, which is not supported by the CUDA 12 legacy LLVM 7 NVVM dialect",
            decl.export_name
        ));
    }
    if legacy_typed_pointers
        && (decl.return_type.ext_attr().is_some()
            || decl.param_types.iter().any(|ty| ty.ext_attr().is_some()))
    {
        return Err(format!(
            "device extern `@{}` passes a sub-32-bit integer or `bool` by value, which is not supported by the CUDA 12 legacy LLVM 7 NVVM dialect; use i32/u32 or pass a pointer",
            decl.export_name
        ));
    }

    // Render both forms up front. This validates zero-width integers,
    // `void*`, and invalid array element types even if only one dialect is
    // selected for this compilation.
    decl.return_type.llvm_string(false)?;
    decl.return_type.llvm_string(true)?;
    for ty in &decl.param_types {
        ty.llvm_string(false)?;
        ty.llvm_string(true)?;
    }
    Ok(())
}

fn index_device_externs(
    state: &mut ModuleExportState<'_>,
    device_externs: &[DeviceExternDecl],
) -> Result<(), String> {
    for decl in device_externs {
        validate_device_extern_decl(decl, state.legacy_typed_pointers())?;
        validate_device_extern_function_shape(state, decl)?;
        if let Some(previous) = state.device_externs.get(&decl.export_name) {
            if previous != decl {
                return Err(format!(
                    "conflicting device-extern declarations for `@{}`",
                    decl.export_name
                ));
            }
            continue;
        }
        state
            .device_externs
            .insert(decl.export_name.clone(), decl.clone());
    }
    Ok(())
}

fn verify_legacy_text(output: &str, state: &ModuleExportState<'_>) -> Result<(), String> {
    if !state.legacy_typed_pointers() {
        return Ok(());
    }
    if contains_opaque_pointer_type(output) {
        return Err(
            "legacy LLVM 7 output still contains unsupported opaque `ptr` syntax".to_string(),
        );
    }
    Ok(())
}

/// Detect the opaque-pointer type token without mistaking `@ptr`, `%ptr`,
/// quoted filenames, metadata, comments, or dotted intrinsic names for a
/// type. This is a final check after structural type printing, not a full LLVM
/// lexer.
fn contains_opaque_pointer_type(output: &str) -> bool {
    let bytes = output.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b';' => {
                index += 1;
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            b'"' => {
                index += 1;
                while index < bytes.len() {
                    if bytes[index] == b'\\' {
                        index = (index + 2).min(bytes.len());
                    } else if bytes[index] == b'"' {
                        index += 1;
                        break;
                    } else {
                        index += 1;
                    }
                }
            }
            byte if byte.is_ascii_alphanumeric() || byte == b'_' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
                {
                    index += 1;
                }
                if &bytes[start..index] == b"ptr"
                    && !matches!(
                        start.checked_sub(1).map(|previous| bytes[previous]),
                        Some(b'@' | b'%' | b'!' | b'$' | b'.')
                    )
                {
                    return true;
                }
            }
            _ => index += 1,
        }
    }
    false
}

/// Internal implementation of module export with device externs.
pub(super) fn export_module_with_externs_impl(
    ctx: &Context,
    module: &ModuleOp,
    device_externs: &[DeviceExternDecl],
    config: &dyn ExportBackendConfig,
) -> Result<ExportedModule, String> {
    validate_export_config(config)?;
    let mut output = String::new();
    let emit_all_annotations = config.emit_all_kernel_annotations();
    let emit_ptx_kernel_keyword = config.emit_ptx_kernel_keyword();
    let mut state = ModuleExportState::new(
        ctx,
        emit_ptx_kernel_keyword,
        config.debug_kind(),
        config.nvvm_ir_dialect(),
        config.function_local_static_placement(),
    );
    index_module_symbols(&mut state, module)?;
    prepare_debug_shared_function_scopes(&mut state, module)?;
    index_device_externs(&mut state, device_externs)?;

    // 1. Header
    writeln!(
        &mut output,
        "; ModuleID = '{}'",
        Operation::get_opid(module.get_operation(), ctx)
    )
    .unwrap();
    writeln!(
        &mut output,
        "source_filename = \"{}\"",
        module.get_symbol_name(ctx)
    )
    .unwrap();
    writeln!(
        &mut output,
        "target datalayout = \"{}\"",
        config.datalayout()
    )
    .unwrap();
    writeln!(&mut output, "target triple = \"nvptx64-nvidia-cuda\"").unwrap();
    writeln!(&mut output).unwrap();

    // 2. Device extern declarations (before function definitions)
    //
    // NOTE: We intentionally do NOT emit LLVM attributes on these declarations.
    // The external LTOIR (from nvcc -dc -dlto) already contains proper attributes
    // (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
    // When nvJitLink performs LTO linking, it uses the definition's attributes.
    // Attributes on declarations are redundant and were causing issues where
    // all externs incorrectly got the same attribute group.
    if !device_externs.is_empty() {
        writeln!(
            &mut output,
            "; External device function declarations (resolved by nvJitLink)"
        )
        .unwrap();
        let mut emitted_names = std::collections::HashSet::new();
        for decl in device_externs {
            // Identical duplicates have already been validated. Emit a single
            // declaration for each link symbol.
            if !emitted_names.insert(decl.export_name.as_str()) {
                continue;
            }
            write!(&mut output, "declare ").unwrap();
            // Return-position attributes precede the type: `declare signext i8 @f()`.
            if let Some(attr) = decl.return_type.ext_attr() {
                write!(&mut output, "{attr} ").unwrap();
            }
            decl.return_type
                .write_llvm(&mut output, state.legacy_typed_pointers())?;
            write!(&mut output, " @{}(", decl.export_name).unwrap();
            for (index, param) in decl.param_types.iter().enumerate() {
                if index != 0 {
                    write!(&mut output, ", ").unwrap();
                }
                param.write_llvm_with_attr(&mut output, state.legacy_typed_pointers())?;
            }
            writeln!(&mut output, ")").unwrap();
        }
        writeln!(&mut output).unwrap();
    }

    // 3. Process Globals and Functions (including intrinsic declarations)
    // Skip device extern declarations - they were already emitted in section 2 with proper attributes
    let device_extern_names: std::collections::HashSet<&str> = device_externs
        .iter()
        .map(|d| d.export_name.as_str())
        .collect();

    let region = module.get_region(ctx).deref(ctx);
    if let Some(block) = region.iter(ctx).next() {
        let mut last_was_decl = false;
        for op in block.deref(ctx).iter(ctx) {
            if let Some(func) = Operation::get_op::<ops::FuncOp>(op, ctx) {
                let is_decl = func.get_operation().deref(ctx).regions().count() == 0;
                let func_name = func.get_symbol_name(ctx);

                // Skip device extern declarations - already emitted in section 2
                if is_decl && device_extern_names.contains(func_name.as_ref()) {
                    continue;
                }

                if !is_decl && last_was_decl {
                    writeln!(&mut output).unwrap();
                }

                state.export_function(&func, &mut output)?;
                last_was_decl = is_decl;
            } else if let Some(global) = Operation::get_op::<ops::GlobalOp>(op, ctx) {
                state.export_global(&global, &mut output)?;
                last_was_decl = false;
            } else {
                if state.legacy_typed_pointers() {
                    return Err(format!(
                        "legacy LLVM 7 export does not support top-level operation `{}`",
                        Operation::get_opid(op, ctx)
                    ));
                }
                writeln!(
                    &mut output,
                    "; Unsupported top-level op: {}",
                    Operation::get_opid(op, ctx)
                )
                .unwrap();
                last_was_decl = false;
            }
        }
    }

    // 4. @llvm.used — preserve kernels and/or standalone device functions from DCE
    //
    // Kernels have no callers in the device module (invoked from host), and standalone
    // device functions have no callers when compiled without a kernel (consumed by
    // external C++ via LTOIR). Both need @llvm.used to survive optimization.
    if config.emit_llvm_used() {
        emit_llvm_used(&mut output, &state)?;
    }

    // 5. Debug intrinsic declarations used by full-debug local variables.
    if state.debug_declare_used || state.debug_value_used {
        writeln!(&mut output).unwrap();
        state.emit_debug_intrinsic_declarations(&mut output);
    }

    // 6. Emit attribute groups for convergent intrinsics used by module functions
    // Note: Device extern declarations no longer get attribute groups - see section 2 comment.
    if state.convergent_used {
        writeln!(&mut output).unwrap();
        writeln!(&mut output, "attributes #0 = {{ convergent }}").unwrap();
    }

    // 7. nvvm.annotations metadata
    if needs_nvvm_annotations(&state, emit_all_annotations) {
        writeln!(&mut output).unwrap();
        emit_nvvm_annotations(&mut output, &mut state, emit_all_annotations)?;
    }

    // 8. nvvmir.version metadata (if backend requires)
    if config.emit_nvvmir_version() {
        writeln!(&mut output).unwrap();
        emit_nvvmir_version(&mut output, &mut state, config.nvvmir_version());
    }

    // 9. DWARF metadata (if requested and source locations exist)
    if state.has_debug_metadata() {
        writeln!(&mut output).unwrap();
        state.emit_debug_metadata(&mut output);
    }

    verify_legacy_text(&output, &state)?;
    Ok(ExportedModule {
        llvm_ir: output,
        public_symbols: public_symbols(&state),
    })
}

fn public_symbols(state: &ModuleExportState<'_>) -> Vec<String> {
    let mut symbols: Vec<String> = root_function_names(state)
        .into_iter()
        .map(str::to_owned)
        .collect();
    symbols.extend(state.public_globals.iter().cloned());
    symbols.sort_unstable();
    symbols.dedup();
    symbols
}

/// Export a module op to a String containing LLVM IR with custom backend configuration.
///
/// The `config` parameter controls backend-specific IR generation options like
/// data layout, metadata emission, and symbol preservation.
pub(super) fn export_module_to_string_with_config(
    ctx: &Context,
    module: &ModuleOp,
    config: &dyn ExportBackendConfig,
) -> Result<String, String> {
    validate_export_config(config)?;
    let mut output = String::new();
    let emit_all_annotations = config.emit_all_kernel_annotations();
    let emit_ptx_kernel_keyword = config.emit_ptx_kernel_keyword();
    let mut state = ModuleExportState::new(
        ctx,
        emit_ptx_kernel_keyword,
        config.debug_kind(),
        config.nvvm_ir_dialect(),
        config.function_local_static_placement(),
    );
    index_module_symbols(&mut state, module)?;
    prepare_debug_shared_function_scopes(&mut state, module)?;

    // 1. Header
    writeln!(
        &mut output,
        "; ModuleID = '{}'",
        Operation::get_opid(module.get_operation(), ctx)
    )
    .unwrap();
    writeln!(
        &mut output,
        "source_filename = \"{}\"",
        module.get_symbol_name(ctx)
    )
    .unwrap();

    // Use backend-specific data layout
    writeln!(
        &mut output,
        "target datalayout = \"{}\"",
        config.datalayout()
    )
    .unwrap();
    writeln!(&mut output, "target triple = \"nvptx64-nvidia-cuda\"").unwrap();
    writeln!(&mut output).unwrap(); // Separate header from body

    // 2. Process Globals and Functions (including intrinsic declarations)
    let region = module.get_region(ctx).deref(ctx);
    if let Some(block) = region.iter(ctx).next() {
        let mut last_was_decl = false;
        for op in block.deref(ctx).iter(ctx) {
            if let Some(func) = Operation::get_op::<ops::FuncOp>(op, ctx) {
                let is_decl = func.get_operation().deref(ctx).regions().count() == 0;

                // If we are transitioning from a declaration to a definition (or anything else)
                // insert a newline to separate the declaration block from the definitions.
                if !is_decl && last_was_decl {
                    writeln!(&mut output).unwrap();
                }

                state.export_function(&func, &mut output)?;
                last_was_decl = is_decl;
            } else if let Some(global) = Operation::get_op::<ops::GlobalOp>(op, ctx) {
                // Export global variable (typically shared memory)
                state.export_global(&global, &mut output)?;
                last_was_decl = false;
            } else {
                if state.legacy_typed_pointers() {
                    return Err(format!(
                        "legacy LLVM 7 export does not support top-level operation `{}`",
                        Operation::get_opid(op, ctx)
                    ));
                }
                writeln!(
                    &mut output,
                    "; Unsupported top-level op: {}",
                    Operation::get_opid(op, ctx)
                )
                .unwrap();
                last_was_decl = false;
            }
        }
    }

    // Emit @llvm.used if backend requests it (prevents symbols from being optimized away).
    //
    // WHY THIS IS NEEDED:
    // Kernels have no callers within the device module - they're invoked by host code.
    // Standalone device functions have no callers when compiled without a kernel - they're
    // consumed by external C++ via LTOIR linking.
    // Without explicit marking, LLVM's optimizer sees them as "dead code" and removes them.
    // The @llvm.used global tells LLVM: "preserve these symbols, they're used externally."
    if config.emit_llvm_used() {
        emit_llvm_used(&mut output, &state)?;
    }

    // Emit debug intrinsic declarations used by full-debug local variables.
    if state.debug_declare_used || state.debug_value_used {
        writeln!(&mut output).unwrap();
        state.emit_debug_intrinsic_declarations(&mut output);
    }

    // Emit attributes section if convergent operations were used
    if state.convergent_used {
        writeln!(&mut output).unwrap();
        writeln!(&mut output, "attributes #0 = {{ convergent }}").unwrap();
    }

    // Emit nvvm.annotations metadata
    // - Default: Only for kernels with cluster configuration or launch bounds
    // - Alternate backends: May require annotations for ALL kernels
    if needs_nvvm_annotations(&state, emit_all_annotations) {
        writeln!(&mut output).unwrap();
        emit_nvvm_annotations(&mut output, &mut state, emit_all_annotations)?;
    }

    // Emit !nvvmir.version metadata if backend requests it
    if config.emit_nvvmir_version() {
        writeln!(&mut output).unwrap();
        emit_nvvmir_version(&mut output, &mut state, config.nvvmir_version());
    }

    // Emit DWARF line-table metadata if debug export requested it and at least
    // one function had a real source location.
    if state.has_debug_metadata() {
        writeln!(&mut output).unwrap();
        state.emit_debug_metadata(&mut output);
    }

    verify_legacy_text(&output, &state)?;
    Ok(output)
}

#[cfg(test)]
mod legacy_text_tests {
    use super::contains_opaque_pointer_type;

    #[test]
    fn opaque_pointer_scan_ignores_names_strings_and_comments() {
        assert!(!contains_opaque_pointer_type(
            "; ptr in a comment\nsource_filename = \"ptr\"\ndefine void @ptr() {\n%ptr = add i32 1, 2\nret void\n}\n"
        ));
        assert!(!contains_opaque_pointer_type(
            "declare i8* @llvm.ptr.annotation.p0(i8*, i8*, i8*, i32, i8*)"
        ));
        assert!(contains_opaque_pointer_type("define void @f(ptr %value)"));
        assert!(contains_opaque_pointer_type(
            "load i32, ptr addrspace(1) %p"
        ));
    }
}
