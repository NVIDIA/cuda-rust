/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Function and basic block emission.
//!
//! Contains the pre-pass that assigns deterministic anonymous-value names so the
//! textual IR is stable across runs, and the block-argument → PHI-node translation
//! that bridges pliron's basic-block argument convention to LLVM's PHI-node convention.

use rustc_hash::FxHashMap;
use std::fmt::Write;

use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr, TypeAttr},
        op_interfaces::{BranchOpInterface, SymbolOpInterface},
        type_interfaces::FunctionTypeInterface,
        types::IntegerType,
    },
    context::Ptr,
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    printable::Printable,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use reserved_oxide_symbols::{
    LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX, LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
};

use crate::{
    attributes::FPHalfAttr,
    ops::{self, FuncOp, GlobalOpExt},
    types::{ArrayType, FuncType, PointerType, StructLayout, StructType},
};

use super::{
    literals::{format_float_literal, format_half_literal},
    names::{decode_intrinsic_identifier, has_device_prefix, strip_device_prefix},
    state::{
        FunctionAbiAlignment, GridConstantParameter, KernelBlockGeometry, KernelClusterConfig,
        KernelGridConstants, KernelInfo, KernelLaunchBounds, ModuleExportState, PredecessorMap,
    },
};

/// Flatten a descriptive string so it cannot escape a single `;` comment line.
///
/// An LLVM comment runs to end of line, so an embedded newline would turn the
/// remainder of the text into IR the parser has to accept. Control characters
/// become spaces; nothing else is altered, which keeps ordinary Rust paths
/// (`::`, generics, `<impl …>`) readable verbatim.
fn sanitize_comment(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

/// Final LLVM spelling of a pliron function symbol.
///
/// Shared-global owner attributes carry the raw symbol, so the module prepass,
/// function definition, and global attachment must all normalize it exactly
/// once through the same helper.
pub(super) fn exported_function_name(raw_name: &str) -> String {
    if raw_name.starts_with("llvm_") {
        decode_intrinsic_identifier(raw_name)
    } else {
        strip_device_prefix(raw_name)
    }
}

impl<'a> ModuleExportState<'a> {
    /// Export a global variable (typically shared memory for GPU kernels).
    pub(super) fn export_global(
        &mut self,
        global: &ops::GlobalOp,
        output: &mut String,
    ) -> Result<(), String> {
        use crate::attributes::LinkageAttr;

        let name = global.get_symbol_name(self.ctx);
        let ty = global.get_type(self.ctx);
        let address_space = global.address_space(self.ctx);

        // NVVM only permits module-scope variables in generic/global,
        // shared, or constant memory. Local memory (address space 5) is for
        // per-thread allocations, not LLVM globals.
        if self.nvvm_ir_dialect.is_some() && !matches!(address_space, 0 | 1 | 3 | 4) {
            return Err(format!(
                "NVVM global `@{name}` uses unsupported address space {address_space}; expected generic (0), global (1), shared (3), or constant (4)"
            ));
        }

        // Lowering names every static shared allocation `__shared_mem_N`, so
        // the emitted symbol carries no hint of which Rust `static` it came
        // from. Anyone attributing a kernel's shared-memory footprint back to
        // source has only the exported IR to work from, so record the Rust
        // path as a comment on the line above the definition. It is a comment
        // rather than a symbol rename because the generated names are matched
        // literally elsewhere, and because a source path is not a legal LLVM
        // identifier in general.
        if let Some(source_name) = global.shared_source_name(self.ctx) {
            writeln!(
                output,
                "; shared source: {}",
                sanitize_comment(&source_name)
            )
            .unwrap();
        }

        // Check for external linkage (dynamic shared memory)
        let is_external = global
            .get_attr_llvm_global_linkage(self.ctx)
            .map(|linkage| matches!(*linkage, LinkageAttr::ExternalLinkage))
            .unwrap_or(false);

        // Get alignment from attribute, or compute natural alignment from type
        let alignment = global.get_alignment(self.ctx).unwrap_or_else(|| {
            // Compute natural alignment from array element type
            // For [N x T], alignment is size_of(T) (common case: f32 = 4, i64 = 8)
            let ty_ref = ty.deref(self.ctx);
            if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
                let elem_ty = array_ty.elem_type();
                let elem_ref = elem_ty.deref(self.ctx);
                if elem_ref.is::<pliron::builtin::types::IntegerType>() {
                    let int_ty = elem_ref
                        .downcast_ref::<pliron::builtin::types::IntegerType>()
                        .unwrap();
                    u64::from(int_ty.width() / 8)
                } else if elem_ref.is::<pliron::builtin::types::FP32Type>() {
                    4
                } else {
                    8 // Default alignment (FP64Type and unknown types)
                }
            } else {
                8 // Default alignment
            }
        });
        let debug_attachment = if !is_external
            && matches!(
                address_space,
                crate::types::address_space::GLOBAL
                    | crate::types::address_space::SHARED
                    | crate::types::address_space::CONSTANT
            ) {
            match ops::debug_global_variable(self.ctx, global.get_operation()) {
                Some(info) => {
                    let owner = ops::debug_global_owner_function(self.ctx, global.get_operation())
                        .and_then(|raw| {
                            let exported = exported_function_name(&raw);
                            self.function_source_names
                                .get(&exported)
                                .is_some_and(|indexed| indexed == &raw)
                                .then_some(exported)
                        });
                    self.debug_global_variable(
                        name.as_ref(),
                        Some(alignment),
                        address_space,
                        owner.as_deref(),
                        &info,
                    )?
                    .map(|id| format!(", !dbg !{id}"))
                    .unwrap_or_default()
                }
                None => String::new(),
            }
        } else {
            String::new()
        };

        if is_external {
            // External linkage: declaration with size determined elsewhere.
            write!(
                output,
                "@{name} = external addrspace({address_space}) global "
            )
            .unwrap();
            self.export_type(ty, output)?;
            writeln!(output, ", align {alignment}{debug_attachment}").unwrap();
        } else {
            self.public_globals.push(name.to_string());
            // Defined static storage in the global's address space. The LLVM
            // definition retains external linkage for host-side symbol lookup.
            //
            // `constant` rather than `global` when the LLVM global is marked
            // constant. That keyword lets `opt` treat a read of this storage as
            // invariant: it both
            // enables `isOnlyCopiedFromConstantMemory` to delete a copy of the
            // data into a stack slot, and makes `llc` select `ld.global.nc`
            // (the read-only data cache) for the load. External linkage is
            // retained either way; `constant` constrains writes, not visibility.
            let storage_keyword = if global.is_constant(self.ctx) {
                "constant"
            } else {
                "global"
            };
            write!(
                output,
                "@{name} = addrspace({address_space}) {storage_keyword} "
            )
            .unwrap();
            self.export_type(ty, output)?;
            if let Some(encoded) = global.initializer_relocations(self.ctx) {
                let hex = global.initializer_hex(self.ctx).ok_or_else(|| {
                    format!(
                        "global `@{name}` carries pointer relocations without initializer bytes"
                    )
                })?;
                let bytes = decode_hex_initializer(&hex)?;
                write!(output, " ").unwrap();
                self.export_relocated_initializer(ty, &bytes, &encoded, output)?;
                writeln!(output, ", align {alignment}{debug_attachment}").unwrap();
            } else if let Some(hex) = global.initializer_hex(self.ctx) {
                let bytes = decode_hex_initializer(&hex)?;
                write!(output, " ").unwrap();
                self.export_byte_initializer(ty, &bytes, output)?;
                writeln!(output, ", align {alignment}{debug_attachment}").unwrap();
            } else {
                // NVVM forbids initialized shared variables. `undef` denotes
                // uninitialized shared storage and is required by both legacy and
                // modern NVVM IR. Keep the ordinary llc/PTX path's historical zero
                // initializer for compatibility.
                let initializer = if self.nvvm_ir_dialect.is_some() && address_space == 3 {
                    "undef"
                } else {
                    "zeroinitializer"
                };
                writeln!(
                    output,
                    " {initializer}, align {alignment}{debug_attachment}"
                )
                .unwrap();
            }
        }

        Ok(())
    }

    /// Render an evaluated Rust allocation without interpreting its contents.
    ///
    /// The lowering boundary deliberately represents initialized globals as
    /// `[N x i8]`. Escaping every byte keeps all bit patterns intact, including
    /// NaN payloads, and makes Rust padding explicit without teaching this
    /// exporter Rust's layout rules.
    fn export_byte_initializer(
        &self,
        ty: pliron::r#type::TypeHandle,
        bytes: &[u8],
        output: &mut String,
    ) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        let array_ty = ty_ref
            .downcast_ref::<crate::types::ArrayType>()
            .ok_or_else(|| {
                format!(
                    "explicit global initializer requires `[N x i8]` storage, found `{}`",
                    ty_ref.disp(self.ctx)
                )
            })?;
        let elem_ty = array_ty.elem_type();
        let elem_ref = elem_ty.deref(self.ctx);
        let is_i8 = elem_ref
            .downcast_ref::<IntegerType>()
            .is_some_and(|int_ty| int_ty.width() == 8);
        if !is_i8 || array_ty.size() != bytes.len() as u64 {
            return Err(format!(
                "explicit global initializer has {} bytes but storage type is `{}`; expected `[{} x i8]`",
                bytes.len(),
                ty_ref.disp(self.ctx),
                bytes.len()
            ));
        }

        write!(output, "c\"").unwrap();
        for byte in bytes {
            write!(output, "\\{byte:02X}").unwrap();
        }
        write!(output, "\"").unwrap();
        Ok(())
    }

    /// Render a byte-exact initializer whose pointer-width slots are LLVM
    /// relocation expressions rather than literal bytes.
    fn export_relocated_initializer(
        &self,
        ty: pliron::r#type::TypeHandle,
        bytes: &[u8],
        encoded: &str,
        output: &mut String,
    ) -> Result<(), String> {
        let mut relocations = ops::decode_global_initializer_relocations(encoded)?;
        if relocations.is_empty() {
            return Err(
                "global initializer relocation metadata contains no relocations".to_string(),
            );
        }
        relocations.sort_by_key(|relocation| relocation.source_offset);

        let ty_ref = ty.deref(self.ctx);
        let storage = ty_ref.downcast_ref::<StructType>().ok_or_else(|| {
            format!(
                "relocated global initializer requires segmented struct storage, found `{}`",
                ty_ref.disp(self.ctx)
            )
        })?;
        let fields: Vec<_> = storage.fields().collect();
        let mut field_index = 0usize;
        let mut cursor = 0u64;
        let mut first = true;
        let (open, close) = match storage.layout() {
            StructLayout::Packed => ("<{ ", " }>"),
            StructLayout::Unpacked => ("{ ", " }"),
        };

        write!(output, "{open}").unwrap();
        for (index, relocation) in relocations.iter().enumerate() {
            if relocation.width_bytes != 8 {
                return Err(format!(
                    "global initializer relocation {index} uses unsupported {}-byte pointer storage; CUDA global/constant pointers require 8 bytes",
                    relocation.width_bytes
                ));
            }
            if relocation.source_offset < cursor {
                return Err(format!(
                    "global initializer relocation {index} overlaps the previous relocation"
                ));
            }
            let end = relocation
                .source_offset
                .checked_add(u64::from(relocation.width_bytes))
                .ok_or_else(|| format!("global initializer relocation {index} offset overflows"))?;
            if end > bytes.len() as u64 {
                return Err(format!(
                    "global initializer relocation {index} occupies bytes {}..{} but the initializer has {} bytes",
                    relocation.source_offset,
                    end,
                    bytes.len()
                ));
            }

            if relocation.source_offset > cursor {
                let span = &bytes[cursor as usize..relocation.source_offset as usize];
                let field_ty = *fields.get(field_index).ok_or_else(|| {
                    "segmented global storage is missing a literal-byte field".to_string()
                })?;
                self.validate_initializer_byte_field(field_ty, span.len() as u64)?;
                if !first {
                    write!(output, ", ").unwrap();
                }
                self.export_type(field_ty, output)?;
                write!(output, " ").unwrap();
                self.export_byte_initializer(field_ty, span, output)?;
                first = false;
                field_index += 1;
            }

            let field_ty = *fields.get(field_index).ok_or_else(|| {
                "segmented global storage is missing a relocation field".to_string()
            })?;
            self.validate_initializer_pointer_field(field_ty, relocation.width_bytes)?;
            if !first {
                write!(output, ", ").unwrap();
            }
            self.export_type(field_ty, output)?;
            write!(output, " ").unwrap();
            self.export_global_relocation_constant(relocation, output)?;
            first = false;
            field_index += 1;
            cursor = end;
        }

        if cursor < bytes.len() as u64 {
            let span = &bytes[cursor as usize..];
            let field_ty = *fields.get(field_index).ok_or_else(|| {
                "segmented global storage is missing its trailing literal-byte field".to_string()
            })?;
            self.validate_initializer_byte_field(field_ty, span.len() as u64)?;
            if !first {
                write!(output, ", ").unwrap();
            }
            self.export_type(field_ty, output)?;
            write!(output, " ").unwrap();
            self.export_byte_initializer(field_ty, span, output)?;
            field_index += 1;
        }

        if field_index != fields.len() {
            return Err(format!(
                "segmented global storage has {} fields, but relocation metadata consumed {field_index}",
                fields.len()
            ));
        }
        write!(output, "{close}").unwrap();
        Ok(())
    }

    fn validate_initializer_byte_field(
        &self,
        ty: pliron::r#type::TypeHandle,
        expected_len: u64,
    ) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        let array = ty_ref.downcast_ref::<ArrayType>().ok_or_else(|| {
            format!(
                "literal initializer span requires `[N x i8]`, found `{}`",
                ty_ref.disp(self.ctx)
            )
        })?;
        let elem = array.elem_type();
        let elem_ref = elem.deref(self.ctx);
        let is_i8 = elem_ref
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 8);
        if !is_i8 || array.size() != expected_len {
            return Err(format!(
                "literal initializer span has {expected_len} bytes but storage field is `{}`",
                ty_ref.disp(self.ctx)
            ));
        }
        Ok(())
    }

    fn validate_initializer_pointer_field(
        &self,
        ty: pliron::r#type::TypeHandle,
        width_bytes: u32,
    ) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        let width = ty_ref
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width)
            .ok_or_else(|| {
                format!(
                    "relocation slot requires an integer field, found `{}`",
                    ty_ref.disp(self.ctx)
                )
            })?;
        if width != width_bytes * 8 {
            return Err(format!(
                "relocation slot is i{width}, expected i{}",
                width_bytes * 8
            ));
        }
        Ok(())
    }

    fn export_global_relocation_constant(
        &self,
        relocation: &ops::GlobalInitializerRelocation,
        output: &mut String,
    ) -> Result<(), String> {
        let target = self
            .global_sources
            .get(&relocation.target_key)
            .ok_or_else(|| {
                format!(
                    "global initializer relocation references unknown rustc global key `{}`",
                    relocation.target_key
                )
            })?;
        if target.address_space != relocation.target_address_space {
            return Err(format!(
                "global initializer relocation target `{}` claims address space {}, but `@{}` is in address space {}",
                relocation.target_key,
                relocation.target_address_space,
                target.symbol,
                target.address_space
            ));
        }
        if !matches!(target.address_space, 1 | 4) {
            return Err(format!(
                "global initializer relocation target `@{}` uses unsupported address space {}",
                target.symbol, target.address_space
            ));
        }
        if let Some(size) = target.initializer_size
            && relocation.target_addend > size
        {
            return Err(format!(
                "global initializer relocation target `@{}` has byte addend {}, but the allocation is only {size} bytes",
                target.symbol, relocation.target_addend
            ));
        }
        if relocation.width_bytes != 8 {
            return Err(format!(
                "global initializer relocation uses unsupported {}-byte destination",
                relocation.width_bytes
            ));
        }

        write!(output, "ptrtoint (").unwrap();
        self.export_global_relocation_pointer(target, relocation.target_addend, output)?;
        write!(output, " to i64)").unwrap();
        Ok(())
    }

    fn export_global_relocation_pointer(
        &self,
        target: &super::state::GlobalSourceInfo,
        byte_addend: u64,
        output: &mut String,
    ) -> Result<(), String> {
        if self.legacy_typed_pointers() {
            write!(output, "i8* addrspacecast (").unwrap();
            if byte_addend == 0 {
                self.export_legacy_global_byte_pointer(target, output)?;
            } else {
                write!(
                    output,
                    "i8 addrspace({})* getelementptr (i8, ",
                    target.address_space
                )
                .unwrap();
                self.export_legacy_global_byte_pointer(target, output)?;
                write!(output, ", i64 {byte_addend})").unwrap();
            }
            write!(output, " to i8*)").unwrap();
        } else {
            write!(output, "ptr addrspacecast (").unwrap();
            if byte_addend == 0 {
                write!(
                    output,
                    "ptr addrspace({}) @{}",
                    target.address_space, target.symbol
                )
                .unwrap();
            } else {
                write!(
                    output,
                    "ptr addrspace({}) getelementptr (i8, ptr addrspace({}) @{}, i64 {byte_addend})",
                    target.address_space,
                    target.address_space,
                    target.symbol
                )
                .unwrap();
            }
            write!(output, " to ptr)").unwrap();
        }
        Ok(())
    }

    fn export_legacy_global_byte_pointer(
        &self,
        target: &super::state::GlobalSourceInfo,
        output: &mut String,
    ) -> Result<(), String> {
        write!(output, "i8 addrspace({})* bitcast (", target.address_space).unwrap();
        self.export_type(target.value_type, output)?;
        write!(
            output,
            " addrspace({})* @{} to i8 addrspace({})*)",
            target.address_space, target.symbol, target.address_space
        )
        .unwrap();
        Ok(())
    }

    /// Validate and retain by-value pointees before any function is emitted.
    /// LLVM 7 uses them in the pointer type itself, including symbol references.
    pub(super) fn grid_constant_parameters(
        &self,
        func: &FuncOp,
    ) -> Result<Vec<GridConstantParameter>, String> {
        let name = func.get_symbol_name(self.ctx);
        let fixed_func_name = exported_function_name(name.as_ref());
        let func_type = func.get_type(self.ctx);
        let func_ty = func_type.deref(self.ctx);
        let attrs = &func.get_operation().deref(self.ctx).attributes;
        let kernel_key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
        let is_kernel = attrs
            .get::<pliron::builtin::attributes::StringAttr>(&kernel_key)
            .is_some_and(|value| value.as_str() == "true");
        for key_id in attrs.0.keys() {
            let key: &str = key_id.as_ref();
            let (prefix, index, typed) =
                if let Some(index) = key.strip_prefix(LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX) {
                    (
                        LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
                        index,
                        attrs.get::<TypeAttr>(key_id).is_some(),
                    )
                } else if let Some(index) = key.strip_prefix(LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX) {
                    (
                        LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX,
                        index,
                        attrs.get::<IntegerAttr>(key_id).is_some(),
                    )
                } else {
                    continue;
                };
            let index = index.parse::<usize>().map_err(|_| {
                format!(
                    "function `@{fixed_func_name}` has malformed grid-constant attribute `{key}`"
                )
            })?;
            if key != format!("{prefix}{index}") || !typed {
                return Err(format!(
                    "function `@{fixed_func_name}` has malformed grid-constant attribute `{key}`"
                ));
            }
            if index >= func_ty.arg_types().len() {
                return Err(format!(
                    "function `@{fixed_func_name}` grid-constant parameter index {index} is out of range for {} parameters",
                    func_ty.arg_types().len()
                ));
            }
        }
        let mut grid_constant_params = Vec::new();
        for (index, arg_ty) in func_ty.arg_types().iter().enumerate() {
            let pointee_key = format!("{LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX}{index}");
            let align_key = format!("{LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX}{index}");
            let pointee_key_id: pliron::identifier::Identifier = pointee_key
                .as_str()
                .try_into()
                .map_err(|_| format!("invalid grid-constant attribute name `{pointee_key}`"))?;
            let align_key_id: pliron::identifier::Identifier = align_key
                .as_str()
                .try_into()
                .map_err(|_| format!("invalid grid-constant attribute name `{align_key}`"))?;
            let pointee = attrs.get::<TypeAttr>(&pointee_key_id);
            let alignment = attrs.get::<IntegerAttr>(&align_key_id);
            let (Some(pointee), Some(alignment)) = (pointee, alignment) else {
                if pointee.is_some() || alignment.is_some() {
                    return Err(format!(
                        "function `@{fixed_func_name}` parameter {index} has incomplete grid-constant metadata"
                    ));
                }
                continue;
            };
            if !is_kernel {
                return Err(format!(
                    "non-kernel function `@{fixed_func_name}` parameter {index} is marked grid-constant"
                ));
            }
            let arg_ref = arg_ty.deref(self.ctx);
            let Some(pointer) = arg_ref.downcast_ref::<PointerType>() else {
                return Err(format!(
                    "kernel `@{fixed_func_name}` grid-constant parameter {index} is not a pointer"
                ));
            };
            if pointer.address_space() != 0 {
                return Err(format!(
                    "kernel `@{fixed_func_name}` grid-constant parameter {index} must use generic address space 0, found {}",
                    pointer.address_space()
                ));
            }
            let pointee = pointee.get_type(self.ctx);
            if self.fixed_type_has_storage(pointee) != Some(true) {
                return Err(format!(
                    "kernel `@{fixed_func_name}` grid-constant parameter {index} requires a sized, non-zero pointee"
                ));
            }
            let value = alignment.value();
            if value.bw() > 64 {
                return Err(format!(
                    "kernel `@{fixed_func_name}` grid-constant alignment for parameter {index} is wider than 64 bits"
                ));
            }
            let alignment = value.to_u64();
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(format!(
                    "kernel `@{fixed_func_name}` grid-constant alignment for parameter {index} must be a non-zero power of two, found {alignment}"
                ));
            }
            grid_constant_params.push(GridConstantParameter {
                index,
                pointee,
                alignment,
            });
        }
        Ok(grid_constant_params)
    }

    fn export_grid_constant_parameter_attrs(
        &self,
        pointee: TypeHandle,
        alignment: u64,
        output: &mut String,
    ) -> Result<(), String> {
        if self.legacy_typed_pointers() {
            write!(output, " byval align {alignment}").unwrap();
        } else {
            write!(output, " byval(").unwrap();
            self.export_type(pointee, output)?;
            write!(output, ") align {alignment}").unwrap();
        }
        Ok(())
    }

    pub(super) fn export_function(
        &mut self,
        func: &FuncOp,
        output: &mut String,
    ) -> Result<(), String> {
        let func_name = func.get_symbol_name(self.ctx);
        let func_name_str: &str = func_name.as_ref();
        // LLVM intrinsics (NVVM and standard, e.g. llvm.fptosi.sat) use dots in IR
        // but Pliron IR identifiers use underscores; convert for export.
        // Strip cuda_oxide_device_ prefixes and restore LLVM intrinsic dots at
        // the final export boundary. Calls and AS3 owner scopes share this map.
        let fixed_func_name = exported_function_name(func_name_str);

        // Check for kernel attribute
        let kernel_key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
        let attrs = &func.get_operation().deref(self.ctx).attributes;
        let llvm_func_attrs = func.get_attr_llvm_func_attrs(self.ctx);
        let is_kernel = attrs
            .get::<pliron::builtin::attributes::StringAttr>(&kernel_key)
            .is_some();

        // Check for cluster dimension attributes (from #[cluster(x,y,z)])
        // These will be emitted as nvvm.annotations metadata
        if is_kernel {
            let get_int = |key: &str| -> Option<u32> {
                let key: pliron::identifier::Identifier = key.try_into().unwrap();
                attrs
                    .get::<pliron::builtin::attributes::IntegerAttr>(&key)
                    .map(|int_attr| int_attr.value().to_u32())
            };

            if let (Some(dim_x), Some(dim_y), Some(dim_z)) = (
                get_int("cluster_dim_x"),
                get_int("cluster_dim_y"),
                get_int("cluster_dim_z"),
            ) {
                self.cluster_kernels.push(KernelClusterConfig {
                    name: fixed_func_name.clone(),
                    dim_x,
                    dim_y,
                    dim_z,
                });
            }

            // Check for launch bounds attributes (from #[launch_bounds(max, min)])
            // and an exact block shape (from #[launch_contract(block = (x,y,z))]).
            // These are emitted as nvvm.annotations metadata for maxntid,
            // minctasm and reqntid.
            let exact_block = match (
                get_int("reqntid_x"),
                get_int("reqntid_y"),
                get_int("reqntid_z"),
            ) {
                (Some(x), Some(y), Some(z)) => Some(KernelBlockGeometry::ExactBlock(x, y, z)),
                _ => None,
            };
            // An exact shape displaces the thread maximum: ptxas rejects an
            // entry carrying both directives, and the shape already bounds the
            // thread count on every axis.
            let geometry =
                exact_block.or_else(|| get_int("maxntid").map(KernelBlockGeometry::MaxThreads));
            if let Some(geometry) = geometry {
                let min_blocks = get_int("minctasm");
                self.launch_bounds_kernels.push(KernelLaunchBounds {
                    name: fixed_func_name.clone(),
                    geometry,
                    min_blocks: if min_blocks == Some(0) {
                        None
                    } else {
                        min_blocks
                    },
                });
            }
        }

        let ft: pliron::r#type::TypeHandle = func.get_type(self.ctx).into();
        let ft_ref = ft.deref(self.ctx);
        let func_ty = ft_ref
            .downcast_ref::<FuncType>()
            .ok_or("Not a function type")?;
        let legacy_atomic_add = self.legacy_nvvm_atomic_add_signature(&fixed_func_name, func_ty)?;

        // Reference validity is transported as a typed LLVM-dialect fact.
        // The exporter does not infer Rust semantics: it only checks that a
        // fact is attached to an in-range pointer parameter of a kernel entry
        // before emitting the corresponding LLVM attributes.
        let mut reference_param_validities = vec![None; func_ty.arg_types().len()];
        for (index, validity) in
            ops::kernel_reference_param_validity_entries(self.ctx, func.get_operation())?
        {
            if !is_kernel {
                return Err(format!(
                    "function `@{fixed_func_name}` carries kernel reference parameter validity but is not a kernel entry"
                ));
            }
            let Some(arg_ty) = func_ty.arg_types().get(index).copied() else {
                return Err(format!(
                    "kernel `@{fixed_func_name}` reference validity parameter index {index} is out of range for {} parameters",
                    func_ty.arg_types().len()
                ));
            };
            if !arg_ty.deref(self.ctx).is::<PointerType>() {
                return Err(format!(
                    "kernel `@{fixed_func_name}` reference validity parameter {index} is not an LLVM pointer"
                ));
            }
            reference_param_validities[index] = Some(validity.0);
        }

        let is_declaration = func.get_operation().deref(self.ctx).regions().count() == 0;
        if legacy_atomic_add.is_some() && !is_declaration {
            return Err(format!(
                "legacy NVVM atomic-add intrinsic `@{fixed_func_name}` must be a declaration, not a definition"
            ));
        }

        self.function_types.insert(fixed_func_name.clone(), ft);

        let grid_constant_params = self
            .function_grid_constants
            .get(&fixed_func_name)
            .cloned()
            .unwrap_or_default();
        if !grid_constant_params.is_empty() {
            let positions = grid_constant_params
                .iter()
                .map(|parameter| {
                    let index = parameter.index;
                    u32::try_from(index + 1).map_err(|_| {
                        format!(
                            "kernel `@{fixed_func_name}` grid-constant parameter index {index} exceeds NVVM's 32-bit position field"
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.grid_constant_kernels.push(KernelGridConstants {
                name: fixed_func_name.clone(),
                positions,
            });
        }

        // MIR lowering records source-language ABI alignments only when the
        // direct LLVM aggregate type is naturally under-aligned. NVVM represents
        // these contracts with the function-level `"align"` property rather than
        // an ordinary LLVM parameter attribute. The low 16 bits are the byte
        // alignment; the high 16 bits are the position (0 = return, arguments
        // start at 1).
        let read_abi_alignment = |key: &str| -> Result<Option<u16>, String> {
            let key_id: pliron::identifier::Identifier = key
                .try_into()
                .map_err(|_| format!("invalid ABI alignment attribute name `{key}`"))?;
            let Some(attribute) = attrs.get::<IntegerAttr>(&key_id) else {
                return Ok(None);
            };
            let value = attribute.value();
            if value.bw() > 64 {
                return Err(format!(
                    "ABI alignment attribute `{key}` is wider than 64 bits"
                ));
            }
            let alignment = value.to_u64();
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(format!(
                    "ABI alignment attribute `{key}` must be a non-zero power of two, found {alignment}"
                ));
            }
            let alignment = u16::try_from(alignment).map_err(|_| {
                format!(
                    "ABI alignment attribute `{key}` exceeds NVVM's 16-bit alignment field: {alignment}"
                )
            })?;
            Ok(Some(alignment))
        };

        let mut abi_alignments = Vec::new();
        if let Some(alignment) = read_abi_alignment("cuda_oxide_return_abi_align")? {
            abi_alignments.push(FunctionAbiAlignment {
                name: fixed_func_name.clone(),
                position: 0,
                alignment,
            });
        }

        if is_kernel {
            for index in 0..func_ty.arg_types().len() {
                let key = format!("cuda_oxide_param_abi_align_{index}");
                let Some(alignment) = read_abi_alignment(&key)? else {
                    continue;
                };
                let position = u16::try_from(index + 1).map_err(|_| {
                    format!(
                        "kernel `@{fixed_func_name}` parameter index {index} exceeds NVVM's 16-bit position field"
                    )
                })?;
                abi_alignments.push(FunctionAbiAlignment {
                    name: fixed_func_name.clone(),
                    position,
                    alignment,
                });
            }
        }
        self.function_abi_alignments.extend(abi_alignments);

        // Track every kernel as an external root. Backends independently decide
        // whether to emit annotations for all of them.
        if is_kernel {
            self.all_kernels.push(KernelInfo {
                name: fixed_func_name.clone(),
            });
        }

        // Track device function definitions (not declarations) for @llvm.used
        // preservation in standalone device-function compilation.
        if !is_declaration && !is_kernel && has_device_prefix(func_name_str) {
            self.device_functions.push(fixed_func_name.clone());
        }

        // A kernel receives its parameters in `.param` space, filled by the
        // host at launch. Shared and local memory are allocated per block and
        // per thread by the device, so the host has no address in either to
        // pass. Carrying one of those state spaces into the entry signature
        // produces a module that ptxas assembles and the driver then refuses,
        // which takes every other kernel in the module down with it.
        //
        // Global and constant are left alone: the host allocates there, so a
        // pointer parameter in those spaces is something it can supply, and
        // the codegen tests rely on it. A device function may carry any state
        // space on its parameters, which is why this is checked for kernels
        // alone.
        //
        // The refusal is deliberately scoped to address spaces 3 (shared) and
        // 5 (local). addrspace(4) constant parameters stay allowed by design:
        // the host owns constant-bank allocation, so it does have an address
        // to pass. addrspace(7) (shared::cluster) is not user-reachable as a
        // parameter type today; if a frontend type ever surfaces it at a
        // kernel boundary, it needs a new arm here for the same reason as
        // plain shared.
        if is_kernel {
            for (index, arg_ty) in func_ty.arg_types().iter().enumerate() {
                let arg_ref = arg_ty.deref(self.ctx);
                let Some(pointer) = arg_ref.downcast_ref::<PointerType>() else {
                    continue;
                };
                let space = match pointer.address_space() {
                    3 => "shared",
                    5 => "local",
                    _ => continue,
                };
                return Err(format!(
                    "kernel `@{fixed_func_name}` parameter {index} is a pointer into {space} \
                     memory (address space {}); {space} memory is allocated by the device at \
                     launch, so the host has no address to pass, and the driver refuses a \
                     module whose entry declares one",
                    pointer.address_space()
                ));
            }
        }

        let ret_ty = func_ty.result_type();

        // Check if function has a body
        if func.get_operation().deref(self.ctx).regions().count() == 0 {
            // Function Declaration
            write!(output, "declare ").unwrap();
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let args = func_ty.arg_types();
            for (i, arg_ty) in args.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                if i == 0
                    && let Some((pointee, address_space)) = legacy_atomic_add
                {
                    self.export_pointer_to(pointee, address_space, output)?;
                } else {
                    self.export_function_parameter_type(&fixed_func_name, i, *arg_ty, output)?;
                }
                let grid_constant = grid_constant_params
                    .iter()
                    .find(|parameter| parameter.index == i);
                if let Some(alignment) = reference_param_validities[i] {
                    write!(output, " nonnull").unwrap();
                    if alignment > 1 && grid_constant.is_none() {
                        write!(output, " align {alignment}").unwrap();
                    }
                }
                if let Some(parameter) = grid_constant {
                    self.export_grid_constant_parameter_attrs(
                        parameter.pointee,
                        parameter.alignment,
                        output,
                    )?;
                }
            }
            write!(output, ")").unwrap();
            if let Some(attrs) = llvm_func_attrs.as_deref() {
                self.export_llvm_attributes(attrs, output)?;
            }

            // Check if this is a known convergent intrinsic
            let is_convergent_intrinsic = Self::is_convergent_intrinsic(&fixed_func_name);
            if is_convergent_intrinsic {
                writeln!(output, " #0").unwrap();
                self.convergent_used = true;
            } else {
                writeln!(output).unwrap();
            }
            // No extra newline after declarations to keep them grouped
            return Ok(());
        }

        // Function Body
        let entry_block_opt = func
            .get_operation()
            .deref(self.ctx)
            .get_region(0)
            .deref(self.ctx)
            .iter(self.ctx)
            .next();

        // Check for alwaysinline attribute (from #[inline(always)]).
        // Emitted as a function attribute keyword between the parameter
        // list and the body open brace.
        let alwaysinline_key: pliron::identifier::Identifier = "alwaysinline".try_into().unwrap();
        let is_alwaysinline = attrs
            .get::<pliron::builtin::attributes::StringAttr>(&alwaysinline_key)
            .is_some();

        if let Some(entry_block) = entry_block_opt {
            let func_loc = func.get_operation().deref(self.ctx).loc();
            let debug_name = ops::debug_function_name(self.ctx, func.get_operation())
                .unwrap_or_else(|| fixed_func_name.clone());
            let debug_scope =
                self.debug_subprogram_for_function(&debug_name, &fixed_func_name, &func_loc);
            if let Some(scope_id) = debug_scope {
                self.register_debug_source_scopes_for_function(scope_id, func.get_operation());
            }

            write!(output, "define ").unwrap();
            if is_kernel && self.emit_ptx_kernel_keyword {
                write!(output, "ptx_kernel ").unwrap();
            }
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let mut value_names = FxHashMap::default();
            let mut next_value_id = 0;

            let block = entry_block.deref(self.ctx);
            let args = block.arguments();
            let mut entry_prologue = String::new();
            // Kernel Rust references may carry importer-proven `nonnull` and
            // pointee `align N`. Every other parameter remains bare.
            //
            // In particular this deliberately does NOT infer `noalias`,
            // `readonly`, `dereferenceable`, or any validity fact from LLVM
            // pointer shape. `DisjointSlice` and raw pointers therefore retain
            // the conservative behavior required by their unsafe construction
            // contracts. Future alias attributes need their own audited proof
            // source rather than extending this mechanical export step.
            for (i, arg) in args.enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                let arg_ty = arg.get_type(self.ctx);
                self.export_function_parameter_type(&fixed_func_name, i, arg_ty, output)?;
                let grid_constant = grid_constant_params
                    .iter()
                    .find(|parameter| parameter.index == i);
                if let Some(alignment) = reference_param_validities[i] {
                    write!(output, " nonnull").unwrap();
                    if alignment > 1 && grid_constant.is_none() {
                        write!(output, " align {alignment}").unwrap();
                    }
                }
                if let Some(parameter) = grid_constant {
                    self.export_grid_constant_parameter_attrs(
                        parameter.pointee,
                        parameter.alignment,
                        output,
                    )?;
                }
                let name = format!("%v{next_value_id}");
                value_names.insert(arg, name.clone());
                if self.legacy_typed_pointers()
                    && let Some(parameter) = grid_constant
                    && !self.is_i8_type(parameter.pointee)
                {
                    let abi_name = format!("{name}.byval");
                    write!(output, " {abi_name}").unwrap();
                    write!(entry_prologue, "  {name} = bitcast ").unwrap();
                    self.export_function_parameter_type(
                        &fixed_func_name,
                        i,
                        arg_ty,
                        &mut entry_prologue,
                    )?;
                    write!(entry_prologue, " {abi_name} to ").unwrap();
                    self.export_type(arg_ty, &mut entry_prologue)?;
                    writeln!(entry_prologue).unwrap();
                } else {
                    write!(output, " {name}").unwrap();
                }
                next_value_id += 1;
            }
            write!(output, ")").unwrap();
            if let Some(attrs) = llvm_func_attrs.as_deref() {
                self.export_llvm_attributes(attrs, output)?;
            }

            // Mark every emitted device function `convergent` (attr group #0).
            // GPU code is convergent-by-default, as in Clang/nvcc: a function
            // that (transitively) performs a barrier / shuffle / vote must not
            // have those ops sunk or duplicated into divergent control flow by
            // `opt -O2`. Without this, an inlined `grid::sync()` / warp collective
            // gets its `bar.sync.aligned` pushed into a `tid`-dependent branch
            // and deadlocks. opt's FunctionAttrs strips `convergent` from
            // functions it proves never reach a convergent op.
            // alwaysinline (from #[inline(always)]) and !dbg are independent:
            // either, both, or neither can be present. Emit the inline keyword
            // before the convergent attr group #0, then the debug scope.
            if is_alwaysinline {
                write!(output, " alwaysinline").unwrap();
            }
            if let Some(scope_id) = debug_scope {
                writeln!(output, " #0 !dbg !{scope_id} {{").unwrap();
            } else {
                writeln!(output, " #0 {{").unwrap();
            }
            self.convergent_used = true;

            // Assign labels to all blocks
            let mut block_labels = FxHashMap::default();
            let mut next_label_id = 0;
            for (i, block_node) in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
                .enumerate()
            {
                if i == 0 {
                    // Entry block usually doesn't need label in LLVM if it's first
                    block_labels.insert(block_node, "entry".to_string());
                } else {
                    let label = format!("bb{next_label_id}");
                    next_label_id += 1;
                    block_labels.insert(block_node, label);
                }
            }

            // PRE-PASS: Assign names to ALL values before exporting.
            // This is needed because PHI nodes may reference values from blocks that
            // come later in the block list (e.g., back-edges in loops).
            for block_node in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
            {
                // Name block arguments (skip entry block which was already done)
                if block_node != entry_block {
                    for arg in block_node.deref(self.ctx).arguments() {
                        let name = format!("%v{next_value_id}");
                        next_value_id += 1;
                        value_names.insert(arg, name);
                    }
                }

                // Name operation results
                for op in block_node.deref(self.ctx).iter(self.ctx) {
                    let op_ref = op.deref(self.ctx);
                    let op_obj = Operation::get_op_dyn(op, self.ctx);
                    let op_dyn = op_obj.as_ref();

                    // PHIs can reference an undef defined in a block that is
                    // printed later. Register its literal spelling during the
                    // pre-pass just like constants.
                    if let Some(undef_op) = op_dyn.downcast_ref::<ops::UndefOp>() {
                        let result = undef_op.get_operation().deref(self.ctx).get_result(0);
                        value_names.insert(result, "undef".to_string());
                        continue;
                    }

                    // CRITICAL: ConstantOp MUST be registered in pre-pass, not during export!
                    // PHI nodes may reference constants from blocks that appear later in the
                    // iteration order. If we delay constant naming until export, the PHI
                    // export will fail to find the constant in value_names and emit "undef".
                    //
                    // Example: bb6 has PHI receiving constant 0 from bb14, but bb6 is
                    // exported before bb14. Without pre-pass registration, the constant's
                    // Value is not in value_names when bb6's PHI is emitted.
                    if let Some(const_op) = op_dyn.downcast_ref::<ops::ConstantOp>() {
                        let val_attr = const_op.get_value(self.ctx);
                        let val_attr = &*val_attr as &dyn Attribute;

                        let const_str = if let Some(int_attr) =
                            val_attr.downcast_ref::<IntegerAttr>()
                        {
                            int_attr.value().to_string_unsigned_decimal()
                        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
                            format_half_literal(crate::fp16_attr_to_bits(fp16_attr))
                        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
                            let float_val: f32 = fp32_attr.clone().into();
                            format_float_literal(f64::from(float_val))
                        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
                            let float_val: f64 = fp64_attr.clone().into();
                            format_float_literal(float_val)
                        } else if self.legacy_typed_pointers() {
                            return Err(format!(
                                "legacy LLVM 7 export cannot render constant attribute `{}`",
                                val_attr.disp(self.ctx)
                            ));
                        } else {
                            "0".to_string() // Fallback
                        };

                        let res = op_ref.get_result(0);
                        value_names.insert(res, const_str);
                        continue;
                    }

                    // AddressOfOp is also virtual in textual LLVM IR: uses
                    // must print the global symbol directly. Pre-register
                    // the result as `@<global_name>` here so CFG order
                    // cannot expose a stale temporary name when a
                    // later-printed block defines the address used by an
                    // earlier-printed block. The op-emit arm in `export_op`
                    // for AddressOfOp asserts this invariant.
                    if !self.legacy_typed_pointers()
                        && let Some(address_of) = op_dyn.downcast_ref::<ops::AddressOfOp>()
                    {
                        let symbol_name = address_of.get_global_name(self.ctx).to_string();
                        let res = op_ref.get_result(0);
                        let result_type = res.get_type(self.ctx);
                        let result_ref = result_type.deref(self.ctx);
                        let result_pointer =
                            result_ref.downcast_ref::<PointerType>().ok_or_else(|| {
                                format!(
                                    "addressof result is not a pointer: `{}`",
                                    result_ref.disp(self.ctx)
                                )
                            })?;

                        let exported_name = if let Some(info) =
                            self.global_symbols.get(&symbol_name)
                        {
                            if result_pointer.address_space() != info.address_space {
                                return Err(format!(
                                    "addressof `@{symbol_name}` address-space mismatch: result is {}, global is {}",
                                    result_pointer.address_space(),
                                    info.address_space
                                ));
                            }
                            symbol_name
                        } else {
                            let function_name = if symbol_name.starts_with("llvm_") {
                                decode_intrinsic_identifier(&symbol_name)
                            } else {
                                strip_device_prefix(&symbol_name)
                            };
                            if self.function_grid_constants.contains_key(&function_name) {
                                return Err(format!(
                                    "cannot take the device function address of grid-constant kernel entry `@{function_name}`; its launch ABI is not an ordinary function ABI"
                                ));
                            }
                            if !self.function_types.contains_key(&function_name) {
                                return Err(format!(
                                    "addressof references unknown symbol `@{symbol_name}`"
                                ));
                            }
                            if result_pointer.address_space() != 0 {
                                return Err(format!(
                                    "function addressof `@{function_name}` must produce a program-address-space (0) pointer, got address space {}",
                                    result_pointer.address_space()
                                ));
                            }
                            function_name
                        };
                        value_names.insert(res, format!("@{exported_name}"));
                        continue;
                    }

                    for res in op_ref.results() {
                        let name = format!("%v{next_value_id}");
                        next_value_id += 1;
                        value_names.insert(res, name);
                    }
                }
            }

            // Build predecessor map for PHI generation
            let mut pred_map: PredecessorMap = FxHashMap::default();
            let validate_edge = |source: Ptr<BasicBlock>, dest: Ptr<BasicBlock>, args: &[Value]| {
                let expected: Vec<_> = dest.deref(self.ctx).arguments().collect();
                let source_label = block_labels
                    .get(&source)
                    .map(String::as_str)
                    .unwrap_or("<unknown>");
                let dest_label = block_labels
                    .get(&dest)
                    .map(String::as_str)
                    .unwrap_or("<unknown>");
                if args.len() != expected.len() {
                    return Err(format!(
                        "predecessor `%{source_label}` supplies {} values to `%{dest_label}`, which expects {} block arguments",
                        args.len(),
                        expected.len()
                    ));
                }
                for (index, (value, argument)) in args.iter().copied().zip(expected).enumerate() {
                    if value.get_type(self.ctx) != argument.get_type(self.ctx) {
                        return Err(format!(
                            "predecessor `%{source_label}` value {index} has type `{}`, but `%{dest_label}` block argument has type `{}`",
                            value.get_type(self.ctx).disp(self.ctx),
                            argument.get_type(self.ctx).disp(self.ctx)
                        ));
                    }
                }
                Ok(())
            };
            for block in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
            {
                let block_ref = block.deref(self.ctx);
                if let Some(term) = block_ref.iter(self.ctx).last() {
                    let term_obj = Operation::get_op_dyn(term, self.ctx);
                    let term_dyn = term_obj.as_ref();

                    if let Some(branch) = term_dyn.downcast_ref::<ops::BrOp>() {
                        // BrOp has 1 successor and all operands are passed to it
                        let dest = term.deref(self.ctx).successors().next().unwrap();
                        let args = branch.successor_operands(self.ctx, 0);
                        validate_edge(block, dest, &args)?;
                        pred_map.entry(dest).or_default().push((block, args));
                    } else if let Some(branch) = term_dyn.downcast_ref::<ops::CondBrOp>() {
                        let succs: Vec<_> = term.deref(self.ctx).successors().collect();
                        let true_dest = succs[0];
                        let false_dest = succs[1];
                        let true_args = branch.successor_operands(self.ctx, 0);
                        let false_args = branch.successor_operands(self.ctx, 1);
                        validate_edge(block, true_dest, &true_args)?;
                        validate_edge(block, false_dest, &false_args)?;
                        if true_dest == false_dest {
                            if true_args != false_args {
                                return Err(format!(
                                    "conditional branch `%{}` reaches `%{}` on both edges with different forwarded values; LLVM PHIs cannot distinguish duplicate edges from one predecessor",
                                    block_labels.get(&block).unwrap(),
                                    block_labels.get(&true_dest).unwrap()
                                ));
                            }
                            pred_map
                                .entry(true_dest)
                                .or_default()
                                .push((block, true_args));
                        } else {
                            pred_map
                                .entry(true_dest)
                                .or_default()
                                .push((block, true_args));
                            pred_map
                                .entry(false_dest)
                                .or_default()
                                .push((block, false_args));
                        }
                    }
                }
            }

            // Export blocks
            for (i, block_node) in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
                .enumerate()
            {
                self.export_block(
                    block_node,
                    &mut value_names,
                    &mut next_value_id,
                    &block_labels,
                    &pred_map,
                    i == 0,
                    &entry_prologue,
                    debug_scope,
                    output,
                )?;
            }

            writeln!(output, "}}").unwrap();
        } else {
            // get_num_regions() >= 1 but the first region has no entry block (empty function).
            // Treat it as a declaration.
            write!(output, "declare ").unwrap();
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let args = func_ty.arg_types();
            for (i, arg_ty) in args.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_function_parameter_type(&fixed_func_name, i, *arg_ty, output)?;
                let grid_constant = grid_constant_params
                    .iter()
                    .find(|parameter| parameter.index == i);
                if let Some(alignment) = reference_param_validities[i] {
                    write!(output, " nonnull").unwrap();
                    if alignment > 1 && grid_constant.is_none() {
                        write!(output, " align {alignment}").unwrap();
                    }
                }
                if let Some(parameter) = grid_constant {
                    self.export_grid_constant_parameter_attrs(
                        parameter.pointee,
                        parameter.alignment,
                        output,
                    )?;
                }
            }
            writeln!(output, ")").unwrap();
        }

        writeln!(output).unwrap();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn export_block(
        &mut self,
        block: Ptr<BasicBlock>,
        value_names: &mut FxHashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &FxHashMap<Ptr<BasicBlock>, String>,
        pred_map: &PredecessorMap,
        is_entry: bool,
        entry_prologue: &str,
        debug_scope: Option<usize>,
        output: &mut String,
    ) -> Result<(), String> {
        // Always print label to ensure it can be referenced by PHI nodes
        let label = block_labels.get(&block).unwrap();
        writeln!(output, "{label}:").unwrap();
        if is_entry {
            output.push_str(entry_prologue);
        }

        // Generate PHI nodes for block arguments (except entry block which uses function args)
        let args: Vec<_> = block.deref(self.ctx).arguments().collect();
        if !args.is_empty() && !is_entry {
            let preds = pred_map
                .get(&block)
                .ok_or_else(|| "Block with args has no predecessors".to_string())?;

            for (arg_idx, arg) in args.iter().enumerate() {
                // Use pre-assigned name or generate new one
                let arg_name = if let Some(name) = value_names.get(arg) {
                    name.clone()
                } else {
                    let name = format!("%v{next_value_id}");
                    *next_value_id += 1;
                    value_names.insert(*arg, name.clone());
                    name
                };

                write!(output, "  {arg_name} = phi ").unwrap();
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();

                for (i, (pred_block, pred_args)) in preds.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }

                    if arg_idx < pred_args.len() {
                        let val = pred_args[arg_idx];
                        write!(output, "[ ").unwrap();
                        self.export_value(val, value_names, output)?;
                        let label = block_labels.get(pred_block).unwrap();
                        write!(output, ", %{label} ]").unwrap();
                    } else {
                        return Err(format!(
                            "predecessor `%{}` does not supply block argument {arg_idx} for `%{label}`",
                            block_labels.get(pred_block).unwrap()
                        ));
                    }
                }
                writeln!(output).unwrap();
            }
        }

        for op in block.deref(self.ctx).iter(self.ctx) {
            self.export_op(
                op,
                value_names,
                next_value_id,
                block_labels,
                debug_scope,
                output,
            )?;
        }
        Ok(())
    }
}

fn decode_hex_initializer(hex: &str) -> Result<Vec<u8>, String> {
    if !hex.len().is_multiple_of(2) {
        return Err("global initializer hex string has odd length".to_string());
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.as_bytes().as_chunks::<2>().0 {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

fn hex_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!("invalid hex digit {:?}", byte as char)),
    }
}
