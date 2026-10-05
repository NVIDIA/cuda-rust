/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conversion of `mir.global_alloc` to device globals, including relocated initializers.

use super::common::{anyhow_to_pliron, counter_named_global};
use crate::context::{DeviceGlobalDeclaration, DeviceGlobalRecord, DeviceGlobalsMap};
use crate::convert::types::{
    convert_type, llvm_type_size_align, validate_initialized_global_layout,
    validate_relocated_initialized_global_layout,
};
use crate::helpers;
use llvm_export::ops as llvm;
use llvm_export::ops::GlobalOpExt;
use llvm_export::types::{ArrayType, StructLayout, StructType};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};

/// Convert `mir.global_alloc` to an LLVM global in CUDA global memory.
///
/// Ordinary Rust `static` / `static mut` values have grid scope and
/// application lifetime, so they live in address space 1 rather than the
/// per-block shared-memory address space.
pub fn convert_global_alloc_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    device_globals: &mut DeviceGlobalsMap,
    next_device_global_index: &mut usize,
) -> Result<()> {
    use pliron::builtin::attributes::{StringAttr, TypeAttr};

    let (
        global_key,
        mir_global_type,
        alignment,
        addr_space,
        initializer_hex,
        initializer_relocations,
        immutable,
        debug_info,
    ) = {
        let global_op = dialect_mir::ops::MirGlobalAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let global_key_attr = op_ref
            .attributes
            .get::<StringAttr>(&"global_key".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_key StringAttr attribute"
                ))
            })?;
        let global_key = String::from((*global_key_attr).clone());

        let global_type_attr = op_ref
            .attributes
            .get::<TypeAttr>(&"global_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_type TypeAttr attribute"
                ))
            })?;
        let mir_global_type = global_type_attr.get_type(ctx);

        let alignment = global_op.get_alignment_value(ctx).unwrap_or(0);
        let initializer_hex = op_ref
            .attributes
            .get::<StringAttr>(&"global_initializer_hex".try_into().unwrap())
            .map(|attr| String::from((*attr).clone()));
        let initializer_relocations = op_ref
            .attributes
            .get::<StringAttr>(&"global_initializer_relocations".try_into().unwrap())
            .map(|attr| String::from((*attr).clone()));
        let debug_info = llvm::debug_global_variable(ctx, op);

        // Read the address space the op's result already carries — set by
        // mir-importer based on the static's type (`ConstantMemory<T>` → 4,
        // ordinary → 1). The dialect verifier accepts both.
        let res_ty = op_ref.get_result(0).get_type(ctx);
        let addr_space = res_ty
            .deref(ctx)
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .map(|p| {
                if p.address_space == dialect_mir::types::address_space::CONSTANT {
                    llvm_export::types::address_space::CONSTANT
                } else {
                    llvm_export::types::address_space::GLOBAL
                }
            })
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp result is not a MirPtrType"
                ))
            })?;

        (
            global_key,
            mir_global_type,
            alignment,
            addr_space,
            initializer_hex,
            initializer_relocations,
            global_op.is_immutable(ctx),
            debug_info,
        )
    };

    let declaration = DeviceGlobalDeclaration {
        mir_type: mir_global_type,
        alignment,
        addr_space,
        initializer_hex: initializer_hex.clone(),
        initializer_relocations: initializer_relocations.clone(),
        debug_info: debug_info.clone(),
        immutable,
    };

    let global_name = if let Some(existing) = device_globals.get(&global_key) {
        if existing.declaration != declaration {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "duplicate global_key {:?} has an incompatible declaration",
                global_key
            )));
        }
        existing.symbol.clone()
    } else {
        create_device_global(
            ctx,
            op,
            device_globals,
            next_device_global_index,
            DeviceGlobalSpec {
                key: &global_key,
                mir_type: mir_global_type,
                alignment,
                addr_space,
                initializer_hex: initializer_hex.as_deref(),
                initializer_relocations: initializer_relocations.as_deref(),
                debug_info: debug_info.as_ref(),
                immutable,
            },
        )?
    };

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, addr_space);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
}

struct DeviceGlobalSpec<'a> {
    key: &'a str,
    mir_type: TypeHandle,
    alignment: u64,
    addr_space: u32,
    initializer_hex: Option<&'a str>,
    initializer_relocations: Option<&'a str>,
    debug_info: Option<&'a llvm::DebugGlobalVariableInfo>,
    /// Nothing writes this storage, so it exports as LLVM `constant`. Set only
    /// for the compiler's own promoted constants; see `MirGlobalAllocOp`.
    immutable: bool,
}

/// `next_device_global_index` is scoped to one `MirToLlvmConversionDriver`
/// instance (one module), not a process-global counter: `N` is a function of
/// this module's own MIR walk order, not of how many other modules have
/// lowered a device global earlier in the process (#706).
fn create_device_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    device_globals: &mut DeviceGlobalsMap,
    next_device_global_index: &mut usize,
    spec: DeviceGlobalSpec<'_>,
) -> Result<pliron::identifier::Identifier> {
    // An explicit initializer is already the evaluated Rust allocation image.
    // Pointer-free data stays `[N x i8]`. Initializers with relocations use a
    // segmented LLVM struct whose literal spans remain byte arrays and whose
    // pointer slots become pointer-width integers. This preserves both exact
    // bytes and linker-visible pointer provenance.
    let semantic_llvm_type = convert_type(ctx, spec.mir_type).map_err(anyhow_to_pliron)?;
    let (llvm_global_type, alignment) = if let Some(initializer_hex) = spec.initializer_hex {
        let byte_count = initializer_hex_byte_count(initializer_hex).map_err(anyhow_to_pliron)?;
        if spec.alignment == 0 {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "device global initializer is missing its evaluated Rust allocation alignment"
            )));
        }
        let storage_type = if let Some(encoded) = spec.initializer_relocations {
            validate_relocated_initialized_global_layout(
                ctx,
                spec.mir_type,
                byte_count,
                spec.alignment,
            )
            .map_err(anyhow_to_pliron)?;
            relocated_initializer_storage_type(ctx, byte_count, spec.alignment, encoded)
                .map_err(anyhow_to_pliron)?
        } else {
            validate_initialized_global_layout(ctx, spec.mir_type, byte_count, spec.alignment)
                .map_err(anyhow_to_pliron)?;
            let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
            ArrayType::get(ctx, i8_ty.into(), byte_count).into()
        };
        (storage_type, spec.alignment)
    } else {
        if spec.initializer_relocations.is_some() {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "device global carries relocation metadata without initializer bytes"
            )));
        }
        (semantic_llvm_type, spec.alignment)
    };

    // Constant-memory globals reuse the Rust-side mangled name so host code can
    // resolve them by name via `cuModuleGetGlobal`. Ordinary device globals
    // are private to the kernel and get a counter-based unique name.
    let name: pliron::identifier::Identifier =
        if spec.addr_space == llvm_export::types::address_space::CONSTANT {
            let symbol = dialect_mir::ops::rust_static_symbol_from_global_key(spec.key)
                .ok_or_else(|| {
                    anyhow_to_pliron(anyhow::anyhow!(
                        "constant global_key {:?} is not a tagged Rust static identity",
                        spec.key
                    ))
                })?;
            symbol.try_into().map_err(|e| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "constant Rust static symbol {:?} is not a valid identifier: {e:?}",
                    symbol
                ))
            })?
        } else {
            let counter = *next_device_global_index;
            *next_device_global_index += 1;
            counter_named_global(ctx, "__device_global", counter)
                .try_into()
                .unwrap()
        };

    let global_op = if alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), llvm_global_type, alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), llvm_global_type)
    };
    global_op.set_address_space(ctx, spec.addr_space);
    global_op.set_source_global_key(ctx, spec.key);
    if matches!(
        spec.addr_space,
        llvm_export::types::address_space::GLOBAL | llvm_export::types::address_space::CONSTANT
    ) && let Some(info) = spec.debug_info
    {
        llvm::set_debug_global_variable(ctx, global_op.get_operation(), info);
    }
    if let Some(initializer_hex) = spec.initializer_hex {
        global_op.set_initializer_hex(ctx, initializer_hex);
    }
    if let Some(initializer_relocations) = spec.initializer_relocations {
        global_op.set_initializer_relocations(ctx, initializer_relocations);
    }
    if spec.immutable {
        global_op.set_constant(ctx, true);
    }

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);
    device_globals.insert(
        spec.key.to_string(),
        DeviceGlobalRecord {
            symbol: name.clone(),
            declaration: DeviceGlobalDeclaration {
                mir_type: spec.mir_type,
                alignment: spec.alignment,
                addr_space: spec.addr_space,
                initializer_hex: spec.initializer_hex.map(str::to_owned),
                initializer_relocations: spec.initializer_relocations.map(str::to_owned),
                debug_info: spec.debug_info.cloned(),
                immutable: spec.immutable,
            },
        },
    );

    Ok(name)
}

pub(super) fn relocated_initializer_storage_type(
    ctx: &mut Context,
    byte_count: u64,
    allocation_alignment: u64,
    encoded: &str,
) -> std::result::Result<TypeHandle, anyhow::Error> {
    let mut relocations =
        llvm::decode_global_initializer_relocations(encoded).map_err(anyhow::Error::msg)?;
    if relocations.is_empty() {
        anyhow::bail!("device global relocation metadata contains no relocations");
    }
    relocations.sort_by_key(|relocation| relocation.source_offset);

    let mut cursor = 0u64;
    let mut fields = Vec::with_capacity(relocations.len() * 2 + 1);
    let mut requires_packed_storage = false;
    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);

    for (index, relocation) in relocations.iter().enumerate() {
        if relocation.width_bytes != 8 {
            anyhow::bail!(
                "device global relocation {index} uses unsupported {}-byte pointer storage; CUDA global/constant pointers require 8 bytes",
                relocation.width_bytes
            );
        }
        if !matches!(relocation.target_address_space, 1 | 4) {
            anyhow::bail!(
                "device global relocation {index} targets unsupported CUDA address space {}",
                relocation.target_address_space
            );
        }
        if relocation.target_key.is_empty() {
            anyhow::bail!("device global relocation {index} has an empty target key");
        }

        let width = u64::from(relocation.width_bytes);
        requires_packed_storage |=
            allocation_alignment < width || !relocation.source_offset.is_multiple_of(width);
        if relocation.source_offset < cursor {
            anyhow::bail!(
                "device global relocation {index} overlaps the previous relocation or literal span"
            );
        }
        let end = relocation
            .source_offset
            .checked_add(width)
            .ok_or_else(|| anyhow::anyhow!("device global relocation {index} offset overflows"))?;
        if end > byte_count {
            anyhow::bail!(
                "device global relocation {index} occupies bytes {}..{} but the initializer is only {} bytes",
                relocation.source_offset,
                end,
                byte_count
            );
        }

        if relocation.source_offset > cursor {
            fields
                .push(ArrayType::get(ctx, i8_ty.into(), relocation.source_offset - cursor).into());
        }
        fields.push(IntegerType::get(ctx, relocation.width_bytes * 8, Signedness::Signless).into());
        cursor = end;
    }

    if cursor < byte_count {
        fields.push(ArrayType::get(ctx, i8_ty.into(), byte_count - cursor).into());
    }

    let layout = if requires_packed_storage {
        StructLayout::Packed
    } else {
        StructLayout::Unpacked
    };
    let storage: TypeHandle = StructType::get_unnamed(ctx, (fields, layout)).into();
    // Exact or error, never guessed: the storage type is built from i8 arrays
    // and integer slots, so its natural size is always computable, and it must
    // land exactly on rustc's byte count or the relocation offsets are wrong.
    let Some((lowered_size, _)) = llvm_type_size_align(ctx, storage) else {
        anyhow::bail!(
            "relocated device global storage `{}` has no exact size",
            storage.deref(ctx).disp(ctx)
        );
    };
    if lowered_size != byte_count {
        anyhow::bail!(
            "relocated device global storage lowers to {} bytes, but rustc evaluated {} bytes",
            lowered_size,
            byte_count
        );
    }
    Ok(storage)
}

fn initializer_hex_byte_count(hex: &str) -> std::result::Result<u64, anyhow::Error> {
    if !hex.len().is_multiple_of(2) {
        anyhow::bail!("device global initializer has an odd-length hex byte string");
    }
    if let Some(invalid) = hex.bytes().find(|byte| !byte.is_ascii_hexdigit()) {
        anyhow::bail!(
            "device global initializer contains invalid hex digit {:?}",
            invalid as char
        );
    }
    u64::try_from(hex.len() / 2)
        .map_err(|_| anyhow::anyhow!("device global initializer is too large for LLVM"))
}
