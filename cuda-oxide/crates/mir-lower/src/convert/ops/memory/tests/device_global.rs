/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::disallowed_methods)]

use super::*;

pub(super) fn append_global_alloc(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    global_key: &str,
    constant: bool,
) -> Ptr<Operation> {
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    append_global_alloc_typed(ctx, block, global_key, constant, !constant, i32_ty)
}

fn append_global_alloc_typed(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    global_key: &str,
    constant: bool,
    is_mutable: bool,
    global_ty: TypeHandle,
) -> Ptr<Operation> {
    let result_ty = if constant {
        MirPtrType::get_constant(ctx, global_ty, is_mutable)
    } else {
        MirPtrType::get_global(ctx, global_ty, is_mutable)
    };
    let op = Operation::new(
        ctx,
        mir::MirGlobalAllocOp::get_concrete_op_info(),
        vec![result_ty.into()],
        vec![],
        vec![],
        0,
    );
    let alloc = mir::MirGlobalAllocOp::new(op);
    alloc.set_attr_global_type(ctx, TypeAttr::new(global_ty));
    alloc.set_attr_global_key(ctx, StringAttr::new(global_key.to_string()));
    op.insert_at_back(block, ctx);
    op
}

#[test]
fn convert_global_alloc_deduplicates_matching_storage_across_pointer_mutability() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    append_global_alloc_typed(&mut ctx, block, "same-global", false, false, i32_ty);
    append_global_alloc_typed(&mut ctx, block, "same-global", false, true, i32_ty);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("pointer-carrier mutability does not change physical global storage");

    let top = module_top_block(&ctx, module_ptr);
    let globals = top
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .filter(|global| global.source_global_key(&ctx).as_deref() == Some("same-global"))
        .count();
    assert_eq!(globals, 1);
    assert_eq!(
        count_ops::<llvm::AddressOfOp>(&ctx, &kernel_blocks(&ctx, module_ptr)),
        2
    );
}

#[test]
fn convert_global_alloc_rejects_conflicting_key_declarations() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let byte_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
    let word_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let empty_words: TypeHandle = MirArrayType::get(&mut ctx, word_ty, 0).into();

    let byte =
        append_global_alloc_typed(&mut ctx, block, "conflicting-global", false, false, byte_ty);
    let byte = mir::MirGlobalAllocOp::new(byte);
    byte.set_alignment_value(&mut ctx, 1);
    byte.mark_immutable(&mut ctx);

    let words = append_global_alloc_typed(
        &mut ctx,
        block,
        "conflicting-global",
        false,
        false,
        empty_words,
    );
    let words = mir::MirGlobalAllocOp::new(words);
    words.set_alignment_value(&mut ctx, 4);
    words.mark_immutable(&mut ctx);
    append_mir_return(&mut ctx, block, vec![]);

    let error = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("one global_key must not name incompatible type/alignment storage");
    assert!(
        error.to_string().contains("incompatible declaration"),
        "{error}"
    );
}

#[test]
fn device_global_declaration_identity_covers_every_storage_property() {
    let ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
    let base = DeviceGlobalDeclaration {
        mir_type: i32_ty,
        alignment: 4,
        addr_space: llvm_addr::GLOBAL,
        initializer_hex: Some("00000000".to_string()),
        initializer_relocations: Some("reloc-a".to_string()),
        debug_info: None,
        immutable: true,
    };

    let mut changed = base.clone();
    changed.mir_type = i64_ty;
    assert!(base != changed);
    changed = base.clone();
    changed.alignment = 8;
    assert!(base != changed);
    changed = base.clone();
    changed.addr_space = llvm_addr::CONSTANT;
    assert!(base != changed);
    changed = base.clone();
    changed.initializer_hex = Some("01000000".to_string());
    assert!(base != changed);
    changed = base.clone();
    changed.initializer_relocations = Some("reloc-b".to_string());
    assert!(base != changed);
    changed = base.clone();
    changed.debug_info = Some(llvm::DebugGlobalVariableInfo {
        name: "COUNTER".to_string(),
        namespace: vec!["my_crate".to_string()],
        ty: llvm::DebugLocalTypeKind::Basic {
            name: "u32".to_string(),
            size_bits: 32,
            encoding: "DW_ATE_unsigned",
        },
        declaration: llvm::DebugSourcePosition {
            file: PathBuf::from("/tmp/global.rs"),
            line: 7,
            column: 1,
        },
        is_local_to_unit: true,
        is_function_local: false,
    });
    assert!(base != changed);
    changed = base.clone();
    changed.immutable = false;
    assert!(base != changed);
}

#[test]
fn convert_global_alloc_places_in_global_or_constant_addrspace() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_global_alloc(&mut ctx, block, "ordinary_static", false);
    let constant_key = dialect_mir::ops::encode_rust_static_global_key("_ZN7my_mod3KEYE");
    append_global_alloc(&mut ctx, block, &constant_key, true);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let top = module_top_block(&ctx, module_ptr);
    let globals: Vec<_> = top
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .collect();
    let global_addr_global = globals
        .iter()
        .find(|g| g.address_space(&ctx) == llvm_addr::GLOBAL)
        .expect("expected one global in addrspace(1)");
    let global_addr_const = globals
        .iter()
        .find(|g| g.address_space(&ctx) == llvm_addr::CONSTANT)
        .expect("expected one global in addrspace(4)");

    // Constant-memory globals reuse the Rust mangled name so host code can
    // resolve them by name via `cuModuleGetGlobal`; ordinary globals get
    // a counter-suffixed `__device_global_N`.
    assert_eq!(
        global_addr_const.get_symbol_name(&ctx).to_string(),
        "_ZN7my_mod3KEYE",
        "constant globals must keep the raw rustc symbol payload as their name"
    );
    assert!(
        global_addr_global
            .get_symbol_name(&ctx)
            .to_string()
            .starts_with("__device_global_"),
        "ordinary device globals get the __device_global_ prefix"
    );
}

#[test]
fn as1_and_as4_global_debug_info_survive_lowering_with_distinct_semantic_types() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let ordinary = append_global_alloc(&mut ctx, block, "crate::GLOBAL_COUNTER", false);
    let constant_key = dialect_mir::ops::encode_rust_static_global_key("_ZN5crate5COEFFE");
    let constant = append_global_alloc(&mut ctx, block, &constant_key, true);
    let ordinary_info = llvm::DebugGlobalVariableInfo {
        name: "GLOBAL_COUNTER".to_string(),
        namespace: vec!["crate".to_string(), "state".to_string()],
        ty: llvm::DebugLocalTypeKind::Basic {
            name: "u32".to_string(),
            size_bits: 32,
            encoding: "DW_ATE_unsigned",
        },
        declaration: llvm::DebugSourcePosition {
            file: PathBuf::from("/tmp/global.rs"),
            line: 7,
            column: 1,
        },
        is_local_to_unit: true,
        is_function_local: false,
    };
    let constant_info = llvm::DebugGlobalVariableInfo {
        name: "COEFF".to_string(),
        namespace: vec!["crate".to_string(), "kernels".to_string()],
        ty: llvm::DebugLocalTypeKind::Struct {
            name: "ConstantMemory".to_string(),
            size_bits: 32,
            members: vec![llvm::DebugTypeMember {
                name: "0".to_string(),
                offset_bits: 0,
                ty: llvm::DebugLocalTypeKind::Struct {
                    name: "UnsafeCell".to_string(),
                    size_bits: 32,
                    members: vec![llvm::DebugTypeMember {
                        name: "value".to_string(),
                        offset_bits: 0,
                        ty: llvm::DebugLocalTypeKind::Basic {
                            name: "f32".to_string(),
                            size_bits: 32,
                            encoding: "DW_ATE_float",
                        },
                    }],
                },
            }],
        },
        declaration: llvm::DebugSourcePosition {
            file: PathBuf::from("/tmp/global.rs"),
            line: 11,
            column: 1,
        },
        is_local_to_unit: true,
        is_function_local: false,
    };
    llvm::set_debug_global_variable(&mut ctx, ordinary, &ordinary_info);
    llvm::set_debug_global_variable(&mut ctx, constant, &constant_info);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let top = module_top_block(&ctx, module_ptr);
    let globals: Vec<_> = top
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .collect();
    let ordinary = globals
        .iter()
        .find(|global| global.address_space(&ctx) == llvm_addr::GLOBAL)
        .expect("ordinary global");
    let constant = globals
        .iter()
        .find(|global| global.address_space(&ctx) == llvm_addr::CONSTANT)
        .expect("constant global");

    assert_eq!(
        llvm::debug_global_variable(&ctx, ordinary.get_operation()),
        Some(ordinary_info),
        "source identity and semantic type must survive MIR-to-LLVM lowering"
    );
    assert_eq!(
        llvm::debug_global_variable(&ctx, constant.get_operation()),
        Some(constant_info),
        "AS4 must retain the source ConstantMemory/UnsafeCell/f32 graph rather than its physical byte storage"
    );
}

#[test]
fn immutable_marking_survives_lowering_and_is_not_assumed() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

    // Two ordinary addrspace(1) globals, distinguished by their source key.
    // Only the promoted one claims immutability; the plain static must not
    // acquire it, or the exporter would write `constant` for storage the
    // host can still overwrite by symbol.
    let promoted = append_global_alloc(&mut ctx, block, "promoted_table", false);
    mir::MirGlobalAllocOp::new(promoted).mark_immutable(&mut ctx);
    append_global_alloc(&mut ctx, block, "plain_static", false);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let top = module_top_block(&ctx, module_ptr);
    let globals: Vec<llvm::GlobalOp> = top
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .collect();
    let by_key = |key: &str| -> llvm::GlobalOp {
        *globals
            .iter()
            .find(|g| g.source_global_key(&ctx).as_deref() == Some(key))
            .unwrap_or_else(|| panic!("no lowered global carries source key {key}"))
    };

    assert!(
        by_key("promoted_table").is_constant(&ctx),
        "a global marked immutable in MIR must stay immutable through lowering"
    );
    assert!(
        !by_key("plain_static").is_constant(&ctx),
        "lowering must not infer immutability; only the promoted-constant \
             sites may claim it"
    );
}

#[test]
fn convert_global_alloc_rejects_conflicting_immutability() {
    // A plain static and a promoted constant that end up sharing one key
    // differ only in the immutable flag; that alone must fail closed
    // instead of silently reusing the first allocation's storage.
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_global_alloc(&mut ctx, block, "collision", false);
    let promoted = append_global_alloc(&mut ctx, block, "collision", false);
    mir::MirGlobalAllocOp::new(promoted).mark_immutable(&mut ctx);
    append_mir_return(&mut ctx, block, vec![]);

    let error = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("a shared string with different origins must fail closed");
    assert!(
        error.to_string().contains("incompatible declaration"),
        "{error}"
    );
}

#[test]
fn initialized_global_lowers_to_byte_storage() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let op = append_global_alloc(&mut ctx, block, "nan_payload", false);
    let alloc = mir::MirGlobalAllocOp::new(op);
    alloc.set_alignment_value(&mut ctx, 4);
    let initializer_key: Identifier = "global_initializer_hex".try_into().unwrap();
    op.deref_mut(&ctx)
        .attributes
        .set(initializer_key, StringAttr::new("3412c07f".to_string()));
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let top = module_top_block(&ctx, module_ptr);
    let global = top
        .deref(&ctx)
        .iter(&ctx)
        .find_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .expect("expected lowered device global");
    let global_ty = global.get_type(&ctx);
    let global_ty_ref = global_ty.deref(&ctx);
    let array_ty = global_ty_ref
        .downcast_ref::<ArrayType>()
        .expect("initialized global must use byte-array storage");
    assert_eq!(array_ty.size(), 4);
    let elem_ty = array_ty.elem_type();
    let elem_ty_ref = elem_ty.deref(&ctx);
    let elem = elem_ty_ref
        .downcast_ref::<IntegerType>()
        .expect("byte-array element must be an integer");
    assert_eq!(elem.width(), 8);
    assert_eq!(global.get_alignment(&ctx), Some(4));
    assert_eq!(global.initializer_hex(&ctx).as_deref(), Some("3412c07f"));
}

#[test]
fn relocated_global_lowers_to_segmented_storage() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let word_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let mir_global_ty: TypeHandle = MirArrayType::get(&mut ctx, word_ty, 3).into();
    let result_ty = MirPtrType::get_global(&mut ctx, mir_global_ty, false);
    let op = Operation::new(
        &mut ctx,
        mir::MirGlobalAllocOp::get_concrete_op_info(),
        vec![result_ty.into()],
        vec![],
        vec![],
        0,
    );
    let alloc = mir::MirGlobalAllocOp::new(op);
    alloc.set_attr_global_type(&ctx, TypeAttr::new(mir_global_ty));
    alloc.set_attr_global_key(&ctx, StringAttr::new("reference_table".to_string()));
    alloc.set_alignment_value(&mut ctx, 8);
    op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new("000000000000000000000000000000000000000000000000".to_string()),
    );
    let encoded =
        llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
            source_offset: 8,
            width_bytes: 8,
            target_address_space: llvm_addr::GLOBAL,
            target_addend: 4,
            target_key: "target_static".to_string(),
        }]);
    op.deref_mut(&ctx).attributes.set(
        "global_initializer_relocations".try_into().unwrap(),
        StringAttr::new(encoded.clone()),
    );
    op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let top = module_top_block(&ctx, module_ptr);
    let global = top
        .deref(&ctx)
        .iter(&ctx)
        .find_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .expect("expected lowered device global");
    let global_ty = global.get_type(&ctx);
    let global_ty_ref = global_ty.deref(&ctx);
    let storage = global_ty_ref
        .downcast_ref::<StructType>()
        .expect("relocated initializer must use segmented struct storage");
    assert_eq!(storage.num_fields(), 3);
    assert_eq!(
        global.source_global_key(&ctx).as_deref(),
        Some("reference_table")
    );
    assert_eq!(
        global.initializer_relocations(&ctx).as_deref(),
        Some(encoded.as_str())
    );
}

#[test]
fn relocated_global_uses_packed_storage_for_unaligned_pointer_slot() {
    let mut ctx = make_ctx();
    let encoded =
        llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
            source_offset: 1,
            width_bytes: 8,
            target_address_space: llvm_addr::GLOBAL,
            target_addend: 0,
            target_key: "target_static".to_string(),
        }]);

    let storage = relocated_initializer_storage_type(&mut ctx, 9, 1, &encoded)
        .expect("unaligned relocation should use packed storage");
    let storage_ref = storage.deref(&ctx);
    let struct_ty = storage_ref
        .downcast_ref::<StructType>()
        .expect("relocated initializer must use struct storage");
    assert_eq!(struct_ty.layout(), StructLayout::Packed);
    assert_eq!(struct_ty.num_fields(), 2);
    assert_eq!(
        crate::convert::types::llvm_type_size_align(&ctx, storage),
        Some((9, 1))
    );

    let fields: Vec<_> = struct_ty.fields().collect();
    let literal_ref = fields[0].deref(&ctx);
    let literal = literal_ref
        .downcast_ref::<ArrayType>()
        .expect("leading literal span must be a byte array");
    assert_eq!(literal.size(), 1);
    let pointer_ref = fields[1].deref(&ctx);
    let pointer = pointer_ref
        .downcast_ref::<IntegerType>()
        .expect("relocation slot must be an integer carrier");
    assert_eq!(pointer.width(), 64);
}

#[test]
fn relocated_global_uses_packed_storage_for_underaligned_allocation() {
    let mut ctx = make_ctx();
    let encoded =
        llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
            source_offset: 0,
            width_bytes: 8,
            target_address_space: llvm_addr::GLOBAL,
            target_addend: 0,
            target_key: "target_static".to_string(),
        }]);

    let storage = relocated_initializer_storage_type(&mut ctx, 8, 1, &encoded)
        .expect("underaligned allocation should use packed storage");
    let storage_ref = storage.deref(&ctx);
    let struct_ty = storage_ref
        .downcast_ref::<StructType>()
        .expect("relocated initializer must use struct storage");
    assert_eq!(struct_ty.layout(), StructLayout::Packed);
    assert_eq!(
        crate::convert::types::llvm_type_size_align(&ctx, storage),
        Some((8, 1))
    );
}

#[test]
fn relocated_global_keeps_naturally_aligned_storage_unpacked() {
    let mut ctx = make_ctx();
    let encoded =
        llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
            source_offset: 8,
            width_bytes: 8,
            target_address_space: llvm_addr::GLOBAL,
            target_addend: 0,
            target_key: "target_static".to_string(),
        }]);

    let storage = relocated_initializer_storage_type(&mut ctx, 16, 8, &encoded)
        .expect("aligned relocation should keep ordinary storage");
    let storage_ref = storage.deref(&ctx);
    let struct_ty = storage_ref
        .downcast_ref::<StructType>()
        .expect("relocated initializer must use struct storage");
    assert_eq!(struct_ty.layout(), StructLayout::Unpacked);
}

#[test]
fn relocated_global_rejects_overlapping_slots() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let word_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let mir_global_ty: TypeHandle = MirArrayType::get(&mut ctx, word_ty, 2).into();
    let result_ty = MirPtrType::get_global(&mut ctx, mir_global_ty, false);
    let op = Operation::new(
        &mut ctx,
        mir::MirGlobalAllocOp::get_concrete_op_info(),
        vec![result_ty.into()],
        vec![],
        vec![],
        0,
    );
    let alloc = mir::MirGlobalAllocOp::new(op);
    alloc.set_attr_global_type(&ctx, TypeAttr::new(mir_global_ty));
    alloc.set_attr_global_key(&ctx, StringAttr::new("overlap".to_string()));
    alloc.set_alignment_value(&mut ctx, 8);
    op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new("00000000000000000000000000000000".to_string()),
    );
    let encoded = llvm::encode_global_initializer_relocations(&[
        llvm::GlobalInitializerRelocation {
            source_offset: 0,
            width_bytes: 8,
            target_address_space: llvm_addr::GLOBAL,
            target_addend: 0,
            target_key: "a".to_string(),
        },
        llvm::GlobalInitializerRelocation {
            source_offset: 0,
            width_bytes: 8,
            target_address_space: llvm_addr::GLOBAL,
            target_addend: 0,
            target_key: "b".to_string(),
        },
    ]);
    op.deref_mut(&ctx).attributes.set(
        "global_initializer_relocations".try_into().unwrap(),
        StringAttr::new(encoded),
    );
    op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    let error = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("overlapping relocations must fail closed");
    assert!(error.to_string().contains("overlaps"), "{error}");
}
