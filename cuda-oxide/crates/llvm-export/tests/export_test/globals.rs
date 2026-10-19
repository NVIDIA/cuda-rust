/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        NvvmExportConfig, NvvmIrDialect, export_module_to_string,
        export_module_to_string_with_config,
    },
    ops::{
        GlobalInitializerRelocation, GlobalOp, GlobalOpExt, encode_global_initializer_relocations,
    },
    types::{ArrayType, StructLayout, StructType},
};
use pliron::{
    builtin::{
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    op::Op,
};

use crate::common::module_top_block;

/// Export a module holding one shared global, optionally labelled with the
/// Rust path of the `static` it came from.
fn export_shared_global_with_source_name(source_name: Option<&str>) -> String {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let array_ty = ArrayType::get(&ctx, i32_ty.to_handle(), 64);
    let global = GlobalOp::new(
        &mut ctx,
        "__shared_mem_7".try_into().unwrap(),
        array_ty.to_handle(),
    );
    global.set_address_space(&mut ctx, 3);
    if let Some(source_name) = source_name {
        global.set_shared_source_name(&mut ctx, source_name);
    }
    global.get_operation().insert_at_back(module_block, &ctx);

    export_module_to_string(&ctx, &module).expect("export succeeds")
}

#[test]
fn shared_global_source_name_is_exported_as_a_comment_above_the_definition() {
    let ir = export_shared_global_with_source_name(Some("my_kernel::TILE"));

    let definition_index = ir
        .find("@__shared_mem_7 = addrspace(3) global")
        .expect("module must declare the shared global");
    let comment_index = ir
        .find("; shared source: my_kernel::TILE")
        .unwrap_or_else(|| panic!("shared global must name its Rust source:\n{ir}"));
    assert!(
        comment_index < definition_index,
        "the source comment must precede the definition it describes:\n{ir}"
    );
}

#[test]
fn shared_global_without_a_source_name_exports_no_comment() {
    let ir = export_shared_global_with_source_name(None);

    assert!(
        ir.contains("@__shared_mem_7 = addrspace(3) global"),
        "module must declare the shared global:\n{ir}"
    );
    assert!(
        !ir.contains("; shared source:"),
        "an unlabelled global must not gain a comment:\n{ir}"
    );
}

#[test]
fn shared_global_source_name_cannot_escape_its_comment_line() {
    // A newline in the label would end the comment and leave the remainder to
    // be parsed as IR. Nothing in the current pipeline produces such a name,
    // so this pins the exporter's own guarantee rather than a live bug.
    let ir = export_shared_global_with_source_name(Some("EVIL\n@injected = addrspace(3) global"));

    assert!(
        ir.lines().all(|line| !line.starts_with("@injected")),
        "a control character in the label must not open a new IR line:\n{ir}"
    );
    assert!(
        ir.contains("; shared source: EVIL @injected = addrspace(3) global"),
        "the label must survive on one line with controls flattened:\n{ir}"
    );
}

#[test]
fn nvvm_export_rejects_invalid_global_address_spaces() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(
        &mut ctx,
        "thread_local_global".try_into().unwrap(),
        i32_ty.to_handle(),
    );
    global.set_address_space(&mut ctx, 5);
    global.get_operation().insert_at_back(module_block, &ctx);

    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("NVVM module-scope local-memory global must be rejected");
    assert!(error.contains("unsupported address space 5"), "{error}");

    // The ordinary LLVM/PTX exporter retains its prior behavior; this
    // restriction is specifically part of the NVVM IR contract.
    assert!(export_module_to_string(&ctx, &module).is_ok());
}

#[test]
fn initialized_globals_export_exact_bytes() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "exact_global_bytes".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);

    // 0x7fc01234 is a quiet f32 NaN with a non-canonical payload. Treating
    // these bytes as an f32 before printing would collapse it to 0x7fc00000.
    let nan_ty = ArrayType::get(&ctx, i8_ty.into(), 4);
    let nan = GlobalOp::new_with_alignment(
        &mut ctx,
        "nan_payload".try_into().unwrap(),
        nan_ty.into(),
        4,
    );
    nan.set_address_space(&mut ctx, 1);
    nan.set_initializer_hex(&mut ctx, "3412c07f");
    nan.get_operation().insert_at_back(module_block, &ctx);

    // Byte 0 is a u8, bytes 1..4 are zeroed repr(C) padding, and bytes 4..8
    // are a little-endian u32. The exporter must not recompute those offsets.
    let padded_ty = ArrayType::get(&ctx, i8_ty.into(), 8);
    let padded = GlobalOp::new_with_alignment(
        &mut ctx,
        "padded_struct".try_into().unwrap(),
        padded_ty.into(),
        4,
    );
    padded.set_address_space(&mut ctx, 1);
    padded.set_initializer_hex(&mut ctx, "ab00000078563412");
    padded.get_operation().insert_at_back(module_block, &ctx);

    for config in [
        NvvmExportConfig::new(NvvmIrDialect::Modern),
        NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    ] {
        let ir = export_module_to_string_with_config(&ctx, &module, &config)
            .expect("byte-exact global export succeeds");
        assert!(
            ir.contains(r#"@nan_payload = addrspace(1) global [4 x i8] c"\34\12\C0\7F", align 4"#),
            "NaN payload bytes changed:\n{ir}"
        );
        assert!(
            ir.contains(
                r#"@padded_struct = addrspace(1) global [8 x i8] c"\AB\00\00\00\78\56\34\12", align 4"#
            ),
            "repr(C) layout bytes changed:\n{ir}"
        );
    }
}

#[test]
fn immutable_globals_export_the_constant_keyword() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "constant_keyword".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let table_ty = ArrayType::get(&ctx, i8_ty.into(), 4);

    // The compiler's own promoted table: marked never-written, so it must
    // export as `constant`. That keyword is the whole point of the property:
    // it is what lets `opt` treat reads as invariant (deleting a copy into a
    // stack slot) and what makes `llc` select `ld.global.nc`.
    let promoted = GlobalOp::new_with_alignment(
        &mut ctx,
        "promoted_table".try_into().unwrap(),
        table_ty.into(),
        4,
    );
    promoted.set_address_space(&mut ctx, 1);
    promoted.set_initializer_hex(&mut ctx, "01020304");
    promoted.set_constant(&mut ctx, true);
    promoted.get_operation().insert_at_back(module_block, &ctx);

    // An identically shaped global without the constant property: the host may
    // still write such storage by symbol, so it must keep `global`. The
    // constant property is opt-in per global, never inferred from the shape of
    // the initializer.
    let plain = GlobalOp::new_with_alignment(
        &mut ctx,
        "plain_static".try_into().unwrap(),
        table_ty.into(),
        4,
    );
    plain.set_address_space(&mut ctx, 1);
    plain.set_initializer_hex(&mut ctx, "01020304");
    plain.get_operation().insert_at_back(module_block, &ctx);

    // The absent property and a cleared true property both denote mutable
    // storage. The latter catches an incorrect presence-only interpretation.
    for clear_constant in [false, true] {
        if clear_constant {
            plain.set_constant(&mut ctx, true);
            plain.set_constant(&mut ctx, false);
        }
        for config in [
            NvvmExportConfig::new(NvvmIrDialect::Modern),
            NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        ] {
            let ir = export_module_to_string_with_config(&ctx, &module, &config)
                .expect("immutable global export succeeds");
            assert!(
                ir.contains(
                    r#"@promoted_table = addrspace(1) constant [4 x i8] c"\01\02\03\04", align 4"#
                ),
                "promoted global lost the constant keyword:\n{ir}"
            );
            assert!(
                ir.contains(
                    r#"@plain_static = addrspace(1) global [4 x i8] c"\01\02\03\04", align 4"#
                ),
                "unmarked global must not become constant:\n{ir}"
            );
        }
    }
}

#[test]
fn initialized_global_exports_static_pointer_relocation() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "static_relocation".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    // Insert the reference first. Module symbol indexing must make relocation
    // resolution independent of textual global order.
    let reference_ty = StructType::get_unnamed(&ctx, (vec![i64_ty.into()], StructLayout::Unpacked));
    let reference = GlobalOp::new_with_alignment(
        &mut ctx,
        "reference".try_into().unwrap(),
        reference_ty.into(),
        8,
    );
    reference.set_address_space(&mut ctx, 1);
    reference.set_source_global_key(&mut ctx, "REFERENCE");
    reference.set_initializer_hex(&mut ctx, "0000000000000000");
    let encoded = encode_global_initializer_relocations(&[GlobalInitializerRelocation {
        source_offset: 0,
        width_bytes: 8,
        target_address_space: 1,
        target_addend: 0,
        target_key: "TARGET".to_string(),
    }]);
    reference.set_initializer_relocations(&mut ctx, &encoded);
    reference.get_operation().insert_at_back(module_block, &ctx);

    let target_ty = ArrayType::get(&ctx, i8_ty.into(), 4);
    let target =
        GlobalOp::new_with_alignment(&mut ctx, "target".try_into().unwrap(), target_ty.into(), 4);
    target.set_address_space(&mut ctx, 1);
    target.set_source_global_key(&mut ctx, "TARGET");
    target.set_initializer_hex(&mut ctx, "78563412");
    target.get_operation().insert_at_back(module_block, &ctx);

    let modern = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern relocated initializer export succeeds");
    assert!(
        modern.contains(
            "@reference = addrspace(1) global { i64 } { i64 ptrtoint (ptr addrspacecast (ptr addrspace(1) @target to ptr) to i64) }, align 8"
        ),
        "{modern}"
    );

    let legacy = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy relocated initializer export succeeds");
    assert!(
        legacy.contains(
            "@reference = addrspace(1) global { i64 } { i64 ptrtoint (i8* addrspacecast (i8 addrspace(1)* bitcast ([4 x i8] addrspace(1)* @target to i8 addrspace(1)*) to i8*) to i64) }, align 8"
        ),
        "{legacy}"
    );
}

#[test]
fn initialized_global_exports_unaligned_pointer_relocation_as_packed_storage() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "packed_static_relocation".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    let literal_ty = ArrayType::get(&ctx, i8_ty.into(), 1);
    let reference_ty = StructType::get_unnamed(
        &ctx,
        (vec![literal_ty.into(), i64_ty.into()], StructLayout::Packed),
    );
    let reference = GlobalOp::new_with_alignment(
        &mut ctx,
        "packed_reference".try_into().unwrap(),
        reference_ty.into(),
        1,
    );
    reference.set_address_space(&mut ctx, 1);
    reference.set_source_global_key(&mut ctx, "PACKED_REFERENCE");
    reference.set_initializer_hex(&mut ctx, "7b0000000000000000");
    let encoded = encode_global_initializer_relocations(&[GlobalInitializerRelocation {
        source_offset: 1,
        width_bytes: 8,
        target_address_space: 1,
        target_addend: 0,
        target_key: "TARGET".to_string(),
    }]);
    reference.set_initializer_relocations(&mut ctx, &encoded);
    reference.get_operation().insert_at_back(module_block, &ctx);

    let target_ty = ArrayType::get(&ctx, i8_ty.into(), 4);
    let target =
        GlobalOp::new_with_alignment(&mut ctx, "target".try_into().unwrap(), target_ty.into(), 4);
    target.set_address_space(&mut ctx, 1);
    target.set_source_global_key(&mut ctx, "TARGET");
    target.set_initializer_hex(&mut ctx, "78563412");
    target.get_operation().insert_at_back(module_block, &ctx);

    let modern = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern packed relocated initializer export succeeds");
    assert!(
        modern.contains(
            r#"@packed_reference = addrspace(1) global <{ [1 x i8], i64 }> <{ [1 x i8] c"\7B", i64 ptrtoint (ptr addrspacecast (ptr addrspace(1) @target to ptr) to i64) }>, align 1"#
        ),
        "{modern}"
    );

    let legacy = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy packed relocated initializer export succeeds");
    assert!(
        legacy.contains(
            r#"@packed_reference = addrspace(1) global <{ [1 x i8], i64 }> <{ [1 x i8] c"\7B", i64 ptrtoint (i8* addrspacecast (i8 addrspace(1)* bitcast ([4 x i8] addrspace(1)* @target to i8 addrspace(1)*) to i8*) to i64) }>, align 1"#
        ),
        "{legacy}"
    );
}

#[test]
fn initialized_global_exports_multiple_relocations_and_addends() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "multiple_static_relocations".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    let target_a_ty = ArrayType::get(&ctx, i8_ty.into(), 16);
    let target_a = GlobalOp::new_with_alignment(
        &mut ctx,
        "target_a".try_into().unwrap(),
        target_a_ty.into(),
        8,
    );
    target_a.set_address_space(&mut ctx, 1);
    target_a.set_source_global_key(&mut ctx, "TARGET_A");
    target_a.set_initializer_hex(&mut ctx, "000102030405060708090a0b0c0d0e0f");
    target_a.get_operation().insert_at_back(module_block, &ctx);

    let target_b_ty = ArrayType::get(&ctx, i8_ty.into(), 8);
    let target_b = GlobalOp::new_with_alignment(
        &mut ctx,
        "target_b".try_into().unwrap(),
        target_b_ty.into(),
        8,
    );
    target_b.set_address_space(&mut ctx, 4);
    target_b.set_source_global_key(&mut ctx, "TARGET_B");
    target_b.set_initializer_hex(&mut ctx, "1011121314151617");
    target_b.get_operation().insert_at_back(module_block, &ctx);

    let table_ty = StructType::get_unnamed(
        &ctx,
        (vec![i64_ty.into(), i64_ty.into()], StructLayout::Unpacked),
    );
    let table = GlobalOp::new_with_alignment(
        &mut ctx,
        "reference_table".try_into().unwrap(),
        table_ty.into(),
        8,
    );
    table.set_address_space(&mut ctx, 1);
    table.set_source_global_key(&mut ctx, "REFERENCE_TABLE");
    table.set_initializer_hex(&mut ctx, "00000000000000000000000000000000");
    let encoded = encode_global_initializer_relocations(&[
        GlobalInitializerRelocation {
            source_offset: 0,
            width_bytes: 8,
            target_address_space: 1,
            target_addend: 4,
            target_key: "TARGET_A".to_string(),
        },
        GlobalInitializerRelocation {
            source_offset: 8,
            width_bytes: 8,
            target_address_space: 4,
            target_addend: 0,
            target_key: "TARGET_B".to_string(),
        },
    ]);
    table.set_initializer_relocations(&mut ctx, &encoded);
    table.get_operation().insert_at_back(module_block, &ctx);

    for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
        let ir =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .expect("relocated initializer export succeeds");
        assert!(
            ir.contains("@reference_table = addrspace(1) global { i64, i64 }"),
            "{ir}"
        );
        assert!(ir.contains("getelementptr (i8"), "{ir}");
        assert!(ir.contains("@target_a"), "{ir}");
        assert!(ir.contains("@target_b"), "{ir}");
        assert!(!ir.contains("inttoptr"), "{ir}");
    }
}

#[test]
fn initialized_global_relocation_rejects_unknown_target_key() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "unknown_relocation_target".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let reference_ty = StructType::get_unnamed(&ctx, (vec![i64_ty.into()], StructLayout::Unpacked));
    let reference = GlobalOp::new_with_alignment(
        &mut ctx,
        "reference".try_into().unwrap(),
        reference_ty.into(),
        8,
    );
    reference.set_address_space(&mut ctx, 1);
    reference.set_source_global_key(&mut ctx, "REFERENCE");
    reference.set_initializer_hex(&mut ctx, "0000000000000000");
    let encoded = encode_global_initializer_relocations(&[GlobalInitializerRelocation {
        source_offset: 0,
        width_bytes: 8,
        target_address_space: 1,
        target_addend: 0,
        target_key: "MISSING".to_string(),
    }]);
    reference.set_initializer_relocations(&mut ctx, &encoded);
    reference.get_operation().insert_at_back(module_block, &ctx);

    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect_err("unknown relocation target must fail");
    assert!(
        error.contains("unknown rustc global key `MISSING`"),
        "{error}"
    );
}
