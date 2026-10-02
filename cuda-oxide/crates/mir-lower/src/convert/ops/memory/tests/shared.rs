/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::disallowed_methods)]

use super::device_global::append_global_alloc;
use super::*;

/// Build a `mir.shared_alloc` returning `MirPtrType<i32, addrspace=3>` of
/// length `size`, with the given alloc_key, and append it to `block`.
fn append_shared_alloc(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    alloc_key: &str,
    size: u64,
) -> Ptr<Operation> {
    append_shared_alloc_named(ctx, block, alloc_key, size, None)
}

/// As [`append_shared_alloc`], additionally carrying the Rust path of the
/// `static` the allocation came from.
fn append_shared_alloc_named(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    alloc_key: &str,
    size: u64,
    source_name: Option<&str>,
) -> Ptr<Operation> {
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    append_shared_alloc_typed(ctx, block, alloc_key, i32_ty, size, source_name, 0)
}

#[allow(clippy::too_many_arguments)]
fn append_shared_alloc_typed(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    alloc_key: &str,
    element_type: TypeHandle,
    size: u64,
    source_name: Option<&str>,
    alignment: u64,
) -> Ptr<Operation> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::utils::apint::APInt;

    let i32_ty = element_type;
    let result_ty = MirPtrType::get_shared(ctx, i32_ty, true);
    let op = Operation::new(
        ctx,
        mir::MirSharedAllocOp::get_concrete_op_info(),
        vec![result_ty.into()],
        vec![],
        vec![],
        0,
    );
    let alloc = mir::MirSharedAllocOp::new(op);
    alloc.set_attr_elem_type(ctx, TypeAttr::new(i32_ty));
    let size_attr = IntegerAttr::new(
        IntegerType::get(ctx, 64, Signedness::Signless),
        APInt::from_u64(size, std::num::NonZeroUsize::new(64).unwrap()),
    );
    alloc.set_attr_size(ctx, size_attr);
    alloc.set_attr_alloc_key(ctx, StringAttr::new(alloc_key.to_string()));
    if let Some(source_name) = source_name {
        alloc.set_attr_source_name(ctx, StringAttr::new(source_name.to_string()));
    }
    if alignment != 0 {
        alloc.set_alignment_value(ctx, alignment);
    }
    op.insert_at_back(block, ctx);
    op
}

fn shared_array_debug_info(count: u64) -> llvm::DebugGlobalVariableInfo {
    llvm::DebugGlobalVariableInfo {
        name: "TILE".to_string(),
        namespace: vec!["fixture".to_string(), "kernel".to_string()],
        ty: llvm::DebugLocalTypeKind::Array {
            name: format!("[i32; {count}]"),
            size_bits: count * 32,
            element: Box::new(llvm::DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            }),
            count,
        },
        declaration: llvm::DebugSourcePosition {
            file: PathBuf::from("/tmp/shared.rs"),
            line: 7,
            column: 5,
        },
        is_local_to_unit: true,
        is_function_local: true,
    }
}

fn shared_static_key(symbol: &str) -> String {
    dialect_mir::ops::encode_rust_static_global_key(symbol)
}

#[test]
fn convert_shared_alloc_creates_global_in_addrspace_3() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_shared_alloc(&mut ctx, block, "k1", 64);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    // Global lives at module level; addressof lives in the function body.
    let top = module_top_block(&ctx, module_ptr);
    let global = top
        .deref(&ctx)
        .iter(&ctx)
        .find_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .expect("expected an llvm.global for the shared allocation");
    assert_eq!(
        global.address_space(&ctx),
        llvm_addr::SHARED,
        "shared_alloc global must live in addrspace 3"
    );
    assert!(
        global
            .get_symbol_name(&ctx)
            .to_string()
            .starts_with("__shared_mem_"),
        "shared global should have __shared_mem_ prefix"
    );

    let body = kernel_blocks(&ctx, module_ptr);
    let addrof = find_first::<llvm::AddressOfOp>(&ctx, &body).expect("expected an llvm.addressof");
    assert_eq!(
        ptr_addrspace(
            &ctx,
            addrof
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
        ),
        llvm_addr::SHARED,
    );
}

#[test]
fn convert_shared_alloc_deduplicates_by_alloc_key() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    // Two allocations sharing the same alloc_key — they must collapse to
    // a single underlying global (this is what enables a single `static`
    // referenced from multiple sites to land in one shared region).
    append_shared_alloc(&mut ctx, block, "same-key", 64);
    append_shared_alloc(&mut ctx, block, "same-key", 64);
    // A third with a different key must NOT dedupe with them.
    append_shared_alloc(&mut ctx, block, "other-key", 32);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let top = module_top_block(&ctx, module_ptr);
    let shared_globals = top
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .filter(|g| g.address_space(&ctx) == llvm_addr::SHARED)
        .count();
    assert_eq!(
        shared_globals, 2,
        "two distinct alloc_keys must produce two globals"
    );

    // Each of the three mir.shared_alloc ops becomes one addressof.
    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(count_ops::<llvm::AddressOfOp>(&ctx, &body), 3);
}

#[test]
fn convert_shared_alloc_rejects_conflicting_key_declarations() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_shared_alloc(&mut ctx, block, "conflicting-key", 1);
    let differently_aligned = append_shared_alloc(&mut ctx, block, "conflicting-key", 1);
    mir::MirSharedAllocOp::new(differently_aligned).set_alignment_value(&mut ctx, 16);
    append_mir_return(&mut ctx, block, vec![]);

    let error = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("one shared alloc_key must not name differently aligned storage");
    assert!(
        error.to_string().contains("incompatible declaration"),
        "{error}"
    );
}

#[test]
fn shared_alloc_repeated_debug_identity_is_attached_once() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let key = shared_static_key("_Rfixture4TILE");
    for _ in 0..2 {
        let op =
            append_shared_alloc_named(&mut ctx, block, &key, 32, Some("fixture::kernel::TILE"));
        let info = shared_array_debug_info(32);
        llvm::set_debug_global_variable(&mut ctx, op, &info);
        llvm::set_debug_global_owner_function(&mut ctx, op, "fixture_kernel");
    }
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");
    let top = module_top_block(&ctx, module_ptr);
    let globals: Vec<_> = top
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .filter(|global| global.address_space(&ctx) == llvm_addr::SHARED)
        .collect();
    assert_eq!(
        globals.len(),
        1,
        "repeated references must share one global"
    );
    assert_eq!(
        llvm::debug_global_variable(&ctx, globals[0].get_operation()),
        Some(shared_array_debug_info(32))
    );
    assert_eq!(
        llvm::debug_global_owner_function(&ctx, globals[0].get_operation()).as_deref(),
        Some("fixture_kernel")
    );
}

/// One function-local shared static materialized in two owning functions
/// (for example a `#[device]` helper inlined into two kernels) is a valid
/// release build, so debug metadata must not turn it into an error. DWARF
/// cannot truthfully scope one AS3 object to two subprograms either, so
/// the divergent identity fails open: the storage is still shared and the
/// debug attachment is dropped.
#[test]
fn shared_alloc_owner_conflict_drops_debug_metadata() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let key = shared_static_key("_Rfixture4TILE");
    for owner in ["fixture_kernel", "other_kernel"] {
        let op =
            append_shared_alloc_named(&mut ctx, block, &key, 32, Some("fixture::kernel::TILE"));
        let info = shared_array_debug_info(32);
        llvm::set_debug_global_variable(&mut ctx, op, &info);
        llvm::set_debug_global_owner_function(&mut ctx, op, owner);
    }
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("divergent debug identity must not fail a build release accepts");
    let top = module_top_block(&ctx, module_ptr);
    let globals: Vec<_> = top
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
        .filter(|global| global.address_space(&ctx) == llvm_addr::SHARED)
        .collect();
    assert_eq!(
        globals.len(),
        1,
        "the physical storage must still be shared"
    );
    assert!(llvm::debug_global_variable(&ctx, globals[0].get_operation()).is_none());
    assert!(llvm::debug_global_owner_function(&ctx, globals[0].get_operation()).is_none());
}

#[test]
fn dynamic_extern_then_static_shared_rejects_reserved_key_collision() {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::utils::apint::APInt;

    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
    let shared_i8 = MirPtrType::get_shared(&mut ctx, i8_ty, true);

    // Lower the dynamic declaration first. Its internal cache key is
    // deliberately then reused by a static allocation with the same
    // physical type, size, and alignment. Those declarations must still
    // be rejected because one is extern storage and one is fixed storage.
    let dynamic = Operation::new(
        &mut ctx,
        mir::MirExternSharedOp::get_concrete_op_info(),
        vec![shared_i8.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirExternSharedOp::new(dynamic).set_alignment_value(&mut ctx, 128);
    dynamic.insert_at_back(block, &ctx);

    let fixed = Operation::new(
        &mut ctx,
        mir::MirSharedAllocOp::get_concrete_op_info(),
        vec![shared_i8.into()],
        vec![],
        vec![],
        0,
    );
    let fixed_alloc = mir::MirSharedAllocOp::new(fixed);
    fixed_alloc.set_attr_elem_type(&ctx, TypeAttr::new(i8_ty));
    fixed_alloc.set_attr_size(
        &ctx,
        IntegerAttr::new(
            IntegerType::get(&ctx, 64, Signedness::Unsigned),
            APInt::from_u64(0, std::num::NonZeroUsize::new(64).unwrap()),
        ),
    );
    fixed_alloc.set_attr_alloc_key(
        &ctx,
        StringAttr::new("__dynamic_smem_global_created_kernel_func".to_string()),
    );
    fixed_alloc.set_alignment_value(&mut ctx, 128);
    fixed.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    let error = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("a static allocation must not reuse a dynamic extern declaration");
    let message = error.to_string();
    assert!(message.contains("incompatible declaration"), "{message}");
    assert!(
        message.contains("duplicate shared alloc_key"),
        "the dynamic declaration must be observed first, got: {message}"
    );
}

#[test]
fn shared_alloc_mismatched_debug_type_fails_closed_without_failing_lowering() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let key = shared_static_key("_Rfixture4TILE");
    let op = append_shared_alloc_named(&mut ctx, block, &key, 32, Some("fixture::kernel::TILE"));
    llvm::set_debug_global_variable(&mut ctx, op, &shared_array_debug_info(31));
    llvm::set_debug_global_owner_function(&mut ctx, op, "fixture_kernel");
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("bad optional metadata must not fail compilation");
    let top = module_top_block(&ctx, module_ptr);
    let global = top
        .deref(&ctx)
        .iter(&ctx)
        .find_map(|candidate| Operation::get_op::<llvm::GlobalOp>(candidate, &ctx))
        .expect("shared global");
    assert!(llvm::debug_global_variable(&ctx, global.get_operation()).is_none());
    assert!(llvm::debug_global_owner_function(&ctx, global.get_operation()).is_none());
}

#[test]
fn barrier_semantic_struct_matches_single_i64_shared_backing() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let key = shared_static_key("_Rfixture7BARRIER");
    let op = append_shared_alloc_typed(
        &mut ctx,
        block,
        &key,
        i64_ty,
        1,
        Some("fixture::kernel::BARRIER"),
        8,
    );
    let info = llvm::DebugGlobalVariableInfo {
        name: "BARRIER".to_string(),
        namespace: vec!["fixture".to_string(), "kernel".to_string()],
        ty: llvm::DebugLocalTypeKind::Struct {
            name: "Barrier".to_string(),
            size_bits: 64,
            members: vec![llvm::DebugTypeMember {
                name: "_state".to_string(),
                offset_bits: 0,
                ty: llvm::DebugLocalTypeKind::Basic {
                    name: "u64".to_string(),
                    size_bits: 64,
                    encoding: "DW_ATE_unsigned",
                },
            }],
        },
        declaration: llvm::DebugSourcePosition {
            file: PathBuf::from("/tmp/barrier.rs"),
            line: 8,
            column: 9,
        },
        is_local_to_unit: true,
        is_function_local: true,
    };
    llvm::set_debug_global_variable(&mut ctx, op, &info);
    llvm::set_debug_global_owner_function(&mut ctx, op, "fixture_kernel");
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("Barrier lowering succeeds");
    let top = module_top_block(&ctx, module_ptr);
    let global = top
        .deref(&ctx)
        .iter(&ctx)
        .find_map(|candidate| Operation::get_op::<llvm::GlobalOp>(candidate, &ctx))
        .expect("Barrier shared global");
    assert_eq!(global.get_alignment(&ctx), Some(8));
    assert_eq!(
        llvm::debug_global_variable(&ctx, global.get_operation()),
        Some(info)
    );
}

/// Collect `(symbol, source_name)` for every shared global in the module.
fn shared_global_source_names(
    ctx: &Context,
    module_ptr: Ptr<Operation>,
) -> Vec<(String, Option<String>)> {
    use llvm_export::ops::GlobalOpExt;

    let top = module_top_block(ctx, module_ptr);
    let mut named: Vec<_> = top
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, ctx))
        .filter(|g| g.address_space(ctx) == llvm_addr::SHARED)
        .map(|g| {
            (
                g.get_symbol_name(ctx).to_string(),
                g.shared_source_name(ctx),
            )
        })
        .collect();
    // Globals are inserted at the front of the module block, so iteration
    // order is the reverse of creation order. Sort for a stable assertion.
    named.sort();
    named
}

#[test]
fn shared_alloc_source_name_reaches_the_generated_global() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_shared_alloc_named(&mut ctx, block, "k1", 64, Some("my_kernel::TILE"));
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let named = shared_global_source_names(&ctx, module_ptr);
    assert_eq!(named.len(), 1, "expected exactly one shared global");
    let (symbol, source_name) = &named[0];
    // The symbol itself must stay anonymous: the whole point of the
    // sidecar attribute is that it does not perturb the emitted name.
    assert!(
        symbol.starts_with("__shared_mem_"),
        "the generated symbol must not be renamed, got `{symbol}`"
    );
    assert_eq!(source_name.as_deref(), Some("my_kernel::TILE"));
}

#[test]
fn shared_alloc_without_source_name_leaves_the_global_unlabelled() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_shared_alloc(&mut ctx, block, "k1", 64);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let named = shared_global_source_names(&ctx, module_ptr);
    assert_eq!(named.len(), 1);
    assert_eq!(named[0].1, None, "an unnamed allocation must stay unnamed");
}

#[test]
fn shared_alloc_source_names_are_per_global_not_shared_across_them() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    // Two references to one static dedupe onto a single global, and a
    // second static gets its own. Each global must carry its own name —
    // the failure this guards is one name leaking onto every allocation.
    append_shared_alloc_named(&mut ctx, block, "tile", 64, Some("my_kernel::TILE"));
    append_shared_alloc_named(&mut ctx, block, "tile", 64, Some("my_kernel::TILE"));
    append_shared_alloc_named(&mut ctx, block, "scratch", 32, Some("my_kernel::SCRATCH"));
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let names: Vec<_> = shared_global_source_names(&ctx, module_ptr)
        .into_iter()
        .map(|(_, source_name)| source_name)
        .collect();
    assert_eq!(names.len(), 2, "the shared alloc_key must still dedupe");
    let mut names: Vec<_> = names.into_iter().map(|n| n.expect("named")).collect();
    names.sort();
    assert_eq!(names, vec!["my_kernel::SCRATCH", "my_kernel::TILE"]);
}

#[test]
fn shared_alloc_source_name_reaches_the_exported_llvm_ir() {
    // The end the feature exists for: a consumer holding only the emitted
    // artifact can tell which Rust `static` a `__shared_mem_N` block is.
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_shared_alloc_named(&mut ctx, block, "tile", 64, Some("my_kernel::TILE"));
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let module = Operation::get_op::<pliron::builtin::ops::ModuleOp>(module_ptr, &ctx)
        .expect("lowered top-level op is a module");
    let ir = llvm_export::export::export_module_to_string(&ctx, &module).expect("export");

    let comment_index = ir
        .find("; shared source: my_kernel::TILE")
        .unwrap_or_else(|| panic!("exported IR must name the shared source:\n{ir}"));
    let definition_index = ir
        .find("__shared_mem_")
        .expect("exported IR must declare the shared global");
    assert!(
        comment_index < definition_index,
        "the source comment must precede the global it describes:\n{ir}"
    );
}

/// A `__shared_mem_N` or `__device_global_N` index must depend only on
/// the module being lowered, not on how many allocations any OTHER
/// module has already lowered in this process (#706). Before the fix,
/// each `N` came from a `static AtomicUsize` shared across every call in
/// the process, so lowering the second of these two modules would have
/// produced `__shared_mem_1` and `__device_global_1`, not the `_0` names.
#[test]
fn shared_and_device_global_indices_are_per_module_not_process_global() {
    for _ in 0..2 {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        append_shared_alloc(&mut ctx, block, "k", 64);
        append_global_alloc(&mut ctx, block, "ordinary_static", false);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let names: Vec<String> = top
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .map(|g| g.get_symbol_name(&ctx).to_string())
            .collect();
        assert!(
            names.iter().any(|n| n == "__shared_mem_0"),
            "a module with exactly one shared allocation must always name it \
                 __shared_mem_0, regardless of how many other modules already lowered \
                 one in this process (got {names:?})"
        );
        assert!(
            names.iter().any(|n| n == "__device_global_0"),
            "a module with exactly one ordinary device global must always name it \
                 __device_global_0, regardless of how many other modules already \
                 lowered one in this process (got {names:?})"
        );
    }
}

/// Per-module counters restart at zero in every module, so two crates'
/// modules both define `__shared_mem_0` / `__device_global_0`, and the
/// textual PTX bundle merge in `load_all_ptx_bundles_merged` fails driver
/// JIT compilation on the duplicate definition (#1277). A caller whose
/// output can be merged passes a per-compilation disambiguator; the name
/// must then be a pure function of (disambiguator, family, module-local
/// counter) — deterministic for one crate, distinct across crates.
#[test]
fn module_disambiguator_separates_counter_named_globals_across_modules() {
    let lower_one_module = |disambiguator: Option<u64>| -> Vec<String> {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        append_shared_alloc(&mut ctx, block, "k", 64);
        append_global_alloc(&mut ctx, block, "ordinary_static", false);
        append_extern_shared(&mut ctx, block, 16, 0);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            crate::LoweringOptions {
                module_disambiguator: disambiguator,
                ..Default::default()
            },
        )
        .expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        top.deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .map(|g| g.get_symbol_name(&ctx).to_string())
            .collect()
    };

    // The exact shape is part of the cross-crate uniqueness contract: the
    // disambiguator sits between family and counter, zero-padded to the full
    // u64 width so no (disambiguator, counter) pair can collide with another
    // by concatenation.
    let crate_a = lower_one_module(Some(0x00c0_ffee));
    for expected in [
        "__shared_mem_0000000000c0ffee_0",
        "__device_global_0000000000c0ffee_0",
        // The dynamic pool symbol keeps its owner suffix: within one module
        // the owner keeps per-function pools apart, and the crate namespace
        // keeps the same owner's declarations apart across merged bundles,
        // whose alignments would otherwise silently resolve by bundle order.
        "__dynamic_smem_0000000000c0ffee_kernel_func",
    ] {
        assert!(
            crate_a.iter().any(|n| n == expected),
            "expected {expected} among disambiguated globals (got {crate_a:?})"
        );
    }

    // The same module lowered under a different crate identity shares no
    // module-scope symbol with the first: this is the #1277 merge invariant.
    let crate_b = lower_one_module(Some(0xdead_beef));
    for name in &crate_b {
        assert!(
            !crate_a.contains(name),
            "module-scope symbol {name} collides across differently \
             disambiguated modules (a: {crate_a:?}, b: {crate_b:?})"
        );
    }

    // No disambiguator keeps the historical undecorated names.
    let undecorated = lower_one_module(None);
    for expected in [
        "__shared_mem_0",
        "__device_global_0",
        "__dynamic_smem_kernel_func",
    ] {
        assert!(
            undecorated.iter().any(|n| n == expected),
            "expected historical name {expected} without a disambiguator \
             (got {undecorated:?})"
        );
    }
}

/// Append a `mir.extern_shared` (dynamic shared memory) to `block` with the
/// given launch-contract alignment and byte offset into the pool.
fn append_extern_shared(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    alignment: u64,
    byte_offset: u64,
) -> Ptr<Operation> {
    let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let shared_i8 = MirPtrType::get_shared(ctx, i8_ty, true);
    let op = Operation::new(
        ctx,
        mir::MirExternSharedOp::get_concrete_op_info(),
        vec![shared_i8.into()],
        vec![],
        vec![],
        0,
    );
    let extern_shared = mir::MirExternSharedOp::new(op);
    extern_shared.set_alignment_value(ctx, alignment);
    extern_shared.set_byte_offset_value(ctx, byte_offset);
    op.insert_at_back(block, ctx);
    op
}

/// A `mir.extern_shared` with a nonzero byte offset addresses into an extern
/// `[0 x i8]` addrspace(3) global whose real size only exists at launch, so
/// the lowering cannot promise LLVM's `inbounds` allocated-object contract
/// for that GEP:
///
/// ```text
/// @__dynamic_smem_*  external addrspace(3) global [0 x i8]      size known at launch only
///        |
///        +-- getelementptr i8, ptr addrspace(3) @__dynamic_smem_*, i64 64     plain: fine
///        +-- getelementptr inbounds i8, ...                                    promise we cannot keep
/// ```
///
/// Before the upstream no-wrap flags, the exporter treated an unset attribute
/// as `inbounds`, so this GEP shipped as `getelementptr inbounds`. It must
/// stay a plain `getelementptr`.
///
/// Scope: today the importer only sets `byte_offset = 0` on this op (that is
/// `DynamicSharedArray::get()`); `offset(N)` goes through a separate
/// `mir.ptr_offset`. So this locks the op's own contract; the `offset(N)`
/// path real kernels take is a separate follow-up.
#[test]
fn extern_shared_byte_offset_gep_carries_no_no_wrap_flags() {
    use llvm_export::attributes::GepNoWrapFlags;

    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_extern_shared(&mut ctx, block, 16, 64);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body)
        .expect("a nonzero byte offset must lower to one llvm.gep");
    assert_eq!(
        gep.no_wrap_flags(&ctx),
        GepNoWrapFlags::empty(),
        "the dynamic shared-memory GEP must not claim inbounds, nusw, or nuw"
    );

    let module = Operation::get_op::<pliron::builtin::ops::ModuleOp>(module_ptr, &ctx)
        .expect("lowered top-level op is a module");
    let ir = llvm_export::export::export_module_to_string(&ctx, &module).expect("export");
    assert!(
        ir.contains("getelementptr i8,"),
        "the byte-offset GEP must export as a plain i8 getelementptr:\n{ir}"
    );
    assert!(
        !ir.contains("getelementptr inbounds"),
        "the dynamic shared-memory GEP must not export as inbounds:\n{ir}"
    );
}

/// Control for the test above: `DynamicSharedArray::get()` (offset 0) is
/// just the pool base, so the lowering hands back the `llvm.addressof`
/// directly and there is no GEP whose flags could be wrong.
#[test]
fn extern_shared_zero_byte_offset_is_the_bare_address_of() {
    let mut ctx = make_ctx();
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
    append_extern_shared(&mut ctx, block, 16, 0);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(
        count_ops::<llvm::GetElementPtrOp>(&ctx, &body),
        0,
        "a zero byte offset must not emit a GEP at all"
    );
    assert_eq!(
        count_ops::<llvm::AddressOfOp>(&ctx, &body),
        1,
        "a zero byte offset lowers to exactly one llvm.addressof of the pool"
    );
}
