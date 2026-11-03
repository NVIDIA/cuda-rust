/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

#[test]
fn test_packed_atomic_add_lowers_to_exact_side_effecting_ptx() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into()]);
    let address = entry.deref(&ctx).get_argument(0);
    let addend = entry.deref(&ctx).get_argument(1);

    for op_info in [
        nvvm::NvvmAtomAddF16x2Op::get_concrete_op_info(),
        nvvm::NvvmAtomAddBf16x2Op::get_concrete_op_info(),
    ] {
        Operation::new(
            &mut ctx,
            op_info,
            vec![u32_ty.into()],
            vec![address, addend],
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);
    }
    for format in [
        nvvm::PackedAtomicFormatAttr::F16x2,
        nvvm::PackedAtomicFormatAttr::Bf16x2,
    ] {
        nvvm::PackedAtomicAddOp::build(&mut ctx, address, addend, format)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        "atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "atom.global.add.noftz.bf16x2 $0, [$1], $2;",
    ];
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut lowered = Vec::new();

    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                assert!(
                    !template.contains("atom.cas"),
                    "packed atomic exact-native lowering must not use a CAS loop: {template}"
                );
                if !template.starts_with("atom.global.add.noftz.") {
                    continue;
                }
                lowered.push((
                    template,
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .unwrap_or_default(),
                    llvm::asm_kind(&ctx, &asm),
                    body_op.deref(&ctx).get_num_operands(),
                    body_op.deref(&ctx).get_num_results(),
                ));
            }
        }
    }

    assert_eq!(lowered.len(), expected.len() * 2);
    for instruction in expected {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _)| template == instruction)
            .collect();
        assert_eq!(
            matches.len(),
            2,
            "legacy and generated paths must have exact PTX parity for {instruction}"
        );
        for (_, constraints, kind, operands, results) in matches {
            assert_eq!(constraints, "=r,l,r,~{memory}");
            assert_eq!(*kind, llvm::AsmKind::SideEffect);
            assert_eq!(*operands, 2);
            assert_eq!(*results, 1);
        }
    }

    Ok(())
}

#[test]
fn test_generated_packed_atomic_add_libnvvm_route_is_exact() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into()]);
    let address = entry.deref(&ctx).get_argument(0);
    let addend = entry.deref(&ctx).get_argument(1);
    for format in [
        nvvm::PackedAtomicFormatAttr::F16x2,
        nvvm::PackedAtomicFormatAttr::Bf16x2,
    ] {
        nvvm::PackedAtomicAddOp::build(&mut ctx, address, addend, format)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LibNvvm,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;

    let lowered = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|op| {
            let asm = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx)?;
            let template = asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))?;
            template.starts_with("atom.global.add.noftz.").then(|| {
                (
                    template,
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone())),
                    llvm::asm_kind(&ctx, &asm),
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(lowered.len(), 2);
    assert!(
        lowered
            .iter()
            .any(|(template, _, _)| { template == "atom.global.add.noftz.f16x2 $0, [$1], $2;" })
    );
    assert!(
        lowered
            .iter()
            .any(|(template, _, _)| { template == "atom.global.add.noftz.bf16x2 $0, [$1], $2;" })
    );
    for (_, constraints, kind) in lowered {
        assert_eq!(constraints.as_deref(), Some("=r,l,r,~{memory}"));
        assert_eq!(kind, llvm::AsmKind::SideEffect);
    }
    Ok(())
}

/// Scoped atomic loads and stores must lower to inline PTX, not to
/// `load atomic` / `store atomic`.
///
/// libNVVM rejects the IR-level form outright ("Atomic loads/stores are not
/// supported"), so lowering to `llvm::AtomicLoadOp` / `llvm::AtomicStoreOp`
/// produced modules that no `--materialize-cubin` build could consume, making
/// every `DeviceAtomic*::load` and `::store` call a build failure. The PTX
/// instructions themselves have existed since sm_70, so the ops lower through
/// inline assembly instead.
///
/// This asserts the exact template, constraints and asm kind, because all
/// three are load-bearing: the scope qualifier is the whole point of the
/// feature, the constraint register class must match the operand width, and
/// `SideEffect` plus the memory clobber is what stops a publication spin being
/// hoisted out of its loop.
#[test]
fn test_scoped_atomic_load_store_lower_to_inline_ptx() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::atomic::{
        AtomicOrdering, AtomicScope, NvvmAtomicLoadOp, NvvmAtomicStoreOp,
    };
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get_global(&mut ctx, u32_ty.into(), true);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into(), u64_ty.into()]);
    let address = entry.deref(&ctx).get_argument(0);
    let val32 = entry.deref(&ctx).get_argument(1);
    let val64 = entry.deref(&ctx).get_argument(2);

    // One per scope, so a regression that hardcodes a scope is caught rather
    // than passing on the Device case alone.
    NvvmAtomicLoadOp::build(
        &mut ctx,
        address,
        u32_ty.into(),
        AtomicOrdering::Relaxed,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    NvvmAtomicLoadOp::build(
        &mut ctx,
        address,
        u32_ty.into(),
        AtomicOrdering::Acquire,
        AtomicScope::Block,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    NvvmAtomicLoadOp::build(
        &mut ctx,
        address,
        u64_ty.into(),
        AtomicOrdering::Relaxed,
        AtomicScope::System,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    NvvmAtomicStoreOp::build(
        &mut ctx,
        val32,
        address,
        AtomicOrdering::Release,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    NvvmAtomicStoreOp::build(
        &mut ctx,
        val64,
        address,
        AtomicOrdering::Relaxed,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut lowered = Vec::new();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                // The IR-level forms are what libNVVM rejects; neither may survive.
                assert!(
                    Operation::get_op::<llvm::AtomicLoadOp>(body_op, &ctx).is_none(),
                    "scoped atomic load must not lower to `load atomic`: libNVVM rejects it"
                );
                assert!(
                    Operation::get_op::<llvm::AtomicStoreOp>(body_op, &ctx).is_none(),
                    "scoped atomic store must not lower to `store atomic`: libNVVM rejects it"
                );
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                if !template.starts_with("ld.") && !template.starts_with("st.") {
                    continue;
                }
                lowered.push((
                    template,
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .unwrap_or_default(),
                    llvm::asm_kind(&ctx, &asm),
                ));
            }
        }
    }

    // Every access keeps the `~{memory}` clobber, Relaxed included. Without
    // it LLVM may move plain loads and stores of the same address across the
    // asm, breaking the single-thread coherence Rust still guarantees for
    // Relaxed atomics; libcu++ keeps the clobber on its relaxed accesses too.
    let expected = [
        ("ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"),
        ("ld.acquire.cta.b32 $0, [$1];", "=r,l,~{memory}"),
        ("ld.relaxed.sys.b64 $0, [$1];", "=l,l,~{memory}"),
        ("st.release.gpu.b32 [$0], $1;", "l,r,~{memory}"),
        ("st.relaxed.gpu.b64 [$0], $1;", "l,l,~{memory}"),
    ];
    assert_eq!(
        lowered.len(),
        expected.len(),
        "expected one inline-asm op per scoped atomic load/store, got {lowered:?}"
    );
    for (template, constraints) in expected {
        let found = lowered
            .iter()
            .find(|(t, _, _)| t == template)
            .unwrap_or_else(|| panic!("missing lowering for {template}; got {lowered:?}"));
        assert_eq!(found.1, constraints, "constraints for {template}");
        // Not Convergent: these are per-thread accesses, not warp-synchronous.
        // SideEffect with the memory clobber is what keeps a spin re-reading.
        assert_eq!(
            found.2,
            llvm::AsmKind::SideEffect,
            "asm kind for {template}"
        );
    }
    Ok(())
}

#[test]
fn test_pointer_atomic_load_store_use_b64_pointer_registers() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::atomic::{
        AtomicOrdering, AtomicScope, NvvmAtomicLoadOp, NvvmAtomicStoreOp,
    };
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);

    // Atomic value type: *mut u64.
    let value_ptr_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_generic(&mut ctx, u64_ty.into(), true).into();

    // Atomic storage type: *mut (*mut u64).
    let address_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_global(&mut ctx, value_ptr_ty, true).into();

    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![address_ty, value_ptr_ty]);
    let address = entry.deref(&ctx).get_argument(0);
    let pointer_value = entry.deref(&ctx).get_argument(1);

    NvvmAtomicLoadOp::build(
        &mut ctx,
        address,
        value_ptr_ty,
        AtomicOrdering::Relaxed,
        AtomicScope::System,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    NvvmAtomicStoreOp::build(
        &mut ctx,
        pointer_value,
        address,
        AtomicOrdering::Relaxed,
        AtomicScope::System,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut lowered = Vec::new();

    for op in lowered_kernel_body(&ctx, module_ptr) {
        let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };

        let template = asm
            .get_attr_llvm_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();

        if !template.starts_with("ld.") && !template.starts_with("st.") {
            continue;
        }

        let constraints = asm
            .get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();

        lowered.push((template, constraints));
    }

    assert_eq!(
        lowered,
        vec![
            (
                "ld.relaxed.sys.b64 $0, [$1];".to_string(),
                "=l,l,~{memory}".to_string(),
            ),
            (
                "st.relaxed.sys.b64 [$0], $1;".to_string(),
                "l,l,~{memory}".to_string(),
            ),
        ]
    );

    Ok(())
}

#[test]
fn test_pointer_atomic_load_rejects_shared_value_pointer() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::atomic::{AtomicOrdering, AtomicScope, NvvmAtomicLoadOp};
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);

    let shared_value_ptr_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_shared(&mut ctx, u64_ty.into(), true).into();

    let address_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_generic(&mut ctx, shared_value_ptr_ty, true).into();

    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![address_ty]);
    let address = entry.deref(&ctx).get_argument(0);

    NvvmAtomicLoadOp::build(
        &mut ctx,
        address,
        shared_value_ptr_ty,
        AtomicOrdering::Relaxed,
        AtomicScope::System,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    append_return(&mut ctx, entry);

    let error = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("shared pointer atomic values must fail closed")
        .to_string();

    assert!(
        error.contains("nvvm.atomic_load has an unsupported value type"),
        "unexpected error: {error}",
    );

    Ok(())
}

/// Orderings a PTX load or store cannot carry must be rejected, not silently
/// weakened.
///
/// Acquire is load-only, Release is store-only, and AcqRel makes no sense on
/// a single access. Approximating any of them with `relaxed` would be a
/// correctness bug that no runtime test would catch, because the wrong answer
/// only appears under contention on hardware that reorders.
#[test]
fn test_scoped_atomic_rejects_inexpressible_orderings() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::atomic::{
        AtomicOrdering, AtomicScope, NvvmAtomicLoadOp, NvvmAtomicStoreOp,
    };
    use pliron::builtin::types::{IntegerType, Signedness};

    for (is_load, ordering) in [
        (true, AtomicOrdering::Release), // release is store-only
        (true, AtomicOrdering::AcqRel),
        (false, AtomicOrdering::Acquire), // acquire is load-only
        (false, AtomicOrdering::AcqRel),
    ] {
        let mut ctx = make_test_ctx();
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into()]);
        let address = entry.deref(&ctx).get_argument(0);
        let val = entry.deref(&ctx).get_argument(1);

        if is_load {
            NvvmAtomicLoadOp::build(
                &mut ctx,
                address,
                u32_ty.into(),
                ordering.clone(),
                AtomicScope::Device,
            )
            .get_operation()
            .insert_at_back(entry, &ctx);
        } else {
            NvvmAtomicStoreOp::build(
                &mut ctx,
                val,
                address,
                ordering.clone(),
                AtomicScope::Device,
            )
            .get_operation()
            .insert_at_back(entry, &ctx);
        }
        append_return(&mut ctx, entry);

        let result = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr);
        assert!(
            result.is_err(),
            "{} with {ordering:?} ordering must be rejected, not approximated",
            if is_load { "load" } else { "store" }
        );
    }
    Ok(())
}

/// SeqCst load and store lower to one asm op whose template fuses the
/// `fence.sc` with the access, at every scope.
///
/// PTX has no sequentially consistent load or store instruction. libcu++ maps
/// a SeqCst load to `fence.sc.{scope}` followed by an acquire load at the
/// same scope, and a SeqCst store to `fence.sc.{scope}` followed by a release
/// store. Fusing the fence into the same template keeps the pair inseparable.
#[test]
fn test_seqcst_atomic_load_store_fuse_fence_into_template() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::atomic::{
        AtomicOrdering, AtomicScope, NvvmAtomicLoadOp, NvvmAtomicStoreOp,
    };
    use pliron::builtin::types::{IntegerType, Signedness};

    for (scope, ptx) in [
        (AtomicScope::Device, "gpu"),
        (AtomicScope::Block, "cta"),
        (AtomicScope::System, "sys"),
    ] {
        let mut ctx = make_test_ctx();
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let ptr_ty = MirPtrType::get_global(&mut ctx, u32_ty.into(), true);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into()]);
        let address = entry.deref(&ctx).get_argument(0);
        let val = entry.deref(&ctx).get_argument(1);

        NvvmAtomicLoadOp::build(
            &mut ctx,
            address,
            u32_ty.into(),
            AtomicOrdering::SeqCst,
            scope.clone(),
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        NvvmAtomicStoreOp::build(&mut ctx, val, address, AtomicOrdering::SeqCst, scope)
            .get_operation()
            .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
            .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut lowered = Vec::new();
        for op in lowered_kernel_body(&ctx, module_ptr) {
            let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                continue;
            };
            lowered.push((
                asm.get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default(),
                asm.get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default(),
            ));
        }
        let expected = vec![
            (
                format!("fence.sc.{ptx}; ld.acquire.{ptx}.b32 $0, [$1];"),
                "=r,l,~{memory}".to_string(),
            ),
            (
                format!("fence.sc.{ptx}; st.release.{ptx}.b32 [$0], $1;"),
                "l,r,~{memory}".to_string(),
            ),
        ];
        assert_eq!(lowered, expected, "SeqCst lowering at scope {ptx}");
    }
    Ok(())
}

/// Float atomic loads and stores travel through integer registers with an
/// LLVM-level bitcast on each side.
///
/// The register classes in the constraints are integer (`h`/`r`/`l`), so
/// handing llc a float-typed asm operand or result is a constraint mismatch.
/// The store bitcasts the float to the same-width integer before the asm; the
/// load returns the integer and bitcasts it back to the float type. The f16
/// case also covers the `b16`/`h` arm end to end, which is what makes
/// `DeviceAtomicF16::load`/`store` compile at all.
#[test]
fn test_float_atomic_load_store_bitcast_through_integer_registers() -> Result<(), anyhow::Error> {
    use dialect_mir::types::{MirFP16Type, MirPtrType};
    use dialect_nvvm::ops::atomic::{
        AtomicOrdering, AtomicScope, NvvmAtomicLoadOp, NvvmAtomicStoreOp,
    };
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::{FP32Type, IntegerType};
    use pliron::r#type::Typed;

    let is_expected_float = |ctx: &Context, ty: pliron::r#type::TypeHandle, width: u32| -> bool {
        let ty_ref = ty.deref(ctx);
        match width {
            16 => ty_ref.is::<llvm_types::HalfType>(),
            32 => ty_ref.is::<FP32Type>(),
            _ => false,
        }
    };
    let integer_width_of = |ctx: &Context, ty: pliron::r#type::TypeHandle| -> Option<u32> {
        ty.deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width)
    };

    for (is_f16, ptx_ty, reg, width) in [(false, "b32", "r", 32u32), (true, "b16", "h", 16u32)] {
        let mut ctx = make_test_ctx();
        let elem_ty: pliron::r#type::TypeHandle = if is_f16 {
            MirFP16Type::get(&ctx).into()
        } else {
            FP32Type::get(&ctx).into()
        };
        let ptr_ty = MirPtrType::get_generic(&mut ctx, elem_ty, true);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), elem_ty]);
        let address = entry.deref(&ctx).get_argument(0);
        let val = entry.deref(&ctx).get_argument(1);

        NvvmAtomicLoadOp::build(
            &mut ctx,
            address,
            elem_ty,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        NvvmAtomicStoreOp::build(
            &mut ctx,
            val,
            address,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
            .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut ld_asm = None;
        let mut st_asm = None;
        let mut bitcasts = Vec::new();
        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                let constraints = asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                if template.starts_with("ld.") {
                    ld_asm = Some((op, template, constraints));
                } else if template.starts_with("st.") {
                    st_asm = Some((op, template, constraints));
                }
            }
            if Operation::get_op::<llvm::BitcastOp>(op, &ctx).is_some() {
                bitcasts.push(op);
            }
        }

        // Load direction: asm produces the staging integer, then one bitcast
        // turns it back into the float.
        let (ld_op, ld_template, ld_constraints) =
            ld_asm.expect("float atomic load must lower to inline asm");
        assert_eq!(ld_template, format!("ld.relaxed.gpu.{ptx_ty} $0, [$1];"));
        assert_eq!(ld_constraints, format!("={reg},l,~{{memory}}"));
        let ld_result = ld_op.deref(&ctx).get_result(0);
        assert_eq!(
            integer_width_of(&ctx, ld_result.get_type(&ctx)),
            Some(width),
            "load asm must produce the staging integer, not the float"
        );
        let load_cast = bitcasts
            .iter()
            .copied()
            .find(|&cast| cast.deref(&ctx).get_operand(0) == ld_result)
            .expect("load asm result must be bitcast back to the float type");
        let load_cast_ty = load_cast.deref(&ctx).get_result(0).get_type(&ctx);
        assert!(
            is_expected_float(&ctx, load_cast_ty, width),
            "load bitcast must produce the float type"
        );

        // Store direction: the float is bitcast to the staging integer, and
        // that integer is what the asm consumes.
        let (st_op, st_template, st_constraints) =
            st_asm.expect("float atomic store must lower to inline asm");
        assert_eq!(st_template, format!("st.relaxed.gpu.{ptx_ty} [$0], $1;"));
        assert_eq!(st_constraints, format!("l,{reg},~{{memory}}"));
        let stored = st_op.deref(&ctx).get_operand(1);
        assert_eq!(
            integer_width_of(&ctx, stored.get_type(&ctx)),
            Some(width),
            "store asm must consume the staging integer, not the float"
        );
        let store_cast = bitcasts
            .iter()
            .copied()
            .find(|&cast| cast.deref(&ctx).get_result(0) == stored)
            .expect("store asm value must come from a float-to-integer bitcast");
        let store_src_ty = store_cast.deref(&ctx).get_operand(0).get_type(&ctx);
        assert!(
            is_expected_float(&ctx, store_src_ty, width),
            "store bitcast must consume the float type"
        );
    }
    Ok(())
}

/// A SeqCst fence must go through the typed NVVM intrinsic, and a weaker one
/// through inline PTX.
///
/// PTX defines `membar.level` as a synonym for `fence.sc.level`, so
/// `llvm.nvvm.membar.{cta,gl,sys}` is an exact match for SeqCst and is the route
/// the rest of the crate already uses for fences. No intrinsic exists for a
/// weaker fence, and emitting `membar` for AcqRel would silently upgrade the
/// caller's request to sequential consistency, so that case emits
/// `fence.acq_rel.{scope}` directly.
#[test]
fn test_fence_uses_membar_intrinsic_only_for_seqcst() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::atomic::{AtomicOrdering, AtomicRmwKind, AtomicScope, NvvmAtomicRmwOp};
    use pliron::builtin::types::{IntegerType, Signedness};

    for (ordering, want_intrinsic, want_asm) in [
        (AtomicOrdering::SeqCst, Some("llvm_nvvm_membar_gl"), None),
        (AtomicOrdering::AcqRel, None, Some("fence.acq_rel.gpu;")),
    ] {
        let mut ctx = make_test_ctx();
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into()]);
        let address = entry.deref(&ctx).get_argument(0);
        let addend = entry.deref(&ctx).get_argument(1);

        NvvmAtomicRmwOp::build(
            &mut ctx,
            address,
            addend,
            u32_ty.into(),
            AtomicRmwKind::Add,
            ordering.clone(),
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
            .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut saw_intrinsic = false;
        let mut saw_asm = None;
        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
                && callee.to_string().contains("membar")
            {
                saw_intrinsic = true;
            }
            if let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let t = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|v| String::from((*v).clone()))
                    .unwrap_or_default();
                if t.starts_with("fence.") {
                    saw_asm = Some(t);
                }
            }
        }
        assert_eq!(
            saw_intrinsic,
            want_intrinsic.is_some(),
            "{ordering:?}: membar intrinsic presence"
        );
        assert_eq!(saw_asm.as_deref(), want_asm, "{ordering:?}: inline fence");
        // A SeqCst fence must never also emit `membar` as assembly: that would
        // mean both routes fired.
        if want_intrinsic.is_some() {
            assert!(saw_asm.is_none(), "{ordering:?}: emitted both routes");
        }
    }
    Ok(())
}

/// A first-class atomic fence must lower through the same libNVVM-safe routes
/// used by the RMW ordering workaround.
///
/// `core::sync::atomic::fence` imports with system scope, so the source-level
/// path depends on this exact mapping: Acquire/Release/AcqRel become
/// `fence.acq_rel.sys`, while SeqCst becomes `llvm.nvvm.membar.sys`.
#[test]
fn test_first_class_atomic_fence_lowers_at_system_scope() -> Result<(), anyhow::Error> {
    use dialect_nvvm::ops::atomic::{AtomicOrdering, AtomicScope, NvvmAtomicFenceOp};

    for ordering in [
        AtomicOrdering::Acquire,
        AtomicOrdering::Release,
        AtomicOrdering::AcqRel,
        AtomicOrdering::SeqCst,
    ] {
        let mut ctx = make_test_ctx();
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);

        NvvmAtomicFenceOp::build(&mut ctx, ordering.clone(), AtomicScope::System)
            .get_operation()
            .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
            .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut membar_calls = 0usize;
        let mut fence_asm = Vec::new();
        for op in lowered_kernel_body(&ctx, module_ptr) {
            assert!(
                Operation::get_op::<NvvmAtomicFenceOp>(op, &ctx).is_none(),
                "first-class atomic fence must be fully consumed by lowering"
            );

            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
                && callee.to_string() == "llvm_nvvm_membar_sys"
            {
                membar_calls += 1;
            }

            let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                continue;
            };
            let template = asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default();
            if template.starts_with("fence.") {
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some("~{memory}")
                );
                assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::SideEffect);
                fence_asm.push(template);
            }
        }

        match ordering {
            AtomicOrdering::SeqCst => {
                assert_eq!(membar_calls, 1, "SeqCst must use membar.sys");
                assert!(
                    fence_asm.is_empty(),
                    "SeqCst must not also emit an inline PTX fence"
                );
            }
            AtomicOrdering::Acquire | AtomicOrdering::Release | AtomicOrdering::AcqRel => {
                assert_eq!(membar_calls, 0, "weak fences must not use membar.sys");
                assert_eq!(
                    fence_asm,
                    vec!["fence.acq_rel.sys;".to_owned()],
                    "{ordering:?} must lower to one system-scope acq_rel fence"
                );
            }
            AtomicOrdering::Relaxed => unreachable!("Relaxed is not a valid Rust fence ordering"),
        }
    }

    Ok(())
}

#[test]
fn test_first_class_atomic_fence_rejects_relaxed() -> Result<(), anyhow::Error> {
    use dialect_nvvm::ops::atomic::{AtomicOrdering, AtomicScope, NvvmAtomicFenceOp};

    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    NvvmAtomicFenceOp::build(&mut ctx, AtomicOrdering::Relaxed, AtomicScope::System)
        .get_operation()
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    let result = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr);
    assert!(
        result.is_err(),
        "Relaxed is not a valid atomic fence ordering and must be rejected"
    );
    Ok(())
}
