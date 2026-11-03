/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

fn lower_basic_mbarrier(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let bar_ptr_ty = MirPtrType::get_shared(&mut ctx, u64_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![bar_ptr_ty.into(), u32_ty.into()]);
    let barrier = entry.deref(&ctx).get_argument(0);
    let count = entry.deref(&ctx).get_argument(1);

    nvvm::MbarrierInitSharedOp::build(&mut ctx, barrier, count).insert_at_back(entry, &ctx);
    let arrive = nvvm::MbarrierArriveSharedOp::build(&mut ctx, barrier);
    let token = arrive.deref(&ctx).get_result(0);
    arrive.insert_at_back(entry, &ctx);
    nvvm::MbarrierTestWaitSharedOp::build(&mut ctx, barrier, token).insert_at_back(entry, &ctx);
    nvvm::MbarrierInvalSharedOp::build(&mut ctx, barrier).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_generated_basic_mbarrier_uses_shared_lowering_on_both_backends() -> Result<(), anyhow::Error>
{
    let expected_calls = [
        ("llvm_nvvm_mbarrier_init_shared", 2),
        ("llvm_nvvm_mbarrier_arrive_shared", 1),
        ("llvm_nvvm_mbarrier_inval_shared", 1),
    ];
    let init_template = "mbarrier.init.shared.b64 [$0], $1;";
    let arrive_template = "mbarrier.arrive.shared.b64 $0, [$1];";
    let test_wait_template =
        "{ .reg .pred %p0; mbarrier.test_wait.shared.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }";
    let inval_template = "mbarrier.inval.shared.b64 [$0];";

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_basic_mbarrier(backend)?;
        let mut call_counts = [0usize; 3];
        let expected_asm = match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                vec![(test_wait_template, "=r,l,l,~{memory}", 2)]
            }
            mir_lower::IntrinsicBackend::LibNvvm => vec![
                (init_template, "l,r,~{memory}", 2),
                (arrive_template, "=l,l,~{memory}", 1),
                (test_wait_template, "=r,l,l,~{memory}", 2),
                (inval_template, "l,~{memory}", 1),
            ],
        };
        let mut asm_counts = vec![0usize; expected_asm.len()];
        let mut trunc_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) {
                let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                    continue;
                };
                let callee = callee.to_string();
                let Some(index) = expected_calls
                    .iter()
                    .position(|(expected, _)| callee == *expected)
                else {
                    assert_ne!(callee, "llvm_nvvm_mbarrier_test_wait_shared");
                    continue;
                };
                call_counts[index] += 1;
                assert_eq!(op.deref(&ctx).get_num_operands(), expected_calls[index].1);
            }

            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let index = expected_asm
                    .iter()
                    .position(|(expected, _, _)| template.as_deref() == Some(*expected))
                    .unwrap_or_else(|| panic!("unexpected {backend:?} inline PTX: {template:?}"));
                let (_, constraints, operand_count) = expected_asm[index];
                asm_counts[index] += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some(constraints)
                );
                assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                let asm = op.deref(&ctx);
                assert_eq!(asm.get_num_operands(), operand_count);
                assert_eq!(asm.get_num_results(), 1);
            }

            if Operation::get_op::<llvm::TruncOp>(op, &ctx).is_some() {
                trunc_count += 1;
            }
        }

        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => assert_eq!(call_counts, [1; 3]),
            mir_lower::IntrinsicBackend::LibNvvm => assert_eq!(call_counts, [0; 3]),
        }
        assert_eq!(asm_counts, vec![1; expected_asm.len()]);
        assert_eq!(
            trunc_count, 1,
            "test-wait must adapt its i32 predicate to i1"
        );

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert!(
                    ir.contains("call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(
                    ir.contains("call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(
                    ir.contains("call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(!ir.contains(init_template), "{ir}");
                assert!(!ir.contains(arrive_template), "{ir}");
                assert!(!ir.contains(inval_template), "{ir}");
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                for symbol in [
                    "@llvm.nvvm.mbarrier.init.shared",
                    "@llvm.nvvm.mbarrier.arrive.shared",
                    "@llvm.nvvm.mbarrier.inval.shared",
                ] {
                    assert!(
                        !ir.contains(symbol),
                        "libNVVM route retained {symbol}:\n{ir}"
                    );
                }
                for template in [init_template, arrive_template, inval_template] {
                    assert!(ir.contains(template), "{ir}");
                }
            }
        }
        assert!(ir.contains(test_wait_template), "{ir}");
        assert!(ir.contains("asm sideeffect"), "{ir}");
        assert!(ir.contains("trunc i32") && ir.contains("to i1"), "{ir}");
        assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    }
    Ok(())
}

fn lower_cluster_barriers(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    for mode in [
        nvvm::ClusterBarrierModeAttr::Arrive,
        nvvm::ClusterBarrierModeAttr::ArriveAligned,
        nvvm::ClusterBarrierModeAttr::ArriveRelaxed,
        nvvm::ClusterBarrierModeAttr::ArriveRelaxedAligned,
        nvvm::ClusterBarrierModeAttr::Wait,
        nvvm::ClusterBarrierModeAttr::WaitAligned,
    ] {
        nvvm::ClusterBarrierOp::build(&mut ctx, mode).insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);
    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn generated_cluster_barriers_lower_exactly_on_both_backends() -> Result<(), anyhow::Error> {
    let recipes = [
        (
            "llvm_nvvm_barrier_cluster_arrive",
            "barrier.cluster.arrive;",
        ),
        (
            "llvm_nvvm_barrier_cluster_arrive_aligned",
            "barrier.cluster.arrive.aligned;",
        ),
        (
            "llvm_nvvm_barrier_cluster_arrive_relaxed",
            "barrier.cluster.arrive.relaxed;",
        ),
        (
            "llvm_nvvm_barrier_cluster_arrive_relaxed_aligned",
            "barrier.cluster.arrive.relaxed.aligned;",
        ),
        ("llvm_nvvm_barrier_cluster_wait", "barrier.cluster.wait;"),
        (
            "llvm_nvvm_barrier_cluster_wait_aligned",
            "barrier.cluster.wait.aligned;",
        ),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_cluster_barriers(backend)?;
        let mut calls = [0usize; 6];
        let mut asm = [0usize; 6];
        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) {
                let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                    continue;
                };
                if let Some(index) = recipes
                    .iter()
                    .position(|(symbol, _)| callee.to_string() == *symbol)
                {
                    calls[index] += 1;
                }
            }
            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                if let Some(index) = recipes
                    .iter()
                    .position(|(_, expected)| template.as_deref() == Some(*expected))
                {
                    asm[index] += 1;
                    assert_eq!(
                        inline_asm
                            .get_attr_llvm_inline_asm_constraints(&ctx)
                            .map(|value| String::from((*value).clone()))
                            .as_deref(),
                        Some("~{memory}")
                    );
                    assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                }
            }
        }
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert_eq!(calls, [1; 6]);
                assert_eq!(asm, [0; 6]);
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                assert_eq!(calls, [0; 6]);
                assert_eq!(asm, [1; 6]);
            }
        }
    }
    Ok(())
}

#[test]
fn test_cluster_mbarrier_and_fences_lower_to_exact_inline_ptx() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let bar_ptr_ty = MirPtrType::get_shared(&mut ctx, i64_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![bar_ptr_ty.into(), i32_ty.into()]);

    let bar_ptr = entry.deref(&ctx).get_argument(0);
    let bytes_or_parity = entry.deref(&ctx).get_argument(1);

    let arrive = Operation::new(
        &mut ctx,
        nvvm::MbarrierArriveExpectTxClusterOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![bar_ptr, bytes_or_parity],
        vec![],
        0,
    );
    arrive.insert_at_back(entry, &ctx);

    let try_wait = Operation::new(
        &mut ctx,
        nvvm::MbarrierTryWaitParityClusterOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![bar_ptr, bytes_or_parity],
        vec![],
        0,
    );
    try_wait.insert_at_back(entry, &ctx);

    let mbarrier_fence = Operation::new(
        &mut ctx,
        nvvm::FenceMbarrierInitReleaseClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    mbarrier_fence.insert_at_back(entry, &ctx);

    let proxy_release_fence = Operation::new(
        &mut ctx,
        nvvm::FenceProxyAsyncGenericReleaseSharedCtaClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    proxy_release_fence.insert_at_back(entry, &ctx);

    let proxy_acquire_fence = Operation::new(
        &mut ctx,
        nvvm::FenceProxyAsyncGenericAcquireSharedClusterClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    proxy_acquire_fence.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        (
            "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 $0, [$1], $2;",
            "=l,l,r,~{memory}",
        ),
        (
            "{ .reg .pred %p0; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
            "=r,l,r,~{memory}",
        ),
        ("fence.mbarrier_init.release.cluster;", "~{memory}"),
        (
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
            "~{memory}",
        ),
        (
            "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
            "~{memory}",
        ),
    ];
    let mut matches = [0usize; 5];

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let Some(index) = expected.iter().position(|(expected_template, _)| {
                    template.as_deref() == Some(*expected_template)
                }) else {
                    continue;
                };

                matches[index] += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some(expected[index].1)
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &inline_asm),
                    Some(llvm::AsmKind::Convergent)
                );
            }
        }
    }

    assert_eq!(
        matches, [1; 5],
        "each cluster barrier/fence must lower to its exact PTX template once"
    );
    Ok(())
}

/// The mir-importer encodes `core::sync::atomic::compiler_fence` (issue #781)
/// as an empty, volatile, non-convergent inline-PTX block whose only content
/// is a `~{memory}` clobber. It must lower to a void side-effecting inline asm
/// call with the clobber intact and no instruction text, so no hardware
/// `fence` or `membar` can reach the emitted PTX.
#[test]
fn test_compiler_fence_encoding_lowers_to_empty_sideeffect_asm() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::r#type::Typed;

    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);

    let barrier = nvvm::InlinePtxOp::build(&mut ctx, vec![], vec![], "", "~{memory}", true, false);
    barrier.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_barrier = false;
    for op in lowered_kernel_body(&ctx, module_ptr) {
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        assert!(
            !found_barrier,
            "expected exactly one inline asm op in the lowered kernel"
        );
        found_barrier = true;

        let template = inline_asm
            .get_attr_llvm_inline_asm_template(&ctx)
            .map(|s| String::from((*s).clone()));
        assert_eq!(
            template.as_deref(),
            Some(""),
            "compiler fence must emit no PTX instruction"
        );
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|s| String::from((*s).clone()))
                .as_deref(),
            Some("~{memory}"),
            "compiler fence must keep its memory clobber"
        );
        assert!(
            llvm::inline_asm_sideeffect(&ctx, inline_asm.get_operation()),
            "compiler fence must stay side-effecting so the optimizer cannot drop it"
        );
        assert!(
            !inline_asm
                .get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent")),
            "compiler fence must not be convergent"
        );
        // The zero-result lowering path models the void call as a single
        // result of `llvm.void` type, so assert on the type rather than on
        // the result count.
        let result_ty = inline_asm
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        assert!(
            result_ty
                .deref(&ctx)
                .downcast_ref::<llvm_types::VoidType>()
                .is_some(),
            "compiler fence lowers to a void inline asm call"
        );
    }

    assert!(
        found_barrier,
        "expected the compiler-fence inline asm op in the lowered kernel"
    );
    Ok(())
}

#[test]
fn test_cluster_grid_compatibility_ops_keep_original_lowering() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_type = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    for op_info in [
        nvvm::ReadPtxSregClusterIdxOp::get_concrete_op_info(),
        nvvm::ReadPtxSregNclusterIdOp::get_concrete_op_info(),
    ] {
        Operation::new(&mut ctx, op_info, vec![i32_type.into()], vec![], vec![], 0)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let lowered = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx))
        .filter_map(|asm| {
            let template = asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))?;
            (template.contains("%clusterid") || template.contains("%nclusterid"))
                .then_some((template, asm))
        })
        .collect::<Vec<_>>();
    assert_eq!(lowered.len(), 2);
    assert!(lowered.iter().any(|(template, _)| {
        template
            == "{ .reg .u32 %cx, %cy, %cz, %nx, %ny, %nxy, %xy; mov.u32 %cx, %clusterid.x; mov.u32 %cy, %clusterid.y; mov.u32 %cz, %clusterid.z; mov.u32 %nx, %nclusterid.x; mov.u32 %ny, %nclusterid.y; mul.lo.u32 %nxy, %nx, %ny; mad.lo.u32 %xy, %cy, %nx, %cx; mad.lo.u32 $0, %cz, %nxy, %xy; }"
    }));
    assert!(lowered.iter().any(|(template, _)| {
        template
            == "{ .reg .u32 %nx, %ny, %nz, %nxy; mov.u32 %nx, %nclusterid.x; mov.u32 %ny, %nclusterid.y; mov.u32 %nz, %nclusterid.z; mul.lo.u32 %nxy, %nx, %ny; mul.lo.u32 $0, %nxy, %nz; }"
    }));
    for (_, asm) in lowered {
        assert_eq!(
            asm.get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r")
        );
        assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::Convergent);
    }
    Ok(())
}

fn lower_sync_threads(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    Operation::new(
        &mut ctx,
        nvvm::Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_sync_threads_llvm_nvptx_uses_typed_intrinsic_with_fixed_zero() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;

    let (ctx, module_ptr) = lower_sync_threads(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut found = false;
    for op in body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX sync_threads must use the typed intrinsic"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        if callee.to_string() != "llvm_nvvm_barrier_cta_sync_aligned_all" {
            continue;
        }
        let call = op.deref(&ctx);
        assert_eq!(call.get_num_operands(), 1);
        let barrier_id = call.get_operand(0);
        let defining_op = barrier_id.defining_op().expect("barrier ID is constant");
        let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
            .expect("barrier ID is an LLVM constant");
        let value = constant.get_value(&ctx);
        let integer = (&*value as &dyn pliron::attribute::Attribute)
            .downcast_ref::<IntegerAttr>()
            .expect("barrier ID is an integer");
        assert_eq!(integer.value().bw(), 32);
        assert_eq!(integer.value().to_u64(), 0);
        found = true;
    }
    assert!(found, "modern typed CTA barrier call was not emitted");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(ir.contains("@llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"));
    assert!(!ir.contains("@llvm.nvvm.barrier0"));
    Ok(())
}

#[test]
fn test_sync_threads_libnvvm_uses_exact_convergent_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_sync_threads(mir_lower::IntrinsicBackend::LibNvvm)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut found = false;
    for op in body {
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                assert_ne!(callee.to_string(), "llvm_nvvm_barrier_cta_sync_aligned_all");
            }
            continue;
        };
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("bar.sync 0;")
        );
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("~{memory}")
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
        assert_eq!(op.deref(&ctx).get_num_operands(), 0);
        found = true;
    }
    assert!(found, "exact libNVVM barrier inline PTX was not emitted");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(
        ir.contains("call void asm sideeffect \"bar.sync 0;\", \"~{memory}\"() #0"),
        "{ir}"
    );
    assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    assert!(!ir.contains("@llvm.nvvm.barrier.cta.sync.aligned.all"));
    Ok(())
}

fn lower_warp_barrier(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
    let member_mask = entry.deref(&ctx).get_argument(0);
    nvvm::BarWarpSyncOp::build(&mut ctx, member_mask).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_warp_barrier_uses_typed_intrinsic_on_both_backends() -> Result<(), anyhow::Error> {
    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_warp_barrier(backend)?;
        let body = lowered_kernel_body(&ctx, module_ptr);
        let mut calls = 0;
        for op in body {
            assert!(
                Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
                "warp barrier must use its typed intrinsic"
            );
            let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                continue;
            };
            let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                continue;
            };
            if callee.to_string() == "llvm_nvvm_bar_warp_sync" {
                assert_eq!(op.deref(&ctx).get_num_operands(), 1);
                assert_eq!(op.deref(&ctx).get_num_results(), 1);
                calls += 1;
            }
        }
        assert_eq!(calls, 1, "expected one typed warp-barrier call");

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        assert!(ir.contains("@llvm.nvvm.bar.warp.sync(i32"), "{ir}");
    }
    Ok(())
}

// =============================================================================
// cp.async lowering tests
// =============================================================================
