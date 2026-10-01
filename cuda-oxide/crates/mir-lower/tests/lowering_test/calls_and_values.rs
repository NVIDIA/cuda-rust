/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;

/// Regression cover for explicit MIR address-space coercion before a call.
///
/// When a caller passes a pointer in one address space to a callee whose
/// declared parameter lives in a different address space (the
/// `*mut SharedArray<T, N>` / `addrspace(3)` case that surfaces from
/// `block_reduce` and friends), MIR must record the representational cast
/// before the exact call-signature verifier sees the argument. Lowering then
/// emits the corresponding `llvm.addrspacecast`.
///
/// This test builds two MIR functions in one module:
///   - `callee(p: *mut i32 in addrspace(3))`
///   - `caller(p: *mut i32 in addrspace(0)) { callee(p) }`
///
/// and asserts the lowered `caller` body contains an `AddrSpaceCastOp`.
#[test]
fn addrspace_coercion_inserts_addrspacecast_at_call_site() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use llvm_export::ops::AddrSpaceCastOp;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_addrspace_coercion".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let shared_ptr_ty = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let generic_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let expected_call_location = Location::Named {
        name: "source-device-call".to_string(),
        child_loc: Box::new(Location::Unknown),
    };

    // Callee: takes a *mut i32 in addrspace(3), returns ().
    let callee_func_ty = FunctionType::get(&ctx, vec![shared_ptr_ty.into()], vec![]);
    let callee_func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let callee_func = mir::MirFuncOp::new(
        &mut ctx,
        callee_func_op,
        TypeAttr::new(callee_func_ty.into()),
    );
    callee_func.set_symbol_name(&mut ctx, "callee".try_into().unwrap());
    {
        let region = callee_func.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![shared_ptr_ty.into()]);
        block.insert_at_back(region, &ctx);

        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    callee_func
        .get_operation()
        .insert_at_back(module_block, &ctx);

    // Caller: takes a *mut i32 in addrspace(0), calls `callee` with that
    // pointer. The lowerer is responsible for inserting an addrspacecast
    // since the callee's declared addrspace differs.
    let caller_func_ty = FunctionType::get(&ctx, vec![generic_ptr_ty.into()], vec![]);
    let caller_func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let caller_func = mir::MirFuncOp::new(
        &mut ctx,
        caller_func_op,
        TypeAttr::new(caller_func_ty.into()),
    );
    caller_func.set_symbol_name(&mut ctx, "caller".try_into().unwrap());
    {
        let region = caller_func.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![generic_ptr_ty.into()]);
        block.insert_at_back(region, &ctx);
        let arg = block.deref(&ctx).get_argument(0);
        let cast_op_ptr = Operation::new(
            &mut ctx,
            mir::MirCastOp::get_concrete_op_info(),
            vec![shared_ptr_ty.into()],
            vec![arg],
            vec![],
            0,
        );
        mir::MirCastOp::new(cast_op_ptr)
            .set_attr_cast_kind(&ctx, dialect_mir::attributes::MirCastKindAttr::PtrToPtr);
        cast_op_ptr.insert_at_back(block, &ctx);
        let coerced_arg = cast_op_ptr.deref(&ctx).get_result(0);

        let call_op_ptr = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![],
            vec![coerced_arg],
            vec![],
            0,
        );
        let call_op = mir::MirCallOp::new(call_op_ptr);
        call_op.set_attr_callee(&ctx, StringAttr::new("callee".to_string()));
        call_op_ptr
            .deref_mut(&ctx)
            .set_loc(expected_call_location.clone());
        call_op_ptr.insert_at_back(block, &ctx);

        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    caller_func
        .get_operation()
        .insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_addrspace_cast = false;
    let mut found_call = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "caller" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if Operation::get_op::<AddrSpaceCastOp>(body_op, &ctx).is_some() {
                    found_addrspace_cast = true;
                }
                if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                    && matches!(
                        call.callee(&ctx),
                        CallOpCallable::Direct(symbol) if symbol.to_string() == "callee"
                    )
                {
                    found_call = true;
                    assert_eq!(
                        call.get_operation().deref(&ctx).loc(),
                        expected_call_location
                    );
                }
            }
        }
    }

    assert!(
        found_addrspace_cast,
        "caller body must contain llvm.addrspacecast for the addrspace(0) -> (3) coercion at the call site",
    );
    assert!(found_call, "caller body must contain the lowered call");
    Ok(())
}

/// A zero-sized MIR result is erased from the NVPTX function ABI, but its
/// typed value can remain live inside MIR (for example, when one ZST-returning
/// function returns the result of another). Lowering must keep the void call
/// for side effects and replace only its value result with a typed LLVM undef.
#[test]
fn zst_union_call_result_keeps_void_call_and_replaces_live_uses() -> Result<(), anyhow::Error> {
    use dialect_mir::types::{MirTupleType, MirUnionType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::FunctionType;
    use pliron::r#type::{TypeHandle, Typed};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let unit_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
    let union_ty: TypeHandle = MirUnionType::get(
        &mut ctx,
        "AlignedZeroUnion".into(),
        vec!["unit".into()],
        vec![unit_ty],
        0,
        16,
    )
    .into();

    let module = ModuleOp::new(&mut ctx, "test_zst_union_call".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let callee_ty = FunctionType::get(&ctx, vec![], vec![union_ty]);
    let callee_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let callee = mir::MirFuncOp::new(&mut ctx, callee_ptr, TypeAttr::new(callee_ty.into()));
    callee.set_symbol_name(&mut ctx, "make_zero".try_into().unwrap());
    {
        let region = callee.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(region, &ctx);

        let undef = mir::MirUndefOp::new(&mut ctx, union_ty);
        undef.get_operation().insert_at_back(block, &ctx);
        let value = undef.get_operation().deref(&ctx).get_result(0);

        let ret = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        ret.insert_at_back(block, &ctx);
    }
    callee.get_operation().insert_at_back(module_block, &ctx);

    let caller_ty = FunctionType::get(&ctx, vec![], vec![union_ty]);
    let caller_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let caller = mir::MirFuncOp::new(&mut ctx, caller_ptr, TypeAttr::new(caller_ty.into()));
    caller.set_symbol_name(&mut ctx, "return_called_zero".try_into().unwrap());
    {
        let region = caller.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(region, &ctx);

        let call_ptr = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![union_ty],
            vec![],
            vec![],
            0,
        );
        let call = mir::MirCallOp::new(call_ptr);
        call.set_attr_callee(&ctx, StringAttr::new("make_zero".to_string()));
        call_ptr.insert_at_back(block, &ctx);
        let value = call_ptr.deref(&ctx).get_result(0);

        let ret = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        ret.insert_at_back(block, &ctx);
    }
    caller.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut found_void_call = false;
    let mut caller_undefs = 0;
    let mut found_void_return = false;
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func.get_symbol_name(&ctx).to_string() != "return_called_zero" {
            continue;
        }
        let region = func.get_operation().deref(&ctx).get_region(0);
        for block in region.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                    && let CallOpCallable::Direct(callee) = call.callee(&ctx)
                    && callee.to_string() == "make_zero"
                {
                    let call_op = call.get_operation().deref(&ctx);
                    assert_eq!(call_op.get_num_results(), 1);
                    assert!(
                        call_op
                            .get_result(0)
                            .get_type(&ctx)
                            .deref(&ctx)
                            .is::<llvm_export::types::VoidType>(),
                        "the ZST-returning call must use the void ABI"
                    );
                    found_void_call = true;
                }
                if Operation::get_op::<llvm::UndefOp>(body_op, &ctx).is_some() {
                    caller_undefs += 1;
                }
                if let Some(ret) = Operation::get_op::<llvm::ReturnOp>(body_op, &ctx) {
                    assert_eq!(
                        ret.get_operation().deref(&ctx).get_num_operands(),
                        0,
                        "the caller's ZST return must also use the void ABI"
                    );
                    found_void_return = true;
                }
            }
        }
    }

    assert!(
        found_void_call,
        "the LLVM call must be retained because the callee may have side effects"
    );
    assert_eq!(
        caller_undefs, 1,
        "the live MIR result must be replaced by one typed LLVM undef"
    );
    assert!(
        found_void_return,
        "the caller must retain its return terminator"
    );
    Ok(())
}

/// Lock the comparison-predicate lowering table to the rustc_codegen_ssa
/// reference (`bin_op_to_fcmp_predicate` / `bin_op_to_icmp_predicate`):
///
/// | MIR op   | float `fcmp`      | signed `icmp` | unsigned `icmp` |
/// |----------|-------------------|---------------|-----------------|
/// | `mir.eq` | `oeq` (ordered)   | `eq`          | `eq`            |
/// | `mir.ne` | `une` (UNordered) | `ne`          | `ne`            |
/// | `mir.lt` | `olt`             | `slt`         | `ult`           |
/// | `mir.le` | `ole`             | `sle`         | `ule`           |
/// | `mir.gt` | `ogt`             | `sgt`         | `ugt`           |
/// | `mir.ge` | `oge`             | `sge`         | `uge`           |
///
/// `ne` is the one float predicate that must be UNordered: Rust requires
/// `a != b == !(a == b)`, so `x != x` must be true for NaN (issue #123;
/// the ordered `one` folds the canonical NaN check to `false`).
///
/// The test also locks fastmath flags to *empty* on every lowered `fcmp`:
/// a future `nnan` default would make `fcmp nnan une x, x` poison for NaN
/// and silently re-break NaN detection while the predicate assertion above
/// stays green.
#[test]
fn test_cmp_predicate_lowering() -> Result<(), anyhow::Error> {
    use llvm_export::attributes::{FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr};
    use llvm_export::op_interfaces::FastMathFlags;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_signed = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signed,
    );
    let u32_unsigned = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let bool_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        1,
        pliron::builtin::types::Signedness::Signless,
    );

    // Args: (f32, f32, i32, u32). The integer args carry pre-conversion
    // signedness, which is what selects signed vs unsigned icmp predicates.
    let arg_tys: Vec<pliron::r#type::TypeHandle> = vec![
        f32_ty.into(),
        f32_ty.into(),
        i32_signed.into(),
        u32_unsigned.into(),
    ];
    let func_name = "cmp_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, arg_tys.clone(), vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, arg_tys);
        b.insert_at_back(region, &ctx);
        b
    };
    let fa = block.deref(&ctx).get_argument(0);
    let fb = block.deref(&ctx).get_argument(1);
    let si = block.deref(&ctx).get_argument(2);
    let ui = block.deref(&ctx).get_argument(3);

    // One comparison op per table row, in a fixed program order. The raw
    // `Operation::new` construction mirrors how the importer builds these
    // ops (mir-importer translator/rvalue/expr.rs, `Rvalue::BinaryOp` arm).
    let cmp_infos = [
        // Floats: all six predicates.
        (mir::MirEqOp::get_concrete_op_info(), fa, fb),
        (mir::MirNeOp::get_concrete_op_info(), fa, fb),
        (mir::MirLtOp::get_concrete_op_info(), fa, fb),
        (mir::MirLeOp::get_concrete_op_info(), fa, fb),
        (mir::MirGtOp::get_concrete_op_info(), fa, fb),
        (mir::MirGeOp::get_concrete_op_info(), fa, fb),
        // Signed integers: eq/ne are sign-agnostic, the rest must be s*.
        (mir::MirEqOp::get_concrete_op_info(), si, si),
        (mir::MirNeOp::get_concrete_op_info(), si, si),
        (mir::MirLtOp::get_concrete_op_info(), si, si),
        (mir::MirLeOp::get_concrete_op_info(), si, si),
        (mir::MirGtOp::get_concrete_op_info(), si, si),
        (mir::MirGeOp::get_concrete_op_info(), si, si),
        // Unsigned integers: the relational predicates must be u*.
        (mir::MirLtOp::get_concrete_op_info(), ui, ui),
        (mir::MirLeOp::get_concrete_op_info(), ui, ui),
        (mir::MirGtOp::get_concrete_op_info(), ui, ui),
        (mir::MirGeOp::get_concrete_op_info(), ui, ui),
    ];
    for (info, lhs, rhs) in cmp_infos {
        let op = Operation::new(
            &mut ctx,
            info,
            vec![bool_ty.into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
    }

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op_ptr.insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Collect lowered predicates in program order.
    let mut fcmp_preds = Vec::new();
    let mut icmp_preds = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != func_name {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if let Some(fcmp) = Operation::get_op::<llvm::FCmpOp>(body_op, &ctx) {
                    fcmp_preds.push(fcmp.predicate(&ctx));
                    // fcmp carries `contract` (set by add_fastmath_flags) which is a
                    // no-op for comparisons at the LLVM / PTX level. Critically, nnan
                    // is NOT set, so NaN checks like `x != x` still evaluate correctly.
                    let expected: FastmathFlagsAttr =
                        llvm_export::attributes::FastmathFlags::CONTRACT.into();
                    assert_eq!(
                        fcmp.fast_math_flags(&ctx),
                        expected,
                        "fcmp must carry only the contract flag (nnan would poison NaN checks)"
                    );
                }
                if let Some(icmp) = Operation::get_op::<llvm::ICmpOp>(body_op, &ctx) {
                    icmp_preds.push(icmp.predicate(&ctx));
                }
            }
        }
    }

    assert_eq!(
        fcmp_preds,
        vec![
            FCmpPredicateAttr::OEQ,
            FCmpPredicateAttr::UNE,
            FCmpPredicateAttr::OLT,
            FCmpPredicateAttr::OLE,
            FCmpPredicateAttr::OGT,
            FCmpPredicateAttr::OGE,
        ],
        "float comparison predicates must mirror rustc: ordered except Ne (une)"
    );
    assert_eq!(
        icmp_preds,
        vec![
            ICmpPredicateAttr::EQ,
            ICmpPredicateAttr::NE,
            ICmpPredicateAttr::SLT,
            ICmpPredicateAttr::SLE,
            ICmpPredicateAttr::SGT,
            ICmpPredicateAttr::SGE,
            ICmpPredicateAttr::ULT,
            ICmpPredicateAttr::ULE,
            ICmpPredicateAttr::UGT,
            ICmpPredicateAttr::UGE,
        ],
        "integer comparison predicates must respect pre-conversion signedness"
    );
    Ok(())
}

/// Regression cover for PR #141: comparisons whose operand is a bool phi.
///
/// Bools are signless i1, which `can_convert_type` rejects (signless is
/// already the LLVM form), so DialectConversion records no type history for
/// a bool block argument. `is_signed_int_op` used to error out for such
/// operands ("expected IntegerType or MirPtrType operand in arithmetic op");
/// it must instead fall back to the live operand type and lower the
/// comparison as unsigned.
///
/// The function mirrors the MIR of a short-circuit kernel:
///
/// ```text
/// let p = a || b;            // bool phi: merge block argument
/// out = (p == q, p < q);     // icmp eq i1 / icmp ult i1
/// ```
///
/// ```text
/// bb0(a: i1, b: i1, q: i1):  mir.cond_br a, bb2(a), bb1()
/// bb1():                     mir.goto bb2(b)
/// bb2(p: i1):                mir.eq p, q ; mir.lt p, q ; mir.return
/// ```
#[test]
fn test_bool_phi_cmp_lowers_to_unsigned_i1_icmp() -> Result<(), anyhow::Error> {
    use llvm_export::attributes::ICmpPredicateAttr;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
    use pliron::r#type::Typed;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let bool_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let arg_tys: Vec<pliron::r#type::TypeHandle> =
        vec![bool_ty.into(), bool_ty.into(), bool_ty.into()];
    let func_name = "bool_phi_cmp";
    let func_ty = FunctionType::get(&ctx, arg_tys.clone(), vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);

    // bb0(a, b, q): the function entry.
    let bb0 = BasicBlock::new(&mut ctx, None, arg_tys);
    bb0.insert_at_back(region, &ctx);
    let a = bb0.deref(&ctx).get_argument(0);
    let b = bb0.deref(&ctx).get_argument(1);
    let q = bb0.deref(&ctx).get_argument(2);

    // bb1(): the short-circuit "evaluate b" block.
    let bb1 = BasicBlock::new(&mut ctx, None, vec![]);
    bb1.insert_at_back(region, &ctx);

    // bb2(p): the merge block; `p` is the bool phi.
    let bb2 = BasicBlock::new(&mut ctx, None, vec![bool_ty.into()]);
    bb2.insert_at_back(region, &ctx);
    let p = bb2.deref(&ctx).get_argument(0);

    // bb0: cond_br a, bb2(a), bb1(). On the true edge `a` is true, so
    // passing `a` itself is `a || b` without needing a constant.
    let (flat_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![a], vec![a], vec![]]);
    let cond_br = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat_operands,
        vec![bb2, bb1],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(cond_br, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    cond_br.insert_at_back(bb0, &ctx);

    // bb1: goto bb2(b).
    let goto = Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![b],
        vec![bb2],
        0,
    );
    goto.insert_at_back(bb1, &ctx);

    // bb2: p == q, then p < q.
    for info in [
        mir::MirEqOp::get_concrete_op_info(),
        mir::MirLtOp::get_concrete_op_info(),
    ] {
        let cmp = Operation::new(&mut ctx, info, vec![bool_ty.into()], vec![p, q], vec![], 0);
        cmp.insert_at_back(bb2, &ctx);
    }
    let ret_op = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op.insert_at_back(bb2, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    // Before the fallback, this failed with "expected IntegerType or
    // MirPtrType operand in arithmetic op".
    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut icmps = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != func_name {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if let Some(icmp) = Operation::get_op::<llvm::ICmpOp>(body_op, &ctx) {
                    let lhs_ty = body_op.deref(&ctx).get_operand(0).get_type(&ctx);
                    icmps.push((icmp.predicate(&ctx), lhs_ty));
                }
            }
        }
    }

    let i1: pliron::r#type::TypeHandle = bool_ty.into();
    assert_eq!(
        icmps,
        vec![(ICmpPredicateAttr::EQ, i1), (ICmpPredicateAttr::ULT, i1),],
        "bool-phi comparisons must lower to `icmp eq i1` and `icmp ult i1`"
    );
    Ok(())
}

// =============================================================================
// Scalar math lowering tests
// =============================================================================

/// Builds a module holding a single `mir.func` whose body calls the `bswap`
/// placeholder on a `u8` argument, with the call result carrying `result_ty`.
/// Shared by the valid (u8 result) and malformed (tuple result) 8-bit bswap
/// tests below.
fn build_bswap8_call_module(
    ctx: &mut Context,
    result_ty: pliron::r#type::TypeHandle,
) -> pliron::context::Ptr<Operation> {
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};

    let u8_ty = IntegerType::get(ctx, 8, Signedness::Unsigned);

    let module = ModuleOp::new(ctx, "m".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_block = module
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .unwrap();

    let func_ty = FunctionType::get(ctx, vec![u8_ty.into()], vec![]);
    let func_op = Operation::new(
        ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(ctx, func_op, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(ctx, "f".try_into().unwrap());
    {
        let region = func.get_operation().deref(ctx).get_region(0);
        let block = BasicBlock::new(ctx, None, vec![u8_ty.into()]);
        block.insert_at_back(region, ctx);
        let arg = block.deref(ctx).get_argument(0);

        let call_ptr = Operation::new(
            ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![result_ty],
            vec![arg],
            vec![],
            0,
        );
        let call = mir::MirCallOp::new(call_ptr);
        call.set_attr_callee(
            ctx,
            StringAttr::new(dialect_mir::rust_intrinsics::CALLEE_BSWAP.to_string()),
        );
        call_ptr.insert_at_back(block, ctx);

        let ret_op = Operation::new(
            ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, ctx);
    }
    func.get_operation().insert_at_back(module_block, ctx);

    module_ptr
}

/// An 8-bit `bswap` takes an early bitcast path, since Rust's semantics for a
/// single byte are identity and LLVM has no useful intrinsic for it. That path
/// returned before reaching the cast every other arm goes through, so a result
/// type that was not an 8-bit integer produced `bitcast i8 to <aggregate>`,
/// which LLVM rejects when the module is read back.
///
/// The known producer of this shape is an importer bug that hands the call
/// the type of the destination *local* rather than of the projected place it
/// writes, so `RET.1 = bswap(x)` on a `(f64, u8)` return arrives here
/// carrying the whole tuple. That bug is the importer's to fix, and whatever
/// the importer does, a malformed producer can still build such a `mir.call`.
/// Lowering cannot repair it, and refusing it is what keeps the emitted
/// module valid.
#[test]
fn bswap8_with_an_aggregate_result_is_refused_rather_than_bitcast() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirTupleType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let f64_ty = pliron::builtin::types::FP64Type::get(&ctx);
    // The tuple result is the whole point: an 8-bit operand with a result
    // the bitcast cannot legally produce.
    let tuple_ty = MirTupleType::get(&mut ctx, vec![f64_ty.into(), u8_ty.into()]);
    let module_ptr = build_bswap8_call_module(&mut ctx, tuple_ty.into());

    let result = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr);
    let error =
        result.expect_err("an 8-bit bswap with a tuple result must be refused, not lowered");
    assert!(
        error
            .to_string()
            .contains("expected integer type for Rust bit intrinsic"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// The valid counterpart: an 8-bit `bswap` whose result is a plain `u8` must
/// still take the identity-bitcast path and lower cleanly, so the guard above
/// cannot over-reject the case Rust actually produces for `u8::swap_bytes`.
#[test]
fn bswap8_with_a_u8_result_lowers_to_an_identity_bitcast() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let module_ptr = build_bswap8_call_module(&mut ctx, u8_ty.into());

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("an 8-bit bswap with a u8 result must lower");

    // The call must have become a bitcast, not an intrinsic call: there is no
    // llvm.bswap.i8.
    let mut found_bitcast = false;
    let module_block = module_ptr
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .iter(&ctx)
        .next()
        .unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                assert!(
                    Operation::get_op::<llvm::CallOp>(body_op, &ctx).is_none(),
                    "an 8-bit bswap must not lower to an intrinsic call"
                );
                if Operation::get_op::<llvm::BitcastOp>(body_op, &ctx).is_some() {
                    found_bitcast = true;
                }
            }
        }
    }
    assert!(
        found_bitcast,
        "an 8-bit bswap with a u8 result must lower to an identity bitcast"
    );
    Ok(())
}
