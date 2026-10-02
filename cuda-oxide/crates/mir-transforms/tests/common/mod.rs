/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helpers for building small `dialect-mir` CFGs in integration tests, so the
//! analyses (`LoopInfo`, `induction`) and the unroll pass can be exercised
//! without going through the whole rustc -> MIR pipeline.
//!
//! Each test binary (`loop_info`, `induction`, `unroll`) pulls this in via
//! `mod common;` and uses only the builders it needs, hence the crate-wide
//! `dead_code` allow.

#![allow(dead_code)]

use core::num::NonZero;

use dialect_mir::ops::{
    MirAddOp, MirCondBranchOp, MirConstantOp, MirFuncOp, MirGeOp, MirGotoOp, MirGtOp, MirLeOp,
    MirLtOp, MirNotOp, MirReturnOp, MirSubOp,
};
use mir_transforms::analyses::induction::CmpPred;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
use pliron::builtin::op_interfaces::{
    OperandSegmentInterface, SingleBlockRegionInterface, SymbolOpInterface,
};
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::region::Region;
use pliron::r#type::{TypeHandle, TypedHandle};
use pliron::utils::apint::APInt;
use pliron::value::Value;

/// A fresh context with the `mir` dialect registered (builtin is registered by
/// `Context::new`).
pub fn mir_ctx() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    ctx
}

/// A signless `i1` type.
pub fn i1(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 1, Signedness::Signless)
}

/// An unsigned 32-bit type (the IV type our kernels use).
pub fn u32t(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 32, Signedness::Unsigned)
}

/// A signed 32-bit type.
pub fn i32t(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 32, Signedness::Signed)
}

/// A signed 128-bit type, for constants at the edge of the analysis range.
pub fn i128t(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 128, Signedness::Signed)
}

/// Create `fn foo(inputs...) -> outputs...` inside a module and return
/// `(module_op, region)`. Blocks are appended to `region` by the caller; the
/// first block is the entry and must have the same argument types as `inputs`.
pub fn func(
    ctx: &mut Context,
    inputs: Vec<TypeHandle>,
    outputs: Vec<TypeHandle>,
) -> (Ptr<Operation>, Ptr<Region>) {
    let module = ModuleOp::new(ctx, "test".try_into().unwrap());
    let func_ty = FunctionType::get(ctx, inputs, outputs);
    let func_op = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = MirFuncOp::new(ctx, func_op, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(ctx, "foo".try_into().unwrap());
    module.append_operation(ctx, func_op, 0);
    let region = func_op.deref(ctx).get_region(0);
    (module.get_operation(), region)
}

/// Create an empty `fn foo()` inside a module and return `(module_op, region)`.
pub fn empty_func(ctx: &mut Context) -> (Ptr<Operation>, Ptr<Region>) {
    func(ctx, vec![], vec![])
}

/// Append an empty block (with the given argument types) to `region`.
pub fn block(ctx: &mut Context, region: Ptr<Region>, args: Vec<TypeHandle>) -> Ptr<BasicBlock> {
    let b = BasicBlock::new(ctx, None, args);
    b.insert_at_back(region, ctx);
    b
}

/// Append an integer constant to `b` and return its result value.
pub fn iconst(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    ty: TypedHandle<IntegerType>,
    val: i64,
) -> Value {
    let width = ty.deref(ctx).width() as usize;
    let apint = APInt::from_i64(val, NonZero::new(width).unwrap());
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, IntegerAttr::new(ty, apint));
    op.insert_at_back(b, ctx);
    op.deref(ctx).get_result(0)
}

/// Append an integer constant given as an `i128`, for types wider than 64 bits.
pub fn iconst_i128(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    ty: TypedHandle<IntegerType>,
    val: i128,
) -> Value {
    let width = ty.deref(ctx).width() as usize;
    let apint = APInt::from_i128(val, NonZero::new(width).unwrap());
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, IntegerAttr::new(ty, apint));
    op.insert_at_back(b, ctx);
    op.deref(ctx).get_result(0)
}

/// Append an unconditional `goto target(operands)` to `b`.
pub fn goto(ctx: &mut Context, b: Ptr<BasicBlock>, target: Ptr<BasicBlock>, operands: Vec<Value>) {
    let op = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![target],
        0,
    );
    op.insert_at_back(b, ctx);
}

/// Append `cond_br cond [true_succ(true_operands),
/// false_succ(false_operands)]` to `b`.
pub fn cond_br_args(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    cond: Value,
    true_succ: Ptr<BasicBlock>,
    true_operands: Vec<Value>,
    false_succ: Ptr<BasicBlock>,
    false_operands: Vec<Value>,
) {
    let (flat, segs) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond], true_operands, false_operands]);
    let op = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![true_succ, false_succ],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(op, ctx)
        .unwrap()
        .set_operand_segment_sizes(ctx, segs);
    op.insert_at_back(b, ctx);
}

/// Append `cond_br cond [true_succ, false_succ]` (no successor operands) to `b`.
pub fn cond_br(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    cond: Value,
    true_succ: Ptr<BasicBlock>,
    false_succ: Ptr<BasicBlock>,
) {
    cond_br_args(ctx, b, cond, true_succ, vec![], false_succ, vec![]);
}

/// Append `return values...` to `b`.
pub fn ret_values(ctx: &mut Context, b: Ptr<BasicBlock>, values: Vec<Value>) {
    let op = Operation::new(
        ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        values,
        vec![],
        0,
    );
    op.insert_at_back(b, ctx);
}

/// Append a `return` (no value) to `b`.
pub fn ret(ctx: &mut Context, b: Ptr<BasicBlock>) {
    ret_values(ctx, b, vec![]);
}

/// Append a two-operand op (built from `info`) of `result_ty` to `b`, returning
/// its result. A macro rather than a fn because `ConcreteOpInfo` is crate-private
/// in pliron, so it cannot be named in a parameter type here.
macro_rules! op2 {
    ($ctx:expr, $b:expr, $info:expr, $ty:expr, $lhs:expr, $rhs:expr) => {{
        let op = Operation::new($ctx, $info, vec![$ty], vec![$lhs, $rhs], vec![], 0);
        op.insert_at_back($b, $ctx);
        op.deref($ctx).get_result(0)
    }};
}

/// A built counted loop and the blocks worth asserting on.
pub struct CountedLoop {
    pub module: Ptr<Operation>,
    pub region: Ptr<Region>,
    pub preheader: Ptr<BasicBlock>,
    pub header: Ptr<BasicBlock>,
    pub latch: Ptr<BasicBlock>,
    pub exit: Ptr<BasicBlock>,
}

/// Build the canonical counted loop `while i < n { acc += i; i += 1 }`, in the
/// shape mem2reg leaves it (carried values are header block arguments, the exit
/// test is `not(i < n)`):
///
/// ```text
///   preheader:        acc0=0; i0=0;            goto header(acc0, i0)
///   header(acc, i):   nlt = not(i < n);        cond_br nlt [exit, latch]
///   latch:            acc1=acc+i; i1=i+1;      goto header(acc1, i1)
///   exit:             return
/// ```
pub fn counted_loop(ctx: &mut Context, n: i64) -> CountedLoop {
    counted_loop_from(ctx, 0, n)
}

/// Build the same unsigned counted loop with an explicit starting value. This
/// variant covers constants whose high bit is set without changing the common
/// zero-based test shape.
pub fn counted_loop_from(ctx: &mut Context, start: i64, n: i64) -> CountedLoop {
    counted_loop_from_step(ctx, start, n, 1)
}

/// Build the unsigned counted loop with explicit start, bound, and positive
/// step. This exposes wraparound cases without complicating ordinary tests.
pub fn counted_loop_from_step(ctx: &mut Context, start: i64, n: i64, step: i64) -> CountedLoop {
    let (module, region) = empty_func(ctx);
    let u32 = u32t(ctx);
    let i1 = i1(ctx);

    let preheader = block(ctx, region, vec![]);
    let header = block(ctx, region, vec![u32.into(), u32.into()]); // (acc, i)
    let latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![]);

    // preheader: acc0 = 0; i0 = start; goto header(acc0, i0)
    let acc0 = iconst(ctx, preheader, u32, 0);
    let i0 = iconst(ctx, preheader, u32, start);
    goto(ctx, preheader, header, vec![acc0, i0]);

    // header(acc, i): nlt = not(i < n); cond_br nlt [exit, latch]
    let acc = header.deref(ctx).get_argument(0);
    let i = header.deref(ctx).get_argument(1);
    let nconst = iconst(ctx, header, u32, n);
    let lt = op2!(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let nlt = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![lt],
            vec![],
            0,
        );
        op.insert_at_back(header, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br(ctx, header, nlt, exit, latch);

    // latch: acc1 = acc + i; i1 = i + step; goto header(acc1, i1)
    let acc1 = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        acc,
        i
    );
    let one = iconst(ctx, latch, u32, step);
    let inext = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        one
    );
    goto(ctx, latch, header, vec![acc1, inext]);

    // exit: return
    ret(ctx, exit);

    CountedLoop {
        module,
        region,
        preheader,
        header,
        latch,
        exit,
    }
}

/// Where an offset loop's limit `n` comes from.
#[derive(Debug, Clone, Copy)]
pub enum OffsetBound {
    /// A compile-time constant.
    Const(i128),
    /// The function's only argument, so the limit is known only at runtime.
    Param,
    /// The carried accumulator plus this constant, so both sides of the exit
    /// test are computed from header arguments.
    AccPlus(i128),
}

/// Build `while i + off <pred> n { acc += i; i += step }` in the shape mem2reg
/// leaves it. A negative `off` is written `i - |off|`:
///
/// ```text
///   preheader(n?):    acc0=0; i0=start;          goto header(acc0, i0)
///   header(acc, i):   v = i + off; t = not(v <pred> n); cond_br t [exit(acc), latch]
///   latch:            acc1=acc+i; i1=i+step;     goto header(acc1, i1)
///   exit(result):     return result
/// ```
///
/// All values have type `ty`. With [`OffsetBound::Param`] the function takes
/// `n` as its only argument.
pub fn offset_counted_loop(
    ctx: &mut Context,
    ty: TypedHandle<IntegerType>,
    start: i128,
    step: i128,
    off: i128,
    pred: CmpPred,
    bound: OffsetBound,
) -> CountedLoop {
    offset_loop(ctx, ty, start, step, off, pred, bound, false)
}

/// The same loop with the counter expression on the right of the exit test:
/// `while n <pred> i + off`. `pred` is the operator as written, so
/// `n >= i + off` passes [`CmpPred::Ge`].
pub fn offset_counted_loop_iv_on_right(
    ctx: &mut Context,
    ty: TypedHandle<IntegerType>,
    start: i128,
    step: i128,
    off: i128,
    pred: CmpPred,
    bound: OffsetBound,
) -> CountedLoop {
    offset_loop(ctx, ty, start, step, off, pred, bound, true)
}

#[allow(clippy::too_many_arguments)]
fn offset_loop(
    ctx: &mut Context,
    ty: TypedHandle<IntegerType>,
    start: i128,
    step: i128,
    off: i128,
    pred: CmpPred,
    bound: OffsetBound,
    iv_on_right: bool,
) -> CountedLoop {
    let i1 = i1(ctx);
    let inputs = match bound {
        OffsetBound::Param => vec![ty.into()],
        OffsetBound::Const(_) | OffsetBound::AccPlus(_) => vec![],
    };
    let (module, region) = func(ctx, inputs.clone(), vec![ty.into()]);

    let preheader = block(ctx, region, inputs);
    let header = block(ctx, region, vec![ty.into(), ty.into()]); // (acc, i)
    let latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![ty.into()]);

    let acc0 = iconst_i128(ctx, preheader, ty, 0);
    let i0 = iconst_i128(ctx, preheader, ty, start);
    goto(ctx, preheader, header, vec![acc0, i0]);

    let acc = header.deref(ctx).get_argument(0);
    let i = header.deref(ctx).get_argument(1);
    let tested = if off < 0 {
        let c = iconst_i128(ctx, header, ty, -off);
        op2!(
            ctx,
            header,
            MirSubOp::get_concrete_op_info(),
            ty.into(),
            i,
            c
        )
    } else {
        let c = iconst_i128(ctx, header, ty, off);
        op2!(
            ctx,
            header,
            MirAddOp::get_concrete_op_info(),
            ty.into(),
            i,
            c
        )
    };
    let n = match bound {
        OffsetBound::Const(n) => iconst_i128(ctx, header, ty, n),
        OffsetBound::Param => preheader.deref(ctx).get_argument(0),
        OffsetBound::AccPlus(c) => {
            let c = iconst_i128(ctx, header, ty, c);
            op2!(
                ctx,
                header,
                MirAddOp::get_concrete_op_info(),
                ty.into(),
                acc,
                c
            )
        }
    };
    let (lhs, rhs) = if iv_on_right {
        (n, tested)
    } else {
        (tested, n)
    };
    let info = match pred {
        CmpPred::Lt => MirLtOp::get_concrete_op_info(),
        CmpPred::Le => MirLeOp::get_concrete_op_info(),
        CmpPred::Gt => MirGtOp::get_concrete_op_info(),
        CmpPred::Ge => MirGeOp::get_concrete_op_info(),
    };
    let keep_going = op2!(ctx, header, info, i1.into(), lhs, rhs);
    let done = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![keep_going],
            vec![],
            0,
        );
        op.insert_at_back(header, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br_args(ctx, header, done, exit, vec![acc], latch, vec![]);

    let acc1 = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        ty.into(),
        acc,
        i
    );
    let step = iconst_i128(ctx, latch, ty, step);
    let inext = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        ty.into(),
        i,
        step
    );
    goto(ctx, latch, header, vec![acc1, inext]);

    let result = exit.deref(ctx).get_argument(0);
    ret_values(ctx, exit, vec![result]);

    CountedLoop {
        module,
        region,
        preheader,
        header,
        latch,
        exit,
    }
}

/// A built nested counted loop (outer `while i < n` containing inner
/// `while j < m`) and the blocks worth asserting on.
pub struct NestedLoop {
    pub module: Ptr<Operation>,
    pub region: Ptr<Region>,
    pub preheader: Ptr<BasicBlock>,
    pub outer_header: Ptr<BasicBlock>,
    pub outer_body: Ptr<BasicBlock>,
    pub inner_header: Ptr<BasicBlock>,
    pub inner_body: Ptr<BasicBlock>,
    pub outer_latch: Ptr<BasicBlock>,
    pub exit: Ptr<BasicBlock>,
}

/// Build `while i < n { while j < m { j += 1 } i += 1 }` in the shape mem2reg
/// leaves it (carried values are header block arguments, exit tests are
/// `not(_ < _)`):
///
/// ```text
///   preheader:        i0=0;             goto outer_header(i0)
///   outer_header(i):  nlt = not(i < n); cond_br nlt [exit, outer_body]
///   outer_body:       j0=0;             goto inner_header(j0)   // inner preheader
///   inner_header(j):  mlt = not(j < m); cond_br mlt [outer_latch, inner_body]
///   inner_body:       j1 = j+1;         goto inner_header(j1)
///   outer_latch:      i1 = i+1;         goto outer_header(i1)
///   exit:             return
/// ```
///
/// The outer loop *contains* the inner loop, so this is the shape the
/// nested-unroll path must handle: unrolling the outer clones the inner loop
/// wholesale (it stays a loop in each copy), it is never flattened.
pub fn nested_counted_loop(ctx: &mut Context, n: i64, m: i64) -> NestedLoop {
    let (module, region) = empty_func(ctx);
    let u32 = u32t(ctx);
    let i1 = i1(ctx);

    let preheader = block(ctx, region, vec![]);
    let outer_header = block(ctx, region, vec![u32.into()]); // (i)
    let outer_body = block(ctx, region, vec![]);
    let inner_header = block(ctx, region, vec![u32.into()]); // (j)
    let inner_body = block(ctx, region, vec![]);
    let outer_latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![]);

    let not = |ctx: &mut Context, b: Ptr<BasicBlock>, v: Value| -> Value {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![v],
            vec![],
            0,
        );
        op.insert_at_back(b, ctx);
        op.deref(ctx).get_result(0)
    };

    // preheader: i0 = 0; goto outer_header(i0)
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, outer_header, vec![i0]);

    // outer_header(i): nlt = not(i < n); cond_br nlt [exit, outer_body]
    let i = outer_header.deref(ctx).get_argument(0);
    let nconst = iconst(ctx, outer_header, u32, n);
    let lt = op2!(
        ctx,
        outer_header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let nlt = not(ctx, outer_header, lt);
    cond_br(ctx, outer_header, nlt, exit, outer_body);

    // outer_body: j0 = 0; goto inner_header(j0)
    let j0 = iconst(ctx, outer_body, u32, 0);
    goto(ctx, outer_body, inner_header, vec![j0]);

    // inner_header(j): mlt = not(j < m); cond_br mlt [outer_latch, inner_body]
    let j = inner_header.deref(ctx).get_argument(0);
    let mconst = iconst(ctx, inner_header, u32, m);
    let jlt = op2!(
        ctx,
        inner_header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        j,
        mconst
    );
    let jnlt = not(ctx, inner_header, jlt);
    cond_br(ctx, inner_header, jnlt, outer_latch, inner_body);

    // inner_body: j1 = j + 1; goto inner_header(j1)
    let one_j = iconst(ctx, inner_body, u32, 1);
    let j1 = op2!(
        ctx,
        inner_body,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        j,
        one_j
    );
    goto(ctx, inner_body, inner_header, vec![j1]);

    // outer_latch: i1 = i + 1; goto outer_header(i1)
    let one_i = iconst(ctx, outer_latch, u32, 1);
    let inext = op2!(
        ctx,
        outer_latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        one_i
    );
    goto(ctx, outer_latch, outer_header, vec![inext]);

    // exit: return
    ret(ctx, exit);

    NestedLoop {
        module,
        region,
        preheader,
        outer_header,
        outer_body,
        inner_header,
        inner_body,
        outer_latch,
        exit,
    }
}

/// A counted loop whose `continue` and normal paths are distinct back-edges.
pub struct MultiLatchLoop {
    pub module: Ptr<Operation>,
    pub region: Ptr<Region>,
    pub preheader: Ptr<BasicBlock>,
    pub header: Ptr<BasicBlock>,
    pub choose: Ptr<BasicBlock>,
    pub continue_latch: Ptr<BasicBlock>,
    pub normal_latch: Ptr<BasicBlock>,
    pub exit: Ptr<BasicBlock>,
}

/// Build this two-latch loop, returning the final accumulator:
///
/// ```text
///   preheader:          acc0=0; i0=0; goto header(acc0, i0)
///   header(acc, i):     if i >= n -> exit(acc), else -> choose
///   choose:             if i < 2 -> continue_latch, else -> normal_latch
///   continue_latch:     ic = i + continue_step; goto header(acc, ic)
///   normal_latch:       acc1 = acc+i; in = i + normal_step;
///                       goto header(acc1, in)
///   exit(result):       return result
/// ```
///
/// With both steps equal to one and `n=4`, this models a `continue` for `i=0,1`
/// and produces `2+3=5`. Giving the two latches different steps constructs the
/// unsafe non-affine recurrence that the unroller must reject.
pub fn multi_latch_counted_loop(
    ctx: &mut Context,
    n: i64,
    continue_step: i64,
    normal_step: i64,
) -> MultiLatchLoop {
    let u32 = u32t(ctx);
    let i1 = i1(ctx);
    let (module, region) = func(ctx, vec![], vec![u32.into()]);

    let preheader = block(ctx, region, vec![]);
    let header = block(ctx, region, vec![u32.into(), u32.into()]); // (acc, i)
    let choose = block(ctx, region, vec![]);
    let continue_latch = block(ctx, region, vec![]);
    let normal_latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![u32.into()]);

    let acc0 = iconst(ctx, preheader, u32, 0);
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, header, vec![acc0, i0]);

    let acc = header.deref(ctx).get_argument(0);
    let i = header.deref(ctx).get_argument(1);
    let nconst = iconst(ctx, header, u32, n);
    let lt_n = op2!(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let done = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![lt_n],
            vec![],
            0,
        );
        op.insert_at_back(header, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br_args(ctx, header, done, exit, vec![acc], choose, vec![]);

    let two = iconst(ctx, choose, u32, 2);
    let take_continue = op2!(
        ctx,
        choose,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        two
    );
    cond_br(ctx, choose, take_continue, continue_latch, normal_latch);

    let cstep = iconst(ctx, continue_latch, u32, continue_step);
    let ic = op2!(
        ctx,
        continue_latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        cstep
    );
    goto(ctx, continue_latch, header, vec![acc, ic]);

    let acc1 = op2!(
        ctx,
        normal_latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        acc,
        i
    );
    let nstep = iconst(ctx, normal_latch, u32, normal_step);
    let inext = op2!(
        ctx,
        normal_latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        nstep
    );
    goto(ctx, normal_latch, header, vec![acc1, inext]);

    let result = exit.deref(ctx).get_argument(0);
    ret_values(ctx, exit, vec![result]);

    MultiLatchLoop {
        module,
        region,
        preheader,
        header,
        choose,
        continue_latch,
        normal_latch,
        exit,
    }
}

/// A counted loop with one normal exit and one early `break` edge.
pub struct EarlyExitLoop {
    pub module: Ptr<Operation>,
    pub region: Ptr<Region>,
    pub preheader: Ptr<BasicBlock>,
    pub header: Ptr<BasicBlock>,
    pub body: Ptr<BasicBlock>,
    pub latch: Ptr<BasicBlock>,
    pub exit: Ptr<BasicBlock>,
}

/// Build `while i < n { acc += i; if i >= break_at { break } i += 1 }`.
/// Both the normal header exit and the early body exit pass the path-specific
/// accumulator into the shared exit block.
pub fn early_exit_counted_loop(ctx: &mut Context, n: i64, break_at: i64) -> EarlyExitLoop {
    let u32 = u32t(ctx);
    let i1 = i1(ctx);
    let (module, region) = func(ctx, vec![], vec![u32.into()]);

    let preheader = block(ctx, region, vec![]);
    let header = block(ctx, region, vec![u32.into(), u32.into()]); // (acc, i)
    let body = block(ctx, region, vec![]);
    let latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![u32.into()]);

    let acc0 = iconst(ctx, preheader, u32, 0);
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, header, vec![acc0, i0]);

    let acc = header.deref(ctx).get_argument(0);
    let i = header.deref(ctx).get_argument(1);
    let nconst = iconst(ctx, header, u32, n);
    let lt_n = op2!(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let done = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![lt_n],
            vec![],
            0,
        );
        op.insert_at_back(header, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br_args(ctx, header, done, exit, vec![acc], body, vec![]);

    let acc1 = op2!(
        ctx,
        body,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        acc,
        i
    );
    let break_const = iconst(ctx, body, u32, break_at);
    let before_break = op2!(
        ctx,
        body,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        break_const
    );
    // Continue while `i < break_at`; the false edge is the early `break`.
    cond_br_args(ctx, body, before_break, latch, vec![], exit, vec![acc1]);

    let one = iconst(ctx, latch, u32, 1);
    let inext = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        one
    );
    goto(ctx, latch, header, vec![acc1, inext]);

    let result = exit.deref(ctx).get_argument(0);
    ret_values(ctx, exit, vec![result]);

    EarlyExitLoop {
        module,
        region,
        preheader,
        header,
        body,
        latch,
        exit,
    }
}

/// Build the same early-exit loop, but let `exit` read the header accumulator
/// directly rather than receiving it as a block argument. This is valid before
/// unrolling because the header dominates the exit. With an early exit, however,
/// no single replacement value is correct for every cloned path, so v1 must skip.
pub fn early_exit_with_direct_liveout(ctx: &mut Context, n: i64) -> EarlyExitLoop {
    let u32 = u32t(ctx);
    let i1 = i1(ctx);
    let (module, region) = func(ctx, vec![], vec![u32.into()]);

    let preheader = block(ctx, region, vec![]);
    let header = block(ctx, region, vec![u32.into(), u32.into()]); // (acc, i)
    let body = block(ctx, region, vec![]);
    let latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![]);

    let acc0 = iconst(ctx, preheader, u32, 0);
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, header, vec![acc0, i0]);

    let acc = header.deref(ctx).get_argument(0);
    let i = header.deref(ctx).get_argument(1);
    let nconst = iconst(ctx, header, u32, n);
    let lt_n = op2!(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let done = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![lt_n],
            vec![],
            0,
        );
        op.insert_at_back(header, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br(ctx, header, done, exit, body);

    let acc1 = op2!(
        ctx,
        body,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        acc,
        i
    );
    let two = iconst(ctx, body, u32, 2);
    let before_break = op2!(
        ctx,
        body,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        two
    );
    let break_now = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![before_break],
            vec![],
            0,
        );
        op.insert_at_back(body, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br(ctx, body, break_now, exit, latch);

    let one = iconst(ctx, latch, u32, 1);
    let inext = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        one
    );
    goto(ctx, latch, header, vec![acc1, inext]);

    ret_values(ctx, exit, vec![acc]);

    EarlyExitLoop {
        module,
        region,
        preheader,
        header,
        body,
        latch,
        exit,
    }
}

/// A counted loop with two early-exit targets plus its normal exit.
pub struct MultipleExitLoop {
    pub module: Ptr<Operation>,
    pub region: Ptr<Region>,
    pub preheader: Ptr<BasicBlock>,
    pub header: Ptr<BasicBlock>,
    pub check_a: Ptr<BasicBlock>,
    pub check_b: Ptr<BasicBlock>,
    pub latch: Ptr<BasicBlock>,
    pub normal_exit: Ptr<BasicBlock>,
    pub exit_a: Ptr<BasicBlock>,
    pub exit_b: Ptr<BasicBlock>,
}

/// Build a counted loop with two runtime-controlled `break` targets. `flag_a`
/// and `flag_b` are function arguments, so SCCP cannot erase either path:
///
/// ```text
///   header(acc, i):  done -> normal_exit(acc), otherwise -> check_a
///   check_a:         acc1=acc+i; flag_a -> exit_a(acc1), else -> check_b
///   check_b:         flag_b -> exit_b(acc1), else -> latch
///   latch:           i1=i+1; goto header(acc1, i1)
/// ```
///
/// The exits return the normal value, `value+100`, and `value+200` respectively.
pub fn multiple_exit_counted_loop(ctx: &mut Context, n: i64) -> MultipleExitLoop {
    let u32 = u32t(ctx);
    let i1 = i1(ctx);
    let (module, region) = func(ctx, vec![i1.into(), i1.into()], vec![u32.into()]);

    let preheader = block(ctx, region, vec![i1.into(), i1.into()]);
    let header = block(ctx, region, vec![u32.into(), u32.into()]); // (acc, i)
    let check_a = block(ctx, region, vec![]);
    let check_b = block(ctx, region, vec![]);
    let latch = block(ctx, region, vec![]);
    let normal_exit = block(ctx, region, vec![u32.into()]);
    let exit_a = block(ctx, region, vec![u32.into()]);
    let exit_b = block(ctx, region, vec![u32.into()]);

    let flag_a = preheader.deref(ctx).get_argument(0);
    let flag_b = preheader.deref(ctx).get_argument(1);
    let acc0 = iconst(ctx, preheader, u32, 0);
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, header, vec![acc0, i0]);

    let acc = header.deref(ctx).get_argument(0);
    let i = header.deref(ctx).get_argument(1);
    let nconst = iconst(ctx, header, u32, n);
    let lt_n = op2!(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let done = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![lt_n],
            vec![],
            0,
        );
        op.insert_at_back(header, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br_args(ctx, header, done, normal_exit, vec![acc], check_a, vec![]);

    let acc1 = op2!(
        ctx,
        check_a,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        acc,
        i
    );
    cond_br_args(ctx, check_a, flag_a, exit_a, vec![acc1], check_b, vec![]);
    cond_br_args(ctx, check_b, flag_b, exit_b, vec![acc1], latch, vec![]);

    let one = iconst(ctx, latch, u32, 1);
    let inext = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        one
    );
    goto(ctx, latch, header, vec![acc1, inext]);

    let normal_value = normal_exit.deref(ctx).get_argument(0);
    ret_values(ctx, normal_exit, vec![normal_value]);

    let value_a = exit_a.deref(ctx).get_argument(0);
    let tag_a = iconst(ctx, exit_a, u32, 100);
    let tagged_a = op2!(
        ctx,
        exit_a,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        value_a,
        tag_a
    );
    ret_values(ctx, exit_a, vec![tagged_a]);

    let value_b = exit_b.deref(ctx).get_argument(0);
    let tag_b = iconst(ctx, exit_b, u32, 200);
    let tagged_b = op2!(
        ctx,
        exit_b,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        value_b,
        tag_b
    );
    ret_values(ctx, exit_b, vec![tagged_b]);

    MultipleExitLoop {
        module,
        region,
        preheader,
        header,
        check_a,
        check_b,
        latch,
        normal_exit,
        exit_a,
        exit_b,
    }
}
