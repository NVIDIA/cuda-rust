/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Works out how a loop's counters and accumulators change each iteration.
//!
//! An **induction variable** (IV) is a value that changes by a fixed amount on
//! every trip through the loop: the `i` in `for i in 0..n` (it goes `0, 1, 2,
//! ...`), or one stepping by 4 to give `0, 4, 8, ...`. This analysis finds the
//! IVs of one loop and, when it can, how many times the loop runs.
//!
//! How the values flow. After mem2reg, a loop's per-iteration values live as the
//! header block's **block arguments** (mem2reg is the pass that turns memory
//! slots into SSA values; "block arguments" are pliron's way of passing values
//! into a block, the same role LLVM gives to phi nodes). Each predecessor branch
//! supplies the values for those arguments. So a header argument gets its value
//! from two edges: the **preheader edge** (the branch that enters the loop)
//! gives the starting value, and the **latch edge** (the branch that loops back)
//! gives the value to use next time round.
//!
//! This analysis looks at those two values for each header argument and labels
//! it one of:
//!
//!   * **Basic induction variable.** The latch feeds back `arg + c` for a
//!     constant `c` (the per-iteration step). With a constant starting value we
//!     describe it by a **recurrence** `{init, step}`, meaning value =
//!     `init + step * iteration_number`. For example `{0, 4}` is the sequence
//!     `0, 4, 8, 12, ...`. Such an IV is always a multiple of `step` away from
//!     `init`; that fact ("`arg` is **congruent** to `init` modulo `step`",
//!     i.e. `arg` and `init` leave the same remainder when divided by `step`) is
//!     what the unroller exploits when it folds `& mask` operations to
//!     constants.
//!   * **Reduction** (a loop-carried accumulator). A value carried across
//!     iterations and updated by something other than a constant step, e.g.
//!     `acc = acc + (i & 3)`. It is not a counter, so we cannot replace it with
//!     a formula; the unroller just threads it from one unrolled body copy to
//!     the next.
//!   * **Invariant.** Fed back unchanged, so it has the same value every
//!     iteration.
//!
//! The **trip count** is how many times the loop body runs. We read it off the
//! header's exit test `IV <pred> bound` (e.g. `i < 16`) when `init`, `step`, and
//! a constant `bound` are all known. For `i = 0; i < 16; i += 4` the trip count
//! is 4. The test may also add a constant to the counter first, as in
//! `i + 2 <= 16` ("a whole tile of two still fits"); that is the same as
//! `i <= 14` when the addition does not wrap.
//!
//! This is a small, reusable stand-in for full scalar evolution that the
//! unroller (and later loop passes) build on. It is deliberately cautious:
//! anything that does not match the simple counted-loop shape it recognises is
//! reported as `Unknown` / `None` rather than guessed at.

use dialect_mir::ops::arithmetic::{MirAddOp, MirNotOp, MirSubOp};
use dialect_mir::ops::comparison::{MirGeOp, MirGtOp, MirLeOp, MirLtOp};
use dialect_mir::ops::constants::MirConstantOp;
use dialect_mir::ops::control_flow::MirCondBranchOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::op_cast;
use pliron::operation::Operation;
use pliron::r#type::{Typed, TypedHandle};
use pliron::value::Value;
use rustc_hash::FxHashSet;

use crate::analyses::loop_info::{Loop, LoopId, LoopInfo};

/// A comparison operator, the `<pred>` in a test written `lhs <pred> rhs`:
/// less-than, less-or-equal, greater-than, greater-or-equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpPred {
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpPred {
    /// The opposite test: the one that is true exactly when this one is false.
    /// (`<` becomes `>=`, and so on.)
    fn negate(self) -> CmpPred {
        match self {
            CmpPred::Lt => CmpPred::Ge,
            CmpPred::Le => CmpPred::Gt,
            CmpPred::Gt => CmpPred::Le,
            CmpPred::Ge => CmpPred::Lt,
        }
    }
    /// The test you get by swapping the two sides while keeping the same
    /// meaning: `a < b` is the same fact as `b > a`, so `swap(<)` is `>`.
    fn swap(self) -> CmpPred {
        match self {
            CmpPred::Lt => CmpPred::Gt,
            CmpPred::Gt => CmpPred::Lt,
            CmpPred::Le => CmpPred::Ge,
            CmpPred::Ge => CmpPred::Le,
        }
    }
}

/// What one header block argument turned out to be: a counter, an accumulator,
/// a constant-across-iterations value, or something we don't recognise.
#[derive(Debug, Clone)]
pub enum ArgKind {
    /// A counter: its value is `init + step * iteration`, so it forms the
    /// sequence `init, init+step, init+2*step, ...`.
    BasicIv { init: i128, step: i128 },
    /// A value carried across iterations and updated by something other than a
    /// fixed step, e.g. an accumulator `acc = acc + (i & 3)`.
    Reduction,
    /// The same value on every iteration (fed back unchanged).
    Invariant,
    /// Did not match any of the patterns above.
    Unknown,
}

/// Everything this analysis learned about one loop.
#[derive(Debug, Clone)]
pub struct LoopRecurrences {
    /// What each header argument is, in header-argument order: `args[i]`
    /// describes header argument `i`.
    pub args: Vec<ArgKind>,
    /// Which header argument is the counter the loop tests against to decide
    /// whether to keep going (its index in `args`), if we found one.
    pub primary_iv: Option<usize>,
    /// The loop's limit as a plain number, when it is a compile-time constant.
    /// It is normalized so the body runs while `IV <continue_pred> bound`: for
    /// a test written `IV + iv_offset <pred> n` it is `n - iv_offset`.
    pub bound: Option<i128>,
    /// The value the exit test actually compares against, as written (`n`, not
    /// `n - iv_offset`). It can be a value only known at runtime (e.g. an array
    /// length), which is fine for partial unrolling, so we keep it here even
    /// when `bound` is `None`.
    pub bound_value: Option<Value>,
    /// The constant the exit test adds to the counter before comparing: the
    /// body runs while `IV + iv_offset <continue_pred> bound_value`. It is 0 for
    /// `i < n`, 2 for `i + 2 <= n`, and -1 for `i - 1 < n`.
    pub iv_offset: i128,
    /// The test that keeps the loop going: the body runs while
    /// `IV <continue_pred> bound` holds (e.g. `<` for `while i < n`).
    pub continue_pred: Option<CmpPred>,
    /// How many times the body runs, when `init`, `step`, `bound`, and the
    /// predicate are all known constants; `None` otherwise. This is the count
    /// over mathematical integers: consumers must prove that the counter and
    /// its exit-test offset do not wrap in the actual integer type.
    pub trip_count: Option<u64>,
}

/// If `v` is a compile-time integer constant (a `mir.constant` op), return its
/// mathematical value; `None` if `v` is computed at runtime or cannot fit in
/// the analysis's `i128` representation.
///
/// APInt's `to_i128` sign-extends narrow values, so unsigned and signless
/// constants must be zero-extended explicitly. Otherwise an unsigned `u8`
/// constant such as 200 would be mistaken for -56 and could produce a wrong
/// trip count.
pub(crate) fn const_i128(ctx: &Context, v: Value) -> Option<i128> {
    let def = v.defining_op()?;
    let c = Operation::get_op::<MirConstantOp>(def, ctx)?;
    let attr = c.get_attr_value(ctx)?;
    let ty = TypedHandle::<IntegerType>::from_handle(v.get_type(ctx), ctx).ok()?;
    if ty.deref(ctx).width() > 128 {
        return None;
    }
    match ty.deref(ctx).signedness() {
        Signedness::Signed => Some(attr.value().to_i128()),
        Signedness::Unsigned | Signedness::Signless => i128::try_from(attr.value().to_u128()).ok(),
    }
}

/// The values that the branch from `pred` to `header` supplies for `header`'s
/// block arguments. (`pred`'s last instruction is its branch; we find the slot
/// that targets `header` and read off the values passed along that edge.)
/// Returns `None` if `pred` does not actually branch to `header`.
pub(crate) fn edge_operands(
    ctx: &Context,
    pred: Ptr<BasicBlock>,
    header: Ptr<BasicBlock>,
) -> Option<Vec<Value>> {
    let term = pred.deref(ctx).get_terminator(ctx)?;
    let succs: Vec<Ptr<BasicBlock>> = term.deref(ctx).successors().collect();
    let idx = succs.iter().position(|&s| s == header)?;
    let opobj = Operation::get_op_dyn(term, ctx);
    let br = op_cast::<dyn BranchOpInterface>(opobj.as_ref())?;
    Some(br.successor_operands(ctx, idx))
}

/// Peel off any run of `mir.not` (boolean negation) wrapping `v`. Returns the
/// value underneath and whether the number peeled off was odd (so `!!x` reports
/// "not negated" and `!x` reports "negated"). Used so a guard written `!(i < n)`
/// is understood the same as `i >= n`.
fn unwrap_not(ctx: &Context, mut v: Value) -> (Value, bool) {
    let mut negated = false;
    while let Some(def) = v.defining_op() {
        if Operation::get_op::<MirNotOp>(def, ctx).is_some() {
            v = def.deref(ctx).get_operand(0);
            negated = !negated;
        } else {
            break;
        }
    }
    (v, negated)
}

/// If `op` is a comparison (`<`, `<=`, `>`, `>=`), return which one it is and
/// its two sides (left, right); `None` for any other op.
fn match_cmp(ctx: &Context, op: Ptr<Operation>) -> Option<(CmpPred, Value, Value)> {
    let pred = if Operation::get_op::<MirLtOp>(op, ctx).is_some() {
        CmpPred::Lt
    } else if Operation::get_op::<MirLeOp>(op, ctx).is_some() {
        CmpPred::Le
    } else if Operation::get_op::<MirGtOp>(op, ctx).is_some() {
        CmpPred::Gt
    } else if Operation::get_op::<MirGeOp>(op, ctx).is_some() {
        CmpPred::Ge
    } else {
        return None;
    };
    let o = op.deref(ctx);
    Some((pred, o.get_operand(0), o.get_operand(1)))
}

/// Run the full analysis on loop `id`: classify every header argument, find the
/// counter the loop tests, and compute the trip count when possible. `preheader`
/// is the block that enters the loop (see [`LoopInfo::preheader`]); we read each
/// counter's starting value off the branch from it.
///
/// [`LoopInfo::preheader`]: crate::analyses::loop_info::LoopInfo::preheader
pub fn analyze(
    ctx: &Context,
    info: &LoopInfo,
    id: LoopId,
    preheader: Ptr<BasicBlock>,
) -> LoopRecurrences {
    let l = &info.loops()[id];
    let header = l.header;
    let nargs = header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..nargs)
        .map(|i| header.deref(ctx).get_argument(i))
        .collect();

    // Starting values come in on the preheader edge. Next-iteration values may
    // come from several latches, or through a synthetic latch block whose block
    // arguments merge several old back-edges. Trace those block arguments to
    // their leaf incoming values and require every path to agree before calling
    // something an induction variable.
    let pre_ops = edge_operands(ctx, preheader, header);

    // Classify each header argument.
    let mut args = Vec::with_capacity(nargs);
    for (i, &arg) in header_args.iter().enumerate() {
        let recurrence_values = recurrence_values(ctx, l, header, arg, i);
        args.push(classify_arg(
            ctx,
            arg,
            i,
            pre_ops.as_deref(),
            recurrence_values.as_deref(),
        ));
    }

    // Read the header's exit test to find the counter it checks, the limit, and
    // the keep-going predicate.
    let guard = analyze_guard(ctx, info, id, &header_args, &args);
    let primary_iv = guard.as_ref().map(|g| g.iv);
    let bound = guard.as_ref().and_then(|g| g.bound);
    let bound_value = guard.as_ref().map(|g| g.bound_value);
    let iv_offset = guard.as_ref().map_or(0, |g| g.offset);
    let continue_pred = guard.as_ref().map(|g| g.pred);

    let trip_count = match (primary_iv, bound, continue_pred) {
        (Some(iv), Some(b), Some(p)) => match &args[iv] {
            ArgKind::BasicIv { init, step } => trip_count(*init, *step, b, p),
            _ => None,
        },
        _ => None,
    };

    LoopRecurrences {
        args,
        primary_iv,
        bound,
        bound_value,
        iv_offset,
        continue_pred,
        trip_count,
    }
}

/// Values that can arrive at header argument `arg_index` on any back-edge.
///
/// A unique-latch canonicalizer represents the merge as a latch block argument.
/// Looking only at the latch-to-header operand would therefore see a block
/// argument instead of `arg + step`. Follow in-loop block arguments backwards
/// until reaching the actual values supplied by the former latch edges.
fn recurrence_values(
    ctx: &Context,
    lp: &Loop,
    header: Ptr<BasicBlock>,
    arg: Value,
    arg_index: usize,
) -> Option<Vec<Value>> {
    let mut out = Vec::new();
    for edge_use in header.uses(ctx) {
        let term = edge_use.user_op();
        let source = term.deref(ctx).get_parent_block()?;
        if !lp.latches.contains(&source) {
            continue;
        }
        let opobj = Operation::get_op_dyn(term, ctx);
        let branch = op_cast::<dyn BranchOpInterface>(opobj.as_ref())?;
        let value = branch
            .successor_operands(ctx, edge_use.find_index(ctx))
            .get(arg_index)
            .copied()?;
        let mut visiting = FxHashSet::default();
        if !expand_block_argument(ctx, lp, header, arg, value, &mut visiting, &mut out) {
            return None;
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Recursively replace an in-loop block argument with every value passed to it.
/// `arg` itself is a leaf: following the header argument would walk around the
/// loop forever. `visiting` rejects any other block-argument cycle rather than
/// guessing at a recurrence.
fn expand_block_argument(
    ctx: &Context,
    lp: &Loop,
    header: Ptr<BasicBlock>,
    arg: Value,
    value: Value,
    visiting: &mut FxHashSet<Value>,
    out: &mut Vec<Value>,
) -> bool {
    if value == arg {
        out.push(value);
        return true;
    }
    let Some(block) = value.defining_block() else {
        out.push(value);
        return true;
    };
    if block == header || !lp.blocks.contains(&block) {
        out.push(value);
        return true;
    }
    if !visiting.insert(value) {
        return false;
    }

    let index = value.find_index(ctx);
    let incoming_edges = block.uses(ctx);
    if incoming_edges.is_empty() {
        visiting.remove(&value);
        return false;
    }
    for edge_use in incoming_edges {
        let term = edge_use.user_op();
        let Some(source) = term.deref(ctx).get_parent_block() else {
            visiting.remove(&value);
            return false;
        };
        if !lp.blocks.contains(&source) {
            visiting.remove(&value);
            return false;
        }
        let opobj = Operation::get_op_dyn(term, ctx);
        let Some(branch) = op_cast::<dyn BranchOpInterface>(opobj.as_ref()) else {
            visiting.remove(&value);
            return false;
        };
        let operands = branch.successor_operands(ctx, edge_use.find_index(ctx));
        let Some(&incoming) = operands.get(index) else {
            visiting.remove(&value);
            return false;
        };
        if !expand_block_argument(ctx, lp, header, arg, incoming, visiting, out) {
            visiting.remove(&value);
            return false;
        }
    }
    visiting.remove(&value);
    true
}

fn classify_arg(
    ctx: &Context,
    arg: Value,
    i: usize,
    pre_ops: Option<&[Value]>,
    recurrence_values: Option<&[Value]>,
) -> ArgKind {
    let values = match recurrence_values {
        Some(values) if !values.is_empty() => values,
        _ => return ArgKind::Unknown,
    };
    // Fed back unchanged on every path -> same value every iteration.
    if values.iter().all(|&value| value == arg) {
        return ArgKind::Invariant;
    }

    // Every back-edge must carry `arg + c`, `c + arg`, or `arg - c`, and every
    // path must agree on c. Choosing one arbitrary latch is unsound.
    let mut steps = values.iter().map(|&value| constant_offset(ctx, value, arg));
    let first_step = steps.next().flatten();
    if let Some(step) = first_step
        && steps.all(|candidate| candidate == Some(step))
    {
        if let Some(init) = pre_ops
            .and_then(|o| o.get(i).copied())
            .and_then(|v| const_i128(ctx, v))
        {
            return ArgKind::BasicIv { init, step };
        }
        // Steps like a counter, but its starting value isn't a constant, so we
        // can't give it a numeric formula; treat it as a carried value instead.
        return ArgKind::Reduction;
    }
    // Changes each iteration but not by a fixed step: an accumulator.
    ArgKind::Reduction
}

/// If `v` is `arg + c`, `c + arg`, or `arg - c` for a constant `c`, return how
/// far `v` is from `arg` (`c`, or `-c` for the subtraction). `None` otherwise.
/// On a latch edge this is the per-iteration step; in an exit test it is the
/// offset added to the counter before comparing.
fn constant_offset(ctx: &Context, v: Value, arg: Value) -> Option<i128> {
    let def = v.defining_op()?;
    if Operation::get_op::<MirAddOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if a == arg {
            return const_i128(ctx, b);
        }
        if b == arg {
            return const_i128(ctx, a);
        }
    } else if Operation::get_op::<MirSubOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if a == arg {
            return const_i128(ctx, b).and_then(i128::checked_neg);
        }
    }
    None
}

/// A header exit test the analysis understood: the body runs while
/// `header_args[iv] + offset <pred> bound_value`.
struct Guard {
    iv: usize,
    offset: i128,
    /// `bound_value - offset` when `bound_value` is a constant, so the body
    /// runs while `IV <pred> bound`.
    bound: Option<i128>,
    bound_value: Value,
    pred: CmpPred,
}

/// If `v` is `a +/- c` for a header argument `a` and constant `c`, return
/// the argument's index and the offset.
fn header_arg_plus_constant(
    ctx: &Context,
    v: Value,
    header_args: &[Value],
) -> Option<(usize, i128)> {
    header_args
        .iter()
        .enumerate()
        .find_map(|(i, &a)| constant_offset(ctx, v, a).map(|c| (i, c)))
}

/// Read the header's conditional branch (its `i < n`-style exit test) and pull
/// out which header argument is the counter being tested, the constant added to
/// it before the comparison, the limit it is compared against (as a constant
/// when possible, always as a value), and the predicate under which the body
/// keeps running. Returns `None` if the header doesn't have a recognisable
/// counted-loop test.
fn analyze_guard(
    ctx: &Context,
    info: &LoopInfo,
    id: LoopId,
    header_args: &[Value],
    args: &[ArgKind],
) -> Option<Guard> {
    let l = &info.loops()[id];
    let term = l.header.deref(ctx).get_terminator(ctx)?;
    // Successor position and operand 0 have true/false-condition meaning only
    // for `mir.cond_br`. Do not infer a guard from another two-way branch op that
    // happens to expose the same raw operation layout.
    Operation::get_op::<MirCondBranchOp>(term, ctx)?;
    let succs: Vec<Ptr<BasicBlock>> = term.deref(ctx).successors().collect();
    if succs.len() != 2 {
        return None;
    }
    // The header branches two ways: into the body, or out of the loop. Find
    // which of the two successors is the body (the one still inside the loop).
    let body_idx = if l.blocks.contains(&succs[0]) {
        0
    } else if l.blocks.contains(&succs[1]) {
        1
    } else {
        return None;
    };
    // The branch's first operand is the boolean condition; the body is the
    // successor taken when that condition is true (successor 0 is the true side).
    // Peel off any `!` so we compare against the real underlying test, and track
    // whether peeling flipped the sense.
    let cond = term.deref(ctx).get_operand(0);
    let (cmp_val, negated) = unwrap_not(ctx, cond);
    let body_when_cmp_true = (body_idx == 0) ^ negated;

    let (pred_written, lhs, rhs) = match_cmp(ctx, cmp_val.defining_op()?)?;
    // We want the predicate that is true when the body runs. If the body runs
    // when the comparison is true, that's the comparison itself; if it runs when
    // the comparison is false, flip the comparison to its opposite.
    let mut pred = if body_when_cmp_true {
        pred_written
    } else {
        pred_written.negate()
    };

    // The test could be written `i < n` or `n > i`. Figure out which side is the
    // counter (a header argument) and which is the limit, and if the counter is
    // on the right, swap the predicate so we always end up with `IV <pred> bound`.
    let iv_is_lhs = header_args.iter().position(|&a| a == lhs);
    let iv_is_rhs = header_args.iter().position(|&a| a == rhs);
    let (iv_index, offset, bound_val) = match (iv_is_lhs, iv_is_rhs) {
        (Some(idx), _) => (idx, 0, rhs),
        (None, Some(idx)) => {
            pred = pred.swap();
            (idx, 0, lhs)
        }
        // Neither side is a header argument itself. The test may still be
        // `i + c <pred> n` (or `n <pred> i + c`), with the counter plus a
        // constant on exactly one side.
        (None, None) => match (
            header_arg_plus_constant(ctx, lhs, header_args),
            header_arg_plus_constant(ctx, rhs, header_args),
        ) {
            (Some((idx, c)), None) => (idx, c, rhs),
            (None, Some((idx, c))) => {
                pred = pred.swap();
                (idx, c, lhs)
            }
            _ => return None,
        },
    };
    // The thing being tested must actually be a counter for this to be a
    // counted loop.
    if !matches!(args[iv_index], ArgKind::BasicIv { .. }) {
        return None;
    }
    // `IV + offset <pred> n` is `IV <pred> n - offset`. A limit the analysis
    // cannot represent is no recognizable test at all.
    let bound = match const_i128(ctx, bound_val) {
        Some(n) => Some(n.checked_sub(offset)?),
        None => None,
    };
    Some(Guard {
        iv: iv_index,
        offset,
        bound,
        bound_value: bound_val,
        pred,
    })
}

/// How many times the body runs for a loop that continues while
/// `IV <pred> bound`, given the counter's `init` and `step`. For example
/// `i = 0; i < 16; i += 4` gives 4. Returns `None` when the step points the
/// wrong way for the test (e.g. `i < n` while `i` decreases), which would be an
/// infinite or zero loop we don't count.
fn trip_count(init: i128, step: i128, bound: i128, pred: CmpPred) -> Option<u64> {
    let count = match pred {
        // Counting up toward an upper limit.
        CmpPred::Lt if step > 0 => div_ceil(bound.checked_sub(init)?, step)?,
        CmpPred::Le if step > 0 => div_ceil(bound.checked_sub(init)?.checked_add(1)?, step)?,
        // Counting down toward a lower limit.
        CmpPred::Gt if step < 0 => div_ceil(init.checked_sub(bound)?, step.checked_neg()?)?,
        CmpPred::Ge if step < 0 => div_ceil(
            init.checked_sub(bound)?.checked_add(1)?,
            step.checked_neg()?,
        )?,
        _ => return None,
    };
    u64::try_from(count).ok()
}

/// Divide and round up (so 5/4 is 2). Returns 0 when the numerator is zero or
/// negative, which corresponds to a loop that never runs. Returns `None` if
/// the host-side analysis arithmetic would overflow.
fn div_ceil(num: i128, den: i128) -> Option<i128> {
    if num <= 0 {
        Some(0)
    } else {
        Some(num.checked_add(den.checked_sub(1)?)? / den)
    }
}
