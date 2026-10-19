/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Residual placement: where a goal that no JIT-stage proof source decided
//! becomes a device check, and in what instantiated form.
//!
//! When the check sits directly in a loop body whose every iteration runs
//! to completion (no `continue`), and the index is loop-invariant (or
//! affine in the unit-step induction variable, or statically ranged over
//! the values the loop actually attains), the whole comparison chain is
//! emitted in the loop's preheader instead of the hot body: the upper goal
//! takes the strongest extreme instance of the index, the lower goal the
//! weakest, and the pair is wrapped in an `upper <= lower || in_bounds`
//! guard so an empty loop — whose accesses never execute — cannot trap
//! spuriously. A hoisted check traps exactly when some iteration's access
//! would have, but it traps *before the loop*: the iterations preceding the
//! offending one (and their stores) no longer execute. Everything else
//! stays an in-place check at the access block.
//!
//! Either way the check is about to become a device instruction, so
//! placement is also where the `deny_in_kernel_checks` policy fires,
//! through the one shared gate ([`CUDATileFunctionCompiler::deny_residual_check`]).

use proc_macro2::Span;

use super::super::_function::CUDATileFunctionCompiler;
use super::super::_value::{CompilerContext, LoopFrame};
use super::super::shared_utils::TileBinaryOp;
use super::goals::AxisGoals;
use crate::compiler::_value::TileRustValue;
use crate::compiler::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::generics::GenericVars;

use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{BlockId, Module};

/// How the residual check's index is instantiated in the emission block.
pub(super) enum HoistIndex {
    /// A statically-known index range: `max` drives the upper check, `min`
    /// decides the lower goal (`0 <= index`).
    Const { min: i32, max: i32 },
    /// A loop-invariant runtime index: guarded directly, in the preheader.
    Invariant,
    /// `scale * induction + offset`: each goal substitutes the loop extreme
    /// that is worst for it (identity is scale 1, offset 0). Carries the
    /// bounds of the loop the affine was CLASSIFIED against (the innermost),
    /// because the outward walk may choose an outer frame as the emission
    /// target and the extremes must still be the inner loop's — substituting
    /// the target frame's bounds would test the wrong index instance
    /// (audit finding F1, 2026-08-08: latent today only because a bound that
    /// makes the inner loop known_non_empty also gives the iterand interval
    /// bounds, which classify as `Const` first; nothing enforces that
    /// coincidence).
    InductionAffine {
        scale: i64,
        offset: i64,
        lower: cutile_ir::ir::Value,
        upper: cutile_ir::ir::Value,
    },
}

/// Which extreme instance of an affine index over `[lower, upper)` a goal
/// substitutes: the strongest instance drives the upper goal, the weakest
/// the lower goal. For positive scale the strongest sits at `upper - 1`
/// and the weakest at `lower`; a negative scale mirrors them.
enum Extreme {
    Strongest,
    Weakest,
}

/// Placement may substitute a bound for the index only when the bound is
/// exactly representable in the index's runtime type: range analysis is
/// mathematical `i64`, the kernel computes wrapping `i32`, and a narrowing
/// `as i32` here turned a hoisted check into one that passes while the
/// actual wrapped indices run wild (issue #213). A range outside `i32` is
/// treated as no range at all — the check falls back to testing the raw
/// runtime value, which wraps exactly as the access does.
///
/// Since the 2026-08-12 review (S1), [`crate::value_facts::transfer`]
/// enforces this per op — an interval survives only if the op that made it
/// provably cannot wrap — so an `i32` value's range always fits here. The
/// gate stays as defense in depth for facts minted outside that path.
fn fits_i32(bounds: &crate::bounds::Bounds<i64>) -> bool {
    bounds.start >= i32::MIN as i64 && bounds.end <= i32::MAX as i64
}

impl<'m> CUDATileFunctionCompiler<'m> {
    /// Classify where the residual check for `goals` can be emitted:
    /// against the innermost loop first, then walking outward to the
    /// maximal preheader the check's operands dominate. `None` means the
    /// check stays in place; the second component records the
    /// innermost-classification failure for diagnostics.
    fn classify_hoist(
        &self,
        goals: &AxisGoals<'_>,
        dynamic_extent: Option<&TileRustValue>,
        block_id: BlockId,
        ctx: &CompilerContext,
    ) -> (Option<(LoopFrame, HoistIndex)>, Option<&'static str>) {
        let mut no_hoist_why: Option<&'static str> = None;
        let hoist = 'classify: {
            if !self.check_opts.hoist_to_preheaders {
                no_hoist_why = Some("preheader hoisting disabled by the check policy");
                break 'classify None;
            }
            let frames = &ctx.loop_frames;
            let Some(innermost) = frames.last() else {
                break 'classify None;
            };
            if block_id != innermost.body_block {
                no_hoist_why = Some("check sits inside a conditional block");
                break 'classify None;
            }
            // A body that can skip the rest of an iteration (`continue`) does
            // not execute the access on every iteration, so the extreme
            // instance a hoisted check would test may never be attained
            // (differential harness defect D2).
            if innermost.has_early_exit {
                no_hoist_why = Some("the loop body has an early exit (`continue`)");
                break 'classify None;
            }
            // Values the emitted check will reference; each must dominate
            // the target preheader (index() below the target watermark).
            let mut operand_deps: Vec<cutile_ir::ir::Value> = vec![];
            if let Some(shape_value) = dynamic_extent {
                let Some(value) = shape_value.value else {
                    no_hoist_why = Some("dynamic shape operand has no direct value");
                    break 'classify None;
                };
                operand_deps.push(value);
            }
            let kind = if let Some(bounds) = goals.index.bounds.filter(fits_i32) {
                HoistIndex::Const {
                    min: bounds.start as i32,
                    max: bounds.end as i32,
                }
            } else {
                let Some(value) = goals.index.value else {
                    no_hoist_why = Some("index has no direct value");
                    break 'classify None;
                };
                let affine = if innermost.induction_values.contains(&value) {
                    Some((value.index(), 1i64, 0i64))
                } else {
                    // Project the index's symbolic term to the single-var
                    // affine fragment `scale * iv + offset`, and only when
                    // that `iv` is the innermost loop's induction variable.
                    goals
                        .index
                        .term
                        .as_ref()
                        .and_then(|term| term.as_single_affine())
                        .and_then(|(atom, scale, offset)| match atom {
                            cuda_async::predicate::Atom::Iv(id)
                                if innermost.induction_values.iter().any(|v| v.index() == id) =>
                            {
                                Some((id, scale, offset))
                            }
                            _ => None,
                        })
                };
                if let Some((iv_id, scale, offset)) = affine {
                    if !innermost.unit_step {
                        no_hoist_why = Some("index depends on a non-unit-step induction variable");
                        break 'classify None;
                    }
                    if scale == 0 {
                        no_hoist_why = Some("degenerate affine index");
                        break 'classify None;
                    }
                    // When the loop's induction range is a compile-time
                    // constant, the index's range follows from its term
                    // (value_facts::term_range): discharge it as a static
                    // constant — the strongest instance is the range's max —
                    // instead of a runtime strongest-instance substitution.
                    let static_max = innermost.induction_range.and_then(|iv_range| {
                        crate::value_facts::term_range(
                            &cuda_async::predicate::Term::affine(
                                cuda_async::predicate::Atom::Iv(iv_id),
                                scale,
                                offset,
                            ),
                            &|atom| match atom {
                                cuda_async::predicate::Atom::Iv(id) if *id == iv_id => {
                                    Some(iv_range)
                                }
                                _ => None,
                            },
                        )
                    });
                    if let Some(range) = static_max.filter(fits_i32) {
                        HoistIndex::Const {
                            min: range.start as i32,
                            max: range.end as i32,
                        }
                    } else if !(scale == 1 && offset == 0) {
                        // A non-identity instance is emitted as wrapping i32
                        // multiply/add; without a static range proving the
                        // image fits, the substituted extreme can wrap and
                        // pass while the real indices are out of range
                        // (issue #213). Identity is exact: the instance IS
                        // the loop bound.
                        no_hoist_why = Some("non-identity affine instance could overflow i32");
                        break 'classify None;
                    } else {
                        // The substituted instances reference BOTH loop
                        // bounds: the strongest instance uses one extreme and
                        // the lower guard (always emitted for affine — the
                        // index carries no bounds here) uses the other. Both
                        // must dominate wherever the walk emits the check.
                        operand_deps.push(innermost.upper);
                        operand_deps.push(innermost.lower);
                        HoistIndex::InductionAffine {
                            scale,
                            offset,
                            lower: innermost.lower,
                            upper: innermost.upper,
                        }
                    }
                } else if value.index() < innermost.value_watermark {
                    operand_deps.push(value);
                    HoistIndex::Invariant
                } else {
                    no_hoist_why = Some("index is computed inside the loop body");
                    break 'classify None;
                }
            };
            // Walk outward: cross a loop only when it is directly nested
            // (its preheader is the enclosing body), statically non-empty
            // (its accesses execute on every enclosing iteration), the
            // enclosing body cannot skip it with an early exit, and every
            // operand dominates the enclosing preheader.
            let mut target = frames.len() - 1;
            while target > 0 {
                let inner = &frames[target];
                let outer = &frames[target - 1];
                let contiguous = inner.preheader_block == outer.body_block;
                let deps_dominate = operand_deps
                    .iter()
                    .all(|value| value.index() < outer.value_watermark);
                if contiguous && inner.known_non_empty && !outer.has_early_exit && deps_dominate {
                    target -= 1;
                } else {
                    break;
                }
            }
            Some((frames[target].clone(), kind))
        };
        (hoist, no_hoist_why)
    }

    /// One extreme instance of `scale * j + offset` over the frame's
    /// induction range `[lower, upper)`. The strongest and weakest builders
    /// are exact mirrors — the same op sequence at opposite loop extremes —
    /// so one parameterized emitter serves both goals.
    #[allow(clippy::too_many_arguments)]
    fn affine_extreme_instance(
        &self,
        module: &mut Module,
        check_block: BlockId,
        lower: cutile_ir::ir::Value,
        upper: cutile_ir::ir::Value,
        scale: i64,
        offset: i64,
        extreme: Extreme,
        index_ty: &TileRustType,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        span: &Span,
    ) -> Result<TileRustValue, JITError> {
        let at_last_iteration = (scale > 0) == matches!(extreme, Extreme::Strongest);
        let mut instance = if at_last_iteration {
            let upper_value = TileRustValue::new_primitive(upper, index_ty.clone(), None);
            let one = self.compile_constant(module, check_block, generic_vars, 1)?;
            self.compile_binary_op_from_values(
                module,
                check_block,
                upper_value,
                one,
                &TileBinaryOp::Sub,
                generic_vars,
                ctx,
                None,
                span,
            )?
        } else {
            TileRustValue::new_primitive(lower, index_ty.clone(), None)
        };
        if scale != 1 {
            let scale_value =
                self.compile_constant(module, check_block, generic_vars, scale as i32)?;
            instance = self.compile_binary_op_from_values(
                module,
                check_block,
                instance,
                scale_value,
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
        }
        if offset != 0 {
            let offset_value =
                self.compile_constant(module, check_block, generic_vars, offset as i32)?;
            instance = self.compile_binary_op_from_values(
                module,
                check_block,
                instance,
                offset_value,
                &TileBinaryOp::Add,
                generic_vars,
                ctx,
                None,
                span,
            )?;
        }
        Ok(instance)
    }

    /// Place the residual device check for one axis's goals: classify the
    /// hoist target, decide the lower goal statically when a known range
    /// allows it, apply the deny gate, then emit the comparison chain and
    /// assert at the chosen block.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place_residual_check(
        &self,
        module: &mut Module,
        block_id: BlockId,
        goals: &AxisGoals<'_>,
        dynamic_extent: Option<TileRustValue>,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        span: &Span,
    ) -> Result<(), JITError> {
        let axis = goals.axis;
        let (hoist, no_hoist_why) =
            self.classify_hoist(goals, dynamic_extent.as_ref(), block_id, ctx);
        let (check_block, guard_bounds, hoist_kind) = match &hoist {
            Some((frame, kind)) => (
                frame.preheader_block,
                // Vacuous-trip guard, elided when static bounds prove the
                // target loop runs at least once.
                (!frame.known_non_empty).then_some((frame.lower, frame.upper)),
                Some(kind),
            ),
            None => (block_id, None, None),
        };
        // The lower access goal `0 <= index` is shape-independent, so any
        // statically-known range decides it here: a provably negative block
        // index is rejected outright (mirroring the static-fold rung), and a
        // proven-nonnegative one discharges the lower guard. Only a genuinely
        // unknown index pays a runtime guard below. Previously the dynamic
        // path emitted ONLY the upper comparison, so a signed `-1` passed
        // `index < ceil(extent/tile)` and reached the access (2026-08-04
        // review, finding Ask3-1).
        // Without proof discharge nothing is inferred: the reference build
        // tests the actual value on BOTH sides, or it cannot referee the
        // inference that removed a side (2026-08-12 review, S2).
        let lower_static: Option<i64> = if !self.check_opts.discharge_proofs {
            None
        } else {
            goals
                .index
                .bounds
                .filter(fits_i32)
                .map(|b| b.start)
                .or(match &hoist_kind {
                    Some(HoistIndex::Const { min, .. }) => Some(*min as i64),
                    _ => None,
                })
        };
        if let Some(min) = lower_static {
            if min < 0 {
                return self.jit_error_result(
                    span,
                    &format!(
                        "partition access out of bounds: dim {axis}, block index can be {min} \
                         (0 <= index is required)"
                    ),
                );
            }
        }
        let needs_lower_guard = lower_static.is_none();
        // The check could not be discharged at compile time or hoisted to
        // launch, so it is about to be emitted *in* the kernel (either at
        // the access block or, hoisted, the loop preheader — both cost
        // device registers). The shared gate makes that a hard error under
        // `deny_in_kernel_checks`.
        let placement_detail = no_hoist_why
            .map(|why| format!(" ({why})"))
            .unwrap_or_default();
        // Name the two fixes concretely. The overwhelmingly common cause is an
        // index computed by hand — `n / TILE`, or a `Dim` built from one —
        // which is the right *number* carrying none of the provenance the
        // checker needs, so the remedy is to iterate the count the compiler
        // minted instead. The second form covers an index that legitimately
        // comes from another tensor.
        self.deny_residual_check(
            &format!("the bounds check for partition axis {axis}{placement_detail}"),
            &format!(
                "For a loop counter, iterate `0..num_tiles(&partition, {axis})`, whose \
                 result carries the axis it counts (a hand-computed `n / TILE` is the \
                 same number but proves nothing); for an index from a different tensor, \
                 relate the extents with `preconditions = (dim(a, i) == dim(b, j),)`; \
                 a tile-block id cannot be proven against a partition's tile count \
                 today, since the launch grid counts CTA slabs rather than tiles"
            ),
            span,
        )?;
        if hoist.is_some() {
            self.check_stats
                .hoisted
                .set(self.check_stats.hoisted.get() + 1);
        } else {
            self.check_stats
                .in_place
                .set(self.check_stats.in_place.get() + 1);
            if let Some(why) = no_hoist_why {
                if crate::cuda_tile_runtime_utils::jit_hoist_log_enabled() {
                    eprintln!(
                        "[cutile::jit] bounds check for dim {axis} stays in the loop body: {why}"
                    );
                }
            }
        }
        let tile_dim_value =
            self.compile_constant(module, check_block, generic_vars, goals.tile)?;
        // Upper goal operand: the strongest instance of the index.
        let index_instance = match hoist_kind {
            Some(HoistIndex::InductionAffine {
                scale,
                offset,
                lower,
                upper,
            }) => {
                // The extremes are the CLASSIFIED loop's bounds, carried in
                // the variant — never the walk target's (audit F1).
                self.affine_extreme_instance(
                    module,
                    check_block,
                    *lower,
                    *upper,
                    *scale,
                    *offset,
                    Extreme::Strongest,
                    &goals.index.ty,
                    generic_vars,
                    ctx,
                    span,
                )?
            }
            Some(HoistIndex::Const { max, .. }) => {
                self.compile_constant(module, check_block, generic_vars, *max)?
            }
            // An in-place check (and a hoisted loop-invariant one) tests the
            // index's ACTUAL value. A check stays in place precisely when the
            // access is conditional — under an `if`, after a `continue`, or
            // outside any loop with a merely-ranged index — so the range's
            // extreme may never be attained and substituting it (as this arm
            // once did) traps spuriously: `for k in 0..64 { if k >= limit
            // { continue; } p.load([k]) }` tested `63` against ten tiles.
            // Only a hoisted `Const` check may test the extreme, and
            // `classify_hoist` admits that only for a body every iteration
            // of which reaches the access.
            Some(HoistIndex::Invariant) | None => goals.index.clone(),
        };
        let shape_dim_value = match dynamic_extent {
            Some(shape_value) => shape_value,
            None => {
                let extent = goals
                    .static_extent
                    .expect("residual check without a dynamic extent has a static one");
                self.compile_constant(module, check_block, generic_vars, extent)?
            }
        };
        // Compute ceil_div(shape, tile) as `shape / tile + min(shape % tile, 1)`
        // (exact for `shape >= 0`, `tile >= 1`). Every intermediate is bounded
        // by `shape`, so the chain cannot wrap even for an extent near
        // `i32::MAX` — the former `(shape + tile - 1) / tile` did, turning the
        // check into `index < <negative>` (audit 2026-08). It also avoids the
        // `positive_inf` rounding mode, which can be misoptimized when the
        // dividend carries assume hints. A static extent folds the whole chain
        // to a constant.
        let quotient = self.compile_binary_op_from_values(
            module,
            check_block,
            shape_dim_value.clone(),
            tile_dim_value.clone(),
            &TileBinaryOp::Div,
            generic_vars,
            ctx,
            None,
            span,
        )?;
        let remainder = self.compile_binary_op_from_values(
            module,
            check_block,
            shape_dim_value.clone(),
            tile_dim_value,
            &TileBinaryOp::Rem,
            generic_vars,
            ctx,
            None,
            span,
        )?;
        let one = self.compile_constant(module, check_block, generic_vars, 1)?;
        let carry = self.compile_binary_op_from_values(
            module,
            check_block,
            remainder,
            one,
            &TileBinaryOp::Min,
            generic_vars,
            ctx,
            None,
            span,
        )?;
        let div_result_value = self.compile_binary_op_from_values(
            module,
            check_block,
            quotient,
            carry,
            &TileBinaryOp::Add,
            generic_vars,
            ctx,
            None,
            span,
        )?;
        let ineq_result_value = self.compile_binary_op_from_values(
            module,
            check_block,
            index_instance,
            div_result_value,
            &TileBinaryOp::Lt,
            generic_vars,
            ctx,
            None,
            span,
        )?;
        // Runtime lower guard, only when no static range decided the lower
        // goal above. For an affine hoisted index the weakest instance
        // mirrors the strongest one at the opposite loop extreme; otherwise
        // the raw (invariant or in-place) value is guarded directly.
        let ineq_result_value = if needs_lower_guard {
            let guard_operand = match hoist_kind {
                Some(HoistIndex::InductionAffine {
                    scale,
                    offset,
                    lower,
                    upper,
                }) => self.affine_extreme_instance(
                    module,
                    check_block,
                    *lower,
                    *upper,
                    *scale,
                    *offset,
                    Extreme::Weakest,
                    &goals.index.ty,
                    generic_vars,
                    ctx,
                    span,
                )?,
                _ => goals.index.clone(),
            };
            let zero = self.compile_constant(module, check_block, generic_vars, 0)?;
            let lower_ok = self.compile_binary_op_from_values(
                module,
                check_block,
                guard_operand,
                zero,
                &TileBinaryOp::Ge,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            self.compile_binary_op_from_values(
                module,
                check_block,
                lower_ok,
                ineq_result_value,
                &TileBinaryOp::BitAnd,
                generic_vars,
                ctx,
                None,
                span,
            )?
        } else {
            ineq_result_value
        };
        // Hoisted checks are guarded against vacuous loops: when
        // `upper <= lower` the body never runs, so no access exists to
        // be out of bounds.
        let checked_value = if let Some((lower, upper)) = guard_bounds {
            let upper_value =
                TileRustValue::new_primitive(upper, self.scalar_i32_type(span)?, None);
            let lower_value =
                TileRustValue::new_primitive(lower, self.scalar_i32_type(span)?, None);
            let vacuous = self.compile_binary_op_from_values(
                module,
                check_block,
                upper_value,
                lower_value,
                &TileBinaryOp::Le,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            self.compile_binary_op_from_values(
                module,
                check_block,
                vacuous,
                ineq_result_value,
                &TileBinaryOp::BitOr,
                generic_vars,
                ctx,
                None,
                span,
            )?
        } else {
            ineq_result_value
        };
        let result_value = checked_value
            .value
            .ok_or_else(|| self.jit_error(span, "failed to compile a binary expression operand"))?;
        let shape_desc = match goals.static_extent {
            Some(extent) => format!("{extent}"),
            None => "?".to_string(),
        };
        let suffix = if needs_lower_guard {
            " or index < 0"
        } else {
            ""
        };
        let message = format!(
            "partition access out of bounds: dim {axis}, block index >= ceil({shape_desc}/{})\
             {suffix}",
            goals.tile
        );
        let (assert_op_id, _) = OpBuilder::new(Opcode::Assert, self.ir_location(span))
            .attr("message", cutile_ir::ir::Attribute::String(message))
            .operand(result_value)
            .build(module);
        append_op(module, check_block, assert_op_id);
        Ok(())
    }
}
