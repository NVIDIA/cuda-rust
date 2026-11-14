/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single unified `dialect-mir` translator powered by [`SsaEnv`].
//!
//! Both pure scalar functions (`arithmetic.mir`) and verified GPU kernels
//! (`vecadd.mir`) walk their MIR instructions through this module using the
//! same typed lookup table (`HashMap<Value, TypedSsa>`):
//!
//! ```text
//!                     translate_function(ctx, mir_func)
//!                                    │
//!                       Builds `SsaEnv` for entry args
//!                                    │
//!             ┌──────────────────────┴──────────────────────┐
//!             ▼                                             ▼
//!     Scalar `mir.func`                         Verified Kernel `mir.func`
//!   • Binds args as `ScalarWord`              • Binds args as `BufferParam(i)`
//!   • Walks ops, checks `ScalarWord`,         • Walks ops, tracks `Index1D`,
//!     emits `let vN := .mul/.add ...`           `SliceLen`, `IndexLtLen`, `ElementPtr`
//!   • Returns `TranslatedFunction::Scalar`    • Returns `TranslatedFunction::Kernel`
//! ```

use std::collections::{BTreeSet, HashMap, HashSet};

use pliron::{
    attribute::AttributeDict,
    basic_block::BasicBlock,
    builtin::{
        attributes::{IdentifierAttr, IntegerAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, IntegerType},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::{Operation, verify_operation},
    r#type::{TypeHandle, Typed},
};

use crate::{
    ExportError, check_attributes, get_attribute, reject,
    spec::{
        KernelVerificationSpec, get_string_attr, has_verification_attributes,
        parse_kernel_verification_spec,
    },
    ssa::{
        GlobalStripKind, GuardExpr, IdxExpr, LoopScopeCheck, SsaEnv, SsaKind, TileAccessPatternKind,
    },
};

/// Direction of an extracted buffer access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessKind {
    /// Read from a slice or buffer parameter (`mir.load`).
    Read,
    /// Write to a mutable slice or `DisjointSlice` (`mir.store`).
    Write,
}

/// One guarded 1D buffer access translated from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExtractedAccess {
    /// Read or write operation.
    pub kind: AccessKind,
    /// Accessed kernel parameter index.
    pub target_param: usize,
    /// Kernel parameter index whose `.len()` bounds the dominating branch guard.
    pub guard_param: usize,
    /// Constant element offset added to the 1D thread index (`0` for `buf[idx]`).
    pub index_offset: u64,
}

/// One phased shared-memory tile or reduction access extracted from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhasedSharedAccess {
    /// Read or write operation.
    pub kind: AccessKind,
    /// Static `SharedArray` index (`0` for `TILE_A`, `1` for `TILE_B`, ...).
    pub shared_id: usize,
    /// Short name of the `SharedArray` static (`"TILE_A"`, `"WARP_SUMS"`, ...).
    pub shared_name: String,
    /// Declared element count of the `SharedArray`.
    pub buffer_len: u64,
    /// Barrier phase index (`0` before first `sync_threads()`, `1` after first, ...).
    pub phase: usize,
    /// Coordinate formula used to index the shared buffer.
    pub pattern: TileAccessPatternKind,
    /// Exclusive upper bound on loop index `i` (`1` outside a scan loop).
    pub loop_bound: u64,
    /// Whether this access is dominated by a leader guard (`lane == 0` / `tx % W == 0`).
    pub leader_only: bool,
    /// Dominating upper bound on `tx` (`block_size` when unguarded, or `N` under `if tid < N`).
    pub active_threads: u64,
    /// Optional residual branch guard (`GuardExpr::True` when unguarded or covered by `leader_only` / `active_threads`).
    pub guard: GuardExpr,
}

/// One 2D global slice strip access (`start + col` or `start + row * stride`)
/// extracted from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalStripAccess2D {
    /// Accessed kernel slice parameter index (`0..arity`).
    pub param: usize,
    /// Row-strip (`start + col`) vs. column-strip (`start + row * stride`) shape.
    pub kind: GlobalStripKind,
    /// Constant element offset added to the strip index (`0` when exact).
    pub offset: u64,
}

/// Output of the unified MIR translator before rendering to `.lean` text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslatedFunction {
    /// Pure single-block scalar function (`CudaOxide.Mir.Scalar.Expr`).
    Scalar {
        /// Bit width (`8`, `16`, `32`, or `64`).
        width: u32,
        /// Number of entry arguments.
        arity: usize,
        /// Ordered `let argN / vN := ...` lines.
        let_lines: Vec<String>,
        /// Final returned variable name (`"arg0"`, `"v1"`, ...).
        returned_var: String,
    },
    /// Verified 1D GPU kernel (`CudaOxide.Mir.Kernel.KernelSummary1D` + contracts).
    Kernel {
        /// Number of kernel parameters.
        arity: usize,
        /// Parsed `#[launch_contract]` and `#[verify(...)]` specification.
        spec: KernelVerificationSpec,
        /// Guarded buffer reads and writes extracted from the kernel CFG.
        accesses: Vec<ExtractedAccess>,
        /// `mir.assert` bounds-check operations proven redundant by `no_oob_idx`.
        proven_assert_ops: Vec<Ptr<Operation>>,
    },
    /// Verified 2D phased shared-memory GPU kernel (`TiledKernelSummary2D` + contracts).
    TiledKernel2D {
        /// Parsed `#[launch_contract]` and `#[verify(...)]` specification.
        spec: KernelVerificationSpec,
        /// Number of `sync_threads()` barriers in the tile loop.
        num_barriers: usize,
        /// Whether all barriers execute in thread-uniform control flow.
        barriers_uniform: bool,
        /// 2D global slice strip accesses extracted from the kernel CFG.
        global_strip_accesses: Vec<GlobalStripAccess2D>,
        /// Phased shared-memory reads and writes extracted from the kernel CFG.
        shared_accesses: Vec<PhasedSharedAccess>,
        /// `mir.assert` bounds-check operations proven redundant by `no_oob_idx`.
        proven_assert_ops: Vec<Ptr<Operation>>,
    },
}

/// Translate any supported `mir.func` (scalar function or verified GPU kernel)
/// using [`SsaEnv`].
pub fn translate_function(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    if has_verification_attributes(ctx, function) {
        translate_verified_kernel(ctx, function)
    } else {
        translate_scalar_function(ctx, function)
    }
}

/// Validate that `ty` is an unsigned integer type of width `8`, `16`, `32`, or `64`
/// and return its bit width.
fn unsigned_width(ctx: &Context, ty: TypeHandle) -> Result<u32, ExportError> {
    let type_ref = ty.deref(ctx);
    let int = type_ref.downcast_ref::<IntegerType>().ok_or_else(|| {
        reject(
            "function signature",
            "only unsigned integer types are supported",
        )
    })?;
    if !int.is_unsigned() || !matches!(int.width(), 8 | 16 | 32 | 64) {
        return Err(reject(
            "function signature",
            "expected an unsigned integer of width 8, 16, 32, or 64",
        ));
    }
    Ok(int.width())
}

/// Translate a single-block pure scalar integer `mir.func` into a
/// [`TranslatedFunction::Scalar`] `BitVec` AST using [`SsaEnv`].
fn translate_scalar_function(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    let func = function.deref(ctx);
    if func.num_regions() != 1
        || func.get_num_operands() != 0
        || func.get_num_results() != 0
        || func.get_num_successors() != 0
    {
        return Err(reject(
            "function",
            "expected one region and no operands, results, or successors",
        ));
    }
    check_attributes(
        &func.attributes,
        &["builtin_sym_name", "mir_func_type"],
        "function",
    )?;
    get_attribute::<IdentifierAttr>(&func.attributes, "builtin_sym_name", "function")?;
    let type_attr = get_attribute::<TypeAttr>(&func.attributes, "mir_func_type", "function")?;
    let fn_type = type_attr.get_type(ctx).deref(ctx);
    let fn_type = fn_type.downcast_ref::<FunctionType>().ok_or_else(|| {
        reject(
            "function",
            "mir_func_type must contain a builtin.function type",
        )
    })?;
    let inputs = fn_type.arg_types();
    let outputs = fn_type.res_types();
    if outputs.len() != 1 {
        return Err(reject("function signature", "expected exactly one result"));
    }
    let value_type = outputs[0];
    let width = unsigned_width(ctx, value_type)?;
    if inputs.iter().any(|ty| *ty != value_type) {
        return Err(reject(
            "function signature",
            "all arguments must have the same type as the result",
        ));
    }
    let arity = inputs.len();
    let region = func.get_region(0).deref(ctx);
    let blocks: Vec<_> = region.iter(ctx).collect();
    if blocks.len() != 1 {
        return Err(reject(
            "function",
            "expected exactly one nonempty basic block",
        ));
    }
    let block = blocks[0].deref(ctx);
    check_attributes(&block.attributes, &[], "entry block")?;
    if block.get_num_arguments() != arity {
        return Err(reject(
            "entry block",
            "argument count does not match function signature",
        ));
    }

    let mut env = SsaEnv::new();
    let mut let_lines = Vec::new();
    for (index, arg) in block.arguments().enumerate() {
        if arg.get_type(ctx) != value_type {
            return Err(reject(
                "entry block",
                "argument type does not match function signature",
            ));
        }
        let name = env.bind_arg(arg, index, SsaKind::ScalarWord { width });
        let_lines.push(format!(
            "  let {name} : CudaOxide.Mir.Scalar.Expr {width} {arity} := .arg ⟨{index}, by decide⟩"
        ));
    }

    let mut returned_var = None;
    for (index, op_ptr) in block.iter(ctx).enumerate() {
        let opid = Operation::get_opid(op_ptr, ctx).to_string();
        let scope = format!("operation {index} ({opid})");
        if returned_var.is_some() {
            return Err(reject(&scope, "operation after mir.return"));
        }
        let (expected_operands, expected_results, constant_key) = match opid.as_str() {
            "mir.add" | "mir.sub" | "mir.mul" => (2, 1, None),
            "mir.constant" => (0, 1, Some("value")),
            "builtin.constant" => (0, 1, Some("builtin_constant_value")),
            "mir.return" => (1, 0, None),
            _ => return Err(reject(&scope, "unsupported operation in scalar subset")),
        };
        let op = op_ptr.deref(ctx);
        if op.num_regions() != 0 || op.get_num_successors() != 0 {
            return Err(reject(&scope, "regions and successors are unsupported"));
        }
        if op.get_num_operands() != expected_operands || op.get_num_results() != expected_results {
            return Err(reject(&scope, "incorrect operand or result count"));
        }
        check_attributes(
            &op.attributes,
            &constant_key.into_iter().collect::<Vec<_>>(),
            &scope,
        )?;
        let mut operands = Vec::new();
        for operand in op.operands() {
            let binding = env.get(&operand, &scope)?;
            if binding.kind != (SsaKind::ScalarWord { width }) {
                return Err(reject(&scope, "operand type mismatch in scalar expression"));
            }
            operands.push(binding.lean_name.clone());
        }
        for result in op.results() {
            if result.get_type(ctx) != value_type {
                return Err(reject(
                    &scope,
                    "result type does not match function signature",
                ));
            }
        }
        if opid == "mir.return" {
            returned_var = Some(operands[0].clone());
            continue;
        }
        let expression = if let Some(key) = constant_key {
            let literal = get_attribute::<IntegerAttr>(&op.attributes, key, &scope)?;
            if TypeHandle::from(literal.get_type()) != value_type
                || literal.value().bw() != width as usize
            {
                return Err(reject(
                    &scope,
                    "constant type/bit width does not match its result",
                ));
            }
            format!(
                ".literal (BitVec.ofNat {width} {})",
                literal.value().to_string_unsigned_decimal()
            )
        } else {
            let constructor = match opid.as_str() {
                "mir.add" => "add",
                "mir.sub" => "sub",
                "mir.mul" => "mul",
                _ => return Err(reject(&scope, "unsupported arithmetic operation")),
            };
            format!(".{constructor} {} {}", operands[0], operands[1])
        };
        let name = env.bind_result(op.get_result(0), SsaKind::ScalarWord { width });
        let_lines.push(format!(
            "  let {name} : CudaOxide.Mir.Scalar.Expr {width} {arity} := {expression}"
        ));
    }
    let returned_var =
        returned_var.ok_or_else(|| reject("function", "missing mir.return terminator"))?;

    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;

    Ok(TranslatedFunction::Scalar {
        width,
        arity,
        let_lines,
        returned_var,
    })
}

/// Translate a `#[verify(...)]` GPU kernel `mir.func` at `PostMem2Reg` into a
/// 1D [`TranslatedFunction::Kernel`] or phased [`TranslatedFunction::TiledKernel2D`].
fn translate_verified_kernel(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;

    let (arity, spec) = parse_kernel_verification_spec(ctx, function)?;
    if spec.domain == 2
        || !spec.no_oob_shared.is_empty()
        || !spec.no_data_race.is_empty()
        || spec.uniform_barriers
    {
        return translate_verified_2d_tiled_kernel(ctx, function, spec);
    }
    let func = function.deref(ctx);
    let region = func.get_region(0).deref(ctx);
    let blocks: Vec<Ptr<BasicBlock>> = region.iter(ctx).collect();
    let Some(&entry_block) = blocks.first() else {
        return Err(reject("kernel", "empty kernel region"));
    };

    let mut env = SsaEnv::new();
    for (idx, arg) in entry_block.deref(ctx).arguments().enumerate() {
        env.bind_arg(arg, idx, SsaKind::BufferParam(idx));
    }

    let mut reachable: BTreeSet<usize> = BTreeSet::new();
    reachable.insert(0);
    let mut edge_guards: HashMap<(usize, usize), Option<usize>> = HashMap::new();
    let mut block_arg_incomings: HashMap<(usize, usize), SsaKind> = HashMap::new();

    for _pass in 0..blocks.len() + 2 {
        let prev_env_len = env.len();
        let prev_reachable_len = reachable.len();
        let prev_edges = edge_guards.clone();
        let prev_incomings = block_arg_incomings.clone();

        for (block_idx, &block_ptr) in blocks.iter().enumerate() {
            if !reachable.contains(&block_idx) {
                continue;
            }
            let current_guard = compute_block_guard(block_idx, &edge_guards);
            let block = block_ptr.deref(ctx);

            if block_idx != 0 && block.get_num_arguments() == 1 {
                let some_info = block_arg_incomings
                    .iter()
                    .filter(|((_, dst), _)| *dst == block_idx)
                    .find_map(|(_, incoming)| match *incoming {
                        SsaKind::OptionSome {
                            param,
                            offset,
                            guard_param,
                        } => Some((param, offset, guard_param)),
                        _ => None,
                    });
                if let Some((param, offset, guard_param)) = some_info {
                    env.bind_result(
                        block.get_argument(0),
                        SsaKind::OptionJoin {
                            param,
                            offset,
                            guard_param,
                        },
                    );
                }
            }

            let _ = translate_kernel_block_ops(
                ctx,
                block_idx,
                block_ptr,
                &blocks,
                arity,
                &spec,
                current_guard,
                &mut env,
                &mut reachable,
                &mut edge_guards,
                &mut block_arg_incomings,
                None,
                None,
            );
        }

        if env.len() == prev_env_len
            && reachable.len() == prev_reachable_len
            && edge_guards == prev_edges
            && block_arg_incomings == prev_incomings
        {
            break;
        }
    }

    let mut accesses = Vec::new();
    let mut proven_assert_ops = Vec::new();
    for (block_idx, &block_ptr) in blocks.iter().enumerate() {
        if !reachable.contains(&block_idx) {
            continue;
        }
        let block = block_ptr.deref(ctx);
        if block_idx != 0 && block.get_num_arguments() > 1 {
            return Err(reject(
                "kernel CFG",
                "only single-argument Option join blocks are supported in v0",
            ));
        }
        let current_guard = compute_block_guard(block_idx, &edge_guards);
        translate_kernel_block_ops(
            ctx,
            block_idx,
            block_ptr,
            &blocks,
            arity,
            &spec,
            current_guard,
            &mut env,
            &mut reachable,
            &mut edge_guards,
            &mut block_arg_incomings,
            Some(&mut accesses),
            Some(&mut proven_assert_ops),
        )?;
    }

    if !spec.writes_all_elems.is_empty() {
        let num_blocks = blocks.len();
        let mut succs_1d: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        let mut return_blocks_1d: Vec<usize> = Vec::new();
        let mut write_blocks_per_param: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut option_some_targets: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut direct_lt_targets: HashMap<usize, Vec<usize>> = HashMap::new();

        for (b_idx, &b_ptr) in blocks.iter().enumerate() {
            if !reachable.contains(&b_idx) {
                continue;
            }
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                let opid = Operation::get_opid(op_ptr, ctx).to_string();
                let op = op_ptr.deref(ctx);
                match opid.as_str() {
                    "mir.return" => return_blocks_1d.push(b_idx),
                    "mir.store" if op.get_num_operands() >= 1 => {
                        if let Some(SsaKind::ElementPtr { param, offset: 0 }) =
                            env.get_kind(&op.get_operand(0))
                        {
                            write_blocks_per_param.entry(param).or_default().push(b_idx);
                        }
                    }
                    "mir.goto" => {
                        if op.get_num_successors() == 1
                            && let Some(dst) = blocks.iter().position(|&b| b == op.get_successor(0))
                            && !succs_1d[b_idx].contains(&dst)
                        {
                            succs_1d[b_idx].push(dst);
                        }
                    }
                    "mir.cond_br" if op.get_num_operands() >= 1 && op.get_num_successors() == 2 => {
                        let t_opt = blocks.iter().position(|&b| b == op.get_successor(0));
                        let f_opt = blocks.iter().position(|&b| b == op.get_successor(1));
                        for dst in [t_opt, f_opt].into_iter().flatten() {
                            if !succs_1d[b_idx].contains(&dst) {
                                succs_1d[b_idx].push(dst);
                            }
                        }
                        match env.get_kind(&op.get_operand(0)) {
                            Some(SsaKind::OptionIsSome {
                                param,
                                offset: 0,
                                guard_param,
                            }) if param == guard_param => {
                                if let Some(t_idx) = t_opt {
                                    option_some_targets.entry(param).or_default().push(t_idx);
                                }
                            }
                            Some(SsaKind::IndexLtLen { guard_param }) => {
                                if let Some(t_idx) = t_opt {
                                    direct_lt_targets
                                        .entry(guard_param)
                                        .or_default()
                                        .push(t_idx);
                                }
                            }
                            Some(SsaKind::IndexGeLen { guard_param }) => {
                                if let Some(f_idx) = f_opt {
                                    direct_lt_targets
                                        .entry(guard_param)
                                        .or_default()
                                        .push(f_idx);
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }

        for &param in &spec.writes_all_elems {
            let entry_blocks = option_some_targets
                .get(&param)
                .or_else(|| direct_lt_targets.get(&param));
            let Some(entries) = entry_blocks.filter(|v| !v.is_empty()) else {
                return Err(reject(
                    "writes_all_elems",
                    format!(
                        "parameter {param} in `writes_all_elems` requires a direct `idx < param.len()` or `get_mut(idx)` guard"
                    ),
                ));
            };
            let write_blks = write_blocks_per_param
                .get(&param)
                .cloned()
                .unwrap_or_default();
            if write_blks.is_empty() {
                continue;
            }
            for &entry_blk in entries {
                let mut visited = vec![false; num_blocks];
                let mut stack = Vec::new();
                if !write_blks.contains(&entry_blk) {
                    visited[entry_blk] = true;
                    stack.push(entry_blk);
                }
                while let Some(curr) = stack.pop() {
                    if return_blocks_1d.contains(&curr) {
                        return Err(reject(
                            "writes_all_elems",
                            format!(
                                "parameter {param} is listed in `writes_all_elems`, but a conditional branch can skip writing it when `idx < len`"
                            ),
                        ));
                    }
                    for &next in &succs_1d[curr] {
                        if !write_blks.contains(&next) && !visited[next] {
                            visited[next] = true;
                            stack.push(next);
                        }
                    }
                }
            }
        }
    }

    Ok(TranslatedFunction::Kernel {
        arity,
        spec,
        accesses,
        proven_assert_ops,
    })
}

/// Extract the numeric index from a `mir.field_index` or enum-variant attribute
/// named `key_name` on `attrs`.
fn attribute_field_index(attrs: &AttributeDict, key_name: &str) -> Option<u32> {
    let text = attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == key_name)
        .map(|(_, v)| format!("{v:?}"))?;
    text.chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse::<u32>()
        .ok()
}

/// Compute the dominating `< param.len()` guard parameter index for `block_idx`
/// when every incoming CFG edge into `block_idx` carries the exact same guard.
fn compute_block_guard(
    block_idx: usize,
    edge_guards: &HashMap<(usize, usize), Option<usize>>,
) -> Option<usize> {
    if block_idx == 0 {
        return None;
    }
    let mut incoming = edge_guards
        .iter()
        .filter(|((_, dst), _)| *dst == block_idx)
        .map(|(_, guard)| *guard);
    let first = incoming.next()??;
    if incoming.all(|g| g == Some(first)) {
        Some(first)
    } else {
        None
    }
}

/// Walk the operations of a single basic block in a 1D verified kernel, updating
/// [`SsaEnv`], outgoing CFG edge guards, and (on the final pass) recording
/// guarded buffer accesses and proven `mir.assert` nodes.
#[allow(clippy::too_many_arguments)]
fn translate_kernel_block_ops(
    ctx: &Context,
    block_idx: usize,
    block_ptr: Ptr<BasicBlock>,
    blocks: &[Ptr<BasicBlock>],
    arity: usize,
    spec: &KernelVerificationSpec,
    current_guard: Option<usize>,
    env: &mut SsaEnv,
    reachable: &mut BTreeSet<usize>,
    edge_guards: &mut HashMap<(usize, usize), Option<usize>>,
    block_arg_incomings: &mut HashMap<(usize, usize), SsaKind>,
    mut record_accesses: Option<&mut Vec<ExtractedAccess>>,
    mut record_proven_asserts: Option<&mut Vec<Ptr<Operation>>>,
) -> Result<(), ExportError> {
    let block = block_ptr.deref(ctx);
    for op_ptr in block.iter(ctx) {
        let opid = Operation::get_opid(op_ptr, ctx).to_string();
        let op = op_ptr.deref(ctx);
        match opid.as_str() {
            "mir.storage_live" | "mir.storage_dead" | "mir.dbg_value" | "mir.dbg_declare" => {}
            "mir.float_constant" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            _ if opid.starts_with("nvvm.shfl_sync_") => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            "mir.alloca" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::LaunchContextAlloca);
                }
            }
            "mir.cast" => {
                if op.get_num_operands() == 1
                    && op.get_num_results() == 1
                    && let Some(kind) = env.get_kind(&op.get_operand(0))
                {
                    let src_ty = op.get_operand(0).get_type(ctx).deref(ctx);
                    let dst_ty = op.get_result(0).get_type(ctx).deref(ctx);
                    let mapped = match (
                        src_ty.downcast_ref::<IntegerType>(),
                        dst_ty.downcast_ref::<IntegerType>(),
                    ) {
                        (Some(si), Some(di)) => {
                            if si.is_unsigned() && di.is_unsigned() {
                                if di.width() >= si.width() {
                                    kind
                                } else if let SsaKind::ConstInt { value, .. } = kind {
                                    let mask = if di.width() >= 128 {
                                        u128::MAX
                                    } else {
                                        (1u128 << di.width()) - 1
                                    };
                                    SsaKind::ConstInt {
                                        width: di.width(),
                                        value: value & mask,
                                    }
                                } else {
                                    SsaKind::DataValue
                                }
                            } else {
                                SsaKind::DataValue
                            }
                        }
                        (None, None) => kind,
                        _ => SsaKind::DataValue,
                    };
                    env.bind_result(op.get_result(0), mapped);
                }
            }
            "mir.call" => {
                let callee = get_string_attr(&op.attributes, "callee").unwrap_or_default();
                if callee.contains("index_1d") && op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::ThreadIndexStruct);
                } else if (callee.contains("warp__reduce_") || callee.contains("warp__shuffle_"))
                    && op.get_num_results() == 1
                {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                } else {
                    return Err(reject(
                        "mir.call",
                        format!("unsupported call `{callee}` in verified kernel"),
                    ));
                }
            }
            "mir.extract_field" => {
                let base = env.get_kind(&op.get_operand(0));
                let field = attribute_field_index(&op.attributes, "index");
                let mapped = match (base, field) {
                    (Some(SsaKind::ThreadIndexStruct), Some(0)) => {
                        Some(SsaKind::Index1D { offset: 0 })
                    }
                    (Some(SsaKind::CheckedIndex1D { offset }), Some(0)) => {
                        Some(SsaKind::Index1D { offset })
                    }
                    (Some(SsaKind::CheckedIndex1D { .. }), Some(1)) => Some(SsaKind::DataValue),
                    (Some(SsaKind::BufferParam(p)), Some(0)) if p < arity => {
                        Some(SsaKind::SlicePtr(p))
                    }
                    (Some(SsaKind::BufferParam(p)), Some(1)) if p < arity => {
                        Some(SsaKind::SliceLen(p))
                    }
                    _ => None,
                };
                if let Some(kind) = mapped {
                    env.bind_result(op.get_result(0), kind);
                } else {
                    return Err(reject(
                        "mir.extract_field",
                        "unsupported field extraction in verified kernel",
                    ));
                }
            }
            "mir.constant" | "builtin.constant" => {
                let key = if opid == "mir.constant" {
                    "value"
                } else {
                    "builtin_constant_value"
                };
                if let Some(int_attr) = op
                    .attributes
                    .0
                    .iter()
                    .find(|(k, _)| k.to_string() == key)
                    .and_then(|(_, v)| v.downcast_ref::<IntegerAttr>())
                {
                    let value = int_attr
                        .value()
                        .to_string_unsigned_decimal()
                        .parse::<u128>()
                        .unwrap_or(0);
                    let width = u32::try_from(int_attr.value().bw()).unwrap_or(0);
                    env.bind_result(op.get_result(0), SsaKind::ConstInt { width, value });
                } else {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            "mir.checked_add" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                match (lhs, rhs) {
                    (
                        Some(SsaKind::Index1D { offset }),
                        Some(SsaKind::ConstInt { value: c, .. }),
                    )
                    | (
                        Some(SsaKind::ConstInt { value: c, .. }),
                        Some(SsaKind::Index1D { offset }),
                    ) => {
                        let new_offset =
                            offset.saturating_add(u64::try_from(c).unwrap_or(u64::MAX));
                        env.bind_result(
                            op.get_result(0),
                            SsaKind::CheckedIndex1D { offset: new_offset },
                        );
                    }
                    _ => {
                        return Err(reject(
                            "mir.checked_add",
                            "unsupported checked_add operands in index expression",
                        ));
                    }
                }
            }
            "mir.add" | "mir.sub" | "mir.mul" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                if opid == "mir.add" {
                    match (&lhs, &rhs) {
                        (
                            Some(SsaKind::Index1D { offset }),
                            Some(SsaKind::ConstInt { value: c, .. }),
                        )
                        | (
                            Some(SsaKind::ConstInt { value: c, .. }),
                            Some(SsaKind::Index1D { offset }),
                        ) => {
                            let new_offset =
                                offset.saturating_add(u64::try_from(*c).unwrap_or(u64::MAX));
                            env.bind_result(
                                op.get_result(0),
                                SsaKind::Index1D { offset: new_offset },
                            );
                            continue;
                        }
                        _ => {}
                    }
                }
                env.bind_result(op.get_result(0), SsaKind::DataValue);
            }
            "mir.eq" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                let mapped = match (lhs, rhs) {
                    (
                        Some(SsaKind::Index1D { offset: 0 }),
                        Some(SsaKind::ConstInt { value: u64_max, .. }),
                    ) if u64_max == u128::from(u64::MAX) => SsaKind::IsSentinel,
                    (
                        Some(SsaKind::OptionDiscriminant {
                            param,
                            offset,
                            guard_param,
                        }),
                        Some(SsaKind::ConstInt { value: 1, .. }),
                    ) => SsaKind::OptionIsSome {
                        param,
                        offset,
                        guard_param,
                    },
                    (
                        Some(SsaKind::OptionDiscriminant { .. }),
                        Some(SsaKind::ConstInt { value: 0, .. }),
                    ) => SsaKind::OptionIsNone,
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.lt" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                let mapped = match (lhs, rhs) {
                    (
                        Some(SsaKind::Index1D { offset: 0 }),
                        Some(SsaKind::SliceLen(guard_param)),
                    ) => SsaKind::IndexLtLen { guard_param },
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.not" => {
                let inner = env.get_kind(&op.get_operand(0));
                let mapped = match inner {
                    Some(SsaKind::IndexLtLen { guard_param }) => {
                        SsaKind::IndexGeLen { guard_param }
                    }
                    Some(SsaKind::IndexGeLen { guard_param }) => {
                        SsaKind::IndexLtLen { guard_param }
                    }
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.ptr_offset" => {
                let base = env.get_kind(&op.get_operand(0));
                let idx = env.get_kind(&op.get_operand(1));
                match (base, idx) {
                    (Some(SsaKind::SlicePtr(param)), Some(SsaKind::Index1D { offset })) => {
                        env.bind_result(op.get_result(0), SsaKind::ElementPtr { param, offset });
                    }
                    _ => {
                        return Err(reject(
                            "mir.ptr_offset",
                            "pointer offset must index a kernel slice parameter by the 1D thread index",
                        ));
                    }
                }
            }
            "mir.construct_enum" => {
                let variant = attribute_field_index(&op.attributes, "construct_enum_variant_index");
                match variant {
                    Some(0) => {
                        env.bind_result(op.get_result(0), SsaKind::OptionNone);
                    }
                    Some(1) if op.get_num_operands() == 1 => {
                        let Some(SsaKind::ElementPtr { param, offset }) =
                            env.get_kind(&op.get_operand(0))
                        else {
                            return Err(reject(
                                "mir.construct_enum",
                                "Option::Some payload must be a slice element pointer",
                            ));
                        };
                        let Some(guard_param) = current_guard else {
                            return Err(reject(
                                "mir.construct_enum",
                                "Option::Some slice element pointer constructed without a dominating bounds guard",
                            ));
                        };
                        env.bind_result(
                            op.get_result(0),
                            SsaKind::OptionSome {
                                param,
                                offset,
                                guard_param,
                            },
                        );
                    }
                    _ => {
                        return Err(reject(
                            "mir.construct_enum",
                            "unsupported enum construction in verified kernel",
                        ));
                    }
                }
            }
            "mir.get_discriminant" => {
                if let Some(SsaKind::OptionJoin {
                    param,
                    offset,
                    guard_param,
                }) = env.get_kind(&op.get_operand(0))
                {
                    env.bind_result(
                        op.get_result(0),
                        SsaKind::OptionDiscriminant {
                            param,
                            offset,
                            guard_param,
                        },
                    );
                } else {
                    return Err(reject(
                        "mir.get_discriminant",
                        "unsupported discriminant operand in verified kernel",
                    ));
                }
            }
            "mir.enum_payload" => {
                if let Some(SsaKind::OptionJoin { param, offset, .. }) =
                    env.get_kind(&op.get_operand(0))
                {
                    env.bind_result(op.get_result(0), SsaKind::ElementPtr { param, offset });
                } else {
                    return Err(reject(
                        "mir.enum_payload",
                        "unsupported enum payload extraction in verified kernel",
                    ));
                }
            }
            "mir.assert" => {
                if let Some(proven_asserts) = record_proven_asserts.as_deref_mut()
                    && current_guard.is_some()
                    && op.get_num_operands() == 1
                    && let Some(SsaKind::IndexLtLen {
                        guard_param: checked_param,
                    }) = env.get_kind(&op.get_operand(0))
                    && spec.no_oob_idx.contains(&checked_param)
                {
                    proven_asserts.push(op_ptr);
                }
            }
            "mir.load" => {
                let Some(SsaKind::ElementPtr { param, offset }) = env.get_kind(&op.get_operand(0))
                else {
                    return Err(reject(
                        "mir.load",
                        "memory load does not target a recognized slice element pointer",
                    ));
                };
                let Some(guard_param) = current_guard else {
                    return Err(reject(
                        "mir.load",
                        format!(
                            "unguarded read from parameter {param}: every buffer read must be dominated by a strict `< len` guard"
                        ),
                    ));
                };
                if let Some(accesses) = record_accesses.as_deref_mut() {
                    accesses.push(ExtractedAccess {
                        kind: AccessKind::Read,
                        target_param: param,
                        guard_param,
                        index_offset: offset,
                    });
                }
                env.bind_result(op.get_result(0), SsaKind::DataValue);
            }
            "mir.store" => {
                let Some(SsaKind::ElementPtr { param, offset }) = env.get_kind(&op.get_operand(0))
                else {
                    return Err(reject(
                        "mir.store",
                        "memory store does not target a recognized slice element pointer",
                    ));
                };
                let Some(guard_param) = current_guard else {
                    return Err(reject(
                        "mir.store",
                        format!(
                            "unguarded write to parameter {param}: every buffer write must be dominated by a strict `< len` guard"
                        ),
                    ));
                };
                if let Some(accesses) = record_accesses.as_deref_mut() {
                    accesses.push(ExtractedAccess {
                        kind: AccessKind::Write,
                        target_param: param,
                        guard_param,
                        index_offset: offset,
                    });
                }
            }
            "mir.goto" => {
                let target = op.get_successor(0);
                if let Some(dst_idx) = blocks.iter().position(|&b| b == target) {
                    reachable.insert(dst_idx);
                    edge_guards.insert((block_idx, dst_idx), current_guard);
                    if op.get_num_operands() == 1
                        && let Some(kind) = env.get_kind(&op.get_operand(0))
                    {
                        block_arg_incomings.insert((block_idx, dst_idx), kind);
                    }
                }
            }
            "mir.cond_br" => {
                let cond = env.get_kind(&op.get_operand(0));
                let true_succ = op.get_successor(0);
                let false_succ = op.get_successor(1);
                let (true_guard, false_guard) = match cond {
                    Some(SsaKind::IndexLtLen { guard_param }) => (Some(guard_param), current_guard),
                    Some(SsaKind::IndexGeLen { guard_param }) => (current_guard, Some(guard_param)),
                    Some(SsaKind::OptionIsSome { guard_param, .. }) => {
                        (Some(guard_param), current_guard)
                    }
                    _ => (current_guard, current_guard),
                };
                if let Some(dst_idx) = blocks.iter().position(|&b| b == true_succ) {
                    reachable.insert(dst_idx);
                    edge_guards.insert((block_idx, dst_idx), true_guard);
                }
                if let Some(dst_idx) = blocks.iter().position(|&b| b == false_succ) {
                    reachable.insert(dst_idx);
                    edge_guards.insert((block_idx, dst_idx), false_guard);
                }
            }
            "mir.return" | "mir.unreachable" => {}
            _ => {
                return Err(reject(
                    &opid,
                    format!("unsupported operation `{opid}` in verified kernel subset"),
                ));
            }
        }
    }
    Ok(())
}

/// Extract a signed/unsigned integer attribute by key as `u64`.
fn int_attr_u64(attrs: &AttributeDict, key: &str) -> Option<u64> {
    attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == key)
        .and_then(|(_, v)| v.downcast_ref::<IntegerAttr>())
        .and_then(|attr| {
            attr.value()
                .to_string_unsigned_decimal()
                .parse::<u64>()
                .ok()
        })
}

/// Extract a compositional [`IdxExpr`] from an [`SsaKind`] when the operand is
/// an index expression or an integer constant literal.
fn as_idx_expr(kind: &SsaKind) -> Option<IdxExpr> {
    match kind {
        SsaKind::Index(expr) => Some(expr.clone()),
        SsaKind::ConstInt { value, .. } => u64::try_from(*value).ok().map(IdxExpr::Const),
        _ => None,
    }
}

/// Compute the byte size of one element in a `mir.shared_alloc` operation from its
/// `elem_type` attribute (`0` if `elem_type` is absent or unrecognized).
fn shared_alloc_elem_bytes(ctx: &Context, op: &Operation) -> u64 {
    use pliron::printable::Printable;
    op.attributes
        .0
        .iter()
        .find(|(k, _)| k.to_string() == "elem_type")
        .and_then(|(_, v)| v.downcast_ref::<TypeAttr>())
        .map(|ta| {
            let ty_handle = ta.get_type(ctx);
            let ty_ref = ty_handle.deref(ctx);
            if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
                u64::from(int_ty.width()).div_ceil(8)
            } else {
                match ty_handle.disp(ctx).to_string().trim() {
                    "builtin.fp16" | "builtin.bf16" => 2,
                    "builtin.fp32" => 4,
                    "builtin.fp64" => 8,
                    _ => 0,
                }
            }
        })
        .unwrap_or(0)
}

/// Return true if `op` is a `mir.shared_alloc` for a 64-bit hardware `Barrier` (`mbarrier`).
fn is_mbarrier_shared_alloc(ctx: &Context, op: &Operation) -> bool {
    let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
    if size != 1 {
        return false;
    }
    op.attributes.0.iter().any(|(k, v)| {
        k.to_string() == "elem_type"
            && v.downcast_ref::<TypeAttr>()
                .and_then(|ta| {
                    ta.get_type(ctx)
                        .deref(ctx)
                        .downcast_ref::<IntegerType>()
                        .map(|i| i.width() == 64)
                })
                .unwrap_or(false)
    })
}

/// Provenance of a hoisted 2D global slice band start (`a_start` or `b_start`),
/// where valid incoming CFG paths checked the strip's last element `< slice.len()`
/// under `extent != 0` and invalid incoming paths pass `u64::MAX`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BandStartProvenance {
    /// Sentinel `u64::MAX` (`18446744073709551615`).
    SentinelMax,
    /// Row strip start `start` proven to satisfy `cols > 0` and `start + (cols - 1) < param.len()`
    /// whenever not equal to `u64::MAX`.
    RowBand {
        /// Target slice parameter index.
        param: usize,
        /// Canonical SSA root of the row width `cols`.
        cols_root: pliron::value::Value,
    },
    /// Column strip start `start` proven to satisfy `rows > 0` and
    /// `(rows - 1) * stride + start < param.len()` whenever not equal to `u64::MAX`.
    ColBand {
        /// Target slice parameter index.
        param: usize,
        /// Canonical SSA root of the column height `rows`.
        rows_root: pliron::value::Value,
        /// Canonical SSA root of the row stride.
        stride_root: pliron::value::Value,
    },
}

/// Branch condition extracted on canonical integer SSA roots for 2D strip verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StripCond {
    /// Unsigned value is strictly positive (`val != 0`).
    NonZero(pliron::value::Value),
    /// Unsigned value equals zero (`val == 0`).
    IsZero(pliron::value::Value),
    /// Value equals sentinel `u64::MAX`.
    IsMax(pliron::value::Value),
    /// Value does not equal sentinel `u64::MAX`.
    NotMax(pliron::value::Value),
    /// Strict inequality `lhs < rhs`.
    Lt(pliron::value::Value, pliron::value::Value),
    /// Negated strict inequality `!(lhs < rhs)`.
    NotLt(pliron::value::Value, pliron::value::Value),
}

impl StripCond {
    /// Invert this condition across `mir.not` or the false arm of `mir.cond_br`.
    fn negate(self) -> Self {
        match self {
            Self::NonZero(v) => Self::IsZero(v),
            Self::IsZero(v) => Self::NonZero(v),
            Self::IsMax(v) => Self::NotMax(v),
            Self::NotMax(v) => Self::IsMax(v),
            Self::Lt(a, b) => Self::NotLt(a, b),
            Self::NotLt(a, b) => Self::Lt(a, b),
        }
    }
}

/// Analyze 2D global slice strip guards (`checked_raw_row_band_start` /
/// `checked_raw_col_band_start` + dominating `col < a_eff` / `row < b_eff` checks)
/// and return a map from verified index SSA values to `(param, GlobalStripKind, offset)`.
fn analyze_verified_2d_strips(
    ctx: &Context,
    blocks: &[Ptr<BasicBlock>],
    succs: &[Vec<usize>],
    env: &SsaEnv,
) -> HashMap<pliron::value::Value, (usize, GlobalStripKind, u64)> {
    let num_blocks = blocks.len();
    let mut cast_parent: HashMap<pliron::value::Value, pliron::value::Value> = HashMap::new();
    for &b_ptr in blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            if Operation::get_opid(op_ptr, ctx).to_string() == "mir.cast" {
                let op = op_ptr.deref(ctx);
                if op.get_num_operands() == 1 && op.get_num_results() == 1 {
                    let src = op.get_operand(0);
                    let dst = op.get_result(0);
                    let src_ty = src.get_type(ctx).deref(ctx);
                    let dst_ty = dst.get_type(ctx).deref(ctx);
                    if let (Some(si), Some(di)) = (
                        src_ty.downcast_ref::<IntegerType>(),
                        dst_ty.downcast_ref::<IntegerType>(),
                    ) && si.is_unsigned()
                        && di.is_unsigned()
                        && si.width() <= di.width()
                    {
                        cast_parent.insert(dst, src);
                    }
                }
            }
        }
    }
    let root_of = |mut v: pliron::value::Value| -> pliron::value::Value {
        let mut hops = 0usize;
        while let Some(&p) = cast_parent.get(&v) {
            v = p;
            hops += 1;
            if hops > 16 {
                break;
            }
        }
        v
    };

    let mut const_vals: HashMap<pliron::value::Value, u128> = HashMap::new();
    let mut slice_lens: HashMap<pliron::value::Value, usize> = HashMap::new();
    let mut sub_one_of: HashMap<pliron::value::Value, pliron::value::Value> = HashMap::new();
    let mut add_ops: HashMap<pliron::value::Value, (pliron::value::Value, pliron::value::Value)> =
        HashMap::new();
    let mut mul_ops: HashMap<pliron::value::Value, (pliron::value::Value, pliron::value::Value)> =
        HashMap::new();
    let mut tx_derived: HashSet<pliron::value::Value> = HashSet::new();
    let mut ty_derived: HashSet<pliron::value::Value> = HashSet::new();

    for &b_ptr in blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            let op = op_ptr.deref(ctx);
            if op.get_num_results() == 1 {
                let res = op.get_result(0);
                if let Some(SsaKind::ConstInt { value, .. }) = env.get_kind(&res) {
                    const_vals.insert(root_of(res), value);
                }
                if let Some(SsaKind::SliceLen(param)) = env.get_kind(&res) {
                    slice_lens.insert(root_of(res), param);
                }
            }
            match opid.as_str() {
                "nvvm.read_ptx_sreg_tid_x" => {
                    tx_derived.insert(root_of(op.get_result(0)));
                }
                "nvvm.read_ptx_sreg_tid_y" => {
                    ty_derived.insert(root_of(op.get_result(0)));
                }
                "mir.add" if op.get_num_operands() == 2 && op.get_num_results() == 1 => {
                    add_ops.insert(
                        root_of(op.get_result(0)),
                        (root_of(op.get_operand(0)), root_of(op.get_operand(1))),
                    );
                }
                "mir.mul" if op.get_num_operands() == 2 && op.get_num_results() == 1 => {
                    mul_ops.insert(
                        root_of(op.get_result(0)),
                        (root_of(op.get_operand(0)), root_of(op.get_operand(1))),
                    );
                }
                _ => {}
            }
        }
    }
    for &b_ptr in blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            if Operation::get_opid(op_ptr, ctx).to_string() == "mir.sub" {
                let op = op_ptr.deref(ctx);
                if op.get_num_operands() == 2 && op.get_num_results() == 1 {
                    let rhs_r = root_of(op.get_operand(1));
                    if const_vals.get(&rhs_r) == Some(&1) {
                        sub_one_of.insert(root_of(op.get_result(0)), root_of(op.get_operand(0)));
                    }
                }
            }
        }
    }
    for _ in 0..8 {
        let mut changed = false;
        for (&res, &(a, b)) in &add_ops {
            if (tx_derived.contains(&a) || tx_derived.contains(&b))
                && !ty_derived.contains(&a)
                && !ty_derived.contains(&b)
                && tx_derived.insert(res)
            {
                changed = true;
            }
            if (ty_derived.contains(&a) || ty_derived.contains(&b))
                && !tx_derived.contains(&a)
                && !tx_derived.contains(&b)
                && ty_derived.insert(res)
            {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut cond_map: HashMap<pliron::value::Value, StripCond> = HashMap::new();
    for _ in 0..4 {
        for &b_ptr in blocks {
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                let opid = Operation::get_opid(op_ptr, ctx).to_string();
                let op = op_ptr.deref(ctx);
                if op.get_num_results() != 1 {
                    continue;
                }
                let res = op.get_result(0);
                match opid.as_str() {
                    "mir.eq" | "mir.ne" if op.get_num_operands() == 2 => {
                        let a = root_of(op.get_operand(0));
                        let b = root_of(op.get_operand(1));
                        let is_eq = opid == "mir.eq";
                        for (val_r, cst_r) in [(a, b), (b, a)] {
                            if let Some(&c) = const_vals.get(&cst_r) {
                                if c == 0 {
                                    cond_map.insert(
                                        res,
                                        if is_eq {
                                            StripCond::IsZero(val_r)
                                        } else {
                                            StripCond::NonZero(val_r)
                                        },
                                    );
                                } else if c == u128::from(u64::MAX) {
                                    cond_map.insert(
                                        res,
                                        if is_eq {
                                            StripCond::IsMax(val_r)
                                        } else {
                                            StripCond::NotMax(val_r)
                                        },
                                    );
                                }
                            }
                        }
                    }
                    "mir.gt" if op.get_num_operands() == 2 => {
                        let a = root_of(op.get_operand(0));
                        let b = root_of(op.get_operand(1));
                        if const_vals.get(&b) == Some(&0) {
                            cond_map.insert(res, StripCond::NonZero(a));
                        }
                    }
                    "mir.lt" if op.get_num_operands() == 2 => {
                        let a = root_of(op.get_operand(0));
                        let b = root_of(op.get_operand(1));
                        if const_vals.get(&a) == Some(&0) {
                            cond_map.insert(res, StripCond::NonZero(b));
                        } else {
                            cond_map.insert(res, StripCond::Lt(a, b));
                        }
                    }
                    "mir.not" if op.get_num_operands() == 1 => {
                        if let Some(&inner) = cond_map.get(&op.get_operand(0)) {
                            cond_map.insert(res, inner.negate());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let mut branch_preds: Vec<(usize, usize, usize, StripCond)> = Vec::new();
    let mut goto_edges: Vec<(usize, usize, pliron::value::Value)> = Vec::new();
    for (src_blk, &b_ptr) in blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            let op = op_ptr.deref(ctx);
            if opid == "mir.cond_br" && op.get_num_operands() >= 1 && op.get_num_successors() == 2 {
                if let Some(&cond) = cond_map.get(&op.get_operand(0))
                    && let (Some(t_idx), Some(f_idx)) = (
                        blocks.iter().position(|&b| b == op.get_successor(0)),
                        blocks.iter().position(|&b| b == op.get_successor(1)),
                    )
                {
                    branch_preds.push((src_blk, t_idx, f_idx, cond));
                    branch_preds.push((src_blk, f_idx, t_idx, cond.negate()));
                }
            } else if opid == "mir.goto"
                && op.get_num_operands() == 1
                && op.get_num_successors() == 1
                && let Some(dst_idx) = blocks.iter().position(|&b| b == op.get_successor(0))
            {
                goto_edges.push((src_blk, dst_idx, op.get_operand(0)));
            }
        }
    }

    let dominates_pred = |b_idx: usize, pred: StripCond| -> bool {
        branch_preds.iter().any(|&(src_blk, in_dst, out_dst, p)| {
            p == pred
                && !can_reach_without(0, b_idx, in_dst, succs, num_blocks)
                && !can_reach_without(out_dst, b_idx, src_blk, succs, num_blocks)
        })
    };

    let mut band_vars: HashMap<pliron::value::Value, BandStartProvenance> = HashMap::new();
    for _ in 0..num_blocks + 2 {
        let mut changed = false;
        for (dst_blk, &b_ptr) in blocks.iter().enumerate().skip(1) {
            let blk = b_ptr.deref(ctx);
            if blk.get_num_arguments() != 1 {
                continue;
            }
            let arg_root = root_of(blk.get_argument(0));
            let incoming: Vec<_> = goto_edges
                .iter()
                .filter(|&&(_, d, _)| d == dst_blk)
                .collect();
            if incoming.is_empty() {
                continue;
            }
            let mut merged: Option<BandStartProvenance> = None;
            let mut all_ok = true;
            for &&(src_blk, _, in_val) in &incoming {
                let in_r = root_of(in_val);
                let prov = if const_vals.get(&in_r) == Some(&u128::from(u64::MAX)) {
                    Some(BandStartProvenance::SentinelMax)
                } else if let Some(&existing) = band_vars.get(&in_r) {
                    Some(existing)
                } else {
                    let mut found = None;
                    for &(_, _, _, pred) in &branch_preds {
                        let StripCond::Lt(last_r, len_r) = pred else {
                            continue;
                        };
                        let Some(&param) = slice_lens.get(&len_r) else {
                            continue;
                        };
                        if !dominates_pred(src_blk, pred) {
                            continue;
                        }
                        if let Some(&(x, y)) = add_ops.get(&last_r) {
                            for (start_cand, rem_cand) in [(x, y), (y, x)] {
                                if start_cand != in_r {
                                    continue;
                                }
                                if let Some(&cols_root) = sub_one_of.get(&rem_cand)
                                    && dominates_pred(src_blk, StripCond::NonZero(cols_root))
                                {
                                    found = Some(BandStartProvenance::RowBand { param, cols_root });
                                    break;
                                }
                                if let Some(&(m1, m2)) = mul_ops.get(&rem_cand) {
                                    for (sub_cand, stride_root) in [(m1, m2), (m2, m1)] {
                                        if let Some(&rows_root) = sub_one_of.get(&sub_cand)
                                            && dominates_pred(
                                                src_blk,
                                                StripCond::NonZero(rows_root),
                                            )
                                        {
                                            found = Some(BandStartProvenance::ColBand {
                                                param,
                                                rows_root,
                                                stride_root,
                                            });
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if found.is_some() {
                            break;
                        }
                    }
                    found
                };
                match prov {
                    Some(BandStartProvenance::SentinelMax) => {}
                    Some(concrete) => match merged {
                        None => merged = Some(concrete),
                        Some(prev) if prev == concrete => {}
                        Some(_) => {
                            all_ok = false;
                            break;
                        }
                    },
                    None => {
                        all_ok = false;
                        break;
                    }
                }
            }
            if all_ok
                && let Some(valid_prov) = merged
                && band_vars.get(&arg_root) != Some(&valid_prov)
            {
                band_vars.insert(arg_root, valid_prov);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut band_effs: HashMap<pliron::value::Value, pliron::value::Value> = HashMap::new();
    for (dst_blk, &b_ptr) in blocks.iter().enumerate().skip(1) {
        let blk = b_ptr.deref(ctx);
        if blk.get_num_arguments() != 1 {
            continue;
        }
        let eff_root = root_of(blk.get_argument(0));
        let incoming: Vec<_> = goto_edges
            .iter()
            .filter(|&&(_, d, _)| d == dst_blk)
            .collect();
        if incoming.is_empty() {
            continue;
        }
        for (&band_root, &prov) in &band_vars {
            let extent_root = match prov {
                BandStartProvenance::RowBand { cols_root, .. } => cols_root,
                BandStartProvenance::ColBand { rows_root, .. } => rows_root,
                BandStartProvenance::SentinelMax => continue,
            };
            let mut saw_extent = false;
            let all_valid = incoming.iter().all(|&&(src_blk, _, op0)| {
                let r = root_of(op0);
                if const_vals.get(&r) == Some(&0) {
                    true
                } else if r == extent_root && dominates_pred(src_blk, StripCond::NotMax(band_root))
                {
                    saw_extent = true;
                    true
                } else {
                    false
                }
            });
            if all_valid && saw_extent {
                band_effs.insert(eff_root, band_root);
                break;
            }
        }
    }

    let step_is_guarded = |b_idx: usize,
                           step_root: pliron::value::Value,
                           band_root: pliron::value::Value,
                           extent_root: pliron::value::Value|
     -> bool {
        let via_eff = band_effs.iter().any(|(&eff_r, &b_r)| {
            b_r == band_root && dominates_pred(b_idx, StripCond::Lt(step_root, eff_r))
        });
        via_eff
            || (dominates_pred(b_idx, StripCond::NotMax(band_root))
                && dominates_pred(b_idx, StripCond::Lt(step_root, extent_root)))
    };

    let peel_const_add = |r: pliron::value::Value| -> Vec<(pliron::value::Value, u64)> {
        let mut candidates = vec![(r, 0u64)];
        if let Some(&(a, b)) = add_ops.get(&r) {
            if let Some(&c) = const_vals.get(&b)
                && let Ok(off) = u64::try_from(c)
            {
                candidates.push((a, off));
            }
            if let Some(&c) = const_vals.get(&a)
                && let Ok(off) = u64::try_from(c)
            {
                candidates.push((b, off));
            }
        }
        candidates
    };

    let mut verified_indices: HashMap<pliron::value::Value, (usize, GlobalStripKind, u64)> =
        HashMap::new();
    for (b_idx, &b_ptr) in blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let op = op_ptr.deref(ctx);
            for res in op.results() {
                let idx_root = root_of(res);
                let mut matched = None;
                for (core_root, outer_off) in peel_const_add(idx_root) {
                    let Some(&(lhs, rhs)) = add_ops.get(&core_root) else {
                        continue;
                    };
                    for (band_cand, step_part) in [(lhs, rhs), (rhs, lhs)] {
                        match band_vars.get(&band_cand) {
                            Some(&BandStartProvenance::RowBand { param, cols_root }) => {
                                for (step_root, inner_off) in peel_const_add(step_part) {
                                    if tx_derived.contains(&step_root)
                                        && step_is_guarded(b_idx, step_root, band_cand, cols_root)
                                    {
                                        matched = Some((
                                            param,
                                            GlobalStripKind::RowStrip,
                                            outer_off.saturating_add(inner_off),
                                        ));
                                        break;
                                    }
                                }
                            }
                            Some(&BandStartProvenance::ColBand {
                                param,
                                rows_root,
                                stride_root,
                            }) => {
                                if let Some(&(m1, m2)) = mul_ops.get(&step_part) {
                                    for (row_cand, st_cand) in [(m1, m2), (m2, m1)] {
                                        if st_cand == stride_root
                                            && ty_derived.contains(&row_cand)
                                            && step_is_guarded(
                                                b_idx, row_cand, band_cand, rows_root,
                                            )
                                        {
                                            matched =
                                                Some((param, GlobalStripKind::ColStrip, outer_off));
                                            break;
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                        if matched.is_some() {
                            break;
                        }
                    }
                    if matched.is_some() {
                        break;
                    }
                }
                if let Some(info) = matched {
                    verified_indices.insert(res, info);
                }
            }
        }
    }
    verified_indices
}

/// Check if `start` can reach `target` in `succs` without passing through either `avoid1` or `avoid2`.
fn can_reach_avoiding_two(
    start: usize,
    target: usize,
    avoid1: usize,
    avoid2: usize,
    succs: &[Vec<usize>],
    num_blocks: usize,
) -> bool {
    if start == avoid1 || start == avoid2 {
        return false;
    }
    let mut visited = vec![false; num_blocks];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(curr) = stack.pop() {
        if curr == target {
            return true;
        }
        for &next in &succs[curr] {
            if next != avoid1 && next != avoid2 && !visited[next] {
                visited[next] = true;
                stack.push(next);
            }
        }
    }
    false
}

/// Check if `start` can reach `target` in `succs` without passing through `forbidden`.
fn can_reach_without(
    start: usize,
    target: usize,
    forbidden: usize,
    succs: &[Vec<usize>],
    num_blocks: usize,
) -> bool {
    can_reach_avoiding_two(start, target, forbidden, usize::MAX, succs, num_blocks)
}

/// Check if `start` can reach `target` in `succs`.
fn can_reach(start: usize, target: usize, succs: &[Vec<usize>], num_blocks: usize) -> bool {
    can_reach_without(start, target, usize::MAX, succs, num_blocks)
}

/// Translate a 2D phased shared-memory kernel (`sgemm_tiled_views`) using `SsaEnv`.
fn translate_verified_2d_tiled_kernel(
    ctx: &Context,
    function: Ptr<Operation>,
    spec: KernelVerificationSpec,
) -> Result<TranslatedFunction, ExportError> {
    let func = function.deref(ctx);
    let region = func.get_region(0).deref(ctx);
    let blocks: Vec<Ptr<BasicBlock>> = region.iter(ctx).collect();
    if blocks.is_empty() {
        return Err(reject("kernel", "empty kernel region"));
    }

    // 1. Discover all static `SharedArray` and `Barrier` (`mbarrier`) allocations
    //    (`mir.shared_alloc`) and assign deterministic `0..N` IDs sorted by short name.
    let mut shared_raw_list: Vec<(String, u64)> = Vec::new();
    let mut bar_raw_list: Vec<String> = Vec::new();
    for &b_ptr in &blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            if Operation::get_opid(op_ptr, ctx).to_string() == "mir.shared_alloc" {
                let op = op_ptr.deref(ctx);
                let full_name = get_string_attr(&op.attributes, "source_name")
                    .unwrap_or_else(|| "SHARED".to_string());
                let short = full_name
                    .rsplit("::")
                    .next()
                    .unwrap_or(&full_name)
                    .to_string();
                if is_mbarrier_shared_alloc(ctx, &op) {
                    if !bar_raw_list.contains(&short) {
                        bar_raw_list.push(short);
                    }
                } else {
                    let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
                    if !shared_raw_list.iter().any(|(n, _)| n == &short) {
                        shared_raw_list.push((short, size));
                    }
                }
            }
        }
    }
    shared_raw_list.sort_by(|a, b| a.0.cmp(&b.0));
    bar_raw_list.sort();

    // 2. Build the block adjacency list, locate `nvvm.barrier0` sites `(block_idx, ord_in_block)`,
    //    and find `mir.return` blocks.
    let num_blocks = blocks.len();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    let mut barrier_sites: Vec<(usize, usize)> = Vec::new();
    let mut return_blocks: Vec<usize> = Vec::new();

    for (b_idx, &b_ptr) in blocks.iter().enumerate() {
        let mut ord_in_blk = 0usize;
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            let op = op_ptr.deref(ctx);
            if opid == "nvvm.barrier0" {
                barrier_sites.push((b_idx, ord_in_blk));
                ord_in_blk += 1;
            } else if opid == "mir.return" {
                return_blocks.push(b_idx);
            } else if opid == "mir.goto" || opid == "mir.cond_br" {
                for s_idx in 0..op.get_num_successors() {
                    let target = op.get_successor(s_idx);
                    if let Some(dst) = blocks.iter().position(|&b| b == target)
                        && !succs[b_idx].contains(&dst)
                    {
                        succs[b_idx].push(dst);
                    }
                }
            }
        }
    }

    // 3. Separate acyclic (`init` / `epilogue`) barriers from cyclic (`in-loop`) barriers,
    //    verify barrier uniformity across all paths, and assign cyclic `passed % K` phases
    //    inside loops regardless of MIR block list ordering.
    let is_cyclic_block = |b: usize| -> bool {
        succs[b]
            .iter()
            .any(|&s| can_reach(s, b, &succs, num_blocks))
    };
    let mut acyclic_barriers: Vec<(usize, usize)> = Vec::new();
    let mut cyclic_barriers: Vec<(usize, usize)> = Vec::new();
    for &site in &barrier_sites {
        if is_cyclic_block(site.0) {
            cyclic_barriers.push(site);
        } else {
            acyclic_barriers.push(site);
        }
    }

    let mut barriers_uniform = !barrier_sites.is_empty();
    for &(ab_blk, _) in &acyclic_barriers {
        for &ret_blk in &return_blocks {
            if can_reach_without(0, ret_blk, ab_blk, &succs, num_blocks) {
                barriers_uniform = false;
            }
        }
    }

    let mut enclosing_loop_header: Option<usize> = None;
    if let Some(&(sample_bar, _)) = cyclic_barriers.first() {
        let candidates: Vec<usize> = (0..num_blocks)
            .filter(|&h| {
                can_reach(h, sample_bar, &succs, num_blocks)
                    && can_reach(sample_bar, h, &succs, num_blocks)
                    && !can_reach_without(0, sample_bar, h, &succs, num_blocks)
            })
            .collect();
        let header = candidates.iter().copied().find(|&h| {
            candidates
                .iter()
                .all(|&other| other == h || !can_reach_without(0, other, h, &succs, num_blocks))
        });
        enclosing_loop_header = header;
        if let Some(hdr) = header {
            for &ret_blk in &return_blocks {
                if can_reach_without(0, ret_blk, hdr, &succs, num_blocks) {
                    barriers_uniform = false;
                }
            }
            for &body_start in &succs[hdr] {
                if can_reach(body_start, hdr, &succs, num_blocks) {
                    for &(cb_blk, _) in &cyclic_barriers {
                        if can_reach_without(body_start, hdr, cb_blk, &succs, num_blocks) {
                            barriers_uniform = false;
                        }
                    }
                    for &ret_blk in &return_blocks {
                        if can_reach_without(body_start, ret_blk, hdr, &succs, num_blocks) {
                            barriers_uniform = false;
                        }
                    }
                }
            }
            cyclic_barriers.sort_by(|&(blk_a, ord_a), &(blk_b, ord_b)| {
                if blk_a == blk_b {
                    return ord_a.cmp(&ord_b);
                }
                let a_before_b = succs[hdr]
                    .iter()
                    .filter(|&&s| can_reach(s, hdr, &succs, num_blocks))
                    .all(|&s| !can_reach_avoiding_two(s, blk_b, blk_a, hdr, &succs, num_blocks));
                if a_before_b {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });
        } else {
            barriers_uniform = false;
        }
    }

    let acyclic_dom_count = |target_blk: usize, seen_in_blk: usize| -> usize {
        let strict = acyclic_barriers
            .iter()
            .filter(|&&(ab_blk, _)| {
                ab_blk != target_blk
                    && can_reach(ab_blk, target_blk, &succs, num_blocks)
                    && !can_reach_without(0, target_blk, ab_blk, &succs, num_blocks)
            })
            .count();
        let in_same = if !is_cyclic_block(target_blk) {
            seen_in_blk
        } else {
            0
        };
        strict + in_same
    };

    let block_phase = |b_idx: usize, seen_barriers_in_block: usize| -> usize {
        if !barriers_uniform || barrier_sites.is_empty() {
            return 0;
        }
        if let Some(hdr) = enclosing_loop_header {
            let base_loop_phase = acyclic_dom_count(hdr, 0);
            let k = cyclic_barriers.len();
            let in_loop = b_idx == hdr
                || (can_reach(hdr, b_idx, &succs, num_blocks)
                    && can_reach(b_idx, hdr, &succs, num_blocks));
            if in_loop {
                if k <= 1 || b_idx == hdr {
                    return base_loop_phase;
                }
                let passed = cyclic_barriers
                    .iter()
                    .filter(|&&(bar_blk, bar_ord)| {
                        if bar_blk == b_idx {
                            bar_ord < seen_barriers_in_block
                        } else {
                            let body_starts: Vec<usize> = succs[hdr]
                                .iter()
                                .copied()
                                .filter(|&s| can_reach(s, hdr, &succs, num_blocks))
                                .collect();
                            !body_starts.is_empty()
                                && body_starts.iter().all(|&s| {
                                    can_reach_without(s, b_idx, hdr, &succs, num_blocks)
                                        && !can_reach_avoiding_two(
                                            s, b_idx, bar_blk, hdr, &succs, num_blocks,
                                        )
                                })
                        }
                    })
                    .count();
                return base_loop_phase + (passed % k);
            }
            if can_reach(hdr, b_idx, &succs, num_blocks) {
                return acyclic_dom_count(b_idx, seen_barriers_in_block) + k;
            }
        }
        acyclic_dom_count(b_idx, seen_barriers_in_block)
    };

    // 4. Fixed-point SSA propagation over the function blocks.
    //    Blocks are not in topological order (e.g., `^block73v1` defines `tid.x`
    //    after `^block59v1` uses it), so we iterate until all `SsaKind` bindings
    //    stabilize before enforcing fail-closed checks on `mir.ptr_offset`.
    let mut env = SsaEnv::new();
    if let Some(&entry_block) = blocks.first() {
        for (idx, arg) in entry_block.deref(ctx).arguments().enumerate() {
            env.bind_arg(arg, idx, SsaKind::BufferParam(idx));
        }
    }
    for _pass in 0..num_blocks + 3 {
        let mut changed = false;
        let mut update = |env: &mut SsaEnv, val: pliron::value::Value, kind: SsaKind| {
            let old = env.get_kind(&val);
            if old.as_ref() != Some(&kind) {
                // Never downgrade a specific kind back to `DataValue`.
                if kind == SsaKind::DataValue && old.is_some() {
                    return;
                }
                env.bind_result(val, kind);
                changed = true;
            }
        };
        for (b_idx, &b_ptr) in blocks.iter().enumerate() {
            let block = b_ptr.deref(ctx);
            if b_idx != 0 {
                for arg in block.arguments() {
                    update(&mut env, arg, SsaKind::DataValue);
                }
            }
            // Detect inner scan loop counter `i < bound` ONLY when `mir.lt(step_arg, bound)`
            // (or `mir.not(mir.lt(step_arg, bound))`) drives this header block's `mir.cond_br`
            // into a cyclic loop body.
            let mut lt_in_blk: HashMap<pliron::value::Value, (pliron::value::Value, u64)> =
                HashMap::new();
            let mut not_in_blk: HashMap<pliron::value::Value, pliron::value::Value> =
                HashMap::new();
            let mut term_cond_br: Option<(pliron::value::Value, usize, usize)> = None;
            for op_ptr in block.iter(ctx) {
                let opid = Operation::get_opid(op_ptr, ctx).to_string();
                let op = op_ptr.deref(ctx);
                if opid == "mir.lt"
                    && op.get_num_operands() == 2
                    && op.get_num_results() == 1
                    && let Some(SsaKind::ConstInt { value: bound, .. }) =
                        env.get_kind(&op.get_operand(1))
                {
                    lt_in_blk.insert(
                        op.get_result(0),
                        (op.get_operand(0), u64::try_from(bound).unwrap_or(u64::MAX)),
                    );
                } else if opid == "mir.not"
                    && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                {
                    not_in_blk.insert(op.get_result(0), op.get_operand(0));
                } else if opid == "mir.cond_br"
                    && op.get_num_operands() >= 1
                    && op.get_num_successors() == 2
                    && let (Some(t_idx), Some(f_idx)) = (
                        blocks.iter().position(|&b| b == op.get_successor(0)),
                        blocks.iter().position(|&b| b == op.get_successor(1)),
                    )
                {
                    term_cond_br = Some((op.get_operand(0), t_idx, f_idx));
                }
            }
            if let Some((cond_val, t_idx, f_idx)) = term_cond_br {
                let decoded = if let Some(&(lhs_val, bound)) = lt_in_blk.get(&cond_val) {
                    Some((lhs_val, bound, t_idx, f_idx))
                } else if let Some(&inner_val) = not_in_blk.get(&cond_val)
                    && let Some(&(lhs_val, bound)) = lt_in_blk.get(&inner_val)
                {
                    Some((lhs_val, bound, f_idx, t_idx))
                } else {
                    None
                };
                if let Some((lhs_val, bound, body_blk, exit_blk)) = decoded
                    && can_reach(body_blk, b_idx, &succs, num_blocks)
                {
                    for (arg_pos, step_arg) in block.arguments().enumerate() {
                        if step_arg == lhs_val {
                            update(
                                &mut env,
                                step_arg,
                                SsaKind::Index(IdxExpr::LoopVar {
                                    loop_id: b_idx * 16 + arg_pos,
                                    header_block: b_idx,
                                    body_block: body_blk,
                                    exit_block: exit_blk,
                                    bound,
                                }),
                            );
                        }
                    }
                }
            }
            for op_ptr in block.iter(ctx) {
                let opid = Operation::get_opid(op_ptr, ctx).to_string();
                let op = op_ptr.deref(ctx);
                match opid.as_str() {
                    "nvvm.read_ptx_sreg_tid_x" => {
                        update(&mut env, op.get_result(0), SsaKind::Index(IdxExpr::Tx));
                    }
                    "nvvm.read_ptx_sreg_tid_y" => {
                        update(&mut env, op.get_result(0), SsaKind::Index(IdxExpr::Ty));
                    }
                    "nvvm.read_ptx_sreg_laneid" => {
                        update(
                            &mut env,
                            op.get_result(0),
                            SsaKind::Index(IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)),
                        );
                    }
                    "mir.call" => {
                        let callee = get_string_attr(&op.attributes, "callee").unwrap_or_default();
                        if callee.contains("lane_id") && op.get_num_results() == 1 {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::Index(IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)),
                            );
                        } else if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                    "mir.div" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: g, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && g > 0
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Div(
                                Box::new(a),
                                u64::try_from(g).unwrap_or(u64::MAX),
                            ))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.shr" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: sh, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && sh < 16
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Div(Box::new(a), 1u64 << sh))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.rem" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: g, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && g > 0
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Mod(
                                Box::new(a),
                                u64::try_from(g).unwrap_or(u64::MAX),
                            ))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.bitand" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = match (&lhs, &rhs) {
                            (Some(l), Some(SsaKind::ConstInt { value: m, .. }))
                            | (Some(SsaKind::ConstInt { value: m, .. }), Some(l))
                                if *m > 0 && (*m + 1).is_power_of_two() =>
                            {
                                if let Some(a) = as_idx_expr(l)
                                    && a.has_coord()
                                {
                                    SsaKind::Index(IdxExpr::Mod(
                                        Box::new(a),
                                        u64::try_from(*m + 1).unwrap_or(u64::MAX),
                                    ))
                                } else {
                                    SsaKind::DataValue
                                }
                            }
                            _ => SsaKind::DataValue,
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.eq" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Guard(GuardExpr::Eq(a, b))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.ne" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Guard(GuardExpr::Ne(a, b))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.not" => {
                        let inner = env.get_kind(&op.get_operand(0));
                        let mapped = match inner {
                            Some(SsaKind::Guard(g)) => SsaKind::Guard(g.negate()),
                            Some(SsaKind::MbarrierReady { bar_id }) => {
                                SsaKind::MbarrierNotReady { bar_id }
                            }
                            Some(SsaKind::MbarrierNotReady { bar_id }) => {
                                SsaKind::MbarrierReady { bar_id }
                            }
                            _ => SsaKind::DataValue,
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "nvvm.mbarrier_try_wait_parity_shared"
                    | "nvvm.mbarrier_try_wait_shared"
                    | "nvvm.mbarrier_test_wait_shared"
                    | "nvvm.mbarrier_try_wait_parity_cluster"
                    | "nvvm.mbarrier_try_wait_parity_cluster_relaxed" => {
                        let mapped = if op.get_num_operands() >= 1
                            && let Some(SsaKind::MbarrierPtr { bar_id }) =
                                env.get_kind(&op.get_operand(0))
                        {
                            SsaKind::MbarrierReady { bar_id }
                        } else {
                            SsaKind::DataValue
                        };
                        if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), mapped);
                        }
                    }
                    "mir.ge" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Guard(GuardExpr::Ge(a, b))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.constant" | "builtin.constant" => {
                        let key = if opid == "mir.constant" {
                            "value"
                        } else {
                            "builtin_constant_value"
                        };
                        if let Some(val) = int_attr_u64(&op.attributes, key) {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::ConstInt {
                                    width: 64,
                                    value: u128::from(val),
                                },
                            );
                        } else if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                    "mir.shared_alloc" => {
                        let full_name = get_string_attr(&op.attributes, "source_name")
                            .unwrap_or_else(|| "SHARED".to_string());
                        let short = full_name
                            .rsplit("::")
                            .next()
                            .unwrap_or(&full_name)
                            .to_string();
                        if is_mbarrier_shared_alloc(ctx, &op) {
                            let bar_id = bar_raw_list.iter().position(|n| n == &short).unwrap_or(0);
                            update(&mut env, op.get_result(0), SsaKind::MbarrierPtr { bar_id });
                        } else {
                            let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
                            let elem_bytes = shared_alloc_elem_bytes(ctx, &op);
                            let shared_id = shared_raw_list
                                .iter()
                                .position(|(n, _)| n == &short)
                                .unwrap_or(0);
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::SharedBasePtr {
                                    shared_id,
                                    size,
                                    elem_bytes,
                                },
                            );
                        }
                    }
                    "mir.cast" => {
                        if op.get_num_operands() == 1
                            && op.get_num_results() == 1
                            && let Some(kind) = env.get_kind(&op.get_operand(0))
                        {
                            let src_ty = op.get_operand(0).get_type(ctx).deref(ctx);
                            let dst_ty = op.get_result(0).get_type(ctx).deref(ctx);
                            let mapped = match (
                                src_ty.downcast_ref::<IntegerType>(),
                                dst_ty.downcast_ref::<IntegerType>(),
                            ) {
                                (Some(si), Some(di)) => {
                                    if si.is_unsigned() && di.is_unsigned() {
                                        if di.width() >= si.width() {
                                            kind
                                        } else {
                                            match kind {
                                                SsaKind::ConstInt { value, .. } => {
                                                    let mask = if di.width() >= 128 {
                                                        u128::MAX
                                                    } else {
                                                        (1u128 << di.width()) - 1
                                                    };
                                                    SsaKind::ConstInt {
                                                        width: di.width(),
                                                        value: value & mask,
                                                    }
                                                }
                                                SsaKind::Index(expr) if di.width() < 64 => {
                                                    let modulus = 1u64 << di.width();
                                                    if expr
                                                        .exclusive_upper_bound(
                                                            spec.block_size,
                                                            spec.block_y,
                                                        )
                                                        .is_some_and(|ub| ub <= modulus)
                                                    {
                                                        SsaKind::Index(expr)
                                                    } else {
                                                        SsaKind::Index(IdxExpr::Mod(
                                                            Box::new(expr),
                                                            modulus,
                                                        ))
                                                    }
                                                }
                                                _ => SsaKind::DataValue,
                                            }
                                        }
                                    } else {
                                        SsaKind::DataValue
                                    }
                                }
                                (None, None) => kind,
                                _ => SsaKind::DataValue,
                            };
                            update(&mut env, op.get_result(0), mapped);
                        }
                    }
                    "mir.shl" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: sh, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && sh < 16
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Mul(
                                Box::new(a),
                                Box::new(IdxExpr::Const(1u64 << sh)),
                            ))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.mul" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            let prod = IdxExpr::Mul(Box::new(a), Box::new(b));
                            if prod.single_loop_scope().is_valid() {
                                SsaKind::Index(prod)
                            } else {
                                SsaKind::DataValue
                            }
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.add" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            let sum = IdxExpr::Add(Box::new(a), Box::new(b));
                            if sum.single_loop_scope().is_valid() {
                                SsaKind::Index(sum)
                            } else {
                                SsaKind::DataValue
                            }
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.bitor" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                            && let Some(disjoint_sum) =
                                IdxExpr::disjoint_bitor(a, b, spec.block_size, spec.block_y)
                            && disjoint_sum.single_loop_scope().is_valid()
                        {
                            SsaKind::Index(disjoint_sum)
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.extract_field" => {
                        let base = env.get_kind(&op.get_operand(0));
                        let field = attribute_field_index(&op.attributes, "index");
                        let mapped = match (base, field) {
                            (Some(SsaKind::BufferParam(p)), Some(0)) => SsaKind::SlicePtr(p),
                            (Some(SsaKind::BufferParam(p)), Some(1)) => SsaKind::SliceLen(p),
                            _ => SsaKind::DataValue,
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.lt" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && a.has_coord()
                        {
                            SsaKind::Guard(GuardExpr::Lt(a, b))
                        } else if let (
                            Some(SsaKind::VerifiedGlobalStripIdx {
                                param, offset: 0, ..
                            }),
                            Some(SsaKind::SliceLen(guard_param)),
                        ) = (lhs, rhs)
                            && param == guard_param
                        {
                            SsaKind::IndexLtLen { guard_param }
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.ptr_offset" => {
                        let base = env.get_kind(&op.get_operand(0));
                        let idx = env.get_kind(&op.get_operand(1));
                        if let (
                            Some(SsaKind::SharedBasePtr {
                                shared_id, size, ..
                            }),
                            Some(SsaKind::Index(addr)),
                        ) = (base.clone(), idx.clone())
                            && addr.has_coord()
                        {
                            let loop_in_scope = match addr.single_loop_scope() {
                                LoopScopeCheck::None => true,
                                LoopScopeCheck::Single {
                                    header_block: hdr_blk,
                                    body_block: body_blk,
                                    exit_block: exit_blk,
                                    ..
                                } => {
                                    !can_reach_without(0, b_idx, body_blk, &succs, num_blocks)
                                        && !can_reach_without(
                                            exit_blk, b_idx, hdr_blk, &succs, num_blocks,
                                        )
                                }
                                LoopScopeCheck::Conflict => false,
                            };
                            if loop_in_scope {
                                update(
                                    &mut env,
                                    op.get_result(0),
                                    SsaKind::SharedElementPtr {
                                        shared_id,
                                        size,
                                        addr,
                                    },
                                );
                            } else if op.get_num_results() == 1 {
                                update(&mut env, op.get_result(0), SsaKind::DataValue);
                            }
                        } else if let (
                            Some(SsaKind::SlicePtr(param)),
                            Some(SsaKind::VerifiedGlobalStripIdx {
                                param: strip_param,
                                kind,
                                offset,
                            }),
                        ) = (base, idx)
                            && param == strip_param
                        {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::GlobalStripElementPtr {
                                    param,
                                    kind,
                                    offset,
                                },
                            );
                        } else if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                    _ => {
                        if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // 4a. Verify 2D global slice strip bounds (`checked_raw_row_band_start` /
    //     `checked_raw_col_band_start` + dominating `col < a_eff` / `row < b_eff`)
    //     and populate `VerifiedGlobalStripIdx`, `IndexLtLen`, and `GlobalStripElementPtr`.
    if !spec.no_oob_idx.is_empty() {
        let verified_strips = analyze_verified_2d_strips(ctx, &blocks, &succs, &env);
        for (val, (param, kind, offset)) in verified_strips {
            env.bind_result(
                val,
                SsaKind::VerifiedGlobalStripIdx {
                    param,
                    kind,
                    offset,
                },
            );
        }
        for _ in 0..num_blocks + 2 {
            let mut changed = false;
            for &b_ptr in &blocks {
                for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                    let opid = Operation::get_opid(op_ptr, ctx).to_string();
                    let op = op_ptr.deref(ctx);
                    match opid.as_str() {
                        "mir.cast" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
                            if let Some(kind @ SsaKind::VerifiedGlobalStripIdx { .. }) =
                                env.get_kind(&op.get_operand(0))
                                && env.get_kind(&op.get_result(0)).as_ref() != Some(&kind)
                            {
                                env.bind_result(op.get_result(0), kind);
                                changed = true;
                            }
                        }
                        "mir.lt" if op.get_num_operands() == 2 && op.get_num_results() == 1 => {
                            if let (
                                Some(SsaKind::VerifiedGlobalStripIdx {
                                    param, offset: 0, ..
                                }),
                                Some(SsaKind::SliceLen(guard_param)),
                            ) = (
                                env.get_kind(&op.get_operand(0)),
                                env.get_kind(&op.get_operand(1)),
                            ) && param == guard_param
                            {
                                let next = SsaKind::IndexLtLen { guard_param };
                                if env.get_kind(&op.get_result(0)).as_ref() != Some(&next) {
                                    env.bind_result(op.get_result(0), next);
                                    changed = true;
                                }
                            }
                        }
                        "mir.ptr_offset"
                            if op.get_num_operands() == 2 && op.get_num_results() == 1 =>
                        {
                            if let (
                                Some(SsaKind::SlicePtr(param)),
                                Some(SsaKind::VerifiedGlobalStripIdx {
                                    param: strip_param,
                                    kind,
                                    offset,
                                }),
                            ) = (
                                env.get_kind(&op.get_operand(0)),
                                env.get_kind(&op.get_operand(1)),
                            ) && param == strip_param
                            {
                                let next = SsaKind::GlobalStripElementPtr {
                                    param,
                                    kind,
                                    offset,
                                };
                                if env.get_kind(&op.get_result(0)).as_ref() != Some(&next) {
                                    env.bind_result(op.get_result(0), next);
                                    changed = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    // 4b. Collect thread-coordinate branch guards (`GuardExpr`) and async `mbarrier`
    //     token chains (`StageProduced` -> `StageReady`) from `mir.cond_br`.
    let mut branch_guards: Vec<(usize, usize, GuardExpr)> = Vec::new();
    let mut arrived_expect_tx_bars: BTreeSet<usize> = BTreeSet::new();
    let mut arrived_tx_bytes_per_bar: HashMap<usize, u64> = HashMap::new();
    let mut async_bar_for_shared: HashMap<usize, usize> = HashMap::new();
    let mut mbarrier_wait_edges: Vec<(usize, usize, usize)> = Vec::new();
    for (w_blk, &b_ptr) in blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            let op = op_ptr.deref(ctx);
            if matches!(
                opid.as_str(),
                "nvvm.mbarrier_arrive_expect_tx_shared" | "nvvm.mbarrier_arrive_expect_tx_cluster"
            ) {
                let (
                    Some(SsaKind::MbarrierPtr { bar_id }),
                    Some(SsaKind::ConstInt { value: raw_tx, .. }),
                ) = (
                    (op.get_num_operands() >= 1)
                        .then(|| env.get_kind(&op.get_operand(0)))
                        .flatten(),
                    (op.get_num_operands() >= 2)
                        .then(|| env.get_kind(&op.get_operand(1)))
                        .flatten(),
                )
                else {
                    return Err(reject(
                        &opid,
                        "mbarrier_arrive_expect_tx requires a static Barrier pointer and non-zero constant byte count",
                    ));
                };
                let tx_bytes = u64::try_from(raw_tx).unwrap_or(u64::MAX);
                if tx_bytes == 0 {
                    return Err(reject(
                        &opid,
                        "mbarrier_arrive_expect_tx byte count must be greater than zero",
                    ));
                }
                arrived_expect_tx_bars.insert(bar_id);
                let entry = arrived_tx_bytes_per_bar.entry(bar_id).or_insert(0);
                *entry = entry.saturating_add(tx_bytes);
            } else if opid.starts_with("nvvm.cp_async_bulk_tensor_g2s") {
                let (
                    Some(SsaKind::SharedBasePtr { shared_id, .. }),
                    Some(SsaKind::MbarrierPtr { bar_id }),
                ) = (
                    (op.get_num_operands() >= 1)
                        .then(|| env.get_kind(&op.get_operand(0)))
                        .flatten(),
                    (op.get_num_operands() >= 2)
                        .then(|| env.get_kind(&op.get_operand(1)))
                        .flatten(),
                )
                else {
                    return Err(reject(
                        &opid,
                        "cp_async_bulk_tensor_g2s requires a static SharedArray destination and static Barrier",
                    ));
                };
                async_bar_for_shared.insert(shared_id, bar_id);
            } else if opid.starts_with("nvvm.cp_async_bulk") {
                return Err(reject(
                    &opid,
                    format!("unsupported async bulk operation `{opid}` in verified kernel"),
                ));
            } else if opid == "mir.cond_br"
                && op.get_num_operands() >= 1
                && op.get_num_successors() == 2
            {
                let cond_kind = env.get_kind(&op.get_operand(0));
                let true_succ = op.get_successor(0);
                let false_succ = op.get_successor(1);
                if let (Some(t_idx), Some(f_idx)) = (
                    blocks.iter().position(|&b| b == true_succ),
                    blocks.iter().position(|&b| b == false_succ),
                ) {
                    match cond_kind {
                        Some(SsaKind::Guard(g)) => {
                            let is_loop_test = matches!(
                                &g,
                                GuardExpr::Lt(a, _) | GuardExpr::Ge(a, _) if a.loop_bound() > 1
                            );
                            if !is_loop_test {
                                if !can_reach(f_idx, t_idx, &succs, num_blocks) {
                                    branch_guards.push((t_idx, f_idx, g.clone()));
                                }
                                if !can_reach(t_idx, f_idx, &succs, num_blocks) {
                                    branch_guards.push((f_idx, t_idx, g.negate()));
                                }
                            }
                        }
                        Some(SsaKind::MbarrierReady { bar_id }) => {
                            mbarrier_wait_edges.push((w_blk, t_idx, bar_id));
                        }
                        Some(SsaKind::MbarrierNotReady { bar_id }) => {
                            mbarrier_wait_edges.push((w_blk, f_idx, bar_id));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    let block_guard_expr = |b_idx: usize| -> GuardExpr {
        let mut guard = GuardExpr::True;
        for (in_dst, out_dst, g) in &branch_guards {
            if !can_reach_without(0, b_idx, *in_dst, &succs, num_blocks)
                && can_reach_without(*in_dst, b_idx, *out_dst, &succs, num_blocks)
            {
                guard = guard.and(g.clone());
            }
        }
        guard
    };
    let is_stage_ready = |b_idx: usize, bar_id: usize| -> bool {
        arrived_expect_tx_bars.contains(&bar_id)
            && mbarrier_wait_edges.iter().any(|&(w_blk, ready_dst, b_id)| {
                b_id == bar_id
                    && b_idx != w_blk
                    && !can_reach_without(0, b_idx, ready_dst, &succs, num_blocks)
                    && can_reach(ready_dst, b_idx, &succs, num_blocks)
            })
    };
    let mut effective_num_barriers = barrier_sites.len();
    let mut final_barriers_uniform = barriers_uniform;
    for &(_w_blk, ready_dst, bar_id) in &mbarrier_wait_edges {
        if arrived_expect_tx_bars.contains(&bar_id) {
            let uniform_wait = return_blocks
                .iter()
                .all(|&ret_blk| !can_reach_without(0, ret_blk, ready_dst, &succs, num_blocks));
            if uniform_wait {
                if effective_num_barriers == 0 {
                    final_barriers_uniform = true;
                }
                effective_num_barriers += 1;
            } else {
                final_barriers_uniform = false;
            }
        }
    }

    // 5. Pair each in-block `mir.assert(mir.lt(idx, size))` with the `SharedElementPtr`
    //    produced by its verified `mir.ptr_offset(shared_base, idx)` in that block, and
    //    only elide the assert once a `mir.load` or `mir.store` on that pointer is
    //    recorded as a Lean `NoOobTileAccess` obligation (preserving `SharedArray::index`
    //    call-terminator splits where `mir.ptr_offset` and `mir.load`/`mir.store` live in
    //    adjacent blocks).
    let mut lt_const_bounds: HashMap<pliron::value::Value, (pliron::value::Value, u64)> =
        HashMap::new();
    for &b_ptr in &blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            if Operation::get_opid(op_ptr, ctx).to_string() == "mir.lt" {
                let op = op_ptr.deref(ctx);
                if op.get_num_operands() == 2 && op.get_num_results() == 1 {
                    let idx_val = op.get_operand(0);
                    let bound_val = op.get_operand(1);
                    if matches!(env.get_kind(&idx_val), Some(SsaKind::Index(_)))
                        && let Some(SsaKind::ConstInt { value, .. }) = env.get_kind(&bound_val)
                        && let Ok(bound) = u64::try_from(value)
                    {
                        lt_const_bounds.insert(op.get_result(0), (idx_val, bound));
                    }
                }
            }
        }
    }

    let mut ptr_asserts: HashMap<pliron::value::Value, Ptr<Operation>> = HashMap::new();
    for &b_ptr in &blocks {
        let mut pending_shared_asserts: Vec<(Ptr<Operation>, pliron::value::Value, u64)> =
            Vec::new();
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            let op = op_ptr.deref(ctx);
            match opid.as_str() {
                "mir.assert" if op.get_num_operands() == 1 => {
                    if let Some(&(idx_val, bound)) = lt_const_bounds.get(&op.get_operand(0)) {
                        pending_shared_asserts.push((op_ptr, idx_val, bound));
                    }
                }
                "mir.array_element_addr" if op.get_num_operands() >= 2 => {
                    let idx_val = op.get_operand(1);
                    if let Some(pos) = pending_shared_asserts
                        .iter()
                        .rposition(|&(_, p_idx, _)| p_idx == idx_val)
                    {
                        pending_shared_asserts.remove(pos);
                    }
                }
                "mir.ptr_offset" if op.get_num_operands() >= 2 && op.get_num_results() == 1 => {
                    let base_val = op.get_operand(0);
                    let idx_val = op.get_operand(1);
                    match env.get_kind(&base_val) {
                        Some(SsaKind::SharedBasePtr {
                            shared_id, size, ..
                        }) => {
                            if let Some(pos) =
                                pending_shared_asserts
                                    .iter()
                                    .rposition(|&(_, p_idx, p_bound)| {
                                        p_idx == idx_val && p_bound == size
                                    })
                            {
                                let (assert_op, _, _) = pending_shared_asserts.remove(pos);
                                if spec.no_oob_shared.contains(&shared_id) {
                                    ptr_asserts.insert(op.get_result(0), assert_op);
                                }
                            }
                        }
                        _ => {
                            if let Some(pos) = pending_shared_asserts
                                .iter()
                                .rposition(|&(_, p_idx, _)| p_idx == idx_val)
                            {
                                pending_shared_asserts.remove(pos);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    for _pass in 0..num_blocks + 2 {
        let mut changed = false;
        for &b_ptr in &blocks {
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                if Operation::get_opid(op_ptr, ctx).to_string() == "mir.cast" {
                    let op = op_ptr.deref(ctx);
                    if op.get_num_operands() == 1
                        && op.get_num_results() == 1
                        && let Some(&assert_op) = ptr_asserts.get(&op.get_operand(0))
                        && !ptr_asserts.contains_key(&op.get_result(0))
                    {
                        ptr_asserts.insert(op.get_result(0), assert_op);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut global_strip_accesses = Vec::new();
    let mut shared_accesses = Vec::new();
    let mut proven_assert_ops = Vec::new();
    for (b_idx, &b_ptr) in blocks.iter().enumerate() {
        let dom_guard = block_guard_expr(b_idx);
        let leader_only = dom_guard.has_zero_leader_check();
        let active_threads = dom_guard.active_threads_bound(u64::from(spec.block_size));
        let residual_guard = dom_guard.custom_residual();
        let mut seen_barriers_in_block = 0usize;
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            if opid == "nvvm.barrier0" {
                seen_barriers_in_block += 1;
                continue;
            }
            let base_phase = block_phase(b_idx, seen_barriers_in_block);
            let op = op_ptr.deref(ctx);
            if opid.starts_with("nvvm.cp_async_bulk_tensor_g2s") {
                if op.get_num_operands() >= 2
                    && let (
                        Some(SsaKind::SharedBasePtr {
                            shared_id,
                            size,
                            elem_bytes,
                        }),
                        Some(SsaKind::MbarrierPtr { bar_id }),
                    ) = (
                        env.get_kind(&op.get_operand(0)),
                        env.get_kind(&op.get_operand(1)),
                    )
                {
                    if elem_bytes == 0 {
                        return Err(reject(
                            &opid,
                            "cannot determine SharedArray element byte size for async bulk copy",
                        ));
                    }
                    let copied_elems = match arrived_tx_bytes_per_bar.get(&bar_id) {
                        Some(&tx_bytes) => tx_bytes.div_ceil(elem_bytes),
                        None => size,
                    };
                    let shared_name = shared_raw_list
                        .get(shared_id)
                        .map(|(n, _)| n.clone())
                        .unwrap_or_else(|| format!("SHARED_{shared_id}"));
                    shared_accesses.push(PhasedSharedAccess {
                        kind: AccessKind::Write,
                        shared_id,
                        shared_name,
                        buffer_len: size,
                        phase: base_phase,
                        pattern: TileAccessPatternKind::LinearScan,
                        loop_bound: copied_elems,
                        leader_only,
                        active_threads,
                        guard: GuardExpr::True,
                    });
                }
                continue;
            }
            match opid.as_str() {
                "mir.ptr_offset" => {
                    if let Some(SsaKind::SharedBasePtr { .. }) = env.get_kind(&op.get_operand(0))
                        && !matches!(
                            env.get_kind(&op.get_result(0)),
                            Some(SsaKind::SharedElementPtr { .. })
                        )
                    {
                        return Err(reject(
                            "mir.ptr_offset",
                            "shared memory pointer offset must use a recognized 2D tile coordinate",
                        ));
                    }
                    if let Some(SsaKind::SlicePtr(param)) = env.get_kind(&op.get_operand(0))
                        && spec.no_oob_idx.contains(&param)
                        && !matches!(
                            env.get_kind(&op.get_result(0)),
                            Some(SsaKind::GlobalStripElementPtr { .. })
                        )
                    {
                        return Err(reject(
                            "mir.ptr_offset",
                            "2D global slice pointer offset must use a row-strip (`start + col`) or col-strip (`start + row * stride`) index",
                        ));
                    }
                }
                "mir.assert" => {
                    if op.get_num_operands() == 1
                        && let Some(SsaKind::IndexLtLen {
                            guard_param: checked_param,
                        }) = env.get_kind(&op.get_operand(0))
                        && spec.no_oob_idx.contains(&checked_param)
                    {
                        proven_assert_ops.push(op_ptr);
                    }
                }
                "mir.load" => match env.get_kind(&op.get_operand(0)) {
                    Some(SsaKind::SharedElementPtr {
                        shared_id,
                        size,
                        addr,
                    }) => {
                        if let Some(assert_op) = ptr_asserts.remove(&op.get_operand(0))
                            && !proven_assert_ops.contains(&assert_op)
                        {
                            proven_assert_ops.push(assert_op);
                        }
                        let mut phase = base_phase;
                        if let Some(&bar_id) = async_bar_for_shared.get(&shared_id)
                            && is_stage_ready(b_idx, bar_id)
                        {
                            phase += 1;
                        }
                        let (pattern, loop_bound) = addr.classify_pattern(spec.block_size);
                        let guard = if matches!(
                            pattern,
                            TileAccessPatternKind::Expr(_) | TileAccessPatternKind::ColScan { .. }
                        ) && active_threads < u64::from(spec.block_size)
                        {
                            dom_guard.clone()
                        } else {
                            residual_guard.clone()
                        };
                        let shared_name = shared_raw_list
                            .get(shared_id)
                            .map(|(n, _)| n.clone())
                            .unwrap_or_else(|| format!("SHARED_{shared_id}"));
                        shared_accesses.push(PhasedSharedAccess {
                            kind: AccessKind::Read,
                            shared_id,
                            shared_name,
                            buffer_len: size,
                            phase,
                            pattern,
                            loop_bound,
                            leader_only,
                            active_threads,
                            guard,
                        });
                    }
                    Some(SsaKind::GlobalStripElementPtr {
                        param,
                        kind,
                        offset,
                    }) => {
                        global_strip_accesses.push(GlobalStripAccess2D {
                            param,
                            kind,
                            offset,
                        });
                    }
                    _ => {}
                },
                "mir.store" => {
                    if op.get_num_operands() >= 2
                        && matches!(
                            env.get_kind(&op.get_operand(1)),
                            Some(
                                SsaKind::SharedBasePtr { .. }
                                    | SsaKind::SharedElementPtr { .. }
                                    | SsaKind::MbarrierPtr { .. }
                            )
                        )
                    {
                        return Err(reject(
                            "mir.store",
                            "verified shared-memory or mbarrier pointer cannot escape by being stored to memory",
                        ));
                    }
                    if let Some(SsaKind::SharedElementPtr {
                        shared_id,
                        size,
                        addr,
                    }) = env.get_kind(&op.get_operand(0))
                    {
                        if let Some(assert_op) = ptr_asserts.remove(&op.get_operand(0))
                            && !proven_assert_ops.contains(&assert_op)
                        {
                            proven_assert_ops.push(assert_op);
                        }
                        let mut phase = base_phase;
                        if let Some(&bar_id) = async_bar_for_shared.get(&shared_id)
                            && is_stage_ready(b_idx, bar_id)
                        {
                            phase += 1;
                        }
                        let (pattern, loop_bound) = addr.classify_pattern(spec.block_size);
                        let guard = if matches!(
                            pattern,
                            TileAccessPatternKind::Expr(_) | TileAccessPatternKind::ColScan { .. }
                        ) && active_threads < u64::from(spec.block_size)
                        {
                            dom_guard.clone()
                        } else {
                            residual_guard.clone()
                        };
                        let shared_name = shared_raw_list
                            .get(shared_id)
                            .map(|(n, _)| n.clone())
                            .unwrap_or_else(|| format!("SHARED_{shared_id}"));
                        shared_accesses.push(PhasedSharedAccess {
                            kind: AccessKind::Write,
                            shared_id,
                            shared_name,
                            buffer_len: size,
                            phase,
                            pattern,
                            loop_bound,
                            leader_only,
                            active_threads,
                            guard,
                        });
                    }
                }
                "mir.cast"
                | "nvvm.mbarrier_init_shared"
                | "nvvm.mbarrier_inval_shared"
                | "nvvm.mbarrier_arrive_shared"
                | "nvvm.mbarrier_arrive_expect_tx_shared"
                | "nvvm.mbarrier_arrive_expect_tx_cluster"
                | "nvvm.mbarrier_try_wait_parity_shared"
                | "nvvm.mbarrier_try_wait_shared"
                | "nvvm.mbarrier_test_wait_shared"
                | "nvvm.mbarrier_try_wait_parity_cluster"
                | "nvvm.mbarrier_try_wait_parity_cluster_relaxed"
                | "nvvm.fence_proxy_async_shared_cta" => {}
                _ => {
                    for operand in op.operands() {
                        match env.get_kind(&operand) {
                            Some(
                                SsaKind::SharedBasePtr { shared_id, .. }
                                | SsaKind::SharedElementPtr { shared_id, .. },
                            ) if spec.no_oob_shared.contains(&shared_id)
                                || spec.no_data_race.contains(&shared_id) =>
                            {
                                return Err(reject(
                                    &opid,
                                    format!(
                                        "verified shared-memory pointer cannot escape into unmodeled operation `{opid}`"
                                    ),
                                ));
                            }
                            Some(SsaKind::MbarrierPtr { .. }) => {
                                return Err(reject(
                                    &opid,
                                    format!(
                                        "verified mbarrier pointer cannot escape into unmodeled operation `{opid}`"
                                    ),
                                ));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    shared_accesses.sort_by_key(|a| (a.shared_id, a.phase, a.kind));

    Ok(TranslatedFunction::TiledKernel2D {
        spec,
        num_barriers: effective_num_barriers,
        barriers_uniform: final_barriers_uniform,
        global_strip_accesses,
        shared_accesses,
        proven_assert_ops,
    })
}
