/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Barrier-free 1D global-slice kernel transcriber (`vecadd`, `warp_reduce_sum` ->
//! `KernelSummary1D` + `ControlFlowCfg` for `WritesOnAllActivePaths`).

use std::collections::{BTreeSet, HashMap};

use dialect_mir::types::{MirDisjointSliceType, MirSliceType};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    r#type::Typed,
};

use super::{
    AccessKind, ExtractedAccess, TranslatedFunction,
    ir_utils::{
        BlockGraph, attribute_field_index, bin_kinds, decode_int_constant, eval_int_cast, op_kind,
        op_name,
    },
};
use crate::{
    ExportError, reject,
    spec::{KernelVerificationSpec, get_string_attr},
    ssa::{CfgBranchCond, SsaEnv, SsaKind, TranscribedCfgBlock, TranscribedControlFlowCfg},
};

/// Compute the dominating `< param.len()` guard parameter index for `block_idx` in a 1D kernel.
/// Returns `Some(param)` only when every incoming CFG edge into `block_idx` carries `Some(param)`.
fn compute_block_guard_1d(
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
    incoming.all(|g| g == Some(first)).then_some(first)
}

/// Mutable state threaded through 1D kernel SSA propagation and CFG transcription.
/// Tracks reachable basic blocks, edge-propagated `< len` guards, and `Option` join inputs.
struct Kernel1DWalk {
    /// Set of reachable basic-block indices from entry `0`.
    reachable: BTreeSet<usize>,
    /// Propagated `< param.len()` guard parameter along each `(src, dst)` CFG edge.
    edge_guards: HashMap<(usize, usize), Option<usize>>,
    /// Incoming `Option` join `SsaKind` along single-argument `mir.goto` edges.
    block_arg_incomings: HashMap<(usize, usize), SsaKind>,
}

/// Optional final-pass output collectors for [`step_kernel_1d_block`].
type Kernel1DFinalOut<'a> = (
    &'a mut Vec<ExtractedAccess>,
    &'a mut Vec<Ptr<Operation>>,
    &'a mut TranscribedCfgBlock,
    &'a mut HashMap<usize, Vec<usize>>,
    &'a mut HashMap<usize, Vec<usize>>,
    &'a mut HashMap<usize, Vec<usize>>,
);

/// Route a 1D `< len` guard onto either the true or false outgoing `cond_br` edge.
/// Returns `(true_guard, false_guard)`.
fn route_1d_guard(
    guard_param: usize,
    fallback: Option<usize>,
    on_true: bool,
) -> (Option<usize>, Option<usize>) {
    if on_true {
        (Some(guard_param), fallback)
    } else {
        (fallback, Some(guard_param))
    }
}

/// Transcribe a barrier-free 1D slice GPU kernel (`vecadd`, `warp_reduce_sum`) into
/// [`TranslatedFunction::Kernel`] and [`TranscribedControlFlowCfg`].
pub(super) fn translate_slice_1d_kernel(
    ctx: &Context,
    function: Ptr<Operation>,
    arity: usize,
    spec: KernelVerificationSpec,
) -> Result<TranslatedFunction, ExportError> {
    let blocks: Vec<Ptr<BasicBlock>> = function
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .collect();
    let Some(&entry_block) = blocks.first() else {
        return Err(reject("kernel", "empty kernel region"));
    };
    let graph = BlockGraph::build(ctx, blocks);

    let mut env = SsaEnv::new();
    for (idx, arg) in entry_block.deref(ctx).arguments().enumerate() {
        let arg_ty = arg.get_type(ctx).deref(ctx);
        let is_slice = arg_ty.downcast_ref::<MirSliceType>().is_some()
            || arg_ty.downcast_ref::<MirDisjointSliceType>().is_some();
        env.bind_arg(
            arg,
            idx,
            if is_slice {
                SsaKind::BufferParam(idx)
            } else {
                SsaKind::DataValue
            },
        );
    }
    let mut walk = Kernel1DWalk {
        reachable: BTreeSet::from([0]),
        edge_guards: HashMap::new(),
        block_arg_incomings: HashMap::new(),
    };

    for _pass in 0..graph.len() + 2 {
        let snap = (
            env.len(),
            walk.reachable.len(),
            walk.edge_guards.clone(),
            walk.block_arg_incomings.clone(),
        );
        for (b_idx, &b_ptr) in graph.blocks.iter().enumerate() {
            if !walk.reachable.contains(&b_idx) {
                continue;
            }
            let cur_guard = compute_block_guard_1d(b_idx, &walk.edge_guards);
            let block = b_ptr.deref(ctx);
            if b_idx != 0
                && block.get_num_arguments() == 1
                && let Some((param, offset, guard_param)) = walk
                    .block_arg_incomings
                    .iter()
                    .filter(|((_, dst), _)| *dst == b_idx)
                    .find_map(|(_, inc)| match *inc {
                        SsaKind::OptionSome {
                            param,
                            offset,
                            guard_param,
                        } => Some((param, offset, guard_param)),
                        _ => None,
                    })
            {
                env.bind_result(
                    block.get_argument(0),
                    SsaKind::OptionJoin {
                        param,
                        offset,
                        guard_param,
                    },
                );
            }
            let _ = step_kernel_1d_block(
                ctx, b_idx, b_ptr, &graph, arity, &spec, cur_guard, &mut env, &mut walk, None,
            );
        }
        if (
            env.len(),
            walk.reachable.len(),
            walk.edge_guards.clone(),
            walk.block_arg_incomings.clone(),
        ) == snap
        {
            break;
        }
    }

    let mut accesses = Vec::new();
    let mut proven_assert_ops = Vec::new();
    let mut cfg_blocks = vec![TranscribedCfgBlock::default(); graph.len()];
    let mut write_blocks_per_param: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut option_some_targets: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut direct_lt_targets: HashMap<usize, Vec<usize>> = HashMap::new();

    for (b_idx, &b_ptr) in graph.blocks.iter().enumerate() {
        if !walk.reachable.contains(&b_idx) {
            continue;
        }
        if b_idx != 0 && b_ptr.deref(ctx).get_num_arguments() > 1 {
            return Err(reject(
                "kernel CFG",
                "only single-argument Option join blocks are supported in v0",
            ));
        }
        let cur_guard = compute_block_guard_1d(b_idx, &walk.edge_guards);
        step_kernel_1d_block(
            ctx,
            b_idx,
            b_ptr,
            &graph,
            arity,
            &spec,
            cur_guard,
            &mut env,
            &mut walk,
            Some((
                &mut accesses,
                &mut proven_assert_ops,
                &mut cfg_blocks[b_idx],
                &mut write_blocks_per_param,
                &mut option_some_targets,
                &mut direct_lt_targets,
            )),
        )?;
    }

    let raw_obligations: Vec<(usize, Vec<usize>, Vec<usize>)> = spec
        .writes_all_elems
        .iter()
        .filter_map(|&param| {
            let writes = write_blocks_per_param.get(&param)?.clone();
            let entries = option_some_targets
                .get(&param)
                .or_else(|| direct_lt_targets.get(&param))
                .cloned()
                .unwrap_or_default();
            (!writes.is_empty()).then_some((param, entries, writes))
        })
        .collect();

    let mut protected = BTreeSet::new();
    for (_, entries, writes) in &raw_obligations {
        protected.extend(entries.iter().copied());
        protected.extend(writes.iter().copied());
    }
    let mut control_flow_cfg = TranscribedControlFlowCfg {
        num_shared: 0,
        num_barriers: 0,
        blocks: cfg_blocks,
    };
    let remap = control_flow_cfg.compact(&protected);
    let write_path_obligations = raw_obligations
        .into_iter()
        .map(|(param, entries, writes)| {
            let new_entries: Vec<usize> = entries
                .into_iter()
                .filter_map(|b| remap.get(&b).copied())
                .collect();
            let new_writes: Vec<usize> = writes
                .into_iter()
                .filter_map(|b| remap.get(&b).copied())
                .collect();
            (param, new_entries, new_writes)
        })
        .collect();

    Ok(TranslatedFunction::Kernel {
        arity,
        spec,
        accesses,
        control_flow_cfg,
        write_path_obligations,
        proven_assert_ops,
    })
}

/// Walk the operations of one basic block in a 1D verified kernel, updating [`SsaEnv`]
/// and (on the final pass) recording accesses, proven `mir.assert`s, and `CfgBlock` edges.
#[allow(clippy::too_many_arguments)]
fn step_kernel_1d_block(
    ctx: &Context,
    block_idx: usize,
    block_ptr: Ptr<BasicBlock>,
    graph: &BlockGraph,
    arity: usize,
    spec: &KernelVerificationSpec,
    current_guard: Option<usize>,
    env: &mut SsaEnv,
    walk: &mut Kernel1DWalk,
    mut final_out: Option<Kernel1DFinalOut<'_>>,
) -> Result<(), ExportError> {
    for op_ptr in block_ptr.deref(ctx).iter(ctx) {
        let opid = op_name(ctx, op_ptr);
        let op = op_ptr.deref(ctx);
        match opid.as_str() {
            "mir.storage_live" | "mir.storage_dead" | "mir.dbg_value" | "mir.dbg_declare" => {}
            "mir.float_constant" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            "nvvm.read_ptx_sreg_tid_x" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::SregTidX { bit_width: 32 });
                }
            }
            "nvvm.read_ptx_sreg_ctaid_x" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::SregCtaidX { bit_width: 32 });
                }
            }
            "nvvm.read_ptx_sreg_ntid_x" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::SregNtidX { bit_width: 32 });
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
                    && let Some(kind) = op_kind(env, &op, 0)
                {
                    let mapped = eval_int_cast(ctx, &op, kind, spec.block_size, spec.block_y);
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
                let mapped = match (
                    op_kind(env, &op, 0),
                    attribute_field_index(&op.attributes, "index"),
                ) {
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
                }
                .ok_or_else(|| {
                    reject(
                        "mir.extract_field",
                        "unsupported field extraction in verified kernel",
                    )
                })?;
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.constant" | "builtin.constant" => {
                let kind =
                    decode_int_constant(&op, &opid).map_or(SsaKind::DataValue, |(width, value)| {
                        SsaKind::ConstInt { width, value }
                    });
                env.bind_result(op.get_result(0), kind);
            }
            "mir.checked_add" => match bin_kinds(env, &op) {
                (Some(SsaKind::Index1D { offset }), Some(SsaKind::ConstInt { value: c, .. }))
                | (Some(SsaKind::ConstInt { value: c, .. }), Some(SsaKind::Index1D { offset })) => {
                    let new_offset = offset.saturating_add(u64::try_from(c).unwrap_or(u64::MAX));
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
            },
            "mir.add" | "mir.sub" | "mir.mul" => {
                let mapped = match (opid.as_str(), bin_kinds(env, &op)) {
                    (
                        "mir.mul",
                        (
                            Some(SsaKind::SregCtaidX { bit_width: w1 }),
                            Some(SsaKind::SregNtidX { bit_width: w2 }),
                        )
                        | (
                            Some(SsaKind::SregNtidX { bit_width: w2 }),
                            Some(SsaKind::SregCtaidX { bit_width: w1 }),
                        ),
                    ) => {
                        if (w1 < 64 || w2 < 64) && !spec.u32_coordinates {
                            return Err(reject(
                                "mir.mul",
                                "32-bit `blockIdx_x() * blockDim_x()` can wrap past `u32::MAX` on large grids; cast to `usize` before multiplying or add `u32_coordinates` to `#[launch_contract]`",
                            ));
                        }
                        SsaKind::BlockBase1D { bit_width: w1 }
                    }
                    (
                        "mir.mul",
                        (
                            Some(SsaKind::SregCtaidX { bit_width: w1 }),
                            Some(SsaKind::ConstInt {
                                width: w2,
                                value: c,
                            }),
                        )
                        | (
                            Some(SsaKind::ConstInt {
                                width: w2,
                                value: c,
                            }),
                            Some(SsaKind::SregCtaidX { bit_width: w1 }),
                        ),
                    ) if c == u128::from(spec.block_size) => {
                        if (w1 < 64 || w2 < 64) && !spec.u32_coordinates {
                            return Err(reject(
                                "mir.mul",
                                "32-bit `blockIdx_x() * BLOCK_SIZE` can wrap past `u32::MAX` on large grids; cast to `usize` before multiplying or add `u32_coordinates` to `#[launch_contract]`",
                            ));
                        }
                        SsaKind::BlockBase1D { bit_width: w1 }
                    }
                    (
                        "mir.add",
                        (
                            Some(SsaKind::BlockBase1D { bit_width: w1 }),
                            Some(SsaKind::SregTidX { bit_width: w2 }),
                        )
                        | (
                            Some(SsaKind::SregTidX { bit_width: w2 }),
                            Some(SsaKind::BlockBase1D { bit_width: w1 }),
                        ),
                    ) => {
                        if (w1 < 64 || w2 < 64) && !spec.u32_coordinates {
                            return Err(reject(
                                "mir.add",
                                "32-bit `blockIdx_x() * blockDim_x() + threadIdx_x()` can wrap past `u32::MAX`; cast operands to `usize` first or add `u32_coordinates` to `#[launch_contract]`",
                            ));
                        }
                        SsaKind::Index1D { offset: 0 }
                    }
                    (
                        "mir.add",
                        (
                            Some(SsaKind::Index1D { offset }),
                            Some(SsaKind::ConstInt { value: c, .. }),
                        )
                        | (
                            Some(SsaKind::ConstInt { value: c, .. }),
                            Some(SsaKind::Index1D { offset }),
                        ),
                    ) => SsaKind::Index1D {
                        offset: offset.saturating_add(u64::try_from(c).unwrap_or(u64::MAX)),
                    },
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.eq" => {
                let mapped = match bin_kinds(env, &op) {
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
                let mapped = match bin_kinds(env, &op) {
                    (
                        Some(SsaKind::Index1D { offset: 0 }),
                        Some(SsaKind::SliceLen(guard_param)),
                    ) => SsaKind::IndexLtLen { guard_param },
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.not" => {
                let mapped = match op_kind(env, &op, 0) {
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
            "mir.ptr_offset" => match bin_kinds(env, &op) {
                (Some(SsaKind::SlicePtr(param)), Some(SsaKind::Index1D { offset })) => {
                    env.bind_result(op.get_result(0), SsaKind::ElementPtr { param, offset });
                }
                _ => {
                    return Err(reject(
                        "mir.ptr_offset",
                        "pointer offset must index a kernel slice parameter by the 1D thread index",
                    ));
                }
            },
            "mir.construct_enum" => {
                match attribute_field_index(&op.attributes, "construct_enum_variant_index") {
                    Some(0) => {
                        env.bind_result(op.get_result(0), SsaKind::OptionNone);
                    }
                    Some(1) if op.get_num_operands() == 1 => {
                        let Some(SsaKind::ElementPtr { param, offset }) = op_kind(env, &op, 0)
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
                let Some(SsaKind::OptionJoin {
                    param,
                    offset,
                    guard_param,
                }) = op_kind(env, &op, 0)
                else {
                    return Err(reject(
                        "mir.get_discriminant",
                        "unsupported discriminant operand in verified kernel",
                    ));
                };
                env.bind_result(
                    op.get_result(0),
                    SsaKind::OptionDiscriminant {
                        param,
                        offset,
                        guard_param,
                    },
                );
            }
            "mir.enum_payload" => {
                let Some(SsaKind::OptionJoin { param, offset, .. }) = op_kind(env, &op, 0) else {
                    return Err(reject(
                        "mir.enum_payload",
                        "unsupported enum payload extraction in verified kernel",
                    ));
                };
                env.bind_result(op.get_result(0), SsaKind::ElementPtr { param, offset });
            }
            "mir.assert" => {
                if let Some((_, proven_asserts, _, _, _, _)) = final_out.as_mut()
                    && current_guard.is_some()
                    && let Some(SsaKind::IndexLtLen {
                        guard_param: checked_param,
                    }) = op_kind(env, &op, 0)
                    && spec.no_oob_idx.contains(&checked_param)
                {
                    proven_asserts.push(op_ptr);
                }
            }
            "mir.load" | "mir.store" => {
                let is_load = opid == "mir.load";
                let Some(SsaKind::ElementPtr { param, offset }) = op_kind(env, &op, 0) else {
                    let msg = if is_load {
                        "memory load does not target a recognized slice element pointer"
                    } else {
                        "memory store does not target a recognized slice element pointer"
                    };
                    return Err(reject(&opid, msg));
                };
                let Some(guard_param) = current_guard else {
                    let (act, noun) = if is_load {
                        ("read from", "read")
                    } else {
                        ("write to", "write")
                    };
                    return Err(reject(
                        &opid,
                        format!(
                            "unguarded {act} parameter {param}: every buffer {noun} must be dominated by a strict `< len` guard"
                        ),
                    ));
                };
                if let Some((accesses, _, _, write_blks, _, _)) = final_out.as_mut() {
                    accesses.push(ExtractedAccess {
                        kind: if is_load {
                            AccessKind::Read
                        } else {
                            AccessKind::Write
                        },
                        target_param: param,
                        guard_param,
                        index_offset: offset,
                    });
                    if !is_load && offset == 0 {
                        let list = write_blks.entry(param).or_default();
                        if !list.contains(&block_idx) {
                            list.push(block_idx);
                        }
                    }
                }
                if is_load {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            "mir.goto" => {
                if let Some(dst_idx) = graph.idx_of(op.get_successor(0)) {
                    walk.reachable.insert(dst_idx);
                    walk.edge_guards.insert((block_idx, dst_idx), current_guard);
                    if op.get_num_operands() == 1
                        && let Some(kind) = op_kind(env, &op, 0)
                    {
                        walk.block_arg_incomings.insert((block_idx, dst_idx), kind);
                    }
                    if let Some((_, _, cfg_blk, _, _, _)) = final_out.as_mut() {
                        cfg_blk.succs.push((CfgBranchCond::Always, dst_idx));
                    }
                }
            }
            "mir.cond_br" => {
                let cond_k = op_kind(env, &op, 0);
                let (true_guard, false_guard) = match &cond_k {
                    Some(SsaKind::IndexLtLen { guard_param })
                    | Some(SsaKind::OptionIsSome { guard_param, .. }) => {
                        route_1d_guard(*guard_param, current_guard, true)
                    }
                    Some(SsaKind::IndexGeLen { guard_param }) => {
                        route_1d_guard(*guard_param, current_guard, false)
                    }
                    _ => (current_guard, current_guard),
                };
                let t_opt = graph.idx_of(op.get_successor(0));
                let f_opt = graph.idx_of(op.get_successor(1));
                if let Some(dst_idx) = t_opt {
                    walk.reachable.insert(dst_idx);
                    walk.edge_guards.insert((block_idx, dst_idx), true_guard);
                }
                if let Some(dst_idx) = f_opt {
                    walk.reachable.insert(dst_idx);
                    walk.edge_guards.insert((block_idx, dst_idx), false_guard);
                }
                if let Some((_, _, cfg_blk, _, opt_targets, lt_targets)) = final_out.as_mut() {
                    if let (Some(t_idx), Some(f_idx)) = (t_opt, f_opt) {
                        cfg_blk.succs.push((CfgBranchCond::Unknown, t_idx));
                        cfg_blk.succs.push((CfgBranchCond::Unknown, f_idx));
                    }
                    match cond_k {
                        Some(SsaKind::OptionIsSome {
                            param,
                            offset: 0,
                            guard_param,
                        }) if param == guard_param => {
                            if let Some(t_idx) = t_opt {
                                opt_targets.entry(param).or_default().push(t_idx);
                            }
                        }
                        Some(SsaKind::IndexLtLen { guard_param }) => {
                            if let Some(t_idx) = t_opt {
                                lt_targets.entry(guard_param).or_default().push(t_idx);
                            }
                        }
                        Some(SsaKind::IndexGeLen { guard_param }) => {
                            if let Some(f_idx) = f_opt {
                                lt_targets.entry(guard_param).or_default().push(f_idx);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "mir.return" => {
                if let Some((_, _, cfg_blk, _, _, _)) = final_out.as_mut() {
                    cfg_blk.is_return = true;
                }
            }
            "mir.unreachable" => {}
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
