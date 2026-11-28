/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Formal verification metadata extraction (`#[verify(...)]` and
//! `#[launch_contract(requires = ...)]`) during MIR translation.
//!
//! ```text
//!   Rust Source (`#[verify(...)]` + `#[launch_contract(...)]`)
//!     │
//!     ▼  injected by `cuda-macros`
//!   MIR Marker Calls (`__verify_*_config`, `__launch_contract_requires_step_config`)
//!     │
//!     ▼  extracted here & attached to `MirFuncOp`
//!   `dialect-mir` Function Attributes (`verify_*`, `contract_requires`)
//!     │
//!     ▼  consumed at `PostMem2Reg`
//!   `crates/lean-exporter`
//! ```

use std::{collections::BTreeSet, num::NonZero};

use dialect_mir::ops::MirFuncOp;
use pliron::{
    builtin::{
        attributes::{IntegerAttr, StringAttr},
        types::{IntegerType, Signedness},
    },
    context::Context,
    identifier::Identifier,
    op::Op,
    utils::apint::APInt,
};
use rustc_public::{
    CrateDef, mir,
    ty::{ConstantKind, GenericArgKind, RigidTy, TyConstKind, TyKind},
};

/// Compiler marker paths injected by `#[verify(...)]` and `#[launch_contract]`.
pub(crate) const VERIFICATION_METADATA_MARKERS: &[&str] = &[
    "cuda_device::__launch_contract_requires_step_config",
    "cuda_device::thread::__launch_contract_requires_step_config",
    "cuda_device::__verify_no_oob_idx_config",
    "cuda_device::thread::__verify_no_oob_idx_config",
    "cuda_device::__verify_disjoint_writes_config",
    "cuda_device::thread::__verify_disjoint_writes_config",
    "cuda_device::__verify_writes_all_elems_config",
    "cuda_device::thread::__verify_writes_all_elems_config",
    "cuda_device::__verify_no_oob_shared_config",
    "cuda_device::thread::__verify_no_oob_shared_config",
    "cuda_device::__verify_no_data_race_config",
    "cuda_device::thread::__verify_no_data_race_config",
    "cuda_device::__verify_no_global_race_config",
    "cuda_device::thread::__verify_no_global_race_config",
    "cuda_device::__verify_uniform_barriers_config",
    "cuda_device::thread::__verify_uniform_barriers_config",
    "cuda_device::__verify_tmem_lifecycle_config",
    "cuda_device::thread::__verify_tmem_lifecycle_config",
];

/// Map a callee path to its canonical marker leaf name when it is a
/// verification metadata marker.
pub(crate) fn verification_marker_leaf(path: &str) -> Option<&'static str> {
    Some(match path {
        "cuda_device::__launch_contract_requires_step_config"
        | "cuda_device::thread::__launch_contract_requires_step_config" => {
            "__launch_contract_requires_step_config"
        }
        "cuda_device::__verify_no_oob_idx_config"
        | "cuda_device::thread::__verify_no_oob_idx_config" => "__verify_no_oob_idx_config",
        "cuda_device::__verify_disjoint_writes_config"
        | "cuda_device::thread::__verify_disjoint_writes_config" => {
            "__verify_disjoint_writes_config"
        }
        "cuda_device::__verify_writes_all_elems_config"
        | "cuda_device::thread::__verify_writes_all_elems_config" => {
            "__verify_writes_all_elems_config"
        }
        "cuda_device::__verify_no_oob_shared_config"
        | "cuda_device::thread::__verify_no_oob_shared_config" => "__verify_no_oob_shared_config",
        "cuda_device::__verify_no_data_race_config"
        | "cuda_device::thread::__verify_no_data_race_config" => "__verify_no_data_race_config",
        "cuda_device::__verify_no_global_race_config"
        | "cuda_device::thread::__verify_no_global_race_config" => "__verify_no_global_race_config",
        "cuda_device::__verify_uniform_barriers_config"
        | "cuda_device::thread::__verify_uniform_barriers_config" => {
            "__verify_uniform_barriers_config"
        }
        "cuda_device::__verify_tmem_lifecycle_config"
        | "cuda_device::thread::__verify_tmem_lifecycle_config" => "__verify_tmem_lifecycle_config",
        _ => return None,
    })
}

/// Extracted `#[launch_contract]` and `#[verify(...)]` metadata markers for
/// Lean verification.
#[derive(Default)]
struct DetectedVerificationMetadata {
    /// Domain declared by `#[launch_contract(domain = D)]`.
    launch_domain: Option<u32>,
    /// Whether `u32_coordinates` was set on `#[launch_contract]`.
    u32_coordinates: bool,
    /// Formatted `requires` relations (e.g. `s1>=4*s0`, `s4>=p0*p2`).
    requires_rels: Vec<String>,
    /// Parameter indices requested in `#[verify(no_oob_idx(...))]`.
    no_oob_idx: Vec<usize>,
    /// Parameter indices requested in `#[verify(disjoint_writes(...))]`.
    disjoint_writes: Vec<usize>,
    /// Parameter indices requested in `#[verify(writes_all_elems(...))]`.
    writes_all_elems: Vec<usize>,
    /// Static `SharedArray` indices requested in `#[verify(no_oob_idx(...))]`.
    no_oob_shared: Vec<usize>,
    /// Static `SharedArray` indices requested in `#[verify(no_data_race(...))]`.
    no_data_race: Vec<usize>,
    /// Global slice parameter indices requested in `#[verify(no_data_race(...))]`.
    no_global_race: Vec<usize>,
    /// Whether `#[verify(uniform_barriers)]` was requested.
    uniform_barriers: bool,
    /// Whether `#[verify(tmem_lifecycle)]` was requested.
    tmem_lifecycle: bool,
}

impl DetectedVerificationMetadata {
    /// Returns true if any `#[verify(...)]` clause is present on this function.
    fn has_verify_obligations(&self) -> bool {
        !self.no_oob_idx.is_empty()
            || !self.disjoint_writes.is_empty()
            || !self.writes_all_elems.is_empty()
            || !self.no_oob_shared.is_empty()
            || !self.no_data_race.is_empty()
            || !self.no_global_race.is_empty()
            || self.uniform_barriers
            || self.tmem_lifecycle
    }
}

/// Evaluate a const-generic unsigned integer argument (`u8`, `u32`, `u64`, `usize`) from a MIR marker call.
fn read_const_u64_arg(arg: &GenericArgKind) -> Option<u64> {
    let GenericArgKind::Const(value) = arg else {
        return None;
    };
    let raw = match value.kind() {
        TyConstKind::Value(_, alloc) => alloc.read_uint().ok()?,
        _ => u128::from(value.eval_target_usize().ok()?),
    };
    u64::try_from(raw).ok()
}

/// Evaluate a const-generic `usize` argument from a MIR marker call.
fn read_const_usize_arg(arg: &GenericArgKind) -> Option<usize> {
    usize::try_from(read_const_u64_arg(arg)?).ok()
}

/// Order reachable MIR blocks starting along the entry prologue chain from block `0`
/// so sequential postfix `__launch_contract_requires_step_config` markers stay in source order.
fn ordered_reachable_blocks(body: &mir::Body, reachable: &BTreeSet<usize>) -> Vec<usize> {
    let mut order = Vec::with_capacity(reachable.len());
    let mut seen = BTreeSet::new();
    let mut cur = Some(0usize);
    while let Some(bb) = cur {
        if !reachable.contains(&bb) || !seen.insert(bb) {
            break;
        }
        order.push(bb);
        cur = match &body.blocks[bb].terminator.kind {
            mir::TerminatorKind::Call { target, .. } => *target,
            mir::TerminatorKind::Goto { target } => Some(*target),
            _ => None,
        };
    }
    for &bb in reachable {
        if seen.insert(bb) {
            order.push(bb);
        }
    }
    order
}

/// Apply one postfix `(tag, val)` step to `req_stack` / `requires_rels`.
fn apply_requires_step(
    tag: u64,
    val: u64,
    req_stack: &mut Vec<(String, u8)>,
    requires_rels: &mut Vec<String>,
) {
    match tag {
        0 => req_stack.push((val.to_string(), 2)),
        1 => req_stack.push((format!("p{val}"), 2)),
        2 => req_stack.push((format!("s{val}"), 2)),
        3 => {
            let (Some((rhs, rprec)), Some((lhs, lprec))) = (req_stack.pop(), req_stack.pop())
            else {
                req_stack.clear();
                return;
            };
            match val {
                0 | 1 => {
                    let op_ch = if val == 0 { '+' } else { '-' };
                    let r = if rprec == 0 { format!("({rhs})") } else { rhs };
                    req_stack.push((format!("{lhs}{op_ch}{r}"), 0));
                }
                2 => {
                    let l = if lprec < 1 { format!("({lhs})") } else { lhs };
                    let r = if rprec < 1 { format!("({rhs})") } else { rhs };
                    req_stack.push((format!("{l}*{r}"), 1));
                }
                _ => req_stack.clear(),
            }
        }
        4 => {
            let (Some((rhs, _)), Some((lhs, _))) = (req_stack.pop(), req_stack.pop()) else {
                req_stack.clear();
                return;
            };
            let cmp_str = match val {
                0 => ">=",
                1 => ">",
                2 => "<=",
                3 => "<",
                4 => "==",
                5 => "!=",
                _ => {
                    req_stack.clear();
                    return;
                }
            };
            let rel = format!("{lhs}{cmp_str}{rhs}");
            if !requires_rels.contains(&rel) {
                requires_rels.push(rel);
            }
        }
        _ => req_stack.clear(),
    }
}

/// Scan reachable MIR blocks for launch-contract and `#[verify(...)]` markers.
fn detect_verification_metadata(
    body: &mir::Body,
    reachable: &BTreeSet<usize>,
) -> DetectedVerificationMetadata {
    let mut meta = DetectedVerificationMetadata::default();
    let mut req_stack: Vec<(String, u8)> = Vec::new();
    for block_idx in ordered_reachable_blocks(body, reachable) {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };
        if def_id.krate().name.as_str() != "cuda_device" {
            continue;
        }
        let fn_name = def_id.name();
        if fn_name == "__launch_contract_config" || fn_name.ends_with("::__launch_contract_config")
        {
            if let Some(domain) = args.0.first().and_then(read_const_usize_arg) {
                meta.launch_domain = u32::try_from(domain).ok();
            }
            if args.0.get(1).and_then(read_const_usize_arg) == Some(1) {
                meta.u32_coordinates = true;
            }
        } else if fn_name == "__launch_contract_requires_step_config"
            || fn_name.ends_with("::__launch_contract_requires_step_config")
        {
            if let (Some(tag), Some(val)) = (
                args.0.first().and_then(read_const_u64_arg),
                args.0.get(1).and_then(read_const_u64_arg),
            ) {
                apply_requires_step(tag, val, &mut req_stack, &mut meta.requires_rels);
            }
        } else if fn_name == "__verify_no_oob_idx_config"
            || fn_name.ends_with("::__verify_no_oob_idx_config")
        {
            if let Some(param) = args.0.first().and_then(read_const_usize_arg)
                && !meta.no_oob_idx.contains(&param)
            {
                meta.no_oob_idx.push(param);
            }
        } else if fn_name == "__verify_disjoint_writes_config"
            || fn_name.ends_with("::__verify_disjoint_writes_config")
        {
            if let Some(param) = args.0.first().and_then(read_const_usize_arg)
                && !meta.disjoint_writes.contains(&param)
            {
                meta.disjoint_writes.push(param);
            }
        } else if (fn_name == "__verify_writes_all_elems_config"
            || fn_name.ends_with("::__verify_writes_all_elems_config"))
            && let Some(param) = args.0.first().and_then(read_const_usize_arg)
            && !meta.writes_all_elems.contains(&param)
        {
            meta.writes_all_elems.push(param);
        } else if (fn_name == "__verify_no_oob_shared_config"
            || fn_name.ends_with("::__verify_no_oob_shared_config"))
            && let Some(shared_idx) = args.0.first().and_then(read_const_usize_arg)
            && !meta.no_oob_shared.contains(&shared_idx)
        {
            meta.no_oob_shared.push(shared_idx);
        } else if (fn_name == "__verify_no_data_race_config"
            || fn_name.ends_with("::__verify_no_data_race_config"))
            && let Some(shared_idx) = args.0.first().and_then(read_const_usize_arg)
            && !meta.no_data_race.contains(&shared_idx)
        {
            meta.no_data_race.push(shared_idx);
        } else if (fn_name == "__verify_no_global_race_config"
            || fn_name.ends_with("::__verify_no_global_race_config"))
            && let Some(param) = args.0.first().and_then(read_const_usize_arg)
            && !meta.no_global_race.contains(&param)
        {
            meta.no_global_race.push(param);
        } else if fn_name == "__verify_uniform_barriers_config"
            || fn_name.ends_with("::__verify_uniform_barriers_config")
        {
            meta.uniform_barriers = true;
        } else if fn_name == "__verify_tmem_lifecycle_config"
            || fn_name.ends_with("::__verify_tmem_lifecycle_config")
        {
            meta.tmem_lifecycle = true;
        }
    }
    meta
}

/// Detect and attach `#[verify(...)]` and `#[launch_contract]` verification
/// attributes onto `mir_func_op`.
pub(crate) fn attach_verification_attributes(
    ctx: &mut Context,
    mir_func_op: MirFuncOp,
    body: &mir::Body,
    reachable: &BTreeSet<usize>,
) {
    let verification_meta = detect_verification_metadata(body, reachable);
    if let Some(domain) = verification_meta.launch_domain {
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let width = NonZero::new(32).unwrap();
        let key: Identifier = "launch_contract_domain".try_into().unwrap();
        mir_func_op.get_operation().deref_mut(ctx).attributes.set(
            key,
            IntegerAttr::new(u32_ty, APInt::from_u32(domain, width)),
        );
    }
    if verification_meta.u32_coordinates {
        let key: Identifier = "launch_contract_u32_coords".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, StringAttr::new("true".to_string()));
    }
    if !verification_meta.requires_rels.is_empty() {
        let formatted = verification_meta.requires_rels.join(",");
        let key: Identifier = "contract_requires".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, StringAttr::new(formatted));
    }
    if verification_meta.has_verify_obligations() {
        let mut op_mut = mir_func_op.get_operation().deref_mut(ctx);
        for (attr_name, indices) in [
            ("verify_no_oob_idx", &verification_meta.no_oob_idx),
            ("verify_disjoint_writes", &verification_meta.disjoint_writes),
            (
                "verify_writes_all_elems",
                &verification_meta.writes_all_elems,
            ),
            ("verify_no_oob_shared", &verification_meta.no_oob_shared),
            ("verify_no_data_race", &verification_meta.no_data_race),
            ("verify_no_global_race", &verification_meta.no_global_race),
        ] {
            if !indices.is_empty() {
                let formatted = indices
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let key: Identifier = attr_name.try_into().unwrap();
                op_mut.attributes.set(key, StringAttr::new(formatted));
            }
        }
        if verification_meta.uniform_barriers {
            let key: Identifier = "verify_uniform_barriers".try_into().unwrap();
            op_mut.attributes.set(key, StringAttr::new("true".into()));
        }
        if verification_meta.tmem_lifecycle {
            let key: Identifier = "verify_tmem_lifecycle".try_into().unwrap();
            op_mut.attributes.set(key, StringAttr::new("true".into()));
        }
    }
}
