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
//!   MIR Marker Calls (`__verify_*_config`, `__launch_contract_requires_len_ge_config`)
//!     │
//!     ▼  extracted here & attached to `MirFuncOp`
//!   `dialect-mir` Function Attributes (`verify_*`, `contract_requires_len_ge`)
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
    "cuda_device::__launch_contract_requires_len_ge_config",
    "cuda_device::thread::__launch_contract_requires_len_ge_config",
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
    "cuda_device::__verify_uniform_barriers_config",
    "cuda_device::thread::__verify_uniform_barriers_config",
    "cuda_device::__verify_tmem_lifecycle_config",
    "cuda_device::thread::__verify_tmem_lifecycle_config",
];

/// Map a callee path to its canonical marker leaf name when it is a
/// verification metadata marker.
pub(crate) fn verification_marker_leaf(path: &str) -> Option<&'static str> {
    Some(match path {
        "cuda_device::__launch_contract_requires_len_ge_config"
        | "cuda_device::thread::__launch_contract_requires_len_ge_config" => {
            "__launch_contract_requires_len_ge_config"
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
    /// Pairs `(lhs, rhs)` from `requires = (lhs.len() >= rhs.len())`.
    requires_len_ge: Vec<(usize, usize)>,
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
            || self.uniform_barriers
            || self.tmem_lifecycle
    }
}

/// Evaluate a const-generic `usize` argument from a MIR marker call.
fn read_const_usize_arg(arg: &GenericArgKind) -> Option<usize> {
    let GenericArgKind::Const(value) = arg else {
        return None;
    };
    let raw = match value.kind() {
        TyConstKind::Value(_, alloc) => alloc.read_uint().ok()?,
        _ => u128::from(value.eval_target_usize().ok()?),
    };
    usize::try_from(raw).ok()
}

/// Scan reachable MIR blocks for launch-contract and `#[verify(...)]` markers.
fn detect_verification_metadata(
    body: &mir::Body,
    reachable: &BTreeSet<usize>,
) -> DetectedVerificationMetadata {
    let mut meta = DetectedVerificationMetadata::default();
    for &block_idx in reachable {
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
        } else if fn_name == "__launch_contract_requires_len_ge_config"
            || fn_name.ends_with("::__launch_contract_requires_len_ge_config")
        {
            if let (Some(lhs), Some(rhs)) = (
                args.0.first().and_then(read_const_usize_arg),
                args.0.get(1).and_then(read_const_usize_arg),
            ) && !meta.requires_len_ge.contains(&(lhs, rhs))
            {
                meta.requires_len_ge.push((lhs, rhs));
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
    if !verification_meta.requires_len_ge.is_empty() {
        let formatted = verification_meta
            .requires_len_ge
            .iter()
            .map(|(lhs, rhs)| format!("{lhs}>={rhs}"))
            .collect::<Vec<_>>()
            .join(",");
        let key: Identifier = "contract_requires_len_ge".try_into().unwrap();
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
