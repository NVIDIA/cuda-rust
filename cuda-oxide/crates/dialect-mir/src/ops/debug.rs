/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug-only MIR operations.
//!
//! These ops do not change program behavior. They carry source-level debugger
//! facts through MIR transformations until `mir-lower` turns them into LLVM
//! debug intrinsics.

use pliron::{
    builtin::{
        attributes::StringAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    location::{Located, Location},
    op::Op,
    operation::Operation,
    result::Error,
    uniqued_any,
    value::Value,
};
use pliron_derive::pliron_op;

// Keep these keys in sync with `llvm_export::ops`; the metadata is deliberately
// stored as generic string attributes so the MIR dialect does not need to depend
// on LLVM export data structures.
const DEBUG_LOCAL_NAME_KEY: &str = "cuda_oxide_debug_local_name";
const DEBUG_LOCAL_ARG_KEY: &str = "cuda_oxide_debug_local_arg";
const DEBUG_LOCAL_TYPE_KEY: &str = "cuda_oxide_debug_local_type";
const DEBUG_LOCAL_DECL_FILE_KEY: &str = "cuda_oxide_debug_local_decl_file";
const DEBUG_LOCAL_DECL_LINE_KEY: &str = "cuda_oxide_debug_local_decl_line";
const DEBUG_LOCAL_DECL_COLUMN_KEY: &str = "cuda_oxide_debug_local_decl_column";
const DEBUG_LOCAL_SCOPE_KEY: &str = "cuda_oxide_debug_local_scope";
const DEBUG_FRAGMENT_COUNT_KEY: &str = "cuda_oxide_debug_fragment_count";
const DEBUG_FRAGMENT_FIELDS: &[&str] = &[
    "name",
    "arg",
    "type",
    "offset_bits",
    "size_bits",
    "scope",
    "file",
    "line",
    "column",
];
const DEBUG_WHOLE_ALIAS_COUNT_KEY: &str = "cuda_oxide_debug_whole_alias_count";
const DEBUG_WHOLE_ALIAS_FIELDS: &[&str] =
    &["name", "arg", "type", "scope", "file", "line", "column"];
const DEBUG_PROJECTED_COUNT_KEY: &str = "cuda_oxide_debug_projected_count";
const DEBUG_PROJECTED_FIELDS: &[&str] = &[
    "name", "arg", "type", "deref", "offset", "scope", "file", "line", "column",
];
const DEBUG_VALUE_EXPRESSION_KEY: &str = "cuda_oxide_debug_value_expression";

const DEBUG_LOCAL_ATTR_KEYS: &[&str] = &[
    DEBUG_LOCAL_NAME_KEY,
    DEBUG_LOCAL_ARG_KEY,
    DEBUG_LOCAL_TYPE_KEY,
    DEBUG_LOCAL_DECL_FILE_KEY,
    DEBUG_LOCAL_DECL_LINE_KEY,
    DEBUG_LOCAL_DECL_COLUMN_KEY,
    DEBUG_LOCAL_SCOPE_KEY,
    DEBUG_VALUE_EXPRESSION_KEY,
];

/// Value-based source-local debug record.
///
/// This is the MIR-side equivalent of LLVM's `dbg.value`: "at this point in
/// the program, this source local is represented by this SSA value." It is
/// emitted by `mem2reg` when a debug-tagged local slot is promoted.
///
/// This op is debug-only, but Pliron currently sees its operand as a normal
/// use. The current pipeline does not run Pliron DCE after this salvage point.
/// If that changes, DCE needs a non-semantic debug-use story so a `dbg_value`
/// does not keep otherwise-dead computation alive.
#[pliron_op(
    name = "mir.dbg_value",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<0>]
)]
pub struct MirDbgValueOp;

impl MirDbgValueOp {
    /// Create a debug-value op for `value`.
    pub fn new(ctx: &mut Context, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        MirDbgValueOp { op }
    }

    /// The SSA value currently representing the source local.
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Verify for MirDbgValueOp {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        Ok(())
    }
}

/// Multi-value source-local debug record.
///
/// LLVM models a source value computed from multiple SSA values with a
/// `DIArgList` plus a `DIExpression` containing `DW_OP_LLVM_arg` selectors.
/// This MIR marker carries the ordered SSA operands; the expression itself is
/// stored as generic debug metadata so `dialect-mir` stays independent of the
/// LLVM exporter data structures.
#[pliron_op(
    name = "mir.dbg_value_list",
    format,
    interfaces = [NResultsInterface<0>]
)]
pub struct MirDbgValueListOp;

impl MirDbgValueListOp {
    /// Create a multi-value debug record from an ordered list of SSA values.
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0);
        MirDbgValueListOp { op }
    }

    /// The ordered SSA values referenced by `DW_OP_LLVM_arg` operations.
    pub fn values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Verify for MirDbgValueListOp {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        Ok(())
    }
}

/// If `slot` is a debug-tagged alloca result, build a `mir.dbg_value` for the
/// promoted SSA `value` and copy the source-local metadata onto it.
pub(crate) fn debug_value_for_promoted_slot(
    ctx: &mut Context,
    slot: Value,
    value: Value,
    loc: Location,
) -> Option<MirDbgValueOp> {
    // A load-before-store of a debug-tagged slot reaches promotion with the
    // synthesized `undef` default-def (mem2reg's get_or_create_default_def).
    // Recording `dbg.value(undef, var)` would tell the debugger the local is
    // poison at a point the source may still treat as live, so skip it; the
    // declaration metadata copied below already describes the variable.
    if let Some(def) = value.defining_op()
        && Operation::get_opid(def, ctx) == crate::ops::constants::MirUndefOp::get_opid_static()
    {
        return None;
    }

    let slot_op = slot.defining_op()?;
    if !has_debug_local_attrs(ctx, slot_op) {
        return None;
    }

    let dbg_value = MirDbgValueOp::new(ctx, value);
    copy_debug_local_attrs(ctx, slot_op, dbg_value.get_operation());
    stamp_declaration_location(ctx, slot_op, dbg_value.get_operation());

    let value_loc = if source_position_from_location(ctx, &loc).is_some() {
        loc
    } else {
        // Many promoted loads/stores are compiler plumbing and carry no span of
        // their own. Keep the debug record alive by falling back to the source
        // local's declaration slot; the declaration attrs above still preserve
        // the variable's real `DILocalVariable` line.
        slot_op.deref(ctx).loc().clone()
    };
    dbg_value.get_operation().deref_mut(ctx).set_loc(value_loc);
    Some(dbg_value)
}

/// Remove source-variable debug metadata from a local's backing slot.
///
/// Use this when a statement-level debug assignment cannot be emitted safely.
/// It prevents misleading `llvm.dbg.declare` records while keeping the
/// local-memory metadata used by diagnostics.
pub fn suppress_debug_local_from_slot(ctx: &mut Context, slot: Value) {
    let Some(slot_op) = slot.defining_op() else {
        return;
    };

    for key in DEBUG_LOCAL_ATTR_KEYS {
        remove_string_attr(ctx, slot_op, key);
    }
    remove_indexed_debug_attrs(
        ctx,
        slot_op,
        DEBUG_WHOLE_ALIAS_COUNT_KEY,
        DEBUG_WHOLE_ALIAS_FIELDS,
        debug_whole_alias_key,
    );
    remove_indexed_debug_attrs(
        ctx,
        slot_op,
        DEBUG_FRAGMENT_COUNT_KEY,
        DEBUG_FRAGMENT_FIELDS,
        debug_fragment_key,
    );
    remove_indexed_debug_attrs(
        ctx,
        slot_op,
        DEBUG_PROJECTED_COUNT_KEY,
        DEBUG_PROJECTED_FIELDS,
        debug_projected_key,
    );
}

fn has_debug_local_attrs(ctx: &Context, op: Ptr<Operation>) -> bool {
    let has_whole_local = get_string_attr(ctx, op, DEBUG_LOCAL_NAME_KEY).is_some()
        && get_string_attr(ctx, op, DEBUG_LOCAL_TYPE_KEY).is_some();
    let has_fragments = get_string_attr(ctx, op, DEBUG_FRAGMENT_COUNT_KEY)
        .and_then(|count| count.parse::<usize>().ok())
        .is_some_and(|count| (1..=1024).contains(&count));
    let has_whole_aliases = get_string_attr(ctx, op, DEBUG_WHOLE_ALIAS_COUNT_KEY)
        .and_then(|count| count.parse::<usize>().ok())
        .is_some_and(|count| (1..=1024).contains(&count));
    has_whole_local || has_whole_aliases || has_fragments
}

pub(crate) fn copy_debug_local_attrs(ctx: &mut Context, from: Ptr<Operation>, to: Ptr<Operation>) {
    for key in DEBUG_LOCAL_ATTR_KEYS {
        if let Some(value) = get_string_attr(ctx, from, key) {
            set_string_attr(ctx, to, key, value);
        }
    }
    copy_debug_fragment_attrs(ctx, from, to);
    copy_indexed_debug_attrs(
        ctx,
        from,
        to,
        DEBUG_WHOLE_ALIAS_COUNT_KEY,
        DEBUG_WHOLE_ALIAS_FIELDS,
        debug_whole_alias_key,
    );
}

fn copy_debug_fragment_attrs(ctx: &mut Context, from: Ptr<Operation>, to: Ptr<Operation>) {
    let Some(count_text) = get_string_attr(ctx, from, DEBUG_FRAGMENT_COUNT_KEY) else {
        return;
    };
    let Ok(count) = count_text.parse::<usize>() else {
        return;
    };
    if count == 0 || count > 1024 {
        return;
    }

    set_string_attr(ctx, to, DEBUG_FRAGMENT_COUNT_KEY, count_text);
    for index in 0..count {
        for field in DEBUG_FRAGMENT_FIELDS {
            let key = debug_fragment_key(index, field);
            if let Some(value) = get_string_attr(ctx, from, &key) {
                set_string_attr(ctx, to, &key, value);
            }
        }
    }
}

fn debug_fragment_key(index: usize, field: &str) -> String {
    format!("cuda_oxide_debug_fragment_{index}_{field}")
}

fn debug_projected_key(index: usize, field: &str) -> String {
    format!("cuda_oxide_debug_projected_{index}_{field}")
}

fn debug_whole_alias_key(index: usize, field: &str) -> String {
    format!("cuda_oxide_debug_whole_alias_{index}_{field}")
}

fn copy_indexed_debug_attrs(
    ctx: &mut Context,
    from: Ptr<Operation>,
    to: Ptr<Operation>,
    count_key: &str,
    fields: &[&str],
    key: fn(usize, &str) -> String,
) {
    let Some(count_text) = get_string_attr(ctx, from, count_key) else {
        return;
    };
    let Ok(count) = count_text.parse::<usize>() else {
        return;
    };
    if count == 0 || count > 1024 {
        return;
    }

    set_string_attr(ctx, to, count_key, count_text);
    for index in 0..count {
        for field in fields {
            let indexed_key = key(index, field);
            if let Some(value) = get_string_attr(ctx, from, &indexed_key) {
                set_string_attr(ctx, to, &indexed_key, value);
            }
        }
    }
}

fn remove_indexed_debug_attrs(
    ctx: &mut Context,
    op: Ptr<Operation>,
    count_key: &str,
    fields: &[&str],
    key: fn(usize, &str) -> String,
) {
    let count = get_string_attr(ctx, op, count_key)
        .and_then(|count| count.parse::<usize>().ok())
        .filter(|count| *count <= 1024)
        .unwrap_or(0);
    remove_string_attr(ctx, op, count_key);
    for index in 0..count {
        for field in fields {
            remove_string_attr(ctx, op, &key(index, field));
        }
    }
}

fn set_string_attr(ctx: &mut Context, op: Ptr<Operation>, key: &str, value: String) {
    let key = Identifier::try_new(key.to_string()).expect("valid identifier");
    op.deref_mut(ctx)
        .attributes
        .set(key, StringAttr::new(value));
}

fn get_string_attr(ctx: &Context, op: Ptr<Operation>, key: &str) -> Option<String> {
    let key = Identifier::try_new(key.to_string()).expect("valid identifier");
    op.deref(ctx)
        .attributes
        .get::<StringAttr>(&key)
        .map(|a| String::from((*a).clone()))
}

fn remove_string_attr(ctx: &mut Context, op: Ptr<Operation>, key: &str) {
    let key = Identifier::try_new(key.to_string()).expect("valid identifier");
    op.deref_mut(ctx).attributes.0.remove(&key);
}

fn stamp_declaration_location(ctx: &mut Context, from: Ptr<Operation>, to: Ptr<Operation>) {
    let decl_loc = from.deref(ctx).loc().clone();
    let Some((file, line, column)) = source_position_from_location(ctx, &decl_loc) else {
        return;
    };

    set_string_attr(ctx, to, DEBUG_LOCAL_DECL_FILE_KEY, file);
    set_string_attr(ctx, to, DEBUG_LOCAL_DECL_LINE_KEY, line.to_string());
    set_string_attr(ctx, to, DEBUG_LOCAL_DECL_COLUMN_KEY, column.to_string());
}

fn source_position_from_location(ctx: &Context, loc: &Location) -> Option<(String, i32, i32)> {
    match loc {
        Location::SrcPos {
            src: pliron::location::Source::File(path_key),
            pos,
        } if pos.line > 0 && pos.column > 0 => {
            let path = uniqued_any::get(ctx, *path_key)
                .to_string_lossy()
                .into_owned();
            Some((path, pos.line, pos.column))
        }
        Location::Named { child_loc, .. } => source_position_from_location(ctx, child_loc),
        Location::Fused { locations, .. } => locations
            .iter()
            .find_map(|loc| source_position_from_location(ctx, loc)),
        Location::CallSite { callee, caller } => source_position_from_location(ctx, callee)
            .or_else(|| source_position_from_location(ctx, caller)),
        Location::SrcPos { .. } | Location::Unknown => None,
    }
}

/// Register debug operations into the given context.
pub fn register(ctx: &mut Context) {
    MirDbgValueOp::register(ctx);
    MirDbgValueListOp::register(ctx);
}
