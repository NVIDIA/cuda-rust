/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Late materialization of the semantic IKET dialect.

use crate::error::PipelineError;
use crate::options::IketInstrumentation;
use dialect_iket::{
    attributes::IketPayloadKindAttr,
    ops::{
        IketMarkOp, IketRangeEndOp, IketRangePopOp, IketRangePushOp, IketRangeStartOp,
        IketSentinelTokenOp,
    },
    types::IketRangeTokenType,
};
use dialect_mir::{
    attributes::MirCastKindAttr,
    ops::{MirAllocaOp, MirCastOp, MirStoreOp},
};
use dialect_nvvm::ops::InlinePtxOp;
use iket_lower::{
    EncodedEventName, EventMetadata, EventPosition, IKET_COMPATIBILITY_PROFILE, InstrumentMethod,
    InstrumentMethodPolicy, RangeMetadata, RangeType, build_placeholder_ptx,
    encode_metadata_objects, fnv1a_32, placeholder_config, plan_instrumentation,
};
use llvm_export::{
    ops::{GlobalOp, GlobalOpExt},
    types::{ArrayType, address_space},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::BranchOpInterface,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, op_cast},
    operation::Operation,
    r#type::Typed,
    value::Value,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};

const RANGE_POP_EVENT_ID: u32 = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeclarationKind {
    Mark,
    StartEnd,
    PushPop,
}

#[derive(Clone, Debug)]
struct Declaration {
    event_id: u32,
    name: EncodedEventName,
    kind: DeclarationKind,
    payload: IketPayloadKindAttr,
}

#[derive(Clone, Copy, Debug)]
struct Site {
    operation: Ptr<Operation>,
    event_id: u32,
    payload: IketPayloadKindAttr,
    payload_operand: Option<Value>,
}

pub(crate) fn has_iket_operations(ctx: &Context, root: Ptr<Operation>) -> bool {
    collect_operations(ctx, root).into_iter().any(|operation| {
        Operation::get_op::<IketMarkOp>(operation, ctx).is_some()
            || Operation::get_op::<IketRangeStartOp>(operation, ctx).is_some()
            || Operation::get_op::<IketRangeEndOp>(operation, ctx).is_some()
            || Operation::get_op::<IketRangePushOp>(operation, ctx).is_some()
            || Operation::get_op::<IketRangePopOp>(operation, ctx).is_some()
            || Operation::get_op::<IketSentinelTokenOp>(operation, ctx).is_some()
    })
}

/// Erase every semantic IKET operation without materializing anything.
///
/// This is the `CUDA_OXIDE_IKET=off` path, split out so the pipeline can run
/// it before its debug-mode preparation gate: an annotated kernel built with
/// instrumentation disabled must compile in every configuration, including a
/// full-debug build where `materialize` would never be reached.
pub(crate) fn strip(ctx: &mut Context, module: Ptr<Operation>) -> Result<(), PipelineError> {
    if !has_iket_operations(ctx, module) {
        return Ok(());
    }
    erase_semantic_operations(ctx, module)
}

pub(crate) fn materialize(
    ctx: &mut Context,
    module: Ptr<Operation>,
    target: Option<&str>,
    control: &IketInstrumentation,
) -> Result<(), PipelineError> {
    if !has_iket_operations(ctx, module) {
        return Ok(());
    }
    let policy = match control {
        IketInstrumentation::Disabled => {
            erase_semantic_operations(ctx, module)?;
            return Ok(());
        }
        IketInstrumentation::Auto => InstrumentMethodPolicy::Auto,
        IketInstrumentation::NativeDump => InstrumentMethodPolicy::NativeDump,
        IketInstrumentation::ExtendedNativeDump => InstrumentMethodPolicy::ExtendedNativeDump,
        IketInstrumentation::Invalid(value) => {
            return Err(iket_error(format!(
                "invalid CUDA_OXIDE_IKET value {value:?}; expected auto, native, extended, or off"
            )));
        }
    };

    let plan = plan_instrumentation(ctx, module, IKET_COMPATIBILITY_PROFILE, policy)
        .map_err(|error| iket_error(error.to_string()))?;
    let placeholder = placeholder_config(target).map_err(|error| iket_error(error.to_string()))?;
    let event_ids = plan
        .event_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let base = match plan.instrument_method {
                InstrumentMethod::NativeDump => 1,
                InstrumentMethod::ExtendedNativeDump => 64,
            };
            (name.full_name.clone(), base + index as u32)
        })
        .collect::<BTreeMap<_, _>>();

    let operations = collect_operations(ctx, module);
    let mut declarations = BTreeMap::<String, Declaration>::new();
    let mut range_names_by_key = BTreeMap::<String, String>::new();
    for operation in &operations {
        if let Some(op) = Operation::get_op::<IketMarkOp>(*operation, ctx) {
            insert_declaration(
                &mut declarations,
                declaration_for_named(
                    ctx,
                    &op,
                    DeclarationKind::Mark,
                    &plan.event_names,
                    &event_ids,
                )?,
            )?;
        } else if let Some(op) = Operation::get_op::<IketRangeStartOp>(*operation, ctx) {
            if let Some(key) = op.range_key(ctx) {
                let name = op
                    .event_name(ctx)
                    .ok_or_else(|| iket_error("range_start is missing event_name"))?;
                if let Some(previous) = range_names_by_key.insert(key.clone(), name.clone())
                    && previous != name
                {
                    return Err(iket_error(format!(
                        "static range key {key:?} names both {previous:?} and {name:?}"
                    )));
                }
            }
            insert_declaration(
                &mut declarations,
                declaration_for_named(
                    ctx,
                    &op,
                    DeclarationKind::StartEnd,
                    &plan.event_names,
                    &event_ids,
                )?,
            )?;
        } else if let Some(op) = Operation::get_op::<IketRangePushOp>(*operation, ctx) {
            insert_declaration(
                &mut declarations,
                declaration_for_named(
                    ctx,
                    &op,
                    DeclarationKind::PushPop,
                    &plan.event_names,
                    &event_ids,
                )?,
            )?;
        }
    }

    let mut sites = Vec::new();
    for operation in &operations {
        if let Some(op) = Operation::get_op::<IketMarkOp>(*operation, ctx) {
            sites.push(named_site(
                ctx,
                *operation,
                op.event_name(ctx),
                op.payload_kind(ctx),
                &declarations,
                0,
            )?);
        } else if let Some(op) = Operation::get_op::<IketRangeStartOp>(*operation, ctx) {
            sites.push(named_site(
                ctx,
                *operation,
                op.event_name(ctx),
                op.payload_kind(ctx),
                &declarations,
                0,
            )?);
        } else if let Some(op) = Operation::get_op::<IketRangePushOp>(*operation, ctx) {
            sites.push(named_site(
                ctx,
                *operation,
                op.event_name(ctx),
                op.payload_kind(ctx),
                &declarations,
                0,
            )?);
        } else if let Some(op) = Operation::get_op::<IketRangeEndOp>(*operation, ctx) {
            let token = operation.deref(ctx).get_operand(0);
            let name = if let Some(key) = op.range_key(ctx) {
                range_names_by_key.get(&key).cloned().ok_or_else(|| {
                    iket_error(format!("range_end has unknown static range key {key:?}"))
                })?
            } else {
                resolve_range_name(ctx, token)?
            };
            let declaration = declarations
                .get(&name)
                .ok_or_else(|| iket_error(format!("range_end resolved unknown range {name:?}")))?;
            if declaration.kind != DeclarationKind::StartEnd {
                return Err(iket_error(format!(
                    "range_end token resolves to non-start/end event {name:?}"
                )));
            }
            let payload = op
                .payload_kind(ctx)
                .ok_or_else(|| iket_error("range_end is missing payload_kind"))?;
            if payload != declaration.payload {
                return Err(iket_error(format!(
                    "range {name:?} uses {payload:?} at end but {:?} at start",
                    declaration.payload
                )));
            }
            sites.push(Site {
                operation: *operation,
                event_id: declaration.event_id,
                payload,
                payload_operand: payload
                    .has_payload()
                    .then(|| operation.deref(ctx).get_operand(1)),
            });
        } else if Operation::get_op::<IketRangePopOp>(*operation, ctx).is_some() {
            sites.push(Site {
                operation: *operation,
                event_id: RANGE_POP_EVENT_ID,
                payload: IketPayloadKindAttr::None,
                payload_operand: None,
            });
        }
    }

    for site in sites {
        insert_physical_site(ctx, site, placeholder, plan.instrument_method)?;
    }
    erase_token_plumbing(ctx, module)?;
    emit_metadata(
        ctx,
        module,
        plan.instrument_method,
        declarations.into_values(),
    )
}

fn declaration_for_named<O>(
    ctx: &Context,
    op: &O,
    kind: DeclarationKind,
    encoded_names: &[EncodedEventName],
    event_ids: &BTreeMap<String, u32>,
) -> Result<Declaration, PipelineError>
where
    O: Op + NamedIketOp,
{
    let name = op
        .iket_name(ctx)
        .ok_or_else(|| iket_error("named IKET operation is missing event_name"))?;
    let payload = op
        .iket_payload(ctx)
        .ok_or_else(|| iket_error(format!("IKET event {name:?} is missing payload_kind")))?;
    let encoded = encoded_names
        .iter()
        .find(|encoded| encoded.full_name == name)
        .cloned()
        .ok_or_else(|| iket_error(format!("IKET plan omitted event {name:?}")))?;
    Ok(Declaration {
        event_id: event_ids[&name],
        name: encoded,
        kind,
        payload,
    })
}

trait NamedIketOp {
    fn iket_name(&self, ctx: &Context) -> Option<String>;
    fn iket_payload(&self, ctx: &Context) -> Option<IketPayloadKindAttr>;
}

macro_rules! impl_named_iket_op {
    ($ty:ty) => {
        impl NamedIketOp for $ty {
            fn iket_name(&self, ctx: &Context) -> Option<String> {
                self.event_name(ctx)
            }
            fn iket_payload(&self, ctx: &Context) -> Option<IketPayloadKindAttr> {
                self.payload_kind(ctx)
            }
        }
    };
}
impl_named_iket_op!(IketMarkOp);
impl_named_iket_op!(IketRangeStartOp);
impl_named_iket_op!(IketRangePushOp);

fn insert_declaration(
    declarations: &mut BTreeMap<String, Declaration>,
    declaration: Declaration,
) -> Result<(), PipelineError> {
    let name = declaration.name.full_name.clone();
    if let Some(previous) = declarations.get(&name)
        && (previous.kind != declaration.kind || previous.payload != declaration.payload)
    {
        return Err(iket_error(format!(
            "IKET event name {name:?} is reused with incompatible kind or payload"
        )));
    }
    declarations.entry(name).or_insert(declaration);
    Ok(())
}

fn named_site(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: Option<String>,
    payload: Option<IketPayloadKindAttr>,
    declarations: &BTreeMap<String, Declaration>,
    payload_operand_index: usize,
) -> Result<Site, PipelineError> {
    let name = name.ok_or_else(|| iket_error("named IKET operation is missing event_name"))?;
    let payload = payload
        .ok_or_else(|| iket_error(format!("IKET event {name:?} is missing payload_kind")))?;
    let declaration = &declarations[&name];
    Ok(Site {
        operation,
        event_id: declaration.event_id,
        payload,
        payload_operand: payload
            .has_payload()
            .then(|| operation.deref(ctx).get_operand(payload_operand_index)),
    })
}

fn resolve_range_name(ctx: &Context, token: Value) -> Result<String, PipelineError> {
    let mut pending = vec![token];
    let mut visited = HashSet::new();
    let mut names = BTreeSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        if let Some(defining_op) = value.defining_op() {
            if let Some(start) = Operation::get_op::<IketRangeStartOp>(defining_op, ctx) {
                names.insert(
                    start
                        .event_name(ctx)
                        .ok_or_else(|| iket_error("range_start is missing event_name"))?,
                );
            } else if Operation::get_op::<IketSentinelTokenOp>(defining_op, ctx).is_none() {
                return Err(iket_error(
                    "range token is defined by an operation other than range_start or sentinel",
                ));
            }
            continue;
        }
        let block = value
            .defining_block()
            .ok_or_else(|| iket_error("range token has no defining entity"))?;
        let argument_index = value.find_index(ctx);
        for edge in block.uses(ctx) {
            let terminator = edge.user_op();
            let operation = Operation::get_op_dyn(terminator, ctx);
            let branch = op_cast::<dyn BranchOpInterface>(operation.as_ref()).ok_or_else(|| {
                iket_error("range token flows through a terminator without branch operands")
            })?;
            let operands = branch.successor_operands(ctx, edge.find_index(ctx));
            let incoming = operands.get(argument_index).copied().ok_or_else(|| {
                iket_error("range-token block argument has no incoming edge operand")
            })?;
            pending.push(incoming);
        }
    }
    if names.len() != 1 {
        return Err(iket_error(format!(
            "range_end must resolve to exactly one static range name, found {names:?}"
        )));
    }
    Ok(names.into_iter().next().unwrap())
}

fn insert_physical_site(
    ctx: &mut Context,
    site: Site,
    config: iket_lower::PlaceholderConfig,
    method: InstrumentMethod,
) -> Result<(), PipelineError> {
    let template = build_placeholder_ptx(config, method, site.event_id, site.payload)
        .map_err(|error| iket_error(error.to_string()))?;
    let location = site.operation.deref(ctx).loc().clone();
    let (inputs, constraints) = if let Some(payload) = site.payload_operand {
        let bits = payload_bits(ctx, site.operation, payload, site.payload)?;
        let constraint = if payload_width(site.payload) == 32 {
            "r"
        } else {
            "l"
        };
        (vec![bits], format!("{constraint},~{{memory}}"))
    } else {
        (vec![], "~{memory}".to_owned())
    };
    let physical = InlinePtxOp::build(ctx, vec![], inputs, &template, &constraints, true, true);
    physical.deref_mut(ctx).set_loc(location);
    physical.insert_before(ctx, site.operation);
    // Keep the token-producing semantic op until every range_end and CFG
    // forwarding operand has been removed. Erasing a live SSA definition here
    // would corrupt the use lists before `erase_token_plumbing` can strip them.
    if Operation::get_op::<IketRangeStartOp>(site.operation, ctx).is_none() {
        Operation::erase(site.operation, ctx);
    }
    Ok(())
}

fn payload_bits(
    ctx: &mut Context,
    before: Ptr<Operation>,
    payload: Value,
    kind: IketPayloadKindAttr,
) -> Result<Value, PipelineError> {
    let width = payload_width(kind);
    let integer_type = IntegerType::get(ctx, width, Signedness::Signless);
    let cast_kind = match kind {
        IketPayloadKindAttr::Pointer => MirCastKindAttr::PointerExposeAddress,
        IketPayloadKindAttr::F32 | IketPayloadKindAttr::F64 => MirCastKindAttr::Transmute,
        IketPayloadKindAttr::None => {
            return Err(iket_error("no-payload event requested payload bits"));
        }
        _ => MirCastKindAttr::IntToInt,
    };
    let cast = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![integer_type.into()],
        vec![payload],
        vec![],
        0,
    );
    cast.deref_mut(ctx).set_loc(before.deref(ctx).loc().clone());
    MirCastOp::new(cast).set_attr_cast_kind(ctx, cast_kind);
    cast.insert_before(ctx, before);
    Ok(cast.deref(ctx).get_result(0))
}

fn payload_width(kind: IketPayloadKindAttr) -> u32 {
    match kind {
        IketPayloadKindAttr::None => 0,
        IketPayloadKindAttr::I8
        | IketPayloadKindAttr::U8
        | IketPayloadKindAttr::I16
        | IketPayloadKindAttr::U16
        | IketPayloadKindAttr::I32
        | IketPayloadKindAttr::U32
        | IketPayloadKindAttr::F32 => 32,
        _ => 64,
    }
}

fn erase_semantic_operations(
    ctx: &mut Context,
    module: Ptr<Operation>,
) -> Result<(), PipelineError> {
    let operations = collect_operations(ctx, module);
    for operation in operations {
        if Operation::get_op::<IketMarkOp>(operation, ctx).is_some()
            || Operation::get_op::<IketRangeEndOp>(operation, ctx).is_some()
            || Operation::get_op::<IketRangePushOp>(operation, ctx).is_some()
            || Operation::get_op::<IketRangePopOp>(operation, ctx).is_some()
        {
            Operation::erase(operation, ctx);
        }
    }
    // The strip path may run before mem2reg (a full-debug build never runs it
    // at all), so range tokens can still be in the translator's memory form:
    // `range_start -> mir.store -> alloca` and `mir.load -> range_end`. Remove
    // that plumbing here; `erase_token_plumbing` handles the SSA and
    // block-argument form and fails closed on anything left over.
    erase_token_memory_plumbing(ctx, module);
    erase_token_plumbing(ctx, module)
}

/// Erase dead memory-form token plumbing to a fixpoint: stores of token
/// values, unused ops whose results are all tokens (e.g. `mir.load`), and
/// unused allocas of token slots.
fn erase_token_memory_plumbing(ctx: &mut Context, module: Ptr<Operation>) {
    loop {
        let mut dead = Vec::new();
        for operation in collect_operations(ctx, module) {
            // `erase_token_plumbing` owns the token-producing semantic ops.
            if Operation::get_op::<IketRangeStartOp>(operation, ctx).is_some()
                || Operation::get_op::<IketSentinelTokenOp>(operation, ctx).is_some()
            {
                continue;
            }
            let erase = if Operation::get_op::<MirStoreOp>(operation, ctx).is_some() {
                is_token(ctx, operation.deref(ctx).get_operand(1))
            } else if let Some(alloca) = Operation::get_op::<MirAllocaOp>(operation, ctx) {
                alloca
                    .pointee_type(ctx)
                    .deref(ctx)
                    .is::<IketRangeTokenType>()
                    && !operation.deref(ctx).has_use()
            } else {
                let op_ref = operation.deref(ctx);
                op_ref.get_num_results() > 0
                    && (0..op_ref.get_num_results())
                        .all(|index| is_token(ctx, op_ref.get_result(index)))
                    && !op_ref.has_use()
            };
            if erase {
                dead.push(operation);
            }
        }
        if dead.is_empty() {
            return;
        }
        for operation in dead {
            Operation::erase(operation, ctx);
        }
    }
}

fn erase_token_plumbing(ctx: &mut Context, module: Ptr<Operation>) -> Result<(), PipelineError> {
    let blocks = collect_blocks(ctx, module);
    for block in blocks {
        let token_arguments = block
            .deref(ctx)
            .arguments()
            .enumerate()
            .filter_map(|(index, argument)| is_token(ctx, argument).then_some(index))
            .collect::<Vec<_>>();
        for argument_index in token_arguments.into_iter().rev() {
            for edge in block.uses(ctx) {
                let terminator = edge.user_op();
                let operation = Operation::get_op_dyn(terminator, ctx);
                let branch =
                    op_cast::<dyn BranchOpInterface>(operation.as_ref()).ok_or_else(|| {
                        iket_error("range-token block predecessor does not expose branch operands")
                    })?;
                branch.remove_successor_operand(ctx, edge.find_index(ctx), argument_index);
            }
            BasicBlock::remove_argument(block, ctx, argument_index);
        }
    }
    for operation in collect_operations(ctx, module) {
        if Operation::get_op::<IketRangeStartOp>(operation, ctx).is_some()
            || Operation::get_op::<IketSentinelTokenOp>(operation, ctx).is_some()
        {
            if operation.deref(ctx).has_use() {
                return Err(iket_error(
                    "range token still has a non-control-flow use after IKET materialization",
                ));
            }
            Operation::erase(operation, ctx);
        }
    }
    Ok(())
}

fn is_token(ctx: &Context, value: Value) -> bool {
    value.get_type(ctx).deref(ctx).is::<IketRangeTokenType>()
}

fn emit_metadata(
    ctx: &mut Context,
    module: Ptr<Operation>,
    method: InstrumentMethod,
    declarations: impl Iterator<Item = Declaration>,
) -> Result<(), PipelineError> {
    let declarations = declarations.collect::<Vec<_>>();
    let events = declarations
        .iter()
        .map(|declaration| {
            let (position, range_id) = match declaration.kind {
                DeclarationKind::Mark => (EventPosition::NotInRange, 0),
                DeclarationKind::StartEnd => (
                    EventPosition::RangeStartEnd,
                    fnv1a_32(declaration.name.full_name.as_bytes()),
                ),
                DeclarationKind::PushPop => (
                    EventPosition::RangeStart,
                    fnv1a_32(declaration.name.full_name.as_bytes()),
                ),
            };
            EventMetadata {
                event_id: declaration.event_id,
                method,
                payload: declaration.payload,
                position,
                range_id,
                name: declaration.name.clone(),
            }
        })
        .collect::<Vec<_>>();
    let ranges = declarations
        .iter()
        .filter_map(|declaration| {
            let range_type = match declaration.kind {
                DeclarationKind::Mark => return None,
                DeclarationKind::StartEnd => RangeType::StartEnd,
                DeclarationKind::PushPop => RangeType::PushPop,
            };
            Some(RangeMetadata {
                range_id: fnv1a_32(declaration.name.full_name.as_bytes()),
                range_type,
                name: declaration.name.clone(),
            })
        })
        .collect::<Vec<_>>();
    let objects = encode_metadata_objects(method, &events, &ranges)
        .map_err(|error| iket_error(error.to_string()))?;
    let module_block = module
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| iket_error("module has no body block"))?;
    for object in objects.into_iter().rev() {
        let byte_type = IntegerType::get(ctx, 8, Signedness::Signless);
        let array_type = ArrayType::get(ctx, byte_type.into(), object.bytes.len() as u64);
        let symbol = object
            .symbol
            .try_into()
            .map_err(|error| iket_error(format!("invalid IKET metadata symbol: {error:?}")))?;
        let global = GlobalOp::new_with_alignment(ctx, symbol, array_type.into(), object.alignment);
        global.set_address_space(ctx, address_space::GLOBAL);
        global.set_initializer_hex(ctx, &hex_bytes(&object.bytes));
        global.mark_retained(ctx);
        global.get_operation().insert_at_front(module_block, ctx);
    }
    Ok(())
}

fn collect_operations(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut result = vec![root];
    let regions = root.deref(ctx).regions().collect::<Vec<_>>();
    for region in regions {
        for block in region.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            for operation in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                result.extend(collect_operations(ctx, operation));
            }
        }
    }
    result
}

fn collect_blocks(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<BasicBlock>> {
    let mut blocks = Vec::new();
    for operation in collect_operations(ctx, root) {
        for region in operation.deref(ctx).regions().collect::<Vec<_>>() {
            blocks.extend(region.deref(ctx).iter(ctx));
        }
    }
    blocks
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn iket_error(message: impl Into<String>) -> PipelineError {
    PipelineError::Lowering(format!("IKET: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::{MirFuncOp, MirGotoOp, MirReturnOp};
    use pliron::builtin::{
        attributes::TypeAttr, op_interfaces::SymbolOpInterface, ops::ModuleOp, types::FunctionType,
    };

    #[test]
    fn invalid_device_hint_fails_before_iket_promotion_on_both_paths() {
        use crate::options::{BackendOptions, DeviceArchHint, IketInstrumentation};
        use crate::pipeline::{
            ModulePipelineRequest, OutputFiles, PipelineTrace, compile_translated_module,
        };
        for (iket, debug_kind) in [
            (
                IketInstrumentation::Auto,
                llvm_export::export::DebugKind::Off,
            ),
            (
                IketInstrumentation::Auto,
                llvm_export::export::DebugKind::Full,
            ),
            (
                IketInstrumentation::Disabled,
                llvm_export::export::DebugKind::Full,
            ),
        ] {
            for nvvm in [false, true] {
                let mut ctx = Context::new();
                let module = test_module(&mut ctx, 1);
                assert!(has_iket_operations(&ctx, module));
                let backend = BackendOptions {
                    iket: iket.clone(),
                    device_arch_hint: Some(DeviceArchHint::parse("compute_120".to_string())),
                    ..BackendOptions::default()
                };
                let files = OutputFiles {
                    llvm_ir: std::path::Path::new("unused.ll"),
                    ptx: std::path::Path::new("unused.ptx"),
                    stale_before_export: &[],
                };
                let request = ModulePipelineRequest::for_rust_pipeline(
                    &[],
                    nvvm,
                    &backend,
                    debug_kind,
                    files,
                    PipelineTrace::default(),
                );
                let Err(error) = compile_translated_module(&mut ctx, module, &request) else {
                    panic!("invalid hint must fail before IKET promotion");
                };
                assert!(matches!(error, PipelineError::TargetSelection { .. }));
                assert_eq!(
                    error.to_string(),
                    "invalid CUDA_OXIDE_DEVICE_ARCH `compute_120`: expected sm_<capability> with an optional `a` suffix"
                );
                assert!(
                    has_iket_operations(&ctx, module),
                    "error must precede materialization"
                );
            }
        }
    }

    #[test]
    fn proven_device_hint_reaches_both_backends_and_iket_promotion() {
        use crate::options::{BackendOptions, DeviceArchHint};
        use crate::pipeline::{
            ModulePipelineRequest, OutputFiles, PipelineTrace, compile_translated_module,
        };
        for nvvm in [false, true] {
            for event_count in [0, 1] {
                let mut ctx = Context::new();
                let module = test_module(&mut ctx, event_count);
                let root = tempfile::tempdir().unwrap();
                let llvm_ir = root.path().join("module.ll");
                let ptx = root.path().join("module.ptx");
                let backend = BackendOptions {
                    device_arch_hint: Some(DeviceArchHint::parse("sm_120".to_string())),
                    verbose: true,
                    ..BackendOptions::default()
                };
                let request = ModulePipelineRequest::for_rust_pipeline(
                    &[],
                    nvvm,
                    &backend,
                    llvm_export::export::DebugKind::Off,
                    OutputFiles {
                        llvm_ir: &llvm_ir,
                        ptx: &ptx,
                        stale_before_export: &[],
                    },
                    PipelineTrace::default(),
                );
                let output = compile_translated_module(&mut ctx, module, &request).unwrap();
                assert_eq!(output.target, "sm_120");
                if !nvvm {
                    let source = if event_count == 0 {
                        "detected GPU"
                    } else {
                        "the detected GPU, pinned by IKET materialization"
                    };
                    assert!(
                        output.diagnostics.iter().any(|line| line
                            .starts_with(&format!("Target: sm_120 (from {source}; detected "))),
                        "{:?}",
                        output.diagnostics
                    );
                }
            }
        }
    }

    fn test_module(ctx: &mut Context, event_count: usize) -> Ptr<Operation> {
        dialect_mir::register(ctx);
        dialect_iket::register(ctx);
        dialect_nvvm::register(ctx);
        GlobalOp::register(ctx);

        let module = ModuleOp::new(ctx, "iket_test".try_into().unwrap());
        let module_region = module.get_operation().deref(ctx).get_region(0);
        let module_block = module_region
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("ModuleOp creates its single body block");

        let function_type = FunctionType::get(ctx, vec![], vec![]);
        let function_operation = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, function_operation, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, "kernel".try_into().unwrap());
        let body = function_operation.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![]);
        entry.insert_at_back(body, ctx);
        for index in 0..event_count {
            IketMarkOp::new(
                ctx,
                format!("event_{index}"),
                IketPayloadKindAttr::None,
                None,
            )
            .get_operation()
            .insert_at_back(entry, ctx);
        }
        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(entry, ctx);
        function_operation.insert_at_back(module_block, ctx);
        module.get_operation()
    }

    #[test]
    fn auto_materialization_switches_to_extended_after_thirty_names() {
        let mut ctx = Context::new();
        let module = test_module(&mut ctx, 31);
        materialize(&mut ctx, module, Some("sm_100"), &IketInstrumentation::Auto).unwrap();

        let inline_ptx = collect_operations(&ctx, module)
            .into_iter()
            .filter_map(|operation| Operation::get_op::<InlinePtxOp>(operation, &ctx))
            .collect::<Vec<_>>();
        assert_eq!(inline_ptx.len(), 31);
        let template = inline_ptx[0].get_attr_ptx_template(&ctx).unwrap();
        assert!(String::from((*template).clone()).contains("st.weak.shared.u64"));

        let meta = collect_operations(&ctx, module)
            .into_iter()
            .find_map(|operation| {
                let global = Operation::get_op::<GlobalOp>(operation, &ctx)?;
                (global.get_symbol_name(&ctx).to_string() == "__iket_meta_info").then_some(global)
            })
            .unwrap();
        let bytes = meta.initializer_hex(&ctx).unwrap();
        assert!(bytes.starts_with("300000000000000007000000ff0f0000"));
        assert!(meta.is_retained(&ctx));
    }

    #[test]
    fn sm120_materialization_has_no_cluster_register_read() {
        let mut ctx = Context::new();
        let module = test_module(&mut ctx, 1);
        materialize(&mut ctx, module, Some("sm_120"), &IketInstrumentation::Auto).unwrap();
        let template = collect_operations(&ctx, module)
            .into_iter()
            .find_map(|operation| {
                Operation::get_op::<InlinePtxOp>(operation, &ctx)?
                    .get_attr_ptx_template(&ctx)
                    .map(|value| String::from((*value).clone()))
            })
            .unwrap();
        assert!(!template.contains("cluster"));
    }

    #[test]
    fn token_paired_range_uses_one_event_id_at_both_sites() {
        let mut ctx = Context::new();
        let module = test_module(&mut ctx, 0);
        let function = collect_operations(&ctx, module)
            .into_iter()
            .find_map(|operation| Operation::get_op::<MirFuncOp>(operation, &ctx))
            .unwrap();
        let entry = function
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let return_op = entry.deref(&ctx).get_terminator(&ctx).unwrap();
        let start = IketRangeStartOp::new(&mut ctx, "mainloop", IketPayloadKindAttr::None, None);
        let token = start.get_operation().deref(&ctx).get_result(0);
        start.get_operation().insert_before(&ctx, return_op);
        IketRangeEndOp::new(&mut ctx, token, IketPayloadKindAttr::None, None)
            .get_operation()
            .insert_before(&ctx, return_op);

        materialize(&mut ctx, module, Some("sm_90"), &IketInstrumentation::Auto).unwrap();
        let templates = collect_operations(&ctx, module)
            .into_iter()
            .filter_map(|operation| {
                Operation::get_op::<InlinePtxOp>(operation, &ctx)?
                    .get_attr_ptx_template(&ctx)
                    .map(|value| String::from((*value).clone()))
            })
            .collect::<Vec<_>>();
        assert_eq!(templates.len(), 2);
        assert!(
            templates
                .iter()
                .all(|template| template.contains("pmevent.mask 1"))
        );
        assert!(!has_iket_operations(&ctx, module));
    }

    #[test]
    fn static_range_key_survives_a_zst_sentinel_token() {
        let mut ctx = Context::new();
        let module = test_module(&mut ctx, 0);
        let function = collect_operations(&ctx, module)
            .into_iter()
            .find_map(|operation| Operation::get_op::<MirFuncOp>(operation, &ctx))
            .unwrap();
        let entry = function
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let return_op = entry.deref(&ctx).get_terminator(&ctx).unwrap();
        let start = IketRangeStartOp::new(&mut ctx, "zst-range", IketPayloadKindAttr::None, None);
        start.set_range_key(&mut ctx, "kernel::__CudaOxideIketRange");
        start.get_operation().insert_before(&ctx, return_op);
        let sentinel = IketSentinelTokenOp::new(&mut ctx);
        let token = sentinel.get_operation().deref(&ctx).get_result(0);
        sentinel.get_operation().insert_before(&ctx, return_op);
        let end = IketRangeEndOp::new(&mut ctx, token, IketPayloadKindAttr::None, None);
        end.set_range_key(&mut ctx, "kernel::__CudaOxideIketRange");
        end.get_operation().insert_before(&ctx, return_op);

        materialize(&mut ctx, module, Some("sm_100"), &IketInstrumentation::Auto).unwrap();
        let templates = collect_operations(&ctx, module)
            .into_iter()
            .filter_map(|operation| {
                Operation::get_op::<InlinePtxOp>(operation, &ctx)?
                    .get_attr_ptx_template(&ctx)
                    .map(|value| String::from((*value).clone()))
            })
            .collect::<Vec<_>>();
        assert_eq!(templates.len(), 2);
        assert!(
            templates
                .iter()
                .all(|template| template.contains("pmevent.mask 1"))
        );
        assert!(!has_iket_operations(&ctx, module));
    }

    #[test]
    fn range_token_cfg_arguments_are_removed_after_materialization() {
        let mut ctx = Context::new();
        let module = test_module(&mut ctx, 0);
        let function = collect_operations(&ctx, module)
            .into_iter()
            .find_map(|operation| Operation::get_op::<MirFuncOp>(operation, &ctx))
            .unwrap();
        let region = function.get_operation().deref(&ctx).get_region(0);
        let entry = region.deref(&ctx).iter(&ctx).next().unwrap();
        let old_return = entry.deref(&ctx).get_terminator(&ctx).unwrap();
        Operation::erase(old_return, &mut ctx);

        let start = IketRangeStartOp::new(&mut ctx, "forwarded", IketPayloadKindAttr::None, None);
        let token = start.get_operation().deref(&ctx).get_result(0);
        start.get_operation().insert_at_back(entry, &ctx);
        let token_type = token.get_type(&ctx);
        let exit = BasicBlock::new(&mut ctx, None, vec![token_type]);
        exit.insert_at_back(region, &ctx);
        Operation::new(
            &mut ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![token],
            vec![exit],
            0,
        )
        .insert_at_back(entry, &ctx);
        let forwarded = exit.deref(&ctx).get_argument(0);
        IketRangeEndOp::new(&mut ctx, forwarded, IketPayloadKindAttr::None, None)
            .get_operation()
            .insert_at_back(exit, &ctx);
        Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(exit, &ctx);

        materialize(&mut ctx, module, Some("sm_100"), &IketInstrumentation::Auto).unwrap();
        assert_eq!(exit.deref(&ctx).get_num_arguments(), 0);
        assert_eq!(
            entry
                .deref(&ctx)
                .get_terminator(&ctx)
                .unwrap()
                .deref(&ctx)
                .get_num_operands(),
            0
        );
    }

    #[test]
    fn materialized_module_is_accepted_by_mir_to_llvm_lowering() {
        let mut ctx = Context::new();
        let module = test_module(&mut ctx, 1);
        materialize(&mut ctx, module, Some("sm_90"), &IketInstrumentation::Auto).unwrap();
        crate::lower::lower_to_llvm(
            &mut ctx,
            module,
            true,
            mir_lower::IntrinsicBackend::LlvmNvptx,
            None,
        )
        .unwrap();
        assert!(!has_iket_operations(&ctx, module));
    }
}
