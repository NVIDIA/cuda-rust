/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Translation of `cuda_device::iket` compiler markers.

use super::super::helpers::{self, emit_goto, insert_op};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::{rvalue, values::ValueMap};
use dialect_iket::{
    attributes::IketPayloadKindAttr,
    ops::{IketMarkOp, IketRangeEndOp, IketRangePopOp, IketRangePushOp, IketRangeStartOp},
};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    input_err,
    location::{Located, Location},
    op::Op,
    operation::Operation,
    value::Value,
};
use rustc_public::{
    mir,
    ty::{FloatTy, IntTy, RigidTy, Ty, TyKind, UintTy},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marker {
    Mark,
    MarkPayload,
    RangeStart,
    RangeStartPayload,
    RangeEnd,
    RangeEndPayload,
    RangePush,
    RangePushPayload,
    RangePop,
}

impl Marker {
    fn from_path(path: &str) -> Option<Self> {
        Some(match path {
            "cuda_device::iket::__iket_mark" => Self::Mark,
            "cuda_device::iket::__iket_mark_payload" => Self::MarkPayload,
            "cuda_device::iket::__iket_range_start" => Self::RangeStart,
            "cuda_device::iket::__iket_range_start_payload" => Self::RangeStartPayload,
            "cuda_device::iket::__iket_range_end" => Self::RangeEnd,
            "cuda_device::iket::__iket_range_end_payload" => Self::RangeEndPayload,
            "cuda_device::iket::__iket_range_push" => Self::RangePush,
            "cuda_device::iket::__iket_range_push_payload" => Self::RangePushPayload,
            "cuda_device::iket::__iket_range_pop" => Self::RangePop,
            _ => return None,
        })
    }

    fn has_name(self) -> bool {
        matches!(
            self,
            Self::Mark
                | Self::MarkPayload
                | Self::RangeStart
                | Self::RangeStartPayload
                | Self::RangePush
                | Self::RangePushPayload
        )
    }

    fn has_payload(self) -> bool {
        matches!(
            self,
            Self::MarkPayload
                | Self::RangeStartPayload
                | Self::RangeEndPayload
                | Self::RangePushPayload
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn try_dispatch(
    ctx: &mut Context,
    body: &mir::Body,
    path: &str,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_substs: &[Ty],
) -> TranslationResult<Option<Ptr<Operation>>> {
    let Some(marker) = Marker::from_path(path) else {
        return Ok(None);
    };

    let expected_args = usize::from(marker.has_name())
        + usize::from(marker.has_payload())
        + usize::from(matches!(marker, Marker::RangeEnd | Marker::RangeEndPayload));
    if args.len() != expected_args {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "IKET marker `{path}` expects {expected_args} argument(s), got {}",
                args.len()
            ))
        );
    }

    let event_name = if marker.has_name() {
        Some(literal_event_name(&args[0], loc.clone())?)
    } else {
        None
    };

    let payload_kind = if marker.has_payload() {
        let payload_type_index = usize::from(matches!(
            marker,
            Marker::RangeStartPayload | Marker::RangeEndPayload
        ));
        type_substs
            .get(payload_type_index)
            .and_then(payload_kind_from_type)
            .ok_or_else(|| {
                pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                    "could not determine IKET payload type for `{path}`"
                )))
            })?
    } else {
        IketPayloadKindAttr::None
    };

    let mut last_op = prev_op;
    let token = if matches!(marker, Marker::RangeEnd | Marker::RangeEndPayload) {
        let (value, next) = rvalue::translate_operand(
            ctx,
            body,
            &args[0],
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = next;
        Some(value)
    } else {
        None
    };

    let payload_index = usize::from(marker.has_name())
        + usize::from(matches!(marker, Marker::RangeEnd | Marker::RangeEndPayload));
    let payload = if marker.has_payload() {
        let (value, next) = rvalue::translate_operand(
            ctx,
            body,
            &args[payload_index],
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = next;
        Some(value)
    } else {
        None
    };

    let range_key = if matches!(
        marker,
        Marker::RangeStart | Marker::RangeStartPayload | Marker::RangeEnd | Marker::RangeEndPayload
    ) {
        Some(
            type_substs
                .first()
                .map(|ty| format!("{ty:?}"))
                .ok_or_else(|| {
                    pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                        "could not determine IKET range descriptor type for `{path}`"
                    )))
                })?,
        )
    } else {
        None
    };

    let prepared_destination = if matches!(marker, Marker::RangeStart | Marker::RangeStartPayload) {
        let (prepared_destination, next_op) = helpers::prepare_destination_write(
            ctx,
            body,
            destination,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = next_op;
        Some(prepared_destination)
    } else {
        None
    };

    let operation = match marker {
        Marker::Mark | Marker::MarkPayload => IketMarkOp::new(
            ctx,
            event_name.expect("named marker"),
            payload_kind,
            payload,
        )
        .get_operation(),
        Marker::RangeStart | Marker::RangeStartPayload => {
            let op = IketRangeStartOp::new(
                ctx,
                event_name.expect("named marker"),
                payload_kind,
                payload,
            );
            op.set_range_key(ctx, range_key.expect("token-paired range"));
            op.get_operation()
        }
        Marker::RangeEnd | Marker::RangeEndPayload => {
            let op =
                IketRangeEndOp::new(ctx, token.expect("range-end token"), payload_kind, payload);
            op.set_range_key(ctx, range_key.expect("token-paired range"));
            op.get_operation()
        }
        Marker::RangePush | Marker::RangePushPayload => IketRangePushOp::new(
            ctx,
            event_name.expect("named marker"),
            payload_kind,
            payload,
        )
        .get_operation(),
        Marker::RangePop => IketRangePopOp::new(ctx).get_operation(),
    };
    operation.deref_mut(ctx).set_loc(loc.clone());
    insert_op(ctx, operation, block_ptr, last_op);

    if matches!(marker, Marker::RangeStart | Marker::RangeStartPayload) {
        let result: Value = operation.deref(ctx).get_result(0);
        return helpers::emit_prepared_result_and_goto(
            ctx,
            prepared_destination.expect("range_start destination must be prepared"),
            result,
            target,
            block_ptr,
            operation,
            value_map,
            block_map,
            loc,
            "IKET range_start call without target block",
        )
        .map(Some);
    }

    let Some(target) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!("IKET marker `{path}` has no target block"))
        );
    };
    Ok(Some(emit_goto(ctx, *target, operation, block_map, loc)))
}

fn literal_event_name(operand: &mir::Operand, loc: Location) -> TranslationResult<String> {
    let mir::Operand::Constant(constant) = operand else {
        return input_err!(
            loc,
            TranslationErr::unsupported("IKET event name must be a string literal".to_string())
        );
    };
    let bytes = rvalue::constant_bytes(constant, "IKET event name", loc.clone())?;
    String::from_utf8(bytes).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "IKET event name must be valid UTF-8: {error}"
        )))
    })
}

fn payload_kind_from_type(ty: &Ty) -> Option<IketPayloadKindAttr> {
    let TyKind::RigidTy(rigid) = ty.kind() else {
        return None;
    };
    payload_kind_from_rigid_type(&rigid)
}

fn payload_kind_from_rigid_type(ty: &RigidTy) -> Option<IketPayloadKindAttr> {
    Some(match ty {
        RigidTy::Int(IntTy::I8) => IketPayloadKindAttr::I8,
        RigidTy::Int(IntTy::I16) => IketPayloadKindAttr::I16,
        RigidTy::Int(IntTy::I32) => IketPayloadKindAttr::I32,
        RigidTy::Int(IntTy::I64) => IketPayloadKindAttr::I64,
        RigidTy::Uint(UintTy::U8) => IketPayloadKindAttr::U8,
        RigidTy::Uint(UintTy::U16) => IketPayloadKindAttr::U16,
        RigidTy::Uint(UintTy::U32) => IketPayloadKindAttr::U32,
        RigidTy::Uint(UintTy::U64) => IketPayloadKindAttr::U64,
        RigidTy::Float(FloatTy::F32) => IketPayloadKindAttr::F32,
        RigidTy::Float(FloatTy::F64) => IketPayloadKindAttr::F64,
        RigidTy::RawPtr(..) => IketPayloadKindAttr::Pointer,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_paths_are_exact() {
        assert_eq!(
            Marker::from_path("cuda_device::iket::__iket_range_pop"),
            Some(Marker::RangePop)
        );
        assert_eq!(Marker::from_path("other::iket::__iket_range_pop"), None);
    }

    #[test]
    fn payload_kind_recognizes_public_payload_types() {
        assert_eq!(
            payload_kind_from_rigid_type(&RigidTy::Int(IntTy::I32)),
            Some(IketPayloadKindAttr::I32)
        );
        assert_eq!(
            payload_kind_from_rigid_type(&RigidTy::Uint(UintTy::U64)),
            Some(IketPayloadKindAttr::U64)
        );
    }
}
