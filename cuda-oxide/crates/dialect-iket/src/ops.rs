/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IKET event and range operations.

use crate::attributes::IketPayloadKindAttr;
use crate::types::IketRangeTokenType;
use pliron::{
    builtin::{
        attributes::StringAttr,
        op_interfaces::{NResultsInterface, OneResultInterface},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

mod attr_keys {
    pliron::dict_key!(EVENT_NAME, "iket_event_name");
    pliron::dict_key!(PAYLOAD_KIND, "iket_payload_kind");
    pliron::dict_key!(RANGE_KEY, "iket_range_key");
}

fn build_named_event<O: Op>(
    ctx: &mut Context,
    result_types: Vec<TypeHandle>,
    event_name: impl Into<String>,
    payload_kind: IketPayloadKindAttr,
    operands: Vec<Value>,
) -> Ptr<Operation> {
    let operation = Operation::new(
        ctx,
        O::get_concrete_op_info(),
        result_types,
        operands,
        vec![],
        0,
    );
    operation.deref_mut(ctx).attributes.set(
        attr_keys::EVENT_NAME.clone(),
        StringAttr::new(event_name.into()),
    );
    operation
        .deref_mut(ctx)
        .attributes
        .set(attr_keys::PAYLOAD_KIND.clone(), payload_kind);
    operation
}

fn event_name<O: Op>(op: &O, ctx: &Context) -> Option<String> {
    op.get_operation()
        .deref(ctx)
        .attributes
        .get::<StringAttr>(&attr_keys::EVENT_NAME)
        .map(|value| String::from((*value).clone()))
}

fn payload_kind<O: Op>(op: &O, ctx: &Context) -> Option<IketPayloadKindAttr> {
    op.get_operation()
        .deref(ctx)
        .attributes
        .get::<IketPayloadKindAttr>(&attr_keys::PAYLOAD_KIND)
        .copied()
}

fn set_range_key<O: Op>(op: &O, ctx: &mut Context, range_key: impl Into<String>) {
    op.get_operation().deref_mut(ctx).attributes.set(
        attr_keys::RANGE_KEY.clone(),
        StringAttr::new(range_key.into()),
    );
}

fn range_key<O: Op>(op: &O, ctx: &Context) -> Option<String> {
    op.get_operation()
        .deref(ctx)
        .attributes
        .get::<StringAttr>(&attr_keys::RANGE_KEY)
        .map(|value| String::from((*value).clone()))
}

fn verify_name_and_payload<O: Op>(
    op: &O,
    ctx: &Context,
    token_operands: usize,
) -> Result<(), Error> {
    let operation = op.get_operation().deref(ctx);
    let Some(name) = event_name(op, ctx) else {
        return verify_err!(operation.loc(), "IKET event operation requires event_name");
    };
    if name.is_empty() {
        return verify_err!(operation.loc(), "IKET event name must not be empty");
    }
    if name.contains('\0') {
        return verify_err!(operation.loc(), "IKET event name must not contain NUL");
    }

    verify_payload_operands(op, ctx, token_operands)
}

fn verify_payload_operands<O: Op>(
    op: &O,
    ctx: &Context,
    token_operands: usize,
) -> Result<(), Error> {
    let operation = op.get_operation().deref(ctx);
    let Some(payload_kind) = payload_kind(op, ctx) else {
        return verify_err!(
            operation.loc(),
            "IKET event operation requires payload_kind"
        );
    };
    let expected_operands = token_operands + usize::from(payload_kind.has_payload());
    if operation.get_num_operands() != expected_operands {
        return verify_err!(
            operation.loc(),
            "IKET payload kind requires {} operand(s), got {}",
            expected_operands,
            operation.get_num_operands()
        );
    }
    Ok(())
}

fn verify_range_token_result<O: Op>(op: &O, ctx: &Context) -> Result<(), Error> {
    let operation = op.get_operation().deref(ctx);
    if operation.get_num_results() != 1
        || operation
            .get_result(0)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<IketRangeTokenType>()
            .is_none()
    {
        return verify_err!(
            operation.loc(),
            "IKET token producer must return one iket.range_token"
        );
    }
    Ok(())
}

/// Record a named point event, optionally with one scalar payload.
#[pliron_op(
    name = "iket.mark",
    format,
    interfaces = [NResultsInterface<0>]
)]
pub struct IketMarkOp;

impl IketMarkOp {
    pub fn new(
        ctx: &mut Context,
        event_name: impl Into<String>,
        payload_kind: IketPayloadKindAttr,
        payload: Option<Value>,
    ) -> Self {
        let op = build_named_event::<Self>(
            ctx,
            vec![],
            event_name,
            payload_kind,
            payload.into_iter().collect(),
        );
        Self { op }
    }

    pub fn event_name(&self, ctx: &Context) -> Option<String> {
        event_name(self, ctx)
    }

    pub fn payload_kind(&self, ctx: &Context) -> Option<IketPayloadKindAttr> {
        payload_kind(self, ctx)
    }
}

impl Verify for IketMarkOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_name_and_payload(self, ctx, 0)
    }
}

/// Start a named token-paired range.
#[pliron_op(
    name = "iket.range_start",
    format,
    interfaces = [NResultsInterface<1>, OneResultInterface]
)]
pub struct IketRangeStartOp;

impl IketRangeStartOp {
    pub fn new(
        ctx: &mut Context,
        event_name: impl Into<String>,
        payload_kind: IketPayloadKindAttr,
        payload: Option<Value>,
    ) -> Self {
        let token_type: TypeHandle = IketRangeTokenType::get(ctx).into();
        let op = build_named_event::<Self>(
            ctx,
            vec![token_type],
            event_name,
            payload_kind,
            payload.into_iter().collect(),
        );
        Self { op }
    }

    pub fn event_name(&self, ctx: &Context) -> Option<String> {
        event_name(self, ctx)
    }

    pub fn payload_kind(&self, ctx: &Context) -> Option<IketPayloadKindAttr> {
        payload_kind(self, ctx)
    }

    /// Attach the frontend's static identity for this token-paired range.
    pub fn set_range_key(&self, ctx: &mut Context, key: impl Into<String>) {
        set_range_key(self, ctx, key);
    }

    pub fn range_key(&self, ctx: &Context) -> Option<String> {
        range_key(self, ctx)
    }
}

impl Verify for IketRangeStartOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_name_and_payload(self, ctx, 0)?;
        verify_range_token_result(self, ctx)
    }
}

/// Create a non-recording token for control-flow initialization.
///
/// This mirrors CuTe DSL's `iket.sentinel_token`. A frontend is responsible
/// for guarding a corresponding `iket.range_end` so the sentinel path does
/// not emit an event.
#[pliron_op(
    name = "iket.sentinel_token",
    format,
    interfaces = [NResultsInterface<1>, OneResultInterface]
)]
pub struct IketSentinelTokenOp;

impl IketSentinelTokenOp {
    pub fn new(ctx: &mut Context) -> Self {
        let token_type: TypeHandle = IketRangeTokenType::get(ctx).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![token_type],
            vec![],
            vec![],
            0,
        );
        Self { op }
    }
}

impl Verify for IketSentinelTokenOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if operation.get_num_operands() != 0 {
            return verify_err!(operation.loc(), "iket.sentinel_token takes no operands");
        }
        verify_range_token_result(self, ctx)
    }
}

/// End a token-paired range, optionally with one scalar payload.
#[pliron_op(
    name = "iket.range_end",
    format,
    interfaces = [NResultsInterface<0>]
)]
pub struct IketRangeEndOp;

impl IketRangeEndOp {
    pub fn new(
        ctx: &mut Context,
        token: Value,
        payload_kind: IketPayloadKindAttr,
        payload: Option<Value>,
    ) -> Self {
        let mut operands = vec![token];
        operands.extend(payload);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        op.deref_mut(ctx)
            .attributes
            .set(attr_keys::PAYLOAD_KIND.clone(), payload_kind);
        Self { op }
    }

    pub fn payload_kind(&self, ctx: &Context) -> Option<IketPayloadKindAttr> {
        payload_kind(self, ctx)
    }

    /// Attach the same frontend range identity carried by `iket.range_start`.
    pub fn set_range_key(&self, ctx: &mut Context, key: impl Into<String>) {
        set_range_key(self, ctx, key);
    }

    pub fn range_key(&self, ctx: &Context) -> Option<String> {
        range_key(self, ctx)
    }
}

impl Verify for IketRangeEndOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_payload_operands(self, ctx, 1)?;
        let operation = self.get_operation().deref(ctx);
        if operation.get_operand(0).get_type(ctx) != IketRangeTokenType::get(ctx).into() {
            return verify_err!(
                operation.loc(),
                "iket.range_end first operand must be !iket.range_token"
            );
        }
        Ok(())
    }
}

/// Enter a named LIFO range, optionally with one scalar payload.
#[pliron_op(
    name = "iket.range_push",
    format,
    interfaces = [NResultsInterface<0>]
)]
pub struct IketRangePushOp;

impl IketRangePushOp {
    pub fn new(
        ctx: &mut Context,
        event_name: impl Into<String>,
        payload_kind: IketPayloadKindAttr,
        payload: Option<Value>,
    ) -> Self {
        let op = build_named_event::<Self>(
            ctx,
            vec![],
            event_name,
            payload_kind,
            payload.into_iter().collect(),
        );
        Self { op }
    }

    pub fn event_name(&self, ctx: &Context) -> Option<String> {
        event_name(self, ctx)
    }

    pub fn payload_kind(&self, ctx: &Context) -> Option<IketPayloadKindAttr> {
        payload_kind(self, ctx)
    }
}

impl Verify for IketRangePushOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_name_and_payload(self, ctx, 0)
    }
}

/// Leave the most recently entered LIFO range.
#[pliron_op(
    name = "iket.range_pop",
    format,
    interfaces = [NResultsInterface<0>]
)]
pub struct IketRangePopOp;

impl IketRangePopOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Self { op }
    }
}

impl Verify for IketRangePopOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if operation.get_num_operands() != 0 {
            return verify_err!(operation.loc(), "iket.range_pop takes no operands");
        }
        Ok(())
    }
}

pub fn register(ctx: &mut Context) {
    IketMarkOp::register(ctx);
    IketRangeStartOp::register(ctx);
    IketSentinelTokenOp::register(ctx);
    IketRangeEndOp::register(ctx);
    IketRangePushOp::register(ctx);
    IketRangePopOp::register(ctx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{
        basic_block::BasicBlock,
        builtin::types::{IntegerType, Signedness},
        context::Ptr,
        operation::Operation,
    };

    fn payload_value(ctx: &mut Context) -> Value {
        let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let block = BasicBlock::new(ctx, None, vec![ty.into()]);
        block.deref(ctx).get_argument(0)
    }

    #[test]
    fn accepts_arbitrary_length_names() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let name = "producer/mainloop/tma/load/longer-than-thirty-two-characters";
        let mark = IketMarkOp::new(&mut ctx, name, IketPayloadKindAttr::None, None);
        assert!(mark.verify(&ctx).is_ok());
        assert_eq!(mark.event_name(&ctx), Some(name.to_string()));
    }

    #[test]
    fn payload_kind_and_operand_presence_must_agree() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let payload = payload_value(&mut ctx);
        let valid = IketMarkOp::new(&mut ctx, "tile", IketPayloadKindAttr::U32, Some(payload));
        assert!(valid.verify(&ctx).is_ok());

        let invalid = IketMarkOp::new(&mut ctx, "tile", IketPayloadKindAttr::U32, None);
        assert!(invalid.verify(&ctx).is_err());
    }

    #[test]
    fn token_range_has_a_first_class_ssa_edge() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let start =
            IketRangeStartOp::new(&mut ctx, "consumer.wait", IketPayloadKindAttr::None, None);
        assert!(start.verify(&ctx).is_ok());
        let token = start.get_operation().deref(&ctx).get_result(0);
        let end = IketRangeEndOp::new(&mut ctx, token, IketPayloadKindAttr::None, None);
        assert!(end.verify(&ctx).is_ok());
    }

    #[test]
    fn token_range_can_carry_frontend_static_identity() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let start = IketRangeStartOp::new(&mut ctx, "mainloop", IketPayloadKindAttr::None, None);
        start.set_range_key(&mut ctx, "kernel::__CudaOxideIketRange");
        let token = start.get_operation().deref(&ctx).get_result(0);
        let end = IketRangeEndOp::new(&mut ctx, token, IketPayloadKindAttr::None, None);
        end.set_range_key(&mut ctx, "kernel::__CudaOxideIketRange");
        assert_eq!(
            start.range_key(&ctx),
            Some("kernel::__CudaOxideIketRange".to_string())
        );
        assert_eq!(end.range_key(&ctx), start.range_key(&ctx));
    }

    #[test]
    fn names_reject_empty_and_nul() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        assert!(
            IketMarkOp::new(&mut ctx, "", IketPayloadKindAttr::None, None)
                .verify(&ctx)
                .is_err()
        );
        assert!(
            IketMarkOp::new(&mut ctx, "a\0b", IketPayloadKindAttr::None, None)
                .verify(&ctx)
                .is_err()
        );
    }

    #[test]
    fn range_pop_is_zero_arity() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let pop = IketRangePopOp::new(&mut ctx);
        assert!(pop.verify(&ctx).is_ok());
        let _: Ptr<Operation> = pop.get_operation();
    }

    #[test]
    fn sentinel_produces_a_range_token() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let sentinel = IketSentinelTokenOp::new(&mut ctx);
        assert!(sentinel.verify(&ctx).is_ok());
        let expected: TypeHandle = IketRangeTokenType::get(&ctx).into();
        let actual = sentinel
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        assert_eq!(actual, expected);
    }

    #[test]
    fn token_producers_reject_non_token_results() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let wrong_type: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        let start = build_named_event::<IketRangeStartOp>(
            &mut ctx,
            vec![wrong_type],
            "mainloop",
            IketPayloadKindAttr::None,
            vec![],
        );
        assert!(IketRangeStartOp { op: start }.verify(&ctx).is_err());

        let sentinel = Operation::new(
            &mut ctx,
            IketSentinelTokenOp::get_concrete_op_info(),
            vec![wrong_type],
            vec![],
            vec![],
            0,
        );
        assert!(IketSentinelTokenOp { op: sentinel }.verify(&ctx).is_err());
    }
}
