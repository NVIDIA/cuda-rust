/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Floating-point math intrinsic lowering: `dialect-mir` placeholders to LLVM dialect and libdevice.

use crate::convert::types::{StructLayoutInfo, build_struct_slot_map, convert_type};
use crate::helpers;
use dialect_mir::rust_intrinsics::{self, RustFloatMathIntrinsic, RustFloatMathLowering};
use dialect_mir::types::MirTupleType;
use llvm_export::attributes::{FCmpPredicateAttr, FastmathFlags, FastmathFlagsAttr};
use llvm_export::op_interfaces::{BinArithOp, CastOpInterface, FloatBinArithOpWithFastMathFlags};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::{Inserter, OpInsertionPoint};
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Type-aware semantic lowering around the generated shared math dispatch.
pub(crate) trait FloatMathLowering {
    /// Return the ordered comparison predicate for a NaN-ignoring `min`/`max` (`*_nsz`) intrinsic.
    fn minmax_nsz_predicate(self) -> Option<FCmpPredicateAttr>;

    /// Resolve the CUDA libdevice symbol name for `self` and `result_ty`.
    fn libdevice_name(
        self,
        ctx: &Context,
        result_ty: TypeHandle,
        loc: pliron::location::Location,
    ) -> Result<&'static str>;

    /// Resolve the native LLVM intrinsic name when available on the NVPTX backend.
    fn llvm_intrinsic_name(
        self,
        ctx: &Context,
        result_ty: TypeHandle,
        loc: pliron::location::Location,
    ) -> Result<Option<&'static str>>;

    /// Return the fast-math binary operation for `core::intrinsics::f*_fast` variants.
    fn fast_binop(self) -> Option<FastFloatBinop>;
}

impl FloatMathLowering for RustFloatMathIntrinsic {
    /// Return the ordered comparison predicate for a NaN-ignoring `min`/`max` (`*_nsz`) intrinsic.
    fn minmax_nsz_predicate(self) -> Option<FCmpPredicateAttr> {
        match self.lowering() {
            RustFloatMathLowering::MinMaxNsz { is_max: true } => Some(FCmpPredicateAttr::OGE),
            RustFloatMathLowering::MinMaxNsz { is_max: false } => Some(FCmpPredicateAttr::OLE),
            _ => None,
        }
    }

    /// Resolve the CUDA libdevice symbol name for `self` and `result_ty`.
    fn libdevice_name(
        self,
        ctx: &Context,
        result_ty: TypeHandle,
        loc: pliron::location::Location,
    ) -> Result<&'static str> {
        if self == Self::Fabs {
            return fabs_libdevice_name(ctx, result_ty, loc);
        }
        self.libdevice_symbol(false).ok_or_else(|| {
            if self.minmax_nsz_predicate().is_some() {
                pliron::input_error!(
                    loc,
                    "float max/min intrinsics lower to a compare/select expansion, never to libdevice"
                )
            } else {
                pliron::input_error!(
                    loc,
                    "f*_fast intrinsics have no libdevice equivalent; they lower to llvm float binops with fast-math flags"
                )
            }
        })
    }

    /// Resolve the native LLVM intrinsic name when available on the NVPTX backend.
    fn llvm_intrinsic_name(
        self,
        ctx: &Context,
        result_ty: TypeHandle,
        loc: pliron::location::Location,
    ) -> Result<Option<&'static str>> {
        if self == Self::Fabs {
            return fabs_llvm_intrinsic_name(ctx, result_ty, loc).map(Some);
        }
        Ok(self.llvm_symbol(false))
    }

    /// Return the fast-math binary operation for `core::intrinsics::f*_fast` variants.
    fn fast_binop(self) -> Option<FastFloatBinop> {
        match self {
            Self::FaddFast => Some(FastFloatBinop::Add),
            Self::FsubFast => Some(FastFloatBinop::Sub),
            Self::FmulFast => Some(FastFloatBinop::Mul),
            Self::FdivFast => Some(FastFloatBinop::Div),
            Self::FremFast => Some(FastFloatBinop::Rem),
            _ => None,
        }
    }
}

/// Which `llvm.f*` binary op a `core::intrinsics::f*_fast` intrinsic lowers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FastFloatBinop {
    /// Floating-point addition (`llvm.fadd`).
    Add,
    /// Floating-point subtraction (`llvm.fsub`).
    Sub,
    /// Floating-point multiplication (`llvm.fmul`).
    Mul,
    /// Floating-point division (`llvm.fdiv`).
    Div,
    /// Floating-point remainder (`llvm.frem`).
    Rem,
}

/// Convert an `anyhow::Error` into a pliron verification error.
fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

/// Lower placeholder calls for rustc's `f32` and `f64` math intrinsics.
///
/// Dispatches via [`RustFloatMathIntrinsic::lowering`] to either an inline LLVM
/// expansion (fast binops, IEEE-754 min/max), a tuple out-parameter adapter
/// (`frexp`, `modf`, `remquo`, `lgamma_r`), a special-cased libdevice call
/// (`j1`, generic `fabs`), or a direct scalar LLVM/libdevice call.
pub fn convert_rust_float_math_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust float math intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let expected_args = intrinsic.arg_count();
    if args.len() != expected_args {
        return pliron::input_err!(
            loc,
            "Rust float math intrinsic requires {expected_args} operand(s)"
        );
    }

    match intrinsic.lowering() {
        RustFloatMathLowering::FastBinop => {
            let binop = intrinsic
                .fast_binop()
                .expect("FastBinop lowering requires a fast binop variant");
            lower_fast_binop(ctx, rewriter, op, &args, binop, loc)
        }
        RustFloatMathLowering::IeeeMinMax {
            is_max,
            propagate_nan,
        } => lower_float_ieee_minmax(ctx, rewriter, op, intrinsic, &args, is_max, propagate_nan),
        RustFloatMathLowering::MinMaxNsz { is_max } => {
            let predicate = if is_max {
                FCmpPredicateAttr::OGE
            } else {
                FCmpPredicateAttr::OLE
            };
            lower_float_minmax_nsz(ctx, rewriter, op, &args, predicate)
        }
        RustFloatMathLowering::Frexp { symbol } => {
            convert_rust_frexp(ctx, rewriter, op, intrinsic, symbol, args[0])
        }
        RustFloatMathLowering::LgammaR => {
            convert_rust_lgamma_r(ctx, rewriter, op, intrinsic, args[0])
        }
        RustFloatMathLowering::Modf { symbol } => {
            convert_rust_out_param_pair(ctx, rewriter, op, intrinsic, symbol, false, &args)
        }
        RustFloatMathLowering::Remquo { symbol } => {
            convert_rust_out_param_pair(ctx, rewriter, op, intrinsic, symbol, true, &args)
        }
        RustFloatMathLowering::BesselJ1 { symbol } => {
            lower_bessel_j1(ctx, rewriter, op, intrinsic, symbol, &args, loc)
        }
        RustFloatMathLowering::ScalarCall { .. } => {
            if intrinsic == RustFloatMathIntrinsic::Fabs {
                return emit_fabs(ctx, rewriter, op, &args, loc);
            }
            lower_scalar_call(ctx, rewriter, op, intrinsic, false, &args, loc)
        }
        RustFloatMathLowering::Sincos { .. } => pliron::input_err!(
            loc,
            "sincos expands to sin/cos calls during MIR import and has no direct placeholder"
        ),
    }
}

/// Validate a scalar math placeholder's operand and result types against the generated signature.
fn validate_scalar_signature(
    ctx: &mut Context,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    is_f64: bool,
    args: &[Value],
    loc: pliron::location::Location,
) -> Result<TypeHandle> {
    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let Some((argument_types, result_type)) = intrinsic.scalar_signature(is_f64) else {
        return pliron::input_err!(loc, "Rust float math intrinsic requires a scalar signature");
    };
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let f64_ty: TypeHandle = FP64Type::get(ctx).into();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let concrete_type = |ty| match ty {
        rust_intrinsics::MathScalarType::F32 => f32_ty,
        rust_intrinsics::MathScalarType::F64 => f64_ty,
        rust_intrinsics::MathScalarType::I32 => i32_ty,
    };
    if result_ty != concrete_type(result_type)
        || args.len() != argument_types.len()
        || args
            .iter()
            .zip(argument_types)
            .any(|(arg, expected)| arg.get_type(ctx) != concrete_type(*expected))
    {
        return pliron::input_err!(loc, "Rust float math intrinsic signature mismatch");
    }
    Ok(result_ty)
}

/// Emit the generic `fabs` placeholder by recovering its concrete `f32`/`f64` width.
fn emit_fabs(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    args: &[Value],
    loc: pliron::location::Location,
) -> Result<()> {
    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let is_f64 = float_math_is_f64(ctx, result_ty, loc.clone())?;
    lower_scalar_call(
        ctx,
        rewriter,
        op,
        RustFloatMathIntrinsic::Fabs,
        is_f64,
        args,
        loc,
    )
}

/// Lower a validated scalar math placeholder to its native LLVM or libdevice call.
fn lower_scalar_call(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    is_f64: bool,
    args: &[Value],
    loc: pliron::location::Location,
) -> Result<()> {
    let result_ty = validate_scalar_signature(ctx, op, intrinsic, is_f64, args, loc.clone())?;
    let intrinsic_name = float_math_intrinsic_symbol(ctx, intrinsic, result_ty, loc.clone())?;
    let arg_types = args.iter().map(|arg| arg.get_type(ctx)).collect::<Vec<_>>();
    let func_ty = llvm_types::FuncType::get(ctx, result_ty, arg_types, false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "Rust float math intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;
    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .try_into()
        .map_err(|e| pliron::input_error!(loc, "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(sym_name),
        func_ty,
        args.to_vec(),
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, llvm_call.get_operation());
    Ok(())
}

/// Lower `libm::j1`/`j1f` to `__nv_j1`/`__nv_j1f` and normalize `+-inf` to `+0.0`.
fn lower_bessel_j1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    symbol: &'static str,
    args: &[Value],
    loc: pliron::location::Location,
) -> Result<()> {
    let result_ty = validate_scalar_signature(ctx, op, intrinsic, false, args, loc.clone())?;
    let x = args[0];
    let func_ty = llvm_types::FuncType::get(ctx, result_ty, vec![result_ty], false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "Rust float math intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, symbol, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;
    let sym_name: pliron::identifier::Identifier = symbol
        .try_into()
        .map_err(|e| pliron::input_error!(loc, "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(ctx, CallOpCallable::Direct(sym_name), func_ty, vec![x]);
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    // CUDA documents j1(-inf) as -0, while libm 0.2.16 returns +0
    // for either infinity. Match the Rust export without touching NaNs.
    let width = if intrinsic == RustFloatMathIntrinsic::J1F32 {
        32
    } else {
        64
    };
    let int_ty = IntegerType::get(ctx, width, Signedness::Signless);
    let bits = llvm::BitcastOp::new(ctx, x, int_ty.into());
    rewriter.insert_operation(ctx, bits.get_operation());
    let bits = bits.get_operation().deref(ctx).get_result(0);
    let constant = |ctx: &mut Context, rewriter: &mut DialectConversionRewriter, value| {
        let attr = IntegerAttr::new(
            int_ty,
            APInt::from_u64(value, NonZeroUsize::new(width as usize).unwrap()),
        );
        let op = llvm::ConstantOp::new(ctx, Box::new(attr));
        rewriter.insert_operation(ctx, op.get_operation());
        op.get_operation().deref(ctx).get_result(0)
    };
    let infinity = if width == 32 {
        0x7f80_0000
    } else {
        0x7ff0_0000_0000_0000
    };
    let abs_mask = constant(
        ctx,
        rewriter,
        if width == 32 {
            0x7fff_ffff
        } else {
            0x7fff_ffff_ffff_ffff
        },
    );
    let infinity_bits = constant(ctx, rewriter, infinity);
    let zero_bits = constant(ctx, rewriter, 0);
    let magnitude = llvm::AndOp::new(ctx, bits, abs_mask);
    rewriter.insert_operation(ctx, magnitude.get_operation());
    let magnitude = magnitude.get_operation().deref(ctx).get_result(0);
    let is_infinite = llvm::ICmpOp::new(
        ctx,
        llvm_export::attributes::ICmpPredicateAttr::EQ,
        magnitude,
        infinity_bits,
    );
    rewriter.insert_operation(ctx, is_infinite.get_operation());
    let is_infinite = is_infinite.get_operation().deref(ctx).get_result(0);
    let positive_zero = llvm::BitcastOp::new(ctx, zero_bits, result_ty);
    rewriter.insert_operation(ctx, positive_zero.get_operation());
    let positive_zero = positive_zero.get_operation().deref(ctx).get_result(0);
    let call_result = llvm_call.get_operation().deref(ctx).get_result(0);
    let adjusted = llvm::SelectOp::new(ctx, is_infinite, positive_zero, call_result);
    rewriter.insert_operation(ctx, adjusted.get_operation());
    rewriter.replace_operation(ctx, op, adjusted.get_operation());
    Ok(())
}

/// Emit a libdevice scalar return plus one out-parameter. The caller supplies
/// the initial out value and handles its Rust tuple and semantic adjustments.
/// Keeping storage and ABI construction here lets modf/remquo share the path.
fn emit_libdevice_out_param_call(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    symbol: &str,
    args: &[Value],
    scalar_ty: TypeHandle,
    initial_out: Value,
) -> Result<(Value, Value)> {
    let loc = op.deref(ctx).loc().clone();
    let block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "libdevice out-parameter call has no parent block"
        )
    })?;
    let parent = helpers::get_parent_func(ctx, block).map_err(anyhow_to_pliron)?;
    let entry = parent
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                "libdevice out-parameter call has no entry block"
            )
        })?;
    let first = entry.deref(ctx).iter(ctx).next().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "libdevice out-parameter call has an empty entry block"
        )
    })?;
    // A call inside a loop must reuse a static slot, rather than allocate
    // additional stack storage on every iteration. Its size constant must
    // dominate it too; initialization stays at the actual call site.
    let one_attr = IntegerAttr::new(
        IntegerType::get(ctx, 32, Signedness::Signless),
        APInt::from_u64(1, NonZeroUsize::new(32).unwrap()),
    );
    let one = llvm::ConstantOp::new(ctx, Box::new(one_attr));
    let saved_ip = rewriter.get_insertion_point();
    rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(first));
    rewriter.insert_operation(ctx, one.get_operation());
    let out_ty = initial_out.get_type(ctx);
    let one_value = one.get_operation().deref(ctx).get_result(0);
    let slot = llvm::AllocaOp::new(ctx, out_ty, one_value, 0);
    rewriter.insert_operation(ctx, slot.get_operation());
    rewriter.set_insertion_point(saved_ip);
    let ptr = slot.get_operation().deref(ctx).get_result(0);
    let initialize = llvm::StoreOp::new(ctx, initial_out, ptr);
    rewriter.insert_operation(ctx, initialize.get_operation());
    let mut call_args = args.to_vec();
    call_args.push(ptr);
    let arg_types = call_args.iter().map(|arg| arg.get_type(ctx)).collect();
    let func_ty = llvm_types::FuncType::get(ctx, scalar_ty, arg_types, false);
    helpers::ensure_intrinsic_declared(ctx, block, symbol, func_ty).map_err(|error| {
        pliron::input_error!(loc.clone(), "Failed to declare {symbol}: {error}")
    })?;
    let symbol = symbol
        .try_into()
        .map_err(|error| pliron::input_error!(loc.clone(), "Invalid libdevice name: {error:?}"))?;
    let call = llvm::CallOp::new(ctx, CallOpCallable::Direct(symbol), func_ty, call_args);
    crate::convert::preserve_location(ctx, op, call.get_operation());
    rewriter.insert_operation(ctx, call.get_operation());
    let out = llvm::LoadOp::new(ctx, ptr, out_ty);
    rewriter.insert_operation(ctx, out.get_operation());
    Ok((
        call.get_operation().deref(ctx).get_result(0),
        out.get_operation().deref(ctx).get_result(0),
    ))
}

/// Build a Rust tuple using declared field order and the target LLVM layout.
fn pack_math_out_param_pair(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    layout: &StructLayoutInfo,
    fields: [Value; 2],
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    let map = build_struct_slot_map(ctx, layout).map_err(anyhow_to_pliron)?;
    let undef = llvm::UndefOp::new(ctx, map.llvm_struct_ty);
    rewriter.insert_operation(ctx, undef.get_operation());
    let mut value = undef.get_operation().deref(ctx).get_result(0);
    let mut last = undef.get_operation();
    for (field, operand) in fields.into_iter().enumerate() {
        let slot = map.decl_to_llvm[field].ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                "Rust math result field has no LLVM storage slot"
            )
        })?;
        let insert = llvm::InsertValueOp::new(ctx, value, operand, vec![slot]);
        rewriter.insert_operation(ctx, insert.get_operation());
        value = insert.get_operation().deref(ctx).get_result(0);
        last = insert.get_operation();
    }
    crate::convert::preserve_location(ctx, op, last);
    rewriter.replace_operation(ctx, op, last);
    Ok(())
}

/// Call an already validated scalar libdevice mapping from a tuple adapter.
fn call_scalar_math_for_pair(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    args: Vec<Value>,
    float_ty: TypeHandle,
) -> Result<Value> {
    let loc = op.deref(ctx).loc().clone();
    let name = intrinsic.libdevice_name(ctx, float_ty, loc.clone())?;
    let block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(loc.clone(), "libdevice scalar call has no parent block")
    })?;
    let arg_types = args.iter().map(|arg| arg.get_type(ctx)).collect();
    let signature = llvm_types::FuncType::get(ctx, float_ty, arg_types, false);
    helpers::ensure_intrinsic_declared(ctx, block, name, signature)
        .map_err(|error| pliron::input_error!(loc.clone(), "Failed to declare {name}: {error}"))?;
    let name = name
        .try_into()
        .map_err(|error| pliron::input_error!(loc, "Invalid libdevice name: {error:?}"))?;
    let call = llvm::CallOp::new(ctx, CallOpCallable::Direct(name), signature, args);
    rewriter.insert_operation(ctx, call.get_operation());
    Ok(call.get_operation().deref(ctx).get_result(0))
}

/// Compose libm's `(log(abs(gamma(x))), signgam)` from validated scalar
/// libdevice entries. floor/fmod determine the parity of negative intervals;
/// bitwise special cases preserve -0 and normalize lgamma(-infinity) on
/// backends that return negative infinity.
fn convert_rust_lgamma_r(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    x: Value,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    let (float_ty, width, sign_mask, infinity_mask, lgamma, floor, fmod): (
        TypeHandle,
        u32,
        u64,
        u64,
        RustFloatMathIntrinsic,
        RustFloatMathIntrinsic,
        RustFloatMathIntrinsic,
    ) = match intrinsic {
        RustFloatMathIntrinsic::LgammaRF32 => (
            FP32Type::get(ctx).into(),
            32,
            0x8000_0000,
            0x7f80_0000,
            RustFloatMathIntrinsic::LgammaF32,
            RustFloatMathIntrinsic::FloorF32,
            RustFloatMathIntrinsic::FmodF32,
        ),
        RustFloatMathIntrinsic::LgammaRF64 => (
            FP64Type::get(ctx).into(),
            64,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            RustFloatMathIntrinsic::LgammaF64,
            RustFloatMathIntrinsic::FloorF64,
            RustFloatMathIntrinsic::FmodF64,
        ),
        _ => unreachable!("lgamma_r converter requires its tuple placeholder"),
    };
    let layout = {
        let result = op.deref(ctx).get_result(0).get_type(ctx);
        let result = result.deref(ctx);
        let Some(tuple) = result.downcast_ref::<MirTupleType>() else {
            return pliron::input_err!(loc, "Rust lgamma_r requires a (float, i32) tuple");
        };
        let i32_mir: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
        if tuple.get_types() != [float_ty, i32_mir] || x.get_type(ctx) != float_ty {
            return pliron::input_err!(loc, "Rust lgamma_r signature mismatch");
        }
        StructLayoutInfo::of_tuple(tuple)
    };
    let mut integer = |ctx: &mut Context, bits: u32, value: u64| {
        let ty = IntegerType::get(ctx, bits, Signedness::Signless);
        let attr = IntegerAttr::new(
            ty,
            APInt::from_u64(value, NonZeroUsize::new(bits as usize).unwrap()),
        );
        let op = llvm::ConstantOp::new(ctx, Box::new(attr));
        rewriter.insert_operation(ctx, op.get_operation());
        op.get_operation().deref(ctx).get_result(0)
    };
    let plus_one = integer(ctx, 32, 1);
    let minus_one = integer(ctx, 32, u32::MAX.into());
    let positive_zero_bits = integer(ctx, width, 0);
    let negative_zero_bits = integer(ctx, width, sign_mask);
    let negative_infinity_bits = integer(ctx, width, sign_mask | infinity_mask);
    let positive_infinity_bits = integer(ctx, width, infinity_mask);
    let two_bits = integer(
        ctx,
        width,
        if width == 32 {
            2.0f32.to_bits() as u64
        } else {
            2.0f64.to_bits()
        },
    );
    let mut as_float = |ctx: &mut Context, bits: Value| {
        let cast = llvm::BitcastOp::new(ctx, bits, float_ty);
        rewriter.insert_operation(ctx, cast.get_operation());
        cast.get_operation().deref(ctx).get_result(0)
    };
    let zero = as_float(ctx, positive_zero_bits);
    let two = as_float(ctx, two_bits);
    let positive_infinity = as_float(ctx, positive_infinity_bits);
    let magnitude = call_scalar_math_for_pair(ctx, rewriter, op, lgamma, vec![x], float_ty)?;
    let whole = call_scalar_math_for_pair(ctx, rewriter, op, floor, vec![x], float_ty)?;
    let parity = call_scalar_math_for_pair(ctx, rewriter, op, fmod, vec![whole, two], float_ty)?;
    let mut compare_float =
        |ctx: &mut Context, predicate: FCmpPredicateAttr, a: Value, b: Value| {
            let cmp = llvm::FCmpOp::new(ctx, predicate, a, b);
            set_empty_fastmath_flags(ctx, cmp.get_operation());
            rewriter.insert_operation(ctx, cmp.get_operation());
            cmp.get_operation().deref(ctx).get_result(0)
        };
    let negative = compare_float(ctx, FCmpPredicateAttr::OLT, x, zero);
    let odd = compare_float(ctx, FCmpPredicateAttr::OLT, parity, zero);
    let is_pole = compare_float(ctx, FCmpPredicateAttr::OEQ, x, whole);
    let parity_sign = llvm::SelectOp::new(ctx, odd, minus_one, plus_one);
    rewriter.insert_operation(ctx, parity_sign.get_operation());
    let parity_sign = parity_sign.get_operation().deref(ctx).get_result(0);
    let negative_sign = llvm::SelectOp::new(ctx, negative, parity_sign, plus_one);
    rewriter.insert_operation(ctx, negative_sign.get_operation());
    let negative_sign = negative_sign.get_operation().deref(ctx).get_result(0);
    let pole_sign = llvm::SelectOp::new(ctx, is_pole, plus_one, negative_sign);
    rewriter.insert_operation(ctx, pole_sign.get_operation());
    let pole_sign = pole_sign.get_operation().deref(ctx).get_result(0);
    let bits_ty: TypeHandle = IntegerType::get(ctx, width, Signedness::Signless).into();
    let x_bits = llvm::BitcastOp::new(ctx, x, bits_ty);
    rewriter.insert_operation(ctx, x_bits.get_operation());
    let x_bits = x_bits.get_operation().deref(ctx).get_result(0);
    let minus_zero = llvm::ICmpOp::new(
        ctx,
        llvm_export::attributes::ICmpPredicateAttr::EQ,
        x_bits,
        negative_zero_bits,
    );
    rewriter.insert_operation(ctx, minus_zero.get_operation());
    let minus_zero = minus_zero.get_operation().deref(ctx).get_result(0);
    let sign = llvm::SelectOp::new(ctx, minus_zero, minus_one, pole_sign);
    rewriter.insert_operation(ctx, sign.get_operation());
    let sign = sign.get_operation().deref(ctx).get_result(0);
    let minus_infinity = llvm::ICmpOp::new(
        ctx,
        llvm_export::attributes::ICmpPredicateAttr::EQ,
        x_bits,
        negative_infinity_bits,
    );
    rewriter.insert_operation(ctx, minus_infinity.get_operation());
    let minus_infinity = minus_infinity.get_operation().deref(ctx).get_result(0);
    let normalized = llvm::SelectOp::new(ctx, minus_infinity, positive_infinity, magnitude);
    rewriter.insert_operation(ctx, normalized.get_operation());
    let magnitude = normalized.get_operation().deref(ctx).get_result(0);
    pack_math_out_param_pair(ctx, rewriter, op, &layout, [magnitude, sign])
}

/// Adapt libdevice's scalar return and pointer result (`modf`/`remquo`) to libm's tuple ABI.
fn convert_rust_out_param_pair(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    symbol: &'static str,
    is_remquo: bool,
    args: &[Value],
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    let float_ty = match intrinsic {
        RustFloatMathIntrinsic::ModfF32 | RustFloatMathIntrinsic::RemquoF32 => {
            FP32Type::get(ctx).into()
        }
        RustFloatMathIntrinsic::ModfF64 | RustFloatMathIntrinsic::RemquoF64 => {
            FP64Type::get(ctx).into()
        }
        _ => unreachable!("out-parameter pair converter requires modf or remquo"),
    };
    let i32_mir: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let second_ty = if is_remquo { i32_mir } else { float_ty };
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let layout = {
        let result = result_ty.deref(ctx);
        let Some(tuple) = result.downcast_ref::<MirTupleType>() else {
            return pliron::input_err!(loc, "Rust math out-parameter requires a tuple result");
        };
        if tuple.get_types() != [float_ty, second_ty]
            || args.iter().any(|arg| arg.get_type(ctx) != float_ty)
        {
            return pliron::input_err!(loc, "Rust math out-parameter signature mismatch");
        }
        StructLayoutInfo::of_tuple(tuple)
    };
    let zero_ty = if is_remquo { i32_mir } else { float_ty };
    let zero = if is_remquo {
        let attr = IntegerAttr::new(
            IntegerType::get(ctx, 32, Signedness::Signless),
            APInt::from_u64(0, NonZeroUsize::new(32).unwrap()),
        );
        let constant = llvm::ConstantOp::new(ctx, Box::new(attr));
        rewriter.insert_operation(ctx, constant.get_operation());
        constant.get_operation().deref(ctx).get_result(0)
    } else {
        // The out slot is overwritten by libdevice; its initial value only
        // prevents an undefined read if a backend optimizes a special case.
        let constant = llvm::UndefOp::new(ctx, zero_ty);
        rewriter.insert_operation(ctx, constant.get_operation());
        constant.get_operation().deref(ctx).get_result(0)
    };
    let (result, out) =
        emit_libdevice_out_param_call(ctx, rewriter, op, symbol, args, float_ty, zero)?;
    // libm returns quotient zero for invalid remquo inputs. CUDA specifies
    // a NaN remainder but does not define the pointer result in those cases.
    let out = if is_remquo {
        let invalid = llvm::FCmpOp::new(ctx, FCmpPredicateAttr::UNO, result, result);
        set_empty_fastmath_flags(ctx, invalid.get_operation());
        rewriter.insert_operation(ctx, invalid.get_operation());
        let predicate = invalid.get_operation().deref(ctx).get_result(0);
        let normalized = llvm::SelectOp::new(ctx, predicate, zero, out);
        rewriter.insert_operation(ctx, normalized.get_operation());
        normalized.get_operation().deref(ctx).get_result(0)
    } else {
        out
    };
    pack_math_out_param_pair(ctx, rewriter, op, &layout, [result, out])
}

/// Adapt libdevice's `(F, i32*) -> F` ABI to libm's `(F) -> (F, i32)`.
fn convert_rust_frexp(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    symbol: &'static str,
    x: Value,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    let (float_ty, width, exponent_mask): (TypeHandle, u32, u64) = match intrinsic {
        RustFloatMathIntrinsic::FrexpF32 => (FP32Type::get(ctx).into(), 32, 0x7f80_0000),
        RustFloatMathIntrinsic::FrexpF64 => (FP64Type::get(ctx).into(), 64, 0x7ff0_0000_0000_0000),
        _ => unreachable!("frexp converter requires a frexp placeholder"),
    };
    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let layout = {
        let ty = result_mir_ty.deref(ctx);
        let Some(tuple) = ty.downcast_ref::<MirTupleType>() else {
            return pliron::input_err!(loc, "Rust frexp intrinsic requires a (float, i32) result");
        };
        let fields = tuple.get_types();
        let i32_mir: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
        if fields != [float_ty, i32_mir] || x.get_type(ctx) != float_ty {
            return pliron::input_err!(loc, "Rust frexp intrinsic signature mismatch");
        }
        StructLayoutInfo::of_tuple(tuple)
    };
    let mut constant = |ctx: &mut Context, value: u64, bits: u32| {
        let ty = IntegerType::get(ctx, bits, Signedness::Signless);
        let attr = IntegerAttr::new(
            ty,
            APInt::from_u64(value, NonZeroUsize::new(bits as usize).unwrap()),
        );
        let constant = llvm::ConstantOp::new(ctx, Box::new(attr));
        rewriter.insert_operation(ctx, constant.get_operation());
        constant.get_operation().deref(ctx).get_result(0)
    };
    let zero = constant(ctx, 0, 32);
    let mask = constant(ctx, exponent_mask, width);
    let (fraction, exponent) =
        emit_libdevice_out_param_call(ctx, rewriter, op, symbol, &[x], float_ty, zero)?;
    // libm returns (x, 0) for infinities and NaNs; libdevice leaves the
    // exponent unspecified. Bit tests preserve NaN payloads without floating
    // comparisons or fast-math assumptions. The select preserves x itself.
    let bits_ty = IntegerType::get(ctx, width, Signedness::Signless).into();
    let bits = llvm::BitcastOp::new(ctx, x, bits_ty);
    rewriter.insert_operation(ctx, bits.get_operation());
    let bits_value = bits.get_operation().deref(ctx).get_result(0);
    let masked = llvm::AndOp::new(ctx, bits_value, mask);
    rewriter.insert_operation(ctx, masked.get_operation());
    let masked_value = masked.get_operation().deref(ctx).get_result(0);
    let non_finite = llvm::ICmpOp::new(
        ctx,
        llvm_export::attributes::ICmpPredicateAttr::EQ,
        masked_value,
        mask,
    );
    rewriter.insert_operation(ctx, non_finite.get_operation());
    let predicate = non_finite.get_operation().deref(ctx).get_result(0);
    let fraction = llvm::SelectOp::new(ctx, predicate, x, fraction);
    rewriter.insert_operation(ctx, fraction.get_operation());
    let exponent = llvm::SelectOp::new(ctx, predicate, zero, exponent);
    rewriter.insert_operation(ctx, exponent.get_operation());
    let fields = [
        fraction.get_operation().deref(ctx).get_result(0),
        exponent.get_operation().deref(ctx).get_result(0),
    ];
    pack_math_out_param_pair(ctx, rewriter, op, &layout, fields)
}

/// Resolve the callee symbol a float-math intrinsic lowers to under the
/// active intrinsic backend.
fn float_math_intrinsic_symbol(
    ctx: &Context,
    intrinsic: RustFloatMathIntrinsic,
    result_ty: TypeHandle,
    loc: pliron::location::Location,
) -> Result<&'static str> {
    match crate::context::lowering_options(ctx).intrinsic_backend {
        crate::IntrinsicBackend::LlvmNvptx => {
            match intrinsic.llvm_intrinsic_name(ctx, result_ty, loc.clone())? {
                Some(llvm_name) => Ok(llvm_name),
                None => intrinsic.libdevice_name(ctx, result_ty, loc),
            }
        }
        crate::IntrinsicBackend::LibNvvm => intrinsic.libdevice_name(ctx, result_ty, loc),
    }
}

/// Preserve IEEE-754-2019 minimum/maximum and minimumNumber/maximumNumber.
/// Ordered comparisons select unequal operands, while the sign bit of x
/// breaks zero ties. The final NaN selects implement propagation or numeric
/// preference without NSZ/NNAN flags, unlike legacy fmin/fmax.
fn lower_float_ieee_minmax(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
    args: &[Value],
    is_max: bool,
    propagate_nan: bool,
) -> Result<()> {
    let [x, y] = args else {
        return pliron::input_err!(op.deref(ctx).loc(), "C23 min/max requires two operands");
    };
    let width = match intrinsic {
        RustFloatMathIntrinsic::MaximumF32
        | RustFloatMathIntrinsic::MinimumF32
        | RustFloatMathIntrinsic::MaximumNumberF32
        | RustFloatMathIntrinsic::MinimumNumberF32 => 32,
        RustFloatMathIntrinsic::MaximumF64
        | RustFloatMathIntrinsic::MinimumF64
        | RustFloatMathIntrinsic::MaximumNumberF64
        | RustFloatMathIntrinsic::MinimumNumberF64 => 64,
        _ => unreachable!("C23 min/max converter requires its own placeholder"),
    };
    let loc = op.deref(ctx).loc().clone();
    let float_ty: TypeHandle = if width == 32 {
        FP32Type::get(ctx).into()
    } else {
        FP64Type::get(ctx).into()
    };
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, mir_result_ty).map_err(anyhow_to_pliron)?;
    if result_ty != float_ty || x.get_type(ctx) != float_ty || y.get_type(ctx) != float_ty {
        return pliron::input_err!(loc, "C23 min/max signature mismatch");
    }

    let x_bits_ty: TypeHandle = IntegerType::get(ctx, width, Signedness::Signless).into();
    let x_bits = llvm::BitcastOp::new(ctx, *x, x_bits_ty);
    rewriter.insert_operation(ctx, x_bits.get_operation());
    let x_bits = x_bits.get_operation().deref(ctx).get_result(0);
    let sign_attr = IntegerAttr::new(
        IntegerType::get(ctx, width, Signedness::Signless),
        APInt::from_u64(
            1_u64 << (width - 1),
            NonZeroUsize::new(width as usize).unwrap(),
        ),
    );
    let sign_mask = llvm::ConstantOp::new(ctx, Box::new(sign_attr));
    rewriter.insert_operation(ctx, sign_mask.get_operation());
    let sign_mask = sign_mask.get_operation().deref(ctx).get_result(0);
    let zero_attr = IntegerAttr::new(
        IntegerType::get(ctx, width, Signedness::Signless),
        APInt::from_u64(0, NonZeroUsize::new(width as usize).unwrap()),
    );
    let zero = llvm::ConstantOp::new(ctx, Box::new(zero_attr));
    rewriter.insert_operation(ctx, zero.get_operation());
    let zero = zero.get_operation().deref(ctx).get_result(0);
    let masked = llvm::AndOp::new(ctx, x_bits, sign_mask);
    rewriter.insert_operation(ctx, masked.get_operation());
    let masked = masked.get_operation().deref(ctx).get_result(0);
    let is_positive = llvm::ICmpOp::new(
        ctx,
        llvm_export::attributes::ICmpPredicateAttr::EQ,
        masked,
        zero,
    );
    rewriter.insert_operation(ctx, is_positive.get_operation());
    let is_positive = is_positive.get_operation().deref(ctx).get_result(0);

    let mut compare = |ctx: &mut Context, predicate: FCmpPredicateAttr, a: Value, b: Value| {
        let cmp = llvm::FCmpOp::new(ctx, predicate, a, b);
        set_empty_fastmath_flags(ctx, cmp.get_operation());
        rewriter.insert_operation(ctx, cmp.get_operation());
        cmp.get_operation().deref(ctx).get_result(0)
    };
    let better_x = compare(
        ctx,
        if is_max {
            FCmpPredicateAttr::OGT
        } else {
            FCmpPredicateAttr::OLT
        },
        *x,
        *y,
    );
    let better_y = compare(
        ctx,
        if is_max {
            FCmpPredicateAttr::OLT
        } else {
            FCmpPredicateAttr::OGT
        },
        *x,
        *y,
    );
    let x_nan = compare(ctx, FCmpPredicateAttr::UNO, *x, *x);
    let y_nan = compare(ctx, FCmpPredicateAttr::UNO, *y, *y);

    let (tie_positive, tie_negative) = if is_max { (*x, *y) } else { (*y, *x) };
    let tie = llvm::SelectOp::new(ctx, is_positive, tie_positive, tie_negative);
    rewriter.insert_operation(ctx, tie.get_operation());
    let tie = tie.get_operation().deref(ctx).get_result(0);
    let ordered_y = llvm::SelectOp::new(ctx, better_y, *y, tie);
    rewriter.insert_operation(ctx, ordered_y.get_operation());
    let ordered_y = ordered_y.get_operation().deref(ctx).get_result(0);
    let ordered = llvm::SelectOp::new(ctx, better_x, *x, ordered_y);
    rewriter.insert_operation(ctx, ordered.get_operation());
    let ordered = ordered.get_operation().deref(ctx).get_result(0);
    let y_case = llvm::SelectOp::new(ctx, y_nan, if propagate_nan { *y } else { *x }, ordered);
    rewriter.insert_operation(ctx, y_case.get_operation());
    let y_case = y_case.get_operation().deref(ctx).get_result(0);
    let result = llvm::SelectOp::new(ctx, x_nan, if propagate_nan { *x } else { *y }, y_case);
    rewriter.insert_operation(ctx, result.get_operation());
    rewriter.replace_operation(ctx, op, result.get_operation());
    Ok(())
}

/// Lower Rust's NaN-ignoring `min` and `max` contract without libdevice.
fn lower_float_minmax_nsz(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    args: &[Value],
    ordered_predicate: FCmpPredicateAttr,
) -> Result<()> {
    let [x, y] = args else {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "Rust float min/max intrinsic requires two operands"
        );
    };

    let x_is_nan = llvm::FCmpOp::new(ctx, FCmpPredicateAttr::UNO, *x, *x).get_operation();
    set_empty_fastmath_flags(ctx, x_is_nan);
    rewriter.insert_operation(ctx, x_is_nan);

    let y_ordered_against_x = llvm::FCmpOp::new(ctx, ordered_predicate, *y, *x).get_operation();
    set_empty_fastmath_flags(ctx, y_ordered_against_x);
    rewriter.insert_operation(ctx, y_ordered_against_x);

    let y_is_ordered_choice = y_ordered_against_x.deref(ctx).get_result(0);
    let ordered_result = llvm::SelectOp::new(ctx, y_is_ordered_choice, *y, *x).get_operation();
    rewriter.insert_operation(ctx, ordered_result);

    let x_is_nan_value = x_is_nan.deref(ctx).get_result(0);
    let ordered_result_value = ordered_result.deref(ctx).get_result(0);
    let result = llvm::SelectOp::new(ctx, x_is_nan_value, *y, ordered_result_value).get_operation();
    rewriter.insert_operation(ctx, result);
    rewriter.replace_operation(ctx, op, result);
    Ok(())
}

/// Attach empty fast-math flags to a float op.
fn set_empty_fastmath_flags(ctx: &mut Context, op: Ptr<Operation>) {
    let key: pliron::identifier::Identifier = "llvm_fast_math_flags".try_into().unwrap();
    op.deref_mut(ctx)
        .attributes
        .set(key, FastmathFlagsAttr::default());
}

/// Compute the LLVM fast-math flag set for `core::intrinsics::f*_fast`.
fn fast_float_intrinsic_flags(ctx: &Context) -> FastmathFlagsAttr {
    let mut flags = FastmathFlags::FAST;
    if !crate::context::lowering_options(ctx).allow_fma_contraction {
        flags.remove(FastmathFlags::CONTRACT);
    }
    flags.into()
}

/// Lower a `core::intrinsics::f*_fast` placeholder call to the matching LLVM float binop.
fn lower_fast_binop(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    args: &[Value],
    binop: FastFloatBinop,
    loc: pliron::location::Location,
) -> Result<()> {
    let (lhs, rhs) = match args {
        [a, b] => (*a, *b),
        _ => {
            return pliron::input_err!(loc, "f*_fast intrinsic call must have exactly 2 operands");
        }
    };

    let flags = fast_float_intrinsic_flags(ctx);
    let llvm_op = match binop {
        FastFloatBinop::Add => {
            llvm::FAddOp::new_with_fast_math_flags(ctx, lhs, rhs, flags).get_operation()
        }
        FastFloatBinop::Sub => {
            llvm::FSubOp::new_with_fast_math_flags(ctx, lhs, rhs, flags).get_operation()
        }
        FastFloatBinop::Mul => {
            llvm::FMulOp::new_with_fast_math_flags(ctx, lhs, rhs, flags).get_operation()
        }
        FastFloatBinop::Div => {
            llvm::FDivOp::new_with_fast_math_flags(ctx, lhs, rhs, flags).get_operation()
        }
        FastFloatBinop::Rem => {
            llvm::FRemOp::new_with_fast_math_flags(ctx, lhs, rhs, flags).get_operation()
        }
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Return the LLVM `fabs` intrinsic for the concrete float type.
fn fabs_llvm_intrinsic_name(
    ctx: &Context,
    ty: TypeHandle,
    loc: pliron::location::Location,
) -> Result<&'static str> {
    let is_f64 = float_math_is_f64(ctx, ty, loc)?;
    Ok(RustFloatMathIntrinsic::Fabs
        .llvm_symbol(is_f64)
        .expect("generated fabs LLVM symbol"))
}

/// Return the libdevice `fabs` entry point for the concrete float type.
fn fabs_libdevice_name(
    ctx: &Context,
    ty: TypeHandle,
    loc: pliron::location::Location,
) -> Result<&'static str> {
    let is_f64 = float_math_is_f64(ctx, ty, loc)?;
    Ok(RustFloatMathIntrinsic::Fabs
        .libdevice_symbol(is_f64)
        .expect("generated fabs libdevice symbol"))
}

/// Generic math placeholders recover their width from the concrete float type.
fn float_math_is_f64(
    ctx: &Context,
    ty: TypeHandle,
    loc: pliron::location::Location,
) -> Result<bool> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<FP32Type>() {
        Ok(false)
    } else if ty_ref.is::<FP64Type>() {
        Ok(true)
    } else {
        pliron::input_err!(
            loc,
            "expected f32 or f64 type for Rust float math intrinsic"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `tgamma`, `ilogb`, and `ldexp` resolve to the expected libdevice symbols on both backends.
    #[test]
    fn special_math_symbols_and_arities_match_on_both_backends() {
        use RustFloatMathIntrinsic as I;
        for backend in [
            crate::IntrinsicBackend::LlvmNvptx,
            crate::IntrinsicBackend::LibNvvm,
        ] {
            let mut ctx = Context::new();
            crate::context::set_lowering_options(
                &mut ctx,
                crate::LoweringOptions {
                    intrinsic_backend: backend,
                    ..Default::default()
                },
            );
            let f32_ty = FP32Type::get(&ctx).into();
            let f64_ty = FP64Type::get(&ctx).into();
            let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless).into();
            for (callee, intrinsic, result_ty, symbol, arity) in [
                (
                    rust_intrinsics::CALLEE_TGAMMA_F32,
                    I::TgammaF32,
                    f32_ty,
                    "__nv_tgammaf",
                    1,
                ),
                (
                    rust_intrinsics::CALLEE_TGAMMA_F64,
                    I::TgammaF64,
                    f64_ty,
                    "__nv_tgamma",
                    1,
                ),
                (
                    rust_intrinsics::CALLEE_ILOGB_F32,
                    I::IlogbF32,
                    i32_ty,
                    "__nv_ilogbf",
                    1,
                ),
                (
                    rust_intrinsics::CALLEE_ILOGB_F64,
                    I::IlogbF64,
                    i32_ty,
                    "__nv_ilogb",
                    1,
                ),
                (
                    rust_intrinsics::CALLEE_LDEXP_F32,
                    I::LdexpF32,
                    f32_ty,
                    "__nv_ldexpf",
                    2,
                ),
                (
                    rust_intrinsics::CALLEE_LDEXP_F64,
                    I::LdexpF64,
                    f64_ty,
                    "__nv_ldexp",
                    2,
                ),
            ] {
                assert_eq!(I::from_placeholder_callee(callee), Some(intrinsic));
                assert_eq!(intrinsic.arg_count(), arity);
                assert_eq!(
                    float_math_intrinsic_symbol(
                        &ctx,
                        intrinsic,
                        result_ty,
                        pliron::location::Location::Unknown,
                    )
                    .unwrap(),
                    symbol
                );
                assert!(rust_intrinsics::is_libdevice_backed_placeholder(callee));
                assert!(!rust_intrinsics::is_backend_dependent_libdevice_placeholder(callee));
            }
        }
    }

    /// Ensure disabling FMA contraction clears only `CONTRACT` while preserving other fast-math flags.
    #[test]
    fn no_fma_policy_removes_only_contract_from_fast_float_intrinsics() {
        let mut ctx = Context::new();
        crate::context::set_lowering_options(
            &mut ctx,
            crate::LoweringOptions {
                allow_fma_contraction: false,
                intrinsic_backend: crate::IntrinsicBackend::LlvmNvptx,
                module_disambiguator: None,
            },
        );

        let flags = fast_float_intrinsic_flags(&ctx).0;
        assert!(!flags.contains(FastmathFlags::CONTRACT));
        assert!(flags.contains(FastmathFlags::REASSOC));
        assert!(flags.contains(FastmathFlags::NNAN));
        assert!(flags.contains(FastmathFlags::NINF));
    }

    /// Sample of the Rust float-math placeholder to `RustFloatMathIntrinsic` round-trip mapping.
    #[test]
    fn test_float_math_placeholder_round_trip() {
        let cases: &[(&str, RustFloatMathIntrinsic)] = &[
            (
                rust_intrinsics::CALLEE_SQRT_F32,
                RustFloatMathIntrinsic::SqrtF32,
            ),
            (
                rust_intrinsics::CALLEE_SQRT_F64,
                RustFloatMathIntrinsic::SqrtF64,
            ),
            (
                rust_intrinsics::CALLEE_POWI_F32,
                RustFloatMathIntrinsic::PowiF32,
            ),
            (
                rust_intrinsics::CALLEE_POWI_F64,
                RustFloatMathIntrinsic::PowiF64,
            ),
            (
                rust_intrinsics::CALLEE_FMA_F32,
                RustFloatMathIntrinsic::FmaF32,
            ),
            (
                rust_intrinsics::CALLEE_FMULADD_F64,
                RustFloatMathIntrinsic::FmuladdF64,
            ),
            (rust_intrinsics::CALLEE_FABS, RustFloatMathIntrinsic::Fabs),
            (
                rust_intrinsics::CALLEE_COPYSIGN_F32,
                RustFloatMathIntrinsic::CopysignF32,
            ),
            (
                rust_intrinsics::CALLEE_LOG2_F64,
                RustFloatMathIntrinsic::Log2F64,
            ),
            (
                rust_intrinsics::CALLEE_MAXNUM_NSZ_F32,
                RustFloatMathIntrinsic::MaxNumNszF32,
            ),
            (
                rust_intrinsics::CALLEE_MAXNUM_NSZ_F64,
                RustFloatMathIntrinsic::MaxNumNszF64,
            ),
            (
                rust_intrinsics::CALLEE_MINNUM_NSZ_F32,
                RustFloatMathIntrinsic::MinNumNszF32,
            ),
            (
                rust_intrinsics::CALLEE_MINNUM_NSZ_F64,
                RustFloatMathIntrinsic::MinNumNszF64,
            ),
            (
                rust_intrinsics::CALLEE_ASIN_F32,
                RustFloatMathIntrinsic::AsinF32,
            ),
            (
                rust_intrinsics::CALLEE_ASIN_F64,
                RustFloatMathIntrinsic::AsinF64,
            ),
            (
                rust_intrinsics::CALLEE_ACOS_F32,
                RustFloatMathIntrinsic::AcosF32,
            ),
            (
                rust_intrinsics::CALLEE_ACOS_F64,
                RustFloatMathIntrinsic::AcosF64,
            ),
            (
                rust_intrinsics::CALLEE_ATAN2_F32,
                RustFloatMathIntrinsic::Atan2F32,
            ),
            (
                rust_intrinsics::CALLEE_ATAN2_F64,
                RustFloatMathIntrinsic::Atan2F64,
            ),
            (
                rust_intrinsics::CALLEE_ATAN_F32,
                RustFloatMathIntrinsic::AtanF32,
            ),
            (
                rust_intrinsics::CALLEE_ATAN_F64,
                RustFloatMathIntrinsic::AtanF64,
            ),
            (
                rust_intrinsics::CALLEE_CBRT_F32,
                RustFloatMathIntrinsic::CbrtF32,
            ),
            (
                rust_intrinsics::CALLEE_CBRT_F64,
                RustFloatMathIntrinsic::CbrtF64,
            ),
        ];

        for (name, expected) in cases {
            assert_eq!(
                RustFloatMathIntrinsic::from_placeholder_callee(name),
                Some(*expected),
                "placeholder `{name}` did not round-trip"
            );
        }

        assert_eq!(
            RustFloatMathIntrinsic::from_placeholder_callee("not_a_placeholder"),
            None,
        );
    }

    /// Check operand counts across unary, binary, and ternary float-math intrinsics.
    #[test]
    fn test_float_math_arg_count() {
        assert_eq!(RustFloatMathIntrinsic::SqrtF32.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::Fabs.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::FloorF64.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::PowiF32.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::PowfF64.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::CopysignF32.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::AsinF32.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::AsinF64.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::AcosF32.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::AcosF64.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::Atan2F32.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::Atan2F64.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::AtanF32.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::AtanF64.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::CbrtF32.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::CbrtF64.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::FmaF32.arg_count(), 3);
        assert_eq!(RustFloatMathIntrinsic::FmuladdF64.arg_count(), 3);
        assert_eq!(RustFloatMathIntrinsic::MaxNumNszF32.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::MaxNumNszF64.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::MinNumNszF32.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::MinNumNszF64.arg_count(), 2);
    }

    /// Verify ordered comparison predicates for NaN-ignoring `min`/`max` (`*_nsz`) variants.
    #[test]
    fn test_float_math_minmax_nsz_predicates() {
        assert_eq!(
            RustFloatMathIntrinsic::MaxNumNszF32.minmax_nsz_predicate(),
            Some(FCmpPredicateAttr::OGE)
        );
        assert_eq!(
            RustFloatMathIntrinsic::MaxNumNszF64.minmax_nsz_predicate(),
            Some(FCmpPredicateAttr::OGE)
        );
        assert_eq!(
            RustFloatMathIntrinsic::MinNumNszF32.minmax_nsz_predicate(),
            Some(FCmpPredicateAttr::OLE)
        );
        assert_eq!(
            RustFloatMathIntrinsic::MinNumNszF64.minmax_nsz_predicate(),
            Some(FCmpPredicateAttr::OLE)
        );
        assert_eq!(RustFloatMathIntrinsic::Fabs.minmax_nsz_predicate(), None);
    }

    /// Confirm `copysign` resolves to native LLVM intrinsics while transcendentals return `None`.
    #[test]
    fn test_float_math_copysign_uses_llvm_and_transcendentals_use_libdevice() {
        let ctx = Context::new();
        let f32_ty = FP32Type::get(&ctx).into();
        let f64_ty = FP64Type::get(&ctx).into();
        let loc = pliron::location::Location::Unknown;

        assert_eq!(
            RustFloatMathIntrinsic::CopysignF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_copysign_f32")
        );
        assert_eq!(
            RustFloatMathIntrinsic::CopysignF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            Some("llvm_copysign_f64")
        );
        assert_eq!(
            RustFloatMathIntrinsic::SinF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc)
                .unwrap(),
            None
        );
    }

    /// Ensure generic `Fabs` dispatches on the concrete float width and rejects non-float types.
    #[test]
    fn test_fabs_llvm_intrinsic_name_dispatches_on_float_width() {
        let ctx = Context::new();
        let f32_ty = FP32Type::get(&ctx).into();
        let f64_ty = FP64Type::get(&ctx).into();
        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let loc = pliron::location::Location::Unknown;

        assert_eq!(
            fabs_llvm_intrinsic_name(&ctx, f32_ty, loc.clone()).unwrap(),
            "llvm_fabs_f32"
        );
        assert_eq!(
            fabs_llvm_intrinsic_name(&ctx, f64_ty, loc.clone()).unwrap(),
            "llvm_fabs_f64"
        );
        assert!(fabs_llvm_intrinsic_name(&ctx, i32_ty, loc).is_err());
    }

    /// Verify native LLVM intrinsic routing for rounding and sign operations.
    #[test]
    fn test_llvm_intrinsic_name_routing() {
        let ctx = Context::new();
        let f32_ty = FP32Type::get(&ctx).into();
        let f64_ty = FP64Type::get(&ctx).into();
        let loc = pliron::location::Location::Unknown;

        assert_eq!(
            RustFloatMathIntrinsic::Fabs
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_fabs_f32")
        );
        assert_eq!(
            RustFloatMathIntrinsic::Fabs
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            Some("llvm_fabs_f64")
        );
        assert_eq!(
            RustFloatMathIntrinsic::CopysignF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_copysign_f32")
        );

        assert_eq!(
            RustFloatMathIntrinsic::FloorF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_floor_f32")
        );
        assert_eq!(
            RustFloatMathIntrinsic::FloorF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            Some("llvm_floor_f64")
        );
        assert_eq!(
            RustFloatMathIntrinsic::CeilF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_ceil_f32")
        );
        assert_eq!(
            RustFloatMathIntrinsic::CeilF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            Some("llvm_ceil_f64")
        );
        assert_eq!(
            RustFloatMathIntrinsic::TruncF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_trunc_f32")
        );
        assert_eq!(
            RustFloatMathIntrinsic::TruncF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            Some("llvm_trunc_f64")
        );
        assert_eq!(
            RustFloatMathIntrinsic::RoundF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_round_f32")
        );
        assert_eq!(
            RustFloatMathIntrinsic::RoundF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            Some("llvm_round_f64")
        );
        assert_eq!(
            RustFloatMathIntrinsic::RoundevenF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            Some("llvm_roundeven_f32")
        );
        assert_eq!(
            RustFloatMathIntrinsic::RoundevenF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            Some("llvm_roundeven_f64")
        );

        assert_eq!(
            RustFloatMathIntrinsic::MaxNumNszF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            None
        );
        assert_eq!(
            RustFloatMathIntrinsic::MinNumNszF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc.clone())
                .unwrap(),
            None
        );

        assert_eq!(
            RustFloatMathIntrinsic::SinF32
                .llvm_intrinsic_name(&ctx, f32_ty, loc.clone())
                .unwrap(),
            None
        );
        assert_eq!(
            RustFloatMathIntrinsic::SqrtF64
                .llvm_intrinsic_name(&ctx, f64_ty, loc)
                .unwrap(),
            None
        );
    }

    /// Verify that backend-dependent operations switch between LLVM intrinsics and libdevice.
    #[test]
    fn float_math_symbol_dispatches_on_intrinsic_backend() {
        let mut ctx = Context::new();
        let f32_ty = FP32Type::get(&ctx).into();
        let loc = pliron::location::Location::Unknown;

        crate::context::set_lowering_options(
            &mut ctx,
            crate::LoweringOptions {
                allow_fma_contraction: true,
                intrinsic_backend: crate::IntrinsicBackend::LlvmNvptx,
                module_disambiguator: None,
            },
        );
        assert_eq!(
            float_math_intrinsic_symbol(
                &ctx,
                RustFloatMathIntrinsic::FloorF32,
                f32_ty,
                loc.clone()
            )
            .unwrap(),
            "llvm_floor_f32"
        );
        assert_eq!(
            float_math_intrinsic_symbol(
                &ctx,
                RustFloatMathIntrinsic::RoundevenF32,
                f32_ty,
                loc.clone()
            )
            .unwrap(),
            "llvm_roundeven_f32"
        );
        assert_eq!(
            float_math_intrinsic_symbol(&ctx, RustFloatMathIntrinsic::Fabs, f32_ty, loc.clone())
                .unwrap(),
            "llvm_fabs_f32"
        );
        assert_eq!(
            float_math_intrinsic_symbol(
                &ctx,
                RustFloatMathIntrinsic::CopysignF32,
                f32_ty,
                loc.clone()
            )
            .unwrap(),
            "llvm_copysign_f32"
        );
        assert_eq!(
            float_math_intrinsic_symbol(&ctx, RustFloatMathIntrinsic::SinF32, f32_ty, loc.clone())
                .unwrap(),
            "__nv_sinf"
        );

        crate::context::set_lowering_options(
            &mut ctx,
            crate::LoweringOptions {
                allow_fma_contraction: true,
                intrinsic_backend: crate::IntrinsicBackend::LibNvvm,
                module_disambiguator: None,
            },
        );
        assert_eq!(
            float_math_intrinsic_symbol(
                &ctx,
                RustFloatMathIntrinsic::FloorF32,
                f32_ty,
                loc.clone()
            )
            .unwrap(),
            "__nv_floorf"
        );
        assert_eq!(
            float_math_intrinsic_symbol(
                &ctx,
                RustFloatMathIntrinsic::RoundevenF32,
                f32_ty,
                loc.clone()
            )
            .unwrap(),
            "__nv_rintf"
        );
        assert_eq!(
            float_math_intrinsic_symbol(&ctx, RustFloatMathIntrinsic::Fabs, f32_ty, loc.clone())
                .unwrap(),
            "__nv_fabsf"
        );
        assert_eq!(
            float_math_intrinsic_symbol(
                &ctx,
                RustFloatMathIntrinsic::CopysignF32,
                f32_ty,
                loc.clone()
            )
            .unwrap(),
            "__nv_copysignf"
        );
        assert_eq!(
            float_math_intrinsic_symbol(&ctx, RustFloatMathIntrinsic::SinF32, f32_ty, loc).unwrap(),
            "__nv_sinf"
        );
    }
}
