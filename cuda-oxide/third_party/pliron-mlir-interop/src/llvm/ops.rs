// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! MLIR translation for LLVM dialect ops.

use std::fmt;

use pliron::{
    attribute::Attribute,
    builtin::{
        attributes::{BoolAttr, IntegerAttr, StringAttr, TypeAttr, UnitAttr},
        op_interfaces::{CallOpInterface, OperandSegmentInterface, SymbolOpInterface},
    },
    context::{Context, Ptr},
    derive::op_interface_impl,
    identifier::Identifier,
    input_err,
    op::Op,
    operation::Operation,
    printable::State,
    result::Result,
    r#type::{Typed, TypedHandle},
    value::Value,
};
use pliron_llvm::{
    attributes::{
        AlignmentAttr, FastmathFlagsAttr, IntegerOverflowFlagsAttr, LinkageAttr, PoisonAttr,
        SymbolAddrAttr, SyncScopeAttr, UndefAttr, ZeroAttr,
    },
    llvm_attrs::{LlvmAttrValue, LlvmAttributesAttr},
    op_interfaces::{
        ATTR_KEY_FAST_MATH_FLAGS, ATTR_KEY_INTEGER_OVERFLOW_FLAGS, ATTR_KEY_NNEG_FLAG,
        AlignableOpInterface, SyncScopeInterface, VolatilityOpInterface,
    },
    ops::{
        AShrOp, AddOp, AddrSpaceCastOp, AddressOfOp, AllocaOp, AndOp, AtomicCmpxchgOp,
        AtomicLoadOp, AtomicRmwOp, AtomicStoreOp, BitcastOp, BlockAddressOp, BlockTagOp, BrOp,
        CallIntrinsicOp, CallOp, CondBrOp, ConstantOp, ExtractElementOp, ExtractValueOp, FAddOp,
        FCmpOp, FDivOp, FMulOp, FNegOp, FPExtOp, FPToSIOp, FPToUIOp, FPTruncOp, FRemOp, FSubOp,
        FenceOp, FreezeOp, FuncOp, GetElementPtrOp, GlobalOp, ICmpOp, IndirectBrOp, InlineAsmOp,
        InsertElementOp, InsertValueOp, IntToPtrOp, LShrOp, LoadOp, MulOp, OrOp, PoisonOp,
        PtrToIntOp, ReturnOp, SDivOp, SExtOp, SIToFPOp, SRemOp, SelectOp, ShlOp, ShuffleVectorOp,
        StoreOp, SubOp, SwitchOp, TruncOp, UDivOp, UIToFPOp, URemOp, UndefOp, UnreachableOp,
        VAArgOp, XorOp, ZExtOp, ZeroOp,
    },
    types::{FuncType, VoidType},
};

use crate::{
    Error, ToMlirOp,
    llvm::lowering::{constant_needs_lowering, global_initializer_needs_lowering},
    printers::{GenericOp, mlir_dense_array, mlir_string_literal},
};

/// `op`'s results, minus any [VoidType] one.
///
/// pliron gives an op that produces nothing a single `!llvm.void` result;
/// MLIR gives it no result at all.
fn non_void_results(op: Ptr<Operation>, ctx: &Context) -> Vec<Value> {
    op.deref(ctx)
        .results()
        .filter(|res| !res.get_type(ctx).deref(ctx).is::<VoidType>())
        .collect()
}

/// `op`'s `key` attribute, if it is set.
///
/// pliron's interface getters for the flag attributes panic when they are
/// unset, but MLIR's default is unset.
fn op_attr<A: Attribute + Clone>(op: Ptr<Operation>, ctx: &Context, key: &Identifier) -> Option<A> {
    op.deref(ctx).attributes.get::<A>(key).cloned()
}

/// MLIR's `alignment` property, which pliron leaves unset when it has none.
fn alignment_attr(op: &impl AlignableOpInterface, ctx: &Context) -> Option<AlignmentAttr> {
    op.alignment(ctx).map(AlignmentAttr)
}

/// MLIR's `syncscope` property, which MLIR leaves unset for the system scope.
fn syncscope_name(op: &impl SyncScopeInterface, ctx: &Context) -> Option<StringAttr> {
    match op.syncscope(ctx) {
        SyncScopeAttr::System => None,
        scope => Some(StringAttr::new(scope.to_name())),
    }
}

/// An [Error::Untranslatable] error at the location of `op`.
fn untranslatable<T>(op: &impl Op, ctx: &Context, msg: impl Into<String>) -> Result<T> {
    input_err!(op.loc(ctx), Error::Untranslatable(msg.into()))
}

/// The error for an `op` that needs
/// [lower_for_mlir](crate::llvm::lowering::lower_for_mlir) before printing.
fn lowering_err(op: &impl Op, ctx: &Context, what: &str) -> Result<()> {
    untranslatable(
        op,
        ctx,
        format!("{what} has no MLIR attribute; run `lower_for_mlir` before printing"),
    )
}

/// Error: `name` (LLVM function attribute) cannot be held in `mnemonic` (op).
fn fn_attr_err<T>(op: &impl Op, ctx: &Context, mnemonic: &str, name: &str) -> Result<T> {
    untranslatable(
        op,
        ctx,
        format!("MLIR's `{mnemonic}` cannot hold the LLVM function attribute `{name}`"),
    )
}

/// The MLIR unit property of `llvm.call` (if `is_call`) or `llvm.func` for the
/// LLVM function attribute `name`.
fn unit_prop(name: &str, is_call: bool) -> Option<&'static str> {
    Some(match name {
        "alwaysinline" => "always_inline",
        "cold" => "cold",
        "convergent" => "convergent",
        "hot" => "hot",
        "inlinehint" => "inline_hint",
        "minsize" => "minsize",
        "no_caller_saved_registers" => "no_caller_saved_registers",
        "nocallback" => "nocallback",
        "noduplicate" => "noduplicate",
        "noinline" => "no_inline",
        "noreturn" => "noreturn",
        "nounwind" => "no_unwind",
        "optsize" => "optsize",
        "returns_twice" => "returns_twice",
        "save-reg-params" => "save_reg_params",
        "willreturn" => "will_return",
        "optnone" if !is_call => "optimize_none",
        "builtin" if is_call => "builtin",
        "nobuiltin" if is_call => "nobuiltin",
        _ => return None,
    })
}

/// Add the `llvm.func` properties for the LLVM function attributes `attrs`
/// to `generic`.
fn add_func_attrs<'a>(
    mut generic: GenericOp<'a>,
    op: &impl Op,
    ctx: &Context,
    attrs: Option<&LlvmAttributesAttr>,
) -> Result<GenericOp<'a>> {
    let mut passthrough = vec![];
    for (name, value) in attrs.iter().flat_map(|attrs| attrs.iter()) {
        let pair = |value: &str| {
            format!(
                "[{}, {}]",
                mlir_string_literal(name),
                mlir_string_literal(value)
            )
        };
        passthrough.push(match (value, unit_prop(name, false)) {
            (LlvmAttrValue::Unit, Some(mlir_name)) => {
                generic = generic.prop_unit(mlir_name, true);
                continue;
            }
            (LlvmAttrValue::Unit, None) => mlir_string_literal(name),
            (LlvmAttrValue::Int(int), _) => pair(&int.to_string()),
            (LlvmAttrValue::Str(str), _) => pair(str),
            (LlvmAttrValue::Type(_), _) => return fn_attr_err(op, ctx, "llvm.func", name),
        });
    }
    if !passthrough.is_empty() {
        generic = generic.prop_raw("passthrough", format!("[{}]", passthrough.join(", ")));
    }
    Ok(generic)
}

/// Add the `llvm.call` properties for the LLVM function attributes `attrs`
/// to `generic`.
fn add_call_attrs<'a>(
    mut generic: GenericOp<'a>,
    op: &impl Op,
    ctx: &Context,
    attrs: Option<&LlvmAttributesAttr>,
) -> Result<GenericOp<'a>> {
    let mut default_func_attrs = vec![];
    for (name, value) in attrs.iter().flat_map(|attrs| attrs.iter()) {
        generic = match (value, unit_prop(name, true), name) {
            (LlvmAttrValue::Unit, Some(mlir_name), _) => generic.prop_unit(mlir_name, true),
            (LlvmAttrValue::Int(int), _, "memory") => {
                generic.prop_raw("memory_effects", memory_effects(*int))
            }
            (LlvmAttrValue::Int(int), _, "allocsize") => {
                generic.prop_raw("allocsize", allocsize(*int))
            }
            (LlvmAttrValue::Str(str), _, _) => {
                default_func_attrs.push(format!(
                    "{} = {}",
                    mlir_string_literal(name),
                    mlir_string_literal(str)
                ));
                generic
            }
            _ => return fn_attr_err(op, ctx, "llvm.call", name),
        };
    }
    if !default_func_attrs.is_empty() {
        generic = generic.prop_raw(
            "default_func_attrs",
            format!("{{{}}}", default_func_attrs.join(", ")),
        );
    }
    Ok(generic)
}

/// Check that `op` has no LLVM function attributes in `attrs` and if it does, return an error.
fn no_fn_attrs(
    op: &impl Op,
    ctx: &Context,
    mnemonic: &str,
    attrs: Option<&LlvmAttributesAttr>,
) -> Result<()> {
    match attrs.and_then(|attrs| attrs.iter().next()) {
        Some((name, _)) => fn_attr_err(op, ctx, mnemonic, name),
        None => Ok(()),
    }
}

/// `#llvm.memory_effects<...>` for an LLVM `memory` attribute.
///
/// LLVM packs a two bit `ModRefInfo` for each location, in the order `argMem`,
/// `inaccessibleMem`, `errnoMem`, `other`, `targetMem0`, `targetMem1`.
fn memory_effects(value: u64) -> String {
    const MOD_REF: [&str; 4] = ["none", "read", "write", "readwrite"];
    let mode = |location: u32| MOD_REF[((value >> (2 * location)) & 3) as usize];
    format!(
        "#llvm.memory_effects<other = {}, argMem = {}, inaccessibleMem = {}, errnoMem = {}, \
         targetMem0 = {}, targetMem1 = {}>",
        mode(3),
        mode(0),
        mode(1),
        mode(2),
        mode(4),
        mode(5)
    )
}

/// `array<i32: ...>` for an LLVM `allocsize` attribute.
///
/// LLVM packs the element size argument in the high 32 bits, and the number
/// of elements argument (`u32::MAX` when there is none) in the low 32 bits.
fn allocsize(value: u64) -> String {
    let elem_size = (value >> 32) as u32;
    match value as u32 {
        u32::MAX => mlir_dense_array("i32", [elem_size]),
        num_elems => mlir_dense_array("i32", [elem_size, num_elems]),
    }
}

/// [alignment_attr], for the atomic accesses on which MLIR insists.
fn required_alignment(op: &impl AlignableOpInterface, ctx: &Context) -> Result<AlignmentAttr> {
    match alignment_attr(op, ctx) {
        Some(alignment) => Ok(alignment),
        None => untranslatable(
            op,
            ctx,
            "MLIR requires an alignment on an atomic memory access",
        ),
    }
}

/// Ops that carry only their operands, results and successors.
macro_rules! direct_translation_ops {
    ($($op:ty => $mnemonic:literal),* $(,)?) => { $(
        #[op_interface_impl]
        impl ToMlirOp for $op {
            fn to_mlir(
                &self,
                ctx: &Context,
                state: &State,
                f: &mut fmt::Formatter<'_>,
            ) -> Result<()> {
                GenericOp::new(self.get_operation(), $mnemonic).print(ctx, state, f)
            }
        }
    )* };
}

direct_translation_ops! {
    ReturnOp => "llvm.return",
    UnreachableOp => "llvm.unreachable",
    BrOp => "llvm.br",
    UDivOp => "llvm.udiv",
    SDivOp => "llvm.sdiv",
    URemOp => "llvm.urem",
    SRemOp => "llvm.srem",
    AndOp => "llvm.and",
    OrOp => "llvm.or",
    XorOp => "llvm.xor",
    LShrOp => "llvm.lshr",
    AShrOp => "llvm.ashr",
    BitcastOp => "llvm.bitcast",
    IntToPtrOp => "llvm.inttoptr",
    PtrToIntOp => "llvm.ptrtoint",
    AddrSpaceCastOp => "llvm.addrspacecast",
    SExtOp => "llvm.sext",
    TruncOp => "llvm.trunc",
    FPToSIOp => "llvm.fptosi",
    FPToUIOp => "llvm.fptoui",
    SIToFPOp => "llvm.sitofp",
    FreezeOp => "llvm.freeze",
    UndefOp => "llvm.mlir.undef",
    PoisonOp => "llvm.mlir.poison",
    ZeroOp => "llvm.mlir.zero",
    InsertElementOp => "llvm.insertelement",
    ExtractElementOp => "llvm.extractelement",
    VAArgOp => "llvm.va_arg",
}

/// Integer arithmetic that carries LLVM's `nsw` / `nuw` flags.
macro_rules! overflow_ops {
    ($($op:ty => $mnemonic:literal),* $(,)?) => { $(
        #[op_interface_impl]
        impl ToMlirOp for $op {
            fn to_mlir(
                &self,
                ctx: &Context,
                state: &State,
                f: &mut fmt::Formatter<'_>,
            ) -> Result<()> {
                let flags: Option<IntegerOverflowFlagsAttr> =
                    op_attr(self.get_operation(), ctx, &ATTR_KEY_INTEGER_OVERFLOW_FLAGS);
                GenericOp::new(self.get_operation(), $mnemonic)
                    .prop_attr_opt("overflowFlags", flags.as_ref())
                    .print(ctx, state, f)
            }
        }
    )* };
}

overflow_ops! {
    AddOp => "llvm.add",
    SubOp => "llvm.sub",
    MulOp => "llvm.mul",
    ShlOp => "llvm.shl",
}

/// Floating point arithmetic that carries LLVM's fast-math flags.
macro_rules! fastmath_ops {
    ($($op:ty => $mnemonic:literal),* $(,)?) => { $(
        #[op_interface_impl]
        impl ToMlirOp for $op {
            fn to_mlir(
                &self,
                ctx: &Context,
                state: &State,
                f: &mut fmt::Formatter<'_>,
            ) -> Result<()> {
                let flags: Option<FastmathFlagsAttr> =
                    op_attr(self.get_operation(), ctx, &ATTR_KEY_FAST_MATH_FLAGS);
                GenericOp::new(self.get_operation(), $mnemonic)
                    .prop_attr_opt("fastmathFlags", flags.as_ref())
                    .print(ctx, state, f)
            }
        }
    )* };
}

fastmath_ops! {
    FAddOp => "llvm.fadd",
    FSubOp => "llvm.fsub",
    FMulOp => "llvm.fmul",
    FDivOp => "llvm.fdiv",
    FRemOp => "llvm.frem",
    FNegOp => "llvm.fneg",
    FPExtOp => "llvm.fpext",
    FPTruncOp => "llvm.fptrunc",
}

/// Widening conversions that carry LLVM's `nneg` flag (MLIR's `nonNeg`).
macro_rules! nneg_ops {
    ($($op:ty => $mnemonic:literal),* $(,)?) => { $(
        #[op_interface_impl]
        impl ToMlirOp for $op {
            fn to_mlir(
                &self,
                ctx: &Context,
                state: &State,
                f: &mut fmt::Formatter<'_>,
            ) -> Result<()> {
                let nneg = UnitAttr::new();
                let set = op_attr::<BoolAttr>(self.get_operation(), ctx, &ATTR_KEY_NNEG_FLAG)
                    .is_some_and(bool::from);
                GenericOp::new(self.get_operation(), $mnemonic)
                    .prop_attr_opt("nonNeg", set.then_some(&nneg))
                    .print(ctx, state, f)
            }
        }
    )* };
}

nneg_ops! {
    ZExtOp => "llvm.zext",
    UIToFPOp => "llvm.uitofp",
}

/// `llvm.icmp`. MLIR keeps the predicate as an integer property.
#[op_interface_impl]
impl ToMlirOp for ICmpOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let pred = self
            .get_attr_llvm_icmp_predicate(ctx)
            .expect("ICmpOp is missing its predicate attribute");
        GenericOp::new(self.get_operation(), "llvm.icmp")
            .prop_attr("predicate", &*pred)
            .print(ctx, state, f)
    }
}

/// `llvm.fcmp`. MLIR keeps the predicate as an integer property.
#[op_interface_impl]
impl ToMlirOp for FCmpOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let pred = self
            .get_attr_llvm_fcmp_predicate(ctx)
            .expect("FCmpOp is missing its predicate attribute");
        let flags: Option<FastmathFlagsAttr> =
            op_attr(self.get_operation(), ctx, &ATTR_KEY_FAST_MATH_FLAGS);
        GenericOp::new(self.get_operation(), "llvm.fcmp")
            .prop_attr_opt("fastmathFlags", flags.as_ref())
            .prop_attr("predicate", &*pred)
            .print(ctx, state, f)
    }
}

/// `llvm.select`.
#[op_interface_impl]
impl ToMlirOp for SelectOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let flags = self.get_attr_llvm_select_fast_math_flags(ctx);
        GenericOp::new(self.get_operation(), "llvm.select")
            .prop_attr_opt("fastmathFlags", flags.as_deref())
            .print(ctx, state, f)
    }
}

/// `llvm.alloca`.
#[op_interface_impl]
impl ToMlirOp for AllocaOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let elem_type = self
            .get_attr_llvm_alloca_element_type(ctx)
            .expect("AllocaOp is missing its element type attribute");
        let alignment = alignment_attr(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.alloca")
            .prop_attr("elem_type", &*elem_type)
            .prop_attr_opt("alignment", alignment.as_ref())
            .print(ctx, state, f)
    }
}

/// `llvm.getelementptr`.
#[op_interface_impl]
impl ToMlirOp for GetElementPtrOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let elem_type = self
            .get_attr_llvm_gep_src_elem_type(ctx)
            .expect("GetElementPtrOp is missing its source element type attribute");
        let indices = self
            .get_attr_llvm_gep_indices(ctx)
            .expect("GetElementPtrOp is missing its indices attribute");
        // MLIR leaves `noWrapFlags` off when no flag is set.
        let no_wrap_flags = self
            .get_attr_llvm_gep_no_wrap_flags(ctx)
            .filter(|flags| !flags.0.is_empty());
        GenericOp::new(self.get_operation(), "llvm.getelementptr")
            .prop_attr("elem_type", &*elem_type)
            .prop_attr_opt("noWrapFlags", no_wrap_flags.as_deref())
            .prop_attr("rawConstantIndices", &*indices)
            .print(ctx, state, f)
    }
}

/// `llvm.load`.
#[op_interface_impl]
impl ToMlirOp for LoadOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let alignment = alignment_attr(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.load")
            .prop_attr_opt("alignment", alignment.as_ref())
            .prop_unit("volatile_", self.is_volatile(ctx))
            .print(ctx, state, f)
    }
}

/// `llvm.store`.
#[op_interface_impl]
impl ToMlirOp for StoreOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let alignment = alignment_attr(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.store")
            .prop_attr_opt("alignment", alignment.as_ref())
            .prop_unit("volatile_", self.is_volatile(ctx))
            .print(ctx, state, f)
    }
}

/// `llvm.load` with an atomic ordering. MLIR has no separate atomic load op.
#[op_interface_impl]
impl ToMlirOp for AtomicLoadOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let ordering = self
            .get_attr_llvm_ld_ordering(ctx)
            .expect("AtomicLoadOp is missing its ordering attribute");
        let syncscope = syncscope_name(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.load")
            .prop_attr("alignment", &required_alignment(self, ctx)?)
            .prop_attr("ordering", &*ordering)
            .prop_attr_opt("syncscope", syncscope.as_ref())
            .print(ctx, state, f)
    }
}

/// `llvm.store` with an atomic ordering. MLIR has no separate atomic store op.
#[op_interface_impl]
impl ToMlirOp for AtomicStoreOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let ordering = self
            .get_attr_llvm_st_ordering(ctx)
            .expect("AtomicStoreOp is missing its ordering attribute");
        let syncscope = syncscope_name(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.store")
            .prop_attr("alignment", &required_alignment(self, ctx)?)
            .prop_attr("ordering", &*ordering)
            .prop_attr_opt("syncscope", syncscope.as_ref())
            .print(ctx, state, f)
    }
}

/// `llvm.atomicrmw`.
#[op_interface_impl]
impl ToMlirOp for AtomicRmwOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let kind = self
            .get_attr_llvm_rmw_kind(ctx)
            .expect("AtomicRmwOp is missing its kind attribute");
        let ordering = self
            .get_attr_llvm_rmw_ordering(ctx)
            .expect("AtomicRmwOp is missing its ordering attribute");
        let syncscope = syncscope_name(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.atomicrmw")
            .prop_attr("bin_op", &*kind)
            .prop_attr("ordering", &*ordering)
            .prop_attr_opt("syncscope", syncscope.as_ref())
            .print(ctx, state, f)
    }
}

/// `llvm.cmpxchg`.
#[op_interface_impl]
impl ToMlirOp for AtomicCmpxchgOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let success = self
            .get_attr_llvm_cas_success_ordering(ctx)
            .expect("AtomicCmpxchgOp is missing its success ordering attribute");
        let failure = self
            .get_attr_llvm_cas_failure_ordering(ctx)
            .expect("AtomicCmpxchgOp is missing its failure ordering attribute");
        let syncscope = syncscope_name(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.cmpxchg")
            .prop_attr("failure_ordering", &*failure)
            .prop_attr("success_ordering", &*success)
            .prop_attr_opt("syncscope", syncscope.as_ref())
            .print(ctx, state, f)
    }
}

/// `llvm.fence`.
#[op_interface_impl]
impl ToMlirOp for FenceOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let ordering = self
            .get_attr_llvm_fence_ordering(ctx)
            .expect("FenceOp is missing its ordering attribute");
        let syncscope = syncscope_name(self, ctx);
        GenericOp::new(self.get_operation(), "llvm.fence")
            .prop_attr("ordering", &*ordering)
            .prop_attr_opt("syncscope", syncscope.as_ref())
            .print(ctx, state, f)
    }
}

/// `llvm.insertvalue`.
#[op_interface_impl]
impl ToMlirOp for InsertValueOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let position = self
            .get_attr_llvm_insert_value_indices(ctx)
            .expect("InsertValueOp is missing its indices attribute");
        GenericOp::new(self.get_operation(), "llvm.insertvalue")
            .prop_attr("position", &*position)
            .print(ctx, state, f)
    }
}

/// `llvm.extractvalue`.
#[op_interface_impl]
impl ToMlirOp for ExtractValueOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let position = self
            .get_attr_llvm_extract_value_indices(ctx)
            .expect("ExtractValueOp is missing its indices attribute");
        GenericOp::new(self.get_operation(), "llvm.extractvalue")
            .prop_attr("position", &*position)
            .print(ctx, state, f)
    }
}

/// `llvm.shufflevector`.
#[op_interface_impl]
impl ToMlirOp for ShuffleVectorOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let mask = self
            .get_attr_llvm_shuffle_vector_mask(ctx)
            .expect("ShuffleVectorOp is missing its mask attribute");
        GenericOp::new(self.get_operation(), "llvm.shufflevector")
            .prop_attr("mask", &*mask)
            .print(ctx, state, f)
    }
}

/// `llvm.cond_br`.
#[op_interface_impl]
impl ToMlirOp for CondBrOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        // pliron's segments -- condition, true operands, false operands -- are
        // MLIR's, in the same order.
        let segments = self.get_operand_segment_sizes(ctx);
        GenericOp::new(self.get_operation(), "llvm.cond_br")
            .prop_attr("operandSegmentSizes", &segments)
            .print(ctx, state, f)
    }
}

/// `llvm.switch`.
#[op_interface_impl]
impl ToMlirOp for SwitchOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        // pliron keeps one operand segment per successor: the condition, then
        // the default destination's operands, then each case's. MLIR lumps all
        // the case operands into one segment and sizes them separately.
        let segments = self.get_operand_segment_sizes(ctx).0;
        let (condition_and_default, cases) = segments.split_at(2.min(segments.len()));
        let case_total: u32 = cases.iter().sum();
        let operand_segments: Vec<u32> = condition_and_default
            .iter()
            .copied()
            .chain([case_total])
            .collect();

        // A switch with no cases has no `dense<>` element type to spell, and
        // MLIR leaves `case_values` off entirely.
        let case_values = self
            .get_attr_llvm_switch_case_values(ctx)
            .filter(|values| !values.0.is_empty());
        GenericOp::new(self.get_operation(), "llvm.switch")
            .prop_raw("case_operand_segments", mlir_dense_array("i32", cases))
            .prop_raw(
                "operandSegmentSizes",
                mlir_dense_array("i32", &operand_segments),
            )
            .prop_attr_opt("case_values", case_values.as_deref())
            .print(ctx, state, f)
    }
}

/// `llvm.indirectbr`.
#[op_interface_impl]
impl ToMlirOp for IndirectBrOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        // pliron's first segment is the address operand; the rest are one per
        // successor, same as MLIR's `indbr_operand_segments`.
        let segments = self.get_operand_segment_sizes(ctx).0;
        let dest_segments = segments.get(1..).unwrap_or(&[]);
        GenericOp::new(self.get_operation(), "llvm.indirectbr")
            .prop_raw(
                "indbr_operand_segments",
                mlir_dense_array("i32", dest_segments),
            )
            .print(ctx, state, f)
    }
}

/// One of:
/// - `llvm.mlir.addressof`, for the address of a symbol.
/// - `llvm.mlir.zero`, `llvm.mlir.undef` or `llvm.mlir.poison`, for LLVM's
///   non-constant-expression values.
/// - `llvm.mlir.constant`, for all other values.
#[op_interface_impl]
impl ToMlirOp for ConstantOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let value = self
            .get_attr_llvm_constant_value(ctx)
            .expect("ConstantOp is missing its value attribute");
        let value: &dyn Attribute = &**value;

        if constant_needs_lowering(ctx, value) {
            return lowering_err(self, ctx, "the value of this constant");
        }
        if let Some(addr) = value.downcast_ref::<SymbolAddrAttr>() {
            return GenericOp::new(self.get_operation(), "llvm.mlir.addressof")
                .prop_raw("global_name", format!("@{}", addr.symbol()))
                .print(ctx, state, f);
        }

        // MLIR's `llvm.mlir.constant` takes integer, float, string and elements
        // attributes only; these three get an op of their own instead.
        let dedicated_op = if value.is::<ZeroAttr>() {
            Some("llvm.mlir.zero")
        } else if value.is::<UndefAttr>() {
            Some("llvm.mlir.undef")
        } else if value.is::<PoisonAttr>() {
            Some("llvm.mlir.poison")
        } else {
            None
        };

        match dedicated_op {
            Some(mnemonic) => GenericOp::new(self.get_operation(), mnemonic).print(ctx, state, f),
            None => GenericOp::new(self.get_operation(), "llvm.mlir.constant")
                .prop_attr("value", value)
                .print(ctx, state, f),
        }
    }
}

/// `llvm.mlir.addressof`.
#[op_interface_impl]
impl ToMlirOp for AddressOfOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let global_name = self
            .get_attr_llvm_global_name(ctx)
            .expect("AddressOfOp is missing its global name attribute");
        GenericOp::new(self.get_operation(), "llvm.mlir.addressof")
            .prop_attr("global_name", &*global_name)
            .print(ctx, state, f)
    }
}

/// `llvm.blocktag`.
#[op_interface_impl]
impl ToMlirOp for BlockTagOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        GenericOp::new(self.get_operation(), "llvm.blocktag")
            .prop_raw(
                "tag",
                format!("#llvm.blocktag<id = {}>", self.get_tag_id(ctx)),
            )
            .print(ctx, state, f)
    }
}

/// `llvm.blockaddress`.
#[op_interface_impl]
impl ToMlirOp for BlockAddressOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let function = self
            .get_attr_llvm_block_address_function(ctx)
            .expect("BlockAddressOp is missing its function name attribute");
        let function: &pliron::identifier::Identifier = (*function).as_ref();
        let tag = self
            .get_attr_llvm_block_address_tag(ctx)
            .expect("BlockAddressOp is missing its tag attribute");
        let block_addr = format!(
            "#llvm.blockaddress<function = @{function}, tag = <id = {}>>",
            IntegerAttr::value(&tag).to_u64()
        );
        GenericOp::new(self.get_operation(), "llvm.blockaddress")
            .prop_raw("block_addr", block_addr)
            .print(ctx, state, f)
    }
}

/// `llvm.inline_asm`.
#[op_interface_impl]
impl ToMlirOp for InlineAsmOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let template = self
            .get_attr_llvm_inline_asm_template(ctx)
            .expect("InlineAsmOp is missing its template attribute");
        let constraints = self
            .get_attr_llvm_inline_asm_constraints(ctx)
            .expect("InlineAsmOp is missing its constraints attribute");
        let side_effects = self
            .get_attr_llvm_inline_asm_side_effects(ctx)
            .is_some_and(|side_effects| bool::from((*side_effects).clone()));
        no_fn_attrs(
            self,
            ctx,
            "llvm.inline_asm",
            self.get_attr_llvm_inline_asm_attrs(ctx).as_deref(),
        )?;
        GenericOp::new(self.get_operation(), "llvm.inline_asm")
            .results(non_void_results(self.get_operation(), ctx))
            .prop_raw("asm_string", mlir_string_literal(template.as_str()))
            .prop_raw("constraints", mlir_string_literal(constraints.as_str()))
            .prop_unit("has_side_effects", side_effects)
            .print(ctx, state, f)
    }
}

/// `llvm.call`.
#[op_interface_impl]
impl ToMlirOp for CallOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let num_operands = self.get_operation().deref(ctx).get_num_operands();
        let flags = self.get_attr_llvm_call_fastmath_flags(ctx);
        let callee = self.get_attr_llvm_call_callee(ctx);

        let mut generic = GenericOp::new(self.get_operation(), "llvm.call")
            .results(non_void_results(self.get_operation(), ctx))
            // MLIR splits a call's operands into callee operands and operand
            // bundle operands; pliron has no operand bundles.
            .prop_raw(
                "op_bundle_sizes",
                mlir_dense_array("i32", Vec::<u32>::new()),
            )
            .prop_raw(
                "operandSegmentSizes",
                mlir_dense_array("i32", [num_operands as u32, 0]),
            );

        // A direct call names its callee; an indirect one passes it as the
        // first operand, in both dialects.
        if let Some(callee) = &callee {
            generic = generic.prop_attr("callee", &**callee);
        }
        if let Some(flags) = &flags {
            generic = generic.prop_attr("fastmathFlags", &**flags);
        }

        // MLIR needs the callee's type spelled out when it is variadic, since
        // the call's own operand types don't imply it.
        let callee_ty = self.callee_type(ctx);
        let var_callee_ty = TypedHandle::<FuncType>::from_handle(callee_ty, ctx)
            .ok()
            .filter(|ty| ty.deref(ctx).is_var_arg())
            .map(|ty| TypeAttr::new(ty.into()));
        if let Some(var_callee_ty) = &var_callee_ty {
            generic = generic.prop_attr("var_callee_type", var_callee_ty);
        }

        add_call_attrs(
            generic,
            self,
            ctx,
            self.get_attr_llvm_call_attrs(ctx).as_deref(),
        )?
        .print(ctx, state, f)
    }
}

/// `llvm.call_intrinsic`.
#[op_interface_impl]
impl ToMlirOp for CallIntrinsicOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let num_operands = self.get_operation().deref(ctx).get_num_operands();
        let name = self
            .get_attr_llvm_intrinsic_name(ctx)
            .expect("CallIntrinsicOp is missing its intrinsic name attribute");
        let flags = self.get_attr_llvm_intrinsic_fastmath_flags(ctx);
        no_fn_attrs(
            self,
            ctx,
            "llvm.call_intrinsic",
            self.get_attr_llvm_intrinsic_attrs(ctx).as_deref(),
        )?;

        let mut generic = GenericOp::new(self.get_operation(), "llvm.call_intrinsic")
            .results(non_void_results(self.get_operation(), ctx))
            .prop_raw("intrin", mlir_string_literal(name.as_str()))
            .prop_raw(
                "op_bundle_sizes",
                mlir_dense_array("i32", Vec::<u32>::new()),
            )
            .prop_raw(
                "operandSegmentSizes",
                mlir_dense_array("i32", [num_operands as u32, 0]),
            );
        if let Some(flags) = &flags {
            generic = generic.prop_attr("fastmathFlags", &**flags);
        }
        generic.print(ctx, state, f)
    }
}

/// `llvm.func`.
#[op_interface_impl]
impl ToMlirOp for FuncOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let func_type = self
            .get_attr_llvm_func_type(ctx)
            .expect("FuncOp is missing its function type attribute");
        // MLIR always spells the linkage out; pliron leaves it off when external.
        let linkage = self
            .get_attr_llvm_function_linkage(ctx)
            .map_or(LinkageAttr::ExternalLinkage, |l| l.clone());
        let generic = GenericOp::new(self.get_operation(), "llvm.func")
            .prop_attr("function_type", &*func_type)
            .prop_attr("linkage", &linkage)
            .prop_raw(
                "sym_name",
                mlir_string_literal(self.get_symbol_name(ctx).as_ref()),
            )
            // MLIR's `llvm.func` always has a body region; a declaration's is
            // empty. pliron leaves the region out entirely.
            .min_regions(1);
        add_func_attrs(
            generic,
            self,
            ctx,
            self.get_attr_llvm_func_attrs(ctx).as_deref(),
        )?
        .print(ctx, state, f)
    }
}

/// `llvm.mlir.global`.
#[op_interface_impl]
impl ToMlirOp for GlobalOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let global_type = self
            .get_attr_llvm_global_type(ctx)
            .expect("GlobalOp is missing its type attribute");
        // MLIR always spells the linkage out; pliron leaves it off when external.
        let linkage = self
            .get_attr_llvm_global_linkage(ctx)
            .map_or(LinkageAttr::ExternalLinkage, |l| l.clone());
        let initializer = self.get_attr_llvm_global_initializer(ctx);
        let alignment = alignment_attr(self, ctx);

        let mut generic = GenericOp::new(self.get_operation(), "llvm.mlir.global")
            .prop_raw("addr_space", format!("{} : i32", self.address_space(ctx)))
            .prop_attr("global_type", &*global_type)
            .prop_attr("linkage", &linkage)
            .prop_raw(
                "sym_name",
                mlir_string_literal(self.get_symbol_name(ctx).as_ref()),
            )
            .prop_attr_opt("alignment", alignment.as_ref())
            .prop_unit("constant", self.is_constant(ctx))
            // MLIR's `llvm.mlir.global` always has an initializer region, empty
            // when the initializer is a plain value or absent.
            .min_regions(1);
        // The initializer is type erased (`AttrObj`), so it cannot go through
        // `prop_attr_opt`.
        if let Some(initializer) = &initializer {
            if global_initializer_needs_lowering(ctx, &***initializer) {
                return lowering_err(self, ctx, "the initializer of this global");
            }
            generic = generic.prop_attr("value", &***initializer);
        }
        generic.print(ctx, state, f)
    }
}
