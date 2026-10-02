/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Operation emission for LLVM IR.

use rustc_hash::FxHashMap;
use std::cell::Ref;
use std::fmt::Write;

use pliron::r#type::Typed;
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr, StringAttr},
        op_interfaces::{CallOpCallable, CallOpInterface},
        types::{FP32Type, FP64Type, IntegerType},
    },
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    value::Value,
};

use crate::{
    attributes::{
        AtomicOrderingAttr, AtomicRmwKindAttr, FCmpPredicateAttr, FPHalfAttr, FastmathFlags,
        FastmathFlagsAttr, GepIndexAttr, GepNoWrapFlags, ICmpPredicateAttr, SyncScopeAttr,
    },
    op_interfaces::{
        ATTR_KEY_FAST_MATH_FLAGS, PointerTypeResult, SyncScopeInterface, VolatilityOpInterface,
    },
    ops,
    types::{ArrayType, FuncType, HalfType, PointerType, VoidType},
};

use super::{
    externs::DeviceExternType,
    literals::{format_float_literal, format_half_literal, format_string_literal},
    state::ModuleExportState,
};

/// Typed view of a dispatched LLVM dialect operation.
///
/// Each variant wraps a reference to the typed op so that handler methods can
/// call op-specific accessors (predicates, ordering, asm template, etc.)
/// without repeating the downcast inside every arm.
///
/// # Maintenance contract
///
/// Wiring an LLVM dialect op (defined upstream in `pliron-llvm`) into
/// textual export touches four places in this file:
///
/// 1. Add a variant to `LlvmOp` below.
/// 2. Add a matching entry in the `classify_op!` invocation below.
/// 3. Add an `emit_*` helper method on [`ModuleExportState`].
/// 4. Add a `Some(LlvmOp::X(op)) => self.emit_x(...)` arm in `export_op`.
///
/// Selective dispatch sites elsewhere in the export pipeline (e.g. the
/// value-name pre-pass in `function.rs`) inspect specific op types via
/// `downcast_ref` directly and do not need to be updated.
enum LlvmOp<'op> {
    // Terminators
    Return(&'op ops::ReturnOp),
    /// `UnreachableOp` carries no operands or attributes, so the typed reference
    /// is intentionally unread. The variant is kept tuple-shaped for uniformity
    /// with the other terminator variants.
    #[allow(dead_code)]
    Unreachable(&'op ops::UnreachableOp),
    Br(&'op ops::BrOp),
    CondBr(&'op ops::CondBrOp),
    // Memory
    Load(&'op ops::LoadOp),
    Store(&'op ops::StoreOp),
    Alloca(&'op ops::AllocaOp),
    GetElementPtr(&'op ops::GetElementPtrOp),
    // Atomics
    AtomicLoad(&'op ops::AtomicLoadOp),
    AtomicStore(&'op ops::AtomicStoreOp),
    AtomicRmw(&'op ops::AtomicRmwOp),
    AtomicCmpxchg(&'op ops::AtomicCmpxchgOp),
    Fence(&'op ops::FenceOp),
    // Integer arithmetic
    Add(&'op ops::AddOp),
    Sub(&'op ops::SubOp),
    Mul(&'op ops::MulOp),
    SDiv(&'op ops::SDivOp),
    UDiv(&'op ops::UDivOp),
    SRem(&'op ops::SRemOp),
    URem(&'op ops::URemOp),
    Shl(&'op ops::ShlOp),
    LShr(&'op ops::LShrOp),
    AShr(&'op ops::AShrOp),
    And(&'op ops::AndOp),
    Or(&'op ops::OrOp),
    Xor(&'op ops::XorOp),
    // Float arithmetic
    FAdd(&'op ops::FAddOp),
    FSub(&'op ops::FSubOp),
    FMul(&'op ops::FMulOp),
    FDiv(&'op ops::FDivOp),
    FRem(&'op ops::FRemOp),
    FNeg(&'op ops::FNegOp),
    // Comparison / select
    ICmp(&'op ops::ICmpOp),
    FCmp(&'op ops::FCmpOp),
    Select(&'op ops::SelectOp),
    // Calls and inline assembly
    Call(&'op ops::CallOp),
    InlineAsm(&'op ops::InlineAsmOp),
    // Casts
    Bitcast(&'op ops::BitcastOp),
    AddrSpaceCast(&'op ops::AddrSpaceCastOp),
    ZExt(&'op ops::ZExtOp),
    SExt(&'op ops::SExtOp),
    Trunc(&'op ops::TruncOp),
    PtrToInt(&'op ops::PtrToIntOp),
    IntToPtr(&'op ops::IntToPtrOp),
    UIToFP(&'op ops::UIToFPOp),
    SIToFP(&'op ops::SIToFPOp),
    FPToUI(&'op ops::FPToUIOp),
    FPToSI(&'op ops::FPToSIOp),
    FPExt(&'op ops::FPExtOp),
    FPTrunc(&'op ops::FPTruncOp),
    // Aggregates
    ExtractValue(&'op ops::ExtractValueOp),
    InsertValue(&'op ops::InsertValueOp),
    // Virtual / constant ops
    Undef(&'op ops::UndefOp),
    Constant(&'op ops::ConstantOp),
    AddressOf(&'op ops::AddressOfOp),
    DebugValue(&'op ops::DebugValueOp),
    DebugValueList(&'op ops::DebugValueListOp),
}

/// Try each `(Variant, OpType)` pair in order; return the first match.
///
/// Uses `return` to short-circuit out of the enclosing `try_from` body.
macro_rules! classify_op {
    ($op_obj:expr, { $($Variant:ident => $OpTy:ty),* $(,)? }) => {{
        let op = $op_obj;
        $(
            if let Some(inner) = op.downcast_ref::<$OpTy>() {
                return Ok(Self::$Variant(inner));
            }
        )*
        Err(())
    }};
}

impl<'op> TryFrom<&'op dyn Op> for LlvmOp<'op> {
    type Error = ();

    fn try_from(op_obj: &'op dyn Op) -> Result<Self, ()> {
        classify_op!(op_obj, {
            // Terminators
            Return       => ops::ReturnOp,
            Unreachable  => ops::UnreachableOp,
            Br           => ops::BrOp,
            CondBr       => ops::CondBrOp,
            // Memory
            Load         => ops::LoadOp,
            Store        => ops::StoreOp,
            Alloca       => ops::AllocaOp,
            GetElementPtr=> ops::GetElementPtrOp,
            // Atomics
            AtomicLoad   => ops::AtomicLoadOp,
            AtomicStore  => ops::AtomicStoreOp,
            AtomicRmw    => ops::AtomicRmwOp,
            AtomicCmpxchg=> ops::AtomicCmpxchgOp,
            Fence        => ops::FenceOp,
            // Integer arithmetic
            Add          => ops::AddOp,
            Sub          => ops::SubOp,
            Mul          => ops::MulOp,
            SDiv         => ops::SDivOp,
            UDiv         => ops::UDivOp,
            SRem         => ops::SRemOp,
            URem         => ops::URemOp,
            Shl          => ops::ShlOp,
            LShr         => ops::LShrOp,
            AShr         => ops::AShrOp,
            And          => ops::AndOp,
            Or           => ops::OrOp,
            Xor          => ops::XorOp,
            // Float arithmetic
            FAdd         => ops::FAddOp,
            FSub         => ops::FSubOp,
            FMul         => ops::FMulOp,
            FDiv         => ops::FDivOp,
            FRem         => ops::FRemOp,
            FNeg         => ops::FNegOp,
            // Comparison / select
            ICmp         => ops::ICmpOp,
            FCmp         => ops::FCmpOp,
            Select       => ops::SelectOp,
            // Calls and inline assembly
            Call         => ops::CallOp,
            InlineAsm    => ops::InlineAsmOp,
            // Casts
            Bitcast      => ops::BitcastOp,
            AddrSpaceCast=> ops::AddrSpaceCastOp,
            ZExt         => ops::ZExtOp,
            SExt         => ops::SExtOp,
            Trunc        => ops::TruncOp,
            PtrToInt     => ops::PtrToIntOp,
            IntToPtr     => ops::IntToPtrOp,
            UIToFP       => ops::UIToFPOp,
            SIToFP       => ops::SIToFPOp,
            FPToUI       => ops::FPToUIOp,
            FPToSI       => ops::FPToSIOp,
            FPExt        => ops::FPExtOp,
            FPTrunc      => ops::FPTruncOp,
            // Aggregates
            ExtractValue => ops::ExtractValueOp,
            InsertValue  => ops::InsertValueOp,
            // Virtual / constant ops
            Undef        => ops::UndefOp,
            Constant     => ops::ConstantOp,
            AddressOf    => ops::AddressOfOp,
            DebugValue   => ops::DebugValueOp,
            DebugValueList => ops::DebugValueListOp,
        })
    }
}

impl LlvmOp<'_> {
    fn emits_real_instruction(&self) -> bool {
        !matches!(
            self,
            Self::Undef(_)
                | Self::Constant(_)
                | Self::AddressOf(_)
                | Self::DebugValue(_)
                | Self::DebugValueList(_)
        )
    }

    fn needs_scoped_debug_location(&self) -> bool {
        matches!(self, Self::Call(_))
    }
}

const LOCAL_MEMORY_VALUE_PREFIX: &str = "__cuda_oxide_local_x";
const LOCAL_MEMORY_ALLOCA_PREFIX: &str = "__cuda_oxide_local_alloca_x";

/// Ceiling for the pre-hex provenance payload.
///
/// Hex encoding doubles the byte count, and LLVM's textual parser mis-lexes
/// value names longer than roughly a kilobyte (empirically the failure
/// surfaces as a bogus "multiple definition of local value" parse error), so
/// the whole `%__cuda_oxide_local_x<hex>_` identifier must stay comfortably
/// below that. The importer already spells types compactly; this cap makes
/// the exporter safe against any producer, since the attribute itself places
/// no bound on field lengths.
const LOCAL_MEMORY_MAX_PAYLOAD_BYTES: usize = 256;

/// Serialize a [`crate::ops::LocalMemoryProvenanceAttr`] into the hex payload
/// carried by the alloca's SSA value name.
///
/// This is the only place the provenance leaves the first-class attribute
/// world: textual `.ll` handed to an external `opt` has no other channel that
/// reliably survives on live instructions. Tabs separate the fields, the hex
/// encoding keeps arbitrary identifiers and type spellings valid in an LLVM
/// value name, and the trailing `_` sentinel terminates the hex run so LLVM's
/// numeric name-uniquing suffixes (a second inlined copy of the same local
/// becomes `<name>1`) cannot extend or garble the payload.
fn encode_local_memory_provenance(provenance: &crate::ops::LocalMemoryProvenanceAttr) -> String {
    let mut payload = format!(
        "{}\t{}\t{}\t{}",
        provenance.local_index,
        provenance.size_bytes,
        provenance.binding_name.as_str(),
        provenance.type_name.as_str()
    );
    if payload.len() > LOCAL_MEMORY_MAX_PAYLOAD_BYTES {
        // Truncate on a char boundary so the decoded bytes stay valid UTF-8;
        // only the trailing type spelling loses detail.
        let mut cut = LOCAL_MEMORY_MAX_PAYLOAD_BYTES;
        while !payload.is_char_boundary(cut) {
            cut -= 1;
        }
        payload.truncate(cut);
    }
    let mut encoded = String::with_capacity(payload.len() * 2 + 1);
    for byte in payload.as_bytes() {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded.push('_');
    encoded
}

impl<'a> ModuleExportState<'a> {
    fn local_memory_value_name(
        &self,
        op: Ptr<Operation>,
        allocation_storage: bool,
    ) -> Option<String> {
        let provenance = crate::ops::local_memory_provenance(self.ctx, op)?;
        let prefix = if allocation_storage {
            LOCAL_MEMORY_ALLOCA_PREFIX
        } else {
            LOCAL_MEMORY_VALUE_PREFIX
        };
        Some(format!(
            "%{prefix}{}",
            encode_local_memory_provenance(&provenance)
        ))
    }

    pub(super) fn export_op(
        &mut self,
        op: Ptr<Operation>,
        value_names: &mut FxHashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &FxHashMap<Ptr<BasicBlock>, String>,
        debug_scope: Option<usize>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let op_loc = op_ref.loc();
        let op_obj = Operation::get_op_dyn(op, self.ctx);
        let llvm_op = LlvmOp::try_from(op_obj.as_ref()).ok();
        let debug_alloca = llvm_op.as_ref().and_then(|llvm_op| match llvm_op {
            LlvmOp::Alloca(op) => Some(*op),
            _ => None,
        });
        let should_attach_debug = llvm_op.as_ref().is_some_and(LlvmOp::emits_real_instruction);
        let allow_scope_debug_fallback = llvm_op
            .as_ref()
            .is_some_and(LlvmOp::needs_scoped_debug_location);
        let output_before = output.len();

        // Rust-local allocas carry an encoded provenance name into textual LLVM IR.
        // `opt -O2` preserves SSA names on instructions that survive, so the
        // post-optimization local-memory diagnostic can attribute a remaining
        // alloca without relying on non-semantic metadata preservation.
        if let Some(LlvmOp::Alloca(alloca)) = llvm_op.as_ref()
            && let Some(name) = self.local_memory_value_name(alloca.get_operation(), false)
        {
            value_names.insert(op_ref.get_result(0), name);
        }

        // Register result names (skip if already named in pre-pass)
        for res in op_ref.results() {
            value_names.entry(res).or_insert_with(|| {
                let name = format!("%v{next_value_id}");
                *next_value_id += 1;
                name.clone()
            });
        }

        match llvm_op {
            // Terminators
            Some(LlvmOp::Return(op)) => self.emit_return(op, value_names, output)?,
            Some(LlvmOp::Unreachable(_)) => writeln!(output, "  unreachable").unwrap(),
            Some(LlvmOp::Br(op)) => self.emit_br(op, block_labels, output)?,
            Some(LlvmOp::CondBr(op)) => self.emit_cond_br(op, value_names, block_labels, output)?,
            // Memory
            Some(LlvmOp::Load(op)) => self.emit_load(op, value_names, next_value_id, output)?,
            Some(LlvmOp::Store(op)) => self.emit_store(op, value_names, next_value_id, output)?,
            Some(LlvmOp::Alloca(op)) => self.emit_alloca(op, value_names, next_value_id, output)?,
            Some(LlvmOp::GetElementPtr(op)) => {
                self.emit_gep(op, value_names, next_value_id, output)?
            }
            // Atomics
            Some(LlvmOp::AtomicLoad(op)) => {
                self.emit_atomic_load(op, value_names, next_value_id, output)?
            }
            Some(LlvmOp::AtomicStore(op)) => {
                self.emit_atomic_store(op, value_names, next_value_id, output)?
            }
            Some(LlvmOp::AtomicRmw(op)) => {
                self.emit_atomic_rmw(op, value_names, next_value_id, output)?
            }
            Some(LlvmOp::AtomicCmpxchg(op)) => {
                self.emit_atomic_cmpxchg(op, value_names, next_value_id, output)?
            }
            Some(LlvmOp::Fence(op)) => self.emit_fence(op, output)?,
            // Integer arithmetic (all map to export_binop)
            Some(LlvmOp::Add(op)) => {
                self.export_binop("add", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Sub(op)) => {
                self.export_binop("sub", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Mul(op)) => {
                self.export_binop("mul", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SDiv(op)) => {
                self.export_binop("sdiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::UDiv(op)) => {
                self.export_binop("udiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SRem(op)) => {
                self.export_binop("srem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::URem(op)) => {
                self.export_binop("urem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Shl(op)) => {
                self.export_binop("shl", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::LShr(op)) => {
                self.export_binop("lshr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::AShr(op)) => {
                self.export_binop("ashr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::And(op)) => {
                self.export_binop("and", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Or(op)) => {
                self.export_binop("or", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Xor(op)) => {
                self.export_binop("xor", op.get_operation(), value_names, output)?
            }
            // Float arithmetic
            Some(LlvmOp::FAdd(op)) => {
                self.export_binop("fadd", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FSub(op)) => {
                self.export_binop("fsub", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FMul(op)) => {
                self.export_binop("fmul", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FDiv(op)) => {
                self.export_binop("fdiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FRem(op)) => {
                self.export_binop("frem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FNeg(op)) => self.emit_fneg(op, value_names, output)?,
            // Comparison / select
            Some(LlvmOp::ICmp(op)) => self.emit_icmp(op, value_names, output)?,
            Some(LlvmOp::FCmp(op)) => self.emit_fcmp(op, value_names, output)?,
            Some(LlvmOp::Select(op)) => self.emit_select(op, value_names, output)?,
            // Calls and inline assembly
            Some(LlvmOp::Call(op)) => self.emit_call(op, value_names, next_value_id, output)?,
            Some(LlvmOp::InlineAsm(op)) => self.emit_inline_asm(op, value_names, output)?,
            // Casts
            Some(LlvmOp::Bitcast(op)) => self.emit_bitcast(op, value_names, output)?,
            Some(LlvmOp::AddrSpaceCast(op)) => self.emit_addrspacecast(op, value_names, output)?,
            Some(LlvmOp::ZExt(op)) => self.emit_zext(op, value_names, output)?,
            Some(LlvmOp::SExt(op)) => {
                self.export_cast("sext", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Trunc(op)) => {
                self.export_cast("trunc", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::PtrToInt(op)) => {
                self.export_cast("ptrtoint", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::IntToPtr(op)) => {
                self.export_cast("inttoptr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::UIToFP(op)) => {
                self.export_cast("uitofp", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SIToFP(op)) => {
                self.export_cast("sitofp", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPToUI(op)) => {
                self.export_cast("fptoui", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPToSI(op)) => {
                self.export_cast("fptosi", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPExt(op)) => {
                self.export_cast("fpext", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPTrunc(op)) => {
                self.export_cast("fptrunc", op.get_operation(), value_names, output)?
            }
            // Aggregates
            Some(LlvmOp::ExtractValue(op)) => self.emit_extract_value(op, value_names, output)?,
            Some(LlvmOp::InsertValue(op)) => self.emit_insert_value(op, value_names, output)?,
            // Virtual ops
            Some(LlvmOp::Undef(op)) => self.emit_undef(op, value_names),
            Some(LlvmOp::Constant(op)) => self.emit_constant(op, value_names)?,
            Some(LlvmOp::AddressOf(op)) => self.emit_address_of(op, value_names, output)?,
            Some(LlvmOp::DebugValue(op)) => {
                self.emit_debug_value(op, value_names, debug_scope, &op_loc, output)?
            }
            Some(LlvmOp::DebugValueList(op)) => {
                self.emit_debug_value_list(op, value_names, debug_scope, &op_loc, output)?
            }
            // Unknown
            None if self.legacy_typed_pointers() => {
                return Err(format!(
                    "legacy LLVM 7 export does not support operation `{}`",
                    Operation::get_opid(op, self.ctx)
                ));
            }
            None => writeln!(
                output,
                "  ; Unknown op: {}",
                Operation::get_opid(op, self.ctx)
            )
            .unwrap(),
        }

        if should_attach_debug {
            self.attach_debug_to_last_line(
                output,
                output_before,
                debug_scope,
                &op_loc,
                allow_scope_debug_fallback,
            );
        }
        if let Some(alloca) = debug_alloca {
            self.emit_debug_declare_for_alloca(alloca, value_names, debug_scope, &op_loc, output)?;
        }

        Ok(())
    }

    fn fresh_value_name(next_value_id: &mut usize) -> String {
        let name = format!("%v{}", *next_value_id);
        *next_value_id += 1;
        name
    }

    fn value_name<'b>(
        &self,
        value: Value,
        value_names: &'b FxHashMap<Value, String>,
    ) -> Result<&'b str, String> {
        value_names
            .get(&value)
            .map(String::as_str)
            .ok_or_else(|| format!("missing exported name for value {value:?}"))
    }

    /// Check the scalar type and pointer address space against a device-extern
    /// declaration. Pointer pointees are stored separately in `DeviceExternType`.
    fn validate_device_extern_value_type(
        &self,
        expected: &DeviceExternType,
        actual: pliron::r#type::TypeHandle,
        position: &str,
    ) -> Result<(), String> {
        let actual_ref = actual.deref(self.ctx);
        let matches = match expected {
            DeviceExternType::Void => actual_ref.is::<VoidType>(),
            DeviceExternType::Integer(bits)
            | DeviceExternType::SignExtInteger(bits)
            | DeviceExternType::ZeroExtInteger(bits) => actual_ref
                .downcast_ref::<IntegerType>()
                .is_some_and(|ty| ty.width() == *bits),
            DeviceExternType::Float16 => actual_ref.is::<HalfType>(),
            DeviceExternType::Float32 => actual_ref.is::<FP32Type>(),
            DeviceExternType::Float64 => actual_ref.is::<FP64Type>(),
            DeviceExternType::Pointer { address_space, .. } => actual_ref
                .downcast_ref::<PointerType>()
                .is_some_and(|ty| ty.address_space() == *address_space),
            DeviceExternType::Array { element, len } => {
                actual_ref.downcast_ref::<ArrayType>().is_some_and(|ty| {
                    ty.size() == *len
                        && self
                            .validate_device_extern_value_type(element, ty.elem_type(), position)
                            .is_ok()
                })
            }
        };
        if matches {
            return Ok(());
        }

        let expected = expected.llvm_string(self.legacy_typed_pointers())?;
        Err(format!(
            "device-extern {position} type mismatch: declaration expects `{expected}`, lowered call has `{}`",
            actual_ref.disp(self.ctx)
        ))
    }

    /// Convert an internal byte pointer to the type required by a legacy
    /// device-extern declaration, such as `float*`.
    fn device_extern_argument(
        &self,
        arg: Value,
        expected: &DeviceExternType,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
        position: &str,
    ) -> Result<String, String> {
        self.validate_device_extern_value_type(expected, arg.get_type(self.ctx), position)?;
        let source_name = self.value_name(arg, value_names)?.to_string();
        if !self.legacy_typed_pointers()
            || expected.pointer_parts().is_none()
            || expected.is_canonical_byte_pointer()
        {
            return Ok(source_name);
        }

        let (_, address_space) = expected.pointer_parts().unwrap();
        let adapted = Self::fresh_value_name(next_value_id);
        write!(output, "  {adapted} = bitcast ").unwrap();
        self.export_canonical_pointer_type(address_space, output);
        write!(output, " {source_name} to ").unwrap();
        expected.write_llvm(output, true)?;
        writeln!(output).unwrap();
        Ok(adapted)
    }

    /// In legacy mode, convert a byte pointer to the pointee type required by a
    /// memory operation. Modern opaque-pointer mode returns the original value.
    fn typed_pointer_operand(
        &self,
        pointer: Value,
        pointee: pliron::r#type::TypeHandle,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<String, String> {
        let source_name = self.value_name(pointer, value_names)?.to_string();
        if !self.legacy_typed_pointers() {
            return Ok(source_name);
        }

        let pointer_ty = pointer.get_type(self.ctx);
        let pointer_ref = pointer_ty.deref(self.ctx);
        let pointer_ty = pointer_ref.downcast_ref::<PointerType>().ok_or_else(|| {
            format!(
                "expected pointer operand, got `{}`",
                pointer_ref.disp(self.ctx)
            )
        })?;
        let addrspace = pointer_ty.address_space();
        if self.is_i8_type(pointee) {
            return Ok(source_name);
        }
        let typed_name = Self::fresh_value_name(next_value_id);

        write!(output, "  {typed_name} = bitcast ").unwrap();
        self.export_canonical_pointer_type(addrspace, output);
        write!(output, " {source_name} to ").unwrap();
        self.export_pointer_to(pointee, addrspace, output)?;
        writeln!(output).unwrap();
        Ok(typed_name)
    }

    fn emit_return(
        &mut self,
        op: &ops::ReturnOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        write!(output, "  ret ").unwrap();
        if op_ref.operands().count() == 0 {
            write!(output, "void").unwrap();
        } else {
            let val = op_ref.operands().next().unwrap();
            self.export_type(val.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(val, value_names, output)?;
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_br(
        &self,
        op: &ops::BrOp,
        block_labels: &FxHashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let dest = op_ref.successors().next().unwrap();
        let label = block_labels.get(&dest).ok_or("Missing block label")?;
        writeln!(output, "  br label %{label}").unwrap();
        Ok(())
    }

    fn emit_cond_br(
        &mut self,
        op: &ops::CondBrOp,
        value_names: &FxHashMap<Value, String>,
        block_labels: &FxHashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let mut succs = op_ref.successors();
        let true_dest = succs.next().unwrap();
        let false_dest = succs.next().unwrap();
        let true_label = block_labels.get(&true_dest).ok_or("Missing true label")?;
        let false_label = block_labels.get(&false_dest).ok_or("Missing false label")?;
        let cond = op_ref.get_operand(0);

        write!(output, "  br i1 ").unwrap();
        self.export_value(cond, value_names, output)?;
        writeln!(output, ", label %{true_label}, label %{false_label}").unwrap();
        Ok(())
    }

    fn emit_load(
        &mut self,
        op: &ops::LoadOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();
        let ty = res.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);
        let volatile_kw = if op.is_volatile(self.ctx) {
            "volatile "
        } else {
            ""
        };

        let pointer_name =
            self.typed_pointer_operand(ptr, ty, value_names, next_value_id, output)?;

        write!(output, "  {res_name} = load {volatile_kw}").unwrap();
        self.export_type(ty, output)?;
        write!(output, ", ").unwrap();
        if self.legacy_typed_pointers() {
            self.export_pointer_to(ty, addrspace, output)?;
            write!(output, " {pointer_name}").unwrap();
        } else {
            write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
            self.export_value(ptr, value_names, output)?;
        }
        if let Some(align) = crate::ops::op_alignment(self.ctx, op.get_operation())
            .or_else(|| self.natural_alignment(ty))
        {
            writeln!(output, ", align {align}").unwrap();
        } else {
            writeln!(output).unwrap();
        }
        Ok(())
    }

    fn emit_store(
        &mut self,
        op: &ops::StoreOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);
        let val_ty = val.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);
        let volatile_kw = if op.is_volatile(self.ctx) {
            "volatile "
        } else {
            ""
        };

        let pointer_name =
            self.typed_pointer_operand(ptr, val_ty, value_names, next_value_id, output)?;

        write!(output, "  store {volatile_kw}").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, ", ").unwrap();
        if self.legacy_typed_pointers() {
            self.export_pointer_to(val_ty, addrspace, output)?;
            write!(output, " {pointer_name}").unwrap();
        } else {
            write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
            self.export_value(ptr, value_names, output)?;
        }
        if let Some(align) = crate::ops::op_alignment(self.ctx, op.get_operation())
            .or_else(|| self.natural_alignment(val_ty))
        {
            writeln!(output, ", align {align}").unwrap();
        } else {
            writeln!(output).unwrap();
        }
        Ok(())
    }

    fn emit_alloca(
        &mut self,
        op: &ops::AllocaOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let elem_ty = op
            .get_attr_llvm_alloca_element_type(self.ctx)
            .expect("Missing alloca_element_type");

        let elem_llvm_ty = elem_ty.get_type(self.ctx);
        let result_type = res.get_type(self.ctx);
        let result_ref = result_type.deref(self.ctx);
        let result_pointer = result_ref.downcast_ref::<PointerType>().ok_or_else(|| {
            format!(
                "alloca result is not a pointer: `{}`",
                result_ref.disp(self.ctx)
            )
        })?;
        if result_pointer.address_space() != 0 {
            return Err(format!(
                "alloca result uses address space {}, but LLVM alloca produces a default-address-space (0) pointer",
                result_pointer.address_space()
            ));
        }
        let needs_normalization = self.legacy_typed_pointers() && !self.is_i8_type(elem_llvm_ty);
        let alloca_name = if needs_normalization {
            self.local_memory_value_name(op.get_operation(), true)
                .unwrap_or_else(|| Self::fresh_value_name(next_value_id))
        } else {
            res_name.clone()
        };

        write!(output, "  {alloca_name} = alloca ").unwrap();
        self.export_type(elem_llvm_ty, output)?;
        let array_size = op_ref.get_operand(0);
        let array_size_name = self.value_name(array_size, value_names)?;
        if array_size_name != "1" {
            write!(output, ", ").unwrap();
            self.export_type(array_size.get_type(self.ctx), output)?;
            write!(output, " {array_size_name}").unwrap();
        }
        if let Some(align) = crate::ops::op_alignment(self.ctx, op.get_operation())
            .or_else(|| self.natural_alignment(elem_llvm_ty))
        {
            writeln!(output, ", align {align}").unwrap();
        } else {
            writeln!(output).unwrap();
        }

        if needs_normalization {
            write!(output, "  {res_name} = bitcast ").unwrap();
            self.export_pointer_to(elem_llvm_ty, 0, output)?;
            write!(output, " {alloca_name} to ").unwrap();
            self.export_canonical_pointer_type(0, output);
            writeln!(output).unwrap();
        }
        Ok(())
    }

    fn debug_expression_for_projection(dereference_base: bool, offset_bytes: u64) -> String {
        match (dereference_base, offset_bytes) {
            (false, 0) => "!DIExpression()".to_string(),
            (false, offset) => format!("!DIExpression(DW_OP_plus_uconst, {offset})"),
            (true, 0) => "!DIExpression(DW_OP_deref)".to_string(),
            (true, offset) => {
                format!("!DIExpression(DW_OP_deref, DW_OP_plus_uconst, {offset})")
            }
        }
    }

    fn debug_expression_for_fragment(offset_bits: u64, size_bits: u64) -> String {
        format!("!DIExpression(DW_OP_LLVM_fragment, {offset_bits}, {size_bits})")
    }

    fn debug_expression_for_value_list(
        expression: &ops::DebugValueExpression,
        operand_count: usize,
    ) -> Result<String, String> {
        if expression.operations.is_empty() {
            return Err("multi-value debug expression must not be empty".to_string());
        }

        let mut parts = Vec::with_capacity(expression.operations.len() * 2);
        let mut saw_arg = false;
        for operation in &expression.operations {
            match operation {
                ops::DebugValueExpressionOp::Arg(index) => {
                    let index = *index as usize;
                    if index >= operand_count {
                        return Err(format!(
                            "multi-value debug expression references argument {index}, but only {operand_count} operands exist"
                        ));
                    }
                    saw_arg = true;
                    parts.push("DW_OP_LLVM_arg".to_string());
                    parts.push(index.to_string());
                }
                ops::DebugValueExpressionOp::ConstU(value) => {
                    parts.push("DW_OP_constu".to_string());
                    parts.push(value.to_string());
                }
                ops::DebugValueExpressionOp::Plus => parts.push("DW_OP_plus".to_string()),
                ops::DebugValueExpressionOp::PlusUConst(value) => {
                    parts.push("DW_OP_plus_uconst".to_string());
                    parts.push(value.to_string());
                }
                ops::DebugValueExpressionOp::Mul => parts.push("DW_OP_mul".to_string()),
                ops::DebugValueExpressionOp::Deref => parts.push("DW_OP_deref".to_string()),
                ops::DebugValueExpressionOp::StackValue => {
                    parts.push("DW_OP_stack_value".to_string())
                }
            }
        }
        if !saw_arg {
            return Err(
                "multi-value debug expression must reference at least one DIArgList operand"
                    .to_string(),
            );
        }

        Ok(format!("!DIExpression({})", parts.join(", ")))
    }

    fn emit_debug_declare_for_alloca(
        &mut self,
        op: &ops::AllocaOp,
        value_names: &FxHashMap<Value, String>,
        debug_scope: Option<usize>,
        loc: &pliron::location::Location,
        output: &mut String,
    ) -> Result<(), String> {
        if !self.debug_kind.variables_enabled() {
            return Ok(());
        }

        let Some(scope) = debug_scope else {
            return Ok(());
        };
        let alloca_result = op.get_operation().deref(self.ctx).get_result(0);
        let alloca_name = value_names
            .get(&alloca_result)
            .ok_or_else(|| "Missing alloca result name for debug declare".to_string())?;

        let mut emitted = false;
        if let Some(info) = crate::ops::debug_local_variable(self.ctx, op.get_operation())
            && let Some((var_id, loc_id)) =
                self.debug_local_variable_for_scope(scope, loc, op.get_operation(), &info)
        {
            writeln!(
                output,
                "  call void @llvm.dbg.declare(metadata ptr {alloca_name}, metadata !{var_id}, metadata !DIExpression()), !dbg !{loc_id}"
            )
            .unwrap();
            emitted = true;
        }

        for alias in crate::ops::debug_whole_variable_aliases(self.ctx, op.get_operation()) {
            let Some((var_id, loc_id)) = self.debug_whole_variable_for_scope(scope, loc, &alias)
            else {
                continue;
            };
            writeln!(
                output,
                "  call void @llvm.dbg.declare(metadata ptr {alloca_name}, metadata !{var_id}, metadata !DIExpression()), !dbg !{loc_id}"
            )
            .unwrap();
            emitted = true;
        }

        for projected in crate::ops::debug_projected_variables(self.ctx, op.get_operation()) {
            let Some((var_id, loc_id)) =
                self.debug_projected_variable_for_scope(scope, loc, &projected)
            else {
                continue;
            };
            let expression = Self::debug_expression_for_projection(
                projected.dereference_base,
                projected.offset_bytes,
            );
            writeln!(
                output,
                "  call void @llvm.dbg.declare(metadata ptr {alloca_name}, metadata !{var_id}, metadata {expression}), !dbg !{loc_id}"
            )
            .unwrap();
            emitted = true;
        }

        for fragment in crate::ops::debug_fragment_variables(self.ctx, op.get_operation()) {
            let projected = crate::ops::DebugProjectedVariableInfo {
                variable: fragment.variable,
                dereference_base: false,
                offset_bytes: 0,
                source_scope: fragment.source_scope,
                declaration: fragment.declaration,
            };
            let Some((var_id, loc_id)) =
                self.debug_projected_variable_for_scope(scope, loc, &projected)
            else {
                continue;
            };
            let expression = Self::debug_expression_for_fragment(
                fragment.fragment.offset_bits,
                fragment.fragment.size_bits,
            );
            writeln!(
                output,
                "  call void @llvm.dbg.declare(metadata ptr {alloca_name}, metadata !{var_id}, metadata {expression}), !dbg !{loc_id}"
            )
            .unwrap();
            emitted = true;
        }

        if emitted {
            self.debug_declare_used = true;
        }
        Ok(())
    }

    fn emit_debug_value(
        &mut self,
        op: &ops::DebugValueOp,
        value_names: &FxHashMap<Value, String>,
        debug_scope: Option<usize>,
        loc: &pliron::location::Location,
        output: &mut String,
    ) -> Result<(), String> {
        if !self.debug_kind.variables_enabled() {
            return Ok(());
        }

        let Some(scope) = debug_scope else {
            return Ok(());
        };
        let value = op.value(self.ctx);
        let mut emitted = false;

        if let Some(info) = crate::ops::debug_local_variable(self.ctx, op.get_operation())
            && let Some((var_id, loc_id)) =
                self.debug_local_variable_for_scope(scope, loc, op.get_operation(), &info)
        {
            write!(output, "  call void @llvm.dbg.value(metadata ").unwrap();
            self.export_type(value.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(value, value_names, output)?;
            writeln!(
                output,
                ", metadata !{var_id}, metadata !DIExpression()), !dbg !{loc_id}"
            )
            .unwrap();
            emitted = true;
        }

        for alias in crate::ops::debug_whole_variable_aliases(self.ctx, op.get_operation()) {
            let Some((var_id, loc_id)) = self.debug_whole_variable_for_scope(scope, loc, &alias)
            else {
                continue;
            };
            write!(output, "  call void @llvm.dbg.value(metadata ").unwrap();
            self.export_type(value.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(value, value_names, output)?;
            writeln!(
                output,
                ", metadata !{var_id}, metadata !DIExpression()), !dbg !{loc_id}"
            )
            .unwrap();
            emitted = true;
        }

        for fragment in crate::ops::debug_fragment_variables(self.ctx, op.get_operation()) {
            let projected = crate::ops::DebugProjectedVariableInfo {
                variable: fragment.variable,
                dereference_base: false,
                offset_bytes: 0,
                source_scope: fragment.source_scope,
                declaration: fragment.declaration,
            };
            let Some((var_id, loc_id)) =
                self.debug_projected_variable_for_scope(scope, loc, &projected)
            else {
                continue;
            };
            let expression = Self::debug_expression_for_fragment(
                fragment.fragment.offset_bits,
                fragment.fragment.size_bits,
            );
            write!(output, "  call void @llvm.dbg.value(metadata ").unwrap();
            self.export_type(value.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(value, value_names, output)?;
            writeln!(
                output,
                ", metadata !{var_id}, metadata {expression}), !dbg !{loc_id}"
            )
            .unwrap();
            emitted = true;
        }

        if emitted {
            self.debug_value_used = true;
        }
        Ok(())
    }

    fn emit_debug_value_list(
        &mut self,
        op: &ops::DebugValueListOp,
        value_names: &FxHashMap<Value, String>,
        debug_scope: Option<usize>,
        loc: &pliron::location::Location,
        output: &mut String,
    ) -> Result<(), String> {
        if !self.debug_kind.variables_enabled() {
            return Ok(());
        }

        let Some(scope) = debug_scope else {
            return Ok(());
        };
        let Some(info) = crate::ops::debug_local_variable(self.ctx, op.get_operation()) else {
            return Ok(());
        };
        let Some((var_id, loc_id)) =
            self.debug_local_variable_for_scope(scope, loc, op.get_operation(), &info)
        else {
            return Ok(());
        };

        let values = op.values(self.ctx);
        if values.len() < 2 {
            return Err("llvm.dbg_value_list requires at least two operands".to_string());
        }
        let expression = crate::ops::debug_value_expression(self.ctx, op.get_operation())
            .ok_or_else(|| "llvm.dbg_value_list is missing its debug expression".to_string())?;
        let expression = Self::debug_expression_for_value_list(&expression, values.len())?;

        write!(output, "  call void @llvm.dbg.value(metadata !DIArgList(").unwrap();
        for (index, value) in values.into_iter().enumerate() {
            if index != 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(value.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(value, value_names, output)?;
        }
        writeln!(
            output,
            "), metadata !{var_id}, metadata {expression}), !dbg !{loc_id}"
        )
        .unwrap();
        self.debug_value_used = true;

        Ok(())
    }

    fn emit_gep(
        &mut self,
        op: &ops::GetElementPtrOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let ptr = op_ref.get_operand(0);
        let elem_ty = op
            .get_attr_llvm_gep_src_elem_type(self.ctx)
            .expect("Missing gep_src_elem_type")
            .get_type(self.ctx);
        let base_type = ptr.get_type(self.ctx);
        let base_ref = base_type.deref(self.ctx);
        let base_pointer = base_ref
            .downcast_ref::<PointerType>()
            .ok_or_else(|| format!("GEP base is not a pointer: `{}`", base_ref.disp(self.ctx)))?;
        let addrspace = base_pointer.address_space();
        let result_type = res.get_type(self.ctx);
        let result_ref = result_type.deref(self.ctx);
        let result_pointer = result_ref.downcast_ref::<PointerType>().ok_or_else(|| {
            format!(
                "GEP result is not a pointer: `{}`",
                result_ref.disp(self.ctx)
            )
        })?;
        if result_pointer.address_space() != addrspace {
            return Err(format!(
                "GEP result address-space mismatch: base is {addrspace}, result is {}",
                result_pointer.address_space()
            ));
        }

        let result_pointee = op.result_pointee_type(self.ctx);
        let pointer_name =
            self.typed_pointer_operand(ptr, elem_ty, value_names, next_value_id, output)?;
        let needs_normalization = self.legacy_typed_pointers() && !self.is_i8_type(result_pointee);
        let gep_name = if needs_normalization {
            Self::fresh_value_name(next_value_id)
        } else {
            res_name.clone()
        };

        write!(output, "  {gep_name} = getelementptr").unwrap();
        let no_wrap_flags = op.no_wrap_flags(self.ctx);
        if no_wrap_flags.contains(GepNoWrapFlags::INBOUNDS) {
            write!(output, " inbounds").unwrap();
        } else if no_wrap_flags.contains(GepNoWrapFlags::NUSW) {
            write!(output, " nusw").unwrap();
        }
        if no_wrap_flags.contains(GepNoWrapFlags::NUW) {
            write!(output, " nuw").unwrap();
        }
        write!(output, " ").unwrap();
        self.export_type(elem_ty, output)?;
        write!(output, ", ").unwrap();
        if self.legacy_typed_pointers() {
            self.export_pointer_to(elem_ty, addrspace, output)?;
            write!(output, " {pointer_name}").unwrap();
        } else {
            write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
            self.export_value(ptr, value_names, output)?;
        }

        for idx_attr in &op.get_attr_llvm_gep_indices(self.ctx).unwrap().0 {
            write!(output, ", ").unwrap();
            match idx_attr {
                GepIndexAttr::Constant(val) => {
                    write!(output, "i32 {val}").unwrap();
                }
                GepIndexAttr::OperandIdx(operand_idx) => {
                    let val = op_ref.get_operand(*operand_idx);
                    self.export_type(val.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(val, value_names, output)?;
                }
            }
        }
        writeln!(output).unwrap();

        if needs_normalization {
            write!(output, "  {res_name} = bitcast ").unwrap();
            self.export_pointer_to(result_pointee, addrspace, output)?;
            write!(output, " {gep_name} to ").unwrap();
            self.export_canonical_pointer_type(addrspace, output);
            writeln!(output).unwrap();
        }
        Ok(())
    }

    fn emit_atomic_load(
        &mut self,
        op: &ops::AtomicLoadOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();
        let ty = res.get_type(self.ctx);
        let syncscope = fmt_syncscope(op.syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_ld_ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);
        let pointer_name =
            self.typed_pointer_operand(ptr, ty, value_names, next_value_id, output)?;

        write!(output, "  {res_name} = load atomic ").unwrap();
        self.export_type(ty, output)?;
        write!(output, ", ").unwrap();
        if self.legacy_typed_pointers() {
            self.export_pointer_to(ty, addrspace, output)?;
            write!(output, " {pointer_name}").unwrap();
        } else {
            write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
            self.export_value(ptr, value_names, output)?;
        }
        // Atomic loads must state an alignment; the operand types are
        // power-of-two scalars, so this is always answerable. Error rather
        // than fabricate if it ever is not.
        let align = self.natural_alignment(ty).ok_or_else(|| {
            format!(
                "atomic load requires an explicit alignment, but `{}` has no known ABI alignment",
                ty.deref(self.ctx).disp(self.ctx)
            )
        })?;
        writeln!(output, "{syncscope} {ordering}, align {align}").unwrap();
        Ok(())
    }

    fn emit_atomic_store(
        &mut self,
        op: &ops::AtomicStoreOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);
        let syncscope = fmt_syncscope(op.syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_st_ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);
        let val_ty = val.get_type(self.ctx);
        let pointer_name =
            self.typed_pointer_operand(ptr, val_ty, value_names, next_value_id, output)?;

        write!(output, "  store atomic ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, ", ").unwrap();
        if self.legacy_typed_pointers() {
            self.export_pointer_to(val_ty, addrspace, output)?;
            write!(output, " {pointer_name}").unwrap();
        } else {
            write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
            self.export_value(ptr, value_names, output)?;
        }
        // Same contract as atomic load: exact alignment or a loud error.
        let align = self.natural_alignment(val_ty).ok_or_else(|| {
            format!(
                "atomic store requires an explicit alignment, but `{}` has no known ABI alignment",
                val_ty.deref(self.ctx).disp(self.ctx)
            )
        })?;
        writeln!(output, "{syncscope} {ordering}, align {align}").unwrap();
        Ok(())
    }

    fn emit_atomic_rmw(
        &mut self,
        op: &ops::AtomicRmwOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();
        let rmw_kind = fmt_rmw_kind(op.get_attr_llvm_rmw_kind(self.ctx));
        let syncscope = fmt_syncscope(op.syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_rmw_ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);
        let val_ty = val.get_type(self.ctx);
        let pointer_name =
            self.typed_pointer_operand(ptr, val_ty, value_names, next_value_id, output)?;

        write!(output, "  {res_name} = atomicrmw {rmw_kind} ").unwrap();
        if self.legacy_typed_pointers() {
            self.export_pointer_to(val_ty, addrspace, output)?;
            write!(output, " {pointer_name}").unwrap();
        } else {
            write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
            self.export_value(ptr, value_names, output)?;
        }
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        writeln!(output, "{syncscope} {ordering}").unwrap();
        Ok(())
    }

    fn emit_atomic_cmpxchg(
        &mut self,
        op: &ops::AtomicCmpxchgOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let cmp = op_ref.get_operand(1);
        let new_val = op_ref.get_operand(2);
        let res_name = value_names.get(&res).unwrap();
        let success_ord = fmt_ordering(op.get_attr_llvm_cas_success_ordering(self.ctx));
        let failure_ord = fmt_ordering(op.get_attr_llvm_cas_failure_ordering(self.ctx));
        let syncscope = fmt_syncscope(op.syncscope(self.ctx));
        let val_ty = cmp.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);
        let pointer_name =
            self.typed_pointer_operand(ptr, val_ty, value_names, next_value_id, output)?;

        // pliron-llvm's cmpxchg result is the full `{ T, i1 }` struct; a
        // separate `extractvalue` op (emitted on its own) pulls out the loaded
        // value, so here we emit only the cmpxchg into the struct-typed result.
        write!(output, "  {res_name} = cmpxchg ").unwrap();
        if self.legacy_typed_pointers() {
            self.export_pointer_to(val_ty, addrspace, output)?;
            write!(output, " {pointer_name}").unwrap();
        } else {
            write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
            self.export_value(ptr, value_names, output)?;
        }
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(cmp, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(new_val, value_names, output)?;
        writeln!(output, "{syncscope} {success_ord} {failure_ord}").unwrap();
        Ok(())
    }

    fn emit_fence(&self, op: &ops::FenceOp, output: &mut String) -> Result<(), String> {
        let syncscope = fmt_syncscope(op.syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_fence_ordering(self.ctx));
        writeln!(output, "  fence{syncscope} {ordering}").unwrap();
        Ok(())
    }

    fn emit_fneg(
        &mut self,
        op: &ops::FNegOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let arg = op_ref.get_operand(0);

        write!(output, "  {res_name} = fneg ").unwrap();
        self.export_type(arg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(arg, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_icmp(
        &mut self,
        op: &ops::ICmpOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred = match op.predicate(self.ctx) {
            ICmpPredicateAttr::EQ => "eq",
            ICmpPredicateAttr::NE => "ne",
            ICmpPredicateAttr::SLT => "slt",
            ICmpPredicateAttr::SLE => "sle",
            ICmpPredicateAttr::SGT => "sgt",
            ICmpPredicateAttr::SGE => "sge",
            ICmpPredicateAttr::ULT => "ult",
            ICmpPredicateAttr::ULE => "ule",
            ICmpPredicateAttr::UGT => "ugt",
            ICmpPredicateAttr::UGE => "uge",
        };

        write!(output, "  {res_name} = icmp {pred} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_fcmp(
        &mut self,
        op: &ops::FCmpOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred = match op.predicate(self.ctx) {
            FCmpPredicateAttr::False => "false",
            FCmpPredicateAttr::OEQ => "oeq",
            FCmpPredicateAttr::OGT => "ogt",
            FCmpPredicateAttr::OGE => "oge",
            FCmpPredicateAttr::OLT => "olt",
            FCmpPredicateAttr::OLE => "ole",
            FCmpPredicateAttr::ONE => "one",
            FCmpPredicateAttr::ORD => "ord",
            FCmpPredicateAttr::UEQ => "ueq",
            FCmpPredicateAttr::UGT => "ugt",
            FCmpPredicateAttr::UGE => "uge",
            FCmpPredicateAttr::ULT => "ult",
            FCmpPredicateAttr::ULE => "ule",
            FCmpPredicateAttr::UNE => "une",
            FCmpPredicateAttr::UNO => "uno",
            FCmpPredicateAttr::True => "true",
        };

        write!(output, "  {res_name} = fcmp {pred} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_select(
        &mut self,
        op: &ops::SelectOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let cond = op_ref.get_operand(0);
        let true_val = op_ref.get_operand(1);
        let false_val = op_ref.get_operand(2);
        let val_ty = true_val.get_type(self.ctx);

        write!(output, "  {res_name} = select i1 ").unwrap();
        self.export_value(cond, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(true_val, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(false_val, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_call(
        &mut self,
        op: &ops::CallOp,
        value_names: &FxHashMap<Value, String>,
        next_value_id: &mut usize,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let llvm_call_attrs = op.get_attr_llvm_call_attrs(self.ctx);
        let callee = op.callee(self.ctx);
        let func_ty = op.callee_type(self.ctx);
        let func_ty_ref = func_ty.deref(self.ctx);
        let llvm_func_ty = func_ty_ref.downcast_ref::<FuncType>().unwrap();
        let ret_ty = llvm_func_ty.result_type();
        let is_void = ret_ty.deref(self.ctx).is::<VoidType>();

        let direct_callee_name = match &callee {
            CallOpCallable::Direct(identifier) => {
                let name = identifier.to_string();
                Some(if name.starts_with("llvm_") {
                    super::names::decode_intrinsic_identifier(&name)
                } else {
                    super::names::strip_device_prefix(&name)
                })
            }
            CallOpCallable::Indirect(_) => None,
        };
        if let Some(name) = &direct_callee_name
            && self.function_grid_constants.contains_key(name)
        {
            return Err(format!(
                "grid-constant kernel entry `@{name}` cannot be called as a device function; move shared code into an ordinary helper"
            ));
        }
        let legacy_atomic_add = if let Some(name) = &direct_callee_name {
            self.legacy_nvvm_atomic_add_signature(name, llvm_func_ty)?
        } else {
            None
        };

        // Device externs keep their pointer types outside the opaque-pointer
        // module. Clone the declaration before updating exporter state.
        let device_extern = direct_callee_name
            .as_deref()
            .and_then(|name| self.device_extern(name))
            .cloned();

        let indirect_callee = match callee {
            CallOpCallable::Indirect(value) => Some(value),
            CallOpCallable::Direct(_) => None,
        };
        let indirect_name = if let Some(value) = indirect_callee {
            let source_ty = value.get_type(self.ctx);
            let source_ref = source_ty.deref(self.ctx);
            let pointer = source_ref.downcast_ref::<PointerType>().ok_or_else(|| {
                format!(
                    "indirect callee is not a pointer: `{}`",
                    source_ref.disp(self.ctx)
                )
            })?;
            if pointer.address_space() != 0 {
                return Err(format!(
                    "indirect callee uses address space {}, but NVPTX function pointers must use program address space 0",
                    pointer.address_space()
                ));
            }
            if self.legacy_typed_pointers() {
                let name = Self::fresh_value_name(next_value_id);
                write!(output, "  {name} = bitcast ").unwrap();
                self.export_canonical_pointer_type(pointer.address_space(), output);
                write!(output, " ").unwrap();
                self.export_value(value, value_names, output)?;
                write!(output, " to ").unwrap();
                self.export_function_pointer_type(func_ty, output)?;
                writeln!(output).unwrap();
                Some(name)
            } else {
                None
            }
        } else {
            None
        };

        let argument_offset = usize::from(indirect_callee.is_some());
        let arguments: Vec<Value> = op_ref.operands().skip(argument_offset).collect();
        let legacy_atomic_pointer_argument = if let Some((pointee, _)) = legacy_atomic_add {
            let pointer = arguments.first().copied().ok_or_else(|| {
                "legacy NVVM atomic-add call is missing its pointer argument".to_string()
            })?;
            Some(self.typed_pointer_operand(
                pointer,
                pointee,
                value_names,
                next_value_id,
                output,
            )?)
        } else {
            None
        };
        let adapted_arguments = if let Some(decl) = &device_extern {
            if arguments.len() != decl.param_types.len() {
                return Err(format!(
                    "device extern `@{}` expects {} arguments, lowered call has {}",
                    decl.export_name,
                    decl.param_types.len(),
                    arguments.len()
                ));
            }
            let mut adapted = Vec::with_capacity(arguments.len());
            for (index, (arg, expected)) in
                arguments.iter().copied().zip(&decl.param_types).enumerate()
            {
                adapted.push(self.device_extern_argument(
                    arg,
                    expected,
                    value_names,
                    next_value_id,
                    output,
                    &format!("`@{}` argument {index}", decl.export_name),
                )?);
            }
            self.validate_device_extern_value_type(
                &decl.return_type,
                ret_ty,
                &format!("`@{}` result", decl.export_name),
            )?;
            Some(adapted)
        } else {
            None
        };

        // Convert a typed pointer returned by a legacy extern back to the
        // internal byte-pointer type.
        let normalize_pointer_result = device_extern.as_ref().is_some_and(|decl| {
            self.legacy_typed_pointers()
                && decl.return_type.pointer_parts().is_some()
                && !decl.return_type.is_canonical_byte_pointer()
        });
        let final_result_name = if is_void {
            None
        } else {
            Some(
                self.value_name(op_ref.get_result(0), value_names)?
                    .to_string(),
            )
        };
        let call_result_name = if normalize_pointer_result {
            Some(Self::fresh_value_name(next_value_id))
        } else {
            final_result_name.clone()
        };

        // Void calls: "call void @func(...)"
        // Non-void:   "%vN = call <type> @func(...)"
        if is_void {
            write!(output, "  call void").unwrap();
        } else {
            write!(output, "  {} = call ", call_result_name.as_deref().unwrap()).unwrap();
            if let Some(decl) = &device_extern {
                // Return-position attributes precede the type:
                // `%r = call signext i8 @f()`.
                if let Some(attr) = decl.return_type.ext_attr() {
                    write!(output, "{attr} ").unwrap();
                }
                decl.return_type
                    .write_llvm(output, self.legacy_typed_pointers())?;
            } else {
                self.export_type(ret_ty, output)?;
            }
        }

        match callee {
            CallOpCallable::Direct(_) => {
                write!(output, " @{}(", direct_callee_name.as_deref().unwrap()).unwrap();
            }
            CallOpCallable::Indirect(val) => {
                write!(output, " ").unwrap();
                if let Some(name) = &indirect_name {
                    write!(output, "{name}").unwrap();
                } else {
                    self.export_value(val, value_names, output)?;
                }
                write!(output, "(").unwrap();
            }
        }

        for (i, arg) in arguments.iter().copied().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            if let Some(decl) = &device_extern {
                decl.param_types[i].write_llvm_with_attr(output, self.legacy_typed_pointers())?;
            } else if i == 0
                && let Some((pointee, address_space)) = legacy_atomic_add
            {
                self.export_pointer_to(pointee, address_space, output)?;
            } else {
                self.export_type(arg.get_type(self.ctx), output)?;
            }
            write!(output, " ").unwrap();
            if let Some(adapted) = &adapted_arguments {
                write!(output, "{}", adapted[i]).unwrap();
            } else if i == 0
                && let Some(adapted) = &legacy_atomic_pointer_argument
            {
                write!(output, "{adapted}").unwrap();
            } else {
                self.export_value(arg, value_names, output)?;
            }
        }

        // Every device call is emitted `convergent` (attr group #0). GPU code is
        // convergent-by-default (as in Clang/nvcc): if the callee transitively
        // performs a barrier / shuffle / vote, `opt -O2` must not sink or
        // duplicate the call across divergent control flow. opt strips the
        // attribute from calls it proves never reach a convergent op.
        write!(output, ")").unwrap();
        if let Some(attrs) = llvm_call_attrs.as_deref() {
            self.export_llvm_attributes(attrs, output)?;
        }
        writeln!(output, " #0").unwrap();
        self.convergent_used = true;

        if normalize_pointer_result {
            let decl = device_extern.as_ref().unwrap();
            let (_, address_space) = decl.return_type.pointer_parts().unwrap();
            write!(
                output,
                "  {} = bitcast ",
                final_result_name.as_deref().unwrap()
            )
            .unwrap();
            decl.return_type.write_llvm(output, true)?;
            write!(output, " {} to ", call_result_name.as_deref().unwrap()).unwrap();
            self.export_canonical_pointer_type(address_space, output);
            writeln!(output).unwrap();
        }
        Ok(())
    }

    fn emit_bitcast(
        &self,
        op: &ops::BitcastOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let source = op_ref.get_operand(0);
        let result = op_ref.get_result(0);
        let source_ty = source.get_type(self.ctx);
        let result_ty = result.get_type(self.ctx);

        if self.is_pointer_type(source_ty) && self.is_pointer_type(result_ty) {
            let source_ref = source_ty.deref(self.ctx);
            let result_ref = result_ty.deref(self.ctx);
            let source_pointer = source_ref.downcast_ref::<PointerType>().unwrap();
            let result_pointer = result_ref.downcast_ref::<PointerType>().unwrap();
            if source_pointer.address_space() != result_pointer.address_space() {
                return Err(format!(
                    "pointer bitcast cannot cross address spaces {} -> {}; use addrspacecast",
                    source_pointer.address_space(),
                    result_pointer.address_space()
                ));
            }

            if !self.legacy_typed_pointers() {
                return self.export_cast("bitcast", op.get_operation(), value_names, output);
            }

            // Opaque pointer-to-pointer bitcasts carry no pointee information.
            // In typed LLVM, use a zero-offset byte GEP for this identity.
            let result_name = self.value_name(result, value_names)?;
            write!(output, "  {result_name} = getelementptr i8, ").unwrap();
            self.export_canonical_pointer_type(source_pointer.address_space(), output);
            write!(output, " ").unwrap();
            self.export_value(source, value_names, output)?;
            writeln!(output, ", i64 0").unwrap();
            return Ok(());
        }

        self.export_cast("bitcast", op.get_operation(), value_names, output)
    }

    fn emit_addrspacecast(
        &self,
        op: &ops::AddrSpaceCastOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let source_ty = op_ref.get_operand(0).get_type(self.ctx);
        let result_ty = op_ref.get_result(0).get_type(self.ctx);
        let source_ref = source_ty.deref(self.ctx);
        let result_ref = result_ty.deref(self.ctx);
        let source_pointer = source_ref
            .downcast_ref::<PointerType>()
            .ok_or_else(|| "addrspacecast source must be a pointer".to_string())?;
        let result_pointer = result_ref
            .downcast_ref::<PointerType>()
            .ok_or_else(|| "addrspacecast result must be a pointer".to_string())?;

        if source_pointer.address_space() == result_pointer.address_space() {
            return Err(format!(
                "addrspacecast must change address spaces; source and result are both address space {}",
                source_pointer.address_space()
            ));
        }

        self.export_cast("addrspacecast", op.get_operation(), value_names, output)
    }

    fn emit_inline_asm(
        &mut self,
        op: &ops::InlineAsmOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let asm_template = read_string_attr(op.get_attr_llvm_inline_asm_template(self.ctx));
        let constraints = read_string_attr(op.get_attr_llvm_inline_asm_constraints(self.ctx));
        let has_sideeffect = op
            .get_attr_llvm_inline_asm_side_effects(self.ctx)
            .map(|attr| bool::from((*attr).clone()))
            .unwrap_or(true);
        let is_convergent = ops::inline_asm_convergence(self.ctx, op)?;

        // pliron-llvm always stores a single result slot (a void result for
        // no-value asm), so decide void vs valued by the result *type*, not the
        // result count.
        let res = op_ref.get_result(0);
        let res_ty = res.get_type(self.ctx);
        if res_ty.deref(self.ctx).is::<VoidType>() {
            write!(output, "  call void").unwrap();
        } else {
            let res_name = value_names.get(&res).unwrap();
            write!(output, "  {res_name} = call ").unwrap();
            self.export_type(res_ty, output)?;
        }

        write!(output, " asm").unwrap();
        if has_sideeffect {
            write!(output, " sideeffect").unwrap();
        }
        let asm_template = format_string_literal(&asm_template);
        let constraints = format_string_literal(&constraints);
        write!(output, " {asm_template}, {constraints}(").unwrap();
        for (i, arg) in op_ref.operands().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(arg.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(arg, value_names, output)?;
        }

        if is_convergent {
            writeln!(output, ") #0").unwrap();
            self.convergent_used = true;
        } else {
            writeln!(output, ")").unwrap();
        }
        Ok(())
    }

    fn emit_zext(
        &mut self,
        op: &ops::ZExtOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let val = op_ref.get_operand(0);

        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        let nneg = op_ref
            .attributes
            .get::<pliron::builtin::attributes::BoolAttr>(&nneg_key)
            .map(|b| bool::from(b.clone()))
            .unwrap_or(false);

        write!(output, "  {res_name} = zext ").unwrap();
        if nneg {
            write!(output, "nneg ").unwrap();
        }
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_extract_value(
        &mut self,
        op: &ops::ExtractValueOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);

        write!(output, "  {res_name} = extractvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        for idx in op.indices(self.ctx) {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_insert_value(
        &mut self,
        op: &ops::InsertValueOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);

        write!(output, "  {res_name} = insertvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        for idx in op.indices(self.ctx) {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_undef(&self, op: &ops::UndefOp, value_names: &mut FxHashMap<Value, String>) {
        let res = op.get_operation().deref(self.ctx).get_result(0);
        value_names.insert(res, "undef".to_string());
    }

    fn emit_constant(
        &self,
        op: &ops::ConstantOp,
        value_names: &mut FxHashMap<Value, String>,
    ) -> Result<(), String> {
        let val_attr = op.get_value(self.ctx);
        let val_attr = &*val_attr as &dyn Attribute;
        let const_str = if let Some(int_attr) = val_attr.downcast_ref::<IntegerAttr>() {
            // Use APInt's proper decimal string conversion instead of parsing debug format.
            // The old code parsed debug strings like "APInt { value: 0x4000_0000_0000_u64 }"
            // by splitting on '_', which broke for values with underscore grouping
            // (e.g., 1u64 << 46 = 0x4000_0000_0000 would become 0x4000 = 16384).
            int_attr.value().to_string_unsigned_decimal()
        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
            format_half_literal(crate::fp16_attr_to_bits(fp16_attr))
        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
            let float_val: f32 = fp32_attr.clone().into();
            format_float_literal(f64::from(float_val))
        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
            let float_val: f64 = fp64_attr.clone().into();
            format_float_literal(float_val)
        } else if self.legacy_typed_pointers() {
            return Err(format!(
                "legacy LLVM 7 export cannot render constant attribute `{}`",
                val_attr.disp(self.ctx)
            ));
        } else {
            "0".to_string() // Fallback
        };

        let res = op.get_operation().deref(self.ctx).get_result(0);
        value_names.insert(res, const_str);
        Ok(())
    }

    fn emit_address_of(
        &self,
        op: &ops::AddressOfOp,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        // AddressOfOp is virtual in textual LLVM IR: every use site prints the
        // global symbol directly. The naming pre-pass in export_function
        // registers the result as `@<global_name>` before any block is emitted,
        // so there is nothing to emit here. The assertion keeps the contract
        // honest if the pre-pass is ever refactored.
        let res = op.get_operation().deref(self.ctx).get_result(0);
        if !self.legacy_typed_pointers() {
            debug_assert!(
                value_names
                    .get(&res)
                    .is_some_and(|name| name.starts_with('@')),
                "AddressOfOp result must be pre-registered as a global symbol by \
                 the naming pre-pass; got {:?}",
                value_names.get(&res),
            );
            return Ok(());
        }

        let symbol_name = op.get_global_name(self.ctx).to_string();
        let result_type = res.get_type(self.ctx);
        let result_ref = result_type.deref(self.ctx);
        let result_pointer = result_ref.downcast_ref::<PointerType>().ok_or_else(|| {
            format!(
                "addressof result is not a pointer: `{}`",
                result_ref.disp(self.ctx)
            )
        })?;

        if let Some(info) = self.global_symbols.get(&symbol_name).copied() {
            if result_pointer.address_space() != info.address_space {
                return Err(format!(
                    "addressof `@{symbol_name}` address-space mismatch: result is {}, global is {}",
                    result_pointer.address_space(),
                    info.address_space
                ));
            }

            let result_name = self.value_name(res, value_names)?;
            if self.is_i8_type(info.value_type) {
                write!(output, "  {result_name} = getelementptr i8, ").unwrap();
                self.export_canonical_pointer_type(info.address_space, output);
                writeln!(output, " @{symbol_name}, i64 0").unwrap();
            } else {
                write!(output, "  {result_name} = bitcast ").unwrap();
                self.export_pointer_to(info.value_type, info.address_space, output)?;
                write!(output, " @{symbol_name} to ").unwrap();
                self.export_canonical_pointer_type(info.address_space, output);
                writeln!(output).unwrap();
            }
            return Ok(());
        }

        let function_name = if symbol_name.starts_with("llvm_") {
            super::names::decode_intrinsic_identifier(&symbol_name)
        } else {
            super::names::strip_device_prefix(&symbol_name)
        };
        if self.function_grid_constants.contains_key(&function_name) {
            return Err(format!(
                "cannot take the device function address of grid-constant kernel entry `@{function_name}`; its launch ABI is not an ordinary function ABI"
            ));
        }
        self.function_types.get(&function_name).ok_or_else(|| {
            format!("legacy addressof references unknown symbol `@{symbol_name}`")
        })?;
        if result_pointer.address_space() != 0 {
            return Err(format!(
                "function addressof `@{function_name}` must produce a program-address-space (0) pointer, got address space {}",
                result_pointer.address_space()
            ));
        }

        let result_name = self.value_name(res, value_names)?;
        write!(output, "  {result_name} = bitcast ").unwrap();
        if let Some(decl) = self.device_extern(&function_name) {
            decl.return_type.write_llvm(output, true)?;
            write!(output, " (").unwrap();
            for (index, parameter) in decl.param_types.iter().enumerate() {
                if index != 0 {
                    write!(output, ", ").unwrap();
                }
                parameter.write_llvm(output, true)?;
            }
            write!(output, ")*").unwrap();
        } else {
            self.export_named_function_pointer_type(&function_name, output)?;
        }
        write!(output, " @{function_name} to i8*").unwrap();
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_binop(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();

        // Float binops (fadd/fsub/fmul/fdiv/frem) may carry fast-math flags;
        // they are emitted right after the opcode (e.g. `fadd fast float ...`).
        // Integer binops never carry the attribute, and float ops lowered from
        // ordinary Rust arithmetic carry empty flags, so this is a no-op for
        // every existing op and only fires for the `f*_fast` intrinsics.
        let fast_math = op_ref
            .attributes
            .get::<FastmathFlagsAttr>(&ATTR_KEY_FAST_MATH_FLAGS)
            .map(|attr| attr.0)
            .unwrap_or_else(FastmathFlags::empty);

        write!(output, "  {res_name} = {op_name} ").unwrap();
        if !fast_math.is_empty() {
            write!(output, "{} ", fastmath_keywords(fast_math)).unwrap();
        }
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    /// Export a cast: `%res = <op_name> <src_type> <val> to <dst_type>`
    pub(super) fn export_cast(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let val = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_value(
        &self,
        val: Value,
        value_names: &FxHashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        if let Some(name) = value_names.get(&val) {
            write!(output, "{name}").unwrap();
        } else {
            return Err(format!("missing exported name for value {val:?}"));
        }
        Ok(())
    }
}

/// Render LLVM fast-math flag keywords for the given flag set.
///
/// The all-bits set (`FastmathFlags::FAST`) prints as the `fast` shorthand;
/// any proper subset prints the individual keywords in LLVM's canonical order.
/// Callers only emit this when the set is non-empty.
fn fastmath_keywords(flags: FastmathFlags) -> String {
    if flags.contains(FastmathFlags::FAST) {
        return "fast".to_string();
    }
    let mut parts = Vec::new();
    if flags.contains(FastmathFlags::NNAN) {
        parts.push("nnan");
    }
    if flags.contains(FastmathFlags::NINF) {
        parts.push("ninf");
    }
    if flags.contains(FastmathFlags::NSZ) {
        parts.push("nsz");
    }
    if flags.contains(FastmathFlags::ARCP) {
        parts.push("arcp");
    }
    if flags.contains(FastmathFlags::CONTRACT) {
        parts.push("contract");
    }
    if flags.contains(FastmathFlags::AFN) {
        parts.push("afn");
    }
    if flags.contains(FastmathFlags::REASSOC) {
        parts.push("reassoc");
    }
    parts.join(" ")
}

/// Return the address space of a pointer type, or 0 for non-pointer types.
fn addrspace_of(ty: pliron::r#type::TypeHandle, ctx: &pliron::context::Context) -> u32 {
    ty.deref(ctx)
        .downcast_ref::<crate::types::PointerType>()
        .map_or(0, crate::types::PointerType::address_space)
}

/// Format the pointer operand prefix for memory instructions.
///
/// Returns `"ptr addrspace(N) "` for non-default address spaces, `"ptr "` otherwise.
fn ptr_qualifier(addrspace: u32) -> String {
    if addrspace != 0 {
        format!("ptr addrspace({addrspace}) ")
    } else {
        "ptr ".to_string()
    }
}

/// Read an optional `StringAttr` to an owned `String` (absent → empty).
fn read_string_attr(attr: Option<Ref<StringAttr>>) -> String {
    attr.map(|s| String::from((*s).clone())).unwrap_or_default()
}

/// LLVM mnemonic for an atomic ordering. Ordering is always present on atomic
/// ops; `monotonic` is LLVM's weakest default if somehow absent.
fn fmt_ordering(ord: Option<Ref<AtomicOrderingAttr>>) -> &'static str {
    match ord.as_deref() {
        Some(AtomicOrderingAttr::Acquire) => "acquire",
        Some(AtomicOrderingAttr::Release) => "release",
        Some(AtomicOrderingAttr::AcqRel) => "acq_rel",
        Some(AtomicOrderingAttr::SeqCst) => "seq_cst",
        Some(AtomicOrderingAttr::Monotonic) | None => "monotonic",
    }
}

/// LLVM mnemonic for an `atomicrmw` operation.
fn fmt_rmw_kind(kind: Option<Ref<AtomicRmwKindAttr>>) -> &'static str {
    match kind.as_deref() {
        Some(AtomicRmwKindAttr::Xchg) | None => "xchg",
        Some(AtomicRmwKindAttr::Add) => "add",
        Some(AtomicRmwKindAttr::Sub) => "sub",
        Some(AtomicRmwKindAttr::And) => "and",
        Some(AtomicRmwKindAttr::Nand) => "nand",
        Some(AtomicRmwKindAttr::Or) => "or",
        Some(AtomicRmwKindAttr::Xor) => "xor",
        Some(AtomicRmwKindAttr::Max) => "max",
        Some(AtomicRmwKindAttr::Min) => "min",
        Some(AtomicRmwKindAttr::UMax) => "umax",
        Some(AtomicRmwKindAttr::UMin) => "umin",
        Some(AtomicRmwKindAttr::FAdd) => "fadd",
        Some(AtomicRmwKindAttr::FSub) => "fsub",
        Some(AtomicRmwKindAttr::FMax) => "fmax",
        Some(AtomicRmwKindAttr::FMin) => "fmin",
    }
}

/// Format a syncscope suffix from pliron-llvm's dedicated [`SyncScopeAttr`].
fn fmt_syncscope(scope: SyncScopeAttr) -> String {
    // `SyncScopeAttr::to_name` returns the LLVM-IR scope name; the system
    // scope is unnamed (empty string) and is omitted entirely in textual IR.
    let name = scope.to_name();
    if name.is_empty() {
        String::new()
    } else {
        format!(" syncscope(\"{name}\")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::LocalMemoryProvenanceAttr;

    fn decode_hex_payload(encoded: &str) -> String {
        let hex = encoded.strip_suffix('_').expect("sentinel-terminated");
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap())
            .collect();
        String::from_utf8(bytes).expect("capped payload stays valid UTF-8")
    }

    #[test]
    fn provenance_encoding_round_trips_and_is_sentinel_terminated() {
        let provenance = LocalMemoryProvenanceAttr {
            local_index: 3,
            size_bytes: 16,
            binding_name: "scratch".into(),
            type_name: "[u32; 4]".into(),
        };
        let encoded = encode_local_memory_provenance(&provenance);
        assert_eq!(decode_hex_payload(&encoded), "3\t16\tscratch\t[u32; 4]");
    }

    #[test]
    fn oversized_provenance_payloads_are_capped_for_llvm() {
        // LLVM's textual parser mis-lexes kilobyte-long value names, so a
        // pathological type spelling must not reach the emitted identifier
        // at full length.
        let provenance = LocalMemoryProvenanceAttr {
            local_index: 7,
            size_bytes: 4096,
            binding_name: "state".into(),
            type_name: "Ty { … }".repeat(1000).into(),
        };
        let encoded = encode_local_memory_provenance(&provenance);
        assert!(encoded.len() <= LOCAL_MEMORY_MAX_PAYLOAD_BYTES * 2 + 1);
        // The capped payload still decodes as UTF-8 (the truncation respects
        // char boundaries even through the multi-byte ellipsis) and keeps the
        // leading fields intact.
        let decoded = decode_hex_payload(&encoded);
        assert!(decoded.starts_with("7\t4096\tstate\tTy { "));
    }
}
