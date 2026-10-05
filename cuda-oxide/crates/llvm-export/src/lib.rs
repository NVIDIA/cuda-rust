/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM dialect for cuda-oxide.
//!
//! The dialect *modeling* (types, ops, attributes, op-interfaces) now lives
//! upstream in [`pliron_llvm`]; this crate is a thin shim that re-exports it so
//! existing `llvm_export::{ops,types,attributes,op_interfaces}` paths keep
//! resolving, plus the small set of GPU-specific extensions pliron-llvm does
//! not carry (named address spaces, fp16 bit helpers). The
//! pure-Rust textual `.ll` exporter ([`export`]) stays here: pliron-llvm only
//! emits real `.ll` via an `llvm-sys` bridge, which is exactly what cuda-oxide
//! is avoiding.
//!
//! Registration is automatic: every dialect/op/type/attribute linked into the
//! binary registers itself when a [`pliron::context::Context`] is created
//! (`Context::default` runs all link-time `CONTEXT_REGISTRATIONS`), so no
//! explicit `register()` entry point is needed.

pub mod export;

/// Stable marker used when an operation must remain in a debug-scoped LLVM
/// function but does not correspond to user-written source.
pub const ARTIFICIAL_DEBUG_LOCATION_NAME: &str = "cuda_oxide.artificial";

/// Build a location which the textual LLVM exporter emits as line zero.
///
/// LLVM requires calls in a debug-scoped, inlinable function to carry a
/// `DILocation`. An ordinary [`pliron::location::Location::Unknown`] therefore
/// falls back to the function's source line. This explicit marker distinguishes
/// compiler-generated setup which must have a location for LLVM validity but
/// must not create a user-visible line-table entry.
pub fn artificial_debug_location() -> pliron::location::Location {
    pliron::location::Location::Named {
        name: ARTIFICIAL_DEBUG_LOCATION_NAME.into(),
        child_loc: Box::new(pliron::location::Location::Unknown),
    }
}

/// Whether `loc` carries the explicit artificial-debug marker.
pub fn is_artificial_debug_location(loc: &pliron::location::Location) -> bool {
    matches!(
        loc,
        pliron::location::Location::Named { name, .. }
            if name == ARTIFICIAL_DEBUG_LOCATION_NAME
    )
}

/// LLVM types: re-exported from pliron-llvm, plus GPU address-space helpers.
pub mod types {
    pub use pliron_llvm::types::*;

    /// `f16` maps to pliron core's builtin `FP16Type`.
    pub use pliron::builtin::types::FP16Type as HalfType;

    /// NVVM address spaces (generic=0, global=1, shared=3, constant=4,
    /// local=5, tmem=6). pliron-llvm's `PointerType` stores a raw `u32`
    /// address space with no named constants, so we keep these here.
    pub mod address_space {
        /// Generic / flat address space.
        pub const GENERIC: u32 = 0;
        /// Global memory.
        pub const GLOBAL: u32 = 1;
        /// Shared (CTA) memory.
        pub const SHARED: u32 = 3;
        /// Constant memory.
        pub const CONSTANT: u32 = 4;
        /// Thread-local memory.
        pub const LOCAL: u32 = 5;
        /// Tensor memory (Blackwell tcgen05).
        pub const TMEM: u32 = 6;
    }

    use pliron::{context::Context, r#type::TypedHandle};
    pub use pliron_llvm::types::PointerType;

    /// Address-space convenience constructors/predicates re-homed from the
    /// pre-migration local `PointerType`. Upstream ships only
    /// `PointerType::get(ctx, address_space)` + `address_space()`.
    pub trait PointerTypeExt {
        /// Pointer into the generic address space.
        fn get_generic(ctx: &mut Context) -> TypedHandle<PointerType>;
        /// Pointer into the shared address space.
        fn get_shared(ctx: &mut Context) -> TypedHandle<PointerType>;
        /// Pointer into the global address space.
        fn get_global(ctx: &mut Context) -> TypedHandle<PointerType>;
        /// Pointer into tensor memory.
        fn get_tmem(ctx: &mut Context) -> TypedHandle<PointerType>;
        /// True if this pointer is in the shared address space.
        fn is_shared(&self) -> bool;
        /// True if this pointer is in tensor memory.
        fn is_tmem(&self) -> bool;
    }

    impl PointerTypeExt for PointerType {
        fn get_generic(ctx: &mut Context) -> TypedHandle<PointerType> {
            PointerType::get(ctx, address_space::GENERIC)
        }
        fn get_shared(ctx: &mut Context) -> TypedHandle<PointerType> {
            PointerType::get(ctx, address_space::SHARED)
        }
        fn get_global(ctx: &mut Context) -> TypedHandle<PointerType> {
            PointerType::get(ctx, address_space::GLOBAL)
        }
        fn get_tmem(ctx: &mut Context) -> TypedHandle<PointerType> {
            PointerType::get(ctx, address_space::TMEM)
        }
        fn is_shared(&self) -> bool {
            self.address_space() == address_space::SHARED
        }
        fn is_tmem(&self) -> bool {
            self.address_space() == address_space::TMEM
        }
    }
}

/// LLVM attributes: re-exported from pliron-llvm, plus the cuda-oxide names
/// for atomic ordering / rmw-kind.
pub mod attributes {
    pub use pliron_llvm::attributes::*;

    /// `f16` constants use pliron core's builtin `FPHalfAttr`.
    pub use pliron::builtin::attributes::FPHalfAttr;

    /// Atomic ordering / rmw-kind were named `Llvm*` locally; upstream calls
    /// them `Atomic*Attr`. Keep the local names resolving.
    pub use pliron_llvm::attributes::{
        AtomicOrderingAttr as LlvmAtomicOrdering, AtomicRmwKindAttr as LlvmAtomicRmwKind,
    };
}

/// LLVM ops: re-exported from pliron-llvm, plus the builtin `ConstantOp` and
/// an `AsmKind` convenience builder backed by upstream inline-asm semantics.
pub mod ops {
    pub use pliron_llvm::ops::*;

    use std::path::PathBuf;

    use combine::stream::position::SourcePosition;

    /// `ConstantOp` moved from the LLVM dialect to pliron core `builtin`.
    pub use pliron::builtin::ops::ConstantOp;

    use pliron::{
        builtin::{
            attributes::{BoolAttr, StringAttr},
            op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface},
        },
        common_traits::Verify,
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        result::Error,
        r#type::TypeHandle,
        value::Value,
    };
    use pliron_derive::{pliron_attr, pliron_op};
    use pliron_llvm::attributes::AlignmentAttr;
    pub use pliron_llvm::llvm_attrs::{LlvmAttrValue, LlvmAttributesAttr};
    pub use pliron_llvm::ops::{GlobalOp, InlineAsmOp};

    /// Inline asm semantics for LLVM optimization hints.
    ///
    /// This is the complete classification: two orthogonal axes (convergent ×
    /// side-effects) produce exactly four variants, all valid for GPU inline
    /// asm. No further axes are needed because:
    ///
    /// - **Memory effects** (`nomem`/`readonly`/`readwrite`) are unnecessary:
    ///   cuda-oxide's inline asm is either a pure register-to-register
    ///   conversion or a full side-effecting op. Fine-grained memory
    ///   classification would only help if we lowered loads/stores through
    ///   inline asm, which we don't — those go through proper LLVM ops.
    ///
    /// - **`noreturn`/`may_unwind`** don't apply: PTX inline asm always
    ///   returns and never unwinds.
    ///
    /// - **`preserves_flags`/`nostack`** are CPU concepts with no PTX
    ///   equivalent.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum AsmKind {
        /// Convergent + side effects. Warp-synchronous operations that
        /// synchronize threads or write memory: `bar.sync`, `mma.sync`,
        /// `wgmma`, `tcgen05`, `cp.async`.
        Convergent,
        /// Convergent, no side effects: the asm must not cross divergent
        /// control flow, but may be merged with an identical call or dropped
        /// when its result is unused.
        ///
        /// Currently unused, and warp collectives are not candidates for it. A
        /// collective's result depends on every lane's input, not just the
        /// operands of this call, so two calls with equal operands are not
        /// interchangeable. Upstream LLVM says the same by declaring `shfl`,
        /// `vote`, `match`, `redux`, and `activemask`
        /// `IntrConvergent, IntrInaccessibleMemOnly` rather than `IntrNoMem`;
        /// those lower through [`AsmKind::Convergent`] here. This variant would
        /// fit a convergent op that is a true function of its own operands, and
        /// no such op is currently modelled.
        ConvergentPure,
        /// Side effects, not convergent. Non-collective operations that
        /// modify memory or hardware state: `st.global` via asm, hardware
        /// timer reads.
        SideEffect,
        /// No side effects, not convergent. Pure register-to-register data
        /// conversions: `cvt.rn.f16x2.f32`, `cvt.rn.bf16x2.f32`, `prmt`.
        Pure,
    }

    /// Kernel-entry parameter validity proven at the Rust MIR import boundary.
    ///
    /// Presence proves `nonnull`; the payload is the rustc ABI alignment of
    /// the pointee represented by this physical LLVM parameter. It deliberately
    /// carries no aliasing, readonly, or dereferenceability promise.
    #[pliron_attr(
        name = "llvm.kernel_reference_param_validity",
        format = "$0",
        verifier = "succ"
    )]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct KernelReferenceParamValidityAttr(pub u64);

    fn kernel_reference_param_validity_key(index: usize) -> Identifier {
        Identifier::try_new(reserved_oxide_symbols::kernel_reference_param_validity_key(
            index,
        ))
        .expect("valid kernel reference parameter validity attribute key")
    }

    /// Attach a proven reference-validity fact to one physical kernel parameter.
    pub fn set_kernel_reference_param_validity(
        ctx: &mut Context,
        op: Ptr<Operation>,
        index: usize,
        validity: KernelReferenceParamValidityAttr,
    ) {
        op.deref_mut(ctx)
            .attributes
            .set(kernel_reference_param_validity_key(index), validity);
    }

    /// Collect and structurally validate all physical kernel parameter facts.
    ///
    /// Semantic proof is owned by `mir-importer`; this helper only validates
    /// the transport representation before textual LLVM export.
    pub fn kernel_reference_param_validity_entries(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Result<Vec<(usize, KernelReferenceParamValidityAttr)>, String> {
        let operation = op.deref(ctx);
        let mut result = Vec::new();
        for (key, _) in &operation.attributes.0 {
            let key_text = key.to_string();
            let Some(index_text) =
                reserved_oxide_symbols::kernel_reference_param_validity_index_text(&key_text)
            else {
                continue;
            };
            let index = index_text.parse::<usize>().map_err(|_| {
                format!(
                    "kernel reference parameter validity attribute `{key_text}` has an invalid parameter index"
                )
            })?;
            let Some(validity) = operation
                .attributes
                .get::<KernelReferenceParamValidityAttr>(key)
                .copied()
            else {
                return Err(format!(
                    "kernel reference parameter validity attribute `{key_text}` has the wrong attribute type"
                ));
            };
            if validity.0 == 0 || !validity.0.is_power_of_two() {
                return Err(format!(
                    "kernel reference parameter {index} alignment must be a non-zero power of two, found {}",
                    validity.0
                ));
            }
            result.push((index, validity));
        }
        result.sort_unstable_by_key(|(index, _)| *index);
        Ok(result)
    }

    impl AsmKind {
        /// Build the convenience classification from LLVM's two semantic axes.
        pub const fn from_semantics(convergent: bool, side_effects: bool) -> Self {
            match (convergent, side_effects) {
                (true, true) => Self::Convergent,
                (true, false) => Self::ConvergentPure,
                (false, true) => Self::SideEffect,
                (false, false) => Self::Pure,
            }
        }

        /// Whether LLVM must treat this asm call as convergent.
        pub const fn is_convergent(self) -> bool {
            matches!(self, Self::Convergent | Self::ConvergentPure)
        }

        /// Whether LLVM must retain the inline asm for side effects.
        pub const fn has_side_effects(self) -> bool {
            matches!(self, Self::Convergent | Self::SideEffect)
        }
    }

    /// Builder extension that maps CUDA's four-way convenience classification
    /// onto Pliron's upstream LLVM inline-asm representation.
    pub trait InlineAsmOpExt {
        /// Build an `InlineAsmOp` with upstream side-effect and call-site attrs.
        fn build(
            ctx: &mut Context,
            result_ty: TypeHandle,
            inputs: Vec<Value>,
            asm_template: &str,
            constraints: &str,
            kind: AsmKind,
        ) -> Self;
    }

    impl InlineAsmOpExt for InlineAsmOp {
        fn build(
            ctx: &mut Context,
            result_ty: TypeHandle,
            inputs: Vec<Value>,
            asm_template: &str,
            constraints: &str,
            kind: AsmKind,
        ) -> Self {
            let op = InlineAsmOp::new(
                ctx,
                result_ty,
                inputs,
                asm_template,
                constraints,
                kind.has_side_effects(),
            );
            if kind.is_convergent() {
                let mut attrs = LlvmAttributesAttr::new();
                attrs.set("convergent", LlvmAttrValue::Unit);
                op.set_attr_llvm_inline_asm_attrs(ctx, attrs);
            }
            op
        }
    }

    /// Derive the CUDA convenience classification from upstream LLVM semantics.
    ///
    /// `None` means a semantic attribute is missing, malformed, or unsupported
    /// by cuda-oxide's exporter. Upstream verification alone does not check the
    /// supported call-site attribute subset.
    pub fn asm_kind_opt(ctx: &Context, op: &InlineAsmOp) -> Option<AsmKind> {
        let side_effects = op
            .get_attr_llvm_inline_asm_side_effects(ctx)
            .map(|attr| bool::from((*attr).clone()))?;
        let convergent = inline_asm_convergence(ctx, op).ok()?;
        Some(AsmKind::from_semantics(convergent, side_effects))
    }

    /// The native exporter supports the unit `convergent` call-site attribute.
    /// Other LLVM attributes must be modeled explicitly before export can
    /// preserve them; silently dropping one can weaken optimizer restrictions.
    pub(crate) fn inline_asm_convergence(ctx: &Context, op: &InlineAsmOp) -> Result<bool, String> {
        let mut convergent = false;
        if let Some(attrs) = op.get_attr_llvm_inline_asm_attrs(ctx) {
            for (name, value) in attrs.iter() {
                match (name, value) {
                    ("convergent", LlvmAttrValue::Unit) => convergent = true,
                    ("convergent", _) => {
                        return Err("LLVM inline asm `convergent` requires a unit attribute".into());
                    }
                    _ => {
                        return Err(format!(
                            "unsupported LLVM inline asm call-site attribute `{name}`; only unit `convergent` is supported"
                        ));
                    }
                }
            }
        }
        Ok(convergent)
    }

    /// Query the CUDA convenience classification for an `InlineAsmOp`.
    ///
    /// Malformed unverified IR falls back to side-effecting, non-convergent asm.
    pub fn asm_kind(ctx: &Context, op: &InlineAsmOp) -> AsmKind {
        asm_kind_opt(ctx, op).unwrap_or(AsmKind::SideEffect)
    }

    /// Op-attribute key for a `GlobalOp`'s explicit alignment.
    const GLOBAL_ALIGNMENT_KEY: &str = "cuda_oxide_global_alignment";
    /// Op-attribute key for a `GlobalOp`'s Rust static initializer bytes,
    /// encoded as lowercase hex.
    const GLOBAL_INITIALIZER_HEX_KEY: &str = "cuda_oxide_global_initializer_hex";
    /// Stable rustc-side identity of the static represented by a `GlobalOp`.
    ///
    /// The emitted LLVM symbol is not sufficient for relocation lookup because
    /// ordinary device globals receive generated `__device_global_N` names.
    const GLOBAL_SOURCE_KEY: &str = "cuda_oxide_global_source_key";
    /// Versioned, length-prefixed pointer-relocation metadata for an initialized
    /// Rust static.
    const GLOBAL_INITIALIZER_RELOCATIONS_KEY: &str = "cuda_oxide_global_initializer_relocations";
    /// Rust path of the shared-memory `static` a generated `__shared_mem_N`
    /// global came from.
    ///
    /// Distinct from [`GLOBAL_SOURCE_KEY`], which is a *relocation identity*:
    /// the exporter indexes it, requires it to be unique across globals, and
    /// resolves initializer pointers through it. This key is purely
    /// descriptive, is never indexed, and carries no uniqueness requirement.
    const GLOBAL_SHARED_SOURCE_NAME_KEY: &str = "cuda_oxide_global_shared_source_name";
    /// Marks a `GlobalOp` as externally consumed even when device code does not
    /// reference it. The exporter adds marked globals to `@llvm.used`, keeping
    /// profiler metadata alive through libNVVM and nvJitLink materialization.
    const GLOBAL_RETAINED_KEY: &str = "cuda_oxide_global_retained";

    /// One pointer-width relocation inside an evaluated Rust static initializer.
    ///
    /// `source_offset` and `target_addend` are byte offsets. `target_key` is the
    /// stable rustc global key, not the generated LLVM symbol name.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct GlobalInitializerRelocation {
        pub source_offset: u64,
        pub width_bytes: u32,
        pub target_address_space: u32,
        pub target_addend: u64,
        pub target_key: String,
    }

    /// Encode initializer relocations using a versioned, length-prefixed format.
    ///
    /// The target key is length-prefixed rather than delimiter-escaped because
    /// Rust mangled symbols may contain punctuation that is meaningful to simpler
    /// ad-hoc formats.
    pub fn encode_global_initializer_relocations(
        relocations: &[GlobalInitializerRelocation],
    ) -> String {
        fn put_u64(out: &mut String, value: u64) {
            out.push_str(&value.to_string());
            out.push(' ');
        }

        fn put_str(out: &mut String, value: &str) {
            put_u64(out, value.len() as u64);
            out.push_str(value);
            out.push(' ');
        }

        let mut encoded = String::from("v1 ");
        put_u64(&mut encoded, relocations.len() as u64);
        for relocation in relocations {
            put_u64(&mut encoded, relocation.source_offset);
            put_u64(&mut encoded, u64::from(relocation.width_bytes));
            put_u64(&mut encoded, u64::from(relocation.target_address_space));
            put_u64(&mut encoded, relocation.target_addend);
            put_str(&mut encoded, &relocation.target_key);
        }
        encoded
    }

    /// Decode the format produced by [`encode_global_initializer_relocations`].
    pub fn decode_global_initializer_relocations(
        encoded: &str,
    ) -> Result<Vec<GlobalInitializerRelocation>, String> {
        fn take_u64(bytes: &[u8], pos: &mut usize, field: &str) -> Result<u64, String> {
            let start = *pos;
            while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
                *pos += 1;
            }
            if start == *pos || *pos >= bytes.len() || bytes[*pos] != b' ' {
                return Err(format!(
                    "malformed global initializer relocation metadata while reading {field}"
                ));
            }
            let value = std::str::from_utf8(&bytes[start..*pos])
                .map_err(|_| format!("non-UTF-8 digits in relocation {field}"))?
                .parse::<u64>()
                .map_err(|_| format!("invalid integer in relocation {field}"))?;
            *pos += 1;
            Ok(value)
        }

        fn take_str(bytes: &[u8], pos: &mut usize, field: &str) -> Result<String, String> {
            let len = usize::try_from(take_u64(bytes, pos, &format!("{field} length"))?)
                .map_err(|_| format!("relocation {field} length does not fit usize"))?;
            let end = (*pos)
                .checked_add(len)
                .ok_or_else(|| format!("relocation {field} length overflows"))?;
            let raw = bytes
                .get(*pos..end)
                .ok_or_else(|| format!("truncated relocation {field}"))?;
            let value = std::str::from_utf8(raw)
                .map_err(|_| format!("relocation {field} is not UTF-8"))?
                .to_string();
            *pos = end;
            if *pos >= bytes.len() || bytes[*pos] != b' ' {
                return Err(format!("relocation {field} is missing its terminator"));
            }
            *pos += 1;
            Ok(value)
        }

        let bytes = encoded.as_bytes();
        if !bytes.starts_with(b"v1 ") {
            return Err("unsupported global initializer relocation metadata version".to_string());
        }
        let mut pos = 3;
        let count = usize::try_from(take_u64(bytes, &mut pos, "relocation count")?)
            .map_err(|_| "relocation count does not fit usize".to_string())?;
        if count > bytes.len() {
            return Err("global initializer relocation count is implausibly large".to_string());
        }

        let mut relocations = Vec::with_capacity(count);
        for index in 0..count {
            let source_offset = take_u64(bytes, &mut pos, "source offset")?;
            let width_bytes = u32::try_from(take_u64(bytes, &mut pos, "pointer width")?)
                .map_err(|_| format!("relocation {index} pointer width does not fit u32"))?;
            let target_address_space =
                u32::try_from(take_u64(bytes, &mut pos, "target address space")?).map_err(
                    |_| format!("relocation {index} target address space does not fit u32"),
                )?;
            let target_addend = take_u64(bytes, &mut pos, "target addend")?;
            let target_key = take_str(bytes, &mut pos, "target key")?;
            relocations.push(GlobalInitializerRelocation {
                source_offset,
                width_bytes,
                target_address_space,
                target_addend,
                target_key,
            });
        }

        if pos != bytes.len() {
            return Err("trailing bytes in global initializer relocation metadata".to_string());
        }
        Ok(relocations)
    }

    /// Op-attribute key under which a memory op's (`load` / `store` / `alloca`)
    /// explicit ABI alignment is stashed. Stamped by the mir-lower alignment
    /// pre-pass (while types are still MIR, so `repr(align(N))` is visible)
    /// and emitted as `align N` during export.
    const OP_ALIGNMENT_KEY: &str = "cuda_oxide_op_alignment";

    // Compatibility accessors below intentionally use Pliron's upstream keys.
    // They keep existing cuda-oxide callers source-compatible without owning a
    // duplicate semantic representation.
    const LLVM_INLINE_ASM_SIDEEFFECT_KEY: &str = "llvm_inline_asm_side_effects";
    const LLVM_INLINE_ASM_ATTRS_KEY: &str = "llvm_inline_asm_attrs";

    /// Debug type metadata for a local variable described by `llvm.dbg.declare`.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum DebugLocalTypeKind {
        /// A scalar `DIBasicType`.
        Basic {
            name: String,
            size_bits: u64,
            encoding: &'static str,
        },
        /// An opaque compatibility pointer/reference `DIDerivedType`.
        ///
        /// This is the pre-existing representation for pointer shapes whose
        /// pointee cannot yet be described safely. Its null `baseType` keeps
        /// the source variable visible, although debuggers may render it as
        /// `*mut ()`.
        Pointer { name: String, size_bits: u64 },
        /// A thin pointer/reference with a safely bounded pointee type.
        ///
        /// The finite pointee tree is required so the exporter never emits a
        /// pointer with a null `baseType`, which debuggers present
        /// misleadingly as `*mut ()`. Recursive composite graphs are not
        /// represented by this tree-shaped model and must be rejected by the
        /// importer.
        TypedPointer {
            name: String,
            size_bits: u64,
            pointee: Box<DebugLocalTypeKind>,
        },
        /// A struct or tuple `DICompositeType` (`DW_TAG_structure_type`).
        ///
        /// Member offsets come from rustc's real layout, not declaration order,
        /// so this is correct even for `repr(Rust)` field reordering. Tuples are
        /// modelled as a struct whose members are named `__0`, `__1`, ...
        Struct {
            name: String,
            size_bits: u64,
            members: Vec<DebugTypeMember>,
        },
        /// A Rust enum represented with DWARF variant-part metadata.
        ///
        /// The top-level enum is a `DW_TAG_structure_type` containing one
        /// `DW_TAG_variant_part`. Each source variant is described by an
        /// artificial struct containing its payload fields. `discriminant` is
        /// absent for single/empty layouts and is the physical tag or niche
        /// carrier for multi-variant layouts.
        Enum {
            name: String,
            size_bits: u64,
            discriminant: Option<DebugEnumDiscriminant>,
            variants: Vec<DebugEnumVariant>,
        },
        /// A fixed-length array `DICompositeType` (`DW_TAG_array_type`).
        Array {
            name: String,
            size_bits: u64,
            element: Box<DebugLocalTypeKind>,
            count: u64,
        },
    }

    /// One member of a [`DebugLocalTypeKind::Struct`].
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugTypeMember {
        pub name: String,
        /// Byte-offset of the member within its parent, in bits.
        pub offset_bits: u64,
        pub ty: DebugLocalTypeKind,
    }

    /// Physical enum tag/niche carrier used by `DW_TAG_variant_part`.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugEnumDiscriminant {
        /// Bit offset of the physical carrier within the enum storage.
        pub offset_bits: u64,
        /// The physical carrier type. Niche pointer carriers are intentionally
        /// represented as an unsigned integer of the same width, matching
        /// rustc's native DWARF strategy.
        pub ty: Box<DebugLocalTypeKind>,
    }

    /// One Rust enum source variant.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugEnumVariant {
        pub name: String,
        /// Physical discriminant value selecting this variant. `None` is used
        /// for single/empty layouts and for the untagged niche variant.
        pub discriminant: Option<u64>,
        /// Payload members at their rustc-reported offsets within the complete
        /// enum object.
        pub members: Vec<DebugTypeMember>,
    }

    impl DebugLocalTypeKind {
        /// Size of this type in bits, used to fill `DIDerivedType`/member sizes.
        pub fn size_bits(&self) -> u64 {
            match self {
                DebugLocalTypeKind::Basic { size_bits, .. }
                | DebugLocalTypeKind::Pointer { size_bits, .. }
                | DebugLocalTypeKind::TypedPointer { size_bits, .. }
                | DebugLocalTypeKind::Struct { size_bits, .. }
                | DebugLocalTypeKind::Enum { size_bits, .. }
                | DebugLocalTypeKind::Array { size_bits, .. } => *size_bits,
            }
        }

        /// Whether this type belongs to the acyclic subset accepted beneath a
        /// [`DebugLocalTypeKind::TypedPointer`].
        ///
        /// Opaque pointers and source composites are excluded recursively so a
        /// typed pointer can never hide a null-base descendant or smuggle a
        /// recursive type graph into this tree-shaped representation.
        pub(crate) fn is_valid_typed_pointer_pointee(&self) -> bool {
            match self {
                DebugLocalTypeKind::Basic { .. } => true,
                DebugLocalTypeKind::TypedPointer { pointee, .. } => {
                    pointee.is_valid_typed_pointer_pointee()
                }
                DebugLocalTypeKind::Array { element, .. } => {
                    element.is_valid_typed_pointer_pointee()
                }
                DebugLocalTypeKind::Pointer { .. }
                | DebugLocalTypeKind::Struct { .. }
                | DebugLocalTypeKind::Enum { .. } => false,
            }
        }
    }

    /// Map a serialized DWARF encoding name back to its `&'static str`.
    fn debug_encoding_from_str(s: &str) -> Option<&'static str> {
        match s {
            "DW_ATE_boolean" => Some("DW_ATE_boolean"),
            "DW_ATE_float" => Some("DW_ATE_float"),
            "DW_ATE_signed" => Some("DW_ATE_signed"),
            "DW_ATE_unsigned" => Some("DW_ATE_unsigned"),
            _ => None,
        }
    }

    /// Serialize a type tree into a compact, escape-safe string.
    ///
    /// Strings are length-prefixed (`<byte-len> <bytes>`) so arbitrary type
    /// names (`&[u32]`, `Foo<'_, u64>`) round-trip without delimiter escaping.
    /// Numbers are space-terminated. This is the value stored under
    /// [`DEBUG_LOCAL_TYPE_KEY`]; the reader is [`deserialize_debug_type`].
    fn serialize_debug_type(ty: &DebugLocalTypeKind, out: &mut String) {
        fn put_u64(out: &mut String, n: u64) {
            out.push_str(&n.to_string());
            out.push(' ');
        }
        fn put_str(out: &mut String, s: &str) {
            put_u64(out, s.len() as u64);
            out.push_str(s);
        }
        match ty {
            DebugLocalTypeKind::Basic {
                name,
                size_bits,
                encoding,
            } => {
                out.push('b');
                put_u64(out, *size_bits);
                put_str(out, encoding);
                put_str(out, name);
            }
            DebugLocalTypeKind::Pointer { name, size_bits } => {
                out.push('p');
                put_u64(out, *size_bits);
                put_str(out, name);
            }
            DebugLocalTypeKind::TypedPointer {
                name,
                size_bits,
                pointee,
            } => {
                assert!(
                    pointee.is_valid_typed_pointer_pointee(),
                    "typed pointer pointee must contain only basic, typed-pointer, or array nodes"
                );
                out.push('t');
                put_u64(out, *size_bits);
                put_str(out, name);
                serialize_debug_type(pointee, out);
            }
            DebugLocalTypeKind::Struct {
                name,
                size_bits,
                members,
            } => {
                out.push('s');
                put_u64(out, *size_bits);
                put_str(out, name);
                put_u64(out, members.len() as u64);
                for member in members {
                    put_str(out, &member.name);
                    put_u64(out, member.offset_bits);
                    serialize_debug_type(&member.ty, out);
                }
            }
            DebugLocalTypeKind::Enum {
                name,
                size_bits,
                discriminant,
                variants,
            } => {
                out.push('e');
                put_u64(out, *size_bits);
                put_str(out, name);
                match discriminant {
                    Some(discriminant) => {
                        put_u64(out, 1);
                        put_u64(out, discriminant.offset_bits);
                        serialize_debug_type(&discriminant.ty, out);
                    }
                    None => put_u64(out, 0),
                }
                put_u64(out, variants.len() as u64);
                for variant in variants {
                    put_str(out, &variant.name);
                    match variant.discriminant {
                        Some(value) => {
                            put_u64(out, 1);
                            put_u64(out, value);
                        }
                        None => put_u64(out, 0),
                    }
                    put_u64(out, variant.members.len() as u64);
                    for member in &variant.members {
                        put_str(out, &member.name);
                        put_u64(out, member.offset_bits);
                        serialize_debug_type(&member.ty, out);
                    }
                }
            }
            DebugLocalTypeKind::Array {
                name,
                size_bits,
                element,
                count,
            } => {
                out.push('a');
                put_u64(out, *size_bits);
                put_str(out, name);
                put_u64(out, *count);
                serialize_debug_type(element, out);
            }
        }
    }

    const MAX_DEBUG_ATTRIBUTE_BYTES: usize = 1024 * 1024;
    const MAX_DEBUG_STRING_BYTES: usize = 64 * 1024;
    const MAX_DEBUG_TYPE_CHILDREN: usize = 16 * 1024;
    const MAX_DEBUG_TYPE_DEPTH: usize = 64;

    /// Reverse of [`serialize_debug_type`]. Returns `None` on malformed input.
    fn deserialize_debug_type(bytes: &[u8], pos: &mut usize) -> Option<DebugLocalTypeKind> {
        if bytes.len() > MAX_DEBUG_ATTRIBUTE_BYTES {
            return None;
        }
        let mut entries = 0;
        deserialize_debug_type_at(bytes, pos, 0, &mut entries)
    }

    fn deserialize_debug_type_at(
        bytes: &[u8],
        pos: &mut usize,
        depth: usize,
        entries: &mut usize,
    ) -> Option<DebugLocalTypeKind> {
        *entries = entries.checked_add(1)?;
        if depth > MAX_DEBUG_TYPE_DEPTH || *entries > MAX_DEBUG_TYPE_CHILDREN {
            return None;
        }

        fn charge(entries: &mut usize, count: usize) -> Option<()> {
            *entries = entries.checked_add(count)?;
            (*entries <= MAX_DEBUG_TYPE_CHILDREN).then_some(())
        }

        fn take_u64(bytes: &[u8], pos: &mut usize) -> Option<u64> {
            let start = *pos;
            while *pos < bytes.len() && bytes[*pos] != b' ' {
                *pos += 1;
            }
            if start == *pos || *pos >= bytes.len() {
                return None;
            }
            let n: u64 = std::str::from_utf8(&bytes[start..*pos])
                .ok()?
                .parse()
                .ok()?;
            *pos += 1; // consume the space
            Some(n)
        }
        fn take_str(bytes: &[u8], pos: &mut usize) -> Option<String> {
            let len = usize::try_from(take_u64(bytes, pos)?).ok()?;
            if len > MAX_DEBUG_STRING_BYTES {
                return None;
            }
            let end = pos.checked_add(len)?;
            if end > bytes.len() {
                return None;
            }
            let s = std::str::from_utf8(&bytes[*pos..end]).ok()?.to_string();
            *pos = end;
            Some(s)
        }

        let tag = *bytes.get(*pos)?;
        *pos += 1;
        match tag {
            b'b' => {
                let size_bits = take_u64(bytes, pos)?;
                let encoding = debug_encoding_from_str(&take_str(bytes, pos)?)?;
                let name = take_str(bytes, pos)?;
                Some(DebugLocalTypeKind::Basic {
                    name,
                    size_bits,
                    encoding,
                })
            }
            b'p' => {
                let size_bits = take_u64(bytes, pos)?;
                let name = take_str(bytes, pos)?;
                Some(DebugLocalTypeKind::Pointer { name, size_bits })
            }
            b't' => {
                let size_bits = take_u64(bytes, pos)?;
                let name = take_str(bytes, pos)?;
                let pointee = Box::new(deserialize_debug_type(bytes, pos)?);
                if !pointee.is_valid_typed_pointer_pointee() {
                    return None;
                }
                Some(DebugLocalTypeKind::TypedPointer {
                    name,
                    size_bits,
                    pointee,
                })
            }
            b's' => {
                let size_bits = take_u64(bytes, pos)?;
                let name = take_str(bytes, pos)?;
                let member_count = usize::try_from(take_u64(bytes, pos)?).ok()?;
                charge(entries, member_count)?;
                let mut members = Vec::with_capacity(member_count);
                for _ in 0..member_count {
                    let member_name = take_str(bytes, pos)?;
                    let offset_bits = take_u64(bytes, pos)?;
                    let ty = deserialize_debug_type_at(bytes, pos, depth + 1, entries)?;
                    members.push(DebugTypeMember {
                        name: member_name,
                        offset_bits,
                        ty,
                    });
                }
                Some(DebugLocalTypeKind::Struct {
                    name,
                    size_bits,
                    members,
                })
            }
            b'e' => {
                let size_bits = take_u64(bytes, pos)?;
                let name = take_str(bytes, pos)?;
                let discriminant = match take_u64(bytes, pos)? {
                    0 => None,
                    1 => {
                        let offset_bits = take_u64(bytes, pos)?;
                        let ty =
                            Box::new(deserialize_debug_type_at(bytes, pos, depth + 1, entries)?);
                        Some(DebugEnumDiscriminant { offset_bits, ty })
                    }
                    _ => return None,
                };
                let variant_count = usize::try_from(take_u64(bytes, pos)?).ok()?;
                charge(entries, variant_count)?;
                let mut variants = Vec::with_capacity(variant_count);
                for _ in 0..variant_count {
                    let variant_name = take_str(bytes, pos)?;
                    let discriminant_value = match take_u64(bytes, pos)? {
                        0 => None,
                        1 => Some(take_u64(bytes, pos)?),
                        _ => return None,
                    };
                    let member_count = usize::try_from(take_u64(bytes, pos)?).ok()?;
                    charge(entries, member_count)?;
                    let mut members = Vec::with_capacity(member_count);
                    for _ in 0..member_count {
                        let member_name = take_str(bytes, pos)?;
                        let offset_bits = take_u64(bytes, pos)?;
                        let ty = deserialize_debug_type_at(bytes, pos, depth + 1, entries)?;
                        members.push(DebugTypeMember {
                            name: member_name,
                            offset_bits,
                            ty,
                        });
                    }
                    variants.push(DebugEnumVariant {
                        name: variant_name,
                        discriminant: discriminant_value,
                        members,
                    });
                }
                Some(DebugLocalTypeKind::Enum {
                    name,
                    size_bits,
                    discriminant,
                    variants,
                })
            }
            b'a' => {
                let size_bits = take_u64(bytes, pos)?;
                let name = take_str(bytes, pos)?;
                let count = take_u64(bytes, pos)?;
                let element = Box::new(deserialize_debug_type_at(bytes, pos, depth + 1, entries)?);
                Some(DebugLocalTypeKind::Array {
                    name,
                    size_bits,
                    element,
                    count,
                })
            }
            _ => None,
        }
    }

    /// Debug metadata attached to the alloca that stores a source local.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugLocalVariableInfo {
        pub name: String,
        pub argument_index: Option<u16>,
        pub ty: DebugLocalTypeKind,
    }

    /// Source identity and semantic type for a module-scope Rust static.
    ///
    /// The physical LLVM global may use a generated symbol and byte-array
    /// storage, so neither its symbol name nor its LLVM value type can recover
    /// this information at export time.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugGlobalVariableInfo {
        /// Source-level leaf name (`COUNTER`, not its qualified path or LLVM symbol).
        pub name: String,
        /// Crate/module/function namespace components, from outermost to innermost.
        pub namespace: Vec<String>,
        pub ty: DebugLocalTypeKind,
        pub declaration: DebugSourcePosition,
        /// Mirrors rustc's `!tcx.is_reachable_non_generic(def_id)` decision.
        pub is_local_to_unit: bool,
        /// A named static declared inside a function. AS3 uses the owning
        /// subprogram as the DIE scope so cuda-gdb resolves the bare leaf while
        /// the subprogram itself remains under the structured namespace chain.
        pub is_function_local: bool,
    }

    const MAX_DEBUG_NAMESPACE_SEGMENTS: usize = 128;

    /// Serialize the complete global identity as one versioned attribute.
    ///
    /// Every string (including the semantic type blob) is byte-length-prefixed;
    /// this keeps source names and paths opaque and makes truncation detectable.
    fn encode_debug_global_info(info: &DebugGlobalVariableInfo) -> Option<String> {
        fn put_u64(out: &mut String, value: u64) {
            out.push_str(&value.to_string());
            out.push(' ');
        }

        fn put_str(out: &mut String, value: &str) {
            put_u64(out, value.len() as u64);
            out.push_str(value);
        }

        fn charge(entries: &mut usize, count: usize) -> bool {
            let Some(next) = entries.checked_add(count) else {
                return false;
            };
            if next > MAX_DEBUG_TYPE_CHILDREN {
                return false;
            }
            *entries = next;
            true
        }

        fn type_is_bounded(ty: &DebugLocalTypeKind, depth: usize, entries: &mut usize) -> bool {
            if depth > MAX_DEBUG_TYPE_DEPTH || !charge(entries, 1) {
                return false;
            }
            let bounded = |value: &str| value.len() <= MAX_DEBUG_STRING_BYTES;
            match ty {
                DebugLocalTypeKind::Basic { name, encoding, .. } => {
                    bounded(name) && bounded(encoding)
                }
                DebugLocalTypeKind::Pointer { name, .. } => bounded(name),
                DebugLocalTypeKind::TypedPointer { name, pointee, .. } => {
                    bounded(name) && type_is_bounded(pointee, depth + 1, entries)
                }
                DebugLocalTypeKind::Struct { name, members, .. } => {
                    bounded(name)
                        && charge(entries, members.len())
                        && members.iter().all(|member| {
                            bounded(&member.name) && type_is_bounded(&member.ty, depth + 1, entries)
                        })
                }
                DebugLocalTypeKind::Enum {
                    name,
                    discriminant,
                    variants,
                    ..
                } => {
                    bounded(name)
                        && charge(entries, variants.len())
                        && discriminant.as_ref().is_none_or(|discriminant| {
                            type_is_bounded(&discriminant.ty, depth + 1, entries)
                        })
                        && variants.iter().all(|variant| {
                            bounded(&variant.name)
                                && charge(entries, variant.members.len())
                                && variant.members.iter().all(|member| {
                                    bounded(&member.name)
                                        && type_is_bounded(&member.ty, depth + 1, entries)
                                })
                        })
                }
                DebugLocalTypeKind::Array { name, element, .. } => {
                    bounded(name) && type_is_bounded(element, depth + 1, entries)
                }
            }
        }

        let file = info.declaration.file.to_str()?;
        let mut type_entries = 0;
        if info.name.is_empty()
            || info.name.len() > MAX_DEBUG_STRING_BYTES
            || info.namespace.is_empty()
            || info.namespace.len() > MAX_DEBUG_NAMESPACE_SEGMENTS
            || info
                .namespace
                .iter()
                .any(|segment| segment.is_empty() || segment.len() > MAX_DEBUG_STRING_BYTES)
            || file.is_empty()
            || file.len() > MAX_DEBUG_STRING_BYTES
            || info.declaration.line <= 0
            || info.declaration.column <= 0
            || (info.is_function_local && info.namespace.len() < 2)
            || !type_is_bounded(&info.ty, 0, &mut type_entries)
        {
            return None;
        }

        let mut encoded_ty = String::new();
        serialize_debug_type(&info.ty, &mut encoded_ty);
        if encoded_ty.len() > MAX_DEBUG_ATTRIBUTE_BYTES {
            return None;
        }

        let mut out = String::from("v2 ");
        put_str(&mut out, &info.name);
        put_u64(&mut out, info.namespace.len() as u64);
        for segment in &info.namespace {
            put_str(&mut out, segment);
        }
        put_u64(&mut out, u64::from(info.is_local_to_unit));
        put_u64(&mut out, u64::from(info.is_function_local));
        put_str(&mut out, file);
        put_u64(&mut out, info.declaration.line as u64);
        put_u64(&mut out, info.declaration.column as u64);
        put_str(&mut out, &encoded_ty);
        (out.len() <= MAX_DEBUG_ATTRIBUTE_BYTES).then_some(out)
    }

    fn decode_debug_global_info(encoded: &str) -> Option<DebugGlobalVariableInfo> {
        fn take_u64(bytes: &[u8], pos: &mut usize) -> Option<u64> {
            let start = *pos;
            while *pos < bytes.len() && bytes[*pos] != b' ' {
                *pos += 1;
            }
            if start == *pos || *pos >= bytes.len() {
                return None;
            }
            let value = std::str::from_utf8(&bytes[start..*pos])
                .ok()?
                .parse()
                .ok()?;
            *pos += 1;
            Some(value)
        }

        fn take_bytes<'a>(bytes: &'a [u8], pos: &mut usize, max_len: usize) -> Option<&'a [u8]> {
            let len = usize::try_from(take_u64(bytes, pos)?).ok()?;
            if len > max_len {
                return None;
            }
            let end = (*pos).checked_add(len)?;
            let value = bytes.get(*pos..end)?;
            *pos = end;
            Some(value)
        }

        fn take_str(bytes: &[u8], pos: &mut usize) -> Option<String> {
            std::str::from_utf8(take_bytes(bytes, pos, MAX_DEBUG_STRING_BYTES)?)
                .ok()
                .map(ToOwned::to_owned)
        }

        if encoded.len() > MAX_DEBUG_ATTRIBUTE_BYTES {
            return None;
        }
        let bytes = encoded.as_bytes();
        if !bytes.starts_with(b"v2 ") {
            return None;
        }
        let mut pos = 3;
        let name = take_str(bytes, &mut pos)?;
        if name.is_empty() {
            return None;
        }

        let namespace_count = usize::try_from(take_u64(bytes, &mut pos)?).ok()?;
        if namespace_count == 0 || namespace_count > MAX_DEBUG_NAMESPACE_SEGMENTS {
            return None;
        }
        let mut namespace = Vec::with_capacity(namespace_count);
        for _ in 0..namespace_count {
            let segment = take_str(bytes, &mut pos)?;
            if segment.is_empty() {
                return None;
            }
            namespace.push(segment);
        }

        let is_local_to_unit = match take_u64(bytes, &mut pos)? {
            0 => false,
            1 => true,
            _ => return None,
        };
        let is_function_local = match take_u64(bytes, &mut pos)? {
            0 => false,
            1 => true,
            _ => return None,
        };
        if is_function_local && namespace.len() < 2 {
            return None;
        }
        let file = PathBuf::from(take_str(bytes, &mut pos)?);
        let line = i32::try_from(take_u64(bytes, &mut pos)?).ok()?;
        let column = i32::try_from(take_u64(bytes, &mut pos)?).ok()?;
        if file.as_os_str().is_empty() || line <= 0 || column <= 0 {
            return None;
        }

        let ty_bytes = take_bytes(bytes, &mut pos, MAX_DEBUG_ATTRIBUTE_BYTES)?;
        let mut ty_pos = 0;
        let ty = deserialize_debug_type(ty_bytes, &mut ty_pos)?;
        if ty_pos != ty_bytes.len() || pos != bytes.len() {
            return None;
        }

        Some(DebugGlobalVariableInfo {
            name,
            namespace,
            ty,
            declaration: DebugSourcePosition { file, line, column },
            is_local_to_unit,
            is_function_local,
        })
    }

    /// One scalarized fragment of a source variable.
    ///
    /// `offset_bits` and `size_bits` use LLVM's `DW_OP_LLVM_fragment` units and
    /// describe where this storage/value belongs inside the complete source
    /// variable. Fragments with zero size are rejected when decoded.
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct DebugFragment {
        pub offset_bits: u64,
        pub size_bits: u64,
    }

    /// A source variable reconstructed from one scalarized MIR storage/value.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugFragmentVariableInfo {
        pub variable: DebugLocalVariableInfo,
        pub fragment: DebugFragment,
        pub source_scope: Option<u32>,
        pub declaration: Option<DebugSourcePosition>,
    }

    /// An additional whole source variable backed by the same MIR
    /// storage/value as the primary [`DebugLocalVariableInfo`].
    ///
    /// Optimized MIR may deliberately map multiple source bindings to one
    /// local (for example, after reference propagation).  Each binding keeps
    /// its own name, type, argument index, scope, and declaration position and
    /// therefore needs its own `DILocalVariable` rather than being disguised
    /// as a zero-offset projection.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugWholeVariableInfo {
        pub variable: DebugLocalVariableInfo,
        pub source_scope: Option<u32>,
        pub declaration: Option<DebugSourcePosition>,
    }

    /// One operation in a multi-value LLVM debug expression.
    ///
    /// `Arg(N)` selects the Nth operand from the `DIArgList`. The remaining
    /// operations are the small DWARF subset needed to combine addresses and
    /// runtime scalar state without exposing raw expression strings to callers.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum DebugValueExpressionOp {
        Arg(u32),
        ConstU(u64),
        Plus,
        PlusUConst(u64),
        Mul,
        Deref,
        StackValue,
    }

    /// Ordered operations for a multi-value `llvm.dbg.value` location recipe.
    #[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
    pub struct DebugValueExpression {
        pub operations: Vec<DebugValueExpressionOp>,
    }

    impl DebugValueExpression {
        pub fn new(operations: Vec<DebugValueExpressionOp>) -> Self {
            Self { operations }
        }
    }

    /// A source variable whose storage is a supported projection of a MIR local.
    ///
    /// `offset_bytes` is measured from the current address after an optional
    /// leading thin-pointer/reference dereference. The textual LLVM exporter emits
    /// `DW_OP_deref` when `dereference_base` is set, then `DW_OP_plus_uconst` for a
    /// non-zero offset. Dynamic indices, repeated dereferences, slices/fat pointers,
    /// and enum downcasts are intentionally not represented.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugProjectedVariableInfo {
        pub variable: DebugLocalVariableInfo,
        pub dereference_base: bool,
        pub offset_bytes: u64,
        pub source_scope: Option<u32>,
        pub declaration: Option<DebugSourcePosition>,
    }

    /// A source position small enough to carry through cuda-oxide attrs.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugSourcePosition {
        pub file: PathBuf,
        pub line: i32,
        pub column: i32,
    }

    /// Extra scope information rustc records for MIR inlining.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugInlinedScope {
        pub callee_name: String,
        pub callsite: Option<DebugSourcePosition>,
    }

    /// One rustc MIR `SourceScope`, flattened into stable data.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugSourceScope {
        pub id: u32,
        pub parent: Option<u32>,
        pub span: Option<DebugSourcePosition>,
        pub inlined: Option<DebugInlinedScope>,
    }

    /// The original rustc MIR source scope for a statement or terminator span.
    ///
    /// stable MIR currently exposes the span, but not the `SourceScope`, on
    /// statements and terminators. The rustc-codegen bridge records that
    /// pairing before the stable-MIR conversion so instruction `!dbg` scopes
    /// can match the lexical scopes used by local variables.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugSourceScopeLocation {
        pub pos: DebugSourcePosition,
        pub scope: u32,
    }

    /// The source-scope table for one function body.
    #[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
    pub struct DebugSourceScopeMap {
        pub scopes: Vec<DebugSourceScope>,
        pub locations: Vec<DebugSourceScopeLocation>,
    }

    const DEBUG_LOCAL_NAME_KEY: &str = "cuda_oxide_debug_local_name";
    const DEBUG_LOCAL_ARG_KEY: &str = "cuda_oxide_debug_local_arg";
    /// The whole source-local type tree, serialized by [`serialize_debug_type`].
    const DEBUG_LOCAL_TYPE_KEY: &str = "cuda_oxide_debug_local_type";
    const DEBUG_LOCAL_DECL_FILE_KEY: &str = "cuda_oxide_debug_local_decl_file";
    const DEBUG_LOCAL_DECL_LINE_KEY: &str = "cuda_oxide_debug_local_decl_line";
    const DEBUG_LOCAL_DECL_COLUMN_KEY: &str = "cuda_oxide_debug_local_decl_column";
    const DEBUG_LOCAL_SCOPE_KEY: &str = "cuda_oxide_debug_local_scope";
    const DEBUG_GLOBAL_INFO_KEY: &str = "cuda_oxide_debug_global_info";
    const DEBUG_WHOLE_ALIAS_COUNT_KEY: &str = "cuda_oxide_debug_whole_alias_count";
    const DEBUG_PROJECTED_COUNT_KEY: &str = "cuda_oxide_debug_projected_count";
    const DEBUG_FRAGMENT_COUNT_KEY: &str = "cuda_oxide_debug_fragment_count";
    const DEBUG_VALUE_EXPRESSION_KEY: &str = "cuda_oxide_debug_value_expression";
    /// Source-facing name of a function, kept separate from its emitted symbol.
    ///
    /// Non-generic device functions use legalized Rust paths as their physical
    /// symbols and generic functions use Rust's mangled symbol.  Neither is the
    /// name users write in source or pass to a debugger, so the importer carries
    /// that spelling explicitly for `DISubprogram::name`.
    const DEBUG_FUNCTION_NAME_KEY: &str = "cuda_oxide_debug_function_name";
    const DEBUG_SOURCE_SCOPE_COUNT_KEY: &str = "cuda_oxide_debug_scope_count";
    const DEBUG_SOURCE_SCOPE_LOCATION_COUNT_KEY: &str = "cuda_oxide_debug_scope_location_count";
    /// Raw LLVM-dialect symbol of the function that owns a function-local AS3
    /// static. The exporter resolves this to the one real `DISubprogram` used
    /// by that definition; it never creates a scope-only duplicate.
    const DEBUG_GLOBAL_OWNER_FUNCTION_KEY: &str = "cuda_oxide_debug_global_owner_function";
    /// Op-attribute key for the alignment an address computation guarantees.
    /// Lowering-internal: never exported.
    const ADDRESS_ALIGNMENT_KEY: &str = "cuda_oxide_address_alignment";

    /// Stamp the ABI alignment (bytes) onto a memory op.
    pub fn set_op_alignment(ctx: &mut Context, op: Ptr<Operation>, align: u32) {
        let key = Identifier::try_new(OP_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref_mut(ctx).attributes.set(key, AlignmentAttr(align));
    }

    /// Read the ABI alignment (bytes) stamped on a memory op, if any.
    pub fn op_alignment(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
        let key = Identifier::try_new(OP_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref(ctx)
            .attributes
            .get::<AlignmentAttr>(&key)
            .map(|a| a.0)
    }

    /// Stamp the alignment (bytes) that an *address-producing* op guarantees.
    ///
    /// Distinct from [`set_op_alignment`], which states the alignment of a
    /// memory op's own access and is what the exporter prints as `align N`.
    /// This records what a computed address proves about itself, so a later
    /// load through it can state an alignment its own result type does not
    /// know. Nothing exports it: it is consumed during lowering and is inert
    /// on the op that carries it.
    pub fn set_address_alignment(ctx: &mut Context, op: Ptr<Operation>, align: u32) {
        let key = Identifier::try_new(ADDRESS_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref_mut(ctx).attributes.set(key, AlignmentAttr(align));
    }

    /// Read the alignment an address-producing op guarantees, if any.
    pub fn address_alignment(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
        let key = Identifier::try_new(ADDRESS_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref(ctx)
            .attributes
            .get::<AlignmentAttr>(&key)
            .map(|a| a.0)
    }

    /// Attach the source-facing name of a function to a MIR/LLVM function op.
    pub fn set_debug_function_name(ctx: &mut Context, op: Ptr<Operation>, name: &str) {
        set_string_attr(ctx, op, DEBUG_FUNCTION_NAME_KEY, name.to_string());
    }

    /// Read the source-facing name attached to a MIR/LLVM function op.
    pub fn debug_function_name(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
        get_string_attr(ctx, op, DEBUG_FUNCTION_NAME_KEY)
    }

    /// Copy a function's source-facing debug name during MIR-to-LLVM lowering.
    pub fn copy_debug_function_name(ctx: &mut Context, from: Ptr<Operation>, to: Ptr<Operation>) {
        let Some(name) = debug_function_name(ctx, from) else {
            return;
        };
        set_debug_function_name(ctx, to, &name);
    }

    /// Set Pliron's upstream inline-asm side-effect attribute.
    ///
    /// Kept as a cuda-oxide compatibility helper; it no longer owns a private
    /// semantic attribute.
    pub fn set_inline_asm_sideeffect(ctx: &mut Context, op: Ptr<Operation>, sideeffect: bool) {
        let key = Identifier::try_new(LLVM_INLINE_ASM_SIDEEFFECT_KEY.to_string())
            .expect("valid upstream inline asm side-effect identifier");
        op.deref_mut(ctx)
            .attributes
            .set(key, BoolAttr::new(sideeffect));
    }

    /// Read Pliron's upstream inline-asm side-effect attribute.
    pub fn inline_asm_sideeffect(ctx: &Context, op: Ptr<Operation>) -> bool {
        let key = Identifier::try_new(LLVM_INLINE_ASM_SIDEEFFECT_KEY.to_string())
            .expect("valid upstream inline asm side-effect identifier");
        op.deref(ctx)
            .attributes
            .get::<BoolAttr>(&key)
            .map(|attr| bool::from((*attr).clone()))
            .unwrap_or(true)
    }

    /// Set or clear LLVM's upstream `convergent` call-site attribute on inline asm.
    pub fn set_inline_asm_convergent(ctx: &mut Context, op: Ptr<Operation>, convergent: bool) {
        let key = Identifier::try_new(LLVM_INLINE_ASM_ATTRS_KEY.to_string())
            .expect("valid upstream inline asm attrs identifier");
        let mut attrs = op
            .deref(ctx)
            .attributes
            .get::<LlvmAttributesAttr>(&key)
            .map(|attrs| (*attrs).clone())
            .unwrap_or_default();
        if convergent {
            attrs.set("convergent", LlvmAttrValue::Unit);
        } else {
            attrs.remove("convergent");
        }
        if attrs.is_empty() {
            op.deref_mut(ctx).attributes.0.remove(&key);
        } else {
            op.deref_mut(ctx).attributes.set(key, attrs);
        }
    }

    /// Attach source-local debug metadata to a memory slot op.
    pub fn set_debug_local_variable(
        ctx: &mut Context,
        op: Ptr<Operation>,
        info: DebugLocalVariableInfo,
    ) {
        set_string_attr(ctx, op, DEBUG_LOCAL_NAME_KEY, info.name);
        if let Some(arg) = info.argument_index {
            set_string_attr(ctx, op, DEBUG_LOCAL_ARG_KEY, arg.to_string());
        }

        let mut encoded = String::new();
        serialize_debug_type(&info.ty, &mut encoded);
        set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_KEY, encoded);
    }

    /// Read source-local debug metadata from a memory slot op, if present.
    pub fn debug_local_variable(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<DebugLocalVariableInfo> {
        let name = get_string_attr(ctx, op, DEBUG_LOCAL_NAME_KEY)?;
        let argument_index =
            get_string_attr(ctx, op, DEBUG_LOCAL_ARG_KEY).and_then(|arg| arg.parse::<u16>().ok());
        let encoded = get_string_attr(ctx, op, DEBUG_LOCAL_TYPE_KEY)?;
        let ty = deserialize_debug_type(encoded.as_bytes(), &mut 0)?;

        Some(DebugLocalVariableInfo {
            name,
            argument_index,
            ty,
        })
    }

    /// Attach additional whole-variable identities backed by this slot/value.
    pub fn set_debug_whole_variable_aliases(
        ctx: &mut Context,
        op: Ptr<Operation>,
        aliases: &[DebugWholeVariableInfo],
    ) {
        set_string_attr(
            ctx,
            op,
            DEBUG_WHOLE_ALIAS_COUNT_KEY,
            aliases.len().to_string(),
        );

        for (index, info) in aliases.iter().enumerate() {
            set_string_attr(
                ctx,
                op,
                &debug_whole_alias_key(index, "name"),
                info.variable.name.clone(),
            );
            if let Some(argument_index) = info.variable.argument_index {
                set_string_attr(
                    ctx,
                    op,
                    &debug_whole_alias_key(index, "arg"),
                    argument_index.to_string(),
                );
            }

            let mut encoded = String::new();
            serialize_debug_type(&info.variable.ty, &mut encoded);
            set_string_attr(ctx, op, &debug_whole_alias_key(index, "type"), encoded);
            if let Some(source_scope) = info.source_scope {
                set_string_attr(
                    ctx,
                    op,
                    &debug_whole_alias_key(index, "scope"),
                    source_scope.to_string(),
                );
            }
            if let Some(declaration) = &info.declaration {
                set_string_attr(
                    ctx,
                    op,
                    &debug_whole_alias_key(index, "file"),
                    declaration.file.to_string_lossy().into_owned(),
                );
                set_string_attr(
                    ctx,
                    op,
                    &debug_whole_alias_key(index, "line"),
                    declaration.line.to_string(),
                );
                set_string_attr(
                    ctx,
                    op,
                    &debug_whole_alias_key(index, "column"),
                    declaration.column.to_string(),
                );
            }
        }
    }

    /// Read additional whole-variable identities attached to a slot/value.
    /// Malformed entries are skipped individually and the count is bounded.
    pub fn debug_whole_variable_aliases(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Vec<DebugWholeVariableInfo> {
        let count = get_string_attr(ctx, op, DEBUG_WHOLE_ALIAS_COUNT_KEY)
            .and_then(|count| count.parse::<usize>().ok())
            .unwrap_or(0);
        if count > 1024 {
            return Vec::new();
        }

        let mut aliases = Vec::with_capacity(count);
        for index in 0..count {
            let Some(name) = get_string_attr(ctx, op, &debug_whole_alias_key(index, "name")) else {
                continue;
            };
            let argument_index = get_string_attr(ctx, op, &debug_whole_alias_key(index, "arg"))
                .and_then(|arg| arg.parse::<u16>().ok());
            let Some(encoded) = get_string_attr(ctx, op, &debug_whole_alias_key(index, "type"))
            else {
                continue;
            };
            let mut pos = 0;
            let Some(ty) = deserialize_debug_type(encoded.as_bytes(), &mut pos) else {
                continue;
            };
            if pos != encoded.len() {
                continue;
            }
            let source_scope = get_string_attr(ctx, op, &debug_whole_alias_key(index, "scope"))
                .and_then(|scope| scope.parse::<u32>().ok());
            let declaration = debug_whole_alias_declaration(ctx, op, index);

            aliases.push(DebugWholeVariableInfo {
                variable: DebugLocalVariableInfo {
                    name,
                    argument_index,
                    ty,
                },
                source_scope,
                declaration,
            });
        }
        aliases
    }

    /// Attach the source identity and semantic type of a Rust static to an op.
    ///
    /// This is usable on both `mir.global_alloc` and the lowered LLVM global,
    /// which lets the information cross dialect conversion without coupling
    /// either dialect's generated attribute schema to the debug representation.
    pub fn set_debug_global_variable(
        ctx: &mut Context,
        op: Ptr<Operation>,
        info: &DebugGlobalVariableInfo,
    ) {
        if let Some(encoded) = encode_debug_global_info(info) {
            set_string_attr(ctx, op, DEBUG_GLOBAL_INFO_KEY, encoded);
        }
    }

    /// Read source-level debug metadata for a Rust static, if complete.
    ///
    /// Malformed or partial attributes fail closed.
    pub fn debug_global_variable(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<DebugGlobalVariableInfo> {
        let encoded = get_string_attr(ctx, op, DEBUG_GLOBAL_INFO_KEY)?;
        decode_debug_global_info(&encoded)
    }

    /// Associate a function-local debug global with its owning function.
    pub fn set_debug_global_owner_function(ctx: &mut Context, op: Ptr<Operation>, owner: &str) {
        if !owner.is_empty() && owner.len() <= MAX_DEBUG_STRING_BYTES {
            set_string_attr(ctx, op, DEBUG_GLOBAL_OWNER_FUNCTION_KEY, owner.to_string());
        }
    }

    /// Read a bounded, non-empty owner function symbol. Malformed attributes
    /// fail closed so they cannot create a CU-scoped or namespace-scoped alias.
    pub fn debug_global_owner_function(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
        get_string_attr(ctx, op, DEBUG_GLOBAL_OWNER_FUNCTION_KEY)
            .filter(|owner| !owner.is_empty() && owner.len() <= MAX_DEBUG_STRING_BYTES)
    }

    /// Detach a global's debug identity (variable info and owner function).
    ///
    /// Fail-open path for one physical allocation reached under divergent
    /// debug identities: the storage stays, only the optional DWARF
    /// attachment is dropped so it cannot misattribute the allocation.
    pub fn clear_debug_global_identity(ctx: &mut Context, op: Ptr<Operation>) {
        remove_string_attr(ctx, op, DEBUG_GLOBAL_INFO_KEY);
        remove_string_attr(ctx, op, DEBUG_GLOBAL_OWNER_FUNCTION_KEY);
    }

    /// Attach every source variable described by a static projection of this slot.
    pub fn set_debug_projected_variables(
        ctx: &mut Context,
        op: Ptr<Operation>,
        projected: &[DebugProjectedVariableInfo],
    ) {
        set_string_attr(
            ctx,
            op,
            DEBUG_PROJECTED_COUNT_KEY,
            projected.len().to_string(),
        );

        for (index, info) in projected.iter().enumerate() {
            set_string_attr(
                ctx,
                op,
                &debug_projected_key(index, "name"),
                info.variable.name.clone(),
            );
            if let Some(argument_index) = info.variable.argument_index {
                set_string_attr(
                    ctx,
                    op,
                    &debug_projected_key(index, "arg"),
                    argument_index.to_string(),
                );
            }

            let mut encoded = String::new();
            serialize_debug_type(&info.variable.ty, &mut encoded);
            set_string_attr(ctx, op, &debug_projected_key(index, "type"), encoded);
            set_string_attr(
                ctx,
                op,
                &debug_projected_key(index, "deref"),
                u8::from(info.dereference_base).to_string(),
            );
            set_string_attr(
                ctx,
                op,
                &debug_projected_key(index, "offset"),
                info.offset_bytes.to_string(),
            );
            if let Some(source_scope) = info.source_scope {
                set_string_attr(
                    ctx,
                    op,
                    &debug_projected_key(index, "scope"),
                    source_scope.to_string(),
                );
            }
            if let Some(declaration) = &info.declaration {
                set_string_attr(
                    ctx,
                    op,
                    &debug_projected_key(index, "file"),
                    declaration.file.to_string_lossy().into_owned(),
                );
                set_string_attr(
                    ctx,
                    op,
                    &debug_projected_key(index, "line"),
                    declaration.line.to_string(),
                );
                set_string_attr(
                    ctx,
                    op,
                    &debug_projected_key(index, "column"),
                    declaration.column.to_string(),
                );
            }
        }
    }

    /// Read the static-projection source variables attached to a local slot.
    ///
    /// Malformed entries are skipped individually. The count is capped so a bad
    /// internal attribute cannot turn export into an unbounded scan.
    pub fn debug_projected_variables(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Vec<DebugProjectedVariableInfo> {
        let count = get_string_attr(ctx, op, DEBUG_PROJECTED_COUNT_KEY)
            .and_then(|count| count.parse::<usize>().ok())
            .unwrap_or(0);
        if count > 1024 {
            return Vec::new();
        }

        let mut projected = Vec::with_capacity(count);
        for index in 0..count {
            let Some(name) = get_string_attr(ctx, op, &debug_projected_key(index, "name")) else {
                continue;
            };
            let argument_index = get_string_attr(ctx, op, &debug_projected_key(index, "arg"))
                .and_then(|arg| arg.parse::<u16>().ok());
            let Some(encoded) = get_string_attr(ctx, op, &debug_projected_key(index, "type"))
            else {
                continue;
            };
            let mut pos = 0;
            let Some(ty) = deserialize_debug_type(encoded.as_bytes(), &mut pos) else {
                continue;
            };
            if pos != encoded.len() {
                continue;
            }
            let dereference_base = get_string_attr(ctx, op, &debug_projected_key(index, "deref"))
                .is_some_and(|value| value == "1");
            let Some(offset_bytes) =
                get_string_attr(ctx, op, &debug_projected_key(index, "offset"))
                    .and_then(|offset| offset.parse::<u64>().ok())
            else {
                continue;
            };
            let source_scope = get_string_attr(ctx, op, &debug_projected_key(index, "scope"))
                .and_then(|scope| scope.parse::<u32>().ok());
            let declaration = debug_projected_declaration(ctx, op, index);

            projected.push(DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name,
                    argument_index,
                    ty,
                },
                dereference_base,
                offset_bytes,
                source_scope,
                declaration,
            });
        }
        projected
    }

    /// Attach every scalarized source-variable fragment backed by this slot/value.
    pub fn set_debug_fragment_variables(
        ctx: &mut Context,
        op: Ptr<Operation>,
        fragments: &[DebugFragmentVariableInfo],
    ) {
        set_string_attr(
            ctx,
            op,
            DEBUG_FRAGMENT_COUNT_KEY,
            fragments.len().to_string(),
        );

        for (index, info) in fragments.iter().enumerate() {
            set_string_attr(
                ctx,
                op,
                &debug_fragment_key(index, "name"),
                info.variable.name.clone(),
            );
            if let Some(argument_index) = info.variable.argument_index {
                set_string_attr(
                    ctx,
                    op,
                    &debug_fragment_key(index, "arg"),
                    argument_index.to_string(),
                );
            }

            let mut encoded = String::new();
            serialize_debug_type(&info.variable.ty, &mut encoded);
            set_string_attr(ctx, op, &debug_fragment_key(index, "type"), encoded);
            set_string_attr(
                ctx,
                op,
                &debug_fragment_key(index, "offset_bits"),
                info.fragment.offset_bits.to_string(),
            );
            set_string_attr(
                ctx,
                op,
                &debug_fragment_key(index, "size_bits"),
                info.fragment.size_bits.to_string(),
            );
            if let Some(source_scope) = info.source_scope {
                set_string_attr(
                    ctx,
                    op,
                    &debug_fragment_key(index, "scope"),
                    source_scope.to_string(),
                );
            }
            if let Some(declaration) = &info.declaration {
                set_string_attr(
                    ctx,
                    op,
                    &debug_fragment_key(index, "file"),
                    declaration.file.to_string_lossy().into_owned(),
                );
                set_string_attr(
                    ctx,
                    op,
                    &debug_fragment_key(index, "line"),
                    declaration.line.to_string(),
                );
                set_string_attr(
                    ctx,
                    op,
                    &debug_fragment_key(index, "column"),
                    declaration.column.to_string(),
                );
            }
        }
    }

    /// Read scalarized source-variable fragments attached to a slot/value.
    pub fn debug_fragment_variables(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Vec<DebugFragmentVariableInfo> {
        let count = get_string_attr(ctx, op, DEBUG_FRAGMENT_COUNT_KEY)
            .and_then(|count| count.parse::<usize>().ok())
            .unwrap_or(0);
        if count > 1024 {
            return Vec::new();
        }

        let mut fragments = Vec::with_capacity(count);
        for index in 0..count {
            let Some(name) = get_string_attr(ctx, op, &debug_fragment_key(index, "name")) else {
                continue;
            };
            let argument_index = get_string_attr(ctx, op, &debug_fragment_key(index, "arg"))
                .and_then(|arg| arg.parse::<u16>().ok());
            let Some(encoded) = get_string_attr(ctx, op, &debug_fragment_key(index, "type")) else {
                continue;
            };
            let mut pos = 0;
            let Some(ty) = deserialize_debug_type(encoded.as_bytes(), &mut pos) else {
                continue;
            };
            if pos != encoded.len() {
                continue;
            }
            let Some(offset_bits) =
                get_string_attr(ctx, op, &debug_fragment_key(index, "offset_bits"))
                    .and_then(|value| value.parse::<u64>().ok())
            else {
                continue;
            };
            let Some(size_bits) = get_string_attr(ctx, op, &debug_fragment_key(index, "size_bits"))
                .and_then(|value| value.parse::<u64>().ok())
            else {
                continue;
            };
            let Some(end_bits) = offset_bits.checked_add(size_bits) else {
                continue;
            };
            if size_bits == 0 || end_bits > ty.size_bits() {
                continue;
            }
            let source_scope = get_string_attr(ctx, op, &debug_fragment_key(index, "scope"))
                .and_then(|scope| scope.parse::<u32>().ok());
            let declaration = debug_fragment_declaration(ctx, op, index);

            fragments.push(DebugFragmentVariableInfo {
                variable: DebugLocalVariableInfo {
                    name,
                    argument_index,
                    ty,
                },
                fragment: DebugFragment {
                    offset_bits,
                    size_bits,
                },
                source_scope,
                declaration,
            });
        }
        fragments
    }

    /// Rust-local provenance for the post-optimization local-memory diagnostic.
    ///
    /// `mir-importer` attaches this to the `mir.alloca` of every named Rust
    /// source local, `mir-lower` copies it to the LLVM alloca, and the textual
    /// exporter folds it into the alloca's SSA value name so it survives the
    /// external `opt` binary exactly as long as the allocation itself does.
    /// The attribute is a first-class IR citizen inside both dialects; only the
    /// exported SSA name uses a string encoding, because the value name is the
    /// sole channel `opt` reliably preserves on surviving instructions.
    #[pliron_attr(name = "llvm.local_memory_provenance", format, verifier = "succ")]
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LocalMemoryProvenanceAttr {
        /// Index of the Rust MIR local backing the allocation.
        pub local_index: u64,
        /// ABI size of the local in bytes (0 when the layout is unavailable).
        pub size_bytes: u64,
        /// Source binding name of the local.
        pub binding_name: StringAttr,
        /// Compact source-level spelling of the local's type.
        pub type_name: StringAttr,
    }

    /// Op-attribute key for [`LocalMemoryProvenanceAttr`] on `mir.alloca` and
    /// `llvm.alloca`.
    const LOCAL_MEMORY_PROVENANCE_KEY: &str = "cuda_oxide_local_memory_provenance";

    /// Attach Rust-local provenance to a stack-slot op.
    pub fn set_local_memory_provenance(
        ctx: &mut Context,
        op: Ptr<Operation>,
        provenance: LocalMemoryProvenanceAttr,
    ) {
        let key = Identifier::try_new(LOCAL_MEMORY_PROVENANCE_KEY.to_string())
            .expect("valid local-memory provenance attribute key");
        op.deref_mut(ctx).attributes.set(key, provenance);
    }

    /// Read Rust-local provenance from a stack-slot op, if present.
    pub fn local_memory_provenance(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<LocalMemoryProvenanceAttr> {
        let key = Identifier::try_new(LOCAL_MEMORY_PROVENANCE_KEY.to_string())
            .expect("valid local-memory provenance attribute key");
        op.deref(ctx)
            .attributes
            .get::<LocalMemoryProvenanceAttr>(&key)
            .cloned()
    }

    /// Attach the MIR source-scope id that owns this source local.
    pub fn set_debug_local_source_scope(ctx: &mut Context, op: Ptr<Operation>, scope: u32) {
        set_string_attr(ctx, op, DEBUG_LOCAL_SCOPE_KEY, scope.to_string());
    }

    /// Read the MIR source-scope id that owns this source local.
    pub fn debug_local_source_scope(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
        get_string_attr(ctx, op, DEBUG_LOCAL_SCOPE_KEY).and_then(|scope| scope.parse().ok())
    }

    /// Attach a typed multi-value location expression to a debug marker.
    pub fn set_debug_value_expression(
        ctx: &mut Context,
        op: Ptr<Operation>,
        expression: &DebugValueExpression,
    ) {
        set_string_attr(
            ctx,
            op,
            DEBUG_VALUE_EXPRESSION_KEY,
            encode_debug_value_expression(expression),
        );
    }

    /// Read a typed multi-value location expression from a debug marker.
    pub fn debug_value_expression(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<DebugValueExpression> {
        let encoded = get_string_attr(ctx, op, DEBUG_VALUE_EXPRESSION_KEY)?;
        decode_debug_value_expression(&encoded)
    }

    fn encode_debug_value_expression(expression: &DebugValueExpression) -> String {
        let mut encoded = String::from("v1");
        for operation in &expression.operations {
            encoded.push(' ');
            match operation {
                DebugValueExpressionOp::Arg(index) => {
                    encoded.push_str("arg:");
                    encoded.push_str(&index.to_string());
                }
                DebugValueExpressionOp::ConstU(value) => {
                    encoded.push_str("constu:");
                    encoded.push_str(&value.to_string());
                }
                DebugValueExpressionOp::Plus => encoded.push_str("plus"),
                DebugValueExpressionOp::PlusUConst(value) => {
                    encoded.push_str("plus_uconst:");
                    encoded.push_str(&value.to_string());
                }
                DebugValueExpressionOp::Mul => encoded.push_str("mul"),
                DebugValueExpressionOp::Deref => encoded.push_str("deref"),
                DebugValueExpressionOp::StackValue => encoded.push_str("stack_value"),
            }
        }
        encoded
    }

    fn decode_debug_value_expression(encoded: &str) -> Option<DebugValueExpression> {
        let mut tokens = encoded.split_ascii_whitespace();
        if tokens.next()? != "v1" {
            return None;
        }

        let mut operations = Vec::new();
        for token in tokens {
            let operation = if let Some(index) = token.strip_prefix("arg:") {
                DebugValueExpressionOp::Arg(index.parse::<u32>().ok()?)
            } else if let Some(value) = token.strip_prefix("constu:") {
                DebugValueExpressionOp::ConstU(value.parse::<u64>().ok()?)
            } else if token == "plus" {
                DebugValueExpressionOp::Plus
            } else if let Some(value) = token.strip_prefix("plus_uconst:") {
                DebugValueExpressionOp::PlusUConst(value.parse::<u64>().ok()?)
            } else if token == "mul" {
                DebugValueExpressionOp::Mul
            } else if token == "deref" {
                DebugValueExpressionOp::Deref
            } else if token == "stack_value" {
                DebugValueExpressionOp::StackValue
            } else {
                return None;
            };
            operations.push(operation);
        }
        Some(DebugValueExpression { operations })
    }

    /// Attach a function's MIR source-scope table.
    pub fn set_debug_source_scope_map(
        ctx: &mut Context,
        op: Ptr<Operation>,
        map: &DebugSourceScopeMap,
    ) {
        // The reader (`debug_source_scope_map`) reconstructs scope ids as
        // `0..count`, so the writer's per-scope attr keys must use exactly those
        // ids. rustc's `SourceScope` indices are dense `0..len`, which makes this
        // hold today. Assert it so a future sparse/reordered producer fails
        // loudly here instead of silently mislabeling parent/scope links.
        debug_assert!(
            map.scopes
                .iter()
                .enumerate()
                .all(|(idx, scope)| scope.id as usize == idx),
            "DebugSourceScopeMap scope ids must be dense 0..len to round-trip"
        );
        set_string_attr(
            ctx,
            op,
            DEBUG_SOURCE_SCOPE_COUNT_KEY,
            map.scopes.len().to_string(),
        );
        set_string_attr(
            ctx,
            op,
            DEBUG_SOURCE_SCOPE_LOCATION_COUNT_KEY,
            map.locations.len().to_string(),
        );

        for scope in &map.scopes {
            let id = scope.id;
            if let Some(parent) = scope.parent {
                set_string_attr(ctx, op, &debug_scope_key(id, "parent"), parent.to_string());
            }
            if let Some(span) = &scope.span {
                set_debug_position_attrs(ctx, op, id, "span", span);
            }
            if let Some(inlined) = &scope.inlined {
                set_string_attr(
                    ctx,
                    op,
                    &debug_scope_key(id, "callee"),
                    inlined.callee_name.clone(),
                );
                if let Some(callsite) = &inlined.callsite {
                    set_debug_position_attrs(ctx, op, id, "callsite", callsite);
                }
            }
        }

        for (idx, location) in map.locations.iter().enumerate() {
            set_string_attr(
                ctx,
                op,
                &debug_scope_location_key(idx, "scope"),
                location.scope.to_string(),
            );
            set_debug_scope_location_position_attrs(ctx, op, idx, &location.pos);
        }
    }

    /// Read a function's MIR source-scope table.
    pub fn debug_source_scope_map(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<DebugSourceScopeMap> {
        let count = get_string_attr(ctx, op, DEBUG_SOURCE_SCOPE_COUNT_KEY)?
            .parse()
            .ok()?;
        let mut scopes = Vec::with_capacity(count);

        for id in 0..count as u32 {
            let parent = get_string_attr(ctx, op, &debug_scope_key(id, "parent"))
                .and_then(|v| v.parse().ok());
            let span = debug_position_attrs(ctx, op, id, "span");
            let inlined = get_string_attr(ctx, op, &debug_scope_key(id, "callee")).map(|name| {
                DebugInlinedScope {
                    callee_name: name,
                    callsite: debug_position_attrs(ctx, op, id, "callsite"),
                }
            });
            scopes.push(DebugSourceScope {
                id,
                parent,
                span,
                inlined,
            });
        }

        let location_count = get_string_attr(ctx, op, DEBUG_SOURCE_SCOPE_LOCATION_COUNT_KEY)
            .and_then(|count| count.parse().ok())
            .unwrap_or(0);
        let mut locations = Vec::with_capacity(location_count);

        for idx in 0..location_count {
            let scope = get_string_attr(ctx, op, &debug_scope_location_key(idx, "scope"))
                .and_then(|v| v.parse().ok())?;
            let pos = debug_scope_location_position_attrs(ctx, op, idx)?;
            locations.push(DebugSourceScopeLocation { pos, scope });
        }

        Some(DebugSourceScopeMap { scopes, locations })
    }

    /// Copy debug source-scope attrs from one operation to another.
    pub fn copy_debug_source_scope_map(
        ctx: &mut Context,
        from: Ptr<Operation>,
        to: Ptr<Operation>,
    ) {
        let Some(map) = debug_source_scope_map(ctx, from) else {
            return;
        };
        set_debug_source_scope_map(ctx, to, &map);
    }

    /// Read an optional source declaration location for a debug local.
    ///
    /// Promoted `dbg.value` records have two useful locations: the operation
    /// location where the value is current, and the source declaration location
    /// for the `DILocalVariable`. This helper returns the latter when it was
    /// preserved during MIR mem2reg promotion.
    pub fn debug_local_declaration_location(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<(PathBuf, SourcePosition)> {
        let file = PathBuf::from(get_string_attr(ctx, op, DEBUG_LOCAL_DECL_FILE_KEY)?);
        let line = get_string_attr(ctx, op, DEBUG_LOCAL_DECL_LINE_KEY)?
            .parse()
            .ok()?;
        let column = get_string_attr(ctx, op, DEBUG_LOCAL_DECL_COLUMN_KEY)?
            .parse()
            .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }

        Some((file, SourcePosition { line, column }))
    }

    /// Attach the source declaration location for a debug local.
    pub fn set_debug_local_declaration_location(
        ctx: &mut Context,
        op: Ptr<Operation>,
        file: PathBuf,
        line: i32,
        column: i32,
    ) {
        set_string_attr(
            ctx,
            op,
            DEBUG_LOCAL_DECL_FILE_KEY,
            file.to_string_lossy().into_owned(),
        );
        set_string_attr(ctx, op, DEBUG_LOCAL_DECL_LINE_KEY, line.to_string());
        set_string_attr(ctx, op, DEBUG_LOCAL_DECL_COLUMN_KEY, column.to_string());
    }

    fn set_debug_position_attrs(
        ctx: &mut Context,
        op: Ptr<Operation>,
        scope: u32,
        prefix: &str,
        pos: &DebugSourcePosition,
    ) {
        set_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_file")),
            pos.file.to_string_lossy().into_owned(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_line")),
            pos.line.to_string(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_column")),
            pos.column.to_string(),
        );
    }

    fn debug_position_attrs(
        ctx: &Context,
        op: Ptr<Operation>,
        scope: u32,
        prefix: &str,
    ) -> Option<DebugSourcePosition> {
        let file = PathBuf::from(get_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_file")),
        )?);
        let line = get_string_attr(ctx, op, &debug_scope_key(scope, &format!("{prefix}_line")))?
            .parse()
            .ok()?;
        let column = get_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_column")),
        )?
        .parse()
        .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }

        Some(DebugSourcePosition { file, line, column })
    }

    fn set_debug_scope_location_position_attrs(
        ctx: &mut Context,
        op: Ptr<Operation>,
        idx: usize,
        pos: &DebugSourcePosition,
    ) {
        set_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "file"),
            pos.file.to_string_lossy().into_owned(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "line"),
            pos.line.to_string(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "column"),
            pos.column.to_string(),
        );
    }

    fn debug_scope_location_position_attrs(
        ctx: &Context,
        op: Ptr<Operation>,
        idx: usize,
    ) -> Option<DebugSourcePosition> {
        let file = PathBuf::from(get_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "file"),
        )?);
        let line = get_string_attr(ctx, op, &debug_scope_location_key(idx, "line"))?
            .parse()
            .ok()?;
        let column = get_string_attr(ctx, op, &debug_scope_location_key(idx, "column"))?
            .parse()
            .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }

        Some(DebugSourcePosition { file, line, column })
    }

    fn debug_projected_key(index: usize, field: &str) -> String {
        format!("cuda_oxide_debug_projected_{index}_{field}")
    }

    fn debug_whole_alias_key(index: usize, field: &str) -> String {
        format!("cuda_oxide_debug_whole_alias_{index}_{field}")
    }

    fn debug_fragment_key(index: usize, field: &str) -> String {
        format!("cuda_oxide_debug_fragment_{index}_{field}")
    }

    fn debug_projected_declaration(
        ctx: &Context,
        op: Ptr<Operation>,
        index: usize,
    ) -> Option<DebugSourcePosition> {
        let file = PathBuf::from(get_string_attr(
            ctx,
            op,
            &debug_projected_key(index, "file"),
        )?);
        let line = get_string_attr(ctx, op, &debug_projected_key(index, "line"))?
            .parse()
            .ok()?;
        let column = get_string_attr(ctx, op, &debug_projected_key(index, "column"))?
            .parse()
            .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }
        Some(DebugSourcePosition { file, line, column })
    }

    fn debug_whole_alias_declaration(
        ctx: &Context,
        op: Ptr<Operation>,
        index: usize,
    ) -> Option<DebugSourcePosition> {
        let file = PathBuf::from(get_string_attr(
            ctx,
            op,
            &debug_whole_alias_key(index, "file"),
        )?);
        let line = get_string_attr(ctx, op, &debug_whole_alias_key(index, "line"))?
            .parse()
            .ok()?;
        let column = get_string_attr(ctx, op, &debug_whole_alias_key(index, "column"))?
            .parse()
            .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }
        Some(DebugSourcePosition { file, line, column })
    }

    fn debug_fragment_declaration(
        ctx: &Context,
        op: Ptr<Operation>,
        index: usize,
    ) -> Option<DebugSourcePosition> {
        let file = PathBuf::from(get_string_attr(
            ctx,
            op,
            &debug_fragment_key(index, "file"),
        )?);
        let line = get_string_attr(ctx, op, &debug_fragment_key(index, "line"))?
            .parse()
            .ok()?;
        let column = get_string_attr(ctx, op, &debug_fragment_key(index, "column"))?
            .parse()
            .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }
        Some(DebugSourcePosition { file, line, column })
    }

    fn debug_scope_key(scope: u32, field: &str) -> String {
        format!("cuda_oxide_debug_scope_{scope}_{field}")
    }

    fn debug_scope_location_key(idx: usize, field: &str) -> String {
        format!("cuda_oxide_debug_scope_location_{idx}_{field}")
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

    /// LLVM debug-value marker used by the textual exporter.
    ///
    /// This is not a runtime instruction. It lowers to an `llvm.dbg.value`
    /// intrinsic call that tells LLVM/DWARF where a source local lives after a
    /// MIR stack slot has been promoted to an SSA value.
    #[pliron_op(
        name = "llvm.dbg_value",
        format,
        interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<0>]
    )]
    pub struct DebugValueOp;

    impl DebugValueOp {
        pub fn new(ctx: &mut Context, value: Value) -> Self {
            let op = Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![value],
                vec![],
                0,
            );
            DebugValueOp { op }
        }

        pub fn value(&self, ctx: &Context) -> Value {
            self.get_operation().deref(ctx).get_operand(0)
        }
    }

    impl Verify for DebugValueOp {
        fn verify(&self, _ctx: &Context) -> Result<(), Error> {
            Ok(())
        }
    }

    /// LLVM multi-value debug marker used by the textual exporter.
    ///
    /// The operands form the ordered `DIArgList`; a `DebugValueExpression`
    /// attached to the op selects and combines them with `DW_OP_LLVM_arg`.
    #[pliron_op(
        name = "llvm.dbg_value_list",
        format,
        interfaces = [NResultsInterface<0>]
    )]
    pub struct DebugValueListOp;

    impl DebugValueListOp {
        pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
            let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0);
            DebugValueListOp { op }
        }

        pub fn values(&self, ctx: &Context) -> Vec<Value> {
            self.get_operation().deref(ctx).operands().collect()
        }
    }

    impl Verify for DebugValueListOp {
        fn verify(&self, _ctx: &Context) -> Result<(), Error> {
            Ok(())
        }
    }

    /// Alignment helpers re-homed from the pre-migration local `GlobalOp`.
    /// Upstream `GlobalOp` carries type/linkage/addrspace but no alignment, so
    /// we keep the alignment in the op's generic attribute dictionary. Address
    /// space uses upstream's native `address_space` / `set_address_space`.
    pub trait GlobalOpExt {
        /// Build a `GlobalOp` carrying an explicit alignment (bytes).
        fn new_with_alignment(
            ctx: &mut Context,
            name: Identifier,
            ty: TypeHandle,
            alignment: u64,
        ) -> Self;
        /// Read the explicit alignment (bytes), if one was set.
        fn get_alignment(&self, ctx: &Context) -> Option<u64>;
        /// Attach lowered Rust static initializer bytes to this global.
        fn set_initializer_hex(&self, ctx: &mut Context, hex: &str);
        /// Read lowered Rust static initializer bytes, encoded as hex.
        fn initializer_hex(&self, ctx: &Context) -> Option<String>;
        /// Attach the stable rustc global key represented by this LLVM global.
        fn set_source_global_key(&self, ctx: &mut Context, key: &str);
        /// Read the stable rustc global key represented by this LLVM global.
        fn source_global_key(&self, ctx: &Context) -> Option<String>;
        /// Attach serialized initializer relocation metadata.
        fn set_initializer_relocations(&self, ctx: &mut Context, encoded: &str);
        /// Read serialized initializer relocation metadata.
        fn initializer_relocations(&self, ctx: &Context) -> Option<String>;
        /// Attach the Rust path of the shared-memory `static` this global came from.
        ///
        /// Descriptive only: the exporter renders it as a comment above the
        /// global and nothing in code generation consumes it.
        fn set_shared_source_name(&self, ctx: &mut Context, source_name: &str);
        /// Read the Rust path of the shared-memory `static` this global came from.
        fn shared_source_name(&self, ctx: &Context) -> Option<String>;
        /// Keep this global alive through LLVM/NVVM internalization and linking.
        fn mark_retained(&self, ctx: &mut Context);
        /// Whether this global was explicitly marked as externally consumed.
        fn is_retained(&self, ctx: &Context) -> bool;
    }

    impl GlobalOpExt for GlobalOp {
        fn new_with_alignment(
            ctx: &mut Context,
            name: Identifier,
            ty: TypeHandle,
            alignment: u64,
        ) -> Self {
            let op = GlobalOp::new(ctx, name, ty);
            let key =
                Identifier::try_new(GLOBAL_ALIGNMENT_KEY.to_string()).expect("valid identifier");
            op.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, AlignmentAttr(alignment as u32));
            op
        }

        fn get_alignment(&self, ctx: &Context) -> Option<u64> {
            let key =
                Identifier::try_new(GLOBAL_ALIGNMENT_KEY.to_string()).expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<AlignmentAttr>(&key)
                .map(|a| a.0 as u64)
        }

        fn set_initializer_hex(&self, ctx: &mut Context, hex: &str) {
            let key = Identifier::try_new(GLOBAL_INITIALIZER_HEX_KEY.to_string())
                .expect("valid identifier");
            self.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, StringAttr::new(hex.to_string()));
        }

        fn initializer_hex(&self, ctx: &Context) -> Option<String> {
            let key = Identifier::try_new(GLOBAL_INITIALIZER_HEX_KEY.to_string())
                .expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<StringAttr>(&key)
                .map(|attr| String::from((*attr).clone()))
        }

        fn set_source_global_key(&self, ctx: &mut Context, source_key: &str) {
            let key = Identifier::try_new(GLOBAL_SOURCE_KEY.to_string()).expect("valid identifier");
            self.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, StringAttr::new(source_key.to_string()));
        }

        fn source_global_key(&self, ctx: &Context) -> Option<String> {
            let key = Identifier::try_new(GLOBAL_SOURCE_KEY.to_string()).expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<StringAttr>(&key)
                .map(|attr| String::from((*attr).clone()))
        }

        fn set_initializer_relocations(&self, ctx: &mut Context, encoded: &str) {
            let key = Identifier::try_new(GLOBAL_INITIALIZER_RELOCATIONS_KEY.to_string())
                .expect("valid identifier");
            self.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, StringAttr::new(encoded.to_string()));
        }

        fn initializer_relocations(&self, ctx: &Context) -> Option<String> {
            let key = Identifier::try_new(GLOBAL_INITIALIZER_RELOCATIONS_KEY.to_string())
                .expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<StringAttr>(&key)
                .map(|attr| String::from((*attr).clone()))
        }

        fn set_shared_source_name(&self, ctx: &mut Context, source_name: &str) {
            let key = Identifier::try_new(GLOBAL_SHARED_SOURCE_NAME_KEY.to_string())
                .expect("valid identifier");
            self.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, StringAttr::new(source_name.to_string()));
        }

        fn shared_source_name(&self, ctx: &Context) -> Option<String> {
            let key = Identifier::try_new(GLOBAL_SHARED_SOURCE_NAME_KEY.to_string())
                .expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<StringAttr>(&key)
                .map(|attr| String::from((*attr).clone()))
        }

        fn mark_retained(&self, ctx: &mut Context) {
            let key =
                Identifier::try_new(GLOBAL_RETAINED_KEY.to_string()).expect("valid identifier");
            self.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, pliron::builtin::attributes::UnitAttr);
        }

        fn is_retained(&self, ctx: &Context) -> bool {
            let key =
                Identifier::try_new(GLOBAL_RETAINED_KEY.to_string()).expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<pliron::builtin::attributes::UnitAttr>(&key)
                .is_some()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            DebugEnumDiscriminant, DebugEnumVariant, DebugGlobalVariableInfo, DebugLocalTypeKind,
            DebugSourcePosition, DebugTypeMember, GlobalInitializerRelocation,
            MAX_DEBUG_NAMESPACE_SEGMENTS, MAX_DEBUG_STRING_BYTES, MAX_DEBUG_TYPE_DEPTH,
            decode_debug_global_info, decode_global_initializer_relocations,
            deserialize_debug_type, encode_debug_global_info,
            encode_global_initializer_relocations, serialize_debug_type,
        };
        use std::path::PathBuf;

        fn round_trip(ty: &DebugLocalTypeKind) -> DebugLocalTypeKind {
            let mut encoded = String::new();
            serialize_debug_type(ty, &mut encoded);
            let mut pos = 0;
            let decoded =
                deserialize_debug_type(encoded.as_bytes(), &mut pos).expect("decode succeeds");
            assert_eq!(pos, encoded.len(), "decoder consumed the whole blob");
            decoded
        }

        #[test]
        fn round_trips_nested_composites() {
            // A struct whose members include a basic, a pointer, a fixed array,
            // and a nested tuple-as-struct: exercises every variant + recursion.
            let ty = DebugLocalTypeKind::Struct {
                name: "Frame<'_, u64>".to_string(),
                size_bits: 256,
                members: vec![
                    DebugTypeMember {
                        name: "len".to_string(),
                        offset_bits: 0,
                        ty: DebugLocalTypeKind::Basic {
                            name: "usize".to_string(),
                            size_bits: 64,
                            encoding: "DW_ATE_unsigned",
                        },
                    },
                    DebugTypeMember {
                        name: "data".to_string(),
                        offset_bits: 64,
                        ty: DebugLocalTypeKind::TypedPointer {
                            name: "*mut u64".to_string(),
                            size_bits: 64,
                            pointee: Box::new(DebugLocalTypeKind::Basic {
                                name: "u64".to_string(),
                                size_bits: 64,
                                encoding: "DW_ATE_unsigned",
                            }),
                        },
                    },
                    DebugTypeMember {
                        name: "lanes".to_string(),
                        offset_bits: 128,
                        ty: DebugLocalTypeKind::Array {
                            name: "[u32; 2]".to_string(),
                            size_bits: 64,
                            element: Box::new(DebugLocalTypeKind::Basic {
                                name: "u32".to_string(),
                                size_bits: 32,
                                encoding: "DW_ATE_signed",
                            }),
                            count: 2,
                        },
                    },
                ],
            };
            assert_eq!(round_trip(&ty), ty);
        }

        #[test]
        fn round_trips_enum_variant_metadata() {
            let ty = DebugLocalTypeKind::Enum {
                name: "Option<&u32>".to_string(),
                size_bits: 64,
                discriminant: Some(DebugEnumDiscriminant {
                    offset_bits: 0,
                    ty: Box::new(DebugLocalTypeKind::Basic {
                        name: "usize".to_string(),
                        size_bits: 64,
                        encoding: "DW_ATE_unsigned",
                    }),
                }),
                variants: vec![
                    DebugEnumVariant {
                        name: "None".to_string(),
                        discriminant: Some(0),
                        members: vec![],
                    },
                    DebugEnumVariant {
                        name: "Some".to_string(),
                        discriminant: None,
                        members: vec![DebugTypeMember {
                            name: "0".to_string(),
                            offset_bits: 0,
                            ty: DebugLocalTypeKind::TypedPointer {
                                name: "&u32".to_string(),
                                size_bits: 64,
                                pointee: Box::new(DebugLocalTypeKind::Basic {
                                    name: "u32".to_string(),
                                    size_bits: 32,
                                    encoding: "DW_ATE_unsigned",
                                }),
                            },
                        }],
                    },
                ],
            };

            assert_eq!(round_trip(&ty), ty);
        }

        #[test]
        fn round_trips_names_with_delimiters() {
            // Length-prefixing must survive names containing spaces/digits/braces.
            let ty = DebugLocalTypeKind::TypedPointer {
                name: "&[(u32, u32); 4] {x: 1}".to_string(),
                size_bits: 64,
                pointee: Box::new(DebugLocalTypeKind::Basic {
                    name: "u8".to_string(),
                    size_bits: 8,
                    encoding: "DW_ATE_unsigned",
                }),
            };
            assert_eq!(round_trip(&ty), ty);
        }

        #[test]
        fn round_trips_nested_pointer_pointees() {
            let ty = DebugLocalTypeKind::TypedPointer {
                name: "*const *mut i32".to_string(),
                size_bits: 64,
                pointee: Box::new(DebugLocalTypeKind::TypedPointer {
                    name: "*mut i32".to_string(),
                    size_bits: 64,
                    pointee: Box::new(DebugLocalTypeKind::Basic {
                        name: "i32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_signed",
                    }),
                }),
            };
            assert_eq!(round_trip(&ty), ty);
        }

        #[test]
        fn opaque_pointer_preserves_the_legacy_encoding() {
            let ty = DebugLocalTypeKind::Pointer {
                name: "*mut i32".to_string(),
                size_bits: 64,
            };
            let mut encoded = String::new();
            serialize_debug_type(&ty, &mut encoded);
            assert_eq!(encoded, "p64 8 *mut i32");
            assert_eq!(round_trip(&ty), ty);
        }

        #[test]
        fn rejects_typed_pointer_without_a_serialized_pointee() {
            // Unlike the legacy `p` record, the new `t` record requires a
            // recursive pointee type and must reject a truncated payload.
            let encoded = b"t64 8 *mut i32";
            let mut pos = 0;
            assert!(deserialize_debug_type(encoded, &mut pos).is_none());
        }

        #[test]
        fn decoder_rejects_typed_pointer_with_forbidden_descendants() {
            let invalid = [
                // Direct legacy opaque pointer descendant.
                "t64 8 *const _p64 8 *mut i32",
                // The opaque pointer is hidden beneath a fixed array.
                "t64 8 *const _a64 8 [ptr; 1]1 p64 8 *mut i32",
                // Source composites are outside the acyclic typed subset.
                "t64 8 *const _s64 3 Foo0 ",
                "t64 8 *const _e64 3 Foo0 0 ",
                // Composite descendants cannot be hidden below an otherwise
                // accepted array or typed-pointer node either.
                "t64 8 *const _a64 8 [Foo; 1]1 s64 3 Foo0 ",
                "t64 15 *const *const _t64 8 *const _e64 3 Foo0 0 ",
            ];

            for encoded in invalid {
                let mut pos = 0;
                assert!(
                    deserialize_debug_type(encoded.as_bytes(), &mut pos).is_none(),
                    "accepted invalid typed pointer payload {encoded:?}"
                );
            }
        }

        #[test]
        #[should_panic(
            expected = "typed pointer pointee must contain only basic, typed-pointer, or array nodes"
        )]
        fn serializer_rejects_hand_built_typed_pointer_with_opaque_descendant() {
            let invalid = DebugLocalTypeKind::TypedPointer {
                name: "*const [*mut _; 1]".to_string(),
                size_bits: 64,
                pointee: Box::new(DebugLocalTypeKind::Array {
                    name: "[*mut _; 1]".to_string(),
                    size_bits: 64,
                    element: Box::new(DebugLocalTypeKind::Pointer {
                        name: "*mut _".to_string(),
                        size_bits: 64,
                    }),
                    count: 1,
                }),
            };
            let mut encoded = String::new();
            serialize_debug_type(&invalid, &mut encoded);
        }

        #[test]
        fn round_trips_global_initializer_relocations() {
            let relocations = vec![
                GlobalInitializerRelocation {
                    source_offset: 0,
                    width_bytes: 8,
                    target_address_space: 1,
                    target_addend: 0,
                    target_key: "ordinary static with spaces".to_string(),
                },
                GlobalInitializerRelocation {
                    source_offset: 16,
                    width_bytes: 8,
                    target_address_space: 4,
                    target_addend: 24,
                    target_key: "_ZN4test6TARGET17h0123456789abcdefE".to_string(),
                },
            ];
            let encoded = encode_global_initializer_relocations(&relocations);
            assert_eq!(
                decode_global_initializer_relocations(&encoded).expect("decode succeeds"),
                relocations
            );
        }

        #[test]
        fn rejects_truncated_global_initializer_relocations() {
            let relocation = GlobalInitializerRelocation {
                source_offset: 0,
                width_bytes: 8,
                target_address_space: 1,
                target_addend: 0,
                target_key: "target".to_string(),
            };
            let mut encoded = encode_global_initializer_relocations(&[relocation]);
            encoded.pop();
            assert!(decode_global_initializer_relocations(&encoded).is_err());
        }

        #[test]
        fn rejects_truncated_blob() {
            let ty = DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            };
            let mut encoded = String::new();
            serialize_debug_type(&ty, &mut encoded);
            encoded.truncate(encoded.len() - 1);
            let mut pos = 0;
            assert!(deserialize_debug_type(encoded.as_bytes(), &mut pos).is_none());
        }

        #[test]
        fn global_debug_identity_round_trips_structured_namespace_and_visibility() {
            let info = DebugGlobalVariableInfo {
                name: "COUNTER with spaces".to_string(),
                namespace: vec![
                    "collision_crate".to_string(),
                    "module with spaces".to_string(),
                    "function".to_string(),
                ],
                ty: DebugLocalTypeKind::Basic {
                    name: "u32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_unsigned",
                },
                declaration: DebugSourcePosition {
                    file: PathBuf::from("/tmp/path with spaces/kernel.rs"),
                    line: 41,
                    column: 9,
                },
                is_local_to_unit: false,
                is_function_local: true,
            };
            let encoded = encode_debug_global_info(&info).expect("valid identity encodes");
            assert_eq!(decode_debug_global_info(&encoded), Some(info));
        }

        #[test]
        fn global_debug_identity_rejects_malformed_namespace_and_visibility() {
            assert!(
                decode_debug_global_info("v2 1 X0 ").is_none(),
                "an empty namespace must not degrade to scope:null"
            );
            assert!(
                decode_debug_global_info("v2 1 X129 ").is_none(),
                "namespace allocation must be bounded"
            );
            assert!(
                decode_debug_global_info("v2 1 X1 1 n2 ").is_none(),
                "visibility accepts only the exact 0/1 encoding"
            );
            assert!(
                decode_debug_global_info("v2 1 X2 1 c1 f0 2 ").is_none(),
                "function-local accepts only the exact 0/1 encoding"
            );

            let info = DebugGlobalVariableInfo {
                name: "X".to_string(),
                namespace: vec!["crate".to_string()],
                ty: DebugLocalTypeKind::Basic {
                    name: "u8".to_string(),
                    size_bits: 8,
                    encoding: "DW_ATE_unsigned",
                },
                declaration: DebugSourcePosition {
                    file: PathBuf::from("/tmp/x.rs"),
                    line: 1,
                    column: 1,
                },
                is_local_to_unit: true,
                is_function_local: false,
            };
            let mut trailing = encode_debug_global_info(&info).expect("valid identity encodes");
            trailing.push('x');
            assert!(decode_debug_global_info(&trailing).is_none());

            let mut truncated = encode_debug_global_info(&info).expect("valid identity encodes");
            truncated.pop();
            assert!(decode_debug_global_info(&truncated).is_none());

            let mut too_many_namespaces = info.clone();
            too_many_namespaces.namespace =
                vec!["scope".to_string(); MAX_DEBUG_NAMESPACE_SEGMENTS + 1];
            assert!(encode_debug_global_info(&too_many_namespaces).is_none());

            let mut oversized_name = info.clone();
            oversized_name.name = "x".repeat(MAX_DEBUG_STRING_BYTES + 1);
            assert!(encode_debug_global_info(&oversized_name).is_none());

            let mut missing_function_scope = info.clone();
            missing_function_scope.is_function_local = true;
            assert!(
                encode_debug_global_info(&missing_function_scope).is_none(),
                "a function-local global needs crate plus function scope"
            );

            let mut too_deep = DebugLocalTypeKind::Basic {
                name: "u8".to_string(),
                size_bits: 8,
                encoding: "DW_ATE_unsigned",
            };
            for depth in 0..=MAX_DEBUG_TYPE_DEPTH {
                too_deep = DebugLocalTypeKind::Array {
                    name: format!("depth_{depth}"),
                    size_bits: 8,
                    element: Box::new(too_deep),
                    count: 1,
                };
            }
            let mut excessive_nesting = info;
            excessive_nesting.ty = too_deep;
            assert!(encode_debug_global_info(&excessive_nesting).is_none());
        }
    }
}

/// LLVM op-interfaces, re-exported from pliron-llvm.
pub mod op_interfaces {
    pub use pliron_llvm::op_interfaces::*;
}

use pliron::builtin::attributes::FPHalfAttr;
use pliron::utils::apfloat::{Float, Half};

/// Build an `FPHalfAttr` from a raw 16-bit IEEE half pattern. pliron's
/// `FPHalfAttr` wraps `apfloat::Half`, whose bit access is `u128`-wide via the
/// `Float` trait, so we widen here.
pub fn fp16_attr_from_bits(bits: u16) -> FPHalfAttr {
    FPHalfAttr(Half::from_bits(bits as u128))
}

/// Extract the raw 16-bit IEEE half pattern from an `FPHalfAttr`.
pub fn fp16_attr_to_bits(attr: &FPHalfAttr) -> u16 {
    attr.0.to_bits() as u16
}

#[cfg(test)]
mod tests {
    use super::ops::{AsmKind, InlineAsmOp, InlineAsmOpExt, asm_kind};
    use super::types::VoidType;
    use pliron::context::Context;

    #[test]
    fn asm_kind_convergent_round_trips() {
        let mut ctx = Context::new();
        let void_ty = VoidType::get(&ctx);
        let op = InlineAsmOp::build(
            &mut ctx,
            void_ty.into(),
            vec![],
            "bar.sync 0;",
            "",
            AsmKind::Convergent,
        );
        assert_eq!(asm_kind(&ctx, &op), AsmKind::Convergent);
    }

    #[test]
    fn asm_kind_pure_round_trips() {
        let mut ctx = Context::new();
        let void_ty = VoidType::get(&ctx);
        let op = InlineAsmOp::build(&mut ctx, void_ty.into(), vec![], "nop;", "", AsmKind::Pure);
        assert_eq!(asm_kind(&ctx, &op), AsmKind::Pure);
    }

    #[test]
    fn asm_kind_side_effect_round_trips() {
        let mut ctx = Context::new();
        let void_ty = VoidType::get(&ctx);
        let op = InlineAsmOp::build(
            &mut ctx,
            void_ty.into(),
            vec![],
            "st.shared [%0], %1;",
            "r,r",
            AsmKind::SideEffect,
        );
        assert_eq!(asm_kind(&ctx, &op), AsmKind::SideEffect);
    }

    #[test]
    fn asm_kind_convergent_pure_round_trips() {
        let mut ctx = Context::new();
        let void_ty = VoidType::get(&ctx);
        let op = InlineAsmOp::build(
            &mut ctx,
            void_ty.into(),
            vec![],
            "shfl.sync.bfly.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            AsmKind::ConvergentPure,
        );
        assert_eq!(asm_kind(&ctx, &op), AsmKind::ConvergentPure);
    }
}
