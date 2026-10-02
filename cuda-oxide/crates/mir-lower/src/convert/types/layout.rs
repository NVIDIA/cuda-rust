/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Size/alignment metrics, padding/filler type builders, byte-faithfulness
//! and `i1`-storage predicates, and natural struct layout walks.

use dialect_mir::types::{
    MirArrayType, MirEnumType, MirFP16Type, MirPtrType, MirStructType, MirTupleType, MirUnionType,
};
use llvm_export::types as llvm_types;
use pliron::builtin::type_interfaces::FloatTypeInterface;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::Context;
use pliron::r#type::{TypeHandle, type_cast};

/// Create a padding type: `[N x i8]` for N bytes of padding.
pub(super) fn make_padding_type(ctx: &mut Context, size: u64) -> TypeHandle {
    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    llvm_types::ArrayType::get(ctx, i8_ty.into(), size).into()
}

/// Storage for `size` bytes of enum filler at byte `offset`: bytes the payload
/// occupies that no typed slot claims.
///
/// `[N x i8]` is byte-exact but costs one leaf *per byte* everywhere a payload
/// value is built, merged, or taken apart. `Option<&[T]>` is the shape that
/// shows it: the niche carrier claims the pointer, the length is left to an
/// 8-byte filler, and building the value then decomposes that length into
/// eight `i8` `insertvalue`s, every block merge carries eight `phi i8`, and
/// reading it back is a chain of `prmt` byte permutes with the aggregate
/// spilled to `.local` in between. Measured on sm_86, it costs the *pointer*
/// too: it rides through the same byte soup, `InferAddressSpaces` can no
/// longer follow it, and the access lands in the generic window (`ld.v2.b64`)
/// instead of `ld.global.v2.b64`.
///
/// One integer of the same width is the same bytes in one leaf. It is only
/// legal where it moves nothing:
///
/// - the width is 2, 4 or 8 bytes, so the integer is byte-faithful (a multiple
///   of 8 bits, hence no padding of its own). 16 is deliberately excluded:
///   `i128` is legal LLVM but lowers to register pairs on NVPTX, so widening
///   that far trades one win for another cost and wants its own measurement;
/// - `offset` is a multiple of the width, so LLVM inserts no gap ahead of the
///   field and every later field keeps its byte offset;
/// - the width does not exceed the enum's alignment, or the struct's natural
///   alignment would rise above rustc's.
///
/// Anything else keeps the byte array. The filler is a *vehicle*, not a claim:
/// nothing reads it field-wise, because a payload that has no typed slot
/// round-trips through memory as a whole aggregate, and both ends of that
/// round trip are byte-exact. `build_enum_slot_map`'s own size and alignment
/// assertions — hard errors, not debug checks — backstop all three conditions.
pub(super) fn make_enum_filler_type(
    ctx: &mut Context,
    offset: u64,
    size: u64,
    abi_align: u64,
) -> TypeHandle {
    if matches!(size, 2 | 4 | 8) && offset.is_multiple_of(size) && size <= abi_align.max(1) {
        let width = u32::try_from(size * 8).expect("size is at most 8, so width is at most 64");
        return IntegerType::get(ctx, width, Signedness::Signless).into();
    }
    make_padding_type(ctx, size)
}

/// Whether this is an aggregate/vector that contains LLVM `i1` storage.
///
/// Rust `bool` is an SSA `i1`, but its memory representation is one complete
/// byte whose only valid values are 0 and 1. Enum storage never uses `i1` as
/// a physical type: scalar bools claim an explicit i8 byte, and aggregates
/// containing bools claim their byte-faithful twin (see
/// [`llvm_byte_faithful_twin`]), with construction canonicalizing the value
/// before the store.
pub(crate) fn llvm_type_contains_i1(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        return integer.width() == 1;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_contains_i1(ctx, array.elem_type());
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        return llvm_type_contains_i1(ctx, vector.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return struct_ty
            .fields()
            .any(|field| llvm_type_contains_i1(ctx, field));
    }
    false
}

/// The byte-faithful storage twin of an LLVM type: every `i1` leaf becomes
/// its canonical `i8` memory byte, recursively through structs and arrays.
///
/// Rust guarantees a bool occupies one full byte holding exactly 0 or 1, so
/// storing the twin (with each bool zero-extended) writes the same bytes the
/// host writes, while re-loading the original type from those canonical
/// bytes remains well-defined. Sizes and alignments are unchanged because an
/// `i1` already occupies one byte of storage.
///
/// Returns `None` for shapes with a different memory story (`i1` vectors are
/// bit-packed masks) or unknown containers; callers must fail closed.
pub(crate) fn llvm_byte_faithful_twin(ctx: &mut Context, ty: TypeHandle) -> Option<TypeHandle> {
    if !llvm_type_contains_i1(ctx, ty) {
        return Some(ty);
    }
    enum Shape {
        Bool,
        Array(TypeHandle, u64),
        Struct {
            fields: Vec<TypeHandle>,
            layout: llvm_types::StructLayout,
        },
        Other,
    }
    let shape = {
        let ty_ref = ty.deref(ctx);
        if ty_ref
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 1)
        {
            Shape::Bool
        } else if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            Shape::Array(array.elem_type(), array.size())
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            Shape::Struct {
                fields: struct_ty.fields().collect(),
                layout: struct_ty.layout(),
            }
        } else {
            Shape::Other
        }
    };
    match shape {
        Shape::Bool => Some(IntegerType::get(ctx, 8, Signedness::Signless).into()),
        Shape::Array(elem, count) => {
            let twin = llvm_byte_faithful_twin(ctx, elem)?;
            Some(llvm_types::ArrayType::get(ctx, twin, count).into())
        }
        Shape::Struct { fields, layout } => {
            let twins = fields
                .into_iter()
                .map(|field| llvm_byte_faithful_twin(ctx, field))
                .collect::<Option<Vec<_>>>()?;
            Some(llvm_types::StructType::get_unnamed(ctx, (twins, layout)).into())
        }
        Shape::Other => None,
    }
}

/// Whether a MIR aggregate contains a semantic Rust bool value.
///
/// This deliberately stops at pointers/slices: a pointee bool does not occupy
/// bytes in the aggregate itself. It also stops at nested enums, whose own
/// slot-map construction is responsible for canonicalizing a top-level bool
/// or rejecting a deeper one. Inspecting the MIR type is necessary because a
/// union may lower to raw i8 storage and thereby hide a bool from the LLVM
/// type-level check above.
pub(super) fn mir_type_contains_i1(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        return integer.width() == 1;
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
        return struct_ty
            .field_types()
            .iter()
            .copied()
            .any(|field| mir_type_contains_i1(ctx, field));
    }
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return tuple_ty
            .get_types()
            .iter()
            .copied()
            .any(|field| mir_type_contains_i1(ctx, field));
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
        return mir_type_contains_i1(ctx, array_ty.element_ty);
    }
    if let Some(union_ty) = ty_ref.downcast_ref::<MirUnionType>() {
        return union_ty
            .field_types()
            .iter()
            .copied()
            .any(|field| mir_type_contains_i1(ctx, field));
    }
    false
}

/// Whether loading and storing this LLVM value preserves every byte in its
/// allocation. In particular, `i1` is not byte-faithful: it occupies one
/// addressable byte, but LLVM does not define the upper seven stored bits.
pub(crate) fn llvm_type_is_byte_faithful(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return int_ty.width().is_multiple_of(8);
    }
    if ty_ref.is::<llvm_types::HalfType>()
        || ty_ref.is::<FP32Type>()
        || ty_ref.is::<FP64Type>()
        || ty_ref.is::<llvm_types::PointerType>()
    {
        return true;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_is_byte_faithful(ctx, array.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        if struct_ty.layout() == llvm_types::StructLayout::Packed {
            return fields
                .into_iter()
                .all(|field| llvm_type_is_byte_faithful(ctx, field));
        }

        let mut end = 0u64;
        let mut max_align = 1u64;
        for field in fields {
            if !llvm_type_is_byte_faithful(ctx, field) {
                return false;
            }
            let Some((field_size, field_align)) = llvm_type_size_align(ctx, field) else {
                return false;
            };
            let field_align = field_align.max(1);
            let aligned_end = end.div_ceil(field_align) * field_align;
            if aligned_end != end {
                // LLVM would insert bytes that are not represented by an SSA
                // field. Loading and re-storing the aggregate would lose them.
                return false;
            }
            end += field_size;
            max_align = max_align.max(field_align);
        }
        // Reject implicit trailing padding for the same reason. Explicit
        // `[N x i8]` padding fields keep `end` equal to the allocation size.
        return end.div_ceil(max_align) * max_align == end;
    }
    false
}

/// Size of a MIR-level type from rustc layout truth, when stored.
///
/// `MirStructType`, `MirTupleType`, `MirUnionType`, and `MirEnumType` carry
/// `total_size` (interior and trailing padding included) straight from
/// rustc's layout query; arrays of such aggregates multiply it out. Returns
/// `None` when no stored size is available (e.g. niched/single-variant enums
/// store 0) and the caller must fall back to the LLVM-level approximation.
pub(super) fn mir_stored_size(ctx: &Context, mir_ty: TypeHandle) -> Option<u64> {
    let ty_ref = mir_ty.deref(ctx);
    if let Some(s) = ty_ref.downcast_ref::<MirStructType>() {
        if s.total_size() > 0 {
            return Some(s.total_size());
        }
        return None;
    }
    if let Some(t) = ty_ref.downcast_ref::<MirTupleType>() {
        if t.total_size > 0 {
            return Some(t.total_size);
        }
        return None;
    }
    if let Some(e) = ty_ref.downcast_ref::<MirEnumType>() {
        if e.total_size() > 0 {
            return Some(e.total_size());
        }
        return None;
    }
    if let Some(u) = ty_ref.downcast_ref::<MirUnionType>() {
        return Some(u.total_size());
    }
    if let Some(a) = ty_ref.downcast_ref::<MirArrayType>() {
        let elem_ty = a.element_ty;
        let size = a.size;
        // Checked: alignment claims consume this, and a wrapped product
        // would masquerade as a small, trusted stride.
        return mir_stored_size(ctx, elem_ty).and_then(|elem_size| elem_size.checked_mul(size));
    }
    None
}

/// Exact byte stride of a MIR array's element, when provable.
///
/// Element `i` of an array lives at byte `i * stride`, so any alignment
/// claim about an element address is only as strong as the stride it
/// multiplies, and a wrong stride is a miscompile. Scalars have exact
/// sizes; MIR aggregates answer with rustc's stored size (interior and
/// trailing padding included) via [`mir_stored_size`]; arrays of scalars
/// multiply out. Everything else answers `None` so the caller declines
/// instead of guessing.
pub(crate) fn mir_element_stride(ctx: &Context, mir_ty: TypeHandle) -> Option<u64> {
    if let Some(size) = mir_stored_size(ctx, mir_ty) {
        return Some(size);
    }
    let ty_ref = mir_ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return Some(u64::from(int_ty.width()).div_ceil(8));
    }
    // The importer's f16 is MirFP16Type; the HalfType arm covers IR that
    // already carries the converted LLVM scalar.
    if ty_ref.is::<MirFP16Type>() || ty_ref.is::<llvm_types::HalfType>() {
        return Some(2);
    }
    if ty_ref.is::<FP32Type>() {
        return Some(4);
    }
    if ty_ref.is::<FP64Type>() {
        return Some(8);
    }
    if let Some(ptr_ty) = ty_ref.downcast_ref::<MirPtrType>() {
        // Generic-space pointers are 64-bit under every data layout the
        // exporter can choose; a shared-space pointer is 32-bit under the
        // modern NVVM layout alone, so its stored stride is target-dependent.
        // Claim the former, decline the latter.
        return (ptr_ty.address_space == 0).then_some(8);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
        // Arrays of sized aggregates already answered through
        // `mir_stored_size` above; this recursion serves arrays of
        // scalars and pointers, and deeper such arrays.
        let element_ty = array_ty.element_type();
        let count = array_ty.size();
        drop(ty_ref);
        return mir_element_stride(ctx, element_ty)?.checked_mul(count);
    }
    None
}

/// Exact ABI alignment carried by a MIR aggregate type, when rustc layout is
/// available.
///
/// LLVM aggregate types cannot encode a Rust `repr(align(N))` raise. Tuples,
/// structs, enums, and unions therefore carry rustc's alignment explicitly in
/// the MIR dialect. Arrays have the same ABI alignment as their element, so
/// recurse through any number of array layers instead of relying on the
/// converted LLVM element's structural alignment.
pub(crate) fn mir_type_abi_align(ctx: &Context, mir_ty: TypeHandle) -> Option<u64> {
    let ty_ref = mir_ty.deref(ctx);
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return Some(tuple_ty.abi_align()).filter(|align| *align > 0);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
        return Some(struct_ty.abi_align).filter(|align| *align > 0);
    }
    if let Some(enum_ty) = ty_ref.downcast_ref::<MirEnumType>() {
        return Some(enum_ty.abi_align()).filter(|align| *align > 0);
    }
    if let Some(union_ty) = ty_ref.downcast_ref::<MirUnionType>() {
        return Some(union_ty.abi_align()).filter(|align| *align > 0);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
        let element_ty = array_ty.element_type();
        return mir_type_abi_align(ctx, element_ty);
    }
    None
}

/// LLVM natural-layout `(size, align)` of an exported LLVM type, in bytes.
///
/// Mirrors LLVM's default data layout for nvptx64 (scalars align to their
/// size, arrays to their element, non-packed structs to their widest field).
/// Computes the real allocation size (interior and trailing padding
/// included), which is what GEP striding and the enum size check below
/// need. Answers `None` rather than guess: sizes here are exact or absent.
pub(crate) fn llvm_type_size_align(ctx: &Context, ty: TypeHandle) -> Option<(u64, u64)> {
    let ty_ref = ty.deref(ctx);

    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        let size = (int_ty.width() as u64).div_ceil(8);
        // i8 → 1, i16 → 2, i32 → 4, i64 → 8, i128 → 16.
        return Some((size, size.next_power_of_two().min(16)));
    }
    if ty_ref.is::<llvm_types::HalfType>() {
        return Some((2, 2));
    }
    if ty_ref.is::<FP32Type>() {
        return Some((4, 4));
    }
    if ty_ref.is::<FP64Type>() {
        return Some((8, 8));
    }
    if ty_ref.is::<llvm_types::PointerType>() {
        // Lowering runs before the exporter chooses the minimal, legacy, or
        // modern NVPTX data layout. The first two use 64-bit pointers in all
        // address spaces; modern NVVM alone uses p3:32. Callers that expose
        // physical shared-pointer width must either genericize a direct
        // pointer first or reject the target-dependent representation.
        return Some((8, 8));
    }
    if let Some(arr_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        let (elem_size, elem_align) = llvm_type_size_align(ctx, arr_ty.elem_type())?;
        return Some((elem_size.checked_mul(arr_ty.size())?, elem_align.max(1)));
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        if vector.is_scalable() {
            return None;
        }
        let element_bits = {
            let element = vector.elem_type();
            let element_ref = element.deref(ctx);
            if let Some(integer) = element_ref.downcast_ref::<IntegerType>() {
                u64::from(integer.width())
            } else if let Some(float) = type_cast::<dyn FloatTypeInterface>(&*element_ref) {
                u64::try_from(float.get_semantics().bits).ok()?
            } else if element_ref.is::<llvm_types::PointerType>() {
                64
            } else {
                return None;
            }
        };
        let total_bits = element_bits.checked_mul(u64::from(vector.num_elements()))?;
        let size = total_bits.div_ceil(8);
        // Both cuda-oxide NVPTX data layouts explicitly define fixed vector
        // ABI alignment only for these widths. Refuse to guess LLVM defaults
        // for any other width in physical Rust layout code.
        let align = match total_bits {
            16 => 2,
            32 => 4,
            64 => 8,
            128 => 16,
            _ => return None,
        };
        return Some((size, align));
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        if struct_ty.layout() == llvm_types::StructLayout::Packed {
            let mut size = 0u64;
            for field in fields {
                let (field_size, _) = llvm_type_size_align(ctx, field)?;
                size = size.checked_add(field_size)?;
            }
            return Some((size, 1));
        }

        let (_end, size, align) = natural_struct_layout(ctx, &fields)?;
        return Some((size, align));
    }

    None
}

/// The one walk defining natural (non-packed) LLVM struct placement: each
/// field starts at the running offset rounded up to its own alignment.
///
/// Returns `(offsets, end, align)`: the byte offset of every field, the
/// unrounded offset just past the last field, and the widest field alignment.
/// [`natural_struct_layout`] and [`StructSlotMap::natural_slot_offsets`] are
/// both views of this walk, so a size question and an offset question can
/// never disagree.
pub(super) fn natural_layout_walk(
    ctx: &Context,
    fields: &[TypeHandle],
) -> Option<(Vec<u64>, u64, u64)> {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut end = 0u64;
    let mut align = 1u64;
    for field in fields {
        let (field_size, field_align) = llvm_type_size_align(ctx, *field)?;
        let field_align = field_align.max(1);
        end = end.div_ceil(field_align) * field_align;
        offsets.push(end);
        end = end.checked_add(field_size)?;
        align = align.max(field_align);
    }
    Some((offsets, end, align))
}

/// Natural (non-packed) LLVM struct layout over `fields`.
///
/// Returns `(end, size, align)` where `end` is the unrounded offset just past
/// the last field, `size` is `end` rounded up to the struct alignment (the
/// allocation size LLVM uses for GEP striding), and `align` is the widest
/// field alignment.
pub(crate) fn natural_struct_layout(
    ctx: &Context,
    fields: &[TypeHandle],
) -> Option<(u64, u64, u64)> {
    let (_offsets, end, align) = natural_layout_walk(ctx, fields)?;
    let size = end.div_ceil(align) * align;
    Some((end, size, align))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{llvm_int, make_ctx, mir_uint};
    use super::*;

    #[test]
    fn mir_abi_alignment_recurses_through_nested_arrays() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let marker: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Align32".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            32,
        )
        .into();
        let tuple: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![marker, byte],
            vec![0, 1],
            vec![0, 0],
            32,
            32,
        )
        .into();
        let inner: TypeHandle = MirArrayType::get(&mut ctx, tuple, 2).into();
        let outer: TypeHandle = MirArrayType::get(&mut ctx, inner, 3).into();
        let plain: TypeHandle = MirArrayType::get(&mut ctx, byte, 4).into();

        assert_eq!(mir_type_abi_align(&ctx, tuple), Some(32));
        assert_eq!(mir_type_abi_align(&ctx, inner), Some(32));
        assert_eq!(mir_type_abi_align(&ctx, outer), Some(32));
        assert_eq!(mir_type_abi_align(&ctx, plain), None);
    }

    #[test]
    fn fixed_vector_layout_uses_packed_bit_width_and_rejects_unknown_widths() {
        let mut ctx = make_ctx();
        let i1 = llvm_int(&mut ctx, 1);
        let v16i1: TypeHandle =
            llvm_types::VectorType::get(&ctx, i1, 16, llvm_types::VectorTypeKind::Fixed).into();
        assert_eq!(
            llvm_type_size_align(&ctx, v16i1),
            Some((2, 2)),
            "<16 x i1> is 16 packed bits, not sixteen bytes"
        );

        let i8 = llvm_int(&mut ctx, 8);
        let v3i8: TypeHandle =
            llvm_types::VectorType::get(&ctx, i8, 3, llvm_types::VectorTypeKind::Fixed).into();
        assert_eq!(
            llvm_type_size_align(&ctx, v3i8),
            None,
            "the NVPTX data layout does not define a 24-bit vector alignment"
        );
    }
}
