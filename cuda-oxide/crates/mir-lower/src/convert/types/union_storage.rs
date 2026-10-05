/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Byte-exact LLVM storage for Rust unions.

use dialect_mir::types::MirUnionType;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::r#type::TypeHandle;

use super::layout::make_padding_type;
use super::pointer_storage::llvm_type_contains_pointer;
use super::{convert_type, llvm_type_is_byte_faithful, llvm_type_size_align};

/// Build byte-exact LLVM storage for a Rust union.
///
/// A union cannot be represented as an LLVM struct containing every declared
/// field: struct fields are consecutive, while union fields all start at byte
/// zero. We choose one byte-faithful field as the storage view and add explicit
/// tail bytes. A zero-length integer array raises the LLVM type's natural
/// alignment without consuming storage. Pointer-bearing fields are preferred
/// so an ordinary union copy keeps LLVM pointer provenance.
///
/// NVPTX gives scalar integers natural alignments up to 16 bytes. Stronger
/// Rust alignment is carried explicitly on memory operations because LLVM
/// aggregate types cannot encode over-alignment; the storage type still keeps
/// the union's exact size and therefore its array stride.
pub(crate) fn build_union_storage_type(
    ctx: &mut Context,
    union_ty: &MirUnionType,
) -> Result<TypeHandle, anyhow::Error> {
    let size = union_ty.total_size();
    let align = union_ty.abi_align();
    if align == 0 || !align.is_power_of_two() {
        return Err(anyhow::anyhow!(
            "union `{}` has invalid ABI alignment {}",
            union_ty.name(),
            align
        ));
    }
    if size > 0 && !size.is_multiple_of(align) {
        return Err(anyhow::anyhow!(
            "union `{}` size {} is not a multiple of its {}-byte ABI alignment",
            union_ty.name(),
            size,
            align
        ));
    }

    let mut fields = Vec::with_capacity(union_ty.field_count());
    let mut pointer_carrier: Option<TypeHandle> = None;
    for (index, &field_ty) in union_ty.field_types().iter().enumerate() {
        let llvm_field_ty = convert_type(ctx, field_ty)?;
        let (field_size, field_align) =
            llvm_type_size_align(ctx, llvm_field_ty).ok_or_else(|| {
                anyhow::anyhow!(
                    "union `{}` field {} has unsupported LLVM size/alignment",
                    union_ty.name(),
                    index
                )
            })?;
        if field_size > size {
            return Err(anyhow::anyhow!(
                "union `{}` field {} lowers to {} bytes but the union is only {} bytes",
                union_ty.name(),
                index,
                field_size,
                size
            ));
        }
        if field_align > align {
            return Err(anyhow::anyhow!(
                "union `{}` field {} lowers with alignment {} but rustc reports union alignment {}",
                union_ty.name(),
                index,
                field_align,
                align
            ));
        }
        let contains_pointer = llvm_type_contains_pointer(ctx, llvm_field_ty);
        if contains_pointer {
            if let Some(first) = pointer_carrier
                && first != llvm_field_ty
            {
                return Err(anyhow::anyhow!(
                    "union `{}` has pointer-bearing fields with different LLVM representations; preserving provenance for that shape is not yet supported",
                    union_ty.name()
                ));
            }
            pointer_carrier = Some(llvm_field_ty);
        }
        fields.push((llvm_field_ty, field_size, field_align, contains_pointer));
    }

    let storage_align = align.min(16);
    let anchor_int = IntegerType::get(ctx, (storage_align * 8) as u32, Signedness::Signless);
    let anchor: TypeHandle = llvm_types::ArrayType::get(ctx, anchor_int.into(), 0).into();
    let mut storage_fields = vec![anchor];
    if size > 0 {
        let representative = fields
            .iter()
            .filter(|(ty, field_size, _, _)| {
                *field_size > 0
                    && llvm_type_is_byte_faithful(ctx, *ty)
                    && (pointer_carrier.is_none() || llvm_type_contains_pointer(ctx, *ty))
            })
            .max_by_key(|(_, field_size, field_align, contains_pointer)| {
                (*contains_pointer, *field_align, *field_size)
            });
        if let Some(representative) = representative {
            storage_fields.push(representative.0);
            if representative.1 < size {
                storage_fields.push(make_padding_type(ctx, size - representative.1));
            }
        } else if pointer_carrier.is_some() {
            return Err(anyhow::anyhow!(
                "union `{}` has pointer-bearing fields but no byte-faithful pointer carrier; lowering it as raw bytes would discard pointer provenance",
                union_ty.name()
            ));
        } else {
            // Pointer-free unions may safely use raw bytes as their SSA
            // carrier. Field loads/stores still use their declared types.
            storage_fields.push(make_padding_type(ctx, size));
        }
    }
    let storage: TypeHandle = llvm_types::StructType::get_unnamed(
        ctx,
        (storage_fields, llvm_types::StructLayout::Unpacked),
    )
    .into();
    let (llvm_size, llvm_align) = llvm_type_size_align(ctx, storage).ok_or_else(|| {
        anyhow::anyhow!(
            "union `{}` storage has unsupported LLVM layout",
            union_ty.name()
        )
    })?;
    if llvm_size != size || llvm_align > align {
        return Err(anyhow::anyhow!(
            "union `{}` storage lowered to incompatible size/alignment {}/{} but rustc requires {}/{}",
            union_ty.name(),
            llvm_size,
            llvm_align,
            size,
            align
        ));
    }
    Ok(storage)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{make_ctx, mir_uint, struct_fields};
    use super::*;
    use dialect_mir::types::{MirArrayType, MirPtrType, MirStructType};

    #[test]
    fn union_storage_has_exact_size_alignment_and_stride() {
        let mut ctx = make_ctx();
        let u8_ty = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let bytes_ty: TypeHandle = MirArrayType::get(&mut ctx, u8_ty, 4).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "Bits".into(),
            vec!["word".into(), "bytes".into()],
            vec![u32_ty, bytes_ty],
            4,
            4,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let storage = build_union_storage_type(&mut ctx, &union_data).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, storage), Some((4, 4)));

        let union_handle: TypeHandle = union_ty.into();
        let array: TypeHandle = MirArrayType::get(&mut ctx, union_handle, 3).into();
        let llvm_array = convert_type(&mut ctx, array).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, llvm_array), Some((12, 4)));
    }

    #[test]
    fn union_storage_prefers_pointer_carrier() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let u64_ty = mir_uint(&mut ctx, 64);
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "PointerBits".into(),
            vec!["ptr".into(), "bits".into()],
            vec![ptr_ty, u64_ty],
            8,
            8,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let storage = build_union_storage_type(&mut ctx, &union_data).unwrap();
        let fields = struct_fields(&ctx, storage);
        assert!(fields[1].deref(&ctx).is::<llvm_types::PointerType>());
        assert_eq!(llvm_type_size_align(&ctx, storage), Some((8, 8)));
    }

    #[test]
    fn union_storage_rejects_incompatible_pointer_address_spaces() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let generic: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "MixedPointers".into(),
            vec!["generic".into(), "shared".into()],
            vec![generic, shared],
            8,
            8,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let err = build_union_storage_type(&mut ctx, &union_data).unwrap_err();
        assert!(err.to_string().contains("different LLVM representations"));
    }

    #[test]
    fn union_storage_rejects_non_byte_faithful_pointer_carrier() {
        let mut ctx = make_ctx();
        let u8_ty = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let bool_ty = mir_uint(&mut ctx, 1);
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let ptr_bool: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PtrBool".into(),
            vec!["ptr".into(), "flag".into()],
            vec![ptr_ty, bool_ty],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let bytes_ty: TypeHandle = MirArrayType::get(&mut ctx, u8_ty, 16).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "PointerBoolBytes".into(),
            vec!["view".into(), "bytes".into()],
            vec![ptr_bool, bytes_ty],
            16,
            8,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let err = build_union_storage_type(&mut ctx, &union_data).unwrap_err();
        assert!(err.to_string().contains("no byte-faithful pointer carrier"));
    }

    #[test]
    fn union_storage_preserves_over_aligned_size_and_stride() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let union_ty = MirUnionType::get(
            &mut ctx,
            "OverAligned".into(),
            vec!["word".into()],
            vec![u32_ty],
            32,
            32,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let storage = build_union_storage_type(&mut ctx, &union_data).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, storage), Some((32, 16)));

        let union_handle: TypeHandle = union_ty.into();
        let array: TypeHandle = MirArrayType::get(&mut ctx, union_handle, 3).into();
        let llvm_array = convert_type(&mut ctx, array).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, llvm_array), Some((96, 16)));
    }
}
