/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor storage, access, format, and selection metadata.

use dialect_mir::types::{MirFP16Type, MirStructType};
use pliron::{
    builtin::types::{FP32Type, IntegerType},
    common_traits::Verify,
    context::Context,
    derive::pliron_type,
    location::Location,
    result::Error,
    r#type::{Type, TypeHandle, TypedHandle},
    verify_err,
};

use crate::attributes::{
    CuteAlignmentAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr,
    CuteTensorLayoutAttr, CuteTensorRoleAttr,
};

/// A typed view of tensor storage.
///
/// Read the fields left to right:
///
/// ```text
/// logical  storage  format  role  space  access  alignment  layout
///   f32       u8      E2M1   Mkl   Gmem    RO        1      KMajor
///    │         │        │     │      │      │         │        │
/// decoded   carrier  bits  coords  where  allowed  base     order
/// ```
///
/// Plain elementwise views have matching logical and storage types. Packed
/// views decode `u8` storage into logical `f32` values and say whether their
/// rows follow the M or N coordinate.
///
/// Alignment here describes the storage pointer known at view construction.
/// A later unsafe transfer may promise a stronger selected-address alignment
/// on its own operation.
///
/// This is a ghost type. It records meaning for compiler passes; it is never a
/// runtime descriptor and cannot be part of a kernel ABI.
#[pliron_type(
    name = "cute.tensor_view",
    format = "`<` $logical `,` $storage `,` $format `,` $role `,` $space `,` $access `,` $alignment `,` $layout `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteTensorViewType {
    /// Decoded element type visible to tensor arithmetic.
    pub logical: TypeHandle,
    /// MIR element type physically stored in memory.
    pub storage: TypeHandle,
    /// Logical interpretation of the bits in each physical storage element.
    pub format: CuteTensorFormatAttr,
    /// Logical M/N coordinate role assigned to the tensor.
    pub role: CuteTensorRoleAttr,
    /// Address space in which the tensor storage resides.
    pub space: CuteTensorAddressSpaceAttr,
    /// Read-only or writable capability of the tensor view.
    pub access: CuteTensorAccessAttr,
    /// Promised storage-base alignment, in bytes.
    pub alignment: CuteAlignmentAttr,
    /// Static coordinate layout used to interpret the storage.
    pub layout: CuteTensorLayoutAttr,
}

impl CuteTensorViewType {
    /// Create a ghost tensor-view type.
    #[allow(clippy::too_many_arguments)]
    pub fn get(
        ctx: &mut Context,
        logical: TypeHandle,
        storage: TypeHandle,
        space: CuteTensorAddressSpaceAttr,
        access: CuteTensorAccessAttr,
        alignment_bytes: u64,
        layout: CuteTensorLayoutAttr,
    ) -> TypedHandle<Self> {
        Self::get_with_facts(
            ctx,
            logical,
            storage,
            space,
            access,
            alignment_bytes,
            CuteTensorFormatAttr::Plain,
            CuteTensorRoleAttr::Generic,
            layout,
        )
    }

    /// Create a tensor view with packed-format and matrix-role facts.
    #[allow(clippy::too_many_arguments)]
    pub fn get_with_facts(
        ctx: &mut Context,
        logical: TypeHandle,
        storage: TypeHandle,
        space: CuteTensorAddressSpaceAttr,
        access: CuteTensorAccessAttr,
        alignment_bytes: u64,
        format: CuteTensorFormatAttr,
        role: CuteTensorRoleAttr,
        layout: CuteTensorLayoutAttr,
    ) -> TypedHandle<Self> {
        Type::instantiate(
            Self {
                logical,
                storage,
                format,
                role,
                space,
                access,
                alignment: CuteAlignmentAttr(alignment_bytes),
                layout,
            },
            ctx,
        )
    }

    /// Copy every tensor fact except its layout.
    #[must_use]
    pub fn with_layout(
        &self,
        ctx: &mut Context,
        layout: CuteTensorLayoutAttr,
    ) -> TypedHandle<Self> {
        Self::get_with_facts(
            ctx,
            self.logical,
            self.storage,
            self.space,
            self.access,
            self.alignment.0,
            self.format,
            self.role,
            layout,
        )
    }

    /// Scalar representation used to transport this element's unchanged bits.
    ///
    /// Logical and storage types retain transparent numeric wrappers. Only
    /// copying looks through a rustc-proven transparent scalar with one field,
    /// zero offset, and exactly the scalar's size and alignment. Other structs
    /// cannot acquire a scalar copy representation merely from their size.
    #[must_use]
    pub fn transport_scalar(&self, ctx: &Context) -> Option<TypeHandle> {
        scalar_transport_layout(ctx, self.storage, 0).map(|(scalar, _)| scalar)
    }

    /// Size of one supported storage element in bytes.
    #[must_use]
    pub fn storage_bytes(&self, ctx: &Context) -> Option<u64> {
        scalar_transport_layout(ctx, self.storage, 0).map(|(_, bytes)| bytes)
    }

    /// Return the selected tile width, if this is a tile view.
    #[must_use]
    pub const fn selected_tile_size(&self) -> Option<u64> {
        match self.layout {
            CuteTensorLayoutAttr::Tile1D(size) => Some(size),
            _ => None,
        }
    }
}

/// Resolve scalar bits only through wrappers with a fully checked identical layout.
fn scalar_transport_layout(
    ctx: &Context,
    ty: TypeHandle,
    depth: usize,
) -> Option<(TypeHandle, u64)> {
    if depth > 16 {
        return None;
    }
    let element = ty.deref(ctx);
    if element.downcast_ref::<MirFP16Type>().is_some() {
        Some((ty, 2))
    } else if element.downcast_ref::<FP32Type>().is_some() {
        Some((ty, 4))
    } else if let Some(integer) = element.downcast_ref::<IntegerType>() {
        let width = u64::from(integer.width());
        (width > 0 && width % 8 == 0).then_some((ty, width / 8))
    } else if let Some(wrapper) = element.downcast_ref::<MirStructType>() {
        if !wrapper.is_transparent_scalar()
            || wrapper.field_count() != 1
            || wrapper.field_names.len() != 1
            || wrapper.field_offsets.as_slice() != [0]
            || wrapper.memory_order().as_slice() != [0]
        {
            return None;
        }
        let (scalar, bytes) = scalar_transport_layout(ctx, wrapper.field_types[0], depth + 1)?;
        (wrapper.total_size == bytes && wrapper.abi_align == bytes).then_some((scalar, bytes))
    } else {
        None
    }
}

impl Verify for CuteTensorViewType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let Some(storage_bytes) = self.storage_bytes(ctx) else {
            return verify_err!(
                Location::Unknown,
                "cute.tensor_view storage must use a whole number of bytes"
            );
        };
        let alignment = self.alignment.0;
        if alignment == 0 || !alignment.is_power_of_two() {
            return verify_err!(
                Location::Unknown,
                "cute.tensor_view alignment must be a positive power of two"
            );
        }
        if alignment < storage_bytes {
            return verify_err!(
                Location::Unknown,
                "cute.tensor_view alignment cannot be smaller than one storage element"
            );
        }
        if self.layout.tile_size().is_some_and(|size| size == 0) {
            return verify_err!(
                Location::Unknown,
                "cute.tensor_view tile size must be greater than zero"
            );
        }

        let logical_is_f32 = self.logical.deref(ctx).downcast_ref::<FP32Type>().is_some();
        let storage_is_u8 = self
            .storage
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 8 && integer.is_unsigned());
        let storage_is_u32 = self
            .storage
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 32 && integer.is_unsigned());
        let storage_is_f16 = self
            .storage
            .deref(ctx)
            .downcast_ref::<MirFP16Type>()
            .is_some();

        match self.format {
            CuteTensorFormatAttr::Plain => {
                if self.logical != self.storage {
                    return verify_err!(
                        Location::Unknown,
                        "plain cute.tensor_view needs matching logical and storage types"
                    );
                }
                // Keep the wrapper in the typed IR; inspect only its validated
                // physical scalar when deciding which bit copies are supported.
                let transport = self
                    .transport_scalar(ctx)
                    .expect("storage width checked above");
                let storage_ty = transport.deref(ctx);
                let plain_integer_is_supported = storage_ty
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|integer| {
                        integer.is_unsigned() && matches!(integer.width(), 8 | 16)
                    });
                let plain_is_supported = storage_ty.downcast_ref::<MirFP16Type>().is_some()
                    || storage_ty.downcast_ref::<FP32Type>().is_some()
                    || plain_integer_is_supported;
                if !plain_is_supported {
                    return verify_err!(
                        Location::Unknown,
                        "plain cute.tensor_view supports u8, u16, f16, and f32 carriers or proven transparent scalar wrappers only"
                    );
                }
                if self.role != CuteTensorRoleAttr::Generic {
                    return verify_err!(
                        Location::Unknown,
                        "plain cute.tensor_view must use the Generic role"
                    );
                }
                if !matches!(
                    self.layout,
                    CuteTensorLayoutAttr::Contiguous1D
                        | CuteTensorLayoutAttr::Zipped1D(_)
                        | CuteTensorLayoutAttr::Tile1D(_)
                        | CuteTensorLayoutAttr::Tma2D
                ) {
                    return verify_err!(
                        Location::Unknown,
                        "plain cute.tensor_view needs an elementwise or TMA transport layout"
                    );
                }
            }
            CuteTensorFormatAttr::E2M1 => {
                let storage_matches_space = match self.space {
                    CuteTensorAddressSpaceAttr::Gmem => storage_is_u8,
                    CuteTensorAddressSpaceAttr::Smem => storage_is_f16,
                };
                if !logical_is_f32 || !storage_matches_space {
                    return verify_err!(
                        Location::Unknown,
                        "E2M1 cute.tensor_view needs logical f32 values in u8 global or f16 shared storage"
                    );
                }
                if self.access != CuteTensorAccessAttr::ReadOnly {
                    return verify_err!(
                        Location::Unknown,
                        "E2M1 cute.tensor_view must be read-only"
                    );
                }
                if !matches!(self.role, CuteTensorRoleAttr::Mkl | CuteTensorRoleAttr::Nkl) {
                    return verify_err!(
                        Location::Unknown,
                        "E2M1 cute.tensor_view needs an Mkl or Nkl role"
                    );
                }
                if self.layout != CuteTensorLayoutAttr::KMajor {
                    return verify_err!(Location::Unknown, "E2M1 cute.tensor_view must use KMajor");
                }
            }
            CuteTensorFormatAttr::UE8M0 | CuteTensorFormatAttr::E4M3 => {
                let storage_matches_space = match self.space {
                    CuteTensorAddressSpaceAttr::Gmem => storage_is_u8,
                    CuteTensorAddressSpaceAttr::Smem => storage_is_u32,
                };
                if !logical_is_f32 || !storage_matches_space {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0/E4M3 cute.tensor_view needs logical f32 scales in u8 global or u32 shared storage"
                    );
                }
                if self.access != CuteTensorAccessAttr::ReadOnly {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0/E4M3 cute.tensor_view must be read-only"
                    );
                }
                if !matches!(self.role, CuteTensorRoleAttr::Mkl | CuteTensorRoleAttr::Nkl) {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0/E4M3 cute.tensor_view needs an Mkl or Nkl role"
                    );
                }
                if !matches!(self.layout, CuteTensorLayoutAttr::BlockScaleKMajor(size) if size > 0)
                {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0/E4M3 cute.tensor_view must use BlockScaleKMajor"
                    );
                }
            }
        }
        Ok(())
    }
}
