/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Descriptor-backed raw transport views and their direct copy contracts.

use pliron::{
    common_traits::Verify,
    context::Context,
    derive::pliron_type,
    location::Location,
    result::Error,
    r#type::{Type, TypeHandle, TypedHandle},
    verify_err,
};

use super::tensor::CuteTensorViewType;
use crate::attributes::{
    CuteComposedLayoutAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr, CuteTensorRoleAttr,
};

/// A raw-carrier tensor view used by one two-dimensional TMA transfer.
///
/// TMA only moves storage boxes. It does not decode FP4 values, apply block
/// scales, or decide whether bytes belong to A or B:
///
/// ```text
/// !cute.tensor_view<elem, elem, Plain, Generic, Gmem, ..., Tma2D>
///                                │
///                                └── exact shared placement
///                                     !cute.tma_view<..., layout>
/// ```
///
/// The same type wraps the descriptor-backed global source and the
/// pointer-backed shared destination. The inner tensor facts distinguish the
/// address space and access. `smem_layout` records the tile shape, row-major
/// stride, and optional hardware swizzle shared by both sides.
#[pliron_type(name = "cute.tma_view", format = "`<` $tensor `,` $smem_layout `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteTmaViewType {
    /// Wrapped tensor-view type containing storage, format, role, and access facts.
    pub tensor: TypeHandle,
    /// Exact shared tile shape, strides, offset unit, and swizzle.
    pub smem_layout: CuteComposedLayoutAttr,
}

impl CuteTmaViewType {
    /// Attach one exact TMA tile placement to a plain carrier tensor.
    pub fn get(
        ctx: &mut Context,
        tensor: TypeHandle,
        smem_layout: crate::layout::ComposedLayout,
    ) -> TypedHandle<Self> {
        Type::instantiate(
            Self {
                tensor,
                smem_layout: CuteComposedLayoutAttr(smem_layout),
            },
            ctx,
        )
    }

    /// Return the plain tensor facts wrapped by this TMA view.
    #[must_use]
    pub fn tensor_view(&self, ctx: &Context) -> Option<CuteTensorViewType> {
        self.tensor
            .deref(ctx)
            .downcast_ref::<CuteTensorViewType>()
            .cloned()
    }

    /// Return the physical carrier type moved by TMA.
    #[must_use]
    pub fn element(&self, ctx: &Context) -> Option<TypeHandle> {
        self.tensor_view(ctx).map(|view| view.storage)
    }

    /// Return the number of carrier elements in one tile.
    #[must_use]
    pub fn tile_elements(&self) -> Option<u64> {
        u64::try_from(self.smem_layout.0.outer().checked_size()?).ok()
    }

    /// Return the physical byte count moved by one complete tile.
    #[must_use]
    pub fn tile_bytes(&self, ctx: &Context) -> Option<u64> {
        let elements = self.tile_elements()?;
        let bytes = self.tensor_view(ctx)?.storage_bytes(ctx)?;
        elements.checked_mul(bytes)
    }
}

impl Verify for CuteTmaViewType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let Some(tensor) = self.tensor_view(ctx) else {
            return verify_err!(
                Location::Unknown,
                "cute.tma_view must wrap a cute.tensor_view"
            );
        };
        tensor.verify(ctx)?;
        if tensor.format != CuteTensorFormatAttr::Plain
            || tensor.role != CuteTensorRoleAttr::Generic
            || tensor.layout != CuteTensorLayoutAttr::Tma2D
        {
            return verify_err!(
                Location::Unknown,
                "cute.tma_view must move a Plain Generic Tma2D carrier tensor"
            );
        }
        if self.smem_layout.0.unit() != crate::layout::OffsetUnit::Elements {
            return verify_err!(
                Location::Unknown,
                "cute.tma_view shared layout offsets must count carrier elements"
            );
        }
        let Some(element_bytes) = tensor.storage_bytes(ctx) else {
            return verify_err!(
                Location::Unknown,
                "cute.tma_view carrier must have a known byte width"
            );
        };
        let Ok(element_bytes) = i64::try_from(element_bytes) else {
            return verify_err!(
                Location::Unknown,
                "cute.tma_view carrier byte width does not fit i64"
            );
        };
        if let Err(error) =
            crate::layout::validate_tma_encodable(&self.smem_layout.0, element_bytes)
        {
            return verify_err!(
                Location::Unknown,
                "cute.tma_view layout is not encodable by TMA: {error}"
            );
        }
        Ok(())
    }
}
