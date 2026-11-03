/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed-value and scale-view contracts for the block-scaled GEMV flow.

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
    CuteScaledLayoutAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr, CuteTensorRoleAttr,
};

/// A packed value tensor bound to its scale tensor.
///
/// This remains a view. It owns no storage and loads nothing:
///
/// ```text
/// E2M1 tensor ──┐
///                ├── !cute.scaled_view<..., Full>
/// UE8M0 tensor ─┘
/// ```
#[pliron_type(
    name = "cute.scaled_view",
    format = "`<` $values `,` $scales `,` $role `,` $layout `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteScaledViewType {
    /// Type of the packed value tensor bound to the scale tensor.
    pub values: TypeHandle,
    /// Type of the scale tensor paired with the packed values.
    pub scales: TypeHandle,
    /// Logical M/N coordinate role assigned to the tensor.
    pub role: CuteTensorRoleAttr,
    /// Static coordinate layout used to interpret the storage.
    pub layout: CuteScaledLayoutAttr,
}

impl CuteScaledViewType {
    /// Bind value and scale tensor types at one visible selection stage.
    pub fn get(
        ctx: &mut Context,
        values: TypeHandle,
        scales: TypeHandle,
        role: CuteTensorRoleAttr,
        layout: CuteScaledLayoutAttr,
    ) -> TypedHandle<Self> {
        Type::instantiate(
            Self {
                values,
                scales,
                role,
                layout,
            },
            ctx,
        )
    }

    /// Copy the bound tensors and role, changing only the selection stage.
    #[must_use]
    pub fn with_layout(
        &self,
        ctx: &mut Context,
        layout: CuteScaledLayoutAttr,
    ) -> TypedHandle<Self> {
        Self::get(ctx, self.values, self.scales, self.role, layout)
    }

    /// Return how many logical K values share one scale.
    #[must_use]
    pub fn values_per_scale(&self, ctx: &Context) -> Option<u64> {
        let scales = self.scales.deref(ctx);
        let view = scales.downcast_ref::<CuteTensorViewType>()?;
        view.layout.values_per_scale()
    }

    /// Return the selected logical K width, if this is a K tile.
    #[must_use]
    pub const fn k_width(&self) -> Option<u64> {
        match self.layout {
            CuteScaledLayoutAttr::KTile(width) => Some(width),
            _ => None,
        }
    }
}

impl Verify for CuteScaledViewType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let values_ty = self.values.deref(ctx);
        let Some(values) = values_ty.downcast_ref::<CuteTensorViewType>() else {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view values must be a tensor view"
            );
        };
        let scales_ty = self.scales.deref(ctx);
        let Some(scales) = scales_ty.downcast_ref::<CuteTensorViewType>() else {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view scales must be a tensor view"
            );
        };
        values.verify(ctx)?;
        scales.verify(ctx)?;

        if values.format != CuteTensorFormatAttr::E2M1
            || values.layout != CuteTensorLayoutAttr::KMajor
        {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view values must be an E2M1 KMajor tensor"
            );
        }
        if !matches!(
            scales.format,
            CuteTensorFormatAttr::UE8M0 | CuteTensorFormatAttr::E4M3
        ) || scales.layout != CuteTensorLayoutAttr::BlockScaleKMajor(16)
        {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view scales must be a UE8M0 or E4M3 BlockScaleKMajor<16> tensor"
            );
        }
        if self.role == CuteTensorRoleAttr::Generic
            || values.role != self.role
            || scales.role != self.role
        {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view values, scales, and result must share one Mkl or Nkl role"
            );
        }
        if values.logical != scales.logical {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view values and scales must decode to the same logical type"
            );
        }
        self.layout.verify(ctx)?;
        if let CuteScaledLayoutAttr::KTile(width) = self.layout
            && width % 16 != 0
        {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view K tile width must contain whole 16-value scale groups"
            );
        }
        Ok(())
    }
}

/// One loaded block-scaled tile held in registers.
///
/// The source type keeps the packed format, scale format, role, and K width
/// available without copying those facts into a second list.
#[pliron_type(name = "cute.fragment", format = "`<` $source `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteFragmentType {
    /// Type of the selected block-scaled view from which the fragment was loaded.
    pub source: TypeHandle,
}

impl CuteFragmentType {
    /// Describe the register fragment loaded from one scaled K tile.
    pub fn get(ctx: &mut Context, source: TypeHandle) -> TypedHandle<Self> {
        Type::instantiate(Self { source }, ctx)
    }

    /// Read the source view carrying this fragment's semantic facts.
    #[must_use]
    pub fn source_view(&self, ctx: &Context) -> Option<CuteScaledViewType> {
        self.source
            .deref(ctx)
            .downcast_ref::<CuteScaledViewType>()
            .cloned()
    }
}

impl Verify for CuteFragmentType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let Some(source) = self.source_view(ctx) else {
            return verify_err!(
                Location::Unknown,
                "cute.fragment source must be a scaled view"
            );
        };
        source.verify(ctx)?;
        if !matches!(source.layout, CuteScaledLayoutAttr::KTile(width) if width > 0) {
            return verify_err!(
                Location::Unknown,
                "cute.fragment source must be a selected K tile"
            );
        }
        Ok(())
    }
}
