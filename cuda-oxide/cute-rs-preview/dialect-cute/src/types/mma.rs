/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared tensor placement, register carriers, and tiled matrix-instruction contracts.

use dialect_mir::types::{MirArrayType, MirStructType, MirTupleType};
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

use super::tensor::CuteTensorViewType;
use crate::attributes::{
    CuteComposedLayoutAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr,
    CuteTensorLayoutAttr,
};

/// Physical scalar registers inside one ordinary MIR aggregate.
///
/// MMA operations keep runtime values in ordinary Rust arrays and structs.
/// This count lets the verifier check their register requirements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MmaCarrierShape {
    /// Count of packed 32-bit registers in the ordinary carrier.
    pub u32_registers: u64,
    /// Count of FP32 registers in the ordinary carrier.
    pub f32_registers: u64,
}

impl MmaCarrierShape {
    /// Describe a carrier containing only the given number of packed 32-bit registers.
    #[must_use]
    pub const fn u32(registers: u64) -> Self {
        Self {
            u32_registers: registers,
            f32_registers: 0,
        }
    }

    /// Describe a carrier containing only the given number of FP32 registers.
    #[must_use]
    pub const fn f32(registers: u64) -> Self {
        Self {
            u32_registers: 0,
            f32_registers: registers,
        }
    }

    /// Bytes carried by all counted registers.
    #[must_use]
    pub const fn bytes(self) -> Option<u64> {
        let Some(registers) = self.u32_registers.checked_add(self.f32_registers) else {
            return None;
        };
        registers.checked_mul(4)
    }

    /// Combine register counts for aggregate fields, returning `None` on overflow.
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            u32_registers: self.u32_registers.checked_add(other.u32_registers)?,
            f32_registers: self.f32_registers.checked_add(other.f32_registers)?,
        })
    }

    /// Scale register counts for an array of carriers, returning `None` on overflow.
    fn checked_repeat(self, count: u64) -> Option<Self> {
        Some(Self {
            u32_registers: self.u32_registers.checked_mul(count)?,
            f32_registers: self.f32_registers.checked_mul(count)?,
        })
    }
}

/// Count the u32 and f32 registers in a Rust-shaped MIR carrier.
///
/// Arrays, tuples, and structs are walked recursively. Empty marker structs
/// such as `PhantomData<Role>` contribute nothing. Any other scalar or
/// aggregate kind returns `None`.
#[must_use]
pub fn mma_carrier_shape(ctx: &Context, ty: TypeHandle) -> Option<MmaCarrierShape> {
    let ty = ty.deref(ctx);
    if let Some(integer) = ty.downcast_ref::<IntegerType>() {
        return (integer.width() == 32 && integer.is_unsigned()).then_some(MmaCarrierShape::u32(1));
    }
    if ty.downcast_ref::<FP32Type>().is_some() {
        return Some(MmaCarrierShape::f32(1));
    }
    if let Some(array) = ty.downcast_ref::<MirArrayType>() {
        return mma_carrier_shape(ctx, array.element_ty)?.checked_repeat(array.size);
    }
    let children = if let Some(tuple) = ty.downcast_ref::<MirTupleType>() {
        Some(tuple.types.as_slice())
    } else {
        ty.downcast_ref::<MirStructType>()
            .map(|structure| structure.field_types.as_slice())
    }?;
    children
        .iter()
        .try_fold(MmaCarrierShape::default(), |sum, child| {
            sum.checked_add(mma_carrier_shape(ctx, *child)?)
        })
}

/// Static meaning attached to a shared-memory pointer/capacity pair.
///
/// This type lives in a `TypeAttr` on `cute.smem_tensor_overlay`; it is not an
/// SSA result and therefore never has to cross a loop edge:
///
/// ```text
/// ordinary AS3 pointer + ordinary u64 capacity
///                  │  {view = !cute.smem_tensor<...>}
///                  ▼
/// ordinary AS3 pointer + ordinary u64 capacity
/// ```
///
/// `placement` explains where logical rows and K positions live inside one
/// shared stage. The wrapped tensor explains whether those carriers are E2M1
/// values or UE8M0 scales and whether their rows follow M or N.
#[pliron_type(name = "cute.smem_tensor", format = "`<` $tensor `,` $placement `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteSmemTensorType {
    /// Wrapped tensor-view type containing storage, format, role, and access facts.
    pub tensor: TypeHandle,
    /// Mapping from logical coordinates to physical shared storage offsets.
    pub placement: CuteComposedLayoutAttr,
}

impl CuteSmemTensorType {
    /// Attach one shared-memory placement to packed tensor facts.
    pub fn get(
        ctx: &mut Context,
        tensor: TypeHandle,
        placement: crate::layout::ComposedLayout,
    ) -> TypedHandle<Self> {
        Type::instantiate(
            Self {
                tensor,
                placement: CuteComposedLayoutAttr(placement),
            },
            ctx,
        )
    }

    /// Return the packed tensor facts wrapped by this placement.
    #[must_use]
    pub fn tensor_view(&self, ctx: &Context) -> Option<CuteTensorViewType> {
        self.tensor
            .deref(ctx)
            .downcast_ref::<CuteTensorViewType>()
            .cloned()
    }

    /// Number of physical carrier elements in one shared stage.
    #[must_use]
    pub fn storage_elements(&self) -> Option<u64> {
        u64::try_from(self.placement.0.inner().checked_size()?).ok()
    }

    /// Physical byte size of one shared stage.
    #[must_use]
    pub fn storage_bytes(&self, ctx: &Context) -> Option<u64> {
        self.storage_elements()?
            .checked_mul(self.tensor_view(ctx)?.storage_bytes(ctx)?)
    }
}

impl Verify for CuteSmemTensorType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let Some(tensor) = self.tensor_view(ctx) else {
            return verify_err!(
                Location::Unknown,
                "cute.smem_tensor must wrap a cute.tensor_view"
            );
        };
        tensor.verify(ctx)?;
        if tensor.space != CuteTensorAddressSpaceAttr::Smem
            || tensor.access != CuteTensorAccessAttr::ReadOnly
        {
            return verify_err!(
                Location::Unknown,
                "cute.smem_tensor v0 must describe a read-only shared tensor"
            );
        }
        if !matches!(
            (tensor.format, tensor.layout),
            (CuteTensorFormatAttr::E2M1, CuteTensorLayoutAttr::KMajor)
                | (
                    CuteTensorFormatAttr::UE8M0 | CuteTensorFormatAttr::E4M3,
                    CuteTensorLayoutAttr::BlockScaleKMajor(_)
                )
        ) {
            return verify_err!(
                Location::Unknown,
                "cute.smem_tensor needs packed E2M1 values or UE8M0/E4M3 block scales"
            );
        }
        if self.placement.0.unit() != crate::layout::OffsetUnit::Elements
            || self.placement.0.offset() != 0
            || self.storage_elements().is_none()
        {
            return verify_err!(
                Location::Unknown,
                "cute.smem_tensor placement must be a positive zero-based element layout"
            );
        }
        Ok(())
    }
}
