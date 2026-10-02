/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor-view types used to check and lower CuTe operations.
//!
//! A tensor view is a compiler description, not a runtime struct:
//!
//! ```text
//! pointer + length ── cute.tensor_make ──► !cute.tensor_view<...>
//!                                           │
//!                                           └─ gone before LLVM lowering
//! ```
//!
//! A selected backend consumes the description while lowering to native
//! operations or MLIR. A view has no runtime storage and must be removed
//! before the backend produces a kernel signature.
//!
//! After MIR preparation, the verifier follows each view through its uses:
//!
//! ```text
//! make -> zipped_divide -> slice -> load/store
//! ```
//!
//! At this stage, a view cannot be stored inside a pointer or aggregate,
//! passed between blocks, or appear in a function signature. Ordinary scalar
//! values may remain in compiler-owned local storage whose address does not
//! escape, or pass between blocks. The verifier follows those values too.

use dialect_mir::types::{MirArrayType, MirFP16Type, MirStructType, MirTupleType};
use pliron::builtin::types::{FP32Type, IntegerType};
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::derive::pliron_type;
use pliron::location::Location;
use pliron::result::Error;
use pliron::r#type::{Type, TypeHandle, TypedHandle};
use pliron::verify_err;

use crate::attributes::{
    CuteAlignmentAttr, CuteComposedLayoutAttr, CuteEpiloguePlanAttr, CuteScaledLayoutAttr,
    CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr,
    CuteTensorRoleAttr, CuteTileGridAttr,
};

/// Physical scalar registers inside one ordinary MIR aggregate.
///
/// MMA operations keep runtime values in ordinary Rust arrays and structs.
/// This count lets the verifier check their register requirements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MmaCarrierShape {
    pub u32_registers: u64,
    pub f32_registers: u64,
}

impl MmaCarrierShape {
    #[must_use]
    pub const fn u32(registers: u64) -> Self {
        Self {
            u32_registers: registers,
            f32_registers: 0,
        }
    }

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

    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            u32_registers: self.u32_registers.checked_add(other.u32_registers)?,
            f32_registers: self.f32_registers.checked_add(other.f32_registers)?,
        })
    }

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
    pub logical: TypeHandle,
    pub storage: TypeHandle,
    pub format: CuteTensorFormatAttr,
    pub role: CuteTensorRoleAttr,
    pub space: CuteTensorAddressSpaceAttr,
    pub access: CuteTensorAccessAttr,
    pub alignment: CuteAlignmentAttr,
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

    /// Size of one v0 storage element in bytes.
    #[must_use]
    pub fn storage_bytes(&self, ctx: &Context) -> Option<u64> {
        let storage = self.storage.deref(ctx);
        if storage.downcast_ref::<MirFP16Type>().is_some() {
            Some(2)
        } else if storage.downcast_ref::<FP32Type>().is_some() {
            Some(4)
        } else if let Some(integer) = storage.downcast_ref::<IntegerType>() {
            let width = u64::from(integer.width());
            (width > 0 && width % 8 == 0).then_some(width / 8)
        } else {
            None
        }
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
                let storage_ty = self.storage.deref(ctx);
                let plain_integer_is_supported = storage_ty
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|integer| {
                        integer.is_unsigned() && matches!(integer.width(), 8 | 16)
                    });
                let plain_is_supported = storage_ty.downcast_ref::<MirFP16Type>().is_some()
                    || logical_is_f32
                    || plain_integer_is_supported;
                if !plain_is_supported {
                    return verify_err!(
                        Location::Unknown,
                        "plain cute.tensor_view supports u8, u16, f16, and f32 carriers only"
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
            CuteTensorFormatAttr::UE8M0 => {
                let storage_matches_space = match self.space {
                    CuteTensorAddressSpaceAttr::Gmem => storage_is_u8,
                    CuteTensorAddressSpaceAttr::Smem => storage_is_u32,
                };
                if !logical_is_f32 || !storage_matches_space {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0 cute.tensor_view needs logical f32 scales in u8 global or u32 shared storage"
                    );
                }
                if self.access != CuteTensorAccessAttr::ReadOnly {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0 cute.tensor_view must be read-only"
                    );
                }
                if !matches!(self.role, CuteTensorRoleAttr::Mkl | CuteTensorRoleAttr::Nkl) {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0 cute.tensor_view needs an Mkl or Nkl role"
                    );
                }
                if !matches!(self.layout, CuteTensorLayoutAttr::BlockScaleKMajor(size) if size > 0)
                {
                    return verify_err!(
                        Location::Unknown,
                        "UE8M0 cute.tensor_view must use BlockScaleKMajor"
                    );
                }
            }
        }
        Ok(())
    }
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
    pub tensor: TypeHandle,
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
                    CuteTensorFormatAttr::UE8M0,
                    CuteTensorLayoutAttr::BlockScaleKMajor(_)
                )
        ) {
            return verify_err!(
                Location::Unknown,
                "cute.smem_tensor needs packed E2M1 values or UE8M0 block scales"
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

/// Static meaning of the shared result allocation used by one tiled MMA.
///
/// This type is carried only by operation attributes. Runtime values remain
/// the same ordinary pointer, warp/lane scalars, and FP32 accumulator used by
/// the Rust kernel:
///
/// ```text
/// AS3 f16 pointer + 64xf32 accumulator
///              │  TypeAttr<!cute.epilogue_tile<...>>
///              ▼
///       two 128x64 shared halves
/// ```
#[pliron_type(name = "cute.epilogue_tile", format = "`<` $storage `,` $plan `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteEpilogueTileType {
    pub storage: TypeHandle,
    pub plan: CuteEpiloguePlanAttr,
}

impl CuteEpilogueTileType {
    /// Attach the current epilogue placement to its physical storage type.
    pub fn get(
        ctx: &mut Context,
        storage: TypeHandle,
        plan: CuteEpiloguePlanAttr,
    ) -> TypedHandle<Self> {
        Type::instantiate(Self { storage, plan }, ctx)
    }

    #[must_use]
    pub fn half_elements(&self) -> Option<u64> {
        self.plan.half_elements()
    }

    #[must_use]
    pub fn full_elements(&self) -> Option<u64> {
        self.plan.full_elements()
    }

    #[must_use]
    pub fn half_bytes(&self) -> Option<u64> {
        self.plan.half_bytes(2)
    }

    #[must_use]
    pub fn full_bytes(&self) -> Option<u64> {
        self.plan.full_bytes(2)
    }
}

impl Verify for CuteEpilogueTileType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        if self
            .storage
            .deref(ctx)
            .downcast_ref::<MirFP16Type>()
            .is_none()
        {
            return verify_err!(
                Location::Unknown,
                "cute.epilogue_tile v0 storage must be f16"
            );
        }
        self.plan.verify(ctx)?;
        if self.half_elements() != Some(8192)
            || self.full_elements() != Some(16384)
            || self.half_bytes() != Some(16 * 1024)
            || self.full_bytes() != Some(32 * 1024)
        {
            return verify_err!(
                Location::Unknown,
                "cute.epilogue_tile v0 must be two 16 KiB halves in one 32 KiB allocation"
            );
        }
        Ok(())
    }
}

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
    pub tensor: TypeHandle,
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
        u64::try_from(self.smem_layout.0.inner().checked_size()?).ok()
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
    pub values: TypeHandle,
    pub scales: TypeHandle,
    pub role: CuteTensorRoleAttr,
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
        if scales.format != CuteTensorFormatAttr::UE8M0
            || scales.layout != CuteTensorLayoutAttr::BlockScaleKMajor(16)
        {
            return verify_err!(
                Location::Unknown,
                "cute.scaled_view scales must be a UE8M0 BlockScaleKMajor<16> tensor"
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

/// One logical output tile selected by a static scheduler.
///
/// The type remembers the static tile grid. Its defining scheduler operation
/// carries the current linear tile number; no `WorkTile` struct reaches the
/// runtime ABI.
///
/// ```text
/// current linear number + grid<16,16,1>
///                    │
///                    ▼
///        !cute.work_tile<grid<16,16,1>>
///                    │ coordinates
///                    ▼
///              linear, m, n, batch
/// ```
#[pliron_type(name = "cute.work_tile", format = "`<` $grid `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteWorkTileType {
    pub grid: CuteTileGridAttr,
}

impl CuteWorkTileType {
    /// Create the short-lived semantic handle for one scheduler iteration.
    pub fn get(ctx: &mut Context, grid: CuteTileGridAttr) -> TypedHandle<Self> {
        Type::instantiate(Self { grid }, ctx)
    }
}

impl Verify for CuteWorkTileType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.grid.verify(ctx)
    }
}

/// The shared barrier ring used by TMA loads.
///
/// This is a compiler-only handle. The real storage is still the shared
/// pointer consumed by `cute.tma_load_pipeline_make`:
///
/// ```text
/// shared barrier base ── make<3, 8 warps, 17408 bytes>
///                              │
///                              ▼
///               !cute.tma_load_pipeline<3, 8, 17408>
/// ```
///
/// The handle must disappear before ABI or LLVM lowering. Slot and phase are
/// deliberately *not* part of this type; ordinary `u32` values carry them
/// through the loop.
#[pliron_type(
    name = "cute.tma_load_pipeline",
    format = "`<` $stages `,` $consumer_warps `,` $transaction_bytes `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteTmaLoadPipelineType {
    pub stages: u64,
    pub consumer_warps: u32,
    pub transaction_bytes: u32,
}

impl CuteTmaLoadPipelineType {
    /// Create a compiler-only TMA load-pipeline handle.
    pub fn get(
        ctx: &mut Context,
        stages: u64,
        consumer_warps: u32,
        transaction_bytes: u32,
    ) -> TypedHandle<Self> {
        Type::instantiate(
            Self {
                stages,
                consumer_warps,
                transaction_bytes,
            },
            ctx,
        )
    }

    /// Return bytes occupied by `[full; stages]` and `[empty; stages]`.
    ///
    /// Each hardware barrier occupies eight bytes.
    #[must_use]
    pub fn storage_bytes(&self) -> Option<u64> {
        self.stages
            .checked_mul(2)
            .and_then(|count| count.checked_mul(8))
    }
}

impl Verify for CuteTmaLoadPipelineType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0 || self.stages > u64::from(u32::MAX) {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline stages must be between 1 and {}",
                u32::MAX
            );
        }
        if !(1..=32).contains(&self.consumer_warps) {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline consumer warps must be between 1 and 32"
            );
        }
        if self.transaction_bytes == 0 {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline transaction bytes must be greater than zero"
            );
        }
        if self.storage_bytes().is_none() {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline barrier storage size overflowed u64"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
    use pliron::builtin::types::Signedness;

    #[test]
    fn tensor_view_records_the_whole_elementwise_view() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let view = CuteTensorViewType::get(
            &mut ctx,
            f32_ty,
            f32_ty,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            4,
            CuteTensorLayoutAttr::Tile1D(4),
        );

        assert_eq!(view.deref(&ctx).selected_tile_size(), Some(4));
        assert!(view.deref(&ctx).verify(&ctx).is_ok());
    }

    #[test]
    fn tensor_view_rejects_invalid_v0_facts() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();

        let mismatched = CuteTensorViewType::get(
            &mut ctx,
            f32_ty,
            f16_ty,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            4,
            CuteTensorLayoutAttr::Contiguous1D,
        );
        assert!(
            mismatched
                .deref(&ctx)
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("matching logical and storage")
        );

        let zero_tile = CuteTensorViewType::get(
            &mut ctx,
            f32_ty,
            f32_ty,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            4,
            CuteTensorLayoutAttr::Tile1D(0),
        );
        assert!(
            zero_tile
                .deref(&ctx)
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("tile size")
        );
    }

    #[test]
    fn packed_views_keep_format_role_scale_and_fragment_facts() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();

        let values: TypeHandle = CuteTensorViewType::get_with_facts(
            &mut ctx,
            f32_ty,
            u8_ty,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            16,
            CuteTensorFormatAttr::E2M1,
            CuteTensorRoleAttr::Mkl,
            CuteTensorLayoutAttr::KMajor,
        )
        .into();
        let scales: TypeHandle = CuteTensorViewType::get_with_facts(
            &mut ctx,
            f32_ty,
            u8_ty,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            4,
            CuteTensorFormatAttr::UE8M0,
            CuteTensorRoleAttr::Mkl,
            CuteTensorLayoutAttr::BlockScaleKMajor(16),
        )
        .into();
        let tile: TypeHandle = CuteScaledViewType::get(
            &mut ctx,
            values,
            scales,
            CuteTensorRoleAttr::Mkl,
            CuteScaledLayoutAttr::KTile(64),
        )
        .into();
        let fragment = CuteFragmentType::get(&mut ctx, tile);

        assert!(values.deref(&ctx).verify(&ctx).is_ok());
        assert!(scales.deref(&ctx).verify(&ctx).is_ok());
        assert!(tile.deref(&ctx).verify(&ctx).is_ok());
        assert!(fragment.deref(&ctx).verify(&ctx).is_ok());
        let source = fragment.deref(&ctx).source_view(&ctx).unwrap();
        assert_eq!(source.role, CuteTensorRoleAttr::Mkl);
        assert_eq!(source.values_per_scale(&ctx), Some(16));
        assert_eq!(source.k_width(), Some(64));
    }

    #[test]
    fn shared_packed_views_keep_their_real_transport_carriers() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        let values: TypeHandle = CuteTensorViewType::get_with_facts(
            &mut ctx,
            f32_ty,
            f16_ty,
            CuteTensorAddressSpaceAttr::Smem,
            CuteTensorAccessAttr::ReadOnly,
            16,
            CuteTensorFormatAttr::E2M1,
            CuteTensorRoleAttr::Mkl,
            CuteTensorLayoutAttr::KMajor,
        )
        .into();
        let scales: TypeHandle = CuteTensorViewType::get_with_facts(
            &mut ctx,
            f32_ty,
            u32_ty,
            CuteTensorAddressSpaceAttr::Smem,
            CuteTensorAccessAttr::ReadOnly,
            4,
            CuteTensorFormatAttr::UE8M0,
            CuteTensorRoleAttr::Mkl,
            CuteTensorLayoutAttr::BlockScaleKMajor(32),
        )
        .into();
        assert!(values.deref(&ctx).verify(&ctx).is_ok());
        assert!(scales.deref(&ctx).verify(&ctx).is_ok());

        let data_inner: Layout = "(128,32):(32,1)".parse().unwrap();
        let data_placement =
            ComposedLayout::new(Swizzle::new(2, 3, 3), 0, data_inner, OffsetUnit::Elements)
                .unwrap();
        let shared = CuteSmemTensorType::get(&mut ctx, values, data_placement);
        assert!(shared.deref(&ctx).verify(&ctx).is_ok());
        assert_eq!(shared.deref(&ctx).storage_elements(), Some(4096));
        assert_eq!(shared.deref(&ctx).storage_bytes(&ctx), Some(8192));

        let wrong_shared_carrier = CuteTensorViewType::get_with_facts(
            &mut ctx,
            f32_ty,
            u8_ty,
            CuteTensorAddressSpaceAttr::Smem,
            CuteTensorAccessAttr::ReadOnly,
            1,
            CuteTensorFormatAttr::E2M1,
            CuteTensorRoleAttr::Mkl,
            CuteTensorLayoutAttr::KMajor,
        );
        assert!(wrong_shared_carrier.deref(&ctx).verify(&ctx).is_err());
    }

    #[test]
    fn mma_carrier_shape_walks_nested_rust_aggregates_and_ignores_markers() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let words: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 4).into();
        let marker: TypeHandle =
            MirStructType::get(&mut ctx, "PhantomData<Role>".into(), vec![], vec![]).into();
        let fragment: TypeHandle = MirTupleType::get(&mut ctx, vec![words, marker]).into();
        let two_fragments: TypeHandle = MirArrayType::get(&mut ctx, fragment, 2).into();
        assert_eq!(
            mma_carrier_shape(&ctx, two_fragments),
            Some(MmaCarrierShape::u32(8))
        );

        let mixed: TypeHandle = MirTupleType::get(&mut ctx, vec![two_fragments, f32_ty]).into();
        assert_eq!(
            mma_carrier_shape(&ctx, mixed),
            Some(MmaCarrierShape {
                u32_registers: 8,
                f32_registers: 1,
            })
        );
        assert_eq!(mma_carrier_shape(&ctx, mixed).unwrap().bytes(), Some(36));
    }
}
