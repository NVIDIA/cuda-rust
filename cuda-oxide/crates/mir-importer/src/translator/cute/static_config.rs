/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural decoder for nested `cute-rs` type markers.
//!
//! The decoder follows ADT identities and generic-argument kinds:
//!
//! ```text
//! L<T2<C<32>, C<8>>, T2<C<8>, C<1>>>
//! │ └────── shape ──────┘  └───── stride ─────┘
//! └─ exact DefId: cute_rs::markers::L
//! ```
//!
//! It never searches a debug-printed type name. A near-miss from another
//! crate, a swapped `Type`/`Const`, or an unevaluated constant is a loud
//! boundary error.

use dialect_cute::layout::{
    ComposedLayout, CooperativeCopyPlan, IntTuple, Layout, OffsetUnit, Swizzle,
    validate_cooperative_copy_plan, validate_ldmatrix_source,
};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::{
    AdtDef, FloatTy, GenericArgKind, GenericArgs, IntTy, RigidTy, Ty, TyConst, TyConstKind, TyKind,
    UintTy,
};

const C: &str = "cute_rs::markers::C";
const T2: &str = "cute_rs::markers::T2";
const LAYOUT: &str = "cute_rs::markers::L";
const ROW_MAJOR: &str = "cute_rs::markers::RowMajor";
const COL_MAJOR: &str = "cute_rs::markers::ColMajor";
const CP_ASYNC: &str = "cute_rs::markers::CpAsync";
const LEADING_DIM: &str = "cute_rs::markers::LeadingDim";
const SWIZZLE: &str = "cute_rs::markers::Swizzle";
const COMPOSED: &str = "cute_rs::markers::Composed";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopyAtom {
    pub bytes: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct CooperativeConfig {
    pub atom: CopyAtom,
    pub thread_layout: Layout,
    pub value_layout: Layout,
    pub tile_layout: Layout,
    pub smem_layout: ComposedLayout,
    /// Divisor of the runtime row pitch, measured in elements.
    pub leading_dim_divisor: u64,
    /// The exact checked maps M3 must lower; never recompute them later.
    #[allow(dead_code)]
    pub plan: CooperativeCopyPlan,
    // M3 emission needs the decoded Rust type; the pre-M3 sentinel only prints it.
    #[allow(dead_code)]
    pub element_type: Ty,
    // Keep the validated byte width beside the type so emission cannot re-guess it.
    #[allow(dead_code)]
    pub element_bytes: i64,
}

fn cooperative_element(ty: &Ty) -> Result<(i64, &'static str), String> {
    let info = match ty.kind() {
        TyKind::RigidTy(RigidTy::Int(kind)) => match kind {
            IntTy::I8 => (1, "i8"),
            IntTy::I16 => (2, "i16"),
            IntTy::I32 => (4, "i32"),
            IntTy::I64 => (8, "i64"),
            IntTy::I128 => (16, "i128"),
            IntTy::Isize => (8, "isize"),
        },
        TyKind::RigidTy(RigidTy::Uint(kind)) => match kind {
            UintTy::U8 => (1, "u8"),
            UintTy::U16 => (2, "u16"),
            UintTy::U32 => (4, "u32"),
            UintTy::U64 => (8, "u64"),
            UintTy::U128 => (16, "u128"),
            UintTy::Usize => (8, "usize"),
        },
        TyKind::RigidTy(RigidTy::Float(kind)) => match kind {
            FloatTy::F16 => (2, "f16"),
            FloatTy::F32 => (4, "f32"),
            FloatTy::F64 => (8, "f64"),
            FloatTy::F128 => {
                return Err("copy_g2s does not support f128 elements".to_string());
            }
        },
        // Rust `char` already lowers exactly to an unsigned 32-bit dialect integer.
        TyKind::RigidTy(RigidTy::Char) => (4, "char"),
        _ => {
            return Err(format!(
                "copy_g2s element must be a supported scalar, found {ty:?}"
            ));
        }
    };
    Ok(info)
}

fn canonical_adt_path(def: &AdtDef) -> String {
    let crate_name = def.krate().name.to_string();
    let mut segments = Vec::new();
    let mut current = Some(def.def_id());
    while let Some(def_id) = current {
        let printed = def_id.name();
        let segment = printed.as_str().rsplit("::").next().unwrap_or_default();
        segments.push(segment.to_owned());
        current = def_id.parent();
    }
    super::canonical_path_from_leaf_segments(&crate_name, segments)
}

fn adt_owned(ty: &Ty, expected: &str) -> Result<GenericArgs, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Err(format!("expected `{expected}`, found non-ADT type {ty:?}"));
    };
    let found = canonical_adt_path(&def);
    if found != expected {
        return Err(format!("expected `{expected}`, found `{found}`"));
    }
    Ok(args)
}

fn expect_schema(args: &GenericArgs, expected: &[&str], owner: &str) -> Result<(), String> {
    let found: Vec<_> = args
        .0
        .iter()
        .map(|arg| match arg {
            GenericArgKind::Type(_) => "Type",
            GenericArgKind::Const(_) => "Const",
            GenericArgKind::Lifetime(_) => "Lifetime",
        })
        .collect();
    if found == expected {
        Ok(())
    } else {
        Err(format!(
            "`{owner}` expects generics [{}], found [{}]",
            expected.join(", "),
            found.join(", ")
        ))
    }
}

fn unsigned_const(value: &TyConst, what: &str) -> Result<u64, String> {
    let raw = match value.kind() {
        TyConstKind::Value(_, allocation) => allocation
            .read_uint()
            .map_err(|error| format!("cannot read {what}: {error:?}"))?,
        _ => u128::from(
            value
                .eval_target_usize()
                .map_err(|error| format!("cannot evaluate {what}: {error:?}"))?,
        ),
    };
    u64::try_from(raw).map_err(|_| format!("{what} value {raw} does not fit in u64"))
}

fn signed_const(value: &TyConst, what: &str) -> Result<i64, String> {
    let raw = match value.kind() {
        TyConstKind::Value(_, allocation) => allocation
            .read_int()
            .map_err(|error| format!("cannot read {what}: {error:?}"))?,
        _ => i128::from(
            value
                .eval_target_usize()
                .map_err(|error| format!("cannot evaluate {what}: {error:?}"))?,
        ),
    };
    i64::try_from(raw).map_err(|_| format!("{what} value {raw} does not fit in i64"))
}

fn type_arg(args: &GenericArgs, index: usize, owner: &str) -> Result<Ty, String> {
    match args.0.get(index) {
        Some(GenericArgKind::Type(value)) => Ok(*value),
        _ => Err(format!("`{owner}` generic {index} must be a type")),
    }
}

fn const_arg(args: &GenericArgs, index: usize, owner: &str) -> Result<TyConst, String> {
    match args.0.get(index) {
        Some(GenericArgKind::Const(value)) => Ok(value.clone()),
        _ => Err(format!("`{owner}` generic {index} must be a const")),
    }
}

fn decode_tuple(ty: &Ty) -> Result<IntTuple, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Err(format!("layout tuple marker must be an ADT, found {ty:?}"));
    };
    match canonical_adt_path(&def).as_str() {
        C => {
            expect_schema(&args, &["Const"], C)?;
            Ok(IntTuple::Leaf(signed_const(
                &const_arg(&args, 0, C)?,
                "C value",
            )?))
        }
        T2 => {
            expect_schema(&args, &["Type", "Type"], T2)?;
            Ok(IntTuple::Tuple(vec![
                decode_tuple(&type_arg(&args, 0, T2)?)?,
                decode_tuple(&type_arg(&args, 1, T2)?)?,
            ]))
        }
        found => Err(format!("expected `{C}` or `{T2}`, found `{found}`")),
    }
}

fn checked_shape_size(shape: &IntTuple) -> Result<i64, String> {
    match shape {
        IntTuple::Leaf(value) if *value > 0 => Ok(*value),
        IntTuple::Leaf(value) => Err(format!("layout shape leaves must be positive, got {value}")),
        IntTuple::Tuple(items) => items.iter().try_fold(1_i64, |size, item| {
            size.checked_mul(checked_shape_size(item)?)
                .ok_or_else(|| "layout shape product overflows i64".to_string())
        }),
    }
}

pub(crate) fn decode_layout(ty: &Ty) -> Result<Layout, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Err(format!("layout marker must be an ADT, found {ty:?}"));
    };
    let path = canonical_adt_path(&def);
    let layout = match path.as_str() {
        LAYOUT => {
            expect_schema(&args, &["Type", "Type"], LAYOUT)?;
            let shape = decode_tuple(&type_arg(&args, 0, LAYOUT)?)?;
            let stride = decode_tuple(&type_arg(&args, 1, LAYOUT)?)?;
            if !shape.congruent(&stride) {
                return Err("layout shape and stride have different nesting".to_string());
            }
            Layout::new(shape, stride)
        }
        ROW_MAJOR | COL_MAJOR => {
            expect_schema(&args, &["Const", "Const"], &path)?;
            let rows = signed_const(&const_arg(&args, 0, &path)?, "row count")?;
            let cols = signed_const(&const_arg(&args, 1, &path)?, "column count")?;
            if rows <= 0 || cols <= 0 {
                return Err(format!("`{path}` dimensions must be positive"));
            }
            let shape = IntTuple::Tuple(vec![IntTuple::Leaf(rows), IntTuple::Leaf(cols)]);
            let stride = if path == ROW_MAJOR {
                IntTuple::Tuple(vec![IntTuple::Leaf(cols), IntTuple::Leaf(1)])
            } else {
                IntTuple::Tuple(vec![IntTuple::Leaf(1), IntTuple::Leaf(rows)])
            };
            Layout::new(shape, stride)
        }
        _ => return Err(format!("expected a cute-rs layout marker, found `{path}`")),
    };
    checked_shape_size(&layout.shape)?;
    Ok(layout)
}

fn decode_atom(ty: &Ty) -> Result<CopyAtom, String> {
    let args = adt_owned(ty, CP_ASYNC)?;
    expect_schema(&args, &["Const"], CP_ASYNC)?;
    let bytes = u32::try_from(unsigned_const(
        &const_arg(&args, 0, CP_ASYNC)?,
        "cp.async width",
    )?)
    .map_err(|_| "cp.async width does not fit in u32".to_string())?;
    if !matches!(bytes, 4 | 8 | 16) {
        return Err(format!(
            "cp.async width must be 4, 8, or 16 bytes, got {bytes}"
        ));
    }
    Ok(CopyAtom { bytes })
}

fn decode_leading_dim_divisor(ty: &Ty) -> Result<u64, String> {
    let args = adt_owned(ty, LEADING_DIM)?;
    expect_schema(&args, &["Const"], LEADING_DIM)?;
    let divisor = unsigned_const(
        &const_arg(&args, 0, LEADING_DIM)?,
        "leading-dimension divisor",
    )?;
    if divisor == 0 {
        return Err("leading-dimension divisor must be greater than zero".to_string());
    }
    Ok(divisor)
}

fn decode_swizzle(ty: &Ty) -> Result<Swizzle, String> {
    let args = adt_owned(ty, SWIZZLE)?;
    expect_schema(&args, &["Const", "Const", "Const"], SWIZZLE)?;
    let bits = u32::try_from(unsigned_const(&const_arg(&args, 0, SWIZZLE)?, "swizzle B")?)
        .map_err(|_| "swizzle B does not fit in u32".to_string())?;
    let base = u32::try_from(unsigned_const(&const_arg(&args, 1, SWIZZLE)?, "swizzle M")?)
        .map_err(|_| "swizzle M does not fit in u32".to_string())?;
    let shift = i32::try_from(signed_const(&const_arg(&args, 2, SWIZZLE)?, "swizzle S")?)
        .map_err(|_| "swizzle S does not fit in i32".to_string())?;
    Swizzle::try_new(bits, base, shift).ok_or_else(|| {
        format!("invalid CuTe swizzle S<{bits},{base},{shift}>: require |S| >= B and M+B+|S| <= 32")
    })
}

pub(crate) fn decode_smem_layout(ty: &Ty) -> Result<ComposedLayout, String> {
    if let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind()
        && canonical_adt_path(&def) == COMPOSED
    {
        expect_schema(&args, &["Type", "Const", "Type"], COMPOSED)?;
        let outer = decode_swizzle(&type_arg(&args, 0, COMPOSED)?)?;
        let offset = signed_const(&const_arg(&args, 1, COMPOSED)?, "composed offset")?;
        let inner = decode_layout(&type_arg(&args, 2, COMPOSED)?)?;
        // Rust marker layouts count elements. Keeping the unit explicit here
        // lets the dialect verifier recast the entire expression—including
        // the swizzle bit fields—to bytes once `T` is known.
        return ComposedLayout::new(outer, offset, inner, OffsetUnit::Elements)
            .map_err(|error| error.to_string());
    }
    Ok(ComposedLayout::from_layout(
        decode_layout(ty)?,
        OffsetUnit::Elements,
    ))
}

/// Decode the six explicit parameters and inferred pitch marker at the
/// frozen `copy_g2s` boundary.
pub(crate) fn decode_cooperative_config(args: &GenericArgs) -> Result<CooperativeConfig, String> {
    expect_schema(
        args,
        &["Type", "Type", "Type", "Type", "Type", "Type", "Type"],
        "cute_rs::cooperative::copy_g2s",
    )?;
    let atom = decode_atom(&type_arg(args, 0, "copy_g2s")?)?;
    let thread_layout = decode_layout(&type_arg(args, 1, "copy_g2s")?)?;
    let value_layout = decode_layout(&type_arg(args, 2, "copy_g2s")?)?;
    let tile_layout = decode_layout(&type_arg(args, 3, "copy_g2s")?)?;
    let smem_layout = decode_smem_layout(&type_arg(args, 4, "copy_g2s")?)?;
    let element_type = type_arg(args, 5, "copy_g2s")?;
    let (element_bytes, _spelling) = cooperative_element(&element_type)?;
    let leading_dim_divisor = decode_leading_dim_divisor(&type_arg(args, 6, "copy_g2s")?)?;
    let leading_dim_alignment = leading_dim_divisor
        .checked_mul(element_bytes as u64)
        .ok_or_else(|| "leading-dimension byte alignment overflows u64".to_string())?;
    if leading_dim_alignment % u64::from(atom.bytes) != 0 {
        return Err(format!(
            "leading-dimension promise is too weak: {leading_dim_divisor} elements x \
             {element_bytes} bytes is not a multiple of the {}-byte atom",
            atom.bytes
        ));
    }

    let plan = validate_cooperative_copy_plan(
        atom.bytes,
        &thread_layout,
        &value_layout,
        &tile_layout,
        &smem_layout,
        element_bytes,
    )
    .map_err(|error| format!("invalid cooperative copy plan: {error}"))?;

    Ok(CooperativeConfig {
        atom,
        thread_layout,
        value_layout,
        tile_layout,
        smem_layout,
        leading_dim_divisor,
        plan,
        element_type,
        element_bytes,
    })
}

/// Reach the static decoder from a direct call operand.
pub(crate) fn decode_copy_g2s_callee(func: &mir::Operand) -> Result<CooperativeConfig, String> {
    let mir::Operand::Constant(constant) = func else {
        return Err("copy_g2s must be a direct constant function call".to_string());
    };
    let TyKind::RigidTy(RigidTy::FnDef(_, args)) = constant.const_.ty().kind() else {
        return Err("copy_g2s callee is not a FnDef".to_string());
    };
    decode_cooperative_config(&args)
}

/// Checked static facts for one warp-cooperative `ldmatrix` fragment load.
#[derive(Debug, Clone)]
pub(crate) struct MatrixLoadConfig {
    /// Shared tile map, in elements; backend lowering recasts it to bytes.
    pub smem_layout: ComposedLayout,
}

/// Decode `load_matrix_{a,b}::<SmemL>`'s one generic and prove the shared
/// tile can feed `ldmatrix` (two modes, 8-multiple extents, 16-byte-aligned
/// contiguous row segments through any swizzle).
pub(crate) fn decode_load_matrix_callee(func: &mir::Operand) -> Result<MatrixLoadConfig, String> {
    let mir::Operand::Constant(constant) = func else {
        return Err("load_matrix must be a direct constant function call".to_string());
    };
    let TyKind::RigidTy(RigidTy::FnDef(_, args)) = constant.const_.ty().kind() else {
        return Err("load_matrix callee is not a FnDef".to_string());
    };
    expect_schema(&args, &["Type"], "cute_rs::mma::load_matrix")?;
    let smem_layout = decode_smem_layout(&type_arg(&args, 0, "load_matrix")?)?;

    let modes = smem_layout.inner().modes();
    if modes.len() != 2 {
        return Err("load_matrix shared layout needs row and column modes".to_string());
    }
    let (Some(rows), Some(columns)) = (modes[0].checked_size(), modes[1].checked_size()) else {
        return Err("load_matrix shared layout extents are invalid".to_string());
    };
    let byte_layout = smem_layout
        .to_byte_offsets(2)
        .map_err(|error| format!("load_matrix shared layout has no byte form: {error}"))?;
    validate_ldmatrix_source(&byte_layout, rows, columns)
        .map_err(|error| format!("load_matrix source is not ldmatrix-loadable: {error}"))?;

    Ok(MatrixLoadConfig { smem_layout })
}

/// Checked static facts for one hardware (TMA) tile copy.
#[derive(Debug, Clone)]
pub(crate) struct TmaCopyConfig {
    pub smem_layout: ComposedLayout,
    pub element_type: Ty,
    /// Minimum shared-base alignment that preserves this layout's swizzle
    /// phase. The recognized unsafe boundary is the exact place where the
    /// source program promises it.
    pub smem_alignment_bytes: u64,
}

fn tma_copy_alignment_bytes(
    smem_layout: &ComposedLayout,
    element_bytes: i64,
) -> Result<u64, String> {
    let alignment = cute_layout::tma_phase_alignment_bytes(smem_layout, element_bytes)
        .map_err(|error| format!("TMA layout has no valid shared-base alignment: {error}"))?;
    u64::try_from(alignment).map_err(|_| "TMA shared-base alignment does not fit u64".to_string())
}

/// Decode `copy_tma_2d::<T, SmemL>` and prove `SmemL` is TMA-encodable at
/// kernel compile time (the host encoder re-checks at descriptor creation).
pub(crate) fn decode_copy_tma_callee(func: &mir::Operand) -> Result<TmaCopyConfig, String> {
    let mir::Operand::Constant(constant) = func else {
        return Err("copy_tma_2d must be a direct constant function call".to_string());
    };
    let TyKind::RigidTy(RigidTy::FnDef(_, args)) = constant.const_.ty().kind() else {
        return Err("copy_tma_2d callee is not a FnDef".to_string());
    };
    expect_schema(&args, &["Type", "Type"], "cute_rs::tma::copy_tma_2d")?;
    let element_type = type_arg(&args, 0, "copy_tma_2d")?;
    let (element_bytes, _) = cooperative_element(&element_type)?;
    let smem_layout = decode_smem_layout(&type_arg(&args, 1, "copy_tma_2d")?)?;
    let smem_alignment_bytes = tma_copy_alignment_bytes(&smem_layout, element_bytes)
        .map_err(|error| format!("copy_tma_2d layout is not TMA-encodable: {error}"))?;
    Ok(TmaCopyConfig {
        smem_layout,
        element_type,
        smem_alignment_bytes,
    })
}

/// Decode `copy_tma_s2g_2d::<T, SmemL>` and prove the shared source layout
/// is TMA-encodable before constructing the store operation.
pub(crate) fn decode_copy_tma_s2g_callee(func: &mir::Operand) -> Result<TmaCopyConfig, String> {
    let mir::Operand::Constant(constant) = func else {
        return Err("copy_tma_s2g_2d must be a direct constant function call".to_string());
    };
    let TyKind::RigidTy(RigidTy::FnDef(_, args)) = constant.const_.ty().kind() else {
        return Err("copy_tma_s2g_2d callee is not a FnDef".to_string());
    };
    expect_schema(&args, &["Type", "Type"], "cute_rs::tma::copy_tma_s2g_2d")?;
    let element_type = type_arg(&args, 0, "copy_tma_s2g_2d")?;
    let (element_bytes, _) = cooperative_element(&element_type)?;
    let smem_layout = decode_smem_layout(&type_arg(&args, 1, "copy_tma_s2g_2d")?)?;
    let smem_alignment_bytes = tma_copy_alignment_bytes(&smem_layout, element_bytes)
        .map_err(|error| format!("copy_tma_s2g_2d layout is not TMA-encodable: {error}"))?;
    Ok(TmaCopyConfig {
        smem_layout,
        element_type,
        smem_alignment_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tma_copy_alignment_keeps_the_selected_swizzle_phase() {
        let byte_layout = ComposedLayout::new(
            Swizzle::new(2, 4, 3),
            0,
            "(128,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap();
        let scale_layout = ComposedLayout::new(
            Swizzle::IDENTITY,
            0,
            "(1,256):(256,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap();

        assert_eq!(tma_copy_alignment_bytes(&byte_layout, 1), Ok(512));
        assert_eq!(tma_copy_alignment_bytes(&scale_layout, 2), Ok(16));
    }
}
