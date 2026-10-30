/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural decoder for nested `cute-rs` type markers.
//!
//! The decoder follows ADT identities and generic-argument kinds:
//!
//! ```text
//! Layout<(C<32>, C<8>), (C<8>, C<1>)>
//! │      └─── shape ──┘  └─ stride ──┘
//! └─ exact DefId: cute_rs::markers::Layout
//! ```
//!
//! Shape and stride modes are Rust tuples of `C<N>` leaves. A tuple type has
//! no defining path of its own, so the identity check holds at the leaves.
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

/// Exact defining path of `cute_rs::markers::C`; public reexports do not change it.
const C: &str = "cute_rs::markers::C";
/// Exact defining path of `cute_rs::markers::Layout`; public reexports do not change it.
const LAYOUT: &str = "cute_rs::markers::Layout";
/// Exact defining path of `cute_rs::markers::RowMajor`; public reexports do not change it.
const ROW_MAJOR: &str = "cute_rs::markers::RowMajor";
/// Exact defining path of `cute_rs::markers::ColMajor`; public reexports do not change it.
const COL_MAJOR: &str = "cute_rs::markers::ColMajor";
/// Exact defining path of `cute_rs::markers::CopyG2S`; public reexports do not change it.
const COPY_G2S: &str = "cute_rs::markers::CopyG2S";
/// Exact defining path of `cute_rs::markers::LeadingDim`; public reexports do not change it.
const LEADING_DIM: &str = "cute_rs::markers::LeadingDim";
/// Exact defining path of `cute_rs::markers::Swizzle`; public reexports do not change it.
const SWIZZLE: &str = "cute_rs::markers::Swizzle";
/// Exact defining path of `cute_rs::markers::ComposedLayout`; public reexports do not change it.
const COMPOSED: &str = "cute_rs::markers::ComposedLayout";
/// Exact defining path of `cute_rs::numeric::BFloat16`; public reexports do not change it.
const BFLOAT16: &str = "cute_rs::numeric::BFloat16";

/// Validated cooperative-copy instruction width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopyAtom {
    /// Bytes transferred by one copy atom.
    pub bytes: u32,
}

/// Decoded copy types and layouts together with the checked cooperative transfer plan.
#[derive(Debug, Clone)]
pub(crate) struct CooperativeConfig {
    /// Validated asynchronous-copy instruction selection.
    pub atom: CopyAtom,
    /// Map from participating thread coordinates to linear thread IDs.
    pub thread_layout: Layout,
    /// Map describing the per-thread values in one cooperative transfer.
    pub value_layout: Layout,
    /// Logical global tile shape and coordinate mapping.
    pub tile_layout: Layout,
    /// Physical destination layout, including its shared-memory swizzle.
    pub smem_layout: ComposedLayout,
    /// Divisor of the runtime row pitch, measured in elements.
    pub leading_dim_divisor: u64,
    /// The exact checked maps M3 must lower; never recompute them later.
    #[allow(dead_code)]
    pub plan: CooperativeCopyPlan,
    // M3 emission needs the decoded Rust type; the pre-M3 sentinel only prints it.
    /// Decoded Rust scalar type retained for IR carrier translation.
    #[allow(dead_code)]
    pub element_type: Ty,
    // Keep the validated byte width beside the type so emission cannot re-guess it.
    /// Validated number of storage bytes per scalar element.
    #[allow(dead_code)]
    pub element_bytes: i64,
}

/// Return the supported scalar element's byte width and diagnostic name.
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
        TyKind::RigidTy(RigidTy::Adt(def, args)) if canonical_adt_path(&def) == BFLOAT16 => {
            expect_schema(&args, &[], BFLOAT16)?;
            (2, "BFloat16")
        }
        _ => {
            return Err(format!(
                "copy_g2s element must be a supported scalar, found {ty:?}"
            ));
        }
    };
    Ok(info)
}

/// Recover an ADT's actual defining path, independent of public aliases.
fn canonical_adt_path(def: &AdtDef) -> String {
    super::identity::canonical_def_path(def.def_id(), &def.krate().name.to_string())
}

/// Require an exact ADT owner and return its generic arguments.
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

/// Check the ordered Type/Const/Lifetime kinds before decoding generic arguments.
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

/// Evaluate a source constant and reject values that do not fit in `u64`.
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

/// Evaluate a signed source constant and reject values outside `i64`.
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

/// Read a type argument at the required generic position, rejecting other argument kinds.
fn type_arg(args: &GenericArgs, index: usize, owner: &str) -> Result<Ty, String> {
    match args.0.get(index) {
        Some(GenericArgKind::Type(value)) => Ok(*value),
        _ => Err(format!("`{owner}` generic {index} must be a type")),
    }
}

/// Read a constant argument at the required generic position, rejecting other argument kinds.
fn const_arg(args: &GenericArgs, index: usize, owner: &str) -> Result<TyConst, String> {
    match args.0.get(index) {
        Some(GenericArgKind::Const(value)) => Ok(value.clone()),
        _ => Err(format!("`{owner}` generic {index} must be a const")),
    }
}

/// Recursively decode integer leaves and tuple modes into a layout tuple.
///
/// A mode is the leaf marker `C<N>` or a Rust tuple `(A, B, ...)` of further
/// modes, so `(C<2>, (C<3>, C<4>))` decodes to `(2,(3,4))`. A tuple type has
/// no defining path, so the near-miss guard applies at the leaves: every leaf
/// must be the exact `cute_rs::markers::C` ADT.
fn decode_tuple(ty: &Ty) -> Result<IntTuple, String> {
    if let TyKind::RigidTy(RigidTy::Tuple(items)) = ty.kind() {
        if items.is_empty() {
            return Err("layout tuple must have at least one mode".to_string());
        }
        return items
            .iter()
            .map(decode_tuple)
            .collect::<Result<Vec<_>, _>>()
            .map(IntTuple::Tuple);
    }
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Err(format!(
            "layout tuple mode must be a tuple or `{C}`, found {ty:?}"
        ));
    };
    match canonical_adt_path(&def).as_str() {
        C => {
            expect_schema(&args, &["Const"], C)?;
            Ok(IntTuple::Leaf(signed_const(
                &const_arg(&args, 0, C)?,
                "C value",
            )?))
        }
        found => Err(format!("expected a tuple or `{C}`, found `{found}`")),
    }
}

/// Compute the shape product with checked arithmetic and positive extents.
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

/// Decode a static layout marker, including nested shape and stride tuples.
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
            if !shape.is_congruent(&stride) {
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

/// Decode and validate a `CopyG2S` copy atom's byte width.
fn decode_atom(ty: &Ty) -> Result<CopyAtom, String> {
    let args = adt_owned(ty, COPY_G2S)?;
    expect_schema(&args, &["Const"], COPY_G2S)?;
    let bytes = u32::try_from(unsigned_const(
        &const_arg(&args, 0, COPY_G2S)?,
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

/// Decode the guaranteed runtime row-pitch divisor from its static marker.
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

/// Decode XOR-swizzle bits, base and shift from constant marker arguments.
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

/// Decode the sealed attention layout identity and its major mode.
pub(crate) fn decode_attention_layout(ty: &Ty) -> Result<(ComposedLayout, bool), String> {
    /// Exact defining path of `cute_rs::attention::sm100::shared::SharedLayout`; public reexports do not change it.
    const PATH: &str = "cute_rs::attention::sm100::shared::SharedLayout";
    let args = adt_owned(ty, PATH)?;
    expect_schema(&args, &["Const", "Const", "Const"], PATH)?;
    let rows = u32::try_from(unsigned_const(
        &const_arg(&args, 0, PATH)?,
        "attention rows",
    )?)
    .map_err(|_| "attention rows exceed u32")?;
    let cols = u32::try_from(unsigned_const(
        &const_arg(&args, 1, PATH)?,
        "attention columns",
    )?)
    .map_err(|_| "attention columns exceed u32")?;
    let major = unsigned_const(&const_arg(&args, 2, PATH)?, "attention major mode")?;
    if major > 1 {
        return Err("attention major mode must be boolean".into());
    }
    let mn = major == 1;
    Ok((
        dialect_cute::sm100_attention_ops::shared_layout(rows, cols, mn)?.0,
        mn,
    ))
}

/// Decode the supported plain or composed shared-memory layout markers.
pub(crate) fn decode_smem_layout(ty: &Ty) -> Result<ComposedLayout, String> {
    if let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind()
        && canonical_adt_path(&def) == COMPOSED
    {
        // `ComposedLayout<Inner, OFFSET, Outer>`: position 0 is the inner swizzle and
        // position 2 the outer layout, in the order of CuTeDSL's
        // `make_composed_layout(inner, offset, outer)`.
        expect_schema(&args, &["Type", "Const", "Type"], COMPOSED)?;
        let swizzle = decode_swizzle(&type_arg(&args, 0, COMPOSED)?)?;
        let offset = signed_const(&const_arg(&args, 1, COMPOSED)?, "composed offset")?;
        let layout = decode_layout(&type_arg(&args, 2, COMPOSED)?)?;
        // Rust marker layouts count elements. Keeping the unit explicit here
        // lets the dialect verifier recast the entire expression—including
        // the swizzle bit fields—to bytes once `T` is known.
        return ComposedLayout::new(swizzle, offset, layout, OffsetUnit::Elements)
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
        "cute_rs::copy::cooperative::copy_g2s",
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
    expect_schema(&args, &["Type", "Type"], "cute_rs::mma::load_matrix")?;
    let element_type = type_arg(&args, 1, "load_matrix")?;
    let (element_bytes, _) = cooperative_element(&element_type)?;
    if element_bytes != 2 {
        return Err("load_matrix requires a 2-byte (f16 or BFloat16) element type".to_string());
    }
    let smem_layout = decode_smem_layout(&type_arg(&args, 0, "load_matrix")?)?;

    let modes = smem_layout.outer().modes();
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
    /// Physical destination layout, including its shared-memory swizzle.
    pub smem_layout: ComposedLayout,
    /// Decoded Rust scalar type retained for IR carrier translation.
    pub element_type: Ty,
    /// Minimum shared-base alignment that preserves this layout's swizzle
    /// phase. The recognized unsafe boundary is the exact place where the
    /// source program promises it.
    pub smem_alignment_bytes: u64,
}

/// Compute the base alignment needed to preserve the selected TMA swizzle phase.
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
    expect_schema(&args, &["Type", "Type"], "cute_rs::copy::tma::copy_tma_2d")?;
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
    expect_schema(
        &args,
        &["Type", "Type"],
        "cute_rs::copy::tma::copy_tma_s2g_2d",
    )?;
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

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Compile `source` in-process as a crate named `cute_rs`, so a `markers`
    /// module inside it sits at the defining paths the decoder pins, then
    /// decode the first parameter type of every function item and return the
    /// results by short function name.
    ///
    /// `Ty` values exist only inside a compiler session, so the decoded
    /// layouts are plain data compared outside it. The rustup shim resolves
    /// the pinned toolchain this test binary was built with, so the in-process
    /// driver and the sysroot agree.
    fn decode_fixture_parameters(
        source: &str,
        tag: &str,
    ) -> BTreeMap<String, Result<Layout, String>> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_static_config_{tag}_{}_{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("fixture.rs");
        std::fs::write(&fixture, source).unwrap();

        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let sysroot_output = std::process::Command::new(rustc)
            .args(["--print", "sysroot"])
            .output()
            .expect("query rustc sysroot");
        assert!(sysroot_output.status.success(), "rustc --print sysroot");
        let sysroot = String::from_utf8(sysroot_output.stdout)
            .expect("sysroot path is UTF-8")
            .trim()
            .to_string();

        let args = vec![
            "rustc".to_string(),
            "--edition=2024".to_string(),
            "--crate-type=rlib".to_string(),
            "--crate-name=cute_rs".to_string(),
            "--emit=metadata".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        // rustc needs more stack than the default test-thread allowance.
        let decoded = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let mut decoded = BTreeMap::new();
                    for item in rustc_public::all_local_items() {
                        if item.kind() != rustc_public::ItemKind::Fn {
                            continue;
                        }
                        let Some(body) = item.body() else {
                            continue;
                        };
                        let Some(parameter) = body.locals().get(1) else {
                            continue;
                        };
                        let name = item.name();
                        let short = name.rsplit("::").next().unwrap_or(&name).to_string();
                        decoded.insert(short, decode_layout(&parameter.ty));
                    }
                    std::ops::ControlFlow::<(), _>::Continue(decoded)
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();
        decoded
    }

    /// Rust tuples carry nested shapes and strides: a tuple of `C<N>` leaves
    /// or of further tuples decodes to the nested layout it spells, an empty
    /// tuple is rejected, and a leaf that is not the exact
    /// `cute_rs::markers::C` ADT fails the identity check.
    #[test]
    fn tuple_shapes_decode_to_nested_layouts() {
        let decoded = decode_fixture_parameters(
            r#"
pub mod markers {
    use core::marker::PhantomData;
    pub struct C<const VALUE: i64>;
    pub struct Layout<Shape, Stride> { _layout: PhantomData<(Shape, Stride)> }
}
pub mod other {
    pub struct C<const VALUE: i64>;
}
use markers::{C, Layout};
pub fn leaf(_: Layout<C<2>, C<17>>) {}
pub fn pitched(_: Layout<(C<2>, C<3>), (C<17>, C<4>)>) {}
pub fn nested(_: Layout<(C<4>, ((C<2>, C<2>), C<2>)), (C<2>, ((C<1>, C<128>), C<8>))>) {}
pub fn triple(_: Layout<(C<2>, C<3>, C<4>), (C<12>, C<4>, C<1>)>) {}
pub fn empty_mode(_: Layout<(), ()>) {}
pub fn primitive_leaf(_: Layout<(u32, C<1>), (C<1>, C<1>)>) {}
pub fn foreign_leaf(_: Layout<(other::C<2>, C<3>), (C<3>, C<1>)>) {}
"#,
            "tuple_shapes",
        );
        let expect = |name: &str, layout: &str| {
            assert_eq!(
                decoded[name],
                Ok(layout.parse::<Layout>().unwrap()),
                "{name}"
            );
        };
        expect("leaf", "2:17");
        expect("pitched", "(2,3):(17,4)");
        expect("nested", "(4,((2,2),2)):(2,((1,128),8))");
        expect("triple", "(2,3,4):(12,4,1)");
        let error = |name: &str| decoded[name].as_ref().expect_err(name).clone();
        assert!(error("empty_mode").contains("at least one mode"));
        assert!(error("primitive_leaf").contains("must be a tuple or `cute_rs::markers::C`"));
        assert_eq!(
            error("foreign_leaf"),
            "expected a tuple or `cute_rs::markers::C`, found `cute_rs::other::C`"
        );
    }

    /// Check that TMA copy alignment preserves the selected swizzle phase.
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
