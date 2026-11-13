/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Read-only views over the records the assembler passes in.
//!
//! Every pointer in those records is compiler-owned and valid only for the
//! callback, and the header says a null pointer is only legal with a zero
//! length; these helpers treat null as empty and copy what the callbacks keep.

use crate::sys::{
    TileIRConstantArgument, TileIRConstantArgumentKind, TileIRDataTypeKind, TileIRDim,
    TileIRDimKind, TileIRExecutionScope, TileIRForeignType, TileIRLPCallsite, TileIRResolvedCall,
    TileIRStringView, TileIRSymbolicExpression, TileIRSymbolicExpressionKind,
};

/// The bytes of a borrowed string view, lossily decoded. Empty for null.
///
/// # Safety
/// `view.data` must point at `view.len` readable bytes, or be null.
pub(crate) unsafe fn text(view: &TileIRStringView) -> String {
    if view.data.is_null() || view.len == 0 {
        return String::new();
    }
    // SAFETY: the caller guarantees `len` readable bytes.
    let bytes = unsafe { std::slice::from_raw_parts(view.data.cast::<u8>(), view.len) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// A borrowed array as a slice. Empty for null.
///
/// # Safety
/// `ptr` must point at `len` initialized `T`, or be null.
pub(crate) unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        return &[];
    }
    // SAFETY: the caller guarantees `len` initialized elements.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// The Tile IR spelling of an element type tag.
pub(crate) fn dtype_name(kind: TileIRDataTypeKind) -> &'static str {
    match kind {
        TileIRDataTypeKind::TILEIR_DTYPE_I1 => "i1",
        TileIRDataTypeKind::TILEIR_DTYPE_I8 => "i8",
        TileIRDataTypeKind::TILEIR_DTYPE_I16 => "i16",
        TileIRDataTypeKind::TILEIR_DTYPE_I32 => "i32",
        TileIRDataTypeKind::TILEIR_DTYPE_I64 => "i64",
        TileIRDataTypeKind::TILEIR_DTYPE_F16 => "f16",
        TileIRDataTypeKind::TILEIR_DTYPE_BF16 => "bf16",
        TileIRDataTypeKind::TILEIR_DTYPE_F32 => "f32",
        TileIRDataTypeKind::TILEIR_DTYPE_TF32 => "tf32",
        TileIRDataTypeKind::TILEIR_DTYPE_F64 => "f64",
        TileIRDataTypeKind::TILEIR_DTYPE_F8E4M3FN => "f8E4M3FN",
        TileIRDataTypeKind::TILEIR_DTYPE_F8E5M2 => "f8E5M2",
        TileIRDataTypeKind::TILEIR_DTYPE_F8E8M0FNU => "f8E8M0FNU",
        TileIRDataTypeKind::TILEIR_DTYPE_F4E2M1FN => "f4E2M1FN",
        TileIRDataTypeKind::TILEIR_DTYPE_POINTER => "ptr",
        _ => "?",
    }
}

/// One dimension as the assembler described it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Dim {
    /// A concrete extent.
    Concrete(i64),
    /// A named parameter; call sites never carry these, declarations may.
    Symbolic(String),
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Concrete(value) => write!(f, "{value}"),
            Dim::Symbolic(name) => write!(f, "[{name}]"),
        }
    }
}

/// A value type at the call boundary: a tile shape and its element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TileShape {
    /// The dimensions, outermost first.
    pub shape: Vec<Dim>,
    /// The element type tag.
    pub dtype: TileIRDataTypeKind,
}

impl std::fmt::Display for TileShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tile<")?;
        for dim in &self.shape {
            write!(f, "{dim}x")?;
        }
        write!(f, "{}>", dtype_name(self.dtype))
    }
}

/// # Safety
/// `dim` must be a record the assembler filled in.
unsafe fn dim(dim: &TileIRDim) -> Dim {
    match dim.kind {
        // SAFETY: the kind selects the active union member.
        TileIRDimKind::TILEIR_DIM_CONCRETE => Dim::Concrete(unsafe { dim.__bindgen_anon_1.value }),
        // SAFETY: as above.
        _ => Dim::Symbolic(unsafe { text(&dim.__bindgen_anon_1.name) }),
    }
}

/// # Safety
/// `ty` must be a record the assembler filled in.
pub(crate) unsafe fn tile_shape(ty: &TileIRForeignType) -> TileShape {
    // SAFETY: the record's shape array is `num_shape` long.
    let dims = unsafe { slice(ty.shape, ty.num_shape) };
    TileShape {
        // SAFETY: each entry is a record the assembler filled in.
        shape: dims.iter().map(|d| unsafe { dim(d) }).collect(),
        dtype: ty.dtype.kind,
    }
}

/// # Safety
/// `types` must point at `len` records the assembler filled in.
pub(crate) unsafe fn tile_shapes(types: *const TileIRForeignType, len: usize) -> Vec<TileShape> {
    // SAFETY: as documented.
    unsafe { slice(types, len) }
        .iter()
        // SAFETY: as documented.
        .map(|ty| unsafe { tile_shape(ty) })
        .collect()
}

/// A symbolic expression in the Tile IR surface syntax, for diagnostics and
/// for matching constant arguments that are plain integers.
///
/// # Safety
/// `expr` must be a record the assembler filled in.
pub(crate) unsafe fn symbolic(expr: &TileIRSymbolicExpression) -> String {
    use TileIRSymbolicExpressionKind as K;
    // SAFETY: `kind` selects the active union member in every arm below.
    unsafe {
        let u = &expr.__bindgen_anon_1;
        let binary = |op: &str| {
            let lhs = u.binary_val.lhs;
            let rhs = u.binary_val.rhs;
            let side = |p: *const TileIRSymbolicExpression| {
                if p.is_null() {
                    "?".to_string()
                } else {
                    symbolic(&*p)
                }
            };
            format!("({} {op} {})", side(lhs), side(rhs))
        };
        match expr.kind {
            K::TILEIR_SYMBOLIC_EXPRESSION_INT => u.int_val.to_string(),
            K::TILEIR_SYMBOLIC_EXPRESSION_STRING => format!("{:?}", text(&u.string_val)),
            K::TILEIR_SYMBOLIC_EXPRESSION_DTYPE => dtype_name(u.dtype_val.kind).to_string(),
            K::TILEIR_SYMBOLIC_EXPRESSION_VARIABLE => text(&u.variable_name),
            K::TILEIR_SYMBOLIC_EXPRESSION_BUILTIN => text(&u.builtin_val.name),
            K::TILEIR_SYMBOLIC_EXPRESSION_TUPLE => {
                let values = slice(u.tuple_val.values, u.tuple_val.num_values);
                let parts: Vec<String> = values.iter().map(|v| symbolic(v)).collect();
                format!("({})", parts.join(", "))
            }
            K::TILEIR_SYMBOLIC_EXPRESSION_MUL => binary("*"),
            K::TILEIR_SYMBOLIC_EXPRESSION_DIV => binary("/"),
            K::TILEIR_SYMBOLIC_EXPRESSION_ADD => binary("+"),
            K::TILEIR_SYMBOLIC_EXPRESSION_SUB => binary("-"),
            K::TILEIR_SYMBOLIC_EXPRESSION_FLOOR_DIV => binary("floordiv"),
            K::TILEIR_SYMBOLIC_EXPRESSION_CEIL_DIV => binary("ceildiv"),
            K::TILEIR_SYMBOLIC_EXPRESSION_NEG => {
                let operand = u.operand;
                if operand.is_null() {
                    "-?".to_string()
                } else {
                    format!("-{}", symbolic(&*operand))
                }
            }
            other => format!("<expression kind {}>", other.0),
        }
    }
}

/// A concrete constant argument in the Tile IR surface syntax.
///
/// # Safety
/// `arg` must be a record the assembler filled in.
pub(crate) unsafe fn constant(arg: &TileIRConstantArgument) -> String {
    use TileIRConstantArgumentKind as K;
    // SAFETY: `kind` selects the active union member in every arm below.
    unsafe {
        let u = &arg.__bindgen_anon_1;
        match arg.kind {
            K::TILEIR_CONSTANT_ARGUMENT_INT => u.int_val.to_string(),
            K::TILEIR_CONSTANT_ARGUMENT_FLOAT => u.float_val.to_string(),
            K::TILEIR_CONSTANT_ARGUMENT_STRING => format!("{:?}", text(&u.string_val)),
            K::TILEIR_CONSTANT_ARGUMENT_DTYPE => dtype_name(u.dtype_val.kind).to_string(),
            K::TILEIR_CONSTANT_ARGUMENT_TUPLE => {
                let values = slice(u.tuple_val.values, u.tuple_val.num_values);
                let parts: Vec<String> = values.iter().map(|v| constant(v)).collect();
                format!("({})", parts.join(", "))
            }
            other => format!("<constant kind {}>", other.0),
        }
    }
}

/// What `provide_impls` is asked about: one call site of one declaration.
#[derive(Debug, Clone)]
pub(crate) struct Callsite {
    /// Discovery id, for logs.
    pub id: u64,
    /// The declaration the kernel calls.
    pub decl: String,
    /// The declaration's constant parameter names.
    pub const_params: Vec<String>,
    /// The call's constant arguments, printed.
    pub const_args: Vec<String>,
    /// The concrete argument tiles.
    pub args: Vec<TileShape>,
    /// The concrete result tiles.
    pub results: Vec<TileShape>,
}

/// # Safety
/// `site` must be a record the assembler filled in.
pub(crate) unsafe fn callsite(site: &TileIRLPCallsite) -> Option<Callsite> {
    if site.decl.is_null() {
        return None;
    }
    // SAFETY: non-null, compiler-owned for the callback.
    let decl = unsafe { &*site.decl };
    // SAFETY: the record's arrays carry their lengths.
    unsafe {
        Some(Callsite {
            id: site.id,
            decl: text(&decl.symbol),
            const_params: slice(decl.const_params, decl.num_const_params)
                .iter()
                .map(|param| text(&param.name))
                .collect(),
            const_args: slice(site.const_arguments, site.num_const_arguments)
                .iter()
                .map(|expr| symbolic(expr))
                .collect(),
            args: tile_shapes(site.arguments, site.num_arguments),
            results: tile_shapes(site.results, site.num_results),
        })
    }
}

/// What `instantiate` is asked for: one selected implementation of one call.
#[derive(Debug, Clone)]
pub(crate) struct Resolved {
    /// The implementation symbol the compiler expects the code to export.
    pub symbol: String,
    /// The declaration it implements.
    pub decl: String,
    /// The target architecture, `sm_120` and the like, or empty.
    pub arch: String,
    /// The execution scope the call was resolved to.
    pub scope: TileIRExecutionScope,
    /// Warps per CTA, or -1.
    pub warps_per_cta: i64,
    /// The concrete constant arguments, printed.
    pub const_args: Vec<String>,
    /// The concrete argument tiles.
    pub args: Vec<TileShape>,
}

/// # Safety
/// `call` must be a record the assembler filled in.
pub(crate) unsafe fn resolved(call: &TileIRResolvedCall) -> Option<Resolved> {
    if call.impl_.is_null() {
        return None;
    }
    // SAFETY: non-null, compiler-owned for the callback.
    let selected = unsafe { &*call.impl_ };
    // SAFETY: the record's arrays carry their lengths.
    unsafe {
        Some(Resolved {
            symbol: text(&selected.symbol),
            decl: if call.decl.is_null() {
                text(&selected.decl_symbol)
            } else {
                text(&(*call.decl).symbol)
            },
            arch: text(&call.resolved_execution_context.architecture),
            scope: call.resolved_execution_context.scope,
            warps_per_cta: call.resolved_execution_context.num_warps_per_cta,
            const_args: slice(call.const_arguments, call.num_const_arguments)
                .iter()
                .map(|arg| constant(arg))
                .collect(),
            args: tile_shapes(call.arguments, call.num_arguments),
        })
    }
}
