/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler utility functions: tile-ir attribute builders.

use cutile_ir::ir::Attribute;

// Re-export from hints.rs (canonical, melior-free location).
pub use crate::hints::{CompileOptions, OptimizationHints, SMHints};

/// Named attribute pair (name, value).
pub type NamedAttr = (String, Attribute);

/// Creates a string attribute.
pub fn str_attr(name: &str, value: &str) -> NamedAttr {
    (name.to_string(), Attribute::String(value.to_string()))
}

/// Creates a type attribute.
pub fn type_attr(name: &str, ty: cutile_ir::ir::Type) -> NamedAttr {
    (name.to_string(), Attribute::Type(ty))
}

/// Creates a bool/flag attribute (unit attribute = true).
pub fn flag_attr(name: &str) -> NamedAttr {
    (name.to_string(), Attribute::Bool(true))
}

/// Creates an integer attribute (i32-typed, suitable for enum values and small constants).
pub fn int_attr(name: &str, value: i64) -> NamedAttr {
    (name.to_string(), Attribute::i32(value))
}

/// Creates a float attribute (f64-typed).
pub fn float_attr(name: &str, value: f64) -> NamedAttr {
    (
        name.to_string(),
        Attribute::float(value, cutile_ir::ir::ScalarType::F64),
    )
}

// ---------------------------------------------------------------------------
// Enum attribute helpers (replace parse_named_attr for enum types)
// ---------------------------------------------------------------------------

/// Comparison predicate attribute.
pub fn cmp_pred_attr(pred: &str) -> NamedAttr {
    let val = match pred {
        "equal" => 0,
        "not_equal" => 1,
        "less_than" => 2,
        "less_than_or_equal" => 3,
        "greater_than" => 4,
        "greater_than_or_equal" => 5,
        _ => panic!("unknown comparison predicate: {pred}"),
    };
    int_attr("comparison_predicate", val)
}

/// Comparison ordering attribute.
pub fn cmp_ordering_attr(ordering: &str) -> NamedAttr {
    let val = match ordering {
        "unordered" => 0,
        "ordered" => 1,
        _ => panic!("unknown comparison ordering: {ordering}"),
    };
    int_attr("comparison_ordering", val)
}

/// The Tile IR signedness (`"signed"` / `"unsigned"`) of a Rust integer
/// element type, by its source name (`"u8"`, `"i32"`, `"bool"`, ...).
///
/// This is the ONE mapping every signedness-carrying op (`cmpi`, `divi`,
/// `remi`, `maxi`/`mini`, `shri`, `exti`, `itof`/`ftoi`, `mmai`) must use.
/// Tile IR widens `u8`/`u16` to `i8`/`i16` storage, so the Rust name is the
/// only place the unsignedness survives; a site that spells its own subset
/// of the unsigned names silently compiles `u8` values `>= 128` as negative
/// (audit 2026-08: `compile_binary_op` and `compile_cuda_tile_op` listed
/// only `u32`/`u64`). `usize` is deliberately absent: the DSL relabels it
/// against `i32` (see `value_facts::int_value_domain`), so it keeps the
/// signed lowering it always had.
pub fn rust_int_signedness(element_type_str: &str) -> &'static str {
    match element_type_str {
        "bool" | "u8" | "u16" | "u32" | "u64" => "unsigned",
        _ => "signed",
    }
}

/// Signedness attribute.
pub fn signedness_attr(name: &str, signedness: &str) -> NamedAttr {
    let val = match signedness {
        "unsigned" => 0,
        "signed" => 1,
        _ => panic!("unknown signedness: {signedness}"),
    };
    int_attr(name, val)
}

/// Rounding mode attribute.
pub fn rounding_mode_attr(mode: &str) -> NamedAttr {
    let val = match mode {
        "nearest_even" => 0,
        "zero" => 1,
        "negative_inf" => 2,
        "positive_inf" => 3,
        "approx" => 4,
        "full" => 5,
        "nearest_int_to_zero" => 6,
        "nearest_away" => 7,
        _ => panic!("unknown rounding mode: {mode}"),
    };
    int_attr("rounding_mode", val)
}

/// Integer overflow attribute.
pub fn overflow_attr(overflow: &str) -> NamedAttr {
    let val = match overflow {
        "none" => 0,
        "no_signed_wrap" | "nsw" => 1,
        "no_unsigned_wrap" | "nuw" => 2,
        "no_wrap" | "nw" => 3,
        _ => panic!("unknown overflow: {overflow}"),
    };
    int_attr("overflow", val)
}

/// Memory ordering semantics attribute.
pub fn memory_ordering_attr(ordering: &str) -> NamedAttr {
    let val = match ordering {
        "weak" => 0,
        "relaxed" => 1,
        "acquire" => 2,
        "release" => 3,
        "acq_rel" => 4,
        _ => panic!("unknown memory ordering: {ordering}"),
    };
    int_attr("memory_ordering_semantics", val)
}

/// Memory scope attribute.
pub fn memory_scope_attr(scope: &str) -> NamedAttr {
    let val = match scope {
        "tl_blk" => 0,
        "device" => 1,
        "sys" => 2,
        _ => panic!("unknown memory scope: {scope}"),
    };
    int_attr("memory_scope", val)
}
