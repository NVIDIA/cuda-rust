/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! DSL syntax validation for GPU kernel entry points.
//!
//! This module validates that kernel functions follow the restrictions and requirements
//! of the cuTile Rust DSL. It ensures type safety and prevents unsupported patterns that
//! would fail during MLIR compilation or GPU execution.
//!
//! ## Validation Rules
//!
//! ### Parameter Types
//!
//! Kernel entry points may only use the following parameter types:
//!
//! - **Scalars** - Primitive types like `i32`, `f32`, etc.
//! - **`&Tensor<T, S>`** - Immutable tensor references (read-only access)
//! - **`&mut Tensor<T, S>`** - Mutable tensor references (partitioned tensors)
//! - **`*mut T`** - Raw pointers (for unsafe kernels only)
//!
//! ### Disallowed Patterns
//!
//! - **Owned tensors** - Cannot move tensors into kernels
//! - **Arbitrary references** - Only `&Tensor` and `&mut Tensor` are supported
//! - **Complex types** - No user-defined structs (except DSL types)
//! - **Closures** - No closure parameters
//!
//! ## Examples
//!
//! ### Valid Entry Points
//!
//! ```rust,ignore
//! #[cutile::entry]
//! fn valid_kernel<const N: i32>(
//!     output: &mut Tensor<f32, {[N]}>,  // ✓ Mutable tensor (partitioned)
//!     input: &Tensor<f32, {[-1]}>,      // ✓ Immutable tensor
//!     scalar: f32,                       // ✓ Scalar parameter
//! ) { }
//! ```
//!
//! ### Invalid Entry Points
//!
//! ```rust,ignore
//! #[cutile::entry]
//! fn invalid_kernel(
//!     owned: Tensor<f32, {[128]}>,      // ✗ Cannot move tensors
//!     vec_ref: &Vec<f32>,                // ✗ Only &Tensor references allowed
//! ) { }
//! ```
//!
//! ## Error Messages
//!
//! The validator provides helpful error messages when validation fails:
//!
//! - Explains why a parameter type is not supported
//! - Suggests correct usage (e.g., using `&mut Tensor` for partitioned tensors)
//! - Points to the specific parameter that caused the error

use cutile_frontend::syn_utils::{get_attribute, get_ident_from_path, get_sig_types, get_type_ident};
use cutile_frontend::types::get_ptr_type;
use quote::ToTokens;
use syn::{ItemFn, Type};

use crate::error::{Error, SpannedError};

/// Validates that kernel entry point parameters follow DSL restrictions.
///
/// This function checks each parameter in a kernel function signature to ensure it uses
/// only supported types. It enforces the safety guarantees of the cuTile Rust DSL.
///
/// ## Supported Parameter Types
///
/// - **Scalars**: `i32`, `f32`, `bool`, etc.
/// - **Tensor references**: `&Tensor<T, S>` (immutable) or `&mut Tensor<T, S>` (mutable/partitioned)
/// - **Raw pointers**: `*mut T` (unsafe kernels only)
///
/// ## Validation Logic
///
/// For each parameter:
/// 1. **References** - Must be `&Tensor` or `&mut Tensor`
/// 2. **Path types** - Disallows owned `Tensor` (suggests using references)
/// 3. **Pointers** - Validates pointer type is supported
/// 4. **Other types** - Assumed to be scalars (validated elsewhere)
///
/// ## Errors
///
/// Returns an `Error` if an unsupported parameter type is encountered.
/// The error message includes the problematic type and suggestions for fixing it.
///
/// ## Examples
///
/// ```rust,ignore
/// // This would pass validation
/// fn valid_kernel(x: &mut Tensor<f32, {[128]}>, y: &Tensor<f32, {[-1]}>) { }
///
/// // This would panic with helpful error message
/// fn invalid_kernel(x: Tensor<f32, {[128]}>) { }
/// // Error: "Tensors cannot be moved into kernel functions. Use &mut Tensor for
/// //         partitioned tensors or &Tensor for tensor references."
/// ```
// Ensure only valid parameters have been specified in function signatures.
// Currently only supporting scalars, &Tensor, &mut Tensor,
// MappedPartitionMut, and *mut T for unsafe kernels.
// * mut T for unsafe kernels.
pub fn validate_entry_point_parameters(item: &ItemFn) -> Result<(), Error> {
    let (input_types, _output_type) = get_sig_types(&item.sig, None);
    for ty in input_types.iter() {
        match ty {
            Type::Reference(_) => {
                let Some(ident) = get_type_ident(ty) else {
                    return ty.err("Not a supported parameter type.");
                };
                let type_name = ident.to_string();
                if type_name == "MappedPartitionMut" {
                    ty.err("MappedPartitionMut parameters are passed by value; use `mut z: MappedPartitionMut<...>`, not `&mut MappedPartitionMut<...>`.")?;
                }
                if type_name != "Tensor" {
                    ty.err(&format!(
                        "References to `{}` as parameters are not supported. \
                         If this is a type alias for `Tensor`, define the alias in the same \
                         `#[cutile::module]` as the entry function; imported Tensor aliases are \
                         not supported by launcher generation.",
                        type_name
                    ))?;
                }
            }
            Type::Path(path_ty) => {
                let ident = get_ident_from_path(&path_ty.path);
                let type_name = ident.to_string();
                if type_name == "Tensor" {
                    ty.err("Tensors cannot be moved into kernel functions. \
                                  &mut Tensor corresponds to a partitioned tensor argument (e.g. x.partition([...])), \
                                  and &Tensor corresponds to a tensor reference argument (e.g. Arc::new(x) or x.into()).")?;
                }
                if type_name == "MappedPartitionMut" {
                    continue;
                }
            }
            Type::Ptr(ptr_type) => {
                let ptr_str = ptr_type.to_token_stream().to_string();
                let Some(_) = get_ptr_type(&ptr_str) else {
                    return ty.err(&format!("{} is not a supported pointer type.", ptr_str));
                };
            }
            _ => {
                ty.err(&format!(
                    "{} is not a supported parameter type.",
                    ty.to_token_stream()
                ))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imported_tensor_alias_parameter_error_mentions_same_module_aliases() {
        let item: ItemFn = syn::parse_quote! {
            fn kernel(x: &ImportedTensorAlias) {}
        };
        let err = validate_entry_point_parameters(&item).expect_err("expected alias rejection");
        let message = err.to_string();
        assert!(
            message.contains("define the alias in the same `#[cutile::module]`")
                && message.contains("imported Tensor aliases are not supported"),
            "{message}"
        );
    }
}

// TODO (hme): Implement a comprehensive validation pass on entire module.

enum EntryValueKind {
    /// A `true` / `false` literal.
    Bool,
    /// A string literal.
    Str,
    /// An expression its consumer parses itself (`preconditions`, `optimization_hints`),
    /// validated there.
    Expr,
}
/// Every key `#[entry(..)]` honors, with the value kind it expects.
///
/// The JIT looks keys up by name and reads them as literals at first launch
/// (`SingleMetaList::parse_bool` / `parse_string`), so a misspelled key is silently ignored
/// and a non-literal value panics at run time. Both are checked here, at expansion, with a
/// span on the offending token.
const ENTRY_KEYS: &[(&str, EntryValueKind)] = &[
    ("print_ir", EntryValueKind::Bool),
    ("dump_mlir_dir", EntryValueKind::Str),
    ("unchecked_accesses", EntryValueKind::Bool),
    ("deny_in_kernel_checks", EntryValueKind::Bool),
    ("preconditions", EntryValueKind::Expr),
    ("optimization_hints", EntryValueKind::Expr),
];

/// Reject a malformed `#[entry(..)]` at expansion rather than at first launch.
///
/// Matched on the attribute's last path segment, so `#[cutile::entry]`, `#[wax::entry]` and a
/// bare `#[entry]` are all recognised - an embedder names the attribute after its own crate.
pub fn validate_entry_attribute(item: &ItemFn) -> Result<(), Error> {
    let Some(attr) = get_attribute("entry", &item.attrs, true) else {
        return Ok(());
    };
    // A bare `#[entry]` carries nothing to check.
    let syn::Meta::List(list) = &attr.meta else {
        return Ok(());
    };
    let entries = list
        .parse_args_with(syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
        .map_err(|e| {
            crate::error::syn_err(
                e.span(),
                &format!("malformed `#[entry(..)]` arguments: {e}"),
            )
        })?;

    for meta in &entries {
        let (path, value) = match meta {
            syn::Meta::NameValue(name_value) => (&name_value.path, Some(&name_value.value)),
            syn::Meta::Path(path) => (path, None),
            syn::Meta::List(list) => {
                return list
                    .err("`#[entry(..)]` arguments must be `key = value` pairs or bare keys");
            }
        };
        let key = path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let Some((_, kind)) = ENTRY_KEYS.iter().find(|(known, _)| *known == key) else {
            let known = ENTRY_KEYS
                .iter()
                .map(|(k, _)| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ");
            return path.err(&format!(
                "unknown `#[entry]` key `{key}`; expected one of {known}"
            ));
        };
        let Some(value) = value else {
            // A bare key reads as `true`, which only a boolean can mean.
            if matches!(kind, EntryValueKind::Bool) {
                continue;
            }
            return path.err(&format!("`{key}` requires a value (`{key} = ...`)"));
        };
        let literal = match value {
            syn::Expr::Lit(lit) => Some(&lit.lit),
            _ => None,
        };
        match (kind, literal) {
            (EntryValueKind::Bool, Some(syn::Lit::Bool(_))) => {}
            (EntryValueKind::Bool, _) => {
                return value.err(&format!(
                    "`{key}` expects a boolean literal (`true` or `false`)"
                ));
            }
            (EntryValueKind::Str, Some(syn::Lit::Str(_))) => {}
            (EntryValueKind::Str, _) => {
                return value.err(&format!("`{key}` expects a string literal"));
            }
            (EntryValueKind::Expr, _) => {}
        }
    }
    Ok(())
}
