/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stable 128-bit type identifiers for kernel PTX naming.
//!
//! The host needs to compute the same per-type hash that the backend computes
//! via `tcx.type_id_hash(ty).as_u128()`. The stable [`core::any::TypeId::of`]
//! API would force a `T: 'static` bound on the kernel marker, which would in
//! turn reject perfectly valid non-`'static` borrowing closures (e.g. a kernel
//! launcher capturing `&[f32]` from a stack frame the caller keeps alive
//! across the launch). The `core::intrinsics::type_id` form has bound
//! `T: ?Sized` — i.e. no `'static` requirement — and produces the exact same
//! 128-bit value that `tcx.type_id_hash` does for that type, because both go
//! through the same `erase_and_anonymize_regions` + stable-hash pipeline.
//!
//! The macro layer calls [`type_id_u128_of_val`] with a concrete kernel
//! function item. A function item's type contains its definition identity and
//! every monomorphized type and const argument, so one hash covers the complete
//! specialization without inventing a parallel encoding for const values.
//!
//! Framing note for future contributors: `core::intrinsics::type_id` is an
//! internal API and requires `#![feature(core_intrinsics)]` on the owning
//! crate. cuda-oxide already ships against `rustc_private` and pins a
//! nightly toolchain, so this is inside our existing risk surface — but the
//! helper cannot be lifted into a stable-feeling utility crate without
//! re-introducing the feature gate there.

use core::any::TypeId;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

type KernelNameKey = (&'static str, u128);

static GENERIC_KERNEL_NAMES: OnceLock<Mutex<HashMap<KernelNameKey, &'static str>>> =
    OnceLock::new();

/// Returns the same 128-bit hash that the cuda-oxide backend uses for
/// kernel export names.
///
/// At runtime the value is just the 16 raw hash bytes (see the layout
/// comment in `core::any::TypeId`). The intrinsic is const-evaluated by
/// rustc using its internal `Ty<'tcx>` representation, so the call site
/// only ever sees a constant `u128`.
///
/// Bound is intentionally `T: ?Sized` (not `T: 'static`). The typed launch
/// path must keep accepting non-`'static` borrowing closures, the same way
/// the legacy `type_name`-based path did. Adding `'static` here would
/// silently tighten the typed API without enforcing the actual launch-
/// outlives-borrow invariant — that responsibility still sits with the
/// caller (the borrow must outlive `stream.synchronize()`).
#[inline]
pub fn type_id_u128<T: ?Sized>() -> u128 {
    let id = const { core::intrinsics::type_id::<T>() };
    unsafe { core::mem::transmute::<TypeId, u128>(id) }
}

/// Returns [`type_id_u128`] for the inferred type of `value`.
///
/// This is primarily useful for function items, whose anonymous type cannot be
/// written in a turbofish. For example, `type_id_u128_of_val(&kernel::<T, 4>)`
/// hashes the concrete `kernel::<T, 4>` function-item type rather than its
/// coerced function-pointer signature. Keeping the item uncoerced is important:
/// the item type retains both the function definition and its generic arguments.
#[inline]
pub fn type_id_u128_of_val<T: ?Sized>(_: &T) -> u128 {
    type_id_u128::<T>()
}

/// Interns the PTX lookup name for one generic kernel specialization.
///
/// This is public only because `#[kernel]` expansions in downstream crates
/// call it. The process-wide table allocates once per `(base_name, type_hash)`
/// pair; repeated launches reuse the same `&'static str` instead of leaking a
/// fresh formatted name on every cache lookup.
#[doc(hidden)]
pub fn __intern_generic_kernel_name(base_name: &'static str, type_hash: u128) -> &'static str {
    let names = GENERIC_KERNEL_NAMES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut names = names
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match names.entry((base_name, type_hash)) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.get(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let name: &'static str =
                Box::leak(format!("{base_name}_TID_{type_hash:032x}").into_boxed_str());
            entry.insert(name);
            name
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_types_hash_distinctly() {
        assert_ne!(type_id_u128::<f32>(), type_id_u128::<i32>());
        assert_ne!(type_id_u128::<u32>(), type_id_u128::<i32>());
    }

    #[test]
    fn same_type_hashes_stably() {
        let a = type_id_u128::<f32>();
        let b = type_id_u128::<f32>();
        assert_eq!(a, b);
    }

    #[test]
    fn static_borrow_collides_with_free_borrow() {
        // Confirms erase_and_anonymize_regions: free lifetimes (including
        // 'static) all hash to the same value. The `'a` is intentionally a
        // free lifetime here, used only in the body's turbofish.
        #[allow(clippy::extra_unused_lifetimes)]
        fn free<'a>() -> u128 {
            type_id_u128::<&'a i32>()
        }
        assert_eq!(type_id_u128::<&'static i32>(), free());
    }

    #[test]
    fn distinct_closure_literals_hash_distinctly() {
        let factor = 2.5f32;
        let cl1 = move |x: f32| x * factor;
        let cl2 = move |x: f32| x * factor;
        fn id<T>(_: &T) -> u128 {
            type_id_u128::<T>()
        }
        assert_ne!(id(&cl1), id(&cl2));
    }

    #[test]
    fn function_item_hash_includes_const_arguments() {
        fn const_item<const N: usize>() {}

        assert_ne!(
            type_id_u128_of_val(&const_item::<0>),
            type_id_u128_of_val(&const_item::<1>)
        );
        assert_ne!(
            type_id_u128_of_val(&const_item::<4>),
            type_id_u128_of_val(&const_item::<8>)
        );
        assert_eq!(
            type_id_u128_of_val(&const_item::<4>),
            type_id_u128_of_val(&const_item::<4>)
        );
    }

    #[test]
    fn function_item_hash_includes_mixed_type_and_const_arguments() {
        fn mixed_item<T, const N: usize>() {}

        assert_ne!(
            type_id_u128_of_val(&mixed_item::<u32, 4>),
            type_id_u128_of_val(&mixed_item::<u32, 8>)
        );
        assert_ne!(
            type_id_u128_of_val(&mixed_item::<u32, 4>),
            type_id_u128_of_val(&mixed_item::<i32, 4>)
        );
    }

    #[test]
    fn function_item_hash_handles_all_stable_const_parameter_kinds() {
        fn primitive_item<const FLAG: bool, const TAG: char, const BYTE: u8>() {}

        let base = type_id_u128_of_val(&primitive_item::<false, 'a', 0>);
        assert_ne!(base, type_id_u128_of_val(&primitive_item::<true, 'a', 0>));
        assert_ne!(base, type_id_u128_of_val(&primitive_item::<false, 'b', 0>));
        assert_ne!(base, type_id_u128_of_val(&primitive_item::<false, 'a', 1>));
    }

    #[test]
    fn generic_kernel_names_are_allocated_once_per_specialization() {
        let first = __intern_generic_kernel_name("tile", 4);
        let repeated = __intern_generic_kernel_name("tile", 4);
        let other_const = __intern_generic_kernel_name("tile", 8);
        let other_kernel = __intern_generic_kernel_name("reduce", 4);

        assert!(core::ptr::eq(first, repeated));
        assert_eq!(first, "tile_TID_00000000000000000000000000000004");
        assert_ne!(first, other_const);
        assert_ne!(first, other_kernel);
    }
}
