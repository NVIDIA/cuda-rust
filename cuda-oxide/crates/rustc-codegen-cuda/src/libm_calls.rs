/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Keeps calls to `libm`'s non-generic free functions intact, so device code
//! can reroute them.
//!
//! The importer lowers a `libm::*` call by recognizing the call site and
//! emitting the matching libdevice function (`libm::sqrtf` becomes
//! `__nv_sqrtf`); the collector never translates a libm body. That only works
//! while the call survives rustc's MIR inliner, and some libm entry points are
//! small enough not to. libm routes its hardware-backed functions through a
//! per-host `arch` module, so on the host the kernel's MIR was built for the
//! inliner sees a leaf like:
//!
//! ```text
//! kernel ──calls──▶ libm::sqrtf ──calls──▶ libm::math::arch::x86::sqrtf
//!                                              asm!("sqrtss {x}, {x}")
//! ```
//!
//! Both bodies are tiny, so rustc marks them cross-crate inlinable and folds
//! them into the kernel, leaving an x86 `asm!` terminator where the call used
//! to be. The same happens on aarch64 hosts to `sqrt`, `sqrtf`, `rint`,
//! `rintf`, `fma` and `fmaf`.
//!
//! Overriding the `cross_crate_inlinable` query for the `libm` crate's free
//! functions fixes this at the source. The override runs while `libm` itself is
//! compiled (cargo-oxide builds every target crate with this backend), so
//! libm's exports and the metadata downstream crates read agree: its functions
//! are exported symbols that other crates call instead of inlining. rustc's
//! inliner does not ask this query about generic callees, so it covers libm's
//! non-generic entry points, which are the ones the importer intercepts.
//!
//! The cost is on the host, where a libm call is no longer inlined either: a
//! host `libm::sqrtf` becomes a call to libm's `sqrtss` wrapper rather than the
//! instruction itself, which costs a call per element in a hot host loop. Thin
//! or fat LTO inlines it again.
//!
//! Associated functions keep rustc's answer. The `Libm::<f32>` helpers are
//! `#[inline(always)]` wrappers whose names drop the `f` suffix
//! (`Libm::<f32>::sin` forwards to `sinf`), and the importer picks the libdevice
//! function from the last path segment, so a surviving `Libm::<f32>::sin` call
//! would be routed to the f64 `__nv_sin`. Inlined, the wrapper leaves a call to
//! the free function, which the override keeps.

use rustc_hir::def::DefKind;
use rustc_hir::def_id::{LOCAL_CRATE, LocalDefId};
use rustc_middle::ty::TyCtxt;
use std::sync::OnceLock;

/// The crate whose calls the importer intercepts. The collector skips the same
/// crate by name.
const LIBM_CRATE: &str = "libm";

type CrossCrateInlinable = for<'tcx> fn(TyCtxt<'tcx>, LocalDefId) -> bool;

/// rustc's own provider, which the override defers to for everything but
/// `libm`'s free functions.
static DEFAULT_CROSS_CRATE_INLINABLE: OnceLock<CrossCrateInlinable> = OnceLock::new();

/// Installs the override on top of whatever provider is already set.
pub(crate) fn provide(providers: &mut rustc_middle::util::Providers) {
    let default = providers.queries.cross_crate_inlinable;
    // The backend is loaded once per rustc process, so the first provider seen
    // is the only one
    let _ = DEFAULT_CROSS_CRATE_INLINABLE.set(default);
    providers.queries.cross_crate_inlinable = cross_crate_inlinable;
}

fn cross_crate_inlinable(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // only free functions: rustc requires `true` for closures and
    // constructors, and the `Libm<T>` methods must inline down to them
    if tcx.crate_name(LOCAL_CRATE).as_str() == LIBM_CRATE && tcx.def_kind(def_id) == DefKind::Fn {
        return false;
    }

    let default = DEFAULT_CROSS_CRATE_INLINABLE
        .get()
        .expect("`provide` installs the default provider before any query runs");
    default(tcx, def_id)
}
