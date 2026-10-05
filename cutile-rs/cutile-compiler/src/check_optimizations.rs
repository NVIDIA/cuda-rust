/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Placement and discharge policy for dynamic bounds checks.
//!
//! Safety never varies: every access is checked under every policy. What
//! varies is which *optimizations* apply to those checks — hoisting them out
//! of loops, relocating them to launch time, or discharging them by proof.
//! Each named constructor is a documented policy; the compiler consults one
//! resolved [`CheckOptimizations`] per compile instead of scattered
//! environment reads.

use std::env;

/// Which bounds-check optimizations a compile may apply.
///
/// Resolved once per compile (see [`CheckOptimizations::from_env`]) and
/// carried on the function compiler, so a policy can never change between
/// classification and emission within one compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckOptimizations {
    /// Hoist loop-invariant residual checks to loop preheaders.
    pub hoist_to_preheaders: bool,
    /// Relocate launch-decidable checks out of the kernel to a host-side
    /// launch check.
    pub relocate_to_launch: bool,
    /// Discharge checks the compiler can prove: axis provenance, static
    /// folds, entailment from declared preconditions, and inferred bounds
    /// that eliminate one side of a guard.
    pub discharge_proofs: bool,
}

impl CheckOptimizations {
    /// The default: every optimization on.
    pub fn release() -> Self {
        Self {
            hoist_to_preheaders: true,
            relocate_to_launch: true,
            discharge_proofs: true,
        }
    }

    /// Device-debug policy: any check that lives in the kernel lives where
    /// the source puts it.
    ///
    /// No preheader hoisting — that is in-kernel code motion, and moving an
    /// assert across inlined-at scopes makes the rangeless `-O0 -G` DWARF
    /// intervals lie about exactly the instructions users stop on. Proof
    /// discharge and launch relocation stay on: both remove a check from
    /// device code entirely (nothing left to debug, no motion inside the
    /// kernel), and the bounded family's launch checks are load-bearing —
    /// without relocation its undischarged checks are compile errors.
    pub fn device_debug() -> Self {
        Self {
            hoist_to_preheaders: false,
            relocate_to_launch: true,
            discharge_proofs: true,
        }
    }

    /// Every optimization disabled: each check is emitted at its access
    /// site, two-sided, over the actual runtime values.
    ///
    /// This is the differential placement harness's semantic reference — a
    /// build too simple to share the optimized build's bugs. It inherits no
    /// proofs deliberately: a reference that inherits the very proofs under
    /// audit cannot catch one that is wrong (2026-08-12 review, S2; an
    /// earlier version kept "verified" static folds and a wrap-tainted fold
    /// sailed through both builds identically). The bounded (`with_bounds`)
    /// family is unaffected — its undischarged checks are compile errors,
    /// so it has no device placement to fall back to.
    pub fn disabled() -> Self {
        Self {
            hoist_to_preheaders: false,
            relocate_to_launch: false,
            discharge_proofs: false,
        }
    }

    /// Resolves the process-level ablation switches, once per compile.
    ///
    /// `CUTILE_FORCE_DEVICE_CHECKS=1` selects [`disabled`](Self::disabled)
    /// (the differential harness's reference build).
    /// `CUTILE_DISABLE_CHECK_HOISTING=1` turns off preheader hoisting only
    /// (the A/B ablation for hoisting itself). Both are read uncached so
    /// A/B toggling across subprocesses never needs a rebuild — they are
    /// compile-time (JIT) knobs, not kernel-time ones.
    pub fn from_env() -> Self {
        let flag = |name: &str| env::var(name).is_ok_and(|v| v == "1");
        if flag("CUTILE_FORCE_DEVICE_CHECKS") {
            return Self::disabled();
        }
        Self {
            hoist_to_preheaders: !flag("CUTILE_DISABLE_CHECK_HOISTING"),
            ..Self::release()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_tile_runtime_utils::{TileirasOptions, DEFAULT_OPT_LEVEL};
    use crate::hints::CompileOptions;

    #[test]
    fn device_debug_implies_opt_zero_unless_explicit() {
        let debug =
            TileirasOptions::from_compile_options(&CompileOptions::new().device_debug(true));
        assert_eq!(debug.opt_level, 0);
        assert!(debug.device_debug);

        let explicit = TileirasOptions::from_compile_options(
            &CompileOptions::new().device_debug(true).opt_level(2),
        );
        assert_eq!(explicit.opt_level, 2, "an explicit level wins");

        let release = TileirasOptions::from_compile_options(
            &CompileOptions::new().debug_info(crate::hints::DebugInfoLevel::None),
        );
        assert_eq!(release.opt_level, DEFAULT_OPT_LEVEL);
        assert_eq!(release, TileirasOptions::default());
    }

    #[test]
    fn flags_byte_is_injective_over_the_flag_combinations() {
        let mut seen = std::collections::BTreeSet::new();
        for dd in [false, true] {
            for li in [false, true] {
                for sm in [false, true] {
                    let o = TileirasOptions {
                        opt_level: 3,
                        device_debug: dd,
                        lineinfo: li,
                        sanitize_memcheck: sm,
                    };
                    assert!(seen.insert(o.flags_byte()), "flags_byte collision");
                }
            }
        }
        assert_eq!(
            TileirasOptions::default().flags_byte(),
            0,
            "release flags must encode as 0, matching the byte old cache entries carry"
        );
    }

    #[test]
    fn named_policies_differ_only_where_documented() {
        let release = CheckOptimizations::release();
        let debug = CheckOptimizations::device_debug();
        let disabled = CheckOptimizations::disabled();

        assert!(release.hoist_to_preheaders && !debug.hoist_to_preheaders);
        // Debug removes in-kernel motion ONLY: discharge and launch
        // relocation still apply (the bounded family's launch checks are
        // load-bearing).
        assert!(debug.relocate_to_launch && debug.discharge_proofs);
        assert!(
            !disabled.hoist_to_preheaders
                && !disabled.relocate_to_launch
                && !disabled.discharge_proofs
        );
    }
}
