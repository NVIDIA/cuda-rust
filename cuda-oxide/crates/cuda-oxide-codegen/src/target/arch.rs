/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::features::DetectedFeatures;
use cuda_target_spec::{CudaArch, recorded_ptx_floor};

/// Does `arch` (e.g. `"sm_120a"`, `"sm_90"`) support the kernel's detected
/// features?
///
/// tcgen05/TMEM and explicit `cta_group` TMA forms exist only in the sm_100
/// datacenter-Blackwell family: consumer Blackwell (sm_120) and Hopper (sm_90)
/// lack them, so an sm_120 GPU cannot run an sm_100 tcgen05 kernel even though
/// 120 > 100. WGMMA is Hopper-only. The remaining features are forward
/// compatible from their floor (TMA / cluster / sm_90 features need sm_90+,
/// sm_80 features need sm_80+, sm_75 features need sm_75+, and basic needs
/// sm_70+).
///
/// Used to decide whether the GPU in this machine (the `CUDA_OXIDE_DEVICE_ARCH`
/// hint) can actually run the kernel, or whether we must build for the arch the
/// IR requires instead.
pub fn arch_satisfies(arch: &CudaArch, features: DetectedFeatures) -> bool {
    let (capability, suffix) = (arch.capability(), arch.suffix());
    if recorded_ptx_floor(arch).is_err() {
        return false;
    }
    features
        .iter()
        .all(|feature| arch_satisfies_feature(capability, suffix, feature))
}

pub(super) fn arch_satisfies_feature(
    capability: u32,
    suffix: Option<char>,
    feature: DetectedFeatures,
) -> bool {
    let major = capability / 10;
    match feature {
        DetectedFeatures::Blackwell | DetectedFeatures::TmaCtaGroup => {
            supports_tcgen_target(capability, suffix)
        }
        DetectedFeatures::BlackwellAccelerated => {
            supports_blackwell_accelerated_target(capability, suffix)
        }
        DetectedFeatures::BlackwellFamily => supports_blackwell_family_target(capability, suffix),
        DetectedFeatures::MatrixBlackwell => supports_blackwell_matrix_target(capability, suffix),
        DetectedFeatures::ReduxF32 => supports_redux_f32_target(capability, suffix),
        DetectedFeatures::MultimemFp8 => supports_multimem_fp8_target(capability, suffix),
        // The PTX ISA requires only sm_90+. The suffixed targets are advised
        // for performance, so target selection still prefers sm_100a.
        DetectedFeatures::TmaMulticast => major >= 9,
        DetectedFeatures::Wgmma => capability == 90 && suffix == Some('a'),
        DetectedFeatures::Sm100 => is_known_blackwell_capability(capability),
        DetectedFeatures::Tma | DetectedFeatures::Cluster | DetectedFeatures::Sm90 => major >= 9,
        DetectedFeatures::Sm80 => major >= 8,
        DetectedFeatures::Sm75 | DetectedFeatures::Movmatrix | DetectedFeatures::Ldmatrix => {
            capability >= 75
        }
        DetectedFeatures::DynamicStack => capability >= 52,
        // Basic kernels are supported on the project's Volta+ floor. The
        // cross-compilation default remains sm_80, but a detected sm_70/sm_75
        // GPU is a valid and more useful target for `cargo oxide run`.
        DetectedFeatures::Basic => major >= 7,
        // `iter` only yields the single-bit constants above.
        _ => false,
    }
}

/// tcgen05/TMEM exists only on the datacenter Blackwell architecture or
/// family targets. Consumer sm_120 and generic targets without an `a`/`f`
/// suffix do not provide Tensor Memory.
fn supports_tcgen_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        // Architecture-specific targets are exact, not numerically forward
        // compatible. `sm_101a` is the PTX 8.x spelling later renamed to
        // `sm_110a`; accept both spellings plus the distinct sm_103 target.
        Some('a') => matches!(capability, 100 | 101 | 103 | 110),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn supports_blackwell_accelerated_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 103 | 110),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn supports_blackwell_family_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 110 | 120),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110 | 120 | 121),
        _ => false,
    }
}

fn supports_blackwell_matrix_target(capability: u32, suffix: Option<char>) -> bool {
    // LLVM's sm_101 aliases stop selecting these instructions at PTX 9.0.
    match suffix {
        Some('a' | 'f') => matches!(capability, 100 | 103 | 110 | 120 | 121),
        _ => false,
    }
}

/// Floating-point `redux.sync` is scoped to the sm_100/sm_103 family.
fn supports_redux_f32_target(capability: u32, suffix: Option<char>) -> bool {
    matches!(suffix, Some('a' | 'f')) && matches!(capability, 100 | 103)
}

/// FP8 / f16-accumulator multimem forms span several Blackwell architecture
/// targets, but consumer family (`f`) targets do not support the sm_120 line.
fn supports_multimem_fp8_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 103 | 110 | 120 | 121),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn is_known_blackwell_capability(capability: u32) -> bool {
    matches!(capability, 100 | 101 | 103 | 110 | 120 | 121)
}

/// Extract the compute-capability *major* version from an `sm_…` target string.
///
/// CUDA concatenates major+minor without a separator, so `"sm_120a"` is cc 12.0
/// (major 12), `"sm_90"` is cc 9.0, `"sm_103a"` is cc 10.3. We read the digit
/// run after `sm_` and divide by ten. Returns `None` when there are no digits.
#[cfg(test)]
pub(super) fn arch_major(arch: &str) -> Option<u32> {
    arch_compute_capability(arch).map(|capability| capability / 10)
}

/// Extract the numeric compute capability from an `sm_…` target.
#[cfg(test)]
pub(super) fn arch_compute_capability(arch: &str) -> Option<u32> {
    arch_compute_capability_and_suffix(arch).map(|(capability, _)| capability)
}

#[cfg(test)]
pub(super) fn arch_compute_capability_and_suffix(arch: &str) -> Option<(u32, Option<char>)> {
    if !arch.starts_with("sm_") {
        return None;
    }
    let target = arch.parse::<CudaArch>().ok()?;
    Some((target.capability(), target.suffix()))
}
