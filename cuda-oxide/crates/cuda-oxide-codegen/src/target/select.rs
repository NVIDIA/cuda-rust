/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::arch::arch_satisfies;
use super::features::{DetectedFeatures, PtxIsaRequirement};
use super::generated_requirements::{
    for_each_resolved_hardware_alternative, generated_hardware_candidate,
    generated_target_satisfied, resolved_requirement, validate_generated_target,
};
use crate::error::PipelineError;
use crate::generated::GeneratedModuleRequirements;
use cuda_target_spec::{CudaArch, RECORDED_PTX_FLOORS, recorded_ptx_floor};

/// Select a concrete architecture that satisfies every detected feature.
///
/// The first candidate preserves the established default for a module's most
/// restrictive-looking feature. The remaining candidates handle intersections
/// such as WGMMA + TMA multicast, whose only common target is `sm_90a`.
pub fn select_target(features: DetectedFeatures) -> Result<CudaArch, String> {
    let preferred = if features.contains(DetectedFeatures::Blackwell)
        || features.contains(DetectedFeatures::TmaCtaGroup)
        || features.contains(DetectedFeatures::BlackwellAccelerated)
        || features.contains(DetectedFeatures::BlackwellFamily)
        || features.contains(DetectedFeatures::ReduxF32)
        || features.contains(DetectedFeatures::MultimemFp8)
        || features.contains(DetectedFeatures::TmaMulticast)
        || features.contains(DetectedFeatures::MatrixBlackwell)
    {
        "sm_100a"
    } else if features.contains(DetectedFeatures::Wgmma) {
        "sm_90a"
    } else if features.contains(DetectedFeatures::Sm100) {
        "sm_100"
    } else if features.contains(DetectedFeatures::Tma) {
        // Plain TMA is compatible with Hopper, but sm_100 is the existing
        // cross-compilation default because it produces forward-compatible
        // PTX for generic Blackwell devices.
        "sm_100"
    } else if features.contains(DetectedFeatures::Cluster)
        || features.contains(DetectedFeatures::Sm90)
    {
        "sm_90"
    } else if features.contains(DetectedFeatures::Sm80) {
        "sm_80"
    } else if features.contains(DetectedFeatures::Sm75)
        || features.contains(DetectedFeatures::Movmatrix)
        || features.contains(DetectedFeatures::Ldmatrix)
    {
        "sm_75"
    } else {
        "sm_80"
    };

    for candidate in [
        preferred, "sm_100a", "sm_90a", "sm_100", "sm_90", "sm_80", "sm_75",
    ] {
        let arch = candidate.parse().expect("known CUDA target candidate");
        if arch_satisfies(&arch, features) {
            return Ok(arch);
        }
    }

    Err(format!(
        "detected CUDA features {features:?} do not share a compatible GPU architecture"
    ))
}

/// Select one concrete target satisfying both text-detected features and every
/// generated intrinsic used by the module.
///
/// Generated hardware requirements are a module-wide AND. Each intrinsic's
/// `AnyOf` list remains an OR, so the search finds one architecture in the
/// intersection rather than selecting a separate target per call.
pub(crate) fn select_target_with_generated(
    features: DetectedFeatures,
    generated: &GeneratedModuleRequirements,
) -> Result<CudaArch, String> {
    let preferred = select_target(features)?;
    if generated.is_empty() {
        return Ok(preferred);
    }

    let mut candidates = vec![preferred];
    let mut push_candidate = |candidate: CudaArch| {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    };

    // Try each catalog spelling before the exhaustive known-target list. This
    // preserves catalog alternative order while still finding intersections
    // such as `minimum sm_80` AND `sm_90a exactly`.
    for resolved in generated.resolved_targets() {
        let requirement = resolved_requirement(generated, resolved)?;
        for_each_resolved_hardware_alternative(requirement, |alternative| {
            push_candidate(generated_hardware_candidate(alternative));
        });
    }

    // Family and architecture spellings are included because a generated
    // requirement may need to intersect with an existing text-detected feature.
    for entry in RECORDED_PTX_FLOORS {
        push_candidate(
            CudaArch::new(entry.capability, entry.suffix)
                .expect("recorded target floor must identify a valid CUDA target"),
        );
    }

    if let Some(candidate) = candidates.into_iter().find(|candidate| {
        arch_satisfies(candidate, features) && generated_target_satisfied(candidate, generated)
    }) {
        return Ok(candidate);
    }

    let generated_ids = generated
        .resolved_targets()
        .iter()
        .map(|resolved| {
            let selector = resolved
                .selector
                .map(|selector| format!(" for {}={}", selector.name, selector.value))
                .unwrap_or_default();
            format!(
                "{} ({}){selector}",
                resolved.target.id, resolved.target.marker
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "detected CUDA features {features:?} and generated intrinsics [{generated_ids}] do not share a compatible GPU architecture"
    ))
}

pub fn validate_target_features(
    target: &CudaArch,
    features: DetectedFeatures,
) -> Result<(), String> {
    let compatible_default = select_target(features)?;
    if arch_satisfies(target, features) {
        return Ok(());
    }

    Err(format!(
        "CUDA target {} cannot lower detected feature {features:?}; \
         cuda-oxide requires a target compatible with {} for this module",
        target.sm(),
        compatible_default.sm()
    ))
}

#[cfg(test)]
pub fn resolve_ptx_target(
    explicit_override: Option<&str>,
    explicit_override_source: &'static str,
    device_hint: Option<&cuda_target_spec::DeviceArch>,
    detected: DetectedFeatures,
) -> Result<(CudaArch, &'static str), PipelineError> {
    resolve_ptx_target_with_generated(
        explicit_override,
        explicit_override_source,
        device_hint,
        detected,
        &GeneratedModuleRequirements::default(),
    )
}

pub(crate) fn resolve_ptx_target_with_generated(
    explicit_override: Option<&str>,
    explicit_override_source: &'static str,
    device_hint: Option<&cuda_target_spec::DeviceArch>,
    detected: DetectedFeatures,
    generated: &GeneratedModuleRequirements,
) -> Result<(CudaArch, &'static str), PipelineError> {
    let device_hint = device_hint.map(AsRef::as_ref);
    if let Some(target) = explicit_override {
        let parsed =
            target
                .parse::<CudaArch>()
                .map_err(|error| PipelineError::TargetSelection {
                    target: target.to_string(),
                    // `error` already reads "invalid CUDA target `x`: ...";
                    // only the provenance needs to be added here.
                    reason: format!("{error} (target from {explicit_override_source})"),
                })?;
        validate_target_features(&parsed, detected).map_err(|reason| {
            PipelineError::TargetSelection {
                target: parsed.sm(),
                reason: format!("{reason} (target from {explicit_override_source})"),
            }
        })?;
        validate_generated_target(&parsed, generated).map_err(|reason| {
            PipelineError::TargetSelection {
                target: parsed.sm(),
                reason: format!("{reason} (target from {explicit_override_source})"),
            }
        })?;
        return Ok((parsed, explicit_override_source));
    }

    if let Some(device) = device_hint.filter(|target| {
        arch_satisfies(target, detected) && generated_target_satisfied(target, generated)
    }) {
        return Ok((device.clone(), "detected GPU"));
    }

    let target =
        select_target_with_generated(detected, generated).map_err(PipelineError::PtxGeneration)?;
    Ok((target, "feature requirement"))
}

/// Select the PTX ISA independently from the GPU architecture.
///
/// LLVM GPU CPUs select a default PTX ISA independently from the hardware
/// feature floor. Raise that ISA only when the selected CPU's default is too
/// old; never force a newer target back to an older PTX version.
pub fn required_ptx_feature(
    target: &CudaArch,
    requirement: PtxIsaRequirement,
) -> Result<Option<&'static str>, String> {
    let minimum = recorded_ptx_floor(target).map_err(|error| error.to_string())?;
    Ok(requirement
        .spelling()
        .and_then(|spelling| spelling.feature_beyond_floor(minimum)))
}

/// Reject targets that the supported LLVM 21 backend silently mishandles.
///
/// A recorded floor of PTX 9.0 identifies the targets LLVM 21 cannot emit
/// reliably: it accepts their `-mcpu` spellings but only prints a warning and
/// emits PTX 6.0, which ptxas then rejects. LLVM 22 is the first backend in
/// cuda-oxide's supported toolchain set that emits valid PTX for them. An
/// unknown backend version is rejected because it cannot prove support, while
/// unknown targets remain the responsibility of the normal target validators.
pub fn validate_target_for_llvm_major(
    target: &CudaArch,
    llc_major: Option<u32>,
) -> Result<(), String> {
    let requires_llvm_22 = recorded_ptx_floor(target).is_ok_and(|floor| floor >= 90);
    if requires_llvm_22 && llc_major.is_none_or(|major| major < 22) {
        let backend = llc_major.map_or_else(
            || "an LLVM backend with an unknown version".to_string(),
            |major| format!("LLVM {major}"),
        );
        return Err(format!(
            "CUDA target {} requires LLVM 22 or newer; {backend} does not reliably emit valid PTX for this PTX 9.0 target",
            target.sm()
        ));
    }
    Ok(())
}

pub(crate) fn validate_ptx_isa_for_llvm_major(
    requirement: PtxIsaRequirement,
    llc_major: Option<u32>,
) -> Result<(), String> {
    if requirement >= PtxIsaRequirement::new(90) && llc_major.is_none_or(|major| major < 22) {
        let backend = llc_major.map_or_else(
            || "an LLVM backend with an unknown version".to_string(),
            |major| format!("LLVM {major}"),
        );
        return Err(format!(
            "PTX 9.0 or newer requires LLVM 22 or newer; {backend} does not support the required PTX feature"
        ));
    }
    Ok(())
}
