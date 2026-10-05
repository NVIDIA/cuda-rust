/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Architecture feature detection and PTX target selection.
//!
//! Detects the architecture and PTX-ISA requirements of exported LLVM IR and
//! selects the minimum `sm_XX` that can lower them. The backend owns this so an
//! standalone frontend gets the same target selection as the Rust MIR path
//! in `mir-importer`.

mod arch;
mod detect;
mod features;
mod generated_requirements;
mod select;
#[cfg(test)]
mod tests;

pub use arch::arch_satisfies;
pub use detect::{detect_features_in_llvm_text, detect_module_requirements_in_llvm_file};
pub use features::{DetectedFeatures, ModuleRequirements};
pub(crate) use generated_requirements::{
    generated_ptx_isa_requirement, generated_target_satisfied, merge_generated_module_requirements,
    merge_generated_module_requirements_for_target, validate_generated_target,
};
pub use select::{required_ptx_feature, validate_target_features, validate_target_for_llvm_major};
pub(crate) use select::{
    resolve_ptx_target_with_generated, select_target_with_generated,
    validate_ptx_isa_for_llvm_major,
};

// The re-exports below preserve the module's pre-split surface. In-crate
// consumers currently reach these names only from test code (or straight
// through the submodules), so without the `allow` the compiler flags them
// as unused imports in non-test builds.
#[allow(unused_imports)]
pub use features::PtxIsaRequirement;
#[allow(unused_imports)]
pub(crate) use generated_requirements::generated_ptx_isa_requirement_for_target;
#[cfg(test)]
#[allow(unused_imports)]
pub use select::resolve_ptx_target;
#[allow(unused_imports)]
pub use select::select_target;
