/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Types used for launch validation.
//!
//! The compiler emits a [`Validator`]. The host checks a launch against it.

use crate::predicate::LaunchCheck;

#[derive(Debug, Clone)]
pub enum ValidParamType {
    Scalar(ScalarParamType),
    Pointer(PointerParamType),
    Tensor(TensorParamType),
}

#[derive(Debug, Clone)]
pub struct ScalarParamType {
    pub element_type: String,
}

#[derive(Debug, Clone)]
pub struct PointerParamType {
    pub mutable: bool,
    pub element_type: String,
}

// TODO (hme): This is note entirely tile-agnostic with this param type.
#[derive(Debug, Clone)]
pub struct TensorParamType {
    pub element_type: String,
    pub shape: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct Validator {
    pub params: Vec<ValidParamType>,
    /// Compiler-emitted checks to run at launch, before `cuLaunchKernel`. Each
    /// is a canonical [`crate::predicate::Predicate`] the compiler hoisted out
    /// of the device kernel; the host evaluates it against the launched tensors'
    /// extents. Empty unless a kernel hoists a launch-known safety check.
    pub launch_checks: Vec<LaunchCheck>,
}
