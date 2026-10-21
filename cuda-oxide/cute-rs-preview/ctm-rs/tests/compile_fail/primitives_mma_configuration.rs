/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject unsupported CTA-group, M-dimension and operand-major combinations.
#![allow(unused_imports)]
use ctm_rs::{groups::{CtaGroup1,CtaGroup2},memory::*,tcgen05::{descriptors::*,mma::*,tmem::*}};
/// Supported group-two M256/N192 specification used to distinguish incompatible MMA configurations.
type L=DenseSharedF16<CtaGroup2,256,192,KMajor,KMajor>;
/// Supported group-two M256/N160 specification used to distinguish incompatible MMA configurations.
type R=DenseSharedF16<CtaGroup2,256,160,KMajor,KMajor>;
/// Unsupported external major-mode marker.
struct MnMajor;
/// Reject unsupported CTA-group, M-dimension and operand-major combinations.
fn wrong() {
 let _ = InstructionDescriptor::<DenseSharedF16<CtaGroup1,256,192,KMajor,KMajor>>::new();
 let _ = InstructionDescriptor::<DenseSharedF16<CtaGroup2,128,192,KMajor,KMajor>>::new();
 let _ = InstructionDescriptor::<DenseSharedF16<CtaGroup2,256,192,MnMajor,KMajor>>::new();
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
