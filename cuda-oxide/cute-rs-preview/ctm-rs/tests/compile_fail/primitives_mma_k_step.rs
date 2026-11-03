/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject K16 slice index four in a four-slice shared operand.
#![allow(unused_imports)]
use ctm_rs::{groups::{CtaGroup1,CtaGroup2},memory::*,tcgen05::{descriptors::*,mma::*,tmem::*}};
/// Supported group-two M256/N192 specification used to distinguish incompatible MMA configurations.
type L=DenseSharedF16<CtaGroup2,256,192,KMajor,KMajor>;
/// Supported group-two M256/N160 specification used to distinguish incompatible MMA configurations.
type R=DenseSharedF16<CtaGroup2,256,160,KMajor,KMajor>;
/// Reject K16 slice index four in a four-slice shared operand.
fn wrong(a:&SmemOperand<OperandA,F16Tile128>) { let _ = a.k16::<4>(); }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
