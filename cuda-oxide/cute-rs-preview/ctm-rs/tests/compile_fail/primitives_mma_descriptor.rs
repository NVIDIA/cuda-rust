/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject an N160 instruction descriptor paired with an N192 MMA specification.
#![allow(unused_imports)]
use ctm_rs::{groups::{CtaGroup1,CtaGroup2},memory::*,tcgen05::{descriptors::*,mma::*,tmem::*}};
/// Supported group-two M256/N192 specification used to distinguish incompatible MMA configurations.
type L=DenseSharedF16<CtaGroup2,256,192,KMajor,KMajor>;
/// Supported group-two M256/N160 specification used to distinguish incompatible MMA configurations.
type R=DenseSharedF16<CtaGroup2,256,160,KMajor,KMajor>;
/// Reject an N160 instruction descriptor paired with an N192 MMA specification.
fn wrong(d:&impl TmemOperand<Group=CtaGroup2,Format=TmemF32<256,192>>,a:&SmemOperand<OperandA,F16Tile128>,b:&SmemOperand<OperandB,F16TileB96>) {
 unsafe { mma::<L,collector_a::Discard>(d,a,b,&InstructionDescriptor::<R>::new(),false) };
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
