// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A safe implementation must not relabel arbitrary bytes as boolean input.
use cutile::cuda_async::{
    device_operation::ExecutionContext, error::DeviceError, launch::AsyncKernelLaunch,
};
use cutile::cutile_compiler::specialization::SpecializationBits;
use cutile::tensor::{KernelInput, KernelInputStored, Tensor};
use std::sync::Arc;

struct Mislabelled(Arc<Tensor<u8>>);
impl KernelInputStored for Mislabelled {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.0.retain(ctx)
    }
    fn push_kernel_args(&self, launch: &mut AsyncKernelLaunch) {
        self.0.push_kernel_args(launch);
    }
    fn shape(&self) -> &[i32] {
        self.0.shape()
    }
    fn strides(&self) -> &[i32] {
        self.0.strides()
    }
    fn spec(&self) -> &SpecializationBits {
        self.0.spec()
    }
    fn dtype_str(&self) -> &'static str {
        "bool"
    }
}
impl KernelInput<bool> for Mislabelled {
    type Stored = Self;
    type Returned = Arc<Tensor<u8>>;
    fn prepare(self) -> Self {
        self
    }
    fn recover(stored: Self) -> Self::Returned {
        stored.0
    }
}
fn main() {}
