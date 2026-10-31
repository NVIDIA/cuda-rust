// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A safe implementation must not relabel boolean storage as u8 kernel output.
use cuda_async::{
    device_operation::{value, DeviceOp, ExecutionContext},
    error::DeviceError,
    launch::AsyncKernelLaunchBuilder,
};
use cutile::cutile_compiler::specialization::SpecializationBits;
use cutile::error::Error;
use cutile::tensor::{
    IntoPartition, KernelOutput, KernelOutputStored, Partition, Tensor, ToHostVec,
};
use cutile::{api, cuda_async};
use std::sync::Arc;

struct Mislabelled(Partition<Tensor<bool>>);
impl KernelOutputStored<u8> for Mislabelled {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.0.retain(ctx)
    }
    fn push_kernel_args(&self, launch: &mut AsyncKernelLaunchBuilder) {
        self.0.push_kernel_args(launch);
    }
    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        self.0.grid()
    }
    fn dtype_str(&self) -> &'static str {
        "u8"
    }
    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.0.partition_shape_as_i32()
    }
    fn partition_shape(&self) -> &[usize] {
        KernelOutputStored::partition_shape(&self.0)
    }
    fn strides_hint(&self) -> Vec<i32> {
        self.0.strides_hint()
    }
    fn spec(&self) -> &SpecializationBits {
        self.0.spec()
    }
    fn shape_as_i32(&self) -> Vec<i32> {
        self.0.shape_as_i32()
    }
}
impl KernelOutput<u8> for Mislabelled {
    type Stored = Self;
    type Returned = Partition<Tensor<bool>>;
    fn prepare(self) -> Self {
        self
    }
    fn recover(stored: Self) -> Self::Returned {
        stored.0
    }
}
fn main() {
    let output = api::copy_host_vec_to_device(&Arc::new(vec![false; 128]))
        .sync()
        .unwrap();
    let (_, output) =
        cutile::kernels::creation::full(2_u8, value(Mislabelled(output.partition([128]))))
            .sync()
            .unwrap();
    let output: Arc<Tensor<bool>> = Arc::new(output.unpartition());
    // Inspect bytes via AnyBitPattern u8. Never materialize invalid Rust bools.
    let bytes = output
        .reinterpret::<u8>(&[128])
        .unwrap()
        .to_host_vec()
        .sync()
        .unwrap();
    println!("Tensor<bool> after safe u8 kernel: {:?}", &bytes[..8]);
    assert!(bytes.iter().all(|&b| b == 2));
}
