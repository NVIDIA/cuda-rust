/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{Context, DeviceBuffer};
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry]
    fn double<const S: [i32; 1]>(out: &mut Tensor<f32, S>, input: &Tensor<f32, { [-1] }>) {
        out.store(load_tile_like(input, out) + load_tile_like(input, out));
    }
}

#[test]
fn typed_buffer_tile_kernel_and_back_preserve_allocation() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let buffer = DeviceBuffer::from_host(&stream, &[3.0f32; 32]).unwrap();
    let pointer = buffer.cu_deviceptr();
    let identity = buffer.storage().allocation_id();
    let tensor = Tensor::from_buffer(buffer, vec![32]);
    assert_eq!(tensor.as_buffer().unwrap().cu_deviceptr(), pointer);
    assert_eq!(
        tensor.as_buffer().unwrap().storage().allocation_id(),
        identity
    );
    assert_eq!(tensor.context(), &context);
    let output = Tensor::from_buffer(DeviceBuffer::<f32>::zeroed(&stream, 32).unwrap(), vec![32]);
    let output_pointer = output.as_buffer().unwrap().cu_deviceptr();
    let output = kernels::double(output.partition([32]), &tensor)
        .sync_on(&stream)
        .unwrap();
    let output = output.0.unpartition().into_buffer();
    assert_eq!(output.cu_deviceptr(), output_pointer);
    assert_eq!(output.to_host_vec(&stream).unwrap(), vec![6.0; 32]);
    let buffer = tensor.into_buffer();
    assert_eq!(buffer.cu_deviceptr(), pointer);
    assert_eq!(buffer.storage().allocation_id(), identity);
}

#[test]
fn tensor_slices_preserve_byte_offset_and_reject_strided_buffer_views() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let tensor = Tensor::from_buffer(
        DeviceBuffer::from_host(&stream, &[1u32; 16]).unwrap(),
        vec![4, 4],
    );
    let slice = tensor.slice(&[1..3]).unwrap();
    let view = slice.as_buffer().unwrap();
    let base = tensor.as_buffer().unwrap();
    assert_eq!(view.cu_deviceptr(), base.cu_deviceptr() + 16);
    assert_eq!(view.len(), 8);
    assert_eq!(view.context(), base.context());
    assert_eq!(
        view.storage().allocation_id(),
        base.storage().allocation_id()
    );
    assert!(tensor.slice(&[0..4, 1..3]).unwrap().as_buffer().is_err());
}

#[test]
fn typed_ownership_transfer_rejects_a_live_boolean_alias() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let tensor = Arc::new(Tensor::from_buffer(
        DeviceBuffer::from_host(&stream, &[true; 8]).unwrap(),
        vec![8],
    ));
    let alias = (&tensor).reshape(&[2, 4]).unwrap();
    let tensor = Arc::try_unwrap(tensor).ok().unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tensor.into_buffer())).is_err()
    );
    assert_eq!(alias.as_buffer().unwrap().len(), 8);
}
