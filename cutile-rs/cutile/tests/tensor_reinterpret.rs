/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::DeviceOp;
use half::f16;

mod common;

#[cutile::module]
mod tensor_reinterpret_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn passthrough_f32(output: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let tile: Tile<f32, { [4] }> = load_tile_like(input, output);
        output.store(tile);
    }
}

#[test]
fn reinterpret_is_zero_copy() {
    common::with_test_stack(|| {
        let bits: Arc<Vec<u32>> = Arc::new(vec![0x3f800000, 0x40000000, 0x40400000, 0x40800000]);
        let base = Arc::new(api::copy_host_vec_to_device(&bits).sync().expect("Failed."));
        let floats_2d = base.reinterpret::<f32>(&[2, 2]).unwrap();

        assert_eq!(
            base.device_pointer().cu_deviceptr(),
            floats_2d.device_pointer().cu_deviceptr()
        );
        assert_eq!(floats_2d.shape(), vec![2, 2]);
        assert_eq!(floats_2d.strides(), vec![2, 1]);
        assert_eq!(floats_2d.size(), 4);

        let host: Vec<f32> = floats_2d.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![1.0, 2.0, 3.0, 4.0]);
    });
}

#[test]
fn reinterpret_rejects_invalid_byte_size() {
    common::with_test_stack(|| {
        let bytes: Arc<Vec<u8>> = Arc::new(vec![1, 2, 3]);
        let base = Arc::new(
            api::copy_host_vec_to_device(&bytes)
                .sync()
                .expect("Failed."),
        );

        assert!(base.reinterpret::<u32>(&[1]).is_err());
        assert!(base.reinterpret::<u32>(&[1]).is_err());
    });
}

#[test]
fn reinterpret_u8_to_i16_succeeds() {
    common::with_test_stack(|| {
        let expected = vec![0x1234i16, 0x2BCDi16];
        let bytes: Arc<Vec<u8>> = Arc::new(
            expected
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        let base = Arc::new(
            api::copy_host_vec_to_device(&bytes)
                .sync()
                .expect("Failed."),
        );
        let words = base.reinterpret::<i16>(&[expected.len()]).unwrap();

        assert_eq!(
            base.device_pointer().cu_deviceptr(),
            words.device_pointer().cu_deviceptr()
        );
        let host: Vec<i16> = words.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, expected);
    });
}

#[test]
fn reinterpret_u8_boundaries_are_enforced() {
    common::with_test_stack(|| {
        let valid_bits = [0x3f800000u32, 0x40000000u32];
        let valid_bytes: Arc<Vec<u8>> = Arc::new(
            valid_bits
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        let valid = Arc::new(
            api::copy_host_vec_to_device(&valid_bytes)
                .sync()
                .expect("Failed."),
        );
        let floats = valid.reinterpret::<f32>(&[2]).unwrap();
        let host: Vec<f32> = floats.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![1.0, 2.0]);

        let odd_bytes: Arc<Vec<u8>> = Arc::new(vec![0, 0, 0, 0, 0, 0]);
        let odd = Arc::new(
            api::copy_host_vec_to_device(&odd_bytes)
                .sync()
                .expect("Failed."),
        );
        assert!(odd.reinterpret::<f32>(&[2]).is_err());

        let odd_words: Arc<Vec<u8>> = Arc::new(vec![1, 2, 3]);
        let odd_words = Arc::new(
            api::copy_host_vec_to_device(&odd_words)
                .sync()
                .expect("Failed."),
        );
        assert!(odd_words.reinterpret::<i16>(&[1]).is_err());
    });
}

#[test]
fn reinterpret_i16_to_f16_succeeds() {
    common::with_test_stack(|| {
        let expected = vec![f16::from_f32(1.0), f16::from_f32(-2.5), f16::from_f32(0.5)];
        let bits: Arc<Vec<i16>> = Arc::new(
            expected
                .iter()
                .map(|value| i16::from_ne_bytes(value.to_bits().to_ne_bytes()))
                .collect(),
        );
        let base = Arc::new(api::copy_host_vec_to_device(&bits).sync().expect("Failed."));
        let halfs = base.reinterpret::<f16>(&[expected.len()]).unwrap();

        assert_eq!(
            base.device_pointer().cu_deviceptr(),
            halfs.device_pointer().cu_deviceptr()
        );
        let host: Vec<f16> = halfs.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, expected);
    });
}

#[test]
fn reinterpreted_tensors_work_with_kernels() {
    common::with_test_stack(|| {
        let bits: Arc<Vec<u32>> = Arc::new(vec![0x3f800000, 0x40000000, 0x40400000, 0x40800000]);
        let base = Arc::new(api::copy_host_vec_to_device(&bits).sync().expect("Failed."));
        let floats = base.reinterpret::<f32>(&[4]).unwrap();

        let output = api::zeros::<f32>(&[4]).sync().expect("Failed.");
        let (result, _input) =
            tensor_reinterpret_module::passthrough_f32(output.partition([4]), floats)
                .sync()
                .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");

        assert_eq!(host, vec![1.0, 2.0, 3.0, 4.0]);
    });
}
