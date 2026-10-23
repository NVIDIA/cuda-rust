// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Only 0 and 1 are valid `bool` values, so a byte tensor must not become a
// `Tensor<bool>` through the safe reinterpreting cast: the target type has to
// implement `bytemuck::AnyBitPattern`, and `bool` does not.
use cutile::tensor::Tensor;
use std::sync::Arc;

fn forge(bytes: Arc<Tensor<u8>>) -> Arc<Tensor<bool>> {
    bytes.reinterpret::<bool>(&[4]).unwrap()
}

fn main() {
    let _ = forge;
}
