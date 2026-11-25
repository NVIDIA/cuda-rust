/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Root and parent-module re-exports must name the same public types as their defining paths.

macro_rules! same_type {
    ($import:ty, $defining:ty) => {
        let _: fn($import) -> $defining = core::convert::identity;
    };
}

#[test]
fn parent_module_re_exports_preserve_type_identity() {
    same_type!(
        cute_rs::Sm120Epilogue128x128,
        cute_rs::nvgpu::warp::epilogue::Sm120Epilogue128x128
    );
}

#[test]
fn root_exports_preserve_type_identity() {
    same_type!(cute_rs::FragmentA, cute_rs::nvgpu::warp::FragmentA);
    same_type!(cute_rs::RegisterTile<f32, 4>, cute_rs::tensor::RegisterTile<f32, 4>);
    same_type!(
        cute_rs::Mxf4TiledMma<()>,
        cute_rs::nvgpu::warp::block_scaled::Mxf4TiledMma<()>
    );
    same_type!(
        cute_rs::Sm120Epilogue128x128,
        cute_rs::nvgpu::warp::epilogue::Sm120Epilogue128x128
    );
    same_type!(cute_rs::SmemTile<f32, ()>, cute_rs::atom::cooperative::SmemTile<f32, ()>);
    same_type!(cute_rs::SharedTensor<f32, f32, ()>, cute_rs::atom::tiled::SharedTensor<f32, f32, ()>);
    same_type!(cute_rs::GlobalStoreTile<f32, 4>, cute_rs::atom::vector::GlobalStoreTile<f32, 4>);
    same_type!(cute_rs::TensorMap<f32, ()>, cute_rs::nvgpu::cpasync::TensorMap<f32, ()>);
    same_type!(cute_rs::Layout<(), ()>, cute_rs::layout::Layout<(), ()>);
    same_type!(cute_rs::ComposedLayout<(), 0, ()>, cute_rs::layout::ComposedLayout<(), 0, ()>);
    same_type!(cute_rs::CopyG2S<16>, cute_rs::layout::CopyG2S<16>);
    same_type!(cute_rs::ThrCopy<(), (), (), ()>, cute_rs::atom::tiled::ThrCopy<(), (), (), ()>);
}

#[cfg(not(feature = "host"))]
#[test]
fn device_tcgen05_shared_layout_is_available_without_host_feature() {
    use cute_rs::nvgpu::tcgen05::{SharedLayout, SmemLayout};

    assert_eq!(SharedLayout::<64, 64>::crd2idx(0, 1), 1);
    assert_eq!(SharedLayout::<64, 64>::ROWS, 64);
}
