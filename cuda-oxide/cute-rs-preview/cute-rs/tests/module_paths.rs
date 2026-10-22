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
        cute_rs::epilogue::Sm120Epilogue128x128,
        cute_rs::epilogue::sm120::Sm120Epilogue128x128
    );
}

#[test]
fn root_exports_preserve_type_identity() {
    same_type!(cute_rs::FragmentA, cute_rs::mma::FragmentA);
    same_type!(cute_rs::RegisterTile<f32, 4>, cute_rs::tensor::RegisterTile<f32, 4>);
    same_type!(
        cute_rs::Mxf4TiledMma<()>,
        cute_rs::mma::block_scaled::Mxf4TiledMma<()>
    );
    same_type!(
        cute_rs::Sm120Epilogue128x128,
        cute_rs::epilogue::sm120::Sm120Epilogue128x128
    );
    same_type!(cute_rs::SmemTile<f32, ()>, cute_rs::copy::cooperative::SmemTile<f32, ()>);
    same_type!(cute_rs::SharedTensor<f32, f32, ()>, cute_rs::copy::tiled::SharedTensor<f32, f32, ()>);
    same_type!(cute_rs::GlobalStoreTile<f32, 4>, cute_rs::copy::vector::GlobalStoreTile<f32, 4>);
    same_type!(cute_rs::TensorMap<f32, ()>, cute_rs::copy::tma::TensorMap<f32, ()>);
    same_type!(cute_rs::Layout<(), ()>, cute_rs::markers::Layout<(), ()>);
    same_type!(cute_rs::ComposedLayout<(), 0, ()>, cute_rs::markers::ComposedLayout<(), 0, ()>);
    same_type!(cute_rs::CopyG2S<16>, cute_rs::markers::CopyG2S<16>);
    same_type!(cute_rs::ThrCopy<(), (), (), ()>, cute_rs::copy::tiled::ThrCopy<(), (), (), ()>);
}

#[cfg(not(feature = "host"))]
#[test]
fn device_attention_is_available_without_host_metadata() {
    use cute_rs::attention::sm100::{SharedLayout, SmemLayout};

    assert_eq!(SharedLayout::<64, 64>::crd2idx(0, 1), 1);
    assert_eq!(SharedLayout::<64, 64>::ROWS, 64);
}

#[cfg(feature = "host")]
#[test]
fn host_attention_metadata_keeps_parent_imports() {
    same_type!(
        cute_rs::attention::AttentionError,
        cute_rs::attention::metadata::AttentionError
    );
    same_type!(cute_rs::attention::AttentionShape<16, 8, 128>, cute_rs::attention::metadata::AttentionShape<16, 8, 128>);
    same_type!(
        cute_rs::attention::AttentionWindow,
        cute_rs::attention::metadata::AttentionWindow
    );
    same_type!(cute_rs::attention::CacheShape<8, 128>, cute_rs::attention::metadata::CacheShape<8, 128>);
}
