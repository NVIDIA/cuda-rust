/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compatibility and grouped imports must name the same public types.

macro_rules! same_type {
    ($legacy:ty, $grouped:ty) => {
        let _: fn($legacy) -> $grouped = core::convert::identity;
    };
}

#[test]
fn hidden_epilogue_hooks_keep_their_import_path() {
    fn same_item<T>(_: T, _: T) {}

    same_item(
        cute_rs::epilogue::__compiler::epilogue_sync_reusable,
        cute_rs::epilogue::sm120::__compiler::epilogue_sync_reusable,
    );
}

#[test]
fn compatibility_aliases_preserve_type_identity() {
    same_type!(
        cute_rs::bf16_mma::Bf16Mma16x8x16,
        cute_rs::mma::bf16::Bf16Mma16x8x16
    );
    same_type!(
        cute_rs::block_scaled_mma::Mxfp4TiledMma<()>,
        cute_rs::mma::block_scaled::Mxfp4TiledMma<()>
    );
    same_type!(cute_rs::sm100_attention::SharedLayout<64, 64>, cute_rs::attention::sm100::SharedLayout<64, 64>);
    same_type!(
        cute_rs::epilogue::Sm120Epilogue128x128,
        cute_rs::epilogue::sm120::Sm120Epilogue128x128
    );
    same_type!(
        cute_rs::partition::Strided1D,
        cute_rs::tensor::partition::Strided1D
    );
    same_type!(cute_rs::cooperative::SmemTile<f32, ()>, cute_rs::copy::cooperative::SmemTile<f32, ()>);
    same_type!(cute_rs::tiled_copy::SharedTensor<f32, f32, ()>, cute_rs::copy::tiled::SharedTensor<f32, f32, ()>);
    same_type!(cute_rs::tile::GlobalStoreTile<f32, 4>, cute_rs::copy::vector::GlobalStoreTile<f32, 4>);
    same_type!(cute_rs::tma::TmaDesc<f32, ()>, cute_rs::copy::tma::TmaDesc<f32, ()>);
}

#[test]
fn root_exports_preserve_type_identity() {
    same_type!(cute_rs::FragA, cute_rs::mma::FragA);
    same_type!(cute_rs::RegisterTile<f32, 4>, cute_rs::tensor::RegisterTile<f32, 4>);
    same_type!(
        cute_rs::Mxfp4TiledMma<()>,
        cute_rs::mma::block_scaled::Mxfp4TiledMma<()>
    );
    same_type!(
        cute_rs::Sm120Epilogue128x128,
        cute_rs::epilogue::sm120::Sm120Epilogue128x128
    );
    same_type!(cute_rs::SmemTile<f32, ()>, cute_rs::copy::cooperative::SmemTile<f32, ()>);
    same_type!(cute_rs::SharedTensor<f32, f32, ()>, cute_rs::copy::tiled::SharedTensor<f32, f32, ()>);
    same_type!(cute_rs::GlobalStoreTile<f32, 4>, cute_rs::copy::vector::GlobalStoreTile<f32, 4>);
    same_type!(cute_rs::TmaDesc<f32, ()>, cute_rs::copy::tma::TmaDesc<f32, ()>);
}

#[cfg(not(feature = "host"))]
#[test]
fn device_attention_is_available_without_host_metadata() {
    use cute_rs::attention::sm100::{SharedLayout, SmemLayout};

    assert_eq!(SharedLayout::<64, 64>::offset(0, 1), 1);
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
