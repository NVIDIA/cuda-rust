/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `Tensor::store` returns its completion token, `Tensor::token` reads the
//! tensor's current ordering token, and `Tensor::set_token` installs one. Pins that both lower through the
//! JIT and that a kernel which threads them (store -> token -> set) still
//! computes the right result.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod token_module {
    use cutile::core::*;

    /// Stores `x + 1`, reads the token back from the store and from the
    /// tensor, then re-installs it: both spellings must lower and agree
    /// with the tensor's own ordering state.
    #[cutile::entry()]
    fn store_then_token(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let tx: Tile<f32, { [4] }> = x.load_like(z);
        let one: Tile<f32, { [4] }> = broadcast_scalar(1.0, shape![4]);
        let stored: Token = z.store(tx + one);
        let current: Token = z.token();
        unsafe { z.set_token(stored) };
        unsafe { z.set_token(current) };
    }
}

use token_module::store_then_token;

#[test]
fn store_returns_a_token_and_token_reads_it_back() {
    common::with_test_stack(|| {
        let len = 16usize;
        let z_host = store_then_token(api::zeros(&[len]).partition([4]), api::arange::<f32>(len))
            .grid(((len / 4) as u32, 1, 1))
            .first()
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("store_then_token kernel");
        for (i, v) in z_host.iter().enumerate() {
            assert_eq!(*v, i as f32 + 1.0, "index {i}");
        }
    });
}
