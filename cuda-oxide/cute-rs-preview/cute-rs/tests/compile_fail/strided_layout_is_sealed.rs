/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::tensor::partition::IndexLayout;
#[derive(Clone, Copy)]
struct Escaping;
impl IndexLayout for Escaping {
    fn len(&self) -> usize {
        1
    }
    fn crd2idx(&self, _: usize) -> usize {
        usize::MAX
    }
}
fn main() {}
