/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[cutile::module]
pub mod debug_helpers {
    pub fn offset(x: i32) -> i32 {
        x + 7
    }

    pub fn nested(x: i32) -> i32 {
        offset(x) * 2
    }

    pub trait Bump {
        fn bump(self) -> i32;
    }

    impl Bump for i32 {
        fn bump(self) -> i32 {
            self + 3
        }
    }
}
