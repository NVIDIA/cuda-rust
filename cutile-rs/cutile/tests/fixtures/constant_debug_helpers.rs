/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[cutile::module]
pub mod constant_helpers {
    pub fn logical_and(a: bool, b: bool) -> bool {
        a && b
    }

    pub fn logical_or(a: bool, b: bool) -> bool {
        a || b
    }

    pub fn folded() -> i32 {
        let left = 17i32;
        let right = 29i32;
        left + right
    }

    pub fn nested(a: bool, b: bool) -> i32 {
        let both = logical_and(a, b);
        let either = logical_or(a, b);
        let sum = folded();
        if both {
            sum
        } else if either {
            7
        } else {
            11
        }
    }

    pub fn bool_function<const FLAG: bool>(x: i32) -> i32 {
        if FLAG {
            x
        } else {
            x + 1
        }
    }

    pub trait BoolMethod {
        fn bool_method<const FLAG: bool>(self) -> i32;
    }

    impl BoolMethod for i32 {
        fn bool_method<const FLAG: bool>(self) -> i32 {
            if FLAG {
                self
            } else {
                self + 2
            }
        }
    }

    pub fn nested_bindings(x: i32) -> i32 {
        let first = bool_function::<true>(x);
        first.bool_method::<false>()
    }
}
