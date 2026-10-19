/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Decoy default bin. The host metadata selects the `scale-offset-device`
//! target with `bin = "scale-offset-device"`, so cargo-oxide must never
//! compile this one. It carries no kernels: if the `--bin` selection ever
//! regressed to a package-default build, device codegen would find nothing
//! to emit and the artifact check would fail loudly.

fn main() {
    unreachable!("cargo-oxide must build the scale-offset-device bin, not the package default");
}
