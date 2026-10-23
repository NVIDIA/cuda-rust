/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cooperative and TMA transfers, including their layout-aware address builders.

pub(in crate::cute) mod cooperative;
pub(in crate::cute) mod emit;
pub(in crate::cute) mod registry;
pub(in crate::cute) mod sm100_attention;
pub(in crate::cute) mod tma;
