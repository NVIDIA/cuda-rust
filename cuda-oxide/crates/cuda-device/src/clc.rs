/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cluster Launch Control intrinsics for Blackwell and newer GPUs.
//!
//! A running CTA can request a not-yet-launched CTA and decode its grid
//! coordinates from the returned 16-byte response.

use crate::barrier::Barrier;

include!("generated/clc.rs");
