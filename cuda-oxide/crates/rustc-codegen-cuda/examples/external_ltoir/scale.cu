/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ int external_add(int value);

extern "C" __device__ int external_scale_add(int value) {
    return external_add(value * 2);
}
