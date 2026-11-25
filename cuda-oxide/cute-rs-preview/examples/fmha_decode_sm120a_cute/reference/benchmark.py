#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked direct-event timing for the dedicated split-KV decode kernel sequence."""

from launch import parse_args, run


if __name__ == "__main__":
    run(parse_args())
