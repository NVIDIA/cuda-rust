# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


#!/bin/bash

export CLOCK=${1:-$1}
export DEV=0
sudo nvidia-smi -pm 0 -i $DEV
sudo nvidia-smi -pm 1 -i $DEV
sudo nvidia-smi -i $DEV -pl 220
sudo nvidia-smi -i $DEV -rmc
sudo nvidia-smi -i $DEV -rgc

sudo nvidia-smi -i $DEV --lock-gpu-clocks=1350,1350
