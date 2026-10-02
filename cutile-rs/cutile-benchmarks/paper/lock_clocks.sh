#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Lock / unlock GPU clocks for reproducible paper-final benchmark numbers.
#
# 2400 MHz is the paper's reference clock (MACHINE.md): well within the
# sustained thermal envelope of the NVIDIA GeForce RTX 5090 and high enough that GEMMs
# stay in the tensor-core peak regime.
#
# Usage:
#   sudo ./lock_clocks.sh          # lock to 2400 MHz
#   sudo ./lock_clocks.sh unlock   # release the lock
#   sudo ./lock_clocks.sh <mhz>    # lock to a specific rate
#
# Needs sudo to talk to the driver.
set -euo pipefail

DEFAULT_MHZ=2400
arg="${1:-lock}"

if [[ "$arg" == "unlock" || "$arg" == "reset" || "$arg" == "-u" ]]; then
    echo "Releasing GPU clock lock..."
    sudo nvidia-smi -rgc
    exit 0
fi

if [[ "$arg" == "lock" ]]; then
    MHZ="$DEFAULT_MHZ"
else
    MHZ="$arg"
fi

if ! [[ "$MHZ" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 [lock|unlock|<mhz>]" >&2
    exit 2
fi

echo "Locking GPU SM clock to ${MHZ} MHz..."
sudo nvidia-smi -lgc "${MHZ},${MHZ}"
