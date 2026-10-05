# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Token ordering comparison: cuTile Python vs cuTile Rust.

HYPOTHESIS: cuTile Python's token ordering pass infers mutability from
usage (stores -> all ops on that alias set get token-ordered). This may
produce SUPERFLUOUS token orderings on read-only tensors. cuTile Rust
knows at the TYPE LEVEL which tensors are mutable (&mut) vs immutable (&),
so only &mut operations get token-ordered. If Python produces more tokens
than Rust for equivalent kernels, the Rust approach gives the Tile IR
compiler more freedom to reorder immutable operations for performance.

WHAT TO LOOK FOR in the IR output:
  - load_view_tko / load_ptr_tko with "token=" on arrays that are NEVER
    stored to. These are superfluous token orderings.
  - In Rust, equivalent loads on &Tensor would have NO tokens.

Usage:
    python3 token_ordering_comparison.py
"""

import os
import sys

# Set CUDA_TILE_LOGS before importing cuda.tile so the IR dump env var
# takes effect on first init.
if "CUDA_TILE_LOGS" not in os.environ:
    os.environ["CUDA_TILE_LOGS"] = "CUTILEIR"

import cuda.tile as ct
import torch
from math import ceil

ConstInt = ct.Constant[int]


# --- Kernel 1: add_accum ---
# c is read+write (should be token-ordered)
# a, b are read-only (should be unconstrained in Rust; check Python)

@ct.kernel
def add_accum(a, b, c, TILE: ConstInt):
    """c[i] += a[i] + b[i]"""
    bid = ct.bid(0)
    tile_a = ct.load(a, index=(bid,), shape=(TILE,))
    tile_b = ct.load(b, index=(bid,), shape=(TILE,))
    tile_c = ct.load(c, index=(bid,), shape=(TILE,))
    ct.store(c, index=(bid,), tile=tile_a + tile_b + tile_c)


# --- Kernel 2: Many reads, one write ---
# dst is write-only (token-ordered in both)
# a, b, c are read-only (should be unconstrained in Rust)

@ct.kernel
def multi_read_single_write(a, b, c, dst, TILE: ConstInt):
    """dst[i] = a[i] + b[i] + c[i]"""
    bid = ct.bid(0)
    ta = ct.load(a, index=(bid,), shape=(TILE,))
    tb = ct.load(b, index=(bid,), shape=(TILE,))
    tc = ct.load(c, index=(bid,), shape=(TILE,))
    ct.store(dst, index=(bid,), tile=ta + tb + tc)


# --- Kernel 3: Interleaved read/write ---
# Most interesting: interleaved immutable reads and mutable writes.
# In Rust, data: &Tensor loads are unconstrained; out: &mut Tensor
# ops are token-chained. In Python, if the compiler orders everything
# conservatively, ALL ops may end up in one token chain.

@ct.kernel
def interleaved_access(data, out, TILE: ConstInt):
    """Read data, accumulate into out, read data again, accumulate again."""
    bid = ct.bid(0)
    t1 = ct.load(data, index=(bid,), shape=(TILE,))
    old = ct.load(out, index=(bid,), shape=(TILE,))
    ct.store(out, index=(bid,), tile=old + t1)
    t2 = ct.load(data, index=(bid,), shape=(TILE,))
    cur = ct.load(out, index=(bid,), shape=(TILE,))
    ct.store(out, index=(bid,), tile=cur + t2)


def compile_and_dump(kernel, args, name):
    """Launch once to trigger JIT; CUDA_TILE_LOGS=CUTILEIR dumps IR to stderr."""
    sys.stderr.write(f"\n{'='*70}\n  {name}\n{'='*70}\n")
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), kernel, args)
    torch.cuda.synchronize()


def main():
    sys.stderr.write("=== Token Ordering Comparison: cuTile Python ===\n")
    sys.stderr.write("QUESTION: Does Python produce superfluous token orderings\n")
    sys.stderr.write("on read-only arrays that Rust's type system would leave\n")
    sys.stderr.write("unconstrained?\n")

    TILE = 128
    N = 1024
    device = "cuda"

    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = torch.randn(N, dtype=torch.float32, device=device)
    dst = torch.zeros(N, dtype=torch.float32, device=device)
    data = torch.randn(N, dtype=torch.float32, device=device)
    out = torch.zeros(N, dtype=torch.float32, device=device)

    compile_and_dump(add_accum, (a, b, c, TILE),
                     "add_accum (a=read, b=read, c=read+write)")
    compile_and_dump(multi_read_single_write, (a, b, c, dst, TILE),
                     "multi_read_single_write (a=read, b=read, c=read, dst=write)")
    compile_and_dump(interleaved_access, (data, out, TILE),
                     "interleaved_access (data=read, out=read+write)")

    sys.stderr.write(f"\n{'='*70}\n  NEXT STEPS\n{'='*70}\n")
    sys.stderr.write("1. Look for _tko ops on read-only arrays (a, b, c, data).\n")
    sys.stderr.write("   These are superfluous -- Rust would leave them unconstrained.\n")
    sys.stderr.write("2. Compile equivalent Rust kernels with CUTILE_DUMP=ir.\n")
    sys.stderr.write("3. Compare token counts on read-only parameters.\n")
    sys.stderr.write("4. If Python has MORE tokens, benchmark the performance impact.\n")


if __name__ == "__main__":
    main()
