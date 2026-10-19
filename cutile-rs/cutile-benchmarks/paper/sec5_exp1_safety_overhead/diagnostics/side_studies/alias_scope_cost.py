# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cost of alias scope broadening in cuTile Python.

When control flow merges a read-only parameter into the same alias set
as a write target, ALL operations on that set are ordered — even reads
that could safely execute in parallel. This is the actual cost of not
having type-level aliasing information.

Kernel: Read from x, conditionally store to either x or y.
  - If x and target are in the same alias set (because target = x in
    one branch), the load from x is ordered relative to the store.
  - In Rust, x: &Tensor and the store requires &mut — they can't alias.

We compare:
  1. safe_version: read from x, store to y (separate alias sets)
  2. ambiguous_version: read from x, store to x-or-y (merged alias set)

Usage:
    python3 alias_scope_cost.py
"""

import cuda.tile as ct
from cuda.tile._compile import _bind_kernel_arguments, _get_final_ir
from cuda.tile._const_utils import get_constant_annotations
from cuda.tile._context import TileContextConfig
import inspect
import torch

ConstInt = ct.Constant[int]


# --- Kernel 1: Clean separation (like Rust) ---
# x is read-only, y is write-only. Separate alias sets.
# Load from x is unconstrained relative to store to y.

@ct.kernel
def clean_separation(x, y, TILE: ConstInt):
    """Read x, write y. No aliasing ambiguity."""
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx + tx)


# --- Kernel 2: Ambiguous store target ---
# The store target might be x or y. The alias analysis merges x into
# the same alias set as target. Now the load from x is in the same
# alias set as the store — it must be ordered.

@ct.kernel
def ambiguous_target(x, y, cond: int, TILE: ConstInt):
    """Read x, store to x-or-y depending on cond."""
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    if cond:
        target = x  # alias analysis merges x and target
    else:
        target = y
    ct.store(target, index=(bid,), tile=tx + tx)


# --- Kernel 3: Multi-read with ambiguous store ---
# Multiple reads from x, plus a conditional store to x-or-y.
# All reads from x get ordered relative to the store if alias sets merge.

@ct.kernel
def multi_read_ambiguous(x, y, cond: int, TILE: ConstInt):
    """Multiple reads from x, conditional store to x-or-y."""
    bid = ct.bid(0)
    t1 = ct.load(x, index=(bid,), shape=(TILE,))
    t2 = ct.load(x, index=(bid,), shape=(TILE,))
    t3 = ct.load(x, index=(bid,), shape=(TILE,))
    if cond:
        target = x
    else:
        target = y
    ct.store(target, index=(bid,), tile=t1 + t2 + t3)


def get_ir(kernel_obj, args, label):
    """Compile and dump IR with token analysis."""
    pyfunc = kernel_obj._pyfunc
    param_names = tuple(inspect.signature(pyfunc).parameters.keys())
    const_args = get_constant_annotations(pyfunc)
    ir_args = _bind_kernel_arguments(param_names, args, const_args)

    config = TileContextConfig(
        temp_dir='/tmp',
        log_keys=frozenset(),
        compiler_timeout_sec=None,
        enable_crash_dump=False,
        cache_dir=None,
        cache_size_limit=100_000_000,
    )

    func_ir = _get_final_ir(pyfunc, ir_args, config)
    ir_text = func_ir.body.to_string(include_loc=False)

    n_tko = ir_text.count("_token_ordered")
    n_join = ir_text.count("join_tokens")

    # Count non-root token dependencies (operations that depend on
    # something other than the initial $token)
    lines = ir_text.split('\n')
    non_root_deps = 0
    for line in lines:
        if '_token_ordered' in line and 'token=' in line:
            # Extract the token= argument
            import re
            m = re.search(r'token=(\$\S+)', line)
            if m and m.group(1) != '$token':
                non_root_deps += 1

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(ir_text)
    print(f"\n  Token stats: {n_tko} tko ops, {n_join} joins, "
          f"{non_root_deps} non-root token deps")
    print(f"  (Non-root deps = operations forced to wait for prior ops)")

    return n_tko, n_join, non_root_deps


def main():
    N = 1024
    TILE = 128
    device = "cuda"

    x = torch.randn(N, dtype=torch.float32, device=device)
    y = torch.zeros(N, dtype=torch.float32, device=device)

    print("=== Alias Scope Cost ===\n")
    print("QUESTION: When alias sets broaden (due to control flow),")
    print("do read-only loads get unnecessarily ordered with writes?\n")

    tko1, join1, deps1 = get_ir(
        clean_separation, (x, y, TILE),
        "CLEAN: read x, write y (separate alias sets)")

    tko2, join2, deps2 = get_ir(
        ambiguous_target, (x, y, 1, TILE),
        "AMBIGUOUS: read x, write x-or-y (merged alias set)")

    tko3, join3, deps3 = get_ir(
        multi_read_ambiguous, (x, y, 1, TILE),
        "MULTI-READ AMBIGUOUS: 3x read x, write x-or-y")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Clean:             {deps1} non-root deps (reads free)")
    print(f"  Ambiguous:         {deps2} non-root deps")
    print(f"  Multi-read ambig:  {deps3} non-root deps")

    if deps2 > deps1 or deps3 > deps1:
        print()
        print("  CONFIRMED: Alias set broadening forces reads to wait.")
        print("  In Rust, &Tensor reads are ALWAYS unconstrained —")
        print("  the type system guarantees non-aliasing.")
        print("  This is the cost Python pays for not having type-level info.")


if __name__ == "__main__":
    main()
