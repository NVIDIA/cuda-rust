# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Token ordering cost of conservative aliasing in cuTile Python.

FINDING: For simple kernels (each parameter used directly), Python's
alias analysis correctly assigns each parameter its own alias set.
Token ordering is equivalent to Rust — no superfluous orderings.

HOWEVER: When control flow makes aliasing ambiguous (e.g., conditional
assignment), Python conservatively merges alias sets. This forces
token ordering across arrays that Rust knows are non-aliasing via types.

This script tests both cases and compares the IR.

Usage:
    python3 alias_group_cost.py
"""

import cuda.tile as ct
from cuda.tile._compile import _bind_kernel_arguments, _get_final_ir
from cuda.tile._const_utils import get_constant_annotations
from cuda.tile._context import TileContextConfig
import inspect
import torch

ConstInt = ct.Constant[int]


# --- Kernel 1: Simple (no aliasing ambiguity) ---
# Each parameter used directly. Python assigns separate alias sets.
# Token ordering equivalent to Rust.

@ct.kernel
def simple_accum(a, b, out, TILE: ConstInt):
    """out += a + b. No control flow aliasing."""
    bid = ct.bid(0)
    ta = ct.load(a, index=(bid,), shape=(TILE,))
    tb = ct.load(b, index=(bid,), shape=(TILE,))
    old = ct.load(out, index=(bid,), shape=(TILE,))
    ct.store(out, index=(bid,), tile=old + ta + tb)


# --- Kernel 2: Control-flow-dependent aliasing ---
# 'target' could be either 'a' or 'b' depending on a runtime condition.
# Python's alias analysis must conservatively merge a and b's alias sets,
# ordering ALL operations on both as if they might be the same memory.
# In Rust, a: &Tensor and b: &Tensor are provably non-aliasing.

@ct.kernel
def conditional_alias(a, b, out, cond: int, TILE: ConstInt):
    """Read from either a or b depending on cond, write to out."""
    bid = ct.bid(0)
    if cond:
        target = a
    else:
        target = b
    t = ct.load(target, index=(bid,), shape=(TILE,))
    old = ct.load(out, index=(bid,), shape=(TILE,))
    ct.store(out, index=(bid,), tile=old + t)


# --- Kernel 3: Loop aliasing ---
# alias changes inside a loop. Conservative analysis merges.

@ct.kernel
def loop_alias(a, b, out, n: int, TILE: ConstInt):
    """Alternately read from a and b in a loop, accumulate into out."""
    bid = ct.bid(0)
    acc = ct.load(out, index=(bid,), shape=(TILE,))
    alias = a
    for i in range(n):
        t = ct.load(alias, index=(bid,), shape=(TILE,))
        acc = acc + t
        alias = b  # alias changes!
    ct.store(out, index=(bid,), tile=acc)


def get_ir(kernel_obj, args, label):
    """Compile kernel to CuTile IR and count tokens."""
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

    n_make = ir_text.count("make_token")
    n_join = ir_text.count("join_tokens")
    n_tko = ir_text.count("_token_ordered")

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(ir_text)
    print(f"\n  Token stats: make_token={n_make}, join_tokens={n_join}, "
          f"token-ordered ops={n_tko}")

    return n_tko, n_join


def main():
    N = 1024
    TILE = 128
    device = "cuda"

    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    out = torch.zeros(N, dtype=torch.float32, device=device)

    print("=== Token Ordering: Simple vs Control-Flow Aliasing ===\n")
    print("QUESTION: Does control-flow aliasing force conservative ordering?")
    print("In Rust, &Tensor parameters are ALWAYS non-aliasing (type system).")
    print("In Python, the alias analysis may merge sets conservatively.\n")

    # Simple case — no aliasing ambiguity
    tko1, join1 = get_ir(
        simple_accum, (a, b, out, TILE),
        "SIMPLE: a, b, out all separate (no control flow aliasing)")

    # Conditional aliasing — target = a or b
    tko2, join2 = get_ir(
        conditional_alias, (a, b, out, 1, TILE),
        "CONDITIONAL: target = a if cond else b (alias sets may merge)")

    # Loop aliasing — alias changes in loop
    tko3, join3 = get_ir(
        loop_alias, (a, b, out, 4, TILE),
        "LOOP: alias switches from a to b inside loop")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Simple:      {tko1} tko ops, {join1} joins")
    print(f"  Conditional: {tko2} tko ops, {join2} joins")
    print(f"  Loop:        {tko3} tko ops, {join3} joins")

    if tko2 > tko1 or tko3 > tko1:
        print()
        print("  CONFIRMED: Control-flow aliasing increases token ordering.")
        print("  Python must conservatively order operations across arrays")
        print("  that MIGHT alias through control flow.")
        print("  Rust's type system avoids this: &Tensor is provably non-aliasing.")
    else:
        print()
        print("  Token counts are the same. Check the actual token DEPENDENCIES")
        print("  — the operations might use different input tokens even with")
        print("  the same count.")


if __name__ == "__main__":
    main()
