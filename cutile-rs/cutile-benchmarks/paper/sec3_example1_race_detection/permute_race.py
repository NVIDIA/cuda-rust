# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data race demonstration: kernel-internal index bug in head permutation.

A kernel permutes head dimensions of a (b, h, m, d) tensor. The KERNEL has
a bug: the store index has the `m` and `h2` dimensions swapped. The launcher
is correct (src and dst are distinct tensors).

With grid (B*H, H, 1), for fixed (b, m, h2), H blocks with different h1
each load a different source tile src[b, h1, m, :] and store to the same
destination dst[b, m, h2, 0]. Concurrent writes of different values to
the same location: a data race in Tile IR's memory model (the accesses
are not morally strong — no device-scope ordering, only token ordering
which is intra-tile-thread).

cuTile Rust's partition model makes this inexpressible: the kernel
receives a &mut Tensor partition view bound to a specific destination
tile; it cannot write to the wrong destination index because there is no
destination index to choose.

Usage:
    python3 permute_race.py
"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def permute_heads(src, dst, H: ConstInt, BM: ConstInt, BD: ConstInt):
    """Kernel with BUG: the store index has `m` and `h2` swapped.
    Grid: (B*H, H, 1) — bid(0) encodes batch*h1, bid(1) encodes h2.
    Correct store would be dst[b, h2, m, 0]; buggy store is dst[b, m, h2, 0].
    For fixed (b, m, h2), the H blocks iterating h1 race on the same
    destination address with different source tiles."""
    bh1 = ct.bid(0)
    h2 = ct.bid(1)
    b = bh1 // H
    h1 = bh1 % H
    num_m_tiles = ct.num_tiles(src, axis=2, shape=(1, 1, BM, BD))
    for m in range(num_m_tiles):
        tile = ct.load(src, index=(b, h1, m, 0), shape=(1, 1, BM, BD))
        # BUG: swapped `m` and `h2` in the store index.
        ct.store(dst, index=(b, m, h2, 0), tile=tile)


def run_buggy_kernel(B, H, M, D, BM, BD, stream, src):
    """Run the buggy kernel once and return the resulting dst tensor."""
    dst = torch.zeros(B, H, M, D, dtype=torch.float32, device="cuda")
    grid = (B * H, H, 1)
    ct.launch(stream, grid, permute_heads, (src, dst, H, BM, BD))
    torch.cuda.synchronize()
    return dst


def test_nondeterminism(B, H, M, D, BM, BD, stream, n_trials=20):
    """Run the buggy kernel multiple times with the same input; if outputs
    differ across runs, a data race is confirmed (observable non-determinism)."""
    src = torch.arange(
        B * H * M * D, dtype=torch.float32, device="cuda"
    ).reshape(B, H, M, D)

    # Reference run.
    ref = run_buggy_kernel(B, H, M, D, BM, BD, stream, src)

    race_detected = False
    mismatches = 0
    for trial in range(n_trials):
        out = run_buggy_kernel(B, H, M, D, BM, BD, stream, src)
        if not torch.equal(out, ref):
            race_detected = True
            mismatches += 1
            n_wrong = (out != ref).sum().item()
            total = out.numel()
            print(
                f"  Trial {trial}: NON-DETERMINISTIC -- "
                f"{n_wrong}/{total} elements differ "
                f"({100 * n_wrong / total:.1f}%)"
            )
            if mismatches >= 3:
                break

    return race_detected


def main():
    torch.manual_seed(42)
    stream = torch.cuda.current_stream()

    # Attention-like tensor shapes.
    configs = [
        # (B, H, M, D, BM, BD)
        (1, 8, 128, 64, 32, 64),    # small
        (2, 16, 256, 128, 64, 128), # medium
        (4, 32, 512, 128, 64, 128), # large -- many blocks
    ]

    any_race = False
    for B, H, M, D, BM, BD in configs:
        n_blocks = B * H * H
        print(f"\n--- B={B}, H={H}, M={M}, D={D} ({n_blocks} blocks) ---")
        detected = test_nondeterminism(B, H, M, D, BM, BD, stream)
        if detected:
            any_race = True
        else:
            print("  No non-determinism in 20 trials (output stable).")

    if any_race:
        print("\n=== Data race confirmed. ===")
        print("The kernel-internal index bug produces non-deterministic")
        print("outputs across runs. Per Tile IR's memory model, concurrent")
        print("writes to the same address without device-scope ordering are")
        print("not morally strong and constitute undefined behaviour.")
        print("cuTile Rust's partition model makes this bug inexpressible:")
        print("each tile thread owns a disjoint &mut partition element, so")
        print("there is no destination index for the programmer to get wrong.")
    else:
        print("\n=== Race not observed. ===")
        print("The race is real per the memory model, but scheduling may")
        print("have been deterministic for these configurations. Try larger")
        print("H, more trials, or run on a different device.")


if __name__ == "__main__":
    main()
