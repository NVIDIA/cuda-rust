# partial_warp_reduce

Reducing over the live lanes of a partial warp.

## What this tests

`warp::reduce_sum_f32` and its siblings shuffle with the full 32-lane member
mask, so every lane must be launched and converged. A block whose width is not
a multiple of 32 leaves its last warp short, and the PTX ISA makes `shfl.sync`
undefined when a thread sources a lane that is inactive or outside the member
mask.

`warp::reduce_sum_f32_partial` takes the live-lane count and reduces over
exactly those lanes. Two block widths cover the two paths through it:

  - 48 threads leave a tail warp of 16. A power-of-two count takes the same
    butterfly as a full warp, with the mask and the first offset cut down, so
    `lane ^ offset` stays inside the mask at every step.
  - 45 threads leave a tail warp of 13. No butterfly reaches that count, so the
    reduction folds the upper part of the span into the lower half instead,
    sourcing a clamped lane so no thread ever reads one that was never
    launched.

`maxima_odd_tail` covers the same 13-lane geometry for maximum, where a lane
read from outside the live set replaces the answer rather than perturbing it.

## Usage

```bash
cargo oxide run partial_warp_reduce
```

To run the repository smoketest:

```bash
cuda-oxide/scripts/smoketest.sh -x -v '^partial_warp_reduce$'
```

## Expected output

```text
sums_pow2_tail: 14 warps, max error 0e0
sums_odd_tail: 14 warps, max error 0e0
maxima_odd_tail: 14 warps, max error 0e0
SUCCESS: partial warps reduce over exactly their live lanes
```

## What this does not show

A wrong answer from the full-warp form. Substituting `warp::reduce_sum_f32`
into these kernels still matches the host reference exactly on sm_120, so the
hardware returns something harmless for a shuffle sourcing a never-launched
lane. The ISA promises nothing about that value, on this architecture or
another, which is why the mask names the live lanes rather than relying on it.
