# math_no_mir_inline

Regression test: `std` float methods (`atan`, `sqrt`, `sinh`, `sin_cos`, ...)
must compile in device code even when rustc's MIR inliner is off.

## The bug

`x.atan()` is a one-line `#[inline]` wrapper in `std`. The collector's
"no std on the GPU" guard knew the function underneath it, but not the
wrapper, and relied on rustc inlining the wrapper away first:

```text
MIR inlining on    kernel -> std::sys::cmath::atanf          ok, whitelisted shim
MIR inlining off   kernel -> std::f32::<impl f32>::atan      FORBIDDEN CRATE panic
                                     |
                                     '-> std::sys::cmath::atanf (never reached)
```

rustc switches MIR inlining off for every `-C incremental` build: cargo's dev
profile, `cargo oxide test`, `cargo oxide build -- ...`, or a shell with
`CARGO_INCREMENTAL=1`. So `cargo oxide run math_atan` (release) worked while
the same `.atan()` in a dev or test build died with:

```text
╔════════════════════════════════════════════════════════════════════╗
║             CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE             ║
║ Device code calls: std::f64::<impl f64>::sin_cos                   ║
║ From crate: 'std'                                                  ║
╚════════════════════════════════════════════════════════════════════╝
```

The fix collects these wrappers as ordinary device functions. Their bodies
bottom out in a `core` intrinsic or a `std::sys::cmath` shim, both of which
already lower to LLVM intrinsics / libdevice.

## Reproducing the original failure

At default release settings the wrapper is inlined and the pattern
disappears, so this crate disables MIR inlining for its own package:

```toml
[profile.release.package.math_no_mir_inline]
rustflags = ["-Zinline-mir=no"]
```

Scoped to this one package on purpose: a global `RUSTFLAGS` entry re-keys
every dependency unit and forces a full second dependency-tree build in CI.

## What the kernels check

Two kernels per width, one per wrapper family:

```text
cmath family   atan atan2 tan sinh exp_m1 hypot     wrapper -> std::sys::cmath shim
core family    sqrt sin exp ln powf sin_cos         wrapper -> core intrinsic
```

In the f32 cmath kernel, `atan` is passed as a function item
(`apply(f32::atan, x)`) rather than called directly, so the collector's
function-item path is exercised as well.

Each thread sums its family's terms for one `(x, y)` pair; the host evaluates
the same expression with `std` and compares within a relative tolerance
(`1e-5` for f32, `1e-12` for f64). Every term is positive on the chosen
inputs (`ln` is applied to `x + 1`, and `x` stays below pi/2 so `tan` is
positive), so the sums have no cancellation and a relative tolerance is
meaningful. The `sin_cos` pair enters as `s + s + c`, which is not symmetric
in `s` and `c`, so a swapped tuple would be caught.

## Run

```bash
cargo oxide run math_no_mir_inline
```

Expected output:

```text
SUCCESS: 16 inputs x 12 std float wrappers x 2 widths match host libm
```
