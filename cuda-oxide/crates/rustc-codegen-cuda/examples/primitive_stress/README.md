# primitive_stress

Small stress test for primitive scalar support in cuda-oxide.

It covers cases that are easy for a MIR importer or lowering pass to miss:

- `char` constants and casts.
- `u128` / `i128` constants, arithmetic, shifts, and ABI passing.
- `usize` / `isize` arithmetic.
- Rust integer bit methods (`rotate_left`, `rotate_right`, `count_ones`,
  `leading_zeros`, `trailing_zeros`, `swap_bytes`, `reverse_bits`).
- Rust integer saturating arithmetic (`saturating_add`, `saturating_sub`).
- Rust float math methods (`abs`, `copysign`, `floor`, `ceil`, `round`,
  `trunc`, `mul_add`, `powi`, `powf`, `sqrt`, `exp`, `exp2`, `ln`, `log2`,
  `log10`, `sin`, `cos`) plus the `core::f32::math` / `core::f64::math`
  free-function forms.

Run it with:

```bash
cargo oxide run primitive_stress
```

## How code reaches the GPU

The integer and bit kernels lower to plain LLVM intrinsics that `llc`
handles fine — the standard `.ll → .ptx → cuModuleLoad` path works.

The float-math kernel lowers to `__nv_*` libdevice calls (`__nv_sinf`,
`__nv_powf`, etc.). `llc` cannot resolve those, so cuda-oxide:

1. Auto-detects the `__nv_*` calls in the lowered LLVM module.
2. Forces NVVM IR mode and emits only `primitive_stress.ll` (no `.ptx`).
3. The example calls the generated `kernels::load(&ctx)` loader, which
   reads the embedded NVVM IR payload and transparently:
     - `dlopen`s `libnvvm.so` and `libnvJitLink.so` from the CUDA Toolkit
       (via the [`libnvvm-sys`](../../../libnvvm-sys) and
       [`nvjitlink-sys`](../../../nvjitlink-sys) crates).
     - Compiles the embedded NVVM IR + `libdevice.10.bc` to LTOIR via libNVVM.
     - Links the LTOIR to a cubin via nvJitLink.
     - Loads the cubin and launches every kernel through the generated typed API.

There are no external C tools, no symlinked `tools/` directory, and no
build-pipeline boilerplate to maintain per example. The same embedded loader
works for any standalone project that depends on `cuda-host` and has the CUDA
Toolkit installed.

`CUDA_OXIDE_TARGET` (set automatically when you pass `--arch=<sm_XX>`)
selects the GPU arch; otherwise it defaults to `sm_120`.
`CUDA_OXIDE_LIBDEVICE`, `LIBNVVM_PATH`, and `LIBNVJITLINK_PATH` override
the corresponding discovery searches; without them the helper probes
`CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, and
`/opt/cuda`.
