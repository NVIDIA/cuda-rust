# Debugging and Profiling

Start debugging with small, deterministic inputs. Read results back to the host, compare against a CPU reference, then inspect generated Tile IR or profile the GPU when correctness is established.

## Printing and Assertions

`cuda_tile_print!` prints from inside a GPU kernel:

```rust
#[cutile::entry()]
fn debug_kernel<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,
    x: &Tensor<f32, { [-1, -1] }>,
) {
    let pid0 = program_id(0);
    let pid1 = program_id(1);
    let tile = x.load_like(z);

    cuda_tile_print!("Program ({}, {}): loaded tile\n", pid0, pid1);
    z.store(tile);
}
```

GPU printing is slow and serializes tile block execution. Use it for small grids and remove it before measuring performance.

`cuda_tile_assert!` checks conditions inside a kernel:

```rust
let n: i32 = x.shape()[0];
cuda_tile_assert!(n > 0, "expected a non-empty input");
```

## Host Readback

Host readback is a `DeviceOp`; execute it before reading the host vector:

```rust
let z_host: Vec<f32> = z
    .unpartition()
    .to_host_vec()
    .sync_on(&stream)?;

assert!(!z_host.iter().any(|x| x.is_nan()));
assert!(!z_host.iter().any(|x| x.is_infinite()));
```

If a fused kernel is wrong, split it into stages and read back each intermediate. Each stage should match a simple CPU implementation on a small input.

## Correctness Tests

Use minimal inputs first:

```rust
#[test]
fn small_add_matches_cpu() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![10.0, 20.0, 30.0, 40.0];
    let expected = vec![11.0, 22.0, 33.0, 44.0];

    let result = run_add_kernel(&a, &b);
    assert_eq!(result, expected);
}
```

Then compare larger random inputs against a CPU reference with an appropriate tolerance:

```rust
for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
    assert!((cpu - gpu).abs() < 1e-5, "CPU={cpu}, GPU={gpu}");
}
```

For numerically sensitive kernels, test edge cases: zeros, large positive values, large negative values, non-divisible shapes if supported, and known overflow-prone inputs.

## Inspecting Tile IR

`print_ir = true` prints the generated entry point wrapper and the Tile IR text during JIT compilation:

```rust
#[cutile::entry(print_ir = true)]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

`dump_mlir_dir` writes the compiled Tile IR text to files:

```rust
#[cutile::entry(dump_mlir_dir = "/tmp/cutile-ir")]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

Module-level dumps are also available with environment variables. They are
written to stderr once per compiled module:

| Variable | Description | Default |
|---|---|---|
| `CUTILE_DUMP` | Comma-separated stages to dump: `ir` (the Tile IR module text) and `bytecode` (alias `bc`; the encoded bytecode decoded back to text), or `all`. The `ast`, `resolved`, `typed`, and `instantiated` names are accepted but no code path emits them today | unset |
| `CUTILE_DUMP_FILTER` | Comma-separated `module::function` paths; the dumps are per module, so only the `module` part is matched. Bare function names do not exclude any module | unset |

## Errors and Crashes

Most cuTile Rust errors are caught before a kernel runs:

| Error | Cause | Fix |
|---|---|---|
| Shape mismatch | Incompatible tile shapes | Align shapes or use `reshape` / `broadcast` |
| Element type mismatch | Different element types in one operation | Add explicit `convert_tile()` |
| Invalid reduction axis | Axis outside the tile rank | Use an axis in `0..rank` |
| Unsupported MMA shape or dtype | No lowering for that combination | Use a supported shape and element type |
| Missing entry | Function is not marked with `#[cutile::entry()]` | Add the entry attribute |

Runtime errors usually come from out-of-bounds accesses, toolkit issues, or invalid raw-pointer usage:

| Error | Cause | Fix |
|---|---|---|
| CUDA error: no kernel image | Wrong GPU architecture or stale cubin | Clear cache, rebuild, verify target SM |
| Failed to load kernel | CUDA toolkit or driver issue | Check `nvidia-smi` and toolkit version |
| Out of memory | Tensor allocation or JIT memory pressure | Reduce allocation size or specialization count |
| Shape mismatch at runtime | Tensor size incompatible with partition | Ensure expected divisibility or bounds handling |

CPU segfaults usually mean the failure happened in host-side FFI, JIT compilation, or raw-pointer lifetime management rather than inside ordinary safe tile code. Get a backtrace first:

```bash
RUST_BACKTRACE=1 cargo run
RUST_BACKTRACE=full cargo run

gdb --args ./target/debug/my_program
(gdb) run
(gdb) bt
```

Check the CUDA driver, CUDA Toolkit path, raw pointer lifetimes, spawned task lifetimes, and host memory use during first-launch compilation.

## Debug Builds and Sanitizers

Device debug information follows Cargo's profile `debug` setting by
default. With the standard profiles, `cargo build` enables device-debug
mode and `cargo build --release` disables it. This is independent of
`debug-assertions`.

For example, turn device debugging off in a development build through the
workspace's `Cargo.toml`:

```toml
[profile.dev]
debug = false
```

Cargo provides only `DEBUG=true|false` to build
scripts, not the exact debug level. Any enabled level currently selects
device-debug mode, including `debug = "line-tables-only"` and `"limited"`.
It also selects device optimization level 0 unless explicitly overridden,
even in a release profile with `debug = true`. Automatic line-only mapping
is not implemented. Use the explicit `line` override below for optimized
profiling. See Cargo's [profile settings][cargo-profiles] and
[build-script environment][cargo-build-env].

`CUDA_RUST_DEBUG` overrides the Cargo-derived default when building the app.
No per-launch `.compile_options()` call is needed:

```bash
# Optimized device code with source lines for profiling.
CUDA_RUST_DEBUG=line cargo build --release

# Unoptimized device code with source locations and inline frames.
CUDA_RUST_DEBUG=full cargo build
cuda-gdb --args ./target/debug/my_program
```

The accepted values are `none`, `line`, and `full`. Unset follows Cargo;
other values fail the build. Cargo tracks changes, so no `cargo clean` is
needed. The setting is captured when the target `cutile-compiler` library
is built, not when the proc macro is built. A package-specific profile
override for `cutile-compiler` affects this shared default; an override
only for a kernel's crate does not. Setting the environment variable only
when running an already-built binary does not change device compilation.

`CompileOptions::new()` and `CompileOptions::default()` inherit this build
default. Use `debug_info()` to replace it for one launch. It sets both debug
flags, so it can also turn off a build-time `full` default:

```rust
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};

my_kernel(args)
    .compile_options(CompileOptions::new().debug_info(DebugInfoLevel::None))
    .sync()?;
```

Each option is part of the JIT cache key, so the modes do not share a cached
compiled kernel. The individual flags remain available, as does sanitizer
instrumentation:

```rust
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};

// cuda-gdb: debug information, no optimization.
my_kernel(args)
    .compile_options(CompileOptions::new().debug_info(DebugInfoLevel::Full))
    .sync()?;

// Profiler correlation: line-number information only, full optimization.
my_kernel(args)
    .compile_options(CompileOptions::new().debug_info(DebugInfoLevel::Line))
    .sync()?;

// Compute Sanitizer: memory-access instrumentation.
my_kernel(args).compile_options(CompileOptions::new().sanitize_memcheck(true)).sync()?;
```

- `debug_info(level)` replaces both debug flags. The individual
  `device_debug(bool)` and `lineinfo(bool)` setters change only their own
  flag, leaving other inherited settings in place.
- `device_debug(true)` passes `--device-debug` to the device compiler and
  implies optimization level 0. An explicit `opt_level` takes precedence,
  but `tileiras` currently rejects full debug information at optimized
  levels. The frontend also stops hoisting bounds checks out of loops.
  Checks it proved unnecessary or moved to launch time are unaffected;
  they never reach device code in any mode.
- `lineinfo(true)` passes `--lineinfo`: source-line correlation for Nsight Compute and Nsight Systems without changing code generation. This is the option for profiling optimized kernels.
- `sanitize_memcheck(true)` passes `--sanitize=memcheck` for `compute-sanitizer --tool memcheck`.
- `opt_level(n)` selects `--opt-level` directly; the default is 3.

Full debug mode describes source locations and inline frames, not Rust
variables or values. Tile IR currently supports control-flow debugging of
unoptimized code but not inspection of user variables; see the
[Tile IR debug-info documentation][tile-debug-info].

User helpers and methods retain their definition's source file and line
when inlined, including across modules. Core operations are attributed to
the user's call site. Multiple generated instructions can still map to one
Rust line; this is not a one-to-one mapping, especially with optimization.

## Profiling

Use NVIDIA **Nsight Compute** (`ncu`, reports opened with `ncu-ui`) for
individual kernels: throughput, occupancy, register use, and stalls.
**Nsight Systems** (`nsys`, `nsys-ui`) shows the application timeline:
CPU/GPU scheduling, transfers, synchronization, and launch gaps.

For source correlation, build with `CUDA_RUST_DEBUG=line`. Keep the Rust
sources available on the machine where you inspect the report. Profile the
built executable, not the Cargo build itself:

```bash
CUDA_RUST_DEBUG=line cargo build --release
ncu --target-processes all ./my_cutile_program
ncu --set full -o profile_report ./my_cutile_program
ncu-ui profile_report.ncu-rep
```

Replace `./my_cutile_program` with your binary's path, such as
`./target/release/my_program`. An `ERR_NVGPUCTRPERM` report means this
machine's GPU performance-counter access needs to be enabled by its
administrator; changing line-info settings will not fix that.

Use Nsight Systems for CPU/GPU scheduling:

```bash
nsys profile ./my_cutile_program
nsys-ui report.nsys-rep
```

Look for launch gaps, unnecessary synchronization, memory transfer overlap, and whether independent kernels actually overlap on separate streams.

Tile IR display is separate from Rust source-line correlation. NVIDIA lists
the Tile IR Source view as a new feature in [Nsight Compute 2026.3][ncu-tile-ir].
If it is missing, record the Nsight Compute and CUDA Toolkit versions,
compile options, cubin, and report before diagnosing a metadata problem.
The standalone Tile IR dumps above remain useful independently of that view.

## Regression Checks

The repository tests source provenance and compile-option cache separation
without a GPU:

```bash
cargo test -p cutile --test debug_info
bash scripts/test_debug_defaults.sh
```

The second command requires `jq`. It checks Cargo profiles, independent
debug assertions and build-dependency settings, explicit overrides, and
the line-only mapping limitation. It also checks that changing the runtime
environment does not change the built default. Additional checks are opt-in:

```bash
# Requires tileiras, nvdisasm, and readelf, but no GPU.
cargo test -p cutile --test debug_info cubin_has_rust_lines_and_inline_frames -- --ignored

# Requires a supported GPU and tileiras.
cargo test -p cutile --test debug_info kernel_executes_in_all_debug_modes -- --ignored

# Also requires cuda-gdb and timeout; stops at a helper's source line and
# checks its nested inline backtrace before completing the kernel.
cargo test -p cutile --test debug_info cuda_gdb_stops_in_cross_file_helper -- --ignored
```

[tile-debug-info]: https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html
[ncu-tile-ir]: https://developer.nvidia.com/nsight-compute-2026_3-new-features
[cargo-profiles]: https://doc.rust-lang.org/cargo/reference/profiles.html#debug
[cargo-build-env]: https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts

## Debugging Checklist

- Shapes match the operation and launch partition.
- Tensor sizes are compatible with the partition shape.
- Element types match or are explicitly converted.
- Small inputs match a CPU reference.
- Numerically sensitive code handles overflow and underflow.
- Raw pointers outlive all GPU work that uses them.
- `print_ir` shows the expected Tile IR operations.
- Profiles are captured after correctness checks pass.

---

Review [Performance](performance.md) for optimization strategies or [Interoperability](interoperability.md) for custom CUDA kernels.
