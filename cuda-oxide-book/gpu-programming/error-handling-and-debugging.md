# Error Handling and Debugging

GPU kernels fail differently from CPU code. The CUDA toolchain does not
support exceptions or stack unwinding today, there are no stack traces in
kernel output, and no `println!`. When something goes wrong, the result is
either silent data corruption, a hardware trap, or a cryptic driver error on the
host. This chapter covers cuda-oxide's tools for diagnosing and fixing kernel
problems.

## What happens when a kernel goes wrong

GPU errors fall into four categories:

| Failure mode          | What you see                                  | Example                                                |
|:----------------------|:----------------------------------------------|:-------------------------------------------------------|
| **Silent corruption** | Wrong results, no error                       | Race condition, off-by-one index                       |
| **Hardware trap**     | `CUDA_ERROR_ILLEGAL_INSTRUCTION` on sync      | `debug::trap()`, no-message `gpu_assert!`, panic, OOB  |
| **Device assertion**  | Assertion diagnostic plus `CUDA_ERROR_ASSERT` | `gpu_assert!(condition, "message")`                    |
| **Launch failure**    | `DriverError` returned immediately            | Wrong grid dims, missing module, out of resources      |

The CUDA toolchain does not expose an exception or stack-unwinding mechanism
today. A hardware trap terminates the kernel without structured assertion
metadata. CUDA's device-side `__assertfail` system call also terminates
execution, but reports the assertion message and call-site metadata before
synchronization returns `CUDA_ERROR_ASSERT`.

Both failures are asynchronous device errors and normally surface when the
stream or context is synchronized. Both also leave the CUDA context unusable,
but recovery differs: after a device assert (`CUDA_ERROR_ASSERT`), destroy and
recreate the context to keep using CUDA; after a hardware trap
(`CUDA_ERROR_ILLEGAL_INSTRUCTION`), CUDA documents the whole process as
poisoned, so terminate and relaunch it.

## `gpu_printf!` -- printing from the GPU

`gpu_printf!` lets you print values from device code for quick debugging. It
uses CUDA's built-in `vprintf` mechanism:

```rust
use cuda_device::{DisjointSlice, gpu_printf, kernel, thread};

#[kernel]
pub fn debug_kernel(data: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if idx.get() < 4 {
        gpu_printf!("Thread {} sees value {}\n", idx.get(), data[idx.get()]);
    }
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = data[idx.get()] * 2.0;
    }
}
```

### Important details

- **Flush requires sync.** Output is buffered on the GPU and only appears on
  the host after a stream or device synchronization (e.g., `to_host_vec` or
  `ctx.synchronize()`).
- **Buffer size.** The default printf buffer is 1 MiB. If many threads print,
  output may be truncated. Enlarge with
  `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size)`.
- **Thread ordering.** Output from different threads appears in arbitrary order.
- **Performance.** Printf serializes across threads -- avoid it in hot paths.
  Use it for debugging, not logging.
- **Format conversion.** The macro converts Rust `{}` format specifiers to C
  printf equivalents (`%d`, `%f`, etc.) at compile time.

### Why not `println!` or `Debug`?

Standard Rust formatting (`fmt::Display`, `fmt::Debug`, `format!`, `println!`)
requires dynamic dispatch, string allocation, and I/O -- none of which exist on
the GPU. `gpu_printf!` bypasses all of this by lowering directly to a CUDA
`vprintf` call.

## `gpu_assert!`, `__assertfail`, and `trap()`

For fatal error checking on the device, use `gpu_assert!` or `debug::trap()`:

```rust
use cuda_device::{DisjointSlice, debug, gpu_assert, kernel, thread};

#[kernel]
pub fn checked_kernel(data: &[f32], len: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();

    // No-message form: lowers to trap()
    gpu_assert!(idx.get() < len as usize);

    // String-literal message form: lowers to CUDA's __assertfail
    gpu_assert!(
        data[idx.get()] >= 0.0,
        "expected non-negative input"
    );

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = data[idx.get()];
    }
}
```

| Operation                           | Device behavior                         | Host effect                          |
| :---------------------------------- | :-------------------------------------- | :----------------------------------- |
| `gpu_assert!(condition)`            | Executes `trap()` if false              | `CUDA_ERROR_ILLEGAL_INSTRUCTION`     |
| `gpu_assert!(condition, "message")` | Calls CUDA's device-side `__assertfail` | Diagnostic plus `CUDA_ERROR_ASSERT`  |
| `debug::trap()`                     | Executes an unconditional trap          | `CUDA_ERROR_ILLEGAL_INSTRUCTION`     |
| `debug::breakpoint()`               | Emits `brkpt`                           | Pauses in cuda-gdb; fails without it |

The message passed to `gpu_assert!` must be a string literal. Formatted
assertion messages are not currently supported.

### The launch-and-synchronize pattern

Device failures are normally reported when the stream is synchronized:

```rust
// Launch kernel.
// SAFETY: config matches vecadd's 1D indexing and all buffer bounds.
unsafe { module.vecadd(&stream, config, &a, &b, &mut c) }
    .expect("Launch failed");

// Surface asynchronous traps or device assertions.
stream.synchronize().expect("Kernel failed on the device");
```

For `gpu_assert!(condition, "message")`, the CUDA driver prints the message,
source file, source line, and module context before synchronization returns
`CUDA_ERROR_ASSERT`.

The no-message form does not carry assertion metadata. Use the message form
when identifying the failing check matters, or use `gpu_printf!` when runtime
values also need to be inspected.

Ordinary Rust `assert!`, `debug_assert!`, and `panic!` paths are unchanged and
continue to use the existing trap-based behavior.

## Host-side error handling

### `DriverError`

The synchronous launch path returns
`Result<(), DriverError>`. The `DriverError` wraps a CUDA driver result code:

```rust
// SAFETY: config matches vecadd's 1D indexing and all buffer bounds.
match unsafe { module.vecadd(&stream, config, &a, &b, &mut c) } {
    Ok(()) => { /* launched successfully */ }
    Err(e) => eprintln!("Launch failed: {e}"),
}
```

### `DeviceError`

The async path (`{kernel}_async` / `DeviceOperation`) uses `DeviceError`,
which wraps driver errors alongside context and scheduling failures:

```rust
use cuda_async::simt::error::DeviceError;

let result: Result<Vec<f32>, DeviceError> = operation.sync();
```

`DeviceError` variants include `Driver`, `Context`, `KernelCache`, `Scheduling`,
`Launch`, and `Internal`.

### `CudaContext::check_err`

After a series of operations, call `check_err()` on the context to surface any
asynchronous errors that may have been recorded:

```rust
ctx.check_err().expect("Asynchronous GPU error detected");
```

## `cargo oxide debug` -- cuda-gdb integration

`cargo oxide debug` builds your kernel with debug info and launches cuda-gdb:

```bash
cargo oxide debug vecadd          # Standard GDB
cargo oxide debug vecadd --tui    # GDB with TUI
cargo oxide debug vecadd --cgdb   # cgdb front-end
```

By default this gives you **source-level debugging**: cuda-gdb can stop in
Rust source files and show a useful backtrace. Local-variable inspection is a
separate, heavier mode that you opt into when you need it.

### Debug info modes

cuda-oxide has three device debug modes:

| Mode | How to enable it | What you get | Cost |
|:-----|:-----------------|:-------------|:-----|
| Off | default for normal `build` / `run` | Fastest generated PTX, no source mapping | none |
| Line tables | `--lineinfo`, `cargo oxide debug`, or `CUDA_OXIDE_DEBUG=line-tables` | Source breakpoints, stepping, backtraces | low |
| Full | `--device-debug`, or `CUDA_OXIDE_DEBUG=full` | Line tables plus basic argument/local inspection | higher |

Think of line tables as a map from machine instructions back to source lines:

```text
PTX instruction  ──debug line table──>  src/main.rs:39
```

Full debug adds variable records:

```text
source local `tid`
      |
      v
LLVM/DWARF says: "tid lives in this stack slot"
      |
      v
cuda-gdb can try: print tid
```

For local variables, the debugger also needs the current instruction to be
inside the same lexical scope as the variable:

```text
function
  └─ if-let block
      └─ loop block
          └─ current instruction
```

If you stop too early, such as at kernel launch or at the first helper call,
the variable may honestly print as `<optimized out>` because it has not been
loaded into a register yet. For variable checks, prefer a source line after the
value is used:

```text
break src/main.rs:412
run
info args
info locals
```

Seeing one variable as `<optimized out>` is not automatically a compiler bug;
it can mean "this value has no live machine location at this exact PC." Debug
info is a map, not a time machine.

For inlined helper calls, cuda-oxide also keeps the original owner of each
argument. That matters because two different functions can both have an
argument numbered `1`:

```text
kernel(data) calls helper(self)

data -> arg #1 in kernel's debug scope
self -> arg #1 in helper's debug scope, with "inlined at" the kernel callsite
```

Without that scope split, LLVM treats the metadata as contradictory and drops
it. Debug info is fussy like that; it wants the family tree, not just the
surname.

Use line tables first. They are enough for most "where did execution go?"
questions, and they avoid the slower CUDA debug target mode. Use full debug
when you specifically want `print idx`, `print ptr`, or similar local-variable
inspection. Debuggers are allowed to be nosy; they are not always allowed to be
fast.

The `CUDA_OXIDE_DEBUG` override works with `build`, `run`, `pipeline`, and
`debug`:

```bash
CUDA_OXIDE_DEBUG=line-tables cargo oxide pipeline vecadd
CUDA_OXIDE_DEBUG=full cargo oxide debug vecadd
```

Each mode also has a flag, which is what the environment variable is a shorthand
for. `--lineinfo` selects line tables and `--device-debug` selects full debug,
matching `nvcc -lineinfo` and `nvcc -G`; both are accepted by `build`, `run`,
`pipeline`, `test`, `sanitize`, `inspect`, and `emit-ltoir`:

```bash
cargo oxide build vecadd --lineinfo
cargo oxide run vecadd --device-debug        # same as CUDA_OXIDE_DEBUG=full
```

Two rules settle what happens when they are combined. `--device-debug`
supersedes `--lineinfo`, so passing both gives full debug. And *omitting* both
does not mean "off": the flags export `CUDA_OXIDE_DEBUG` for the build only when
they ask for something, so an absent flag leaves a level the surrounding
environment already set alone instead of quietly opting out of it. A flag that is
present does export, and so wins over an inherited value.

Useful aliases:

| Value | Meaning |
|:------|:--------|
| `off`, `none`, `0` | no device debug metadata |
| `line-tables`, `line`, `lines`, `1` | source line tables only |
| `full`, `2` | line tables plus basic variable metadata |

### What full debug does to optimization

Full debug keeps supported locals in stable memory locations that cuda-gdb can
read.

```text
rustc MIR                         cuda-oxide / LLVM
------------------------------    ------------------
ScalarReplacement...: off    ->   mem2reg: off
SingleUseConsts: off               loop unrolling: off
ReferencePropagation: on           opt -O2: off
general inlining: on*              llc: -O0

* only DisjointSlice::get_mut is outlined
```

`ReferencePropagation` may replace a pointer assignment with an `AssignRef`
debug event. Full mode handles the supported slice-index form like this:

```text
ptr = &slice[index] -> AssignRef -> debugger stack slot -> cuda-gdb
```

A direct immutable reference into a fixed-size array uses the same bounded
stack-home bridge:

```text
&array[index]
    -> AssignRef
    -> debugger stack slot
    -> cuda-gdb
```

The multi-value `DIArgList` form can represent the address recipe in LLVM IR,
but it is not used for this producer because the resulting runtime-indexed
pointer location is not reliably inspectable after `ptxas`. The initial array
extension therefore accepts exactly one runtime `usize` index on a direct
fixed-size array and materializes the reconstructed reference only in full-debug
mode; surrounding field/deref chains remain unsupported.

Unsupported events omit that debug binding instead of emitting a partial
value. These stack-home stores exist only in full-debug builds. We do not
use `-Zmir-opt-level=0` because it exposes MIR forms the importer cannot yet
translate. A local that rustc removes without an equivalent debug event cannot
be recovered later in the pipeline.

You do not need to set `CUDA_OXIDE_NO_OPT=1`; full mode already disables the
downstream optimization stages shown above.

| Setting | Meaning |
| :------ | :------ |
| `CUDA_OXIDE_DEBUG=off` | no device debug metadata; fully optimized PTX |
| `CUDA_OXIDE_DEBUG=line-tables` | source lines only; still optimized |
| `CUDA_OXIDE_DEBUG=full` | source lines plus supported locals/args; selective MIR controls; downstream optimization off |

Line tables stay on the optimized pipeline because a line map survives
optimization well. Full mode keeps the selective MIR compatibility policy but
steps off cuda-oxide's later optimization stages so supported locals remain
materialized.

> The promotion-aware `mir.dbg_value` salvage that Pliron `mem2reg` performs is
> the building block for a future *optimized* debug tier (locals through
> `opt -O2`, best-effort). It is not what `full` uses today.

### What works today

Line-table mode supports:

- breakpoints by kernel name, e.g. `break vecadd`
- source stepping and backtraces
- helper/inlined source locations from other files, such as stepping from your
  kernel into `cuda-device/src/thread.rs`

Full mode (`-G`) supports inspecting:

- local variables and arguments rustc exposes through `var_debug_info`
- scalar types (`bool`, integers, floats), raw pointers, and references
- structs, tuples, and fixed-size arrays, with their fields shown at the
  correct (real-layout) offsets, e.g.
  `out = DisjointSlice {ptr: 0x..., len: 1}` and `idx = ThreadIndex {raw: 0}`

For device statics, keep cuda-gdb in Rust mode and use crate-qualified names:

```text
set language auto
print device_global::DEVICE_COUNTER
print constant_memory_simple::kernels::SCALE
```

```text
device_global::DEVICE_COUNTER          -> @global u64
constant_memory_simple::kernels::SCALE -> ConstantMemory<f32>, AS4
```

The constant-memory verifier also checks `DW_AT_address_class 4` in the cubin.

End-to-end behavior (breakpoint binds, backtrace, `info args`/`info locals`) is
checked on real hardware by `scripts/debug-smoketest.sh`.

Coverage is still partial. Unsupported cases include bare slice arguments
split into a `(ptr, len)` pair at the ABI boundary, repeated dereferences,
subslices, multiple runtime indices, runtime-index projections with surrounding
field/deref chains, dereference-plus-index/downcast chains, and non-field
composite fragments. Static struct/tuple fields, fixed-array constant indices,
enum payload downcasts, one thin-pointer/reference dereference, and one direct
runtime `usize` index used by a statement-debug reference into a fixed-size
array are supported. Locals in other inlined helper frames may still be sparser
than kernel locals; select the frame that owns the source binding.

### Breakpoint workflow

1. Build with debug: `cargo oxide debug <example>`
2. Set a breakpoint on your kernel: `break vecadd`
3. Run: `run`
4. Inspect threads: `cuda thread`, `cuda block`, `cuda warp`
5. Print variables: `print idx`, `print *c_elem`

`cargo oxide debug` selects the executable from Cargo's build artifact
metadata instead of assuming `target/release/<example>`. That keeps debugging
working with custom binary names, `package.default-run`, configured target
directories, host target triples, and virtual workspaces. Use `--bin <name>`
when a package has multiple binaries and no unambiguous default.

For programmatic breakpoints, use `debug::breakpoint()` in your kernel code.
When cuda-gdb hits the `brkpt` instruction, it pauses execution and lets you
inspect the GPU state.

:::{tip}
`debug::breakpoint()` will **crash** the kernel if no debugger is attached.
Guard it with a compile-time flag or only use it during debugging sessions.
:::

## `cargo oxide sanitize` -- Compute Sanitizer

For memory and synchronization correctness checks, run the same build path under
NVIDIA Compute Sanitizer:

```bash
cargo oxide sanitize vecadd
cargo oxide sanitize sharedmem --tool racecheck
cargo oxide sanitize debug --tool synccheck -- --kernel-name kns=clock
cargo oxide sanitize my_app -- --leak-check full -- --app-flag value
```

`memcheck` is the default tool. Use `racecheck` for shared-memory hazards,
`initcheck` for uninitialized global-memory reads, and `synccheck` for invalid
synchronization usage. Extra arguments after `--` are passed directly to
`compute-sanitizer` before the executable; use a second `--` to pass arguments
to the target program after the executable.

The command enables optimized device line tables by default so findings can
name Rust source files and lines. An explicit `CUDA_OXIDE_DEBUG` process or
project setting still wins.

Compute Sanitizer normally exits with status zero even when it reports a tool
finding. `cargo oxide sanitize` therefore supplies `--error-exitcode 86` by
default, making findings fail scripts and CI. An explicit sanitizer argument
overrides that default:

```bash
# Intentionally keep a zero exit status and inspect the printed report.
cargo oxide sanitize vecadd -- --error-exitcode 0
```

Options such as `--check-exit-code no` and `--require-cuda-init no` weaken what
a zero status proves. The wrapper therefore never declares the report clean
from process status alone; it reports completion and leaves the sanitizer
output visible for inspection.

The `--no-fmad` CLI flag is forwarded through both ordinary and interop builds.
It keeps ordinary multiply and add/subtract operations separate, with one
rounding per operation. Explicit fused operations such as `f32::mul_add`
remain fused.

NVVM IR and LTOIR builds also produce `.options`, `.kernels` and versioned
`.target` sidecars. Keep them with the artifact: later libNVVM and nvJitLink
steps preserve the same FMA policy, and check that the finalized image still
defines every kernel listed in `.kernels` (nvJitLink can report success while
dropping a module's kernels).

Run `memcheck` first when investigating memory safety. The other tools are
complementary and do not perform full memory-access checking.

## `cargo oxide doctor` -- environment validation

Before debugging kernel failures, verify your environment is correctly set up:

```bash
cargo oxide doctor
```

Doctor checks:

| Check           | What it verifies                               |
|:----------------|:-----------------------------------------------|
| Rust toolchain  | Nightly compiler with required components      |
| Codegen backend | `librustc_codegen_cuda.so` built               |
| CUDA headers    | `cuda.h` present under the toolkit root        |
| CUDA toolkit    | `nvcc` found and version compatible            |
| libNVVM         | `libnvvm.so` loadable (libdevice math kernels) |
| nvJitLink       | `libnvJitLink.so` loadable (same)              |
| libdevice       | `libdevice.10.bc` discoverable (same)          |
| LLVM            | `llc` (21+) available for PTX generation       |
| Driver / GPU    | `nvidia-smi` reports a GPU and its compute cap |
| cuda-gdb        | Optional; only needed for `cargo oxide debug`  |
| compute-sanitizer | Optional; only needed for `cargo oxide sanitize` |

The libNVVM / nvJitLink / libdevice checks fire only when a kernel calls
CUDA libdevice math (`sin`, `cos`, `exp`, `pow`, `sqrt`, ...). If your
kernel is pure arithmetic, those three failing is harmless. They all ship
with the CUDA Toolkit -- no separate download. If any check fails, doctor
prints the standard install location for that component.

Doctor itself needs neither the CUDA toolkit nor a driver, and it never
builds anything first, so it works on a machine where nothing is installed
yet. Two checks are informational rather than fatal: the codegen backend (a
missing `.so` just means "run `cargo oxide setup`"; `run`/`build` build it
on demand anyway) and the driver / GPU check (only `cargo oxide run` needs
a GPU; `build` and `pipeline` work without one).

## `cargo oxide inspect` -- show generated PTX

When you only need the final device assembly (not the intermediate dumps), use
`inspect`:

```bash
cargo oxide inspect vecadd
```

`inspect` builds the example the same way as `cargo oxide build`, then prints
the generated PTX. Prefer it for a quick "what did the backend emit?" check.
Use `pipeline` (below) when you need MIR / LLVM dialect / `.ll` stages as
well. `inspect` is strictly a PTX view: it exits with an error when a non-PTX
output mode is active in the environment (`CUDA_OXIDE_MATERIALIZE_CUBIN` or
`CUDA_OXIDE_EMIT_NVVM_IR`).

## `cargo oxide clean` -- remove local build outputs

`clean` deletes project-local Cargo `target/` directories and generated
cuda-oxide device artifacts (`.ptx`, `.ll`, `.opt.ll`, `.ltoir`, `.cubin`,
plus the `.target` / `.options` / `.kernels` / `.cubin.target` sidecars) under the current
workspace or standalone project. It refuses (with an error) to remove a
symlinked `target/` directory or artifact, and it does **not** wipe the
shared codegen backend cache at `~/.cargo/cuda-oxide/`.

```bash
cargo oxide clean
```

## `cargo oxide pipeline` -- inspecting the compilation

When a kernel produces wrong results but no errors, inspect the full
compilation pipeline to see exactly what code was generated:

```bash
cargo oxide pipeline vecadd
```

This prints the full pipeline output:

1. **MIR collection** -- which functions the collector found
2. **`dialect-mir`** -- pliron IR modelling Rust MIR (before and after `mem2reg`)
3. **LLVM dialect** -- pliron IR modelling LLVM IR, provided by `pliron-llvm` (after `mir-lower`)
4. **Textual LLVM IR** -- serialized `.ll` file
5. **Final PTX** -- the generated assembly

### Environment variables

For more targeted inspection:

| Variable                       | Effect                            |
|:-------------------------------|:----------------------------------|
| `CUDA_OXIDE_VERBOSE=1`         | Verbose compiler output           |
| `CUDA_OXIDE_SHOW_RUSTC_MIR=1`  | Dump the rustc MIR before import  |
| `CUDA_OXIDE_DEBUG=line-tables` | Emit source line-table metadata   |
| `CUDA_OXIDE_DEBUG=full`        | Emit full metadata for basic locals and args |

## Profiling with Nsight Compute

For performance debugging, NVIDIA's **Nsight Compute** (`ncu`) provides
roofline analysis, memory throughput, and occupancy metrics:

```bash
ncu --set full ./target/release/my_example
```

cuda-oxide kernels can emit profiler triggers using
`debug::prof_trigger::<N>()`, which generates a `pmevent` instruction that
Nsight Compute and Nsight Systems can capture for timeline annotation.

:::{seealso}
[Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
for the full profiling toolkit.
:::

## Common pitfalls

| Pitfall                          | Symptom                                    | Fix                                                              |
|:---------------------------------|:-------------------------------------------|:-----------------------------------------------------------------|
| Race condition on output buffer  | Wrong results, non-deterministic           | Use `DisjointSlice` plus matching prepared launch geometry      |
| Missing `sync_threads()`         | Stale shared memory reads                  | Add barrier between writes and reads                             |
| Wrong `shared_mem_bytes`         | `LAUNCH_OUT_OF_RESOURCES` or garbage data  | Match `LaunchConfig` to actual `DynamicSharedArray` usage        |
| Out-of-bounds with raw pointers  | Trap or silent corruption                  | Use `DisjointSlice::get_mut` for bounds checking                 |
| `panic!("message")` in kernel    | Thread traps, message never appears        | Expected: no panic runtime on GPU. `gpu_printf!` before the check to see values |
| Forgetting to sync after launch  | Host reads stale data                      | Call `to_host_vec`, `stream.synchronize()`, or `.sync()`         |
| PTX built for wrong arch         | `NO_BINARY_FOR_GPU`                        | Rebuild with `cargo oxide build --arch sm_XX`                    |

```{figure} images/debug-workflow.svg
:align: center
:width: 100%

Debugging decision tree: kernel problems fall into three branches (compile
error, runtime device failure, silent corruption), each with different
diagnostic tools. Common fixes are shown at the bottom.
```
