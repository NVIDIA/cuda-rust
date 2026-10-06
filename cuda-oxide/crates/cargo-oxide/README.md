# cargo-oxide

Cargo subcommand for building and running Rust GPU programs with cuda-oxide.

Replaces the previous `xtask` pattern with a proper cargo subcommand that works both inside the cuda-oxide repo (for developers) and externally (for users who `cargo install`).

## Installation

**Internal developers** (inside the cuda-oxide repo): no installation needed. The workspace alias makes `cargo oxide` work immediately.

**External users**:

Install with the project's pinned nightly toolchain:

```bash
cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend if it's not already available.

## Usage

```bash
cargo oxide new my_project          # scaffold a new cuda-oxide project
cargo oxide new my_project --async  # scaffold with async template (tokio + cuda-async)
cargo oxide run vecadd              # build + run an example
cargo oxide build vecadd            # compile only (no run)
cargo oxide build vecadd --debug-assertions  # keep release optimization, compile debug assertions
cargo oxide build vecadd --materialize-cubin --arch sm_120  # embed native cubin
cargo oxide emit-ltoir vecadd --arch sm_100  # device code -> .ltoir (Tile/SIMT interop)
cargo oxide build -- -p my_app      # arbitrary cargo build through cuda-oxide
cargo oxide test                    # cargo test with cuda-oxide defaults
cargo oxide test -- -p my_app       # arbitrary cargo test through cuda-oxide
cargo oxide pipeline vecadd         # verbose pipeline dump
                                    # (MIR -> dialect-mir -> LLVM dialect -> LLVM IR -> PTX)
cargo oxide lean-export             # inspect scalar MIR and its Lean translation
cargo oxide inspect vecadd          # build + print generated PTX
cargo oxide sanitize vecadd         # build + run under NVIDIA Compute Sanitizer
cargo oxide debug vecadd --tui      # build + launch cuda-gdb
cargo oxide fmt                     # format all crates
cargo oxide fmt --check             # check formatting
cargo oxide list                    # list examples and requirements
cargo oxide list --json             # stable machine-readable output
cargo oxide clean                   # remove local build outputs / generated artifacts
cargo oxide doctor                  # validate environment
cargo oxide setup                   # explicitly build the codegen backend
cargo oxide update                  # refresh cached backend (external projects)
cargo oxide update --force          # inside the workspace, run setup via update
```

### Flags

| Flag                         | Applies to                       | Description                                     |
|------------------------------|----------------------------------|-------------------------------------------------|
| `--materialize-cubin`        | run, sanitize, build, test, pipeline, debug | Finalize and embed target-specific native GPU code during the host build |
| `--emit-nvvm-ir`             | run, build, pipeline             | Generate NVVM IR for libNVVM                    |
| `--arch <sm_XX>`             | run, sanitize, build, test, pipeline, emit-ltoir, inspect, debug | Target architecture override |
| `--features <F>`             | run, sanitize, debug, build, build passthrough, emit-ltoir, inspect | Comma-separated cargo features to enable |
| `--device-features <F>`      | run, build                       | Comma-separated features for metadata-declared device crates only |
| `--bin <NAME>`               | run, sanitize, debug                | Specific binary target to build/run      |
| `--tool <T>`                 | sanitize                         | Compute Sanitizer tool: `memcheck`, `racecheck`, `initcheck`, or `synccheck` |
| `-o, --output <P>`           | emit-ltoir                       | Output path for the `.ltoir` artifact           |
| `--cargo-target-dir <PATH>`  | build/test passthrough           | Cargo target directory                          |
| `--device-codegen-crate <LIST>` | build/test passthrough        | Comma-separated device owner crate filter       |
| `--device-cfg <NAME>`        | build/test passthrough           | Append `--cfg NAME` to rustflags                |
| `-v, --verbose`              | run, sanitize, build, test, emit-ltoir, inspect | Show detailed compilation output           |
| `--no-fmad`                  | run, sanitize, build, test, emit-ltoir, pipeline, inspect | Keep ordinary multiply and add/subtract operations separate |
| `--unchecked-indexing`       | run, sanitize, build, test, emit-ltoir, pipeline, inspect | Elide device slice/array bounds checks (UB on OOB) |
| `--lineinfo`                 | run, sanitize, build, test, emit-ltoir, pipeline, inspect | Emit device line-number info for profilers (nvcc `-lineinfo`) |
| `--device-debug`             | run, sanitize, build, test, emit-ltoir, pipeline, inspect | Emit full device debug info; disables libNVVM optimization (nvcc `-G`) |
| `--debug-assertions`         | build                            | Compile `debug_assert!` / `cfg(debug_assertions)` with release-like optimization and overflow checks disabled |
| `--force`                    | update                           | Inside the workspace, run `setup` instead of advising it |
| `--async`                    | new                              | Use the async template                          |
| `--cgdb`                     | debug                            | Use cgdb instead of cuda-gdb                    |
| `--tui`                      | debug                            | Use GDB's TUI interface                         |
| `--check`                    | fmt                              | Check formatting only                           |
| `--json`                     | list                             | Emit stable machine-readable output             |

`--arch` is required for `emit-ltoir` and explicit NVVM IR output because those
artifacts are architecture-specific. Without an override, `run` detects the
local GPU, while `build` and `pipeline` use the compiler's feature-based target
so they remain useful for cross-compilation.

`--no-fmad` (or `CUDA_OXIDE_NO_FMA=1`) disables implicit contraction of an
ordinary multiply followed by an add or subtract, so the two operations round
separately. Explicit fused operations such as `f32::mul_add` remain fused.
For NVVM IR and LTOIR, cuda-oxide records the policy in matching `.options`
and versioned `.target` files and passes `-fma=0` through libNVVM and
nvJitLink. Keep both sidecars with the artifact when another build system
consumes it.

`--lineinfo` and `--device-debug` select the device debug policy, mirroring nvcc's
`-lineinfo` and `-G`. `--lineinfo` preserves source line mappings with optimization
intact and reaches nvJitLink as `-lineinfo`; `--device-debug` emits full debug
information and additionally passes `-g` with `-opt=0` to libNVVM, so device
optimization is disabled. Asking for both yields full debug, because full debug
already carries line tables.

Both are equivalent to `CUDA_OXIDE_DEBUG=line` and `CUDA_OXIDE_DEBUG=full`, and an
explicit flag outranks that variable. Omitting the flags exports nothing rather than
`off`, so an absent flag never cancels a debug level the environment or
`cuda-oxide.toml` already requested. Because the policy changes what the CUDA compiler
stages are asked to do, it participates in the codegen fingerprint: switching it
rebuilds device code instead of reusing artifacts compiled under a different policy.

`--materialize-cubin` moves the remaining libNVVM and nvJitLink work into the
host build. The executable embeds a native cubin, so deployment does not need
libNVVM, nvJitLink, or a first-load compilation step. This trades portability
for startup simplicity: the cubin is pinned to the requested architecture and
cannot use cuda-oxide's load-time PTX bridge on a newer GPU.

```bash
cargo oxide build vecadd --materialize-cubin --arch sm_120
```

A concrete architecture and a build-host CUDA Toolkit (libNVVM, nvJitLink, and
libdevice) are required. cargo-oxide fingerprints the exact compiler inputs and
rechecks them inside the backend; use the wrapper flag instead of setting
`CUDA_OXIDE_MATERIALIZE_CUBIN` around raw Cargo. That variable is an internal
wrapper/backend signal, not a supported user interface: raw Cargo may reuse a
previous artifact without running the backend, and a backend invocation without
the wrapper's fingerprint is rejected. Generic `#[cuda_module]` kernels and
`#[device] extern` declarations are rejected because their run-time
multi-artifact linking is not yet represented by this single-cubin path.
Metadata-interop projects keep rejecting this global flag; they can instead
request one independently finalized cubin per declared device crate with
`artifact-kind = "cubin"`, as described below. Materialization is also a
final-output mode, so it cannot be combined with `--emit-nvvm-ir` or
`emit-ltoir`.

## Commands

### `cargo oxide list [--json]`

Lists every example bundled with the cuda-oxide workspace, including its short
description and any hardware or toolkit requirements documented in its README.

```bash
cargo oxide list
cargo oxide list --json
```

### `cargo oxide run <example>`

Builds the codegen backend, compiles the example with the custom backend, and runs it. This is the primary command for day-to-day development.

When neither `--arch` nor `CUDA_OXIDE_TARGET` is set, `run` detects the
compute capability of CUDA device 0 and targets that architecture so the
generated PTX can load on the local GPU. Use `--arch <sm_XXX>` or
`CUDA_OXIDE_TARGET=<sm_XXX>` to override this for a specific device or
cross-target workflow.

```bash
cargo oxide run vecadd
cargo oxide run gemm_sol
cargo oxide run device_ffi_test --emit-nvvm-ir --arch sm_120
cargo oxide run cutile_inter_kernel
```

Interop examples can declare extra cuda-oxide device crates with
`[[package.metadata.cuda-oxide.device-crates]]`, plus optional
`[package.metadata.cuda-oxide.interop]` metadata. `cargo oxide run` builds those
device crates with `rustc-codegen-cuda`, writes each configured artifact, and
then builds/runs the host crate normally. PTX remains the default;
`cutile_inter_kernel` uses this path:
the host crate is a cutile-rs program, while `simt/` is a cuda-oxide SIMT PTX
crate loaded by the host at runtime.

Device crates that need a deployment-time native artifact can opt into the
same libNVVM and nvJitLink finalizer used by cuda-oxide's embedded
materialization path. When a device package defines multiple binary targets,
select one explicitly with `bin`:

```toml
[[package.metadata.cuda-oxide.device-crates]]
manifest-path = "device/Cargo.toml"
artifact-dir = "device"
artifact-name = "simt_device"
artifact-kind = "cubin"
source-identity = true
bin = "simt-device"
```

`artifact-kind` accepts `ptx` (the default) or `cubin`. Cubin export requires a
concrete target from `--arch`, `CUDA_OXIDE_TARGET`, project configuration, or
the device detected by `run`; that requirement is checked before the device
build. The cubin itself is always finalized for the target the backend records
in the emitted `<name>.target` sidecar, which can differ from the request hint
when the kernel needs a newer architecture than the detected device. When
`source-identity` is true, cargo-oxide also writes `<artifact>.identity`
containing the built target (read back from the `.target` sidecar, or from the
PTX's own `.target` directive for `ptx` artifacts), the normalized
`--device-features` selection, the artifact digest, and Cargo depfile-derived
source digests. Paths in that sidecar are relative to the artifact directory.

Use `--device-features` when the nested device crate and host crate have
different feature sets:

```bash
cargo oxide build interop-app --arch sm_120a \
  --features host-integration \
  --device-features tensor-cores
```

`inspect` accepts only PTX interop artifacts; use the declared cubin file
directly for native artifact inspection.

`cargo oxide` validates `bin`, builds only that target with `cargo build
--bin`, and uses the selected binary name for the artifact unless
`artifact-name` is set. Omitting `bin` preserves the package-default build
behavior.

### `cargo oxide sanitize <example>`

Builds an example like `cargo oxide run`, then executes the produced release
binary under NVIDIA Compute Sanitizer. `memcheck` is the default tool; use
`--tool racecheck`, `--tool initcheck`, or `--tool synccheck` for shared-memory
hazard, uninitialized global-memory, or synchronization checks.

```bash
cargo oxide sanitize vecadd --arch sm_75
cargo oxide sanitize sharedmem --tool racecheck
cargo oxide sanitize debug --tool synccheck -- --kernel-name kns=sync
cargo oxide sanitize vecadd -- --leak-check full
cargo oxide sanitize my_app -- --leak-check full -- --app-flag value
```

Arguments after the first `--` are passed to `compute-sanitizer` before the
executable. A second `--` splits target program arguments, matching
`compute-sanitizer [options] [your-program] [your-program-options]`.

`sanitize` uses Cargo's reported executable artifact, so custom binary names,
workspace layouts, configured target directories, and host target triples do
not require path guessing. Device line tables are enabled by default while
normal optimization remains on.

Compute Sanitizer's own error exit code defaults to zero. This command supplies
`--error-exitcode 86` unless you pass an explicit value, so tool findings fail
scripts and CI. If you intentionally pass `--error-exitcode 0`, inspect the
printed report; a zero process status no longer implies that the report was
clean. Options such as `--check-exit-code no` and `--require-cuda-init no` also
weaken what a zero status proves. The wrapper reports completion and reminds
you to inspect the sanitizer output instead of declaring the report clean from
status alone.
NVIDIA recommends treating the tools as complementary: `racecheck`,
`initcheck`, and `synccheck` do not replace `memcheck` memory-access checking.

### `cargo oxide build <example>`

Same as `run` but stops after compilation. Useful for examples that require hardware you don't have (e.g., Blackwell tensor cores).

```bash
cargo oxide build htens          # compiles PTX, doesn't try to run on GPU
cargo oxide build tcgen05        # sm_100a only, but PTX generation works anywhere
cargo oxide build debug --debug-assertions
```

`--debug-assertions` keeps the normal release-like `-Copt-level=3` policy but
switches to `-Cdebug-assertions=on`. It also explicitly sets
`-Coverflow-checks=off`: rustc otherwise enables overflow checks with debug
assertions, which adds distinct MIR checks and trap paths. The option applies
to regular builds, passthrough builds, and metadata-declared interop device
crates. It is independent of `--device-debug`, which controls emitted device
debug information and CUDA finalization optimization.

`build` also has a passthrough mode for normal Cargo workspaces. Put the Cargo
arguments after `--`; cargo-oxide supplies the backend, target architecture,
configured environment, and optional device owner filters.

```bash
cargo oxide build --                           # plain `cargo build`
cargo oxide build --debug-assertions -- -p my_app
cargo oxide build --arch sm_86 -- -p my_app --bin app --release
cargo oxide build --cargo-target-dir target/cuda -- -p my_app --release
cargo oxide build --device-codegen-crate gpu-kernels,math_gpu -- -p my_app
```

The owner filter is matched against rustc crate names, so Cargo package hyphens
are normalized to underscores (`gpu-kernels` matches `gpu_kernels`). The CUDA
backend still compiles host code for every crate through LLVM; it emits device
artifacts only for the listed owner crates.

The filter controls compilation, not runtime module names. Code in an excluded
crate or target must not call that target's generated `load()` function. In a
package with several targets, embedded bundles still use the package name, so
an excluded target's loader could otherwise find a selected sibling target's
bundle.

Passthrough builds use two Cargo cache boundaries. The exact backend `.so`
identity is global because that backend compiles every crate. CUDA procedural
macros track target, output mode, FMA policy, owner list, materializer
provenance, and configured codegen environment only in crates that can own or
instantiate device code. Changing those settings therefore regenerates device
artifacts without recompiling unrelated host-only dependencies.

### `cargo oxide emit-ltoir <crate>`

Compiles a crate's device code to a binary LTOIR artifact in one step, for the
Tile-to-SIMT interop workflow ([#96](https://github.com/NVlabs/cuda-oxide/issues/96)):
cuda-oxide is the SIMT participant, producing LTOIR that a tile or CUDA C++ kernel
links against. It builds the crate in NVVM IR mode, then runs the emitted
`<crate>.ll` through libNVVM `-gen-lto`, writing `<crate>.ltoir`.

`--arch` is required, since LTOIR is architecture-specific. It accepts `sm_XX`,
`compute_XX`, or a bare `XX`. The default output path is `<crate-dir>/<crate>.ltoir`;
`-o/--output` overrides it.

```bash
cargo oxide emit-ltoir standalone_device_fn --arch sm_100
cargo oxide emit-ltoir my_simt_crate --arch sm_120 -o build/simt.ltoir
```

cuda-oxide chooses the NVVM IR format from the selected architecture.
Pre-Blackwell targets use LLVM 7 typed pointers; `compute_100` and newer targets
use modern opaque pointers. The target is checked before export and recorded
alongside the artifact so the runtime uses the same format.

### `cargo oxide test`

Runs `cargo test` through the cuda-oxide backend. With no extra arguments it
runs the workspace's normal test selection. Put Cargo test arguments after
`--`, including the test-binary separator when needed.

```bash
cargo oxide test
cargo oxide test -- -p my_app --release --test gpu_smoke -- --nocapture
cargo oxide test --device-codegen-crate gpu_smoke -- -p my_app --test gpu_smoke
```

### `cargo oxide pipeline <example>`

Shows verbose progress plus the selected IR artifacts: MIR collection,
`dialect-mir` before and after `mem2reg`, LLVM dialect, LLVM IR, and final PTX.

```bash
cargo oxide pipeline vecadd
cargo oxide pipeline device_ffi_test --emit-nvvm-ir --arch sm_120
```

### `cargo oxide lean-export [input.mir --function name]`

Shows the input MIR and generated Lean together. With no arguments, it uses the
bundled, hand-authored scalar indexing demo:

```bash
cargo oxide lean-export
cargo oxide lean-export path/to/helper.mir --function helper
```

Run from the `cuda-oxide/` workspace directory in a source checkout. The command builds and runs the Rust
`lean-exporter`; it does not build the CUDA backend, require a GPU, or run Lean.
The current subset supports unsigned scalar constants and wrapping arithmetic.
Unsupported input fails with an error. This is translation inspection, not a
kernel verification result. See the [exporter README](../lean-exporter/README.md)
and [Lean workflow](../../lean/README.md).

### `cargo oxide inspect [example]`

Builds the example or project and prints the generated PTX to stdout. Where `pipeline` shows every stage verbosely, this prints just the finished device code, which makes it the quick way to diff codegen across a change. Takes `--arch`, `--features`, `--no-fmad`, `--lineinfo`, `--device-debug`, `--unchecked-indexing` and `-v`. The example name is required inside the workspace and optional for a standalone project.

### `cargo oxide debug <example>`

Builds with debug info (`-C debuginfo=2`) and launches cuda-gdb. Supports
`--tui` for GDB's TUI mode and `--cgdb` for the cgdb frontend.

`debug` uses Cargo's reported executable artifact, so custom binary names,
`package.default-run`, configured target directories, host target triples, and
virtual workspaces do not require path guessing. Use `--bin <name>` when a
package has multiple binaries and no unambiguous default.

### `cargo oxide new <name> [--async]`

Scaffolds a new standalone cuda-oxide project with `Cargo.toml`, `rust-toolchain.toml`, and a working `src/main.rs` containing a vector addition kernel. The default template uses `#[cuda_module]` with typed synchronous launch methods; `--async` generates a template with `tokio`, `cuda-async`, and typed lazy `DeviceOperation` launches.

```bash
cargo oxide new my_kernel
cd my_kernel
cargo oxide run
```

### `cargo oxide clean`

Removes project-local build outputs and the generated cuda-oxide artifacts beside them. This is the `cargo clean` equivalent that also knows about the device-side files the backend writes, so a stale `.ptx`, `.ll` or embedded artifact cannot survive into the next build.

### `cargo oxide fmt [--check]`

Formats the SIMT workspace, `rustc-codegen-cuda`, and all examples; in the
merged repository it also formats the parent shared-host workspace. With
`--check`, reports files that need formatting without modifying them.

### `cargo oxide doctor`

Validates that your environment is correctly set up: Rust nightly toolchain,
CUDA headers (`cuda.h`), CUDA toolkit (`nvcc`, libNVVM, nvJitLink,
libdevice), LLVM (`llc`), clang/libclang, the NVIDIA driver / GPU, and the
codegen backend `.so`. Optional probes cover `cuda-gdb` (for `debug`) and
`compute-sanitizer` (for `sanitize`). Every check reports what was found or
how to fix it.

`cargo-oxide` itself builds and runs without the CUDA toolkit and without an
NVIDIA driver, and `doctor` never builds anything first, so it works on a
bare machine and tells you exactly what is missing. The driver / GPU check is
informational (only `cargo oxide run` needs a GPU), and a missing backend
`.so` just points at `cargo oxide setup` (`run`/`build` build it on demand
anyway).

### `cargo oxide setup`

Explicitly builds (or rebuilds) the codegen backend. Normally this happens automatically on every `run`/`build`/`pipeline` command, but `setup` is useful after pulling new changes or for CI.

### `cargo oxide update [--force]`

Rebuilds the shared codegen backend cache used by projects outside this
repository (`~/.cargo/cuda-oxide/`) from the commit the project's cuda-oxide
dependency resolves to (see [Backend Discovery](#backend-discovery)). You
rarely need it: a changed dependency rebuilds the cache on the next build by
itself. Inside the cuda-oxide workspace the local source tree is
authoritative, so the default path prints how to run `cargo oxide setup`;
pass `--force` to run setup from this command.

```bash
cargo oxide update
cargo oxide update --force
```

## Backend Discovery

When `cargo oxide` needs the `librustc_codegen_cuda.so` backend, it searches in this order:

1. **`CUDA_OXIDE_BACKEND` env var**: explicit path override
2. **Project config**: `.cargo/cuda-oxide.toml`
3. **Local repo**: detects `cuda-oxide/crates/rustc-codegen-cuda` (or the historical flat `crates/rustc-codegen-cuda`) relative to workspace root, builds from source
4. **Cached `.so`**: `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`, when it was built from the commit the project's cuda-oxide dependency resolves to
5. **Build from the dependency**: builds the codegen crate out of the checkout Cargo already made for the project's `cuda-device` / `cuda-host` dependency, and caches it

Outside the repository the backend follows the project's `Cargo.lock`:

```text
Cargo.lock   cuda-device = git+https://github.com/NVlabs/cuda-oxide.git#a1b4f118...
                 │
                 ▼
~/.cargo/git/checkouts/cuda-oxide-*/a1b4f11/        Cargo's checkout of that commit
                 │  crates/rustc-codegen-cuda  →  cargo build
                 ▼
~/.cargo/cuda-oxide/librustc_codegen_cuda.so         + source-rev.txt = a1b4f118...
```

The kernels compile against the crates at that commit and the backend that
lowers them is built from the same commit, so the two cannot drift. Changing
the dependency (`cargo update`, a new `rev = ...`) rebuilds the cache on the
next build. The checkout's `rust-toolchain.toml` names the nightly that commit
needs; if the project pins a different one, the build stops before compiling
and tells you which channel to set. A path dependency on a local checkout
builds in place, like step 3. Projects with no cuda-oxide dependency fall back
to a one-time clone of `main`. The backend's build tree lives at
`~/.cargo/cuda-oxide/target` (shared by every commit's builds; delete it to
reclaim the space).

Project config can also provide the default architecture, extra rustc flags,
and child-process environment:

```toml
backend = "/path/to/librustc_codegen_cuda.so"
default-arch = "sm_86"
extra-rustflags = ["--cfg", "my_device_cfg"]

[env]
MY_BUILD_FLAG = "1"
```

`cargo oxide doctor` validates this file when present: malformed TOML,
wrong-shaped fields, or a `default-arch` the CUDA arch parser rejects fail
the check (doctor exits non-zero but still completes the full scan), while
ignored `[env]` keys such as `RUSTFLAGS` / `CARGO_ENCODED_RUSTFLAGS` and
non-`sm_XX` `default-arch` spellings (`compute_90`, bare `90`) are reported
as warnings. Build commands remain strict and refuse to run with an invalid
config.
Relative backend paths are resolved from the `.cargo` directory containing the
config file. Each `extra-rustflags` array element remains one rustc argument,
including values that contain spaces.

Configuration values are defaults. Precedence is:

1. explicit `cargo oxide` flags and internal artifact paths;
2. environment variables inherited by `cargo oxide`;
3. `.cargo/cuda-oxide.toml` defaults.

Do not put `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS` in the `[env]` table; use
`extra-rustflags` for project defaults. cargo-oxide combines configured flags,
inherited user flags, explicit `--device-cfg` values, and its required compiler
flags in boundary-preserving `CARGO_ENCODED_RUSTFLAGS`. Required compiler flags
are applied last so inherited settings cannot replace the cuda-oxide backend or
disable its correctness-critical codegen options.

## Architecture

```text
cuda-oxide/crates/cargo-oxide/
├── Cargo.toml
└── src/
    ├── main.rs       # CLI definitions (clap) + dispatch
    ├── backend.rs    # Backend discovery + build logic
    ├── artifact_identity.rs  # Embedded artifact identity/provenance
    └── commands/     # Command implementations, one module per command group
```

## Future Commands

| Command                         | Description                                             |
|---------------------------------|---------------------------------------------------------|
| `cargo oxide bench <example>`   | GPU profiling (nsys/ncu integration), report TFLOPS     |
