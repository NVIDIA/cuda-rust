# cuTile Rust Examples

This crate contains runnable examples for the user-facing API and kernel DSL.

## Running examples

Run these commands from the `cutile-rs/` workspace, not the monorepo root.
Reusable kernel implementations live in [`cutile-kernels`](../cutile-kernels/README.md);
the QKV and MoE examples import those implementations and supply host-side
validation and timing code.

- Start with `cargo run -p cutile-examples --example hello_world` to verify the toolchain.
- Run `cargo run -p cutile-examples --example saxpy` for a small kernel launch example.
- Run `cargo run -p cutile-examples --example nvfp4` for a CUDA 13.3 NVFP4 linear-tile example with output checking on native NVFP4 targets.
- Run `cargo run --release -p cutile-examples --example nvfp4_moe` for an NVFP4 mixture of experts benchmark with device routing, activation quantization, fused FC1/ReLU-squared/requantization, FC2, and weighted finalization. It checks outputs and reports CUDA graph latency per batch size.
- Run `cargo run --release -p cutile-examples --example fused_qkv` for fused QKV projection, Q/K RMSNorm, FP8 conversion and paged KV stores. Add `-- --bench` for the opt-in graph timing sweep; see the [QKV guide](fused_qkv.md) for layouts, validation scope and tuning options.
- Run `cargo run -p cutile-examples --example pdl` for a CUDA 13.4 programmatic dependent launch example: a producer/consumer kernel pair with output checking and a stream-order vs PDL timing comparison (sm_90 or newer).
- Run `cargo run -p cutile-examples --example async_gemm` for a larger async example.
- Run `cargo run --release -p cutile-examples --example autotune --features experimental-tune` for RMSNorm tuning with CPU compilation overlapped with GPU evaluation. It checks correctness and prepared-binary reuse; `-- --workers 4` increases preparation concurrency and `-- --serial` disables overlap for comparison. See the [autotuning guide](../cutile-book/guide/autotuning.md) for deadlines and finalist-confirmation requirements.

## Toolkit and GPU requirements

Every example checks the selected `tileiras` and device before allocating
buffers or launching kernels. An unmet requirement prints `SKIP <example>:`
with the required and selected capabilities, then exits successfully.
Tool discovery, compilation, execution and validation errors still fail.

| Examples | Minimum requirements |
| --- | --- |
| All except those listed below | Tile IR 13.2 and a supported SM80+ target |
| `nvfp4`, `nvfp4_moe`, `mxfp8` | Tile IR 13.3 and SM100+ (including SM120/121) |
| `fused_qkv` | FP8-capable SM90+; SM90 requires Tile IR 13.3, while SM100/120 support 13.2 |
| `pdl` | Tile IR 13.4 and SM90+; the driver must support the extended launch API |

The compiler's target rules also apply: SM90 requires Tile IR 13.3 and SM107
requires 13.4. These checks live in `src/requirements.rs`; they are not Cargo
features and do not change which APIs are compiled.

`nvfp4_moe` measures the full pipeline's CUDA graph replay latency for batch
sizes 1, 2, 4, 8, 16, 32, 64, 128, and 256. Each batch uses 20 samples of
1,000 replays, with 50 warmups per sample. Compilation, allocation, transfers,
and output validation are outside the timed region. The original default
fixture remains H=I=1024, eight allocated experts, four active experts, top-k=4.
Direct launches and graph replay are checked against a deterministic reference.
The example imports [shared kernels](../cutile-kernels/src/experimental/nvfp4_moe.rs).

### B200 reference profiles

The opt-in B200 profiles provide SM100 schedules for evaluation with CUDA 13.5
tooling. Compiler support for individual schedules can vary; selecting this
profile does not guarantee that every schedule is supported by every assembler
build. The minimum execution requirements above remain unchanged for default
profiles.

Select matching public toolkit tools and libraries explicitly, and remove
inherited overrides from other toolkit installations:

```sh
export CUDA_TOOLKIT_PATH=/absolute/path/to/public/cuda-13.5
export CUDA_HOME="$CUDA_TOOLKIT_PATH"
export CUDA_PATH="$CUDA_TOOLKIT_PATH"
export CUDA_ROOT="$CUDA_TOOLKIT_PATH"
export CUDACXX="$CUDA_TOOLKIT_PATH/bin/nvcc"
export CUTILE_TILEIRAS_PATH="$CUDA_TOOLKIT_PATH/bin/tileiras"
export CUTILE_BYTECODE_VERSION=13.5
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/nvvm/lib64:$CUDA_TOOLKIT_PATH/lib64:$CUDA_TOOLKIT_PATH/lib"
export LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64:$CUDA_TOOLKIT_PATH/lib"
unset LD_PRELOAD
export LIBCLANG_PATH=/absolute/path/to/libclang/directory

nvcc --version
"$CUTILE_TILEIRAS_PATH" --version
"$CUTILE_TILEIRAS_PATH" --list-versions
sha256sum "$CUTILE_TILEIRAS_PATH"
```

Preserve the repository's Rust toolchain pin.
Record actual backend-library loading during compilation (for example with
`LD_DEBUG=libs`), not just the version-command output.

From `cutile-rs/`:

```sh
cargo run --release --locked -p cutile-examples --example fused_qkv -- --bench --profile b200
cargo run --release --locked -p cutile-examples --example fused_qkv -- --bench --profile b200 --nontrivial
cargo run --release --locked -p cutile-examples --example nvfp4_moe -- --profile b200
cargo run --release --locked -p cutile-examples --example nvfp4_moe -- --profile b200 --shape large
```

`--profile b200` is explicit and requires SM100. Omitting it preserves the
original schedules; it is never selected based on a compiler filename.
MoE uses bounded persistent FC1/FC2 grids and per-batch tiles, occupancy,
latency and finalization settings. It still processes every device-computed
expert group; launch bounds do not assume four active experts.

`--shape large` selects H=6144, I=1536 and 192 allocated experts without a source
edit. Both shapes retain the same synthetic generator, top-k=4 and four active
experts. Label the shapes separately when reporting results. MoE prints every
selected schedule and all 20 `sample_csv,batch,sample,latency_us` records.
Timing still covers routing, quantization, FC1/ReLU-squared/requantization, FC2,
and weighted finalization.

For focused correctness/sanitizer runs, use `--batch 16 --check-only`, with
either shape or profile. This validates direct launches and graph replay
without timing. It does not suppress checks or change tolerances.

### Validation

An assembler may reject individual B200 schedules, including TN=32. Record the
selected schedule and compiler diagnostics when comparing compilers; silently
substituting another profile changes the comparison. `--profile default` retains
the original launch settings.

Run correctness and sanitizer checks separately from timing:

```sh
cargo build --release --locked -p cutile-examples --example fused_qkv --example nvfp4_moe
CUTILE_JIT_SANITIZE=memcheck "$CUDA_TOOLKIT_PATH/bin/compute-sanitizer" \
  --tool memcheck --error-exitcode 99 target/release/examples/fused_qkv
CUTILE_JIT_SANITIZE=memcheck "$CUDA_TOOLKIT_PATH/bin/compute-sanitizer" \
  --tool memcheck --error-exitcode 99 target/release/examples/nvfp4_moe \
  --shape large --batch 16 --check-only
```

QKV keeps its original unpadded M=67 tail/scatter fixture. Its masked slot load
also uses in-bounds addresses for inactive rows. Keep output checks enabled for
both direct launches and graph replay.

### Known SM120 limitation

Separately, on RTX 5090 (`sm_120`, driver 610.43.02), public `tileiras` 13.5.37
fails the original MoE direct-launch check at batch 16: normalized maximum error
0.5, maximum absolute error 1.578125. This reproduces with 13.4 and 13.5 bytecode.
Its cause remains unknown. This native numerical failure did not reproduce
in the B200 evaluations.
Retain all output checks.

## Testing older assemblers

To check compatibility, select the actual older assembler:

```sh
CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.2/bin/tileiras scripts/run_examples.sh
CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras scripts/run_examples.sh
```

Run those commands from `cutile-rs/`. The runner counts passes,
skips and failures separately and shows error output for failures. It enables
`reference-cpu,experimental-tune` by default so all examples can be built.
On aarch64, the CPU-reference dependency may also need
`RUSTFLAGS="-C target-cpu=native"`; see the comments in `Cargo.toml`.

An example that only uses baseline features must keep running on 13.2/13.3.
Do not raise its requirements to hide a compatibility regression.
