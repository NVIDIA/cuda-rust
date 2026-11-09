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
1,000 replays, with 50 warmup replays per sample. CUDA events time GPU execution;
compilation, allocation, and output validation are outside the timed region.
The synthetic workload uses H=I=1024, eight allocated experts, four active
experts, and top-k=4, with direct launches and graph replay checked against a
deterministic reference. Performance will be optimized in a future version.

The example uses the reusable kernels in
[`cutile-kernels::experimental::nvfp4_moe`](../cutile-kernels/src/experimental/nvfp4_moe.rs).
**Peak performance requires CUDA Toolkit 13.5 and its Tile IR assembler.** The
minimum execution requirements above are unchanged; older assemblers can run
the kernels without the latest compiler optimizations.

### NVFP4 MoE validation gap

On an RTX 5090 (`sm_120`, driver 610.43.02), the public `tileiras` 13.5.37
fails the example's direct-launch numerical check at batch 16: normalized
maximum error 0.5, maximum absolute error 1.578125. This reproduces with both
13.4 and 13.5 bytecode, so changing the bytecode version is not a workaround.
The cause has not been isolated. B200 correctness and performance evaluation
remain pending; retain the output checks when evaluating this example.

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
