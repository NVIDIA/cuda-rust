# cuTile Rust Examples

This crate contains runnable examples for the user-facing API and kernel DSL.

# Running Examples

- Start with `cargo run -p cutile-examples --example hello_world` to verify the toolchain.
- Run `cargo run -p cutile-examples --example saxpy` for a small kernel launch example.
- Run `cargo run -p cutile-examples --example nvfp4` for a CUDA 13.3 NVFP4 linear-tile example with output checking on native NVFP4 targets.
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
| `nvfp4`, `mxfp8` | Tile IR 13.3 and SM100+ (including SM120/121) |
| `pdl` | Tile IR 13.4 and SM90+; the driver must support the extended launch API |

The compiler's target rules also apply: SM90 requires Tile IR 13.3 and SM107
requires 13.4. These checks live in `src/requirements.rs`; they are not Cargo
features and do not change which APIs are compiled.

To check compatibility, select the actual older assembler:

```sh
CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.2/bin/tileiras scripts/run_examples.sh
CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras scripts/run_examples.sh
```

Run those commands from the repository root. The runner counts passes,
skips and failures separately and shows error output for failures. It enables
`reference-cpu,experimental-tune` by default so all examples can be built.
On aarch64, the CPU-reference dependency may also need
`RUSTFLAGS="-C target-cpu=native"`; see the comments in `Cargo.toml`.

An example that only uses baseline features must keep running on 13.2/13.3.
Do not raise its requirements to hide a compatibility regression.
