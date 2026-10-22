# Contracts and validation

[Back to the example](../README.md).

## Arithmetic contract

Standalone cuda-oxide correctness example for the bias-free FFN core
`down(silu(gate(x)) * up(x))`. This composes three BF16 tensor-core projections
and one activation kernel. FP32 projection accumulators round to BF16; SiLU
rounds to BF16 before multiplying the BF16 up projection, and the product rounds
again. RMSNorm and residual addition belong to the surrounding decoder layer.

## Storage and execution

The host-only `cute_rs::ffn` API carries gate/up/down roles and input/output
widths in Rust types. Weights borrow existing `DeviceBuffer<BFloat16>` allocations
with `[output,input]` orientation. Dynamic shapes validate row strides, storage
capacity, arithmetic overflow and addressable byte extents. Owned workspace
buffers keep mutable intermediates distinct; input and weight borrows remain
live through execution. Role tags cannot identify incorrectly loaded weights.

The example's synchronous runner checks buffer contexts and launch geometry,
waits for prior work in the owning context, then queues all four kernels on one
stream and waits for completion, including enqueue-error paths. The initial
context wait makes pending writes on other streams safe without adding a new
readiness-tracking framework. It is intentionally conservative for throughput.
Zero rows perform no launches.

Each full warp computes a 16×8 output tile using typed `Bf16A`/`Bf16B` operands,
FP32 accumulators and K16 MMA steps. Predicated global loads handle row, column
and K tails and padded strides. Warp participation remains an explicit unsafe
contract. Activation uses existing typed 128-bit copies for complete tiles and
scalar copies for tails. This first schedule has no shared-memory pipeline.

## Host and GPU checks

From `cuda-oxide/`:

```bash
cargo test --offline -p cute-rs --features host --all-targets
cargo test --offline -p cute-rs --features host --doc
cargo test --offline --manifest-path cute-rs-preview/examples/ffn_cute/Cargo.toml
CARGO_NET_OFFLINE=true CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run ffn_cute --arch sm_120a
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  cute-rs-preview/examples/ffn_cute/target/release/ffn_cute
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 \
  cute-rs-preview/examples/ffn_cute/target/release/ffn_cute
```

## Acceptance criteria and scope

Synthetic validation covers hidden/intermediate widths 1024/3072 and 2048/6144
at one and three rows, plus 19/23, 32/17 and 1/1 cases, empty rows, and a
17-row tile tail. Asymmetric data and NaN padding expose weight-orientation and
stride errors. Every intermediate and final output is compared with an
independent FP64 reference with integer BF16 rounding; output guards must stay
untouched. The down projection is also checked on its actual GPU input to
separate projection errors from propagated activation error.

For these bounded finite fixtures, gate/up/SiLU/product use
`atol=1e-6, rtol=0.008`; down uses `atol=1e-5, rtol=0.008`; the complete
composition uses `atol=2e-5, rtol=0.02`. A separate exact fixture requires product
bits `0x3fbc`, rejecting the incorrect cast-once result `0x3fbd`. A delayed
producer stream regression checks that safe asynchronous input writes complete
before the runner reads them. Host tests cover shape and grid rejection, and
compile-fail tests cover dtype, weight-role, dimension and borrow contracts.

Validation runs on local RTX 5090 / SM120 using the pinned CUTLASS compiler.
This is synthetic FFN correctness evidence; checkpoint inference, model-level
performance, and execution on other architectures remain separate acceptance
requirements.
