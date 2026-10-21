# Single-token attention decode with CuTe-RS on SM120a (RTX 5090)

For the current matched CUDA-event and Nsight kernel timings, see
[Nsight measurements](nsight-measurements.md). Benchmark results below retain
their original historical protocol.

This cuda-oxide example has its own decode kernels. It does **not** call the
prefill kernel or calculate unused query rows.

- **Input:** one query per head and an existing contiguous FP16 KV cache.
  Defaults: `B=1, Sq=1, Sk=1024, H=8, D=64`, softmax scale `0.125`.
- **Split:** each 128-thread block reads 64 keys and values using typed CuTe
  asynchronous copies. Four lanes cooperate on each query/key dot product.
  The block computes stable softmax and a partial weighted value sum in FP32.
- **Combine:** a second kernel uses one warp per head to combine the partial
  sums, correcting for each split's different softmax maximum, then writes FP16.
- **Tails:** any positive KV length is supported, including 1, 65 and 1,025.
  Out-of-range keys have zero weight. The final split is never empty.
- **Scope:** ordinary multi-head attention, batch one, head dimension 64, on
  SM120. The supplied cache contains only visible tokens; cache allocation,
  appending, paging, grouped-query attention and token sampling are outside
  this example. Prefill and decode share only input/oracle utilities.

The schedule exposes `H * ceil(Sk / 64)` blocks, versus `H` for the old
prefill-at-one-query path. It uses scalar FP32 arithmetic for the single-query
matrix-vector products, with no tensor-core query padding and no TCGen05.
This is a dedicated decode schedule, not a claim that tuning is complete.

## Rust shape specialization

The kernels use Rust const generics: `fmha_decode_partial::<SEQ_K, HEADS>`
and `fmha_decode_merge::<SEQ_K, HEADS>`. The compiler can fold the KV stride,
split count, address calculations and merge-loop bounds for known shapes.
Both kernels share the same shape parameters; static tensor/scratch bounds
are checked at compile time and host inputs are checked before launch.
There is one algorithm implementation, not separately maintained kernels.

| Mode | Behavior in both Rust and CuTeDSL |
| --- | --- |
| `--specialization auto` (default) | Uses a compiled shape when available; otherwise runtime dimensions |
| `--specialization static` | Requires one of the compiled shapes below |
| `--specialization runtime` | Keeps cache length, head count, KV stride and split count dynamic |

The example compiles `(Sk,H) = (64,8), (65,3), (1024,8), (4097,8)` ahead of
time. Other positive cache lengths still work through the runtime path. More
static shapes can be instantiated by extending the host dispatch; this is
an AOT example, not a runtime compiler for every possible cache length.
The internal `<0,0>` instance represents the runtime path. The CuTeDSL
reference offers the same modes and shape set for matched comparisons.

All ten entry points reside in one embedded CUBIN. The runner loads that
artifact explicitly and binds it to the typed module API; the generic
module loader used by PTX examples expects merged PTX bundles instead.

## Build and run

The [example README](../README.md) shows the normal command, which now compares
both implementations with matched direct CUDA events. The direct binary
commands here use `--rust-only` to retain the standalone reports used by this
profiling workflow.

From `cuda-oxide/`, using the CLI/backend built from this branch and the
CUTLASS compiler 4.7.0:

```bash
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir
export CUDA_OXIDE_MLIR_PROFILE=cutlass-full-cute-mlir22
cargo oxide build --arch sm_120a --device-codegen-crate fmha_decode_sm120a_cute \
  --cargo-target-dir /tmp/cuda-oxide-fmha-decode -- \
  --manifest-path cute-rs-preview/examples/fmha_decode_sm120a_cute/Cargo.toml --release
CUDA_VISIBLE_DEVICES=1 /tmp/cuda-oxide-fmha-decode/release/fmha_decode_sm120a_cute \
  --rust-only --seq-k 1024 --heads 8 --json /tmp/decode-rust.json
```

From this example's directory, with the Python environment described in
[reference/README.md](README.md):

```bash
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a python reference/launch.py \
  --seq-k 1024 --heads 8 --json /tmp/decode-cutedsl.json
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a python reference/validate.py \
  --binary /path/to/fmha_decode_sm120a_cute --reference --output-dir /tmp/decode-validation
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a python reference/compare.py \
  --rust-binary /path/to/fmha_decode_sm120a_cute --python /path/to/python \
  --seq-k 1024 --heads 8 --specialization static --output-dir /tmp/decode-comparison
```

Each implementation checks **every** output against independent FP64 stable
softmax, with `abs(actual-expected) <= 2e-4 + 2e-3 * abs(expected)`. Both output
and partial buffers are poisoned before correctness launches; a second launch
must reproduce the output bit for bit. `--zero-query` checks the cache mean;
`--input-scale 16` exercises large logits. `--iters 0` disables timing.

## What is compared

[reference/attention_decode.py](attention_decode.py) is an independently
authored CuTeDSL implementation of the **same** split-KV algorithm. It is not a
vendored or official CUTLASS decode kernel. Both implementations use CUTLASS
4.7.0 and the same data, tile shape, split count, arithmetic precision and
reduction order and specialization mode. This comparison checks the Rust and
Python implementations and compiler paths;
it does not establish parity with a separately tuned upstream decode kernel.

The comparison matches the algorithm and workload, but does not isolate
compiler effects. CuTeDSL interleaves K/V copy groups and preloads Q before
waiting for shared copies; Rust stages K then V and reads Q after the wait.
Tail zero-fill mechanisms also differ. These scheduling differences are part
of the measured implementations, alongside the compiler differences.

CUDA events enclose both kernels. The Nsight comparison reports both the
**sum of their active durations** and the **elapsed span from the first kernel's
start to the second kernel's end**. These differ because the span includes the
inter-kernel gap. Compilation, allocation, host verification and warmup are
excluded. Buffers are reused; this is a warm-cache microbenchmark, not a full
model-generation measurement.

Old 17.791/15.328 µs reports are retained only in
[historical-prefill-baseline/](../historical-prefill-baseline/README.md).
They describe the replaced prefill implementation and do not validate this one.


## Earlier verified results on RTX 5090

On 2026-10-12, `B=1, Sq=1, Sk=1024, H=8, D=64`, using three alternating
rounds, 100 warmups and 31 samples per round:

| Specialization | Rust kernel-active sum | CuTeDSL kernel-active sum | Rust sequence span | CuTeDSL sequence span |
| --- | ---: | ---: | ---: | ---: |
| Runtime on both sides | 2.912 µs | 3.040 µs | 3.168 µs | 3.392 µs |
| Static on both sides | 2.400 µs | 2.176 µs | 2.688 µs | 2.624 µs |

Specialization reduces Rust's kernel-active time by 17.6%. A 10.3% gap
remains against the matching specialized CuTeDSL kernel-active sum. The
sequence span includes the launch gap; these metrics must not be mixed.
See [benchmark-results.json](benchmark-results.json) for all samples,
compiler/source hashes and exact kernel identities.

All 52 validation cases passed for **both** implementations, with 52 output
cross-checks and 12 additional runtime-versus-static comparisons. All 16
Compute Sanitizer runs passed: memcheck, racecheck, synccheck and global/shared
initcheck, for both implementations in runtime and static tail cases. The
host suite has 13 passing tests; the Python suite has 28. Evidence is in
[validation-results.json](validation-results.json).


The example/package was renamed to include the tested `sm_120a` target.
Existing benchmark and validation JSON records preserve the original package
names, paths and hashes from their measurement runs; they are historical
provenance, not build commands for the renamed package.

## Compiler optimization regression check

The prefill compiler follow-up also rebuilt decode and reran 52 Rust numerical
cases, six runtime/static cross-checks, and eight sanitizer checks; all passed.
Fresh three-round Nsight Systems comparisons retain Rust active times of
2.912 µs runtime and 2.400 µs static. The reference measured 3.040 / 2.208 µs;
its static value differs from the earlier run by 0.032 µs, so the changed
percentage gap is not a Rust decode speedup. See
[compiler-regression-results.json](compiler-regression-results.json) for the
latest samples, validation, commands and build hashes. The short example
README uses these latest measurements; the earlier records above are preserved.
