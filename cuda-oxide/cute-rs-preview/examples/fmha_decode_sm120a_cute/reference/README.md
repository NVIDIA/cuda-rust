# Dedicated CuTeDSL decode reference

`attention_decode.py` is authored in this repository using the public CUTLASS
CuTeDSL API. It is **not an upstream CUTLASS kernel**. It implements the same
one-query split-KV schedule as the Rust example: 64 cache rows per 128-thread
CTA, FP32 dot products and stable softmax, FP32 split states, and a second
32-thread-per-head merge kernel producing FP16 output.

`launch.py` owns the decode configuration, allocations and both launches.
It imports only deterministic fixture and CPU-oracle helpers from the prefill
reference. It does not import or call the prefill GPU kernel. Cache lengths
need not be multiples of 64; invalid keys are masked out of softmax.

`--specialization auto|runtime|static` matches the Rust dispatch. Auto specializes
`(Sk,H) = (64,8), (65,3), (1024,8), (4097,8)` and otherwise keeps dimensions
runtime. Static mode rejects other shapes; runtime mode also keeps the KV
stride and merge bounds dynamic. The comparison requires the same effective
mode on both sides, so neither compiler receives an extra shape advantage.

Use `nvidia-cutlass-dsl[cu13]==4.7.0` with a compatible CUDA-enabled PyTorch
build and driver. The shared [`requirements.txt`](../../../requirements.txt)
pins the DSL; dependencies retain their own licenses. All code in this
directory has the indicated first-party Apache-2.0 license.

Prepare the shared Python environment once for both prefill and decode,
from `cuda-oxide/`:

```bash
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

Select a CUDA-enabled PyTorch build that supports SM120 and matches your CUDA
runtime. The normal `cargo oxide run` command finds the shared environment
automatically; no activation or Python environment-variable export is needed.
Python selection is `--python PATH` after `--`, `CUDA_OXIDE_CUTE_PYTHON`, the
example's `.venv/bin/python`, the shared `cute-rs-preview/.venv/bin/python`, then `python3`.
See the [example README](../README.md) for direct-event comparisons.
The main command with `--all-cases --warmup 100 --iters 31` prints runtime
and static direct CUDA-event medians for the selected `(Sk,H)` shape. Decode
has only noncausal single-token attention, so there are two rows. This mode
requires one of the static shapes above and overrides `--specialization`.
Each event interval covers both kernels, their gap and possible host launch
submission delays. It differs from Nsight's sum of kernel-active durations
and from the span between the first kernel's start and the last kernel's end.
Comparing those metrics does not establish a regression.

`compare.py` retains the separate Nsight Systems profiling workflow and
invokes the Rust binary with `--rust-only`. With Nsight Systems installed,
reproduce the archived runtime/static measurement protocol from `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build --arch sm_120a --device-codegen-crate fmha_decode_sm120a_cute \
  --cargo-target-dir /tmp/cuda-oxide-fmha-decode -- \
  --manifest-path cute-rs-preview/examples/fmha_decode_sm120a_cute/Cargo.toml --release

for specialization in runtime static; do
  CUTE_DSL_ARCH=sm_120a cute-rs-preview/.venv/bin/python \
    cute-rs-preview/examples/fmha_decode_sm120a_cute/reference/compare.py \
    --rust-binary /tmp/cuda-oxide-fmha-decode/release/fmha_decode_sm120a_cute \
    --seq-k 1024 --heads 8 --specialization "$specialization" --noncausal \
    --warmup 100 --samples 31 --rounds 3 \
    --output-dir "/tmp/decode-nsys-$specialization"
done
```

The harness reports the median of round medians for active sum and sequence
span separately and retains raw evidence in fresh directories. Hardware state,
clocks and software versions can affect new results; reproducing the protocol
does not guarantee the archived numbers.

From this directory:

```bash
CUTE_DSL_ARCH=sm_120a ../../../.venv/bin/python launch.py \
  --seq-k 65 --heads 3 --warmup 0 --iters 0
../../../.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

`benchmark.py` accepts the same options. Its event measurements include the
whole partial-plus-merge sequence. `compare.py` runs both implementations
serially under Nsight Systems and distinguishes summed kernel-active time
from the elapsed two-kernel span. `validate.py` covers short, ragged and long
caches, multiple heads, uniform attention and large logits.

Each implementation checks every output against an independent FP64 stable
softmax with `abs(actual-expected) <= 2e-4 + 2e-3 * abs(expected)`, poisons
the output and partial buffers before correctness launches, and requires a
second launch to reproduce the output bit for bit. From the example directory,
with the shared environment's interpreter as `python`:

```bash
CUTE_DSL_ARCH=sm_120a python reference/validate.py \
  --binary /path/to/fmha_decode_sm120a_cute --reference --output-dir /tmp/decode-validation
```
