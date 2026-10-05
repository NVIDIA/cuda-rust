# 12. LLM Inference on DGX Spark

DGX Spark's unified memory can hold Qwen3-32B in 16-bit precision. We'll run it
with [Grout](https://github.com/huggingface/grout), an inference engine built on
cuTile Rust, starting with the smaller Qwen3-4B model.

The measurements below are from a DGX Spark (GB10, `sm_121`, 20 CPU cores, 128 GB of
unified LPDDR5X memory) running DGX OS with the 580 driver and CUDA 13.4.
They use cuTile Rust `v0.4.0` and Grout at commit
[`deb7427`](https://github.com/elibol/grout/commit/deb74274c731d4ca9e7f9b356a91892398498309).

---

## Prerequisites

The DGX OS image used here ships CUDA 13.0, which is too old for cuTile and
lacks the Tile IR assembler. Install a current toolkit alongside 13.0:

```bash
sudo apt-get install -y cuda-toolkit-13-4 cuda-tileiras-13-4
sudo apt-get install -y clang libclang-dev cmake build-essential pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
```

Then point the toolchain at the new toolkit (add these to `~/.bashrc`):

```bash
export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_TOOLKIT_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH
```

`CUDA_TOOLKIT_PATH` takes precedence over `CUDA_HOME`. Setting it also overrides
the `/usr/local/cuda-13` default in Grout's `.cargo/config.toml`.

Check that both CUDA tools report release 13.4 and Cargo uses nightly:

```bash
nvcc --version
tileiras --version
cargo --version
```

The 580 driver is a CUDA 13.0 driver. CUDA's minor-version compatibility lets 13.x
toolkits run on it, and the Tile IR assembler emits native `sm_121` code rather than PTX,
so nothing in this tutorial needs a newer driver.

---

## Layout

These commands keep the model directories and cuTile Rust checkout next to Grout:

```text
~/dev/
  cutile-rs/     # cuTile Rust checkout (Grout builds against it)
  grout/         # the inference engine
  hf_models/
    qwen3_4b/
    qwen3_32b/
```

```bash
mkdir -p ~/dev && cd ~/dev
git clone --branch v0.4.0 https://github.com/NVlabs/cutile-rs.git
git clone https://github.com/elibol/grout.git
git -C grout checkout deb74274c731d4ca9e7f9b356a91892398498309
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-4B  --local-dir hf_models/qwen3_4b    # about 8 GB
hf download Qwen/Qwen3-32B --local-dir hf_models/qwen3_32b   # about 66 GB
```

Grout's pinned commit depends on cuTile Rust 0.4.0. Its `[patch.crates-io]`
uses the sibling `cutile-rs` checkout.

Build Grout in release mode. On the Spark this takes well under a minute after the
dependencies are compiled once:

```bash
cd ~/dev/grout && cargo build --release
```

This builds the inference binary. The autotuner requires the `benchmarks`
feature; see [Autotuning](#autotuning) below.

---

## First run: Qwen3-4B

```bash
cargo run --release -- \
  --model ../hf_models/qwen3_4b \
  --prompt "Explain unified memory on DGX Spark in three sentences." \
  --max-new-tokens 200 --profile
```

These runs use Grout's built-in tuning defaults. The pinned revision ships
records for `sm_100` and `sm_120`; with no `benchmarks/tuning/sm_121/` directory,
Grout uses the defaults without printing a warning.

On the first run, cuTile compiles Grout's kernels from Rust through Tile IR to
`sm_121` machine code and caches them on disk. Later runs load the compiled
kernels from the cache.

| Qwen3-4B, 16-bit weights | First run | Warm run |
|---|---|---|
| Model load | 9.9 s | 10.0 s |
| Kernel compilation | 11.7 s (37 kernels) | 0 s (cache hits) |
| Decode | 29.7 tokens/s | 29.5 tokens/s |
| Peak memory in use | 15 GB | 15 GB |

Reading the weights from disk accounts for most of the model load time.

`--profile` prints prefill and decode step averages. `GROUT_PROFILE_OPS=1` adds
a per-operation table; `GROUT_PROFILE_SYNC_OPS=1` synchronizes after each operation
to include GPU execution time. The table covers prefill and warm-up. Decode
replays a CUDA graph, so its operations are not timed individually.

```bash
GROUT_PROFILE_OPS=1 GROUT_PROFILE_SYNC_OPS=1 cargo run --release -- \
  --model ../hf_models/qwen3_4b \
  --prompt "Explain unified memory on DGX Spark in three sentences." \
  --max-new-tokens 200 --profile
```

Use these settings to inspect individual operations, then unset them when
measuring throughput: synchronization changes execution timing.

---

(the-model-that-does-not-fit-a-gpu-qwen3-32b)=
## Qwen3-32B

Qwen3-32B has 32.8 billion parameters, about 66 GB in 16-bit precision, plus a KV cache
that grows with context length (roughly 9 GB at 32K tokens). Both fit in the
Spark's 128 GB of unified memory.

```bash
cargo run --release -- \
  --model ../hf_models/qwen3_32b \
  --prompt "What is 2 + 2? Then explain why in one paragraph." \
  --max-new-tokens 64 --max-seq-len 4096 --profile
```

| Qwen3-32B, 16-bit weights | Measured |
|---|---|
| Decode | 3.7 tokens/s |
| Weights in memory | 66 GB |

Both models use the same 37 compiled kernels. Qwen3-32B reads more weight data
for each generated token.

---

## Where the time goes

Decoding one token requires reading every weight once. At 66 GB per token, 3.7 tokens
per second is about 245 GB/s of sustained memory traffic; the Spark's LPDDR5X peaks near
273 GB/s. For the 4B model, 8 GB per token at 29.5 tokens per second is
about 236 GB/s. In both cases decode runs at roughly 90% of the memory roofline.

`GROUT_ATTN_BN_DECODE` sets Grout's decode attention tile size (default 32).
The following runs varied it manually:

| `GROUT_ATTN_BN_DECODE` | Qwen3-4B decode |
|---|---|
| 16 | 29.1 tokens/s |
| 32 | 29.5 tokens/s |
| 64 | 28.9 tokens/s |

Changing the tile size has little effect on decode: each token still requires
the same weight data. **Prefill** is compute-bound and responds to tile shapes.
Use a long prompt (thousands of tokens) and the per-kernel profiler to measure
the effect of tuning.

**Lower-precision weights** halve or quarter the bytes per token with 8-bit or
4-bit formats. The [NVFP4 tutorial](./11-nvfp4-inference.md) covers the kernels
for 4-bit inference.

Different tile sizes change the order of floating-point accumulation, which can
change the token selected by greedy decoding when logits are nearly tied. The
16 and 64 runs produced the same text as each other, but diverged from the
default after the second sentence.

### Autotuning

Build the autotuner from the Grout checkout:

```bash
cargo build --release --features benchmarks --bin grout_autotune
target/release/grout_autotune --help
```

For a prefill attention sweep, use `--site prefill_attention` and a
`--prompt-dir` containing `pp_512.txt`, `pp_2048.txt`, and `pp_8192.txt`.
Grout's `benchmarks/make_prompts.py` generates these files using the model's
tokenizer; it requires the Python `transformers` package.

The tuner writes records and trial logs under `benchmarks/tuning/sm_121/` by
default. `--budget-min` limits search time per prompt-length bucket. Rerunning
the same command resumes from the trial logs. If allocation recovery exits
with code 3, rerun it in a fresh process.

The measurements above use built-in defaults. Generate records on the Spark
and compare timings with the defaults before adopting them. Keep the
`sm_120` records in their own directory; those were tuned on an RTX 5090.

---

(notes-on-unified-memory)=
## Unified memory

The Spark's CPU and GPU share physical memory through a coherent address space
(ATS mode).

- **Host memory is device memory.** A pageable host buffer's address is valid on the GPU:
  wrapping a `Vec` with `Tensor::from_foreign` and launching a kernel on it works. The
  driver's pointer-attribute query does not classify such pointers, so code that
  validates device pointers that way needs another check on this platform.
- **Readback into fresh pageable memory is slow.** Copying a 256 MB tensor to a newly
  allocated host `Vec` ran at 0.13 GB/s, against 18.6 GB/s for the host-to-device
  direction and 24 GB/s for a host `memcpy`. The page faults on the destination dominate.
  Reuse or pre-touch destination buffers, or use pinned host memory, when readback is on
  a hot path.

---

## Key takeaways

| Concept | What it means on DGX Spark |
|---|---|
| **First launch** | Kernel compilation takes about 12 s on first use; later runs load from the disk cache |
| **Memory capacity** | 128 GB of unified memory holds a 32B model in 16-bit precision |
| **Decode roofline** | Memory bandwidth limits tokens per second; both models run near 90% of that limit |
| **Where tuning helps** | Prefill and other compute-bound phases; decode tile sizes have little effect in 16-bit |
| **Unified memory** | Host pointers work on the device; fresh pageable readback is the slow path |

---

## See also

- [NVFP4 inference](./11-nvfp4-inference.md) — kernels for lower-precision inference
- [Performance](../guide/performance.md) — kernel tuning
- [JIT Compilation](../guide/jit-compilation.md) — compilation and kernel caching
- [Interoperability](../guide/interoperability.md) — `Tensor::from_foreign` and externally owned memory
