# Fused Pre-Attention QKV + Epilogue (`sm100_qkv_epilogue_cute`)

Before an LLM's attention layer can run, every incoming token row has to be projected into **Query (`Q`)**, **Key (`K`)**, and **Value (`V`)** vectors, normalized, quantized to **FP8 (`e4m3fn`)**, and filed into the **Paged KV Cache**.

Instead of launching 4 or 5 separate kernels (and round-tripping intermediate tensors through GPU HBM memory), this kernel fuses the entire pre-attention pipeline into **one persistent GPU kernel** on **NVIDIA B200 (`sm_100a`)**, matching **Python `CuTeDSL` 4.7.0** at **1:1 performance (`~1,400 TFLOP/s`)** and running **`2.5× to 4.8×` faster** than an exposed-epilogue schedule.

For a deeper step-by-step walkthrough, see [`visual.md`](visual.md).

---

## 1. Architecture: One Matrix Multiply, Three Epilogue Paths

![One Fused QKV GEMM and Its Three Epilogue Paths](svg/01-qkv-fused-architecture.svg)

Suppose **`M = 1,024` tokens** arrive at the attention block, and each token is a row of **`K = 4,096` FP16 numbers**:

1. **Stack the three weight matrices into one**:
   We glue `W_Q` (`4,096` columns), `W_K` (`1,024` columns), and `W_V` (`1,024` columns) side by side into a single matrix `W_qkv` of shape `[4096, 6144]`.
2. **Run one big matrix multiply (`X × W_qkv`)**:
   5th-Gen Tensor Cores (`tcgen05.mma`) multiply `[M, 4096] × [4096, 6144]` and leave the `[M, 6144]` FP32 result sitting inside on-chip **Tensor Memory (`TMEM`)** with zero intermediate HBM writes.
3. **Branch by column in the epilogue**:
   - **Columns `0 .. 4,096` (`Q`)**: Run `RMSNorm` + `FP8` quantize + write contiguously to `Q_out`.
   - **Columns `4,096 .. 5,120` (`K`)**: Run `RMSNorm` + `FP8` quantize + scatter into the **Paged KV Cache** (`k_cache[slot_mapping[m]]`).
   - **Columns `5,120 .. 6,144` (`V`)**: **Skip `RMSNorm`** (`V` is never normalized!) + `FP8` quantize + scatter into the **Paged KV Cache** (`v_cache[slot_mapping[m]]`).

---

## 2. Where Do "Q Heads" and "KV Heads" Come From?

![Where Attention Heads Come From: Slicing One Output Row into 128-Wide Chunks](svg/02-why-columns-are-heads.svg)

If you picture the GEMM as *"just concatenate `W_Q | W_K | W_V`, multiply by `X`, and slice the output into `Q`, `K`, and `V`"*, **you are 100% right about the matrix multiply**. During the GEMM, Tensor Cores have no idea what an attention head is: they just compute `6,144` output columns per token.

So why do **heads** (`32` Q heads and `8` KV heads, each `128` numbers wide) show up in the epilogue? Because of **how `RMSNorm` and the KV cache work** right after the GEMM:

1. **Why `Q` is `4,096` columns while `K` and `V` are `1,024` columns**:
   Modern LLMs use **Grouped-Query Attention (GQA)**. Every `128` consecutive columns is one **attention head**:
   - `Q` has **32 heads** × `128` numbers = `4,096` columns.
   - `K` and `V` each have **8 heads** × `128` numbers = `1,024` columns (every 4 `Q` heads share 1 `K` head and 1 `V` head).
2. **`RMSNorm` ("QK-Norm") happens per 128-column head, NOT across all 4,096 columns**:
   When a model normalizes `Q` and `K` before attention, it does **not** average all `4,096` `Q` columns together. Each `128`-number head is normalized **independently** against its own `128` numbers:
   $$\text{Q\_Head}_h[d] = \frac{\text{Q\_Head}_h[d]}{\sqrt{\frac{1}{128}\sum_{i=0}^{127} \text{Q\_Head}_h[i]^2 + \epsilon}} \cdot \gamma_Q[d]$$
3. **Why that is a huge win on the GPU**:
   If `RMSNorm` averaged across all `4,096` columns, different thread blocks would have to synchronize across the GPU just to compute one sum of squares. Because each head is only **`128` columns wide**, a single GPU thread loads all `128` numbers of one head from `TMEM` into its own registers and runs the entire `RMSNorm` locally with **zero cross-thread communication**.

---

## 3. How `cute-rs` & `CuTeDSL` Overlap the Epilogue (2-Stage `TMEM` Ping-Pong)

![Exposed Epilogue vs. Overlapped 2-Stage TMEM Ping-Pong on B200](svg/03-exposed-vs-overlapped-timeline.svg)

- **Why Epilogue Overlap Matters**: When a single accumulator buffer is reused sequentially (`[GEMM] -> [Wait] -> [Full Epilogue] -> [Next Tile]`), Tensor Cores sit idle while CUDA cores compute per-head `RMSNorm` reductions, `Q`/`K`/`V` branching, and indirect paged cache scatters (`slot_mapping[m]`).
- **2-Stage `TMEM` Ping-Pong in `cute-rs` & `CuTeDSL`**:
  We split each CTA's 512-column Tensor Memory into **two accumulator buffers (`TMEM_0` and `TMEM_1`)** and specialize the 6 warps (`192` threads) into three roles:

![Inside One 2-CTA Cluster: 6-Warp (192-Thread) Persistent Pipeline on SM100](svg/04-six-warp-cta-pipeline.svg)

As soon as Epilogue Warps `0..3` load their `128` numbers out of `TMEM_0` into registers (`TmemFragment::<128>::load`), they **immediately signal `acc_empty` to free `TMEM_0`** *before* running `RMSNorm`, `FP8` quantization, and global stores. `Warp 4` keeps firing `tcgen05.mma` back-to-back with zero stalls.

---

## 4. Run on B200 (`sm_100a`)

Run both **Python `CuTeDSL` 4.7.0** and **`cute-rs`** back-to-back with `1,000×` CUDA Graph replay (from `cuda-oxide/`):

```bash
CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run sm100_qkv_epilogue_cute --arch sm_100a
```

Run `cute-rs` standalone or pass custom shapes (`--m-list`, `--k`, `--q-heads`, `--kv-heads`):

```bash
CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run sm100_qkv_epilogue_cute --arch sm_100a -- --rust-only
```

---

## 5. Measured B200 (`sm_100a`) Results (`1,000×` CUDA Graph Replay, `Max Error = 0.000000`)

Shape: `K = 4,096`, `Q Heads = 32` (`N_q = 4,096`), `KV Heads = 8` (`N_k = N_v = 1,024`), `N_total = 6,144`, `Head Dim = 128`.

```text
Tokens |   CuTeDSL (2-TMEM) |   cute-rs (2-TMEM) | cute-rs vs. CuTeDSL | cute-rs TFLOP | Max Error
--------------------------------------------------------------------------------------------------
   128 |          16.384 us |          16.379 us |       1.00x CuTeDSL |       393.3 T |  0.000000
   256 |          16.384 us |          16.380 us |       1.00x CuTeDSL |       786.6 T |  0.000000
   512 |          26.621 us |          26.616 us |       1.00x CuTeDSL |       968.2 T |  0.000000
  1024 |          38.055 us |          36.856 us |       1.03x CuTeDSL |      1398.4 T |  0.000000
  2048 |          76.008 us |          73.735 us |       1.03x CuTeDSL |      1398.0 T |  0.000000
  4096 |         152.872 us |         146.859 us |       1.04x CuTeDSL |      1403.8 T |  0.000000
  8192 |         307.880 us |         295.335 us |       1.04x CuTeDSL |      1396.1 T |  0.000000
```
