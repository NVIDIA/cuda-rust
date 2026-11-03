# Rotary positions, KV cache, and BF16 attention

This cuda-oxide example adds position information to queries and keys, saves
earlier keys and values in a cache, and reuses them for attention. It starts
with projected, normalized queries and keys, unrotated values, and explicit
token positions:

```text
query → rotary pairs ───────────────────────────────┐
key   → rotary pairs → append to owned key cache ───┼→ attention → output
value ───────────────→ append to owned value cache ┘
```

Rotary embeddings pair the first half of each head with the second half.
`StridedTensor` loads each pair into a register tile; both products round to
BF16 before their sum. Standard RoPE and three-axis interleaved MRoPE share
this kernel, using different host-generated coefficient tables.

## The CuTe flow

`AttentionShape` checks query/KV head grouping, and `CacheShape` checks append
capacity. The owned cache stores `[token, head, coordinate]` values and exposes
only its completed prefix to attention. Appends and attention calls wait for
GPU completion before returning.

For attention, CuTe cooperative copies stage 64-key tiles in shared memory.
Typed layouts assign dot-product and output coordinates to threads. A second
kernel merges the split results. Query-key scores, softmax probabilities, and
weighted values are computed in FP32; only the output rounds to BF16. Attention uses 128-wide heads and supports
causal or noncausal windows, including grouped-query attention.

## Run it

From `cuda-oxide/`, with the pinned toolchain, CUDA toolkit, the CUTLASS compiler
(`cargo oxide toolchain install cutlass`) and an RTX 5090:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run rope_cache_cute --arch sm_120a
```

A successful run prints `PASS RoPE/cache` and `PASS BF16 attention` cases,
followed by reference comparison counts. It checks whole and chunked appends,
cache guards, exact rotary rounding, and prefill versus incremental attention.

See [contracts and validation](reference/validation.md) for MRoPE axis mapping,
cache readiness, numerical tolerances, sanitizer commands, and recorded GPU
evidence. Host trigonometry has not been matched bit-for-bit with CUDA/PyTorch;
checkpoint inference and model performance are not established by this example.
