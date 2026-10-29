# Contracts and validation

[Back to the example](../README.md).

## Geometry and rotary arithmetic

This cuda-oxide example starts transformer integration with already projected
and normalized Q/K, unrotated V, and explicit positions. It uses `cute-rs` for
BF16 storage and checked attention geometry, and `ctm-rs` for kernel interfaces
and instruction primitives. It computes causal/noncausal attention over the
completed cache prefix; it does not yet run a model.

`AttentionShape<QH,KVH,D>` keeps query heads, KV heads and head width independent
of hidden width. `CacheShape<KVH,D>` checks capacity and append token ranges.
The example's `KvCache<KVH,D>` privately owns K/V allocations; `RotatedQuery<QH,D>`
retains query dimensions. `HostTables<D>` prevents mixing different rotary head
widths. Dynamic input lengths, grouped heads, byte extents, contexts, positions
and launch sizes are checked before execution.

The runner rotates split-half pairs: coordinate `i` pairs with `i+D/2`, not
with its adjacent coordinate. Both BF16 products round before their BF16 sum.
The predictor uses ordinary RoPE; Talker interleaved MRoPE uses `[24,20,20]`
sections at D=128. Height frequencies occupy indices 1,4,…,58 and width
frequencies occupy 2,5,…,59; all remaining frequencies use temporal positions.
Three-axis positions remain separate from physical cache slots.

## Cache lifecycle and attention schedule

The cache stores rotated K and unrotated V in dense `[token,head,dimension]`
order. An append borrows inputs and the cache until GPU completion, then advances
the valid-token cursor. Input readiness is established with a context wait; all
launched work is synchronized even on enqueue-error paths. Validation failure
does not change the cursor or completed prefix. Reset discards the logical
prefix without promising to erase old allocation bytes. Empty appends launch
no kernels. This is a conservative synchronous API, without paging or eviction.

`Runner::attend` borrows a `RotatedQuery<QH,128>` and `KvCache<KVH,128>` and
returns owned `AttentionOutput<QH>` storage. The head width is fixed by the
Rust signature. `AttentionWindow` checks query counts, visible key counts and
causal offsets; a nonempty query requires a nonempty cache. The causal offset
counts physical cache slots, independently of rotary positions. Attention
preserves the cache and retains all input borrows through completion, including
enqueue-error paths. Empty queries launch nothing.

The attention schedule adapts the SM120 decode example: one CTA per query/head
and 64-key split, followed by a stable merge. It supports Q16/KV8/D128 plus other
valid MHA/GQA/MQA head groupings, arbitrary sequence tails and cached query
offsets. Each partial CTA uses 128 threads and 35,104 shared bytes; each merge
uses 32 threads. Scratch has `queries * query_heads * ceil(keys/64) * 130` FP32
elements, plus guards. This shared prefill/decode baseline is intended for
correct model composition; tensor-core prefill specialization remains future work.

CuTe cooperative copies transport BF16 bits as `u16` inside a private BF16 tile
wrapper, restoring the BF16 type before arithmetic. The copy exporter supports
16-bit integer storage with the same byte-oriented copy and tail zero-fill as
FP16. QK, softmax, PV and split merging remain FP32; only the final output narrows
to BF16. This is a fused-attention numerical contract, not a claim of bitwise
agreement with an eager implementation that rounds scores or probabilities.

## Host and GPU checks

From `cuda-oxide/`:

```bash
cargo test --offline -p cute-rs --features host --all-targets
cargo test --offline -p cute-rs --features host --doc
cargo test --offline --manifest-path cute-rs-preview/examples/rope_cache_cute/Cargo.toml
CARGO_NET_OFFLINE=true CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run rope_cache_cute --arch sm_120a
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  cute-rs-preview/examples/rope_cache_cute/target/release/rope_cache_cute
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 \
  cute-rs-preview/examples/rope_cache_cute/target/release/rope_cache_cute
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
  cute-rs-preview/examples/rope_cache_cute/target/release/rope_cache_cute
```

## Acceptance criteria

Validation covers Q16/KV8/D128 for both rotary modes, unequal position axes,
the last supported positions, Q6/KV2/D6 tails, empty input, cache exhaustion,
invalid lengths, reset and pending writes on another stream. One-shot and
chunked appends must match. An independent FP64 oracle with integer BF16
rounding checks Q/K bits exactly; V bits must remain unchanged. Intermediate
snapshots check every completed prefix, unwritten slot and trailing guard.
An exact fixture requires `0x3eb4` where incorrectly rounding only once produces
`0x3eb5`.

Attention checks every output against a separate FP64 dot/softmax/PV reference
with final integer BF16 rounding (`atol=2e-5`, `rtol=0.008`). Cases cover causal
and noncausal attention, MHA/GQA/MQA, 1/63/64/65/129 keys, unequal rotary axes,
nonzero query offsets, chunked and single-token decoding, amplified scores,
empty batches, reset rejection, unchanged cache storage and allocation guards.
An exact zero-query fixture crosses a split boundary. A poisoned future cache
tests that causal masking excludes both partial tiles and entire future splits.

## Evidence and limitations

The host computes FP32 angles and trigonometry, then narrows coefficients to
BF16. GPU validation checks rotation against those stored coefficients. Host
trigonometry has not been compared bit-for-bit with CUDA/PyTorch table generation;
checkpoint/reference execution is still required for model correctness. Local
execution targets RTX 5090 / SM120; no SM100 execution or performance claim is made.

Validated on RTX 5090 with CUTLASS 4.7 on 2026-10-15: 21 example host tests,
582,652 rotary/cache reference values and 2,464,256 attention reference output
comparisons. The largest attention absolute difference was `0.0078125`, within
the declared tolerance; full prefill and incremental outputs also passed their
comparison. Memcheck and synccheck reported zero errors; racecheck reported
zero hazards, errors or warnings.
The pending-input case exercises ordering but is not an isolated proof of the
context wait: allocation may also serialize prior input writes.
