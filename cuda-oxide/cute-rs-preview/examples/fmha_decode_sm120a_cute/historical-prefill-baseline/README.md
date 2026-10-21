# Historical prefill-at-one-query baseline

These reports predate the dedicated decode implementation. They ran the
`fmha_prefill` kernel with one valid query in a 64-query tile, and compared
it with the unchanged public CUTLASS Ampere FA2 kernel at the same shape.

The 17.791 µs Rust and 15.328 µs CuTeDSL numbers apply only to that old
implementation. They are **not** validation or performance evidence for
the current split-KV decode kernels. The old public reference remains in
`../../fmha_prefill_sm120a_cute/reference/flash_attention_v2.py`.
