# cute-rs

CuTe-style tensor computation for cuda-oxide kernels. Typed views, layouts,
partitions, register tiles, copies, reductions and MMA operations retain the
storage and participation contracts needed by the device compiler.

| Domain | Contents |
| --- | --- |
| `tensor` | Core tensor views and register tiles; `partition` for strided and cyclic views |
| `copy` | `cooperative`, `tiled`, `vector` and `tma` transfers |
| `mma` | Common FP16 fragments and loads; distinct `bf16` and `block_scaled` families |
| `attention` | Checked host `metadata` and device-available `sm100` operations |
| `epilogue` | `sm120` output staging |

The architecture ownership types in `sm100`, readiness protocols in `pipeline`,
block-scaled storage in `block_scaled`, and other focused root modules retain
their separate responsibilities. Each operation family documents its own
layout, lifetime and collective-participation requirements.

The crate is `no_std`. The opt-in `host` feature exposes descriptor encoding,
attention metadata, decoder metadata and FFN metadata. Device SM100 attention
is available without that feature. Compile-time checking of a view does not
establish convergence, allocation readiness or safe asynchronous reuse.

## Import compatibility

Prefer the grouped paths for new code. Existing root item exports remain
available, and the following compatibility names refer to the same modules or
items; they do not wrap or duplicate implementations.

| Existing path | Grouped path |
| --- | --- |
| `bf16_mma` | `mma::bf16` |
| `block_scaled_mma` | `mma::block_scaled` |
| `sm100_attention` | `attention::sm100` |
| `attention::{AttentionError, AttentionShape, AttentionWindow, CacheShape}` | `attention::metadata::{AttentionError, AttentionShape, AttentionWindow, CacheShape}` |
| `epilogue` items | `epilogue::sm120` items |
| `partition` | `tensor::partition` |
| `cooperative` | `copy::cooperative` |
| `tiled_copy` | `copy::tiled` |
| `tile` | `copy::vector` |
| `tma` | `copy::tma` |

For example, `cute_rs::tma::TmaDesc` and `cute_rs::copy::tma::TmaDesc` are
identical types. Common fragments such as `mma::AccC` and core views such as
`tensor::Tensor` keep their original defining paths.

## Checks

Run from the `cuda-oxide/` workspace with its pinned nightly:

```sh
cargo test -p cute-rs --no-default-features
cargo test -p cute-rs --features host
cargo clippy -p cute-rs --all-targets --features host -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p cute-rs --features host --no-deps
```

Host checks cover layout invariants, representations, typed imports and
compile-fail contracts. Executing device operations and validating their
generated instructions additionally requires the cuda-oxide device compiler
and the appropriate GPU.
