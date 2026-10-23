# Native library discovery

This unsafe compiler fixture loads an i32 tile and adds seven through a native
CUDA implementation. Its generic dependency helper is instantiated with one
and four items per thread. The build script registers itself as the provider;
the compiler sends concrete requests and links the returned LTO-IR. No CUB
checkout, hand-written extern declaration or external artifact list is needed.

From `cuda-oxide/`, with a CUDA toolkit and matching GPU:

```sh
cargo oxide run native_library --arch sm_120 --materialize-cubin
cargo oxide test native_library --arch sm_120 --materialize-cubin
```

The provider invokes the selected toolkit's `nvcc`; `CUDAHOSTCXX` may select
its host compiler. Launch preparation uses inferred shared-memory facts and
rejects an equal-product wrong block shape. Both kernels run three blocks and
check every output. The raw helper requires full input/output tiles and exact
participation; this fixture does not claim a safe collective API.

The smoke script supplies materialization and the target in each lane:

```sh
scripts/smoketest.sh --only '^native_library$'
scripts/smoketest.sh --only '^native_library$' --compile-only
scripts/smoketest.sh --only '^native_library$' --full-debug
```

The artifact-only lanes run the fixture's `--check-artifact TARGET OUTPUT`
mode without creating a CUDA context. It verifies the embedded CUBIN and
both kernels' launch facts, then extracts the image to `OUTPUT`. Full debug
adds `--full-debug`, requires the recorded full-debug policy, and uses
`readelf` or `llvm-readelf` to check both kernels' DWARF. That lane uses the
modern libNVVM target (the host target on Blackwell or newer, otherwise
`sm_100`) and launches no kernels.
