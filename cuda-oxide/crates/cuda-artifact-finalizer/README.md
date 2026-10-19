# cuda-artifact-finalizer

Driver-independent finalization of NVVM IR, LTOIR, and PTX into loadable device
code.

This crate is the single owner of cuda-oxide's libNVVM and nvJitLink
compilation policy. It deliberately does **not** link the CUDA Driver, so the
same typed target, FMA, debug, input-order, validation, and provenance rules
apply whether an artifact is materialized at build time (`cargo oxide build
--materialize-cubin`) or finalized at run time as a fallback.

`PtxAssembler` is discovered separately from the ordinary `Finalizer`. It
finds the toolkit `ptxas` executable, pins and fingerprints it, and assembles
already-linked PTX without loading libNVVM, nvJitLink, or the CUDA Driver.
Set `CUDA_OXIDE_PTXAS` to select an explicit executable; otherwise toolkit
roots and `PATH` are searched.

Keeping that policy in one driverless crate is what lets the two paths agree.
A rule that lived in the runtime loader alone could not be applied during a
build, and one duplicated across both would drift.

Build-time materialization passes a versioned `MaterializerHandshakeV1` from
`cargo-oxide` to the codegen backend. Its named fields bind each content digest
to a retained-file identity, so child processes can avoid rereading large CUDA
DSOs while the content-derived combined digest remains Cargo's semantic
fingerprint. Identity mismatches fall back to hashing the newly opened file.

`cargo-oxide` caches the handshake it discovers at
`.oxide-artifacts/materializer-handshake/v1.json` under the workspace root, so
subsequent builds skip rehashing the CUDA DSOs. The cache is self-validating:
a stale or corrupt file is ignored and rediscovered, and deleting it simply
forces a full rehash on the next build.

Consumers:

- `cargo-oxide` and `rustc-codegen-cuda`, for build-time materialization;
- `cuda-host`, for the runtime path.
