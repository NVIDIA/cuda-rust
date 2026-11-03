# cuda-intrinsics

Low-level CUDA intrinsic declarations, generated for cuda-oxide.

**Most code should use `cuda-device` instead.** This crate is the raw compiler
contract: the backend recognizes these functions by their generated paths and
replaces each call with a GPU operation. Their bodies are placeholders and are
never meant to execute — calling one outside a kernel panics.

The declarations here are produced by `cuda-intrinsics-gen` from
`intrinsics/catalog.json`, so they are not edited by hand. Adding an intrinsic
means adding a catalog entry and regenerating, not writing a declaration here.

`cuda-device` wraps these in the safe, typed, documented API that user kernels
are expected to call.
