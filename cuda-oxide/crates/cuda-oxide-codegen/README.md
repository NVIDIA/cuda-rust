# cuda-oxide-codegen

Rustc-independent PTX backend.

It accepts a module already assembled from cuda-oxide's `dialect-mir` and
`dialect-nvvm` operations and produces PTX through the same MIR preparation,
lowering, and LLVM tooling as the rustc frontend uses. The only supported
public surface is the `experimental` module.

The point is the absence of a dependency: this crate has **no rustc linkage**.
It needs neither `rustc_private` nor a nightly toolchain matched to
`rustc_driver`, so a caller that can build the IR itself can reach PTX without
the compiler-plugin machinery `rustc-codegen-cuda` requires.

The v1 surface may change.
