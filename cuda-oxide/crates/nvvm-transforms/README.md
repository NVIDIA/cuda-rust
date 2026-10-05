# nvvm-transforms

Target-aware legalization of the lowered LLVM dialect for libNVVM.

libNVVM does not accept every construct the LLVM dialect can express, and what
it accepts depends on the target. This transform runs after MIR-to-LLVM
lowering and before text export, rewriting operations into the forms the
selected libNVVM will take:

- pre-Blackwell targets receive LLVM 7-compatible operations;
- Blackwell and newer keep modern operations, apart from the NVVM-wide
  compatibility rewrites that apply everywhere.

Ordinary PTX builds skip this stage: it exists for the NVVM path, where
libNVVM rather than `llc` produces the final code.
