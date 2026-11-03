# shared_debug

Persistent regression fixture for function-local static shared-memory DWARF.
It covers the cuda-devtools T27 `TILE` shape, a second array type, `Barrier`,
repeated references, two same-path block-local statics, and a same-leaf static
in another kernel.

Build and validate the retained LLVM debug graph:

```bash
CUDA_OXIDE_DEBUG=full cargo oxide build shared_debug --arch sm_120
./cuda-oxide/crates/rustc-codegen-cuda/examples/shared_debug/verify-debug-info.sh
```

Run the per-block semantics check:

```bash
CUDA_OXIDE_DEBUG=full cargo oxide run shared_debug --arch sm_120
```

For cuda-gdb, break at the line marked `DEBUG_SHARED_BREAK` and require:

- `show language` reports `auto; currently rust`;
- `info locals` lists `TILE`;
- `whatis TILE` / `ptype TILE` report `[i32; 32]`;
- `print sizeof(TILE)` is 128;
- block 0 prints `TILE[0] == 0`, `TILE[1] == 1`, `TILE[7] == 7`;
- block 1 prints `TILE[0] == 100`, `TILE[1] == 101` at the same shared offset;
- `print &TILE` is non-null and reports the same shared-memory offset in both
  blocks (the storage itself is distinct per block);
- the qualified spelling `shared_debug::kernels::shared_debug::TILE` is an
  observational control. The required bare lookup takes precedence because
  the DIE is scoped to its owning subprogram.

CUDA-GDB's Rust formatter does not print the address-space qualifier on
`&TILE`, so the live lookup is not the address-class proof. The verifier runs
ptxas and requires all six shared-static DIEs in the cubin to carry
`DW_AT_address_class 8`; keep the debugger in Rust mode for the source-level
checks.

The owner association is deliberately narrow: these statics materialize in
their declaration function. If MIR inlining materializes one local static in
more than one function, the divergent owners fail open: the storage is still
shared and the debug attachment is dropped, because optional metadata must
never fail a build that a release build accepts and DWARF cannot truthfully
scope one shared object to two subprograms. Representing an inlined lexical
owner is a follow-up.

With the function source/linkage-name split (#1127), the owner cache is keyed
by the physical function linkage. The prepass reads the carried source-facing
function name, preserves that as `DISubprogram::name`, emits the physical
`linkageName`, and uses the static's namespace only as the subprogram parent.
This reuses one node without replacing either naming contract.
