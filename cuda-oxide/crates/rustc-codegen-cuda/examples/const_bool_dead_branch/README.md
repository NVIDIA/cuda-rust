# `const_bool_dead_branch`

This example checks per-instance reachability for a generic branch controlled
by an associated `const bool`.

rustc's monomorphization collector follows only the selected `SwitchInt` edge.
CUDA Oxide must make the same choice during device call-graph collection and
MIR import. Otherwise a statically dead hook may be rejected as panic-only or
left as an unresolved device symbol.

The example covers explicit and default trait hooks. Each has one live
instantiation and one instantiation where the hook is const-dead. A dynamic
control keeps both successors reachable and proves that an unknown
discriminant still collects and emits its hook. A pointer case additionally
proves that a const-dead dynamic-shared arm cannot contaminate the address
space inferred for the live global-pointer arm. A runtime-controlled version
proves that two live, disagreeing pointer spaces remain generic.

```bash
cargo oxide run const_bool_dead_branch
CUDA_OXIDE_NO_OPT=1 cargo oxide run const_bool_dead_branch
./cuda-oxide/crates/rustc-codegen-cuda/examples/const_bool_dead_branch/verify-code-shape.sh
```

The code-shape check requires the `.ll` and `.ptx` artifacts left by
`cargo oxide` and also checks `.opt.ll` when optimization produced it. It
verifies that live hook symbols are present and const-dead hook symbols are
absent in every available stage.

Expected final line:

```text
const_bool_dead_branch: PASS (4 values, seven generic instantiations)
```
