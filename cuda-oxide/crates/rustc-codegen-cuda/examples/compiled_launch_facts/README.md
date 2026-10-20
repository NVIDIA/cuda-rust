# Compiled launch facts

This example uses `dynamic_shared = inferred` with unsafe fixed views of the
CUDA dynamic shared-memory base. `Scratch<1>` and `Scratch<4>` have different
compiler layouts, reached through the same generic helper. A second view aliases
the base and raises its alignment to 128 bytes. The views combine by maximum
extent and alignment; they do not allocate separate regions. The helper writes
one slot per lane and explicitly synchronizes before reading neighboring slots.

The generated prepared launch methods read the facts from the selected embedded
artifact. `LaunchConfig2D::auto_shared` selects 128 or 512 bytes. Preparation
rejects an explicit 511-byte allocation for the larger kernel and rejects a
32-by-1 block even though it has the same thread count as the required 8-by-4
block. The example also checks known-zero scratch and returns an owned output
buffer only after asynchronous launch completion.

From `cuda-oxide/`, run with an architecture supported by your device:

```sh
cargo oxide run compiled_launch_facts --arch sm_120
cargo oxide run compiled_launch_facts --arch sm_120 --features generic
cargo oxide run compiled_launch_facts --arch sm_120 --materialize-cubin
scripts/check-compiled-launch-facts.sh sm_120
```

The script also checks that an inferred kernel containing an unconstrained raw
dynamic view fails to compile. Generic kernels use the existing PTX loader;
generic materialization remains unsupported. Explicit raw pointer access,
initialization, bounds, aliasing, and synchronization remain the responsibility
of the unsafe device code. No synchronization is inserted by `get_fixed`.
Inference follows Rust views through direct calls. Dynamic-base accesses in
external code, indirect calls or inline assembly must fit a fixed view declared
on that path; they are not analyzed. Without declared views, inference records
zero bytes.
