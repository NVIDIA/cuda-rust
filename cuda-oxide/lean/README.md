# Lean kernel verification

**Rust handles the guarantees it can establish; Lean proves the remaining GPU-specific obligations.**

This folder contains the Lean models and proofs. The Rust
[`lean-exporter`](../crates/lean-exporter/) translates a small scalar MIR subset
into Lean. Kernel annotations, buffer accesses, and compiler integration are
not implemented yet; these checks do not certify the Rust kernel.

## Intended workflow

1. Write an ordinary Rust kernel, its launch preconditions, and the properties
   to prove, naming the buffers explicitly:

   ```rust
   #[verify(bounds(a, b, c), unique_writes(c), covers(c))]
   ```

2. Rust checks the program. cuda-oxide preserves the contracts and relevant
   guarantees, then exports its `dialect-mir` at `PostMem2Reg`, before loop
   unrolling and LLVM lowering.
3. The exporter represents the actual kernel in the Lean model. Lean proves
   the requested properties using reusable lemmas and the established facts.
   Unsupported operations or an unproved obligation stop the verified build.
4. Compile the corresponding kernel. Before execution, the checked launcher
   enforces runtime preconditions such as buffer lengths and launch coverage.

For vecadd, `bounds(a, b, c)` means every executed access stays within its
buffer; `unique_writes(c)` means different threads write different elements;
`covers(c)` means every output element has a writer. Numeric `f32` correctness
is outside this first milestone.

## Shared proofs, per-kernel checks

We maintain a reusable library of general proofs. For example,
[`index_lt_of_guard`](CudaOxide/Proofs/Bounds.lean) proves that an index below a
guard limit is also below any buffer length at least as large as that limit. The
vecadd fixture applies this same lemma to both inputs. The
[launch lemmas](CudaOxide/Proofs/Launch.lean) apply to any kernel using the
modeled one-dimensional mapping, regardless of the values it computes.

Each kernel still needs evidence that its actual accesses satisfy its
annotations. The intended verifier generates those obligations and assembles
proofs using the shared rules and automation. For supported patterns, the
intended UX is for kernel authors to write contracts without Lean proofs.
A finite rule library can cover many kernels; it cannot automatically verify
arbitrary programs. New patterns may need richer annotations (such as loop
invariants) or verifier support.

Automatic obligation generation and proof selection remain to be implemented.
The current exporter emits a scalar model; its hand-written test claims check
the translation, not a production kernel-verification workflow.

## Reuse established guarantees

Lean types can carry evidence, just as our Rust APIs do. The initial `Thread1D`
model uses `Fin` for valid block and lane indices. Its bounded address is
constructed using a proof, rather than assuming the index is valid.

Only facts established by Rust or a sound host/device API may enter as
assumptions. Ownership of a buffer for the whole launch does not establish
exclusive access for each thread. Per-thread exclusivity requires a sound
indexing-API guarantee or a Lean proof of the partition. Rust bounds checks
also do not prove that an indexing operation will never fail.

## Exporter trust boundary

`lean-exporter` is written in Rust to read cuda-oxide's IR directly. Lean holds
the model and checks proofs about the exported program.

**The exporter is initially trusted.** Lean checks proofs about the exported
program; it does not automatically prove that the exporter preserved the MIR's
meaning. An incorrect translation could produce a valid proof about the wrong
program. The importer, later compiler lowering, and unsafe API implementations
also remain trusted.

The goal is to make this trusted foundation **small, explicit, and carefully checked**.
Start with a narrow supported subset, reject unknown operations, and test the
translation against its defined semantics. Tests provide evidence, not a proof
of translator correctness. We can later prove the translation correct or
check each translation independently.

## Files and checks

- [`CudaOxide/Model/`](CudaOxide/Model/Launch.lean): launch and scalar models.
- [`CudaOxide/Proofs/Bounds.lean`](CudaOxide/Proofs/Bounds.lean) and
  [`Launch.lean`](CudaOxide/Proofs/Launch.lean): reusable bounds and launch proofs.
- [`tests/vecadd/`](tests/vecadd/Accepted.lean): accepted obligations,
  counterexamples, and intentionally rejected obligations.
- [`check.py`](check.py): build and check both successful and failing cases.
- [`check-exporter.py`](check-exporter.py): export real MIR operations, check
  the generated Lean, and confirm a changed MIR operation breaks the expected result.

To see the exporter's input MIR and output Lean together, run from the `cuda-oxide/` workspace directory:

```bash
cargo oxide lean-export
```

This shows the bundled, hand-authored scalar MIR demo. To inspect another
supported MIR function:

```bash
cargo oxide lean-export path/to/helper.mir --function helper
```

Inspection needs Rust, but does not run Lean or build the CUDA backend. It
shows a translation, not a verification result.

With Lean installed through Elan and Python 3 available, run from the `cuda-oxide/` workspace directory:

```bash
python3 lean/check.py
```

Both check scripts run the Lean build and find `lake` on `PATH`, then in
`$ELAN_HOME/bin` (default: `~/.elan/bin`). They work in a Nix shell even when
Elan's executables are absent from `PATH`.

The toolchain is pinned; no external Lean packages or GPU are needed. The
checker rejects admitted proofs and unexpected axioms in the audited results.
The next milestone is extracting obligations from actual MIR and showing that
changing a Rust access changes the proof outcome. The current exporter checks
use small MIR fixtures; the full vecadd kernel is not supported yet.

To exercise the Rust exporter as well, from `cuda-oxide/` with its pinned Rust toolchain:

```bash
python3 lean/check-exporter.py
```

cLean inspires the separation of IR, semantics, and proofs, particularly its
[two-thread access reasoning](https://github.com/riyazahuja/cLean/blob/854caf29f8f2bf23c0eb821df80cac48ff6db37b/CLean/Verification/GPUVerifyStyle.lean).
Our definitions must reflect cuda-oxide and Rust's established guarantees.
This package uses original proofs; cLean's unfinished proofs and placeholder
conditions are not imported as facts.
