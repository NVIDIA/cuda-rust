# Fuzzing and Differential Testing

The happy path through cuda-oxide is easy to test with examples: `vecadd`,
`gemm`, `sharedmem`, and friends prove that known programs still work. But
compilers do not only fail on known programs. They fail on the weird little
corners no one thought to write by hand -- the cast after the branch, the tuple
inside the call terminator, the integer width that nobody invited to the party.

That is where fuzzing enters the picture.

cuda-oxide uses a small rustlantis-based harness to generate random custom MIR
programs, run them through both the normal Rust CPU backend and cuda-oxide's GPU
backend, and compare a compact trace of intermediate values. The aim is not to
prove all of Rust correct on GPUs. The aim is more modest, and more useful:
stress the MIR importer, lowering pipeline, LLVM export, PTX generation, and
runtime execution with programs we did not hand-author.

---

## What We Are Comparing

At first glance, "compare CPU and GPU execution" sounds suspicious. CPUs and
GPUs have different execution models: one scalar thread versus thousands of
SIMT lanes, divergent control flow, different memory spaces, different
synchronization rules. A general CPU-vs-GPU semantic comparison would be a very
fancy way to lie to ourselves.

So the harness deliberately avoids that.

It runs generated MIR as a scalar GPU program:

```text
<<<1, 1>>>
```

One block. One thread. No inter-thread communication. No scheduling drama. The
GPU is being used as a second codegen target for the same scalar MIR, not as a
parallel programming model.

The comparison is:

```text
same generated MIR
  -> normal rustc/LLVM CPU execution
  -> cuda-oxide -> LLVM IR -> PTX -> CUDA execution
  -> compare trace hash
```

If the hashes match, the generated program observed the same sequence of
intermediate values on both paths. If they differ, something in the cuda-oxide
path deserves attention.

```{note}
The trace is intentionally compact. Instead of copying every intermediate value
back to the host, each `dump_var(...)` call folds values into one `u64` using a
byte-wise hash. This makes generated tests cheap to run and easy to compare.
```

---

## The Moving Pieces

The fuzzing setup is split into four parts:

| Piece                                  | Role                                           |
| :--------------------------------------| :--------------------------------------------- |
| `cuda-oxide/crates/fuzzer`                        | Shared trace API and vendored rustlantis       |
| `cuda-oxide/crates/fuzzer/tools/mir_generator.py` | Seed-to-`generated_case.rs` adapter            |
| `cuda-oxide/crates/fuzzer/tools/run_seed.py`      | Batch runner and artifact recorder             |
| `rustlantis-smoke`                     | Stable CPU/GPU execution harness               |

`cuda-oxide/crates/fuzzer` is a normal workspace crate, but its library surface is
`no_std`. Device code imports `trace_reset`, `trace_finish`, and the generic
`dump_var` from there. The actual rustlantis source is vendored under
`cuda-oxide/crates/fuzzer/rustlantis`; it is invoked as an external generator, not used as
a Rust library dependency.

The `rustlantis-smoke` example lives under
`cuda-oxide/crates/rustc-codegen-cuda/examples/`. It owns the host/GPU launch logic and a
small hand-written sanity test, then includes one generated file:

```text
cuda-oxide/crates/rustc-codegen-cuda/examples/rustlantis-smoke/src/generated_case.rs
```

The fuzzer tools rewrite that file for each seed. Everything else in the
example stays stable. This keeps the harness easy to review: if a seed fails,
the generated MIR is isolated in one place.

---

## The Seed Pipeline

One seed follows this path:

1. **Generate.** rustlantis receives a numeric seed and a small config. Same
   seed plus same config means the same custom-MIR program.
2. **Extract.** `mir_generator.py` pulls out the first generated `#[custom_mir]`
   function.
3. **Adapt.** rustlantis emits calls like `dump_var(a, b, c)`. The adapter
   rewrites those into tuple locals and one generic `fuzzer::dump_var(...)`
   call, because custom MIR call operands are picky about tuple expressions.
4. **Inject.** The adapted function and a small wrapper are written to
   `src/generated_case.rs`.
5. **Run.** `cargo oxide run rustlantis-smoke` executes the CPU oracle and GPU
   kernel.
6. **Classify.** `run_seed.py` records whether the seed passed, mismatched,
   failed to compile, or exceeded the adapter's current support surface.

The checked-in generated case currently uses seed `19`, because it shows the
important property we want from the harness: multiple intermediate dumps, not
just one final value.

```rust
__rl_dump0 = (Move(_1), Move(_2), Move(_3), Move(_4));
Call(_9 = dump_var(Move(__rl_dump0)), ReturnTo(bb4), UnwindUnreachable())

__rl_dump1 = (Move(_6),);
Call(_9 = dump_var(Move(__rl_dump1)), ReturnTo(bb5), UnwindUnreachable())
```

That means the final trace hash includes several values from one point in the
program and another value from a later point. It is still compact, but it is no
longer merely "did the return value match?"

---

## Running It

Run one seed:

```bash
python3 cuda-oxide/crates/fuzzer/tools/run_seed.py --seed 192
```

Run a batch:

```bash
python3 cuda-oxide/crates/fuzzer/tools/run_seed.py --start 0 --count 20 --keep-going
```

Useful flags:

- `--keep-going`: continue after a failing seed.
- `--keep-logs`: write logs for passing seeds too.
- `--no-build`: reuse an already-built rustlantis generator.
- `--append-summary`: append to the existing summary instead of replacing it.

By default, `summary.jsonl` is replaced at the start of each run. This makes it
answer the obvious question: "what happened in the run I just did?" If you want
history, opt in with `--append-summary`.

---

## Reading Results

The runner prints one line per seed and then a full summary:

```text
results:
  seed 0: UNSUPPORTED [adapter] unsupported return type for return-value tracing: *const i8 (...)
  seed 1: COMPILE_FAIL [backend] Unsupported construct: Type translation not yet implemented for: RigidTy(Char) (...)
summary: COMPILE_FAIL=1, UNSUPPORTED=1
```

The statuses mean:

| Status                   | Meaning                                                |
| :----------------------- | :------------------------------------------------------|
| `PASS`                   | CPU and GPU traces matched                             |
| `MISMATCH`               | CPU and GPU traces differed                            |
| `COMPILE_FAIL [backend]` | The adapter produced a case, but cuda-oxide failed it  |
| `UNSUPPORTED [adapter]`  | rustlantis generated MIR, but the adapter declined it  |

`MISMATCH` is the result to take most seriously. Both paths compiled and ran,
but observed different values. That is a potential backend correctness bug.

`COMPILE_FAIL [backend]` means the generated case made it past the adapter and
into cuda-oxide. The failure may still be expected -- for example, a currently
unsupported MIR type -- but the backend is the component that rejected it.

`UNSUPPORTED [adapter]` means rustlantis produced a program, but our adapter
refused to turn it into a smoke case. For example:

```text
unsupported return type for return-value tracing: *const i8
```

That refusal comes from the function boundary, not from a dump site. rustlantis
only dumps values it can hash, and the adapter accepts all of those. What the
adapter can still refuse is the function itself: a return type the trace API
cannot hash, or an argument type it has no literal for. The trace hashes these
scalars:

```text
bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, char,
f32, f64
```

It also hashes arrays and tuples of anything in that list, nested as deep as
you like: the trace just hashes every scalar inside. Tuples work up to 5
elements, matching the `TraceDump` implementations.

```text
[u128; 4]                  hashes fine
(u32, char)                hashes fine
(i8, i8, i8, i8, i8, i8)   rejected: six elements
*const i8                  rejected: pointer
```

What's left out: raw pointers and references, because the CPU and the GPU hold
different addresses for the same object, so a pointer's hash could never match;
and tuples with more than 5 elements. Many adapter-level unsupported cases are
therefore not "bad MIR" and not cuda-oxide bugs. The pointer refusal is
permanent by design; no amount of widening makes those hashes agree. The tuple
limit is just a gap. The harness needs one more `TraceDump` impl, and maybe a
snack, before it can hash a six-element tuple.

---

## Artifacts

Per-seed logs live under:

```text
cuda-oxide/crates/fuzzer/artifacts/
```

Failure logs include:

- the seed
- status and stage
- reason
- return code
- command
- full command output
- the generated `generated_case.rs` snapshot, when one exists

The generated snapshot matters. If a backend failure appears in CI or a long
batch run, the log is enough to see the exact MIR-shaped program that triggered
it. The seed lets you regenerate it, but the snapshot saves you a round trip.

---

## Current Limits

The current config is intentionally small. It keeps the first stage focused on
scalar custom MIR and backend plumbing rather than every Rust construct at once.
That is why many early seeds classify as `UNSUPPORTED [adapter]`.

The widening plan is incremental, and two steps have already landed: the
adapter now accepts every scalar the trace API can hash (f32 and f64 arrived
last, in [#484](https://github.com/NVlabs/cuda-oxide/pull/484)), plus arrays
and tuples built from them
([#792](https://github.com/NVlabs/cuda-oxide/pull/792)). What remains:

1. Decide whether and how to support `char` in cuda-oxide's type translation.
2. Expand control-flow and cast coverage.
3. Add structs and enums; the trace can't look inside those yet.
4. Add minimization for failing seeds.

That order is deliberate. A fuzzer that generates everything on day one mostly
generates noise. A fuzzer that grows one axis at a time tells you what broke,
and why. Much friendlier.
