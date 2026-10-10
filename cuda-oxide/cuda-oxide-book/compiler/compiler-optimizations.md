# Compiler Optimizations

cuda-oxide already benefits from optimizations in rustc and LLVM. The
`mir-transforms` crate is where cuda-oxide adds its own passes while a program
is still represented as `dialect-mir`. Loop unrolling is the first
cuda-oxide-owned optimization pass in this layer.

This page starts with the user-facing idea, then follows one loop through the
compiler. You do not need compiler experience to read it.

---

## What is a MIR transform?

After rustc lowers Rust source to MIR, cuda-oxide imports it into `dialect-mir`,
where typed operations and control-flow branches are explicit.

There are three related jobs in this part of the compiler:

| Job | What it does |
|:----|:-------------|
| **Analysis** | Reads the IR and records facts, such as which blocks form a loop |
| **Transform** | Rewrites the IR without changing what the program computes |
| **Lowering** | Converts the program from one IR dialect into a lower-level dialect |

Loop analysis can identify a loop's counter, step, and bound. The unroll
transform uses those facts to rewrite the loop so that each trip performs
several original iterations. Lowering later converts the result to LLVM.

:::{note}
In this chapter, **MIR transforms** means transforms over cuda-oxide's imported
`dialect-mir`. They do not modify rustc's internal MIR directly.
:::

---

## Where transforms run

The optimization layer sits in the middle of the device compilation pipeline:

```text
Rust source
    → rustc MIR
    → mir-importer
    → dialect-mir
    → mem2reg
    → mir-transforms       ← cuda-oxide optimizations
    → mir-lower
    → LLVM dialect
    → LLVM IR
    → PTX
```

The [MIR importer](mir-importer.md) initially represents local variables with
loads and stores. `mem2reg` promotes eligible locals to **SSA values**, where
each value has one definition and block arguments carry values between blocks.
That makes loop counters and their updates explicit enough for an optimization
pass to reason about.

Transforms run here because `dialect-mir` retains Rust-level types and
operations, making targeted cuda-oxide rewrites easier to express. The
[lowering pipeline](lowering-pipeline.md) then converts the optimized result.

Full variable-debug builds skip `mem2reg` and loop unrolling. Keeping source
variables in stable memory locations gives cuda-gdb a better debugging
experience.

## Diagnosing locals that remain stack-backed

An optimized verbose build also checks the exact LLVM IR that will be passed to
`llc`. If a Rust local's stack allocation survives both cuda-oxide `mem2reg` and
LLVM's `-O2` scalar-replacement pipeline, cuda-oxide reports it before PTX
generation. For example:

```text
warning: local `scratch` (`[u32; 4]`, 16 bytes) in `kernel` could not be promoted to registers
  = note: indexed by a non-constant value; the allocation survives LLVM scalar replacement and remains stack-backed
  = help: use constant indexing or a small match/select when the index range is bounded
```

Dynamic indexing is one common blocker because LLVM cannot replace a small
aggregate with independent SSA values when the selected element is not known.
An address passed to another function, stored as a pointer, returned, or
otherwise made observable can also prevent promotion. When the surviving use
pattern is not one of the conservative cases cuda-oxide recognizes, the warning
still identifies the local but gives a generic scalar-replacement explanation
rather than guessing at the cause.

The check runs only after a successful LLVM optimization pass and only in
verbose mode:

```bash
cargo oxide build <example> --verbose
```

This placement matters. Reporting immediately after cuda-oxide `mem2reg` would
produce false positives for allocations that LLVM SROA removes later. The check
is skipped for no-optimization builds and full variable-debug builds because
those modes intentionally retain storage that an optimized build would normally
try to promote.

---

## The first pass: loop unrolling

A loop normally tests its condition and branches back after every iteration.
**Unrolling** copies the body so that one trip through the new loop performs
several iterations' work.

Consider this small counted loop inside a kernel or device function:

```rust
let mut i = 0u32;
#[unroll]
while i < 4 {
    work(i);
    i += 1;
}
```

Conceptually, full unrolling turns it into:

```rust
work(0);
work(1);
work(2);
work(3);
```

The condition and branch back to the loop header disappear. More importantly,
each copy now has a constant value for `i`, exposing further simplifications.

The tradeoff is larger generated code. A larger body may also need more
registers, so unrolling is a request to the compiler, not a promise of better
performance.

---

## Full and partial unrolling

cuda-oxide provides two forms:

| Annotation | Use it when | What the pass produces |
|:-----------|:------------|:------------------------|
| `#[unroll]` | The trip count is known at compile time | One body copy per iteration; the loop disappears |
| `#[unroll(N)]` | The trip count may be known only at runtime | `N` copies in a main loop, plus a remainder loop |

### Full unrolling

Bare `#[unroll]` needs a compile-time-known trip count:

```rust
let mut i = 0u32;
#[unroll]
while i < 8 {
    sum += i;
    i += 1;
}
```

The pass gives the eight copies counter values `0` through `7`. Constant
propagation can then fold expressions derived from those literals.

### Partial unrolling

`#[unroll(4)]` also works when the limit is a runtime value:

```rust
let mut i = 0u32;
#[unroll(4)]
while i < n {
    process(i);
    i += 1;
}
```

If `n` is `10`, the main loop handles two groups of four iterations. The
remainder loop handles iterations `8` and `9`. The result is still correct
when `n` is not divisible by four.

This removes three out of every four main-loop tests and gives the compiler a
larger straight-line region to optimize.

---

## A useful peephole: folding pipeline stages

Partial unrolling has one wrinkle: the main loop still has a runtime counter.
cuda-oxide includes a small **peephole optimization** for a common
software-pipeline pattern:

```rust
let mut k_idx = 0u32;
#[unroll(4)]
while k_idx < k_iters {
    let stage = k_idx & 3;
    match stage {
        0 => consume_stage_0(),
        1 => consume_stage_1(),
        2 => consume_stage_2(),
        3 => consume_stage_3(),
        _ => {}
    }
    k_idx += 1;
}
```

The unrolled main-loop counter advances by four. Its low two bits therefore do
not change between groups, so the four copies have fixed stage values:

```text
copy 0: (k_idx + 0) & 3  →  0
copy 1: (k_idx + 1) & 3  →  1
copy 2: (k_idx + 2) & 3  →  2
copy 3: (k_idx + 3) & 3  →  3
```

The peephole replaces those expressions with literals. Constant propagation
then removes the dead `match` arms, leaving four straight-line,
stage-specific bodies. The remainder loop stays intact for leftover
iterations.

`k_idx % 4` works too when `k_idx` is unsigned, and constant offsets such as
`(k_idx + 1) & 3` are also recognized. The pass folds these forms only when
their power-of-two period divides the unrolled step.

### Why a runtime base does not fold

The expression must be based directly on the loop counter:

```rust
let stage = k_idx & 3;                    // can fold

let global_k = tile_k_base + k_idx;
let stage = global_k & 3;                 // tile_k_base is runtime: cannot fold
```

Because the low bits of runtime `tile_k_base` are unknown at compile time, the
compiler cannot choose one literal stage for each copy. Rewriting it to the
first form is correct only when the algorithm separately guarantees that
`tile_k_base` is a multiple of four. The pass deliberately refuses to guess.

:::{note}
Full unrolling does not need this peephole. A fully unrolled counter is already
a literal in every copy, so ordinary constant propagation can do the work.
:::

---

## How the annotation reaches the pass

The source attribute travels through the compiler in six stages:

1. The `#[kernel]` or `#[device]` macro consumes `#[unroll]` or `#[unroll(N)]`
   and inserts an internal marker call at the start of that loop.
2. `mir-importer` recognizes the call and emits a `mir.unroll_hint` operation.
   The marker produces no runtime code.
3. `LoopInfo` finds the enclosing loop, while induction analysis finds its
   counter, start, step, bound, and—when possible—trip count.
4. The pass normalizes supported control flow. It joins multiple `continue`
   paths into one path back to the loop header and routes supported
   loop-carried values out of a fully unrolled loop.
5. The transform clones and reconnects the body for full or partial unrolling.
6. A cleanup phase folds constants, simplifies branches, and removes dead or
   unreachable code.

The unroll pass leaves functions without an unroll hint untouched. If it cannot
prove a requested rewrite is safe, it prints a warning and does not apply the
unroll; the loop continues with the same behavior.

---

## Supported loops and safe fallback

The current pass intentionally recognizes a conservative set of loop shapes:

Both forms need a counter the pass can follow: a compile-time-constant starting
value, the same fixed step on every path back to the header, and a simple,
side-effect-free test that directly compares the counter with its bound. The
loop must also have one ordinary entry before its header. For unusual or
irreducible control flow, the compiler warns and does not unroll the loop.

| Shape | Current behavior |
|:------|:-----------------|
| Explicit counted `while` loop | Supported when the requirements above hold |
| Range `for` loop (`for i in a..b`) | Supported when the requirements above hold |
| Other iterator `for` loops (`a..=b`, `.step_by(k)`, `.iter()`) | Not yet recognized; warns |
| Constant trip count with `#[unroll]` | Fully unrolled |
| Runtime, loop-invariant limit with `#[unroll(N)]` | Partially unrolled |
| Several `continue` paths with the same counter step | Joined and unrolled |
| Early `break` or multiple exits with `#[unroll]` | Supported |
| Early `break` or another exit with `#[unroll(N)]` | Warns; partial unroll is skipped |
| Nested loops | Only the annotated loop is unrolled |

Partial unrolling also requires `N >= 2`, a positive counter step, a `<` or
`<=` comparison, and a limit that does not change inside the loop.

When an annotated outer loop contains an unannotated inner loop, each outer
copy receives an intact clone of that inner loop; the inner loop remains a
loop. When both loops are annotated, the compiler handles the inner loop
first.

To bound compile time and memory, one annotation may create at most 1,024 body
copies, 8,192 cloned basic blocks, and 65,536 cloned operations. A request over
one of those limits warns and is not unrolled.

For the complete kernel-author reference, see {ref}`Loop unrolling <loop-unrolling>`.

---

## When should you unroll?

Good candidates are:

- small loops with a fixed trip count;
- hot counted loops where branch overhead matters;
- software pipelines whose stage selection becomes constant after unrolling;
- loops where several independent copies expose instruction-level parallelism.

Be cautious when the body is large or already uses many registers. Extra code
can increase register pressure, reduce occupancy, or put more pressure on the
instruction cache.

Choose a factor that matches the structure of the algorithm—often a pipeline
depth or another small natural group—and benchmark the result. Removing
`#[unroll]` should always remain an easy comparison.

To inspect the pass's analysis, build verbosely:

```bash
cargo oxide build <example> --verbose
```

Verbose mode prints the requested factor and the loop facts the pass found.
If a request is skipped, a warning explains why. Loop numbers and analysis
fields are internal details and may change.

---

## A home for future transforms

The `mir-transforms` crate keeps reusable analysis separate from IR mutation:

```text
cuda-oxide/crates/mir-transforms/src/
├── analyses/
│   ├── loop_info.rs      # which blocks belong to each loop?
│   └── induction.rs      # what are the counter, step, bound, and trip count?
├── canonicalize.rs       # put supported control flow into a simpler shape
└── unroll.rs             # prove, clone, reconnect, and clean up loops
```

This separation gives future optimizations a predictable place to live:
analyses gather facts, canonicalization normalizes equivalent shapes, and
transforms perform behavior-preserving rewrites. `mir-importer/src/pipeline.rs`
controls when each pass runs.

Every transform should follow the same rule as the unroller: prove the rewrite
is safe, bound its resource use, and fall back without changing behavior when a
program is outside the supported shape.

The focused test suite runs with:

```bash
cargo test -p mir-transforms --all-targets
```

From here, [The Lowering Pipeline](lowering-pipeline.md) explains how the
optimized `dialect-mir` becomes the LLVM dialect and, eventually, PTX.
