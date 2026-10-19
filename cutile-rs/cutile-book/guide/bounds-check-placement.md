# Bounds-Check Placement

The safe partition API checks accesses by default. A check inside a hot loop
can disrupt software pipelining between loads and computation, even when the
kernel has few checks. The compiler removes checks it can prove unnecessary
and moves others out of loops where it can.

Each checked dimension has one of three outcomes:

- **Discharged:** no check remains in the kernel. This includes stores through
  `iter_indices()`, loads using a partition's own coordinates, constant indices
  into static shapes, and accesses covered by declared `preconditions` or
  cross-tensor facts verified at launch.
- **Hoisted:** the check runs once before the loop, or before an outer loop
  if the compiler can prove it safe to move there.
- **In place:** the check runs at the access on every iteration.

Set `deny_in_kernel_checks = true` on the entry to reject checks that remain
in the kernel. The diagnostic identifies the access and suggests how to
restructure it.

## When a Check Hoists

A check on an access inside a loop is hoisted to before the loop when all of
the following hold, per index coordinate:

| Index coordinate form | Placement |
|---|---|
| Value computed before the loop | Hoisted |
| Compile-time constant, or value with known constant bounds | Hoisted (or discharged against static shapes) |
| The loop variable `j` of a `for j in lo..hi` loop | Hoisted, checked at `hi - 1` |
| `a * j + b` with constant `a`, `b` | Hoisted at the extreme iteration, when that extreme provably fits `i32`; otherwise in place |
| Loop variable of a `(lo..hi).step_by(k)` loop | Hoisted, checked at the *last attained* index `lo + k * floor((hi - 1 - lo) / k)`, when `lo`, `hi`, and `k` are compile-time constants; otherwise in place |
| Value computed *inside* the loop body (other than the forms above) | In place |
| Any access written inside an `if`/`else` in the loop | In place |
| Any access in a loop body that contains a `continue` | In place |

Additionally, a hoisted check keeps climbing outward through directly
nested loops whose trip counts are statically non-zero and whose bodies
contain no `continue`, stopping at the first loop whose bound it depends
on. Hoisted checks are guarded so that a loop which executes zero times can
never trap. Hoisting does change *when* a violation is reported: a hoisted
check traps before the loop starts, so the iterations that precede the
offending one — and any stores they would have made — do not run. It never
changes *whether* a kernel traps: a hoisted check tests only index values
the loop actually attains, which is why a body with `continue` (some
iterations skip the access) or a stepped loop with a runtime step (the
attained set is unknown) keeps its check in place.

To help the compiler move checks out of a hot loop:

- **Compute index arithmetic above the innermost loop.** `let kv_head =
  q_head / group;` written before the K/V loop hoists every check that uses
  it; the same expression written inside the loop body does not (the
  compiler does not currently chase invariant arithmetic through the loop
  body — it proves invariance by position).
- **Index hot-loop accesses with the loop variable directly**, or an
  affine expression of it, and write the loop as `for j in lo..hi`; a
  `step_by(k)` loop hoists only when `lo`, `hi`, and `k` are all
  compile-time constants.
- **Keep hot-loop accesses unconditional.** A load under an `if` may
  execute on no iteration, so its check cannot move; lift the condition out
  of the loop or accept the in-place check. The same holds for a body with
  a `continue`: every access after it is conditional.
- **Keep index arithmetic wrap-free.** A range fact survives an operation
  only when the operation provably cannot overflow `i32`; an expression
  that can wrap forfeits its facts (even if later `max`/`%` steps pull the
  mathematical range back in bounds), and the access pays an in-place check
  over the actual runtime value.

```rust
for index in out.iter_indices() {
    let (q_tile, q_head, _) = index.components();
    let kv_head = q_head / GROUP;          // above the loop: hoists
    for j in 0i32..kv_tiles {
        let k = k_part.load_pipelined::<L>([kv_head, j, 0i32]);
        //                                  ^ hoisted  ^ hoisted (checked at kv_tiles - 1)
        // ...
    }
}
```

## Checks That Leave the Kernel Entirely

A check whose operands are known at launch can run in the generated host
launcher. Tensor extents qualify; loop variables and loaded values do not.
The launcher checks the actual shapes and refuses an invalid launch with an
error naming the check. No device registers or instructions are needed.

For example, an index from `for j in 0..num_tiles(&p, a)` or a mapped
partition's `iter_indices()` carries bounds from that tensor. Using it to
index another tensor produces a launch check over tile counts, such as
`ceil(dim(x, 1)/BK) <= ceil(dim(y, 0)/BK)`.

The persistent GEMM in `cutile-examples/examples/persistent_gemm.rs` uses
these derived checks. Its mapped components index `x` and `y`, and its K-loop
iterates over `num_tiles`. All shape checks run at launch, so it builds with
`deny_in_kernel_checks = true` without annotations or extra const generics.

Declared `preconditions` go one step further: the launcher already verifies
each declared fact against the real shapes, so the compiler assumes it and
discharges the matching checks at compile time. A kernel that declares
`dim(x, 1) % 64 == 0` emits nothing anywhere for the matching binding, and
`dim(a, i) == dim(b, j)` relates two tensors' axes exactly as the derived
form does. Declare a precondition when you want the contract visible in the
signature or need a fact the walk cannot derive; note the declared equality
is stricter than the derived comparison (`==` on extents versus `<=` on
tile counts), so it rejects some launches the derived form accepts.

The `with_bounds`/`Dim` annotation family is deprecated: everything it
proved is subsumed by the derived facts and declared preconditions above,
with the checks landing at launch instead of possibly in the kernel.

## Reading the Compiler's Decision

`CUTILE_JIT_TIMING=1` reports per-kernel totals on each compile line:

```text
CUTILE_JIT_TIMING module=kernels function=fmha_prefill ... \
    checks_discharged=3 checks_hoisted=4 checks_in_place=2
```

`CUTILE_JIT_LOG=1` explains each check that stays in a loop body:

```text
[cutile::jit] bounds check for dim 1 stays in the loop body: index is
computed inside the loop body
```

Compare the emitted Tile IR with an unchecked version to see the effect on
scheduling. Set `unchecked_accesses = true` on the entry, mark the kernel
`unsafe`, and wrap the launch in `unsafe { ... }`. Dump both variants to
stderr and compare them:

```bash
dump() { CUTILE_DUMP=ir CUTILE_DUMP_FILTER=kernels::fmha_prefill ./my_app 2>&1 \
         | sed -n '/=== CUTILE DUMP: ir/,/^}/p'; }
dump > safe.ir
# flip the kernel to its unsafe twin, then:
dump > raw.ir
diff safe.ir raw.ir
```

The persistent GEMM example emits identical IR for its safe and unsafe
variants: 103 ops with the same loop bodies. Its checks all run in the launcher.

In other kernels, an `assert` means a check remains on the device. If you
expect all checks to leave the kernel, `grep -c assert safe.ir` should return
zero; `deny_in_kernel_checks = true` enforces this during compilation.
Extra comparisons, selects, or branches between loads and `mma` operations
inside a loop can disrupt pipelining. A hoisted check before the loop does
not have that cost.

If the IR matches but performance differs, compare register counts. They
should match when all checks have left the kernel.

Two ablation knobs let you measure placement with the same binary, no
rebuild: `CUTILE_DISABLE_CHECK_HOISTING=1` pins every residual check at its
access site, and `CUTILE_FORCE_DEVICE_CHECKS=1` additionally suppresses
every proof, checking each access two-sided over its actual values — the
reference semantics the test suite diffs placement against.

## When to Reach for `unsafe`

Measure the effect of `unchecked_accesses = true`. On the flash-attention
prefill kernel used to develop check hoisting (RTX 5090,
`checks_in_place=2`), the fully checked kernel runs at 55.0 µs/call against
a 53.6 µs floor with all checks disabled — about 2.5%, all of it from the
two in-place checks on schedule-derived coordinates that execute once per
persistent index, not per inner-loop iteration. The unsafe twin of the same
kernel runs at 56.7 µs; the checked version is faster.

Unchecked accesses can help when checks remain in a hot loop: stepped-loop indices,
data-dependent indices (values loaded from memory), conditional accesses,
and index arithmetic the compiler cannot prove wrap-free. If
`CUTILE_JIT_LOG` shows in-place checks in your inner loop and restructuring
can't move them, measure whether `unchecked_accesses = true` helps.
