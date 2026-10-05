# The Lowering Pipeline

The previous chapters built and prepared the IR.
[MIR Importer](mir-importer.md) translated Stable MIR into `dialect-mir`.
[Compiler Optimizations](compiler-optimizations.md) explained how
`mir-transforms` rewrites that IR in place, while
[Pliron Dialects](mlir-dialects.md) described `dialect-mir`, the LLVM dialect,
and `dialect-nvvm`.

This chapter covers the next bridge: turning Rust-flavored IR into something
LLVM can compile.

For NVVM builds, a separate `nvvm-transforms` pass runs after this lowering.
It keeps `mir-lower` focused on converting `dialect-mir` to the LLVM dialect,
then adjusts that LLVM module for legacy or modern libNVVM before text export.

If you know Rust types, you are about to find out how many of them LLVM has
never heard of.

---

## What Lowering Means

`dialect-mir` speaks Rust. It knows about tuples, enums, slices, checked
arithmetic, and GPU address spaces. LLVM IR knows about none of those things.
It has flat integer and float types, `getelementptr`, PHI nodes, and a general
suspicion toward anything with more than one level of abstraction.

**Lowering** is the process of replacing every `dialect-mir` operation with
an equivalent sequence of LLVM dialect operations, one by one, until no
`dialect-mir` operations remain. Tuples become anonymous structs. Slices
become pointer-length pairs. Checked addition becomes an LLVM overflow
intrinsic followed by an extract. Every Rust concept gets flattened to
something LLVM can digest.

The pass that does all of this lives in `cuda-oxide/crates/mir-lower/` and uses pliron's
`DialectConversion` framework. It is the single largest transformation in the
pipeline, and the rest of this chapter is about how it works.

---

## DialectConversion -- The Lowering Framework

The lowering uses pliron's `DialectConversion` + `DialectConversionRewriter`
rather than a manual walk-and-replace pass. The framework handles IR walking,
def-before-use ordering, type conversion, and block argument patching
automatically.

### How It Works

Each `dialect-mir` and `dialect-nvvm` op declares how to lower itself via the `MirToLlvmConversion`
op interface (defined in `conversion_interface.rs`). The interface has a single
method: `convert(ctx, rewriter, op, operands_info)`. Each op's implementation
lives in `convert/interface_impls.rs`, which dispatches to converter functions
organized by category.

For each `MirFuncOp` in the module, `convert_func` (in `lowering.rs`):

1. **Creates an LLVM function** whose signature depends on whether the
   function is a kernel entry point. Non-kernel functions flatten every
   aggregate arg into its scalar fields for the internal CUDA ABI. Kernel
   entry points keep slices flattened (`(ptr, len)`) but pass structs and
   closures as a single byval value, since the host launcher pushes the
   whole aggregate as one packet slot (see {ref}`Argument Scalarization
   <lowering-argument-scalarization>` below).
2. **Propagates GPU metadata** (`gpu_kernel`, `maxntid`, `cluster_dim_*`).
3. **Uses `inline_region`** to move MIR blocks into the LLVM function. The
   original blocks are preserved -- no manual block mapping needed.
4. **Builds an entry prologue** that reconstructs the MIR-level value
   for every parameter from whatever shape arrived at the LLVM level
   (`insertvalue` of flattened scalars for internal calls and for
   slices; passthrough for byval aggregates at kernel boundaries).
5. **Runs `DialectConversion`** which walks every MIR operation and invokes
   its `MirToLlvmConversion::rewrite` implementation to replace it with
   LLVM operations.

### Converter Modules

The conversion functions are organized into modules by category:

| Category      | Module                        | What It Handles                                                                  |
| :------------ | :---------------------------- | :--------------------------------------------------------------------------------|
| Arithmetic    | `convert/ops/arithmetic.rs`   | `add`→`add`, `sub`→`sub`, `checked_add`→`add`+`extractvalue`                     |
| Memory        | `convert/ops/memory/*.rs`     | `mir.load`→`load`, `mir.store`→`store`, `shared_alloc`→global + `addrspacecast`  |
| Control Flow  | `convert/ops/control_flow.rs` | `mir.goto`→`br`, `mir.cond_br`→`cond_br`, `mir.return`→`return`                  |
| Aggregate     | `convert/ops/aggregate/*.rs`  | Struct/tuple field access → GEP or `extractvalue`/`insertvalue`                  |
| Cast          | `convert/ops/cast.rs`         | `IntToInt`→`zext`/`sext`/`trunc`, `FloatToFloat`→`fpext`/`fptrunc`, etc.         |
| Call          | `convert/ops/call.rs`         | `mir.call`→`call`, with argument flattening and `::` to `__` name conversion     |
| GPU Intrinsic | `convert/intrinsics/*.rs`     | NVVM ops → LLVM intrinsic calls or inline PTX                                    |
| Constants     | `convert/ops/constants.rs`    | `mir.constant`→`llvm.constant`                                                   |

The framework dispatches to these automatically -- each op's `MirToLlvmConversion`
impl calls the right converter function. The complexity lives inside each converter,
where Rust semantics meet LLVM reality.

---

## Type Conversion

Before operations can be converted, their types must be converted. LLVM's type
system is deliberately simpler than Rust's -- there is no signedness on
integers, no tuples, no enums, no fat pointers. Everything must be flattened.

| MIR Type                       | LLVM Type                    | Notes                                                                          |
| :----------------------------- | :--------------------------- | :----------------------------------------------------------------------------- |
| `IntegerType(32, Unsigned)`    | `IntegerType(32, Signless)`  | LLVM integers carry no sign -- signedness is on the *operation*, not the type  |
| `MirTupleType<i32, f32>`       | `{ i32, float }`             | Tuples become anonymous structs                                                |
| `MirSliceType<f32>`            | `{ ptr, i64 }`               | Fat pointer decomposition -- pointer + length                                  |
| `MirStructType`                | `{ fields..., [N x i8] }`    | Explicit padding arrays to match rustc's layout                                |
| `MirPtrType<f32, addrspace:3>` | `ptr addrspace(3)`           | Opaque pointers with address space preserved                                   |
| `MirArrayType<f32, 256>`       | `[256 x float]`              | Direct mapping -- arrays are simple enough even for LLVM                       |
| `MirEnumType`                  | `{ discriminant, [M x i8] }` | Discriminant + payload sized to the largest variant                            |

The integer signedness case deserves emphasis. In Rust, `i32` and `u32` are
different types. In LLVM, both are just `i32`. The sign information shifts to
the operations: a signed less-than comparison is `icmp slt`, an unsigned one is
`icmp ult`. The type converter drops the signedness, and the operation
converters pick it back up when they emit comparison and division instructions.

(lowering-argument-scalarization)=

### Argument Scalarization

Kernel entry points need special treatment. The CUDA driver doesn't
understand Rust fat pointers, so `&[f32]` has to arrive as a separate
pointer and length on both sides of the ABI. Structs and closures by
value, on the other hand, do match a single host packet slot, so the
kernel entry takes them as one byval `.param` -- otherwise the device
would expect N flattened arguments and the host would only push one,
mismatching every later slice.

The lowering pass therefore distinguishes the kernel-entry rule from the
internal-call rule:

| MIR kernel param      | LLVM kernel signature                      |
|:----------------------|:-------------------------------------------|
| `&[f32]`              | `ptr addrspace(1) %ptr, i64 %len`          |
| `T` (scalar)          | passthrough                                |
| `struct { a, b }`     | one byval `{a, b}` value                   |
| closure (N captures)  | one byval closure-struct value             |
| zero-sized aggregate  | dropped (no LLVM arg, no host packet slot) |

The slice case still uses the classic reconstruct-from-flattened pattern
in the entry block:

```text
MIR:  fn kernel(slice: &[f32])
      → entry arg: %slice : MirSliceType

LLVM: fn kernel(ptr addrspace(1) %ptr, i64 %len)
      → entry block reconstructs:
          %slice  = insertvalue {ptr, i64} undef, %ptr, 0
          %slice2 = insertvalue {ptr, i64} %slice, %len, 1
```

The struct/closure case skips the reconstruct -- the byval value is
already the right shape -- and the rest of the function sees it as if
nothing special happened. Internal device-to-device calls keep
flattening aggregates the same way they always have, so the cost of
this rule lives only at the kernel boundary.

---

## Interesting Conversions

Most conversions are straightforward: `mir.add` on integers becomes `llvm.add`,
`mir.load` becomes `llvm.load`, `mir.goto` becomes `llvm.br`. The interesting
cases are the ones where a single MIR operation expands into multiple LLVM
operations, or where GPU-specific concerns change the translation entirely.

### Checked Arithmetic

In debug builds, Rust checks every integer arithmetic operation for overflow.
MIR models this with operations like `mir.checked_add` that return a
`(result, overflow_flag)` tuple. LLVM has no such concept, but it does have
overflow intrinsics:

```text
MIR:  %result = mir.checked_add %a, %b : i32  → mir.tuple<i32, bool>

LLVM: %sum = add i32 %a, %b
      %overflow = extractvalue {i32, i1} @llvm.sadd.with.overflow.i32(%a, %b), 1
```

The overflow flag feeds into an assert that the MIR importer already lowered to
a conditional branch targeting an unreachable block. On the GPU, this
effectively means: if you manage to trigger integer overflow, the kernel traps.
Not the most graceful error handling, but the CUDA toolchain does not support
stack unwinding today.

### Shared Memory

Shared memory in CUDA is block-scoped SRAM -- fast, small, and declared
statically. In `dialect-mir`, it is a `mir.shared_alloc` operation. In LLVM
IR, shared memory must be a module-level global variable in address space 3:

```text
MIR:  %shmem = mir.shared_alloc : mir.array<f32, 256>

LLVM: @shmem_0 = addrspace(3) global [256 x float] zeroinitializer
      %ptr = addrspacecast [256 x float] addrspace(3)* @shmem_0 to ptr
```

The `addrspacecast` produces a generic pointer that the rest of the function
can use without worrying about address spaces. The NVPTX backend in LLVM
handles the rest -- it knows that `addrspace(3)` means shared memory and
generates the appropriate `st.shared` / `ld.shared` instructions.

### Enum Lowering

Rust enums are algebraically rich. LLVM has no concept of tagged unions. The
lowering pass bridges the gap by representing enums as a struct with two
fields: a discriminant (telling you which variant is active) and a payload
area sized to the largest variant:

```text
MIR:  %opt = mir.construct_enum "Some", (%val) : mir.enum<"Option_i32">

LLVM: %tmp = insertvalue { i8, [4 x i8] } zeroinitializer, i8 1, 0
      %result = insertvalue { i8, [4 x i8] } %tmp, <val into payload area>
```

The discriminant is `i8 1` because `Some` is variant 1 of `Option`. The
payload is `[4 x i8]` -- four bytes, enough to hold an `i32`. Variant access
works in reverse: read the discriminant, branch on it, then `extractvalue` the
payload and bitcast to the expected type.

It is not elegant, but it is exactly how C compilers have handled tagged unions
for decades. LLVM's optimizer is quite good at cleaning up the redundant
insertvalue/extractvalue chains.

---

## GPU Intrinsic Conversion

`dialect-nvvm` operations -- thread indexing, warp shuffles, barriers, TMA
bulk copies -- are not lowered to generic LLVM dialect operations. They are
lowered to either LLVM intrinsic calls or inline PTX assembly, depending on
whether LLVM has a built-in intrinsic for the operation.

### Strategy 1: LLVM Intrinsic Call

For operations where LLVM already provides a target-specific intrinsic, the
conversion emits a `call` to that intrinsic:

```text
nvvm.read_ptx_sreg_tid_x
  → call i32 @llvm_nvvm_read_ptx_sreg_tid_x()

nvvm.shfl_sync_bfly_i32
  → call i32 @llvm_nvvm_shfl_sync_bfly_i32(i32 -1, i32 %val, i32 %mask, i32 31)
```

Notice the warp shuffle: the user-facing `cuda_device` API takes two arguments
(value and lane mask), but the LLVM intrinsic takes four (membermask, value,
delta, clamp). The lowering pass fills in the missing arguments -- `membermask =
-1` (all lanes) and `clamp = 31` (full warp width) -- so the user never has to
think about them.

### Strategy 2: Inline PTX Assembly

Newer GPU instructions often lack LLVM intrinsics. For these, the lowering pass
emits inline PTX assembly using LLVM's `asm` syntax:

```text
nvvm.wgmma_fence_sync
  → call void asm sideeffect convergent "wgmma.fence.sync.aligned;", ""()

nvvm.mbarrier_arrive
  → call i64 asm sideeffect convergent "mbarrier.arrive.shared.b64 $0, [$1];", "=l,r"(ptr %bar)
```

The `convergent` attribute is critical here. It tells LLVM: "Do not move,
duplicate, or speculate this instruction across control flow." Without it, LLVM
might hoist a barrier out of a conditional branch or sink a warp-level
instruction past a sync point, resulting in a GPU that hangs or computes garbage
-- neither of which produces a helpful error message.

---

## Block Arguments to PHI Nodes

Pliron IR (MLIR-like) uses **block arguments** for value flow between basic
blocks. LLVM uses **PHI nodes**. They express the same concept -- "this value
comes from different predecessors" -- but the syntax is different enough that
the export step needs a real transformation, not just pretty-printing.

Pliron style (block arguments):

```text
^loop_header(%sum: f32, %i: i64):
    ...
    br ^loop_header(%new_sum, %new_i)
```

LLVM IR style (PHI nodes):

```text
loop_header:
    %sum = phi float [ 0.0, %preheader ], [ %new_sum, %body ]
    %i = phi i64 [ 0, %preheader ], [ %new_i, %body ]
```

The exporter handles this conversion with a two-pass approach:

1. **Pre-pass: name every value.** Before emitting any code, the exporter walks
   all blocks and assigns sequential SSA names (`%v0`, `%v1`, ...) to every
   value. This is critical because PHI nodes can reference values from blocks
   that appear *later* in the listing -- loop back-edges point forward in the
   text but backward in the control flow. Without pre-naming, those references
   would be undefined.

2. **Build a predecessor map.** For each block, the exporter collects
   `(predecessor_block, values_passed)` pairs by inspecting every branch
   instruction in the function.

3. **Emit PHI nodes.** At the entry of each non-entry block, the exporter emits
   one PHI node per block argument, populated with the values and predecessor
   labels from the predecessor map.

The pre-pass is the subtle part. Consider a loop: the PHI in the loop header
references `%new_sum` from the loop body, but the loop body appears *after* the
header in the textual output. If we assigned names on-the-fly during emission,
`%new_sum` would not have a name yet. The pre-pass eliminates this problem by
naming everything upfront.

---

## Debug Locations

rustc already knows where most MIR operations came from in the source program.
cuda-oxide preserves that information as it lowers the kernel:

```text
rustc Span
  ↓
pliron Location
  ↓
LLVM !dbg metadata
  ↓
PTX .loc / DWARF
  ↓
cuda-gdb source line
```

The common case is a source line in the same file as the function:

```llvm
define ptx_kernel void @vecadd(...) !dbg !func {
  %idx = call i64 @cuda_device____internal__index_1d(...), !dbg !loc
}

!loc = !DILocation(line: 39, column: 13, scope: !func)
```

That says: "this instruction belongs to line 39, column 13, inside this
function." It is just enough for cuda-gdb to stop on the right Rust line.

### Cross-file spans

Some source spans point into helper files. For example, a kernel in
`src/main.rs` may step into `cuda-device/src/thread.rs` when calling
`thread::index_1d()`.

The exporter must not attach `thread.rs:292` directly to the kernel's
`src/main.rs` scope. That would tell the debugger "line 292 of main.rs", which
is the kind of small lie that turns debugging into archaeology.

Instead, cuda-oxide creates a file-specific debug scope:

```text
src/main.rs function scope
  |
  +-- cuda-device/src/thread.rs file scope
        |
        +-- line 292, column 19
```

In LLVM metadata this looks like:

```llvm
!thread_scope = !DILexicalBlockFile(scope: !kernel_scope, file: !thread_file, discriminator: 0)
!loc = !DILocation(line: 292, column: 19, scope: !thread_scope)
```

When Pliron gives an explicit callsite location, the exporter also records
where the helper code was reached from:

```llvm
!call = !DILocation(line: 39, column: 13, scope: !kernel_scope)
!loc = !DILocation(line: 292, column: 19, scope: !thread_scope, inlinedAt: !call)
```

Read that as: "we are executing `thread.rs:292`, reached from `main.rs:39`."

### Variables

Full debug mode adds the first variable records on top of line locations:

```llvm
%tid_slot = alloca i32
call void @llvm.dbg.declare(metadata ptr %tid_slot, metadata !tid, metadata !DIExpression())

!tid = !DILocalVariable(name: "tid", scope: !func, file: !file, line: 31, type: !u32)
```

The model is the one every debug build uses:

```text
dbg.declare -> "this source variable lives at this address"
```

Because the address is stable for the variable's whole scope, cuda-gdb can read
it at any breakpoint inside that scope. To keep those addresses real, full-debug
is a `-G`-style build: it **skips** Pliron `mem2reg` (so the stack slots
survive), **skips** LLVM `opt -O2`, and runs `llc` at `-O0`. Promoting a local
to an SSA value would shrink its inspectable range to its register's liveness,
which is exactly how an optimized build ends up showing in-scope locals as
`<optimized out>`.

The aggregate locals (structs, tuples, arrays) get a `DICompositeType` whose
members carry rustc's real layout offsets, so `info locals` prints their fields:

```text
out = DisjointSlice {ptr: 0x..., len: 1}
```

Pliron `mem2reg` still has a promotion-aware salvage path: when it promotes a
debug-tagged slot it emits a `mir.dbg_value` ("this source variable has this SSA
value here") that lowers to `llvm.dbg.value`. The current `full` tier does not
run `mem2reg`, so place-backed locals normally rely on `dbg.declare`.

`ReferencePropagation` can replace a pointer assignment with an internal
`AssignRef` event:

```text
ptr = &slice[i] -> AssignRef(ptr, slice[i])
```

Full mode carries that event across stable MIR. For the supported
borrowed-slice index form, it writes the address into the variable's debugger
stack slot at the same program point:

```text
AssignRef -> address of slice[i] -> ptr's debugger slot
```

The importer validates the complete event set first. Invalid or unsupported
events omit that destination before translation, so no partial instrumentation
is emitted. Default, line-table, and release builds are unchanged. Promotion
salvage remains groundwork for a future *optimized* debug tier.

### Variable scopes and inlining

Variable metadata has one extra rule: `arg: 1` means "formal argument #1 of
the function named by this debug scope." After MIR inlining, two variables in
one kernel body can both honestly be `arg: 1`:

```text
caller kernel:  data  -> arg #1 of caller
inlined helper: self  -> arg #1 of helper
```

So cuda-oxide carries rustc's MIR source-scope table alongside each translated
function. A local debug record stores only the source-scope id it came from;
the LLVM exporter turns that id into the right DWARF shape:

```text
kernel DISubprogram
  |
  +-- helper DISubprogram
        |
        +-- DILocation(..., inlinedAt: kernel callsite)
```

This is why `data` and `self` can both keep `arg: 1` without LLVM rejecting the
module as contradictory.

The current placement is conservative. If `mem2reg` creates a block argument
that acts like a PHI value, cuda-oxide records the variable at the promoted
load/store source point. It does not yet add an eager block-entry debug record
just because the compiler created a join value:

```text
then -> merge(10)
else -> merge(20)

merge(%x):
  ... old load site becomes dbg.value(%x, "x") ...
```

That is correct but not maximally precise. A future pass can make `x` visible
earlier in the merge block once the source location and lexical scope rules are
clear.

One more compiler-maintainer detail: `mir.dbg_value` is debug-only, but inside
Pliron today it still has a normal operand use. cuda-oxide therefore does not
run Pliron DCE after this debug salvage point. If we add that later, DCE needs
to treat debug uses as non-semantic, the same way LLVM treats `dbg.value`.

PHI nodes generated by the exporter normally do not get `!dbg`: they are SSA
bookkeeping, not source operations the user wrote. Later variable metadata can
refer to PHI values through `dbg.value`, but the PHI itself should not become a
surprise stop in cuda-gdb.

---

## Symbol Name Sanitization

Function names flow through several stages, each applying its own constraints:

```text
rustc_public (FQDN)            helper_fn::cuda_oxide_device_<hash>_vecadd
  ↓ body.rs (:: → __)
dialect-mir                    helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ call.rs (:: → __)
LLVM dialect                   helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ llvm-export (strip prefix)
Textual LLVM IR                @vecadd
  ↓ llc
PTX                            vecadd
```

Three conversions happen along this path:

1. **`::` to `__`** -- Both `body.rs` (function definitions) and `call.rs`
   (call targets) replace Rust path separators with double underscores to
   produce valid pliron/LLVM identifiers. Since both sides apply the same
   conversion, definitions and call sites match.

2. **Device prefix stripping** -- `export.rs` strips the reserved
   `cuda_oxide_device_<hash>_` prefix (and any preceding FQDN crate prefix)
   from `#[device]` function names via
   `reserved_oxide_symbols::device_base_name`. This prefix exists for MIR-level
   detection but should not appear in the final LLVM IR, PTX, or LTOIR output.

3. **Device extern prefix stripping** -- For `#[device] unsafe extern "C"`
   functions, `call.rs` strips the `cuda_oxide_device_extern_<hash>_` prefix
   via `reserved_oxide_symbols::device_extern_base_name` so the LLVM IR
   references the original symbol name exported by the external LTOIR (e.g.,
   CCCL libraries).

```{note}
This manual sanitization will be replaced by pliron's `Legaliser` when the
framework is upgraded. The `Legaliser` handles `::` to `_` conversion and
collision detection systematically.
```

---

## PTX Generation

After the LLVM dialect is exported to a textual `.ll` file, the final step is
invoking `llc` -- LLVM's static compiler -- to produce PTX assembly:

```bash
llc -march=nvptx64 -mcpu=sm_90 kernel.ll -o kernel.ptx
```

### Target Selection

The pipeline probes for `llc` in the order below. LLVM 21 is the minimum —
earlier releases reject the TMA / tcgen05 / WGMMA intrinsic signatures
that cuda-oxide emits.

| Priority | Source                                                    | Notes                                                                 |
| :------- | :-------------------------------------------------------- | :-------------------------------------------------------------------- |
| 1st      | `$CUDA_OXIDE_LLC` (if set)                                | Caller-supplied override; whatever binary you point it at.            |
| 2nd      | Rust toolchain's `llvm-tools` llc                         | `<sysroot>/lib/rustlib/<host>/bin/llc` (auto-installed via `rustup`). |
| 3rd      | `llc-23` on `PATH`                                        | Distro / `apt.llvm.org` install of LLVM 23.                           |
| 4th      | `llc-22` on `PATH`                                        | Distro / `apt.llvm.org` install of LLVM 22.                           |
| 5th      | `llc-21` on `PATH`                                        | Distro / `apt.llvm.org` install of LLVM 21.                           |
| 6th      | `llc` on `PATH`                                           | Reporting fallback only; rejected at runtime if older than LLVM 21.   |

The pinned Rust toolchain (`nightly-2026-08-28`) ships LLVM 23 with NVPTX
enabled, so `rustup component add llvm-tools` is the recommended onboarding
path. The PATH probes for `llc-23` / `llc-22` / `llc-21` are kept as a fallback for
users with an existing LLVM install. If none of the probes succeed the
pipeline fails with a clear error. You can opt into a specific (possibly
older) binary by setting `CUDA_OXIDE_LLC=/path/to/llc`, but simple kernels
are the only thing guaranteed to compile on LLVM 20 and below.

If the selected target does not match the physical GPU, the CUDA driver
JIT-compiles the PTX at load time. First launch costs roughly 30ms while the
driver translates; subsequent launches use a cached binary. In practice, you
rarely notice -- the JIT is fast and the cache is persistent across runs.

The `CUDA_OXIDE_TARGET` environment variable overrides auto-detection for cases
where you need a specific target. For example, `sm_100a` enables
Blackwell-specific `tcgen05` features that are not available under the generic
`sm_100` target.

`cargo oxide run` adds a second layer of auto-detection on top of the
backend's feature-based default: when neither `--arch` nor `CUDA_OXIDE_TARGET`
is set, it queries the compute capability of CUDA device 0 and forwards that
to the backend so the generated module is guaranteed to load on the local GPU.
The full precedence is `--arch` > `CUDA_OXIDE_TARGET` > host CC (for `run`
only) > backend feature-based default. `cargo oxide build` and
`cargo oxide pipeline` deliberately skip the host-CC step so they remain
usable for cross-compilation.

If the lowered module calls CUDA libdevice, cuda-oxide switches from `llc` to
the NVVM path automatically. It resolves the target with the same precedence
before choosing typed- or opaque-pointer NVVM IR. Explicit NVVM/LTOIR commands
still require `--arch`; the feature-based fallback applies only when an
ordinary build discovers that libdevice is needed.

```{note}
Why LLVM 21? The 2-D bulk TMA load intrinsic used by `tma_copy`,
`gemm_sol`, and `tcgen05_matmul` gained a 10-operand form with `addrspace(7)`
and a `cta_group` parameter in LLVM 21. Older `llc` versions reject it with
`Intrinsic has incorrect argument type!`. Rather than maintain separate
intrinsic emitters per LLVM version, we set 21 as the minimum.
```

### Picking `opt` to match

`llc` is not the only LLVM binary the pipeline runs: the middle-end
optimization stage needs `opt`, and the two must come from the **same LLVM
major**. Mixing them is the failure this ordering exists to prevent -- `opt`
from one release can emit IR that `llc` from another does not accept.

So `llc` is resolved first, by the table above, and its major is read from
`llc --version`. `opt` is then chosen to match:

| Priority | Source                                             | Notes                                                                 |
| :------- | :------------------------------------------------- | :-------------------------------------------------------------------- |
| 1st      | `$CUDA_OXIDE_OPT` (if set)                         | Always respected. A major mismatch against the chosen `llc` is not silently corrected -- it records a diagnostic naming both binaries. |
| 2nd      | The `opt` beside the chosen `llc`                  | LLVM installs keep their tools side by side. Still version-checked: a mismatched sibling is rejected, and it is only accepted unverified when `llc`'s own version cannot be read. |
| 3rd      | Rust toolchain's `llvm-tools` `opt`                | `<sysroot>/lib/rustlib/<host>/bin/opt`. Filtered to the same major as `llc`. |
| 4th      | `opt-22` / `opt-21` / `opt` on `PATH`              | Filtered to the same major as `llc`.                                  |

If no same-major `opt` exists, resolution records a diagnostic naming every
rejected candidate. The standalone API treats a requested optimization as
strict and fails; the legacy rustc path falls back to running unoptimized.

Because step 2 keys off whichever `llc` won, pinning `CUDA_OXIDE_LLC` alone is
usually enough -- the matching `opt` is normally found next to it, and setting
`CUDA_OXIDE_OPT` as well is only needed when the pair is split across
directories.

### The other two inputs the pipeline looks for

`llvm-link` is needed only when device code calls into libdevice and the PTX
path resolves those calls with an IR-level link. `libdevice.10.bc` is needed
more often: the NVVM path adds it to every module it finalizes, whether or not
the kernel calls into libdevice. Both have their own override:

| Variable | Selects | Discovery when unset |
| :------- | :------ | :------------------- |
| `CUDA_OXIDE_LLVM_LINK` | the `llvm-link` binary | the same four steps as `opt` -- beside the chosen `llc`, then the sysroot, then versioned names on `PATH`, filtered to `llc`'s major. When the variable *is* set it is respected as `CUDA_OXIDE_OPT` is, with a diagnostic naming both binaries if its major differs from `llc`'s |
| `CUDA_OXIDE_LIBDEVICE` | the `libdevice.10.bc` bitcode file; a path that does not exist is skipped silently | `<root>/nvvm/libdevice/libdevice.10.bc` for each of `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda` |

When either piece is missing, the two paths react differently. An ordinary
build does not fail: the backend's path decision sees that the IR-level link
is unavailable and falls back to the NVVM path, the same automatic switch
described under target selection. The standalone API's `Linking::Libdevice`
is strict and fails up front instead, with a `LibdeviceUnavailable` error
naming the missing piece: `libdevice.10.bc`, or an `llvm-link` sharing the
selected `llc`'s major.

## Atomic operations in legacy NVVM IR

An “LLVM-level atomic” is an atomic instruction in the NVVM input. It is not a
statement about whether the GPU has atomic hardware. CUDA 12's LLVM 7 NVVM
dialect accepts only a subset:

| LLVM construct | CUDA 12 NVVM IR | cuda-oxide legacy path today |
|:----------------|:----------------|:-----------------------------|
| Atomic load or store | Not supported as an LLVM atomic instruction | Rejected; it needs another lowering |
| `fence` | LLVM `fence` is unsupported; an NVVM memory-barrier operation is required | Rejected pending an exact ordering/scope mapping |
| `cmpxchg` | `i32`/`i64`, plus `i128` on `compute_90+`; global/shared pointers or generic pointers known to refer there | Kept native for `i32`/`i64` at relaxed (monotonic) ordering in the generic, global and shared address spaces; a success ordering stronger than relaxed, or block/system scope, is rewritten to inline PTX. Other widths (`i128` included, on every capability), other address spaces, and a failure ordering stronger than the success ordering are rejected |
| `atomicrmw` | Integer `xchg`, `add`, `sub`, `and`, `or`, `xor`, `max`, `min`, `umax`, and `umin` on `i32`/`i64`; `i128 xchg` on `compute_90+`; the same address-space restriction | Kept native for `i32`/`i64` in the generic, global and shared address spaces, for those ten operations, at monotonic ordering. Other widths, kinds, orderings and address spaces are rejected |
| NVVM atomic intrinsics | Provide selected additional operations, including floating-point atomic add | Relaxed, device-scoped `atomicrmw fadd` on `f32`/`f64` in generic/global/shared address spaces is lowered to the exact legacy intrinsic; other *floating-point* atomic RMW operations remain rejected |

The normal LLVM-to-PTX path keeps its existing atomic support. The table above
describes only the legacy NVVM legalizer.

Two rules explain that column. The first is that nothing is
emitted whose type, address space, ordering and scope cannot be shown to
survive: [#921](https://github.com/NVlabs/cuda-oxide/pull/921) and
[#923](https://github.com/NVlabs/cuda-oxide/pull/923) admitted integer RMW
and compare-exchange by proving those properties rather than by relaxing the
requirement, which is why the surviving rejections are the cases where the
proof does not hold.

The second is that **scope is never weakened to fit**. libNVVM's legacy dialect
drops what it cannot express: an ordered `cmpxchg` lowers to a bare
`atom.cas.b32`/`.b64` with no ordering qualifier and no surrounding fence,
which PTX then treats as relaxed. So a block- or system-scoped integer atomic,
and a compare-exchange whose success ordering is stronger than relaxed, are
rewritten to inline PTX that states the scope and ordering explicitly instead
of being handed to libNVVM. That rewrite needs `sm_70` or newer. A failure
ordering stronger than the success ordering is never rewritten; it is
rejected outright.

Monotonic is the only ordering an `atomicrmw` reaches here because MIR lowering
has already split Rust's stronger orderings into separate fences, so the
instruction itself carries none.

The legacy specification also accepts but ignores `cmpxchg`'s `weak` marker
and failure ordering.

For the exact accepted types and operations, see the
[CUDA 12.4 NVVM IR specification](https://docs.nvidia.com/cuda/archive/12.4.0/nvvm-ir-spec/index.html).

---

## Putting It All Together

Here is the full sequence of events when `lower_mir_to_llvm` processes a module:

```text
1. LLVM dialect types and operations are registered automatically (link-time)
2. For each MirFuncOp in the module:
   a. Create `llvm.func` with flattened type signature
   b. inline_region: move `dialect-mir` blocks into the LLVM function
   c. Build entry prologue (reconstruct aggregates from flat args)
   d. Run DialectConversion:
      ├── Walk every `dialect-mir`/`dialect-nvvm` op (def-before-use order)
      ├── Invoke MirToLlvmConversion::rewrite for each op
      ├── Converter emits LLVM dialect op(s) via DialectConversionRewriter
      └── Framework patches block arg types automatically
3. Export the LLVM dialect to textual LLVM IR (.ll) (with PHI node conversion)
4. Invoke llc to produce .ptx
```

After step 4, you have a `.ptx` file that the CUDA driver can load and
execute. The journey from `mir.checked_add` to `add.s32` is complete.

---

The lowering pipeline turns Rust-flavored IR into GPU-ready LLVM IR. For a
hands-on walkthrough of adding new GPU operations, see
[Adding New Intrinsics](adding-new-intrinsics.md).
