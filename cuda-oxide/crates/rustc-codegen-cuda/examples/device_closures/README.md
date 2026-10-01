# Unified Device Closures Example

Demonstrates closure patterns in CUDA kernels using unified compilation.

## What This Tests

1. **Inline closures** - Closures defined and called within the kernel
2. **Closures with captures** - Closures that capture kernel parameters
3. **Closures passed to device functions** - Shared `Fn`, mutable `FnMut`, and
   genuine by-value `FnOnce` receiver paths

## Build & Run

```bash
cargo oxide run device_closures
```

## How Closures Work in GPU Code

### Rust's Closure Call Model

`Fn`, `FnMut`, and `FnOnce` are language traits defined in `core::ops`.
`std::ops` re-exports them; it does not provide a different closure-call
implementation. All three call methods use the `extern "rust-call"` ABI, which
passes ordinary arguments in a tuple at the trait-call boundary.

Rustc lowers an overloaded call such as `f(a, b)` through the appropriate
`Fn*` trait method. At a high level:

```text
f(a, b)
  -> <F as Fn*>::call*(receiver, (a, b))
  -> resolved closure body or adapter shim
```

The source-backed paths are:

- `compiler/rustc_mir_build/src/thir/cx/expr.rs`, which constructs the
  overloaded call through the callable traits.
- `compiler/rustc_middle/src/ty/instance.rs`, where
  `Instance::resolve_closure`, `needs_fn_once_adapter_shim`, and
  `ShimKind::ClosureOnce` choose the concrete instance.
- `library/core/src/ops/function.rs`, which defines the three traits and their
  receiver types.

### Why MIR Snapshots Can Look Different

For the same general source pattern, one compilation may expose
`FnMut::call_mut(&mut self, args)` while another exposes
`FnOnce::call_once(self, args)`. Those are useful observed MIR shapes, but they
are not a `no_std` versus `std` semantic rule.

The exact shape depends on the closure's inferred kind, the trait bound used at
the call site, instance resolution, and which MIR transformations have already
run. The backend must therefore inspect the resolved instance and the actual
receiver type instead of guessing from the crate's use of `std` or `no_std`.

### Direct Bodies and `ClosureOnce` Shims

Every closure has a unique anonymous environment type containing its captures.
Rustc also emits a callable body for that type. Instance resolution can select
that body directly or introduce an adapter:

```text
actual FnOnce, requested FnOnce:
  owned self ---------------------------> closure body(self, args...)

actual Fn, requested FnOnce:
  owned self -> ClosureOnce shim -> borrow self -> closure body(&self, args...)

actual FnMut, requested FnOnce:
  owned self -> ClosureOnce shim -> borrow self -> closure body(&mut self, args...)

requested Fn/FnMut:
  &self / &mut self --------------------> closure body(receiver, args...)
```

The adapter is necessary because an `Fn` or `FnMut` closure can be consumed
through an `FnOnce` bound even though its body uses a borrowed receiver. A
genuine by-value `FnOnce` closure does not need that borrow. Consequently, the
closure body does not always take a reference, and a visible
`FnOnce::call_once` does not by itself prove that the backend should create one.

### What the CUDA Backend Does

The collector and translator handle related responsibilities:

1. `collector.rs` recognizes callable-trait methods through rustc's trait
   identity, reads the receiver's closure type, and adds the monomorphized
   closure body to the device worklist. It does not rely on a function name
   containing `call_once` or `call_mut`.
2. `mir-importer/src/translator/terminator/mod.rs` verifies the exact
   `core::{Fn, FnMut, FnOnce}` method and `rust-call` ABI, records whether the
   resolved instance is a shim, extracts the closure body's symbol from the
   receiver type, and unpacks the tuple arguments.
3. The translator resolves the closure body and treats its actual first
   parameter as authoritative. When a shim must be bypassed, `&Closure`
   produces an immutable `SharedRef` and `&mut Closure` produces a mutable
   `UniqueRef`; a direct by-value `FnOnce` parameter remains a value. The
   emitted call is rejected unless its receiver type exactly matches the body.

```text
MIR callable-trait call
  -> identify core Fn* method and receiver closure
  -> resolve instance kind (body or shim)
  -> collect/select closure body
  -> conditionally adapt receiver
  -> unpack rust-call tuple
  -> emit direct device call
```

## Why Closures Need Special Handling

Regular helper functions such as `#[device] fn helper(...)` already have an
ordinary direct-call ABI. Closures add a generated environment receiver, an
`Fn*` trait-call boundary, possible adapter shims, and tuple-packed
`rust-call` arguments.

| Aspect         | Closures                              | Regular functions    |
|----------------|---------------------------------------|----------------------|
| Call boundary  | `Fn*::call*` trait method             | Direct function call |
| ABI            | `rust-call` tuple arguments           | Standard Rust ABI    |
| Receiver       | Value or reference, possibly adapted  | Explicit parameters  |
| Resolution     | Body or `ClosureOnce` adapter shim    | Function instance    |

## Test Cases

| Test                           | Description                                | Receiver path                    |
|--------------------------------|--------------------------------------------|----------------------------------|
| `test_inline_closure`          | `\|x\| x * 2` inside kernel                | Optimized inline                 |
| `scale_kernel`                 | `input[i] * factor`                        | Scalarized capture               |
| `transform_kernel`             | `(x + offset) * scale`                     | Multiple scalarized captures     |
| `inline_with_param`            | Inline closure using kernel parameter      | Captured `Fn`                    |
| `test_closure_constant`        | `\|\| 42`                                  | No captures                      |
| `test_closure_multi_arg`       | `\|a, b\| a + b`                           | Tuple unpacking                  |
| `test_closure_fnonce`          | Capture-free closure through `FnOnce`      | `Fn` adapter -> `&Closure`       |
| `test_closure_capture_fnonce`  | Immutable capture through `FnOnce`         | `Fn` adapter -> `&Closure`       |
| `test_closure_fnmut_adapter`   | Mutating capture through `FnOnce`          | `FnMut` adapter -> `&mut Closure` |
| `test_genuine_fnonce_receiver` | Consumes a non-Copy capture                | Direct closure value             |
| `test_wrapper_method_named_call` | Ordinary method named `call`             | Not callable-trait dispatch      |
| `test_closure_field_projection` | Closure reached through a struct field    | Shared receiver projection       |
| `test_closure_double_ref`      | Closure called through two references      | Shared receiver dereference      |
