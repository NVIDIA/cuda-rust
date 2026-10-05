# Architecture: cuda_tile Macro System

Developer reference for the `#[cuda_tile::*]` attributes in `cutile/src/_core.rs`.
These macros bridge the Rust DSL surface with the JIT compiler and Tile IR backend.

## Overview

The macro system has two concerns:

1. **Variadic expansion** (`variadic_struct`, `variadic_op`, `variadic_impl`):
   Rust lacks variadic const generics (`const D: [i32; N]`), so these macros
   stamp out rank-specific versions at compile time.

2. **JIT metadata** (`ty`, `op`, `compiler_op`): Annotations that the JIT
   compiler reads at runtime to emit Tile IR operations and construct Tile IR types.

Functions annotated with `#[cuda_tile::op]` or `#[cuda_tile::compiler_op]` have
bodies of `unreachable!()` -- they are never executed as Rust. Instead, the JIT
reads their signature and attributes, then emits the corresponding Tile IR.
Functions *without* these annotations have real bodies and are inlined by the JIT
into the caller.

---

## `#[cuda_tile::ty(...)]`

Declares how a Rust type maps to a Tile IR type. Applied to structs and trait impls.

### Attributes

| Attribute              | Purpose |
|------------------------|---------|
| `name`                 | Tile IR type name (e.g., `"!cuda_tile.tile"`, `"!cuda_tile.partition_view"`) |
| `type_params`          | Required Tile IR type parameters, in order |
| `type_params_optional` | Optional Tile IR type parameters, in order (appended after required) |
| `type_meta`            | Runtime metadata fields carried on the value (not part of the Tile IR type string) |
| `pointer_type`         | For pointer types: the Tile IR pointer wrapper type |

### How type params work

The compiler constructs the Tile IR type string by concatenating:
1. Required `type_params`, in order
2. Optional `type_params_optional`, in order -- included only when a matching
   entry exists in the compiler's `type_params` HashMap

The HashMap is populated by constructor ops (see `output_type_params` below).
If a required param cannot be resolved (not in the HashMap, not derivable from
the Rust type), `compile_type` returns `None` and the compiler falls back to
type inference via `derive_type`.

**Ordering matters.** The Tile IR parser expects type parameters in a fixed
order. `type_params_optional` entries must appear in the order the dialect expects
them, after the required params.

### Patterns for type params

| Pattern    | Meaning |
|------------|---------|
| `"{D}xE"`  | Tile shape x element type, e.g., `128x64xf32` |
| `"{D}xP"`  | Tile shape x pointer type |
| `"tile"`   | Tile shape only (derived from the Rust const generic array) |
| `"strides"`| Stride array -- cannot be derived from Rust type alone, triggers deferred inference |
| `"tensor_view"` | Tensor view param -- cannot be derived from Rust type alone, triggers deferred inference |
| `"padding_value"` | Padding specification -- only included when set by a constructor op |
| `"dim_map"` | Dimension permutation map |

When `compile_type` encounters `"tensor_view"` or `"strides"` as a *required*
param that isn't in the HashMap, it returns `Ok(None)` -- signaling that this
type needs constructor context to be fully resolved. The JIT then calls
`derive_type`, which traces through the constructor op's `output_type_params` to
populate the HashMap before retrying.

### Examples

**Simple element type (trait impl):**
```rust
#[cuda_tile::ty(name = "f32")]
impl ElementType for f32 {}
```
Maps `f32` to Tile IR `f32`. No type params.

**Tile (shape x element type):**
```rust
#[cuda_tile::ty(name="!cuda_tile.tile", type_params=["{D}xE"])]
pub struct Tile<E: ElementType, const D: [i32; N]> { .. }
```
`Tile<f32, {[128, 64]}>` becomes `!cuda_tile.tile<128x64xf32>`.

**Tensor (shape x element type + strides + runtime metadata):**
```rust
#[cuda_tile::ty(name="!cuda_tile.tensor_view",
                type_params=["{D}xE", "strides"],
                type_meta=["base", "shape", "strides", "token"])]
pub struct Tensor<E: ElementType, const D: [i32; N]> { .. }
```
The `strides` param triggers deferred inference: `compile_type` returns `None`
from the annotation path, so the JIT derives the type from the constructor
(`make_tensor_view`) that populates `strides` via its `output_type_params`.

`type_meta` declares runtime metadata that travels with the value but isn't
part of the printed Tile IR type. Here, a `Tensor` carries its base pointer, shape,
strides, and a memory-ordering token.

**Partition view (required + optional params):**
```rust
#[cuda_tile::ty(name="!cuda_tile.partition_view",
                type_params=["tile"],
                type_params_optional=["padding_value", "tensor_view", "dim_map"],
                type_meta=["token", "tensor_view.shape()"])]
pub struct Partition<'a, E: ElementType, const D: [i32; N]> { .. }
```

The required param `tile` is always present (derived from `D`). The optional
params appear only when set by the constructor op:
- `make_partition_view(..., padding::None, dim_map::Identity, ...)` sets
  `tensor_view`
- `make_partition_view(..., padding::Zero, dim_map::Identity, ...)` sets
  `tensor_view` + `padding_value`
- `make_partition_view(..., padding::None, dim_map, ...)` sets
  `tensor_view` + `dim_map`

Resulting Tile IR types:
```
partition_view<tile=(64x64), tensor_view<?x?xf32, strides=[?,1]>>
partition_view<tile=(64x64), padding_value = zero, tensor_view<?x?xf32, strides=[?,1]>>
```

---

## `#[cuda_tile::op(...)]`

Declares a function as a primitive Tile IR operation. The function body is
`unreachable!()` -- the JIT emits Tile IR instead.

### Attributes

| Attribute              | Purpose |
|------------------------|---------|
| `name`                 | Tile IR operation name (e.g., `"cuda_tile.make_partition_view"`) |
| `params`               | Function parameters that become Tile IR operands, by name |
| `output_type_params`   | Parameter names whose types are forwarded to the output Tile IR type |
| `output_type_meta`     | Expressions that become runtime metadata on the output value |
| `attribute_params`     | Parameters encoded as Tile IR attributes (not operands). Format: `"name:kind"` |
| `hint_params`          | Parameters that guide compilation but don't appear in Tile IR (e.g., latency hints) |
| `named_attributes`     | Static Tile IR attributes. Format: `"attr_name=attr_value"` |
| `static_params`        | ZST type parameters resolved to Tile IR attributes at compile time |
| `has_variadic_params`  | `true` if operand count varies (adds `operandSegmentSizes` attribute) |

### How constructor ops connect to struct types

This is the central mechanism for how ops produce typed values.

**The problem:** Some Tile IR type parameters can't be derived from Rust type
annotations alone. For example, `Partition<E, D>` only carries the tile shape `D`
in its Rust type, but the Tile IR `partition_view` type also needs `tensor_view`
and optionally `padding_value`.

**The solution:** Constructor ops declare `output_type_params` -- parameter names
whose *types* and *values* are forwarded into the output type:

```rust
// Constructor op:
#[cuda_tile::op(name="cuda_tile.make_partition_view",
                params=["tensor_view"],
                output_type_params=["tensor_view", "padding_value", "dim_map"],
                output_type_meta=["token", "tensor_view.shape()"])]
fn make_partition_view(
    tensor_view: &Tensor<E, TENSOR_SHAPE>,  // forwarded as type param
    tile: Shape<TILE_SHAPE>,
    padding_value: impl padding::Mode,       // forwarded as type param
    dim_map: impl dim_map::Mode,             // forwarded as type param
    token: Token,
) -> Partition<'a, E, TILE_SHAPE> { unreachable!() }
```

When the JIT compiles a call to this function:

1. It compiles each argument and records the parameter name to compiled type
   mapping (`arg_types` HashMap) plus normalized metadata values for marker
   arguments such as `padding::Zero`.

2. For each name in `output_type_params`:
   - Looks up the corresponding argument type from `arg_types`
   - Calls `TypeParam::derive_param_from_type` to create a type parameter
   - For `padding_value` and `dim_map` specifically: if the marker is
     `padding::None` or `dim_map::Identity`, skip the optional type param.
   - For real `padding_value` specifically: if the derived `TypeParam` is a
     `TypeParam::Padding` variant, sets its value from `arg_string_values`
     (i.e., the `"zero"` string literal from the call site)

3. Inserts these into the `type_params` HashMap.

4. Calls `compile_type` on the return type with this HashMap. The struct's
   `type_params_optional` picks up `padding_value` and `tensor_view` from
   the HashMap and includes them in the Tile IR type.

### `output_type_meta`

Declares runtime metadata on the output value. These are expressions evaluated
at the call site and attached to the result:

```rust
output_type_meta=["token", "tensor_view.shape()"]
```

- `"token"` resolves to the `token` parameter's compiled value
- `"tensor_view.shape()"` resolves by substituting the `tensor_view` parameter
  expression, then compiling `<tensor_view_expr>.shape()` -- yielding the
  tensor's dynamic shape at runtime

This metadata is accessed later via `compiler_op(name = "return_type_meta_field")`.

### How `params` maps to Tile IR operands

`params` lists parameter names that become SSA operands in the Tile IR operation.
Parameters NOT in `params` are still compiled (for type inference, string values,
etc.) but don't appear as operands.

Dot notation accesses struct fields:
```rust
params=["base", "shape.dims", "strides.dims"]
```
Here `shape.dims` extracts the `dims` field from the compiled `Shape` struct value.

### `static_params`

Maps Rust ZST (zero-sized type) generic parameters to Tile IR attributes. The
format is a map from ZST variant names to Tile IR attribute strings:

```rust
#[cuda_tile::op(name="cuda_tile.sqrt", params=["x"],
    static_params=[
        "rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, ...}",
        "ftz={Enabled: flush_to_zero=unit}"
    ])]
fn sqrt<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
    x: Tile<E, S>, _r: R, _f: F,
) -> Tile<E, S> { unreachable!() }
```

At the call site `sqrt::<_, _, rounding::NearestEven, ftz::Enabled>(tile)`, the
compiler resolves `R = NearestEven` and `F = Enabled`, then emits the
corresponding Tile IR attributes.

### Examples

**Simple op (no type forwarding):**
```rust
#[cuda_tile::op(name="cuda_tile.cos", params=["x"])]
fn cos<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
    unreachable!()
}
```
Emits `%result = cuda_tile.cos %x : !cuda_tile.tile<128xf32>`.

**Op with hint parameter:**
```rust
#[cuda_tile::op(name="load_view_tko", params=["view", "index"])]
fn load_view_tko(view: &Partition<E, D>, index: [i32; N], latency: Option<i32>, tma: T, ..) { .. }
```
`latency` guides TMA vs non-TMA lowering but doesn't appear as a Tile IR operand.

**Constructor op with type forwarding (detailed walkthrough):**

Given these definitions:
```rust
// Struct type:
#[cuda_tile::ty(name="!cuda_tile.partition_view",
                type_params=["tile"],
                type_params_optional=["padding_value", "tensor_view"])]
pub struct PartitionMut<'a, E: ElementType, const D: [i32; N]> { .. }

// Constructor op:
#[cuda_tile::op(name="cuda_tile.make_partition_view",
                params=["tensor_view"],
                output_type_params=["tensor_view", "padding_value"],
                output_type_meta=["token"])]
fn make_partition_view_mut(
    tensor_view: &Tensor<E, TENSOR_SHAPE>,
    shape: Shape<TILE_SHAPE>,
    padding_value: impl padding::Mode,
    token: Token,
) -> PartitionMut<'a, E, TILE_SHAPE> { unreachable!() }
```

Call site:
```rust
let pv: PartitionMut<E, S> =
    unsafe { make_partition_view_mut(y, tile_shape, padding::Zero, token) };
```

Compilation flow:

1. `compile_type` for annotation `PartitionMut<E, S>` sees required param
   `"tile"` (derivable from `S`) but `type_params_optional` entries
   `"padding_value"` and `"tensor_view"` aren't in the empty HashMap.
   `"tensor_view"` triggers deferred inference and returns `Ok(None)`.

2. Since `ct_ty = None`, `compile_expression` passes `return_type = None` to
   `compile_general_op`.

3. `compile_general_op` calls `derive_type`, which invokes the type derivation
   path in `compile_type.rs`. This path:
   - Looks up `make_partition_view_mut`'s `output_type_params`
   - Finds `["tensor_view", "padding_value"]`
   - Inserts `tensor_view` as the compiled tensor arg type, `padding_value` as
     `TypeParam::Padding { padding_value: Some("zero") }` into the HashMap.

4. `compile_type` retries with the populated HashMap:
   - Required `"tile"` derived from `S`
   - Optional `"padding_value"` found in HashMap
   - Optional `"tensor_view"` found in HashMap

5. Result: `!cuda_tile.partition_view<tile=(64x64), padding_value = zero, tensor_view<?x?xf32, strides=[?,1]>>`

### One Rust op surface for optional Tile IR type params

The partition-view constructor has a single read-only Rust function and one
mutable Rust function. Marker arguments determine which optional Tile IR type
parameters appear:

```rust
fn make_partition_view(
    tensor,
    shape,
    padding_value: impl padding::Mode,
    dim_map: impl dim_map::Mode,
    token,
) -> Partition { .. }

unsafe fn make_partition_view_mut(
    tensor,
    shape,
    padding_value: impl padding::Mode,
    token,
) -> PartitionMut { .. }
```

Both emit the same `cuda_tile.make_partition_view` Tile IR op. The output type
includes `padding_value` only for real padding markers such as `padding::Zero`,
and includes `dim_map` only for non-identity mappings.

---

## `#[cuda_tile::compiler_op(...)]`

Declares a function as a compiler intrinsic. Unlike `cuda_tile::op`, these are
handled by dedicated Rust code in the compiler (`compile_intrinsic.rs`) rather
than the general Tile IR op emission path.

### Attributes

| Attribute         | Purpose |
|-------------------|---------|
| `name`            | Intrinsic name used to dispatch in `compile_compiler_op_call` |
| `type_meta_field` | For `return_type_meta_field` / `set_type_meta_field`: which metadata field to access |

### Common intrinsic names

| Name                     | Purpose |
|--------------------------|---------|
| `"cast"`                 | Type casts (scalar to tile, pointer to pointer tile) |
| `"convert"`              | Element type conversion |
| `"check"`                | Runtime assertions (e.g., bounds checking) |
| `"shape"`                | Extract a dimension from a shape |
| `"return_type_meta_field"` | Read runtime metadata from a value |
| `"set_type_meta_field"`  | Update runtime metadata on a value |
| `"tile"`                 | Tile-level comparison ops (eq, ne, gt, lt, ...) |
| `"arithmetic"`           | Scalar and tile arithmetic (min, max, ceil_div, true_div) |
| `"reduce"`               | Reduction ops (sum, max, min, prod along a dimension) |
| `"mma"`                  | Matrix multiply-accumulate |
| `"assume"`               | Compiler hints (divisibility, bounds) |

### Metadata accessors

The `return_type_meta_field` and `set_type_meta_field` intrinsics read and write
the runtime metadata declared in `type_meta`:

```rust
// Tensor declares: type_meta=["base", "shape", "strides", "token"]

// Read the token from a tensor:
#[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
fn get_tensor_token(tensor: &Tensor<E, S>) -> Token { unreachable!() }

// Write a new token to a tensor:
#[cuda_tile::compiler_op(name = "set_type_meta_field", type_meta_field = "token")]
fn set_tensor_token(tensor: &Tensor<E, S>, token: Token) { unreachable!() }

// Read the shape from a tensor:
#[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "shape")]
fn get_tensor_shape(tensor: &Tensor<E, S>) -> Shape<S> { unreachable!() }
```

These compile to direct reads/writes of the SSA metadata attached to the value --
no Tile IR operation is emitted.

---

## Rank-polymorphism macros

The DSL is rank-polymorphic via const-generic-array (CGA) generics —
parameters of the form `const X: [i32; N]`. Items annotated with the macros
below get specialized over the supported ranks.

### `#[cuda_tile::variadic_struct(N = 6)]`

Emits per-rank struct definitions for ranks 0 through `N`.
`Tile<E, const D: [i32; N]>` produces `Tile_0<E>`, `Tile_1<E, const D_0: i32>`,
`Tile_2<E, const D_0: i32, const D_1: i32>`, etc.

Optional: `constructor = "new"` generates a `const_new()` constructor on
each per-rank variant. If the struct has a slice-typed `dims` field, the
macro additionally emits `new_K(dims: &'a [i32; K])` for each `K`.

### `#[cuda_tile::variadic_impl(N = 6)]`

For inherent impls on a CGA-bearing struct (`impl<E, const D: [i32; N]> Tile<E, D> { … }`),
emits one per-rank impl block, with method bodies rewritten so any reference
to `D`, `Shape<D>`, `Tile<E, D>`, etc. picks up the per-rank concrete form.
Method names are not suffixed; rustc's receiver-type dispatch already
disambiguates.

### `#[cuda_tile::variadic_op(N = 6)]`

For free functions, this emits a single CGA-erased rank-polymorphic **trait** plus
per-rank `impl`s plus a free-fn wrapper that delegates through the trait.
The user-facing free-fn name is preserved; rustc resolves call sites via
normal trait lookup (no rank suffix in user code).

The emitter recognizes three return-type shapes (case-3a same-shape, case-3b
bound `Self::Out`, case-3c free `Out` as trait generic). See
`cutile-macro/README.md` for the worked-out example. Optional parameter:
`method = "name"` overrides the trait method name when the user-facing
method should differ from the fn ident (e.g. `reshape_ptr` whose method is
`reshape`); `trait_name = "Name"` overrides the synthesized trait ident
when the default `PascalCase(fn_name)` would collide with a user trait of
the same name.

### `#[cuda_tile::variadic_trait(N = 6)]` and `#[cuda_tile::variadic_trait_impl()]`

For user-defined traits like `BroadcastScalar` that need rank-polymorphism:
the trait declaration desugars to the same CGA-erased rank-polymorphic form, and
its impl emits per-rank impls of that trait.

### How rank-polymorphism interacts with JIT metadata

The `#[cuda_tile::op]` and `#[cuda_tile::ty]` attributes survive expansion;
they are cloned along with each per-rank item. The JIT, however, doesn't
look at the macro-emitted forms — it works from the *original* generic
source captured by `_module_asts()`. So the macro and the JIT each
instantiate per-rank independently: the macro for rustc, the JIT for code
generation. There is no global registry connecting them.

---

## Inlined (composite) functions

Free functions with `#[cuda_tile::variadic_op]` but no `#[cuda_tile::op]` /
`#[cuda_tile::compiler_op]` have real bodies. The JIT inlines them:

```rust
#[cuda_tile::variadic_op(N = 6)]
pub fn store_tile<E: ElementType, const S: [i32; N]>(y: &mut Tensor<E, S>, result: Tile<E, S>) {
    let tile_shape: Shape<S> = y.shape();
    let tensor_token: Token = get_tensor_token(y);
    let mut y_partition: PartitionMut<E, S> =
        unsafe { make_partition_view_mut(y, tile_shape, padding::Zero, tensor_token) };
    unsafe {
        store_view_tko_mut(
            &mut y_partition,
            result,
            [0i32; N],
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        )
    };
    let new_token: Token = get_partition_token_mut(&y_partition);
    set_tensor_token(y, new_token);
}
```

This composes primitive ops (`get_tensor_token`, `make_partition_view_mut`,
`store_view_tko_mut`, etc.) without emitting a single Tile IR op of its own.

When the JIT encounters a call to `store_tile(...)`:
1. `get_cuda_tile_op_attrs` returns `None` (no `cuda_tile::op`)
2. `get_function_by_name` returns the function item
3. No `cuda_tile::compiler_op` -- falls through to `inline_function_call`
4. The body is compiled in the caller's context with the call's CGA values
   bound to the function's CGA generic, and arguments bound to parameter names

---

## Compilation dispatch summary

When the JIT encounters a function call `f(...)`:

```
get_cuda_tile_op_attrs(f) found?
  YES: compile_cuda_tile_op_call (emit Tile IR op)
  NO:  get_function_by_name(f) found?
         YES: has cuda_tile::compiler_op?
                YES: compile_compiler_op_call (custom intrinsic)
                NO:  inline_function_call (inline the body)
         NO:  error: unsupported call
```

For method calls `x.m(...)`:
```
inline_method_call: resolves impl method, compiles the method body inline
```

Method bodies typically delegate to a standalone function (e.g.,
`Tensor::store` calls `store_tile`), which then goes through the dispatch above.
