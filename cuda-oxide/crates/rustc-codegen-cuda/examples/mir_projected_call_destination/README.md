# mir_projected_call_destination

An intrinsic call whose destination carries a projection, in the three shapes
that reach code generation — plus the two emitters that build their result
value themselves and so carry their own store sites.

## What it checks

rustc lowers an ordinary call with a projected destination into a call to a
temporary followed by a store:

```text
_9 = f(const 10_i32) -> [return: bb3, unwind continue]
(*_8) = move _9
```

An intrinsic keeps its destination instead, so the projection survives to the
importer. Surface Rust cannot write that, which is why all the bodies here
are `#[custom_mir]`:

| body | destination | shape |
|---|---|---|
| `through_deref` | `(*p) = bswap(x)` | dereferenced raw pointer |
| `through_field` | `RET.1 = bswap(x)` | field of a `(f64, u8)` tuple |
| `through_index` | `RET[i] = bswap(x)` | element of a `[i32; 3]` |
| `through_field_float` | `RET.1 = sqrtf32(x)` | field of a `(u64, f32)` tuple, via the float-math placeholder emitter |
| `through_index_float` | `RET[i] = sqrtf32(x)` | element of a `[f32; 3]`, via the float-math placeholder emitter |
| `through_field_fn` | `RET.1 = double_it(x)` | field of a `(u64, u32)` tuple, via the plain function-call path |
| `through_deref_sincos` | `(*p) = sincosf(x)` | dereferenced raw pointer, via the sincos tuple-pack emitter |

Each result has to land at the address the projection names. The device and
the host run the same bodies and their results are compared, so a store aimed
at the local instead of the place shows up as a disagreement rather than as a
value nobody checks. The array case leaves its other two elements at `11` and
`33`, which a store over the whole array would not.

The last two rows cover the emitters that do not return a plain call result:
the float-math placeholder types its result from the destination, so it must
use the projected place's type rather than the whole local's, and `sincos`
packs a `(sin, cos)` tuple and must store it through the projection. Both
use inputs (`sqrt(2)`, angle `0`) whose results are bit-identical on host
and device, keeping the comparison exact.

## Running it

```bash
cargo oxide run mir_projected_call_destination
```

Before the importer took the projection into account, the field case asked for
a cast from a byte to `{ double, i8, [7 x i8] }` and the deref case asked for
the width of a pointer, so this example did not compile.
