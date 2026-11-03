# device_panic_trap

A `#[kernel]` that reaches a panic **carrying a message** must compile, with the
panic path lowered to a device trap.

There is no panic runtime and no `core::fmt` machinery on the GPU, so the
diverging call into `core::panicking` is dropped and replaced by `nvvm.trap`
(`trap;` in PTX) followed by `unreachable`. A thread that reaches it aborts the
kernel; the message is discarded. The statements that would have built the
message are dead once the call is gone, so they are never translated — which is
what makes the kernel compilable, since a materialized `&str` constant has no
device lowering.

## Trigger

A `panic!` guarded by a value read from device memory, so the panic path is
data-dependent and survives `-C opt-level=3`:

```rust
if value > LIMIT {
    panic!("input exceeds the supported range");
}
```

Before the fix this failed the build with:

```
error: device code reaches a panic that builds a message string; panic message
       formatting is not supported on the GPU
```

The same lowering covers every panic whose message is a `&'static str`
materialized on the panic path — `panic!("literal")`, and core functions that
end in one, such as `<[T]>::split_at_mut`'s `panic!("mid > len")`.

## What the example checks

1. `in_range` — every thread scales an accepted input; the outputs must be
   exact. This is the "the panic path did not break the normal path" half.
2. `out_of_range` — thread 0 gets an input the kernel rejects. It must trap,
   which the driver surfaces as a launch failure. A launch that instead
   succeeds means the panic path fell through instead of trapping.

The trapping launch runs last: a trap poisons the CUDA context for every launch
after it.

## Running

```bash
cargo oxide run device_panic_trap
```

Expected output ends with `SUCCESS`. To confirm the lowering directly:

```bash
grep -c 'trap;' device_panic_trap.ptx
```
