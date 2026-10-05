# `error_enum_bool_payload_addr`

Negative test for `&mut` to a `bool` enum payload (`if let Flag::On(value) =
&mut flag { flip_in_place(value) }`).

A bool payload is semantically `i1` but its enum storage byte is a canonical
`i8`; the value paths (construct/extract) zero-extend and truncate exactly at
that boundary. A raw payload address escapes the boundary, and an `i1` store
made through it would leave the byte's upper seven bits undefined for every
`i8` reader, including a niche tag sharing the byte.

Shared borrows of such payloads compile through a sound value copy (the
`shared_borrow_bool_payload` kernel in `enum_payload_addr` runs that path on
GPU). A mutable borrow cannot use a copy because writes through it would be
lost, so the compiler must reject it:

```bash
cargo oxide build error_enum_bool_payload_addr
```

Expected: the build FAILS, and the smoketest checks the log for this
exact diagnostic:

```text
a write that stays inside its function is compiled by rebuilding the enum around the new payload; a borrow that escapes into a call keeps no such rewrite and is refused here
```

together with the `canonical storage type` explanation from the
`mir.field_addr` lowering gate.
