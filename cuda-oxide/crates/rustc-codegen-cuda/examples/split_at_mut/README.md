# split_at_mut

Regression test for a niche-`Option` whose payload carries two pointers —
`<[T]>::split_at_mut_checked`, which returns `Option<(&mut [T], &mut [T])>`.

That payload is two fat slice pointers `{ptr, len, ptr, len}`, with the `None`
niche in the first data pointer. The enum slot map used to back only one pointer
(the niche carrier), so the second slice pointer had no provenance-preserving
`ptr` slot and lowering failed closed:

```
enum slot map: `...` has overlapping pointer and non-identical storage at byte N;
refusing to erase LLVM pointer provenance
```

The fix gives each extra pointer leaf its own `ptr` slot, so both slice pointers
survive the memory round-trip with provenance intact.

## Run

```
cargo oxide run split_at_mut
```

Thread 0 splits a buffer at `k` via `split_at_mut_checked`, bumps the left half
by 1 and the right half by 100 through the two returned slices, and the host
verifies both halves — a dropped or provenance-stripped second pointer yields a
wrong result, not merely a codegen abort.
