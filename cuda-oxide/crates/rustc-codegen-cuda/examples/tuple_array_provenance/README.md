# `tuple_array_provenance`

Positive smoke test: an array of tuples whose elements hold thin references to
device statics. Each pointer field is materialized via `MirGlobalAllocOp`.

This covers aggregate **const** values only. Device-global *initializer*
relocations remain unsupported.

```bash
cargo oxide run tuple_array_provenance
```
