# iket-lower

`iket-lower` converts semantic `dialect-iket` operations into an encoding
accepted by a selected IKET runtime compatibility profile.

The compatibility profile fixes three compiler/runtime boundary facts in
tests:

- NativeDump carries 4 bytes per no-payload event and supports 30 user event
  IDs;
- ExtendedNativeDump carries 8 bytes per no-payload event and supports 4,031
  user event IDs after IKET's reserved-ID offset;
- event metadata has a 32-byte inline name field (31 UTF-8 bytes plus NUL),
  while longer names use the CUDA C++ `h<16hex>` placeholder and
  `__iket_string_decl_*_str` table.

The lowering method policy is `auto` by default. Explicit `native` and
`extended` modes exist for controlled experiments and stable diagnostics, but
do not appear in the semantic dialect.

Planning walks the whole IKET operation tree before rewriting any site. The
method decision therefore counts unique user event names across nested
functions and regions, excludes the reserved range-pop event, and applies one
method consistently to the compiler root.
