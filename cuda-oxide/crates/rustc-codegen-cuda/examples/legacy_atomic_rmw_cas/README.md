# legacy_atomic_rmw_cas

Regression coverage for integer atomic read-modify-write and compare-exchange
operations on the legacy LLVM 7 NVVM path.

The fixture covers both lanes of the legacy legalizer: the native LLVM subset
(device-scope, fully monotonic integer RMW/CAS kept as bare
`atomicrmw`/`cmpxchg`) and the scoped-PTX rewrite lane (block/system scopes
and ordered success or failure orderings lowered to inline
`atom.{sem}.{scope}` PTX):

- `DeviceAtomicU32` and `DeviceAtomicU64` native legacy RMW/CAS coverage;
- `BlockAtomicU32` / `BlockAtomicU64` and `SystemAtomicU32` /
  `SystemAtomicU64` scoped integer RMW coverage;
- strong source RMW ordering through the existing fence-splitting path;
- native-lane `compare_exchange` with `Relaxed` success and failure ordering,
  the only pair libNVVM lowers faithfully as a bare `cmpxchg`;
- block-, system-, and device-scoped compare-exchange paths that require inline
  PTX legalization;
- `compare_exchange` with `AcqRel` success ordering and `Acquire` failure
  ordering, covering the failure-ordering information legacy libNVVM otherwise
  accepts but ignores;
- device-scope `compare_exchange` with `AcqRel` success and `Relaxed` failure
  ordering, covering ordered success orderings that would otherwise be
  silently weakened to an unordered `atom.cas` by libNVVM;
- successful and failed CAS old-value semantics.

The legacy legalizer remains fail-closed for forms whose semantics are not
covered by the supported legacy mapping, including integer widths other than
32 and 64 bits, unsupported address spaces, unknown synchronization scopes,
invalid compare-exchange ordering pairs, and PTX-rewritten scoped or ordered
forms on targets older than `sm_70`.
