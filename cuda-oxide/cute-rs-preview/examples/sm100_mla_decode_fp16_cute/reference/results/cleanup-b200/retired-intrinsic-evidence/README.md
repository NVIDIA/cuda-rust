# Historical intrinsic probe evidence

These files preserve the original LLVM and libNVVM probe records byte-for-byte.
They include the unused hinted mbarrier wait routes formerly assigned `i1034`
and `i1035`, which were removed during the attention cleanup. The active
intrinsic evidence files retain only supported routes.

These are compiler/PTX probe records, not runtime kernel validation. The
retired routes were not used by either attention kernel.

[The prior ledger records](unpublished-abi-records.toml) preserve the exact
experimental identities and signatures. Both IDs existed only in uncommitted
work, so neither was part of the published ABI. They were omitted from the
active ledger rather than introduced as tombstones: ABI history requires new
IDs to begin active. Every previously committed ABI entry remains unchanged;
the ABI-history policy was not relaxed.
