# Optimized B200 decode evidence

See the [performance analysis](../../measurement-notes.md)
for interpretation, conditions and reproduction commands.

- `short-*.json`: 10 warmups, 100 invocations, 5 windows in each timing order.
- `sustained-*.json`: 2,000 warmups, 301 invocations, 7 windows in each order.
- `summary.json`: all-window aggregation, exact source/executable hashes,
  sanitizer input provenance and selected instrumented profile counters.
- `validation.json`: all nine independent-oracle/CuTeDSL correctness cases.
- `memcheck-*.txt`, `synccheck-*.txt`, `racecheck-tail.txt`: final executable logs.
- `compiler-checks.json`: compiler/API checks and native-store fixture provenance.
- `sustained-trace.json`: instrumented kernel-active timings; not benchmark data.
- `prefill-regression/`: shared-compiler regression checks for the preserved
  prefill example.

The initial benchmark reports remain in the parent directory. Profiling used
`ncu --set full --cache-control none --clock-control none`, selecting the main
and reduction kernels after three complete invocations. Nsight Systems used
`--trace=cuda --sample=none --cpuctxsw=none`. Large binary profiler traces are
not checked in; the exported summary preserves the counters used in the report.
