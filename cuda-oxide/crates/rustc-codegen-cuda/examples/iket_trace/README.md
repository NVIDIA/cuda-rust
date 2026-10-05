# IKET trace example

This example shows cuda-oxide's semantic In-Kernel Event Tracing annotations:

- a point event;
- a token-paired range;
- a `u32` payload; and
- an event name longer than `NativeDump`'s 31-byte inline-name limit.

Build the traced executable:

```bash
cargo oxide build iket_trace --arch sm_120
```

When IKET operations are present, `CUDA_OXIDE_IKET=auto` is the default. It
selects `NativeDump` for at most 30 unique event names and
`ExtendedNativeDump` above that limit. The method can also be selected
explicitly:

```bash
CUDA_OXIDE_IKET=native cargo oxide build iket_trace --arch sm_120
CUDA_OXIDE_IKET=extended cargo oxide build iket_trace --arch sm_120
CUDA_OXIDE_IKET=off cargo oxide build iket_trace --arch sm_120
```

The generated CUDA module contains IKET-compatible placeholder instructions
and metadata. A normal launch still computes and verifies the vector result.

Capturing a profile currently requires the public `nvidia-cutlass-dsl`
package, which provides `run-iket`. This is a profiler-time dependency; it is
not linked into the cuda-oxide application. Install the version used to
qualify this example, then wrap the built executable:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python nvidia-cutlass-dsl==4.7.0

.venv/bin/run-iket --output-dir /tmp/iket-trace --clobber \
  profile --postprocess all -- \
  "$PWD/cuda-oxide/crates/rustc-codegen-cuda/examples/iket_trace/target/release/iket_trace"
```

The default profile flow runs a tracker pass to choose the timestamp and
buffer budgets, then runs the application again to capture the trace. The
output directory contains the decoded JSON and Perfetto trace.
