# cuda-gdb debug-info tests

This suite exercises cuda-oxide debug information with `cuda-gdb`. For now,
these tests are intended to be run locally; they are not plugged into CI yet.

## Expected failures

Tests with a known limitation are marked with strict `pytest.xfail`
markers, so an unexpected pass is reported and the marker can be removed when
the underlying behavior is fixed. The current list is:

- `test_execution.py::test_mixed_physical_and_inlined_callstack`
  The generated debug information omits locals for an inline frame.
  Only `No locals.` for `deep_middle` is expected; incorrect values, broken
  frames, missing commands, timeouts and abnormal debugger exits still fail.

This limitation was reproduced with CUDA/cuda-gdb 13.3 on an RTX 5090.

There are no other expected failures; all tests still run.

The tests require an NVIDIA GPU and driver, a CUDA toolkit containing
`cuda-gdb`, and the Python packages in `requirements.txt`. Run them from the
repository root with:

```sh
pytest scripts/debug-tests
```

Use `-s` to show the annotated cuda-gdb transcript and `-k NAME` to select a
test. `CUDA_OXIDE_CUDA_GDB` overrides the cuda-gdb executable, while
`CUDA_OXIDE_TARGET` overrides the detected GPU architecture (for example,
`sm_90`).

The suite builds into the example's own `target` directory, regardless of an
inherited `CARGO_TARGET_DIR`, and explicitly selects the running Rust toolchain's
host target instead of an inherited Cargo build target. The debugger opens that
host binary from `target/<host>/release`.

Each debugger session runs one command file. The first command error aborts the
session and fails the test; the transcript identifies the command that failed.
This also works with cuda-gdb builds that do not include Python support.
