# cuda-bindings

Generated raw Rust FFI bindings to the CUDA toolkit libraries used by this workspace.

Use `cuda-core` for wrappers around these raw bindings.

# Notes

- The bindings are generated at build time from a CUDA 13.0+ toolkit.
- The toolkit root is the first set variable among `CUDA_TOOLKIT_PATH` and
  `CUDA_HOME`; when neither is set, the build searches the standard install
  locations (on Linux `/usr/local/cuda-13.3`, `/usr/local/cuda-13.2`,
  `/usr/local/cuda-13`, `/usr/local/cuda`; on Windows
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` and `v13.2`).
- Within the toolkit, both the standard `include/` layout and the
  redistributable `targets/<dir>/include/` layouts (Jetson/Tegra, sbsa,
  cross-builds, extracted redistributables) are probed for `cuda.h`.
  `CUDA_TOOLKIT_TARGET_DIR` names one `targets/` directory by hand, like
  nvcc's `-target-dir` flag; when set, that tree is the only candidate
  probed.
- Set `CUTILE_SETUP_DIAGNOSTICS=1` to print CUDA toolkit discovery decisions.
