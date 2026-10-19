# cuTile Rust
This is the main user-facing crate in the workspace.
It provides:

- the kernel authoring DSL exposed through `#[cutile::module]`
- host-side tensor API in `cutile::api`
- built-in kernels and launch utilities layered on top of `cuda-async`

For full workspace setup, examples, and environment requirements, start with the repository [README](../README.md).

# Tests
- Run the crate test suite with `cargo test -p cutile`.
- Run a specific test and see its output with `cargo test -p cutile --test span_source_location -- --no-capture`.
