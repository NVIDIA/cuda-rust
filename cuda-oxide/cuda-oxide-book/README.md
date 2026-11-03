# The cuda-oxide Book

This directory contains the source for the cuda-oxide book. To build:

```text
make setup
source .venv/bin/activate
make livehtml
```

By default, the book will be locally hosted at `http://127.0.0.1:8000/`

## Related Documentation

- **API Docs**: Run `cargo doc --open` from the project root
- **Examples**: See `cuda-oxide/crates/rustc-codegen-cuda/examples/`
