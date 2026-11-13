---
orphan: true
---

# The cuTile Rust Book

Source for the cuTile Rust Book. From `cutile-rs/cutile-book/`, build and serve it locally:

```
make setup
source .venv/bin/activate
make livehtml
```

The local server runs at `http://127.0.0.1:8000/` by default.

For local development, use the single-version book build:

```
make livehtml
```

or, from `cutile-rs/`:

```
scripts/run_book.sh serve
```

GitHub Pages uses the versioned site build instead. To run the same build entry
point locally, run from `cutile-rs/`:

```
scripts/build_versioned_book.sh
```

The generated Tile site is written to `cutile-rs/_site/`. For the combined
CUDA Rust site, run `bash docs/build_all_docs.sh` from the monorepo root;
its output is `docs/build/html/`.

## Related Documentation

- **API Docs**: Run `cargo doc -p cutile --no-deps --open` from `cutile-rs/`
- **Examples**: See the [example guide](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-examples/README.md)
- **Reusable kernels**: See the [kernel catalog](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-kernels/README.md)
