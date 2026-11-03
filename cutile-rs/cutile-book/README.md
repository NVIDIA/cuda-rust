---
orphan: true
---

# The cuTile Rust Book

Source for the cuTile Rust Book. To build and serve it locally:

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

or, from the repository root:

```
scripts/run_book.sh serve
```

GitHub Pages uses the versioned site build instead. To run the same build entry
point locally, run from the repository root:

```
scripts/build_versioned_book.sh
```

The generated GitHub Pages site is written to `_site/`.

## Related Documentation

- **API Docs**: Run `cargo doc --open` from the project root
- **Examples**: See the `cutile-examples/` directory
