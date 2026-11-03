# oxide-tile-export

`oxide-tile-export` defines the description of a Rust device function that a
cuTile program calls through a tile library.

Three parts of cuda-oxide exchange these descriptions:

- `#[tile_export]` in `cuda-macros` builds one `TileExportDesc` per exported
  function at macro-expansion time and embeds its JSON in a host-only
  metadata function.
- The `rustc-codegen-cuda` backend reads the JSON back out of that
  function's MIR, validates it, compiles the exported wrapper as a device
  root, and writes every description of a crate to
  `<crate>.tileexports.json` next to the device artifact.
- `cargo oxide` turns that file into the Tile IR interface and implementation
  modules of a tile library.

The crate owns the vocabulary (element types, access modes, synchronization
levels), the validation rules, and the mapping from a description to the
native SIMT ABI that the exported function must have. It performs no file IO
and knows nothing about rustc or MIR.
