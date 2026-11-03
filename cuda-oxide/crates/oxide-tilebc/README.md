# oxide-tilebc

Tile IR bytecode for the tile libraries cuda-oxide ships.

A tile library lets a Tile kernel call a SIMT function. cuda-oxide is the
SIMT provider: it compiles the Rust function to LTOIR and describes it twice,
once as the declaration a Tile kernel calls (the interface) and once as the
native symbol with the fragment each parameter receives (the implementation).
`tileiras --preview-create-tile-lib` packs the three into one `.tilelib`. This
crate writes the two descriptions; nothing else in the Tile IR format.

```text
#[tile_export] descriptions ──> Library { declarations, implementations }
                                   │
          interface_bytecode(v) ───┼──> <crate>.iface.tilebc   (preview$func extern ...)
          implementation_bytecode(v)┴──> <crate>.impl.tilebc    (preview$extern_impl ...)
                                   │
          tileiras --preview-create-tile-lib + <crate>.ltoir ──> <crate>.tilelib
```

The descriptions are public-preview content, which exists from Tile IR 13.5
on and which an assembler reads only at the version it calls current, the
newest numbered version its `--list-versions` prints. `BytecodeVersion::parse`
reads that listing; pass the newest version to the two writers.

```rust
use oxide_tilebc::*;

let tile = TileType::new(vec![128], ScalarType::F32);
let decl = Declaration::new("relu", vec![
    TileArg::new("x", tile.clone(), Access::In),
    TileArg::new("relu_out", tile, Access::Out),
]);
let imp = Implementation::new("relu_impl", "relu", vec![
    FragmentArg::new("x", FragmentType::new(vec![1], ScalarType::F32, MemorySpace::Local,
        Layout::scalar_mem(), Layout::elementwise(128)), Access::In),
    FragmentArg::new("relu_out", FragmentType::new(vec![1], ScalarType::F32, MemorySpace::Local,
        Layout::contiguous_mem(1), Layout::elementwise(128)), Access::Out),
]);
let library = Library::new("relu", vec![decl], vec![imp]);
library.verify()?;
let interface = library.interface_bytecode(BytecodeVersion::new(13, 5))?;
let implementation = library.implementation_bytecode(BytecodeVersion::new(13, 5))?;
println!("{}{}", library.interface_text(), library.implementation_text());
# Ok::<(), oxide_tilebc::Error>(())
```

`verify` applies the rules the Tile compiler's own verifier applies when the
two halves meet: one contiguous run of written arguments, one fragment per
argument with the same access and element type, fragmaps that cover the tile
with the participants the synchronization clause promises, and a rank-0
buffer layout (a scalar passed by value) only for a read-only local fragment.

The wire format was derived from the Tile IR compiler sources and checked
against an assembler that reads 13.5 content: its `--preview-create-tile-lib`
packs this crate's output into a library that a Tile C++ kernel then calls.
