# cuda_module_nested

Demonstrates `#[cuda_module]` collecting kernels from nested inline modules.
Each Rust namespace gets a launcher view for the kernels defined there:

```text
kernels::LoadedModule
├── kernels::init::LoadedModule
├── kernels::scale::LoadedModule
├── kernels::offset::LoadedModule
└── kernels::post::LoadedModule
    └── kernels::post::double::LoadedModule
```

Build a child view from its immediate parent. The views share the same loaded
CUDA module and generic-function cache:

```rust,ignore
let root = kernels::load(&ctx)?;
let scale = kernels::scale::LoadedModule::from_parent(&root)?;
scale.scale_by(&stream, config, &input, &mut output)?;
```

Only inline module bodies are visible to an attribute macro. `mod child;` and
`include!` items are left untouched, but kernels behind those boundaries do
not receive generated launchers. Kernel function names must also remain unique
across the tree because PTX entry symbols are not namespace-qualified yet.

## Run

```
cargo oxide run cuda_module_nested
```

Expected output:

```
✓ SUCCESS: root-loaded nested inline kernels all ran
```
