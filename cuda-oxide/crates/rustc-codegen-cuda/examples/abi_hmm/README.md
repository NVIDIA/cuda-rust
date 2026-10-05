# Unified ABI + HMM Test Example

This example demonstrates **Heterogeneous Memory Management (HMM)** with **unified struct ABI** - the GPU directly reads and writes host memory without explicit `cudaMemcpy` operations.

## What This Tests

1. **HMM (GPU Direct Host Access)**: GPU kernel accesses host stack memory directly
2. **Unified Struct Layout**: Device uses exact same struct layout as host (field ordering + alignment)
3. **Dynamic Layout Matching**: Works without `#[repr(C)]` - compiler matches rustc's layout
4. **Move Closures**: Closure captures by value work without `instantiate!` macro
5. **Reference Captures**: Non-move closures capture by reference - GPU accesses host memory via HMM

## The Challenge: Field Reordering & Alignment

Rust's default `#[repr(Rust)]` allows the compiler to **reorder struct fields** for better packing:

```rust
// User writes:
struct Extreme {
    a: u8,      // declared first
    b: i128,    // declared second
}

// rustc may optimize to:
//   Memory layout: [b @ offset 0][a @ offset 16]
//   (b first to avoid 15 bytes of padding)
```

If the GPU used **declaration order** while the host used **memory order**, field access would be corrupted.

## How cuda-oxide Solves This

### Step 1: Query rustc's Actual Layout

```text
rust_ty.layout()?.shape().fields
  → FieldsShape::Arbitrary { offsets: [16, 0] }
    (field a at offset 16, field b at offset 0)

rust_ty.layout()?.shape().size
  → 32 bytes total
```

### Step 2: Store Layout in MIR Type

```text
mir.struct <Extreme, [a, b], [u8, i128], [1, 0], [16, 0], 32>
            ^^^^^^   ^^^^^^  ^^^^^^^^^^  ^^^^^^  ^^^^^^^  ^^
            name     fields  types       order   offsets  size
```

### Step 3: Build LLVM Struct with Explicit Padding

```text
LLVM struct = { i128, i8, [15 x i8] }
               ├────┤ ├──┤ ├───────┤
               b@0   a@16  padding to 32

The [15 x i8] padding ensures the struct is exactly 32 bytes,
matching rustc's computed size.
```

## Example: Extreme Struct

```rust
// No #[repr(C)] needed!
pub struct Extreme {
    pub a: u8,
    pub b: i128,
}
```

### Host Layout (rustc computes)

```text
┌────────────────────────────────────────────────────────────┐
│ Offset 0-15:  field b (i128, 16 bytes)                     │
│ Offset 16:    field a (u8, 1 byte)                         │
│ Offset 17-31: trailing padding (15 bytes, align to 16)     │
│ Total: 32 bytes                                            │
└────────────────────────────────────────────────────────────┘
```

### Device Layout (cuda-oxide builds to match)

```text
LLVM IR: %Extreme = type { i128, i8, [15 x i8] }

Field access mapping:
  MIR: (*p).b  →  extract field 1 (declaration index)
  LLVM: extract field 0 (memory index)  →  correct!
```

## Test Cases

### Test 1: HMM Direct Host Memory Access

```rust
let mut data = Extreme { a: b'X', b: 42 };
let ptr = &mut data as *mut Extreme;  // HOST stack pointer!

// SAFETY: cfg launches one thread, and both HMM stack values remain alive
// until the stream is synchronized.
unsafe { module.modify_extreme_hmm(stream.as_ref(), cfg, ptr, 2i128, &mut device_ran) }?;

assert_eq!(data.b, 84);   // GPU wrote to host memory ✓
assert_eq!(device_ran, 1); // Kernel executed on GPU ✓
```

### Test 2: HMM + Move Closure

```rust
let scale: i128 = 3;
// move closure: captures `scale` BY VALUE
let closure = move |p: *mut Extreme| unsafe { (*p).b *= scale };

// SAFETY: cfg launches one thread, and the HMM values outlive synchronization.
unsafe { module.with_closure_hmm(stream.as_ref(), cfg, ptr, &mut device_ran, closure) }?;

assert_eq!(data.b, 300);  // 100 * 3 = 300 ✓
```

### Test 3: HMM + Single Reference Capture

```rust
let scale: i128 = 4;
// Non-move closure: captures `scale` BY REFERENCE
// The closure struct contains { scale: &i128 } - a pointer to host memory!
// GPU accesses this host address via HMM

// SAFETY: cfg launches one thread, and the HMM values and captured reference
// remain alive until the stream is synchronized.
unsafe {
    module.with_closure_hmm(
        stream.as_ref(),
        cfg,
        ptr,
        &mut device_ran,
        |p: *mut Extreme| unsafe {
            (*p).b *= scale  // GPU reads &scale via HMM
        },
    )
}?;

assert_eq!(data.b, 200);  // 50 * 4 = 200 ✓
```

This matches C++ capture-by-reference: `[&](auto* p) { p->b *= scale; }`

### Test 4: HMM + Multiple Reference Captures

```rust
let scale: i128 = 5;
let offset: i128 = 7;
// Closure captures BOTH by reference: { scale: &i128, offset: &i128 }

// SAFETY: cfg launches one thread, and the HMM values and captured references
// remain alive until the stream is synchronized.
unsafe {
    module.with_closure_hmm(
        stream.as_ref(),
        cfg,
        ptr,
        &mut device_ran,
        |p: *mut Extreme| unsafe {
            (*p).b = (*p).b * scale + offset  // GPU reads both via HMM
        },
    )
}?;

assert_eq!(data.b, 57);  // 10 * 5 + 7 = 57 ✓
```

## How Reference Captures Work

For **move closures**, the typed launch method passes each capture by value:

```rust
move |p| (*p).b *= scale
// Launch passes the i128 VALUE
```

For **non-move closures**, the typed launch method passes the **address** of each capture:

```rust
|p| (*p).b *= scale
// Launch marshalling stores:
//   let __ref_capture = &scale as *const _;
// which passes the POINTER to host memory
```

The GPU then accesses this host pointer via HMM to read the value.

## Requirements

| Requirement   | Minimum                                           |
|---------------|---------------------------------------------------|
| GPU           | Turing or newer (RTX 20xx+)                       |
| Linux Kernel  | 6.1.24+                                           |
| CUDA          | 12.2+                                             |
| HMM Support   | `nvidia-smi -q \| grep Addressing` shows "HMM"    |

## Build and Run

```bash
# From workspace root
cargo oxide run abi_hmm

# Show full compilation pipeline
cargo oxide pipeline abi_hmm
```

## Why No `#[repr(C)]`?

Traditional CUDA/C++ requires consistent struct layout because:
- C++ compilers use declaration order
- NVCC uses the same rules for host and device

Rust is different:
- `#[repr(Rust)]` (default) allows field reordering
- Host and device might use different compilers

**cuda-oxide's solution**: Query rustc's actual layout and match it exactly with explicit padding. This means:

✅ No `#[repr(C)]` annotation needed
✅ Works with any struct layout rustc chooses
✅ Works on any host architecture (x86_64, aarch64, etc.)
✅ Independent of LLVM's datalayout string

## Technical Details

### Files Involved

| Component     | File                                      | Role                                           |
|---------------|-------------------------------------------|------------------------------------------------|
| Layout query  | `mir-importer/src/translator/types.rs`    | Query `fields_by_offset_order()` and `offsets` |
| MIR type      | `dialect-mir/src/types.rs`                | Store offsets in `MirStructType`               |
| LLVM lowering | `mir-lower/src/convert/types/struct_layout.rs` | Build struct with explicit padding        |
| Field access  | `mir-lower/src/convert/ops/aggregate/fields.rs` | Map declaration → memory index           |

### Pipeline Output

```text
=== dialect-mir module ===
mir.struct <Extreme,[a, b],[ui8, si128],[1, 0],[16, 0],32>
                                        ^^^^^^ ^^^^^^^  ^^
                                        order  offsets  size

=== LLVM dialect struct ===
%Extreme = type { i128, i8, [15 x i8] }
                  ^     ^   ^^^^^^^^^
                  b@0  a@16 padding
```

## See Also

- [rustc-codegen-cuda README](../../README.md) - Unified compilation overview
