# compiler_features

## Compiler Feature Tests - Control Flow, Enums, Loops

Comprehensive test suite for compiler features: multi-way match, Option<T>, for loops, iterators, shared memory address casting, and 64-bit arithmetic.

## What This Example Tests

### Phase 1: Control Flow

- Multi-way match statements (integer switches)
- Binary if-else (baseline)

### Phase 2: Enums

- Option<T> (fundamental for iterators)

### Phase 3: For Loops

- Range-based: `for i in 0..8`
- Iterator-based: `for val in data.iter()`
- With enumerate: `for (i, val) in data.iter().enumerate()`
- Nested loops
- `break` and `continue`

### Phase 4: Parallel Patterns

- Polynomial evaluation
- Chunked sums
- Local averages
- Dot products
- Matrix row sums
- Range counting
- Partial products

### Additional Tests

- Shared memory address casting
- 64-bit arithmetic and shifts

## Key Concepts Demonstrated

### Multi-way Match

```rust
#[kernel]
pub fn test_multiway_match_u32(val: u32, mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        let result = match val {
            0 => 10u32,
            1 => 20u32,
            2 => 30u32,
            _ => 99u32,  // Default case
        };
        *out_elem = result;
    }
}
```

### Option<T> (Enum)

```rust
#[kernel]
pub fn test_option(val: u32, mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        let maybe: Option<u32> = if val > 0 { Some(val) } else { None };
        let result = match maybe {
            Some(x) => x,
            None => 0,
        };
        *out_elem = result;
    }
}
```

### For Loop with Range

```rust
#[kernel]
pub fn test_for_loop_sum(mut out: DisjointSlice<u32>) {
    let mut sum: u32 = 0;
    for i in 0u32..8 {  // 0+1+2+3+4+5+6+7 = 28
        sum += i;
    }
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = sum;
    }
}
```

### Iterator with Enumerate

```rust
#[kernel]
pub fn test_enumerate(data: &[u32], mut out: DisjointSlice<u32>) {
    let mut sum: u32 = 0;
    for (i, val) in data.iter().enumerate() {
        sum += (i as u32) * (*val);  // Weighted sum
    }
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = sum;
    }
}
```

### Break and Continue

```rust
// Break: sum 0+1+2+3+4 = 10
for i in 0u32..100 {
    if i >= 5 { break; }
    sum += i;
}

// Continue: sum odd numbers 1+3+5+7 = 16
for i in 0u32..8 {
    if i % 2 == 0 { continue; }
    sum += i;
}
```

### 64-bit Descriptor Building

```rust
#[kernel]
pub fn test_u64_descriptor_build(addr: u64, ld_bytes: u32, stride_bytes: u32, mut out: DisjointSlice<u64>) {
    let addr_enc = (addr >> 4) & 0x3FFF;
    let ld_enc = ((ld_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    let fixed_bit: u64 = 1u64 << 46;  // Bit 46 set

    let desc = addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit;
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = desc;
    }
}
```

## Build and Run

```bash
cargo oxide run compiler_features
```

## Expected Output

```text
=== Compiler Features Test (Unified) ===

Testing: baseline_while_loop
  ✓ Result: 28 (expected 28)
Testing: baseline_binary_match
  ✓ flag=true: 100 (expected 100)
  ✓ flag=false: 0 (expected 0)
Testing: baseline_vecadd
  ✓ Result: [11.0, 22.0, 33.0, 44.0]
Testing: test_multiway_match_u32
  ✓ val=0: 10 (expected 10)
  ✓ val=1: 20 (expected 20)
  ✓ val=2: 30 (expected 30)
  ✓ val=3: 99 (expected 99)
Testing: test_option
  ✓ val=0: 0 (expected 0)
  ✓ val=42: 42 (expected 42)
Testing: test_for_loop_sum
  ✓ Result: 28 (expected 28)
Testing: test_iter_sum
  ✓ Result: 15 (expected 15)
Testing: test_enumerate
  ✓ Result: 200 (expected 200)
Testing: test_for_loop_break
  ✓ Result: 10 (expected 10)
Testing: test_for_loop_continue
  ✓ Result: 16 (expected 16)
Testing: test_nested_for_loops
  ✓ Result: 36 (expected 36)
Testing: test_u64_shift_by_32
  ✓ Result: 0x0000000800000000 (expected 0x0000000800000000)
Testing: test_u64_shift_by_46
  ✓ Result: 0x0000400000000000 (expected 0x0000400000000000)
Testing: parallel_polynomial_eval
  ✓ Result: 255.0 (expected 255.0)
... (more tests)

=== ALL TESTS PASSED ✓ ===
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+

## Test Categories

| Category     | Tests                          | Purpose                |
|--------------|--------------------------------|------------------------|
| Baseline     | while, if-else, vecadd         | Verify basic operations|
| Control Flow | match, Option                  | Multi-way branching    |
| Loops        | for, break, continue, nested   | Loop lowering to PTX   |
| Iterators    | iter(), enumerate()            | Iterator desugaring    |
| 64-bit       | shifts, descriptor build       | Wide arithmetic        |
| SMEM         | address casting                | Pointer manipulation   |
| Parallel     | polynomial, dot product, etc.  | Real workloads         |

## How For Loops Lower to PTX

Rust `for` loop:

```rust
for i in 0u32..8 {
    sum += i;
}
```

Equivalent while:

```rust
let mut iter = (0u32..8).into_iter();
loop {
    match iter.next() {
        Some(i) => sum += i,
        None => break,
    }
}
```

PTX:

```ptx
loop:
    setp.lt.u32 %p1, %r_i, 8;
    @!%p1 bra done;
    add.u32 %r_sum, %r_sum, %r_i;
    add.u32 %r_i, %r_i, 1;
    bra loop;
done:
```

## Potential Issues

| Issue                | Symptom               | Cause                             |
|----------------------|-----------------------|-----------------------------------|
| Wrong match result   | Unexpected value      | Switch lowering bug               |
| Loop hang            | Kernel timeout        | Iterator implementation issue     |
| Wrong 64-bit result  | Truncated value       | 32-bit operation used             |
| SMEM address wrong   | Large generic pointer | Missing cvta.shared optimization  |

## Shared Memory Address Tests

Tests two approaches to getting shared memory addresses:

```rust
// Direct cast (should give small address)
let addr = &raw const SMEM as u64;

// Via pointer cast (may give generic address)
let addr = &raw const SMEM as *const u8 as u64;
```

The direct cast should give a small (~0-64KB) shared address.
The pointer cast may trigger cvta (convert address) producing a large generic address.
