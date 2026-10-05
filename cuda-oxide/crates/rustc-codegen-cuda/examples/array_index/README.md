# Array Index Operations Test

This example tests all combinations of array index operations to validate the implementation
of array reads and writes with both constant and runtime indices.

## Test Matrix

| Operation | Index Type | Status   | Implementation                                      |
|-----------|------------|----------|-----------------------------------------------------|
| Read      | Constant   | ✓ PASS   | `MirExtractFieldOp` or memory GEP+load              |
| Read      | Runtime    | ✓ PASS   | Memory GEP+load via `MirArrayElementAddrOp`         |
| Write     | Constant   | ✓ PASS   | Memory GEP+store via `MirArrayElementAddrOp`        |
| Write     | Runtime    | ✓ PASS   | Memory GEP+store via `MirArrayElementAddrOp`        |

## Running

```bash
# Compile and run
cargo oxide run array_index
```

## Test Sections

### Section 1: Constant Index Reads (PASS)

- `test_const_index_read` - Read multiple constant indices, sum them
- `test_const_index_read_expr` - Use constant index reads in expressions

### Section 2: Runtime Index Reads (PASS)

- `test_runtime_index_read` - Read at index from kernel parameter
- `test_runtime_index_read_loop` - Read in a loop (sum array elements)
- `test_mixed_read` - Mix constant and runtime index reads

### Section 3: Constant Index Writes (PASS)

- `test_const_index_write` - Write to `arr[0]`, `arr[1]`, etc.
- `test_const_index_write_init` - Initialize array element by element

### Section 4: Runtime Index Writes (PASS)

- `test_runtime_index_write` - Write at runtime index
- `test_runtime_index_write_loop` - Write in a loop (THE MATHDX PATTERN)
- `test_copy_to_local_array` - Copy from slice to local array

### Section 5: Complex Patterns (PASS)

- `test_read_modify_write` - `arr[i] = arr[i] * 2`
- `test_swap_elements` - Swap two elements
- `test_accumulate` - Accumulate into buckets

## Expected Output

```text
=== Array Index Operations Test ===

Device: NVIDIA GeForce RTX 5090

=== SECTION 1: Constant Index Reads ===
test_const_index_read: PASS (result = 100)
test_const_index_read_expr: PASS (result = 14)

=== SECTION 2: Runtime Index Reads ===
test_runtime_index_read: PASS (result = 300)
test_runtime_index_read_loop: PASS (result = 10)
test_mixed_read: PASS (result = 40)

=== SECTION 3: Constant Index Writes ===
test_const_index_write: PASS (result = 26)

=== SECTION 4: Runtime Index Writes ===
test_runtime_index_write_loop: PASS (result = 280)
test_copy_to_local_array: PASS (result = 1000)

=== SECTION 5: Complex Patterns ===
test_read_modify_write: PASS (result = 20)

=== Test Complete ===
```

## Implementation Details

All mutable arrays are **eagerly promoted to memory** at their definition point. This ensures:

1. **Correctness in loops**: No PHI re-initialization issues
2. **Consistent access**: Both reads and writes use memory (GEP + load/store)
3. **Alloca domination**: Allocas placed in entry block, visible from all uses
