/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts `dialect-mir` memory operations to their LLVM dialect equivalents.
//!
//! # Operations
//!
//! | MIR Operation        | LLVM Operation(s)                 | Description                  |
//! |----------------------|-----------------------------------|------------------------------|
//! | `mir.load`           | `llvm.load`                       | Load from pointer            |
//! | `mir.store`          | `llvm.store`                      | Store to pointer             |
//! | `mir.ref`            | `llvm.alloca` + `llvm.store`      | Materialize aggregate in mem |
//! | `mir.ptr_offset`     | `llvm.getelementptr`              | Pointer arithmetic           |
//! | `mir.shared_alloc`   | `llvm.global` + `llvm.addressof`  | Static shared memory         |
//! | `mir.extern_shared`  | `llvm.global` + `llvm.addressof`  | Dynamic shared memory        |
//!
//! # Shared Memory
//!
//! ## Static Shared Memory (`SharedArray<T, N, ALIGN>`)
//!
//! Each static shared memory allocation gets a unique global symbol
//! (`__shared_mem_N`, with the compiling crate's stable id woven in when the
//! lowering options carry a module disambiguator — see
//! `crate::context::module_namespaced_symbol`). Multiple allocations in the
//! same or different kernels each have their own symbol with their own size
//! and alignment.
//!
//! ## Dynamic Shared Memory (`DynamicSharedArray<T, ALIGN>`)
//!
//! Dynamic shared memory uses a symbol for each function that owns an access
//! (`__dynamic_smem_{function_name}`, crate-namespaced the same way).
//! Key characteristics:
//!
//! - **Per-owner symbols**: Each function containing an access gets an extern symbol
//! - **Pre-computed alignment**: A pre-pass combines the owner's body alignment with
//!   the strongest launch-contract marker that can reach it
//! - **Single runtime pool per launch**: The symbols refer to dynamic shared memory
//!   sized by `shared_mem_bytes` at launch
//!
//! ### PTX Output Example
//!
//! ```ptx
//! ; Kernel with 128-byte aligned dynamic shared memory
//! .extern .shared .align 128 .b8 __dynamic_smem_my_kernel[];
//!
//! ; Another kernel with 16-byte aligned (default)
//! .extern .shared .align 16 .b8 __dynamic_smem_other_kernel[];
//! ```

mod access;
mod common;
mod debug;
mod device_global;
mod extern_shared;
mod shared;
mod transfer;

pub(crate) use access::{
    convert_alloca, convert_load, convert_ptr_offset, convert_ref, convert_store,
};
pub(crate) use debug::{convert_dbg_value, convert_dbg_value_list};
pub use device_global::convert_global_alloc_dc;
pub use extern_shared::convert_extern_shared_dc;
pub use shared::convert_shared_alloc_dc;
pub(crate) use transfer::{convert_memcpy, convert_memmove};

#[cfg(test)]
mod tests;
