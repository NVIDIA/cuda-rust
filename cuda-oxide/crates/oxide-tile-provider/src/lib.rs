/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The runtime inside a cuda-oxide tile library that is a linkage provider.
//!
//! A tile library can be one of two things. The static form packs the
//! implementation descriptions and the LTOIR into a `.tilelib` ahead of
//! time. The provider form is a shared object the Tile IR assembler loads
//! while it compiles a Tile kernel; the assembler asks it which
//! implementations exist and then for the device code behind the ones it
//! selected. This crate is that shared object's runtime: it holds the
//! `#[tile_export]` descriptions and the device code (the [`Payload`]) and
//! answers the assembler's callbacks ([`entry`] installs them).
//!
//! ```text
//! #[tile_export] crate --cargo oxide--> descriptions + device code
//!                                            |
//!                              define_provider! embeds both
//!                                            v
//!                       lib<crate>_provider.so  (this runtime + payload)
//!                                            |
//!   tileiras -l <lib>.tilelib: create -> provide_impls -> filter_and_rank -> instantiate
//! ```
//!
//! The shared object is written by `cargo oxide tilelib --provider`; by hand
//! it is a `cdylib` crate with one line:
//!
//! ```ignore
//! oxide_tile_provider::define_provider! {
//!     name: "relu",
//!     descriptions: include_str!("relu.tileexports.json"),
//!     bytecode_version: (13, 5),
//!     ltoir: include_bytes!("relu.ltoir"),
//!     arch: "sm_120",
//! }
//! ```
//!
//! Set `OXIDE_TILE_PROVIDER_LOG=1` in the assembler's environment to see
//! every callback described on stderr.

#![deny(unsafe_op_in_unsafe_fn)]

mod error;
mod ffi;
mod module;
pub mod payload;
mod view;

pub use ffi::{LOG_ENV, entry};
pub use oxide_tile_provider_sys as sys;
pub use oxide_tilebc::BytecodeVersion;
pub use payload::{DeviceCode, Payload, PayloadError};

/// Define `tileir_create_linkage_provider` for a `cdylib` crate from embedded
/// descriptions and LTOIR. `bytecode_version` is the version the assembler
/// this library is built for calls current, the one `cargo oxide tilelib`
/// read from `tileiras --list-versions`.
#[macro_export]
macro_rules! define_provider {
    (
        name: $name:expr,
        descriptions: $descriptions:expr,
        bytecode_version: ($major:expr, $minor:expr),
        ltoir: $ltoir:expr,
        arch: $arch:expr $(,)?
    ) => {
        /// The entry point the Tile IR assembler looks up after `dlopen`.
        ///
        /// # Safety
        /// Called by the assembler with a writable `out_instance`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn tileir_create_linkage_provider(
            compiler_api_version: $crate::sys::TileIRLinkageProviderAPIVersion,
            out_instance: *mut $crate::sys::TileIRLinkageProviderInstance,
        ) -> $crate::sys::TileIRLPResult {
            // SAFETY: the assembler passes a writable instance slot.
            unsafe {
                $crate::entry(compiler_api_version, out_instance, || {
                    $crate::Payload::from_parts(
                        $name,
                        $descriptions,
                        $crate::BytecodeVersion::new($major, $minor),
                        $crate::DeviceCode::Ltoir {
                            bytes: ($ltoir).to_vec(),
                            arch: Some(String::from($arch)),
                        },
                    )
                })
            }
        }
    };
}
