/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The compiled module `instantiate` returns.
//!
//! Same ownership pattern as the errors: the C record is the first field of a
//! `#[repr(C)]` box that owns every buffer and string the record points into,
//! alive until the assembler calls `free_module`.

use crate::error::view;
use crate::sys::{
    TileIRCodeBuffer, TileIRForeignModule, TileIROutputType, TileIRResolvedCallSymbol,
    TileIRStringView,
};

/// One code buffer to return: bytes, their format, and the symbols they define.
pub(crate) struct CodeBufferSpec {
    /// Diagnostic name the linker logs show.
    pub name: String,
    /// The payload bytes.
    pub data: Vec<u8>,
    /// The payload format.
    pub output_type: TileIROutputType,
    /// Every symbol the buffer makes available.
    pub symbols: Vec<String>,
}

/// The box behind every `TileIRForeignModule*` this provider returns.
#[repr(C)]
pub(crate) struct ModuleStorage {
    /// Must stay the first field: the assembler's pointer is a pointer to it.
    module: TileIRForeignModule,
    data: Vec<Vec<u8>>,
    strings: Vec<Box<str>>,
    symbol_views: Vec<Vec<TileIRStringView>>,
    buffers: Vec<TileIRCodeBuffer>,
    call_symbols: Vec<TileIRResolvedCallSymbol>,
}

impl ModuleStorage {
    /// Build the module. `call_symbols` maps every input call index to the
    /// symbol that serves it; each such symbol must be listed by one buffer.
    pub(crate) fn build(
        specs: Vec<CodeBufferSpec>,
        call_symbols: Vec<(usize, String)>,
    ) -> *mut TileIRForeignModule {
        let mut data = Vec::with_capacity(specs.len());
        let mut strings: Vec<Box<str>> = Vec::new();
        let mut symbol_views = Vec::with_capacity(specs.len());
        let mut buffers = Vec::with_capacity(specs.len());
        for spec in specs {
            let name: Box<str> = spec.name.into_boxed_str();
            let name_view = view(&name);
            strings.push(name);
            let views: Vec<TileIRStringView> = spec
                .symbols
                .into_iter()
                .map(|symbol| {
                    let symbol: Box<str> = symbol.into_boxed_str();
                    let symbol_view = view(&symbol);
                    strings.push(symbol);
                    symbol_view
                })
                .collect();
            let bytes = spec.data;
            buffers.push(TileIRCodeBuffer {
                data: bytes.as_ptr().cast(),
                size: bytes.len(),
                output_type: spec.output_type,
                symbols: if views.is_empty() {
                    std::ptr::null()
                } else {
                    views.as_ptr()
                },
                num_symbols: views.len(),
                name: name_view,
            });
            data.push(bytes);
            symbol_views.push(views);
        }
        let call_symbols: Vec<TileIRResolvedCallSymbol> = call_symbols
            .into_iter()
            .map(|(index, symbol)| {
                let symbol: Box<str> = symbol.into_boxed_str();
                let symbol_view = view(&symbol);
                strings.push(symbol);
                TileIRResolvedCallSymbol {
                    call_index: index,
                    symbol: symbol_view,
                }
            })
            .collect();
        let module = TileIRForeignModule {
            code_buffers: if buffers.is_empty() {
                std::ptr::null()
            } else {
                buffers.as_ptr()
            },
            num_code_buffers: buffers.len(),
            call_symbols: if call_symbols.is_empty() {
                std::ptr::null()
            } else {
                call_symbols.as_ptr()
            },
            num_call_symbols: call_symbols.len(),
        };
        // Every pointer above targets heap storage owned by the vectors and
        // boxes below, which the enclosing box keeps alive and unmoved.
        Box::into_raw(Box::new(Self {
            module,
            data,
            strings,
            symbol_views,
            buffers,
            call_symbols,
        }))
        .cast()
    }

    /// Release a module returned by [`ModuleStorage::build`].
    ///
    /// # Safety
    /// `module` must be a pointer `build` returned, not yet freed.
    pub(crate) unsafe fn free(module: *mut TileIRForeignModule) {
        if module.is_null() {
            return;
        }
        // SAFETY: the pointer came from `Box::into_raw` on a `ModuleStorage`
        // whose first field is the record.
        drop(unsafe { Box::from_raw(module.cast::<ModuleStorage>()) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{slice, text};

    #[test]
    fn modules_round_trip_through_the_c_record() {
        let module = ModuleStorage::build(
            vec![CodeBufferSpec {
                name: "lib.ltoir".into(),
                data: vec![1, 2, 3],
                output_type: TileIROutputType::TILEIR_OUTPUT_LTO_IR,
                symbols: vec!["relu_impl".into(), "warp_sum_impl".into()],
            }],
            vec![
                (0, "relu_impl".into()),
                (1, "relu_impl".into()),
                (2, "warp_sum_impl".into()),
            ],
        );
        // SAFETY: the pointer is live until `free`.
        unsafe {
            let record = &*module;
            let buffers = slice(record.code_buffers, record.num_code_buffers);
            assert_eq!(buffers.len(), 1);
            assert_eq!(
                slice(buffers[0].data.cast::<u8>(), buffers[0].size),
                &[1, 2, 3]
            );
            assert_eq!(text(&buffers[0].name), "lib.ltoir");
            let symbols = slice(buffers[0].symbols, buffers[0].num_symbols);
            assert_eq!(text(&symbols[1]), "warp_sum_impl");
            let calls = slice(record.call_symbols, record.num_call_symbols);
            assert_eq!(calls.len(), 3);
            assert_eq!(calls[2].call_index, 2);
            assert_eq!(text(&calls[2].symbol), "warp_sum_impl");
            ModuleStorage::free(module);
            ModuleStorage::free(std::ptr::null_mut());
        }
    }
}
