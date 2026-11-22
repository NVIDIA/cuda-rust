/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provider-owned structured errors.
//!
//! The assembler keeps the pointer we hand out until it calls `free_error`.
//! The C record is the first field of a `#[repr(C)]` box that also owns the
//! strings it points into, so the pointer the assembler holds converts back
//! to the box without a registry.

use crate::sys::{
    TileIRLPError, TileIRLPErrorKind, TileIRLPResult, TileIRLPUnresolvedCall,
    TileIRLPUnresolvedCallsError, TileIRStringView,
};

/// The box behind every `TileIRLPError*` this provider returns.
#[repr(C)]
pub(crate) struct ErrorStorage {
    /// Must stay the first field: the assembler's pointer is a pointer to it.
    error: TileIRLPError,
    message: Box<str>,
    details: Box<str>,
    reasons: Vec<Box<str>>,
    unresolved: Vec<TileIRLPUnresolvedCall>,
}

pub(crate) fn view(text: &str) -> TileIRStringView {
    TileIRStringView {
        data: if text.is_empty() {
            std::ptr::null()
        } else {
            text.as_ptr().cast()
        },
        len: text.len(),
    }
}

impl ErrorStorage {
    /// An error with a message and no typed payload. The returned pointer is
    /// owned by the assembler until `free`.
    pub(crate) fn generic(code: TileIRLPResult, message: String) -> *mut TileIRLPError {
        Self::build(
            code,
            TileIRLPErrorKind::TILEIR_LP_ERROR_KIND_GENERIC,
            message,
            Vec::new(),
        )
    }

    /// The `instantiate` error naming the calls the provider cannot serve,
    /// each with its reason.
    pub(crate) fn unresolved(
        message: String,
        failures: Vec<(usize, String)>,
    ) -> *mut TileIRLPError {
        Self::build(
            TileIRLPResult::TILEIR_LP_ERROR_UNRESOLVED_CALLS,
            TileIRLPErrorKind::TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS,
            message,
            failures,
        )
    }

    fn build(
        code: TileIRLPResult,
        kind: TileIRLPErrorKind,
        message: String,
        failures: Vec<(usize, String)>,
    ) -> *mut TileIRLPError {
        let reasons: Vec<Box<str>> = failures
            .iter()
            .map(|(_, reason)| reason.clone().into_boxed_str())
            .collect();
        let unresolved: Vec<TileIRLPUnresolvedCall> = failures
            .iter()
            .zip(&reasons)
            .map(|((index, _), reason)| TileIRLPUnresolvedCall {
                call_index: *index,
                reason: view(reason),
            })
            .collect();
        let mut storage = Box::new(Self {
            error: TileIRLPError::default(),
            message: message.into_boxed_str(),
            details: String::new().into_boxed_str(),
            reasons,
            unresolved,
        });
        storage.error.code = code;
        storage.error.kind = kind;
        storage.error.message = view(&storage.message);
        storage.error.details = view(&storage.details);
        if kind == TileIRLPErrorKind::TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS {
            storage.error.__bindgen_anon_1.unresolved_calls = TileIRLPUnresolvedCallsError {
                calls: if storage.unresolved.is_empty() {
                    std::ptr::null()
                } else {
                    storage.unresolved.as_ptr()
                },
                num_calls: storage.unresolved.len(),
            };
        }
        // The box owns the heap the views point into; moving the box does not
        // move that heap, and the first-field pointer is the box pointer.
        Box::into_raw(storage).cast()
    }

    /// Release an error returned by one of the constructors above.
    ///
    /// # Safety
    /// `error` must be a pointer this module returned, not yet freed.
    pub(crate) unsafe fn free(error: *mut TileIRLPError) {
        if error.is_null() {
            return;
        }
        // SAFETY: the pointer came from `Box::into_raw` on an `ErrorStorage`
        // whose first field is the record.
        drop(unsafe { Box::from_raw(error.cast::<ErrorStorage>()) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::text;

    #[test]
    fn errors_round_trip_through_the_c_record() {
        let error = ErrorStorage::unresolved(
            "two calls failed".into(),
            vec![(0, "no such symbol".into()), (3, "wrong arch".into())],
        );
        // SAFETY: the pointer is live until `free`.
        unsafe {
            let record = &*error;
            assert_eq!(
                record.code,
                TileIRLPResult::TILEIR_LP_ERROR_UNRESOLVED_CALLS
            );
            assert_eq!(text(&record.message), "two calls failed");
            let calls = record.__bindgen_anon_1.unresolved_calls;
            assert_eq!(calls.num_calls, 2);
            let entries = std::slice::from_raw_parts(calls.calls, calls.num_calls);
            assert_eq!(entries[1].call_index, 3);
            assert_eq!(text(&entries[1].reason), "wrong arch");
            ErrorStorage::free(error);
        }
        let generic = ErrorStorage::generic(TileIRLPResult::TILEIR_LP_ERROR, "boom".into());
        // SAFETY: as above.
        unsafe {
            assert_eq!(text(&(*generic).message), "boom");
            assert_eq!(
                (*generic).kind,
                TileIRLPErrorKind::TILEIR_LP_ERROR_KIND_GENERIC
            );
            ErrorStorage::free(generic);
            ErrorStorage::free(std::ptr::null_mut());
        }
    }
}
