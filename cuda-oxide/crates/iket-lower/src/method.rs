/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Instrumentation-method selection.

use thiserror::Error;

/// Physical event encoding selected after semantic IKET analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrumentMethod {
    /// Event ID in the low five timestamp bits; four bytes per event without a
    /// payload.
    NativeDump,
    /// Separate 32-bit event ID; eight bytes per event without a payload.
    ExtendedNativeDump,
}

/// User control over method selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InstrumentMethodPolicy {
    /// Prefer NativeDump and widen only when its user-event budget is exceeded.
    #[default]
    Auto,
    /// Require NativeDump and diagnose an event-budget overflow.
    NativeDump,
    /// Require ExtendedNativeDump even when NativeDump would fit.
    ExtendedNativeDump,
}

/// Runtime ABI facts that affect compiler lowering.
///
/// Event-ID reservations and metadata encodings belong to the IKET contract,
/// not to `dialect-iket`. Keeping them in one profile allows a future contract
/// revision without changing the semantic dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IketCompatibilityProfile {
    pub native_user_event_capacity: usize,
    pub extended_user_event_capacity: usize,
    pub event_name_inline_bytes: usize,
}

/// IKET compiler/runtime compatibility profile.
///
/// NativeDump user IDs are 1..=30. ExtendedNativeDump adds 63 to the raw user
/// ID and requires the resulting 12-bit ID to be less than 4095, giving user
/// IDs 64..=4094 (4,031 IDs).
pub const IKET_COMPATIBILITY_PROFILE: IketCompatibilityProfile = IketCompatibilityProfile {
    native_user_event_capacity: 30,
    extended_user_event_capacity: 4_031,
    event_name_inline_bytes: 32,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MethodSelectionError {
    #[error(
        "NativeDump supports at most {capacity} unique user event names, got {actual}; use auto or extended"
    )]
    NativeCapacityExceeded { capacity: usize, actual: usize },
    #[error("ExtendedNativeDump supports at most {capacity} unique user event names, got {actual}")]
    ExtendedCapacityExceeded { capacity: usize, actual: usize },
}

pub fn select_instrument_method(
    profile: IketCompatibilityProfile,
    policy: InstrumentMethodPolicy,
    unique_user_event_names: usize,
) -> Result<InstrumentMethod, MethodSelectionError> {
    match policy {
        InstrumentMethodPolicy::Auto
            if unique_user_event_names <= profile.native_user_event_capacity =>
        {
            Ok(InstrumentMethod::NativeDump)
        }
        InstrumentMethodPolicy::Auto | InstrumentMethodPolicy::ExtendedNativeDump
            if unique_user_event_names <= profile.extended_user_event_capacity =>
        {
            Ok(InstrumentMethod::ExtendedNativeDump)
        }
        InstrumentMethodPolicy::NativeDump
            if unique_user_event_names <= profile.native_user_event_capacity =>
        {
            Ok(InstrumentMethod::NativeDump)
        }
        InstrumentMethodPolicy::NativeDump => Err(MethodSelectionError::NativeCapacityExceeded {
            capacity: profile.native_user_event_capacity,
            actual: unique_user_event_names,
        }),
        InstrumentMethodPolicy::Auto | InstrumentMethodPolicy::ExtendedNativeDump => {
            Err(MethodSelectionError::ExtendedCapacityExceeded {
                capacity: profile.extended_user_event_capacity,
                actual: unique_user_event_names,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_switches_after_thirty_unique_names() {
        assert_eq!(
            select_instrument_method(IKET_COMPATIBILITY_PROFILE, InstrumentMethodPolicy::Auto, 30),
            Ok(InstrumentMethod::NativeDump)
        );
        assert_eq!(
            select_instrument_method(IKET_COMPATIBILITY_PROFILE, InstrumentMethodPolicy::Auto, 31),
            Ok(InstrumentMethod::ExtendedNativeDump)
        );
    }

    #[test]
    fn explicit_native_fails_instead_of_silently_widening() {
        assert!(matches!(
            select_instrument_method(
                IKET_COMPATIBILITY_PROFILE,
                InstrumentMethodPolicy::NativeDump,
                31,
            ),
            Err(MethodSelectionError::NativeCapacityExceeded { .. })
        ));
    }

    #[test]
    fn explicit_extended_is_available_for_small_modules() {
        assert_eq!(
            select_instrument_method(
                IKET_COMPATIBILITY_PROFILE,
                InstrumentMethodPolicy::ExtendedNativeDump,
                1
            ),
            Ok(InstrumentMethod::ExtendedNativeDump)
        );
    }

    #[test]
    fn extended_budget_accounts_for_reserved_ids_and_offset() {
        assert_eq!(
            select_instrument_method(
                IKET_COMPATIBILITY_PROFILE,
                InstrumentMethodPolicy::Auto,
                4_031,
            ),
            Ok(InstrumentMethod::ExtendedNativeDump)
        );
        assert!(matches!(
            select_instrument_method(
                IKET_COMPATIBILITY_PROFILE,
                InstrumentMethodPolicy::Auto,
                4_032,
            ),
            Err(MethodSelectionError::ExtendedCapacityExceeded { .. })
        ));
    }
}
