/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering support for [`dialect_iket`].
//!
//! The semantic dialect is method-agnostic. This crate owns compatibility
//! profiles and the eventual dialect-to-instrumentation lowering.

pub mod event_name;
pub mod metadata;
pub mod method;
pub mod physical;
pub mod plan;

pub use event_name::{EncodedEventName, EventNameError, EventNameTable, encode_event_name};
pub use metadata::{
    EventMetadata, EventPosition, IketMetadataError, MetadataObject, RangeMetadata, RangeType,
    encode_metadata_objects, fnv1a_32,
};
pub use method::{
    IKET_COMPATIBILITY_PROFILE, IketCompatibilityProfile, InstrumentMethod, InstrumentMethodPolicy,
    MethodSelectionError, select_instrument_method,
};
pub use physical::{
    IketPhysicalAbiError, PlaceholderConfig, SharedAddressMode, build_placeholder_ptx,
    placeholder_config,
};
pub use plan::{IketLoweringPlan, LoweringPlanError, plan_instrumentation};
