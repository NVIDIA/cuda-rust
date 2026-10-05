/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IKET placeholder ABI.
//!
//! These instruction shapes are compiler-owned placeholders. IKET recognizes
//! and rewrites their final SASS; they are not a standalone event transport
//! implemented by cuda-oxide.

use crate::InstrumentMethod;
use dialect_iket::attributes::IketPayloadKindAttr;
use thiserror::Error;

/// Shared-memory address form required by one architecture family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedAddressMode {
    /// SM90 and SM10x/SM11x use the cluster CTA rank when forming the
    /// deliberately invalid placeholder address.
    ClusterCtaRank,
    /// SM12x uses CTA-local placeholder addressing and does not emit cluster
    /// operations.
    CtaLocal,
}

/// Architecture-specific fields in the IKET placeholder contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceholderConfig {
    pub sts_offset: u32,
    pub shared_address_mode: SharedAddressMode,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IketPhysicalAbiError {
    #[error("IKET materialization requires an explicit sm_* target")]
    MissingTarget,
    #[error(
        "IKET has no placeholder contract for target {target:?}; supported families are SM90, SM10x/SM11x, and SM12x"
    )]
    UnsupportedTarget { target: String },
    #[error("event ID {event_id} is invalid for {method:?}")]
    InvalidEventId {
        method: InstrumentMethod,
        event_id: u32,
    },
}

/// Resolve the IKET placeholder address contract from a CUDA target.
///
/// Safety note on the placeholder addresses: when no IKET tool patches the
/// SASS, the placeholder really executes and stores to a fixed shared-memory
/// address on every normal run. On sm_120 this was verified on hardware to
/// land inside the driver-reserved first 1KB below user allocations (user
/// shared memory starts at 0x400; 1024 reserved bytes per block), so it
/// cannot corrupt user shared-memory state. The sm_90 contract (0x3f0) and
/// the cluster-rank-addressed sm_100/sm_110 variants have not been
/// hardware-verified and rely on the same reserved-region argument.
pub fn placeholder_config(target: Option<&str>) -> Result<PlaceholderConfig, IketPhysicalAbiError> {
    let target = target.ok_or(IketPhysicalAbiError::MissingTarget)?;
    let Some(capability) = parse_compute_capability(target) else {
        return Err(IketPhysicalAbiError::UnsupportedTarget {
            target: target.to_owned(),
        });
    };
    match capability {
        90 => Ok(PlaceholderConfig {
            sts_offset: 0x3f0,
            shared_address_mode: SharedAddressMode::ClusterCtaRank,
        }),
        100..=119 => Ok(PlaceholderConfig {
            sts_offset: 0x20,
            shared_address_mode: SharedAddressMode::ClusterCtaRank,
        }),
        120..=129 => Ok(PlaceholderConfig {
            sts_offset: 0x20,
            shared_address_mode: SharedAddressMode::CtaLocal,
        }),
        _ => Err(IketPhysicalAbiError::UnsupportedTarget {
            target: target.to_owned(),
        }),
    }
}

/// Build the canonical CUDA C++ IKET NativeDump-family placeholder.
pub fn build_placeholder_ptx(
    config: PlaceholderConfig,
    method: InstrumentMethod,
    event_id: u32,
    payload: IketPayloadKindAttr,
) -> Result<String, IketPhysicalAbiError> {
    validate_event_id(method, event_id)?;
    let (rank_read, address_finalize) = address_fragments(config);
    let payload_width = payload_width(payload);

    let ptx = match (method, payload_width) {
        (InstrumentMethod::NativeDump, 0) => format!(
            "{{ .reg .b32 %r, %t; {rank_read} mov.u32 %t, %globaltimer_lo; \
             or.b32 %t, %t, {event_id}; {address_finalize} \
             st.weak.shared.u32 [%r], %t; pmevent.mask {event_id}; }}"
        ),
        (InstrumentMethod::NativeDump, 32) => format!(
            "{{ .reg .pred %p; .reg .b32 %r, %t, %mask, %payload32; \
             activemask.b32 %mask; elect.sync _|%p, %mask; {rank_read} \
             mov.u32 %t, %globaltimer_lo; or.b32 %t, %t, {event_id}; \
             {address_finalize} mov.b32 %payload32, $0; \
             @%p st.weak.shared.u32 [%r], %t; \
             @%p st.weak.shared.b32 [%r+4], %payload32; \
             pmevent.mask {event_id}; }}"
        ),
        (InstrumentMethod::NativeDump, 64) => format!(
            "{{ .reg .pred %p; .reg .b32 %r, %t, %mask; .reg .b64 %payload64; \
             activemask.b32 %mask; elect.sync _|%p, %mask; {rank_read} \
             mov.u32 %t, %globaltimer_lo; or.b32 %t, %t, {event_id}; \
             {address_finalize} mov.b64 %payload64, $0; \
             @%p st.weak.shared.u32 [%r], %t; \
             @%p st.weak.shared.b64 [%r+8], %payload64; \
             pmevent.mask {event_id}; }}"
        ),
        (InstrumentMethod::ExtendedNativeDump, 0) => format!(
            "{{ .reg .b32 %r, %t, %evtid; .reg .b64 %ts_evtid; \
             {rank_read} mov.u32 %t, %globaltimer_lo; {address_finalize} \
             mov.b32 %evtid, {event_id}; mov.b64 %ts_evtid, {{%t, %evtid}}; \
             st.weak.shared.u64 [%r], %ts_evtid; pmevent.mask {event_id}; }}"
        ),
        (InstrumentMethod::ExtendedNativeDump, 32) => format!(
            "{{ .reg .pred %p; .reg .b32 %r, %t, %mask, %evtid, %payload32; \
             .reg .b64 %ts_evtid; activemask.b32 %mask; \
             elect.sync _|%p, %mask; {rank_read} mov.u32 %t, %globaltimer_lo; \
             {address_finalize} mov.b32 %evtid, {event_id}; \
             mov.b32 %payload32, $0; mov.b64 %ts_evtid, {{%t, %evtid}}; \
             @%p st.weak.shared.u64 [%r], %ts_evtid; \
             @%p st.weak.shared.b32 [%r+8], %payload32; \
             pmevent.mask {event_id}; }}"
        ),
        (InstrumentMethod::ExtendedNativeDump, 64) => format!(
            "{{ .reg .pred %p; .reg .b32 %r, %t, %mask, %evtid; \
             .reg .b64 %ts_evtid, %payload64; activemask.b32 %mask; \
             elect.sync _|%p, %mask; {rank_read} mov.u32 %t, %globaltimer_lo; \
             {address_finalize} mov.b32 %evtid, {event_id}; \
             mov.b64 %payload64, $0; mov.b64 %ts_evtid, {{%t, %evtid}}; \
             @%p st.weak.shared.u64 [%r], %ts_evtid; \
             @%p st.weak.shared.b64 [%r+8], %payload64; \
             pmevent.mask {event_id}; }}"
        ),
        _ => unreachable!("payload width is limited to 0, 32, or 64"),
    };
    Ok(ptx)
}

fn parse_compute_capability(target: &str) -> Option<u32> {
    let digits = target
        .strip_prefix("sm_")?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn validate_event_id(method: InstrumentMethod, event_id: u32) -> Result<(), IketPhysicalAbiError> {
    let valid = event_id == 31
        || match method {
            InstrumentMethod::NativeDump => (1..=30).contains(&event_id),
            InstrumentMethod::ExtendedNativeDump => (64..=4094).contains(&event_id),
        };
    if valid {
        Ok(())
    } else {
        Err(IketPhysicalAbiError::InvalidEventId { method, event_id })
    }
}

fn address_fragments(config: PlaceholderConfig) -> (String, String) {
    match config.shared_address_mode {
        SharedAddressMode::ClusterCtaRank => (
            "mov.b32 %r, %cluster_ctarank;".to_owned(),
            format!("mad.lo.u32 %r, %r, 0x1000000, {:#x};", config.sts_offset),
        ),
        SharedAddressMode::CtaLocal => (
            String::new(),
            format!("mov.b32 %r, {:#x};", config.sts_offset),
        ),
    }
}

fn payload_width(payload: IketPayloadKindAttr) -> u32 {
    match payload {
        IketPayloadKindAttr::None => 0,
        IketPayloadKindAttr::I8
        | IketPayloadKindAttr::U8
        | IketPayloadKindAttr::I16
        | IketPayloadKindAttr::U16
        | IketPayloadKindAttr::I32
        | IketPayloadKindAttr::U32
        | IketPayloadKindAttr::F32 => 32,
        IketPayloadKindAttr::I64
        | IketPayloadKindAttr::U64
        | IketPayloadKindAttr::F64
        | IketPayloadKindAttr::Pointer => 64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_families_match_the_runtime_contract() {
        assert_eq!(
            placeholder_config(Some("sm_90")).unwrap(),
            PlaceholderConfig {
                sts_offset: 0x3f0,
                shared_address_mode: SharedAddressMode::ClusterCtaRank,
            }
        );
        for target in ["sm_100", "sm_100a", "sm_103", "sm_110", "sm_119"] {
            assert_eq!(
                placeholder_config(Some(target))
                    .unwrap()
                    .shared_address_mode,
                SharedAddressMode::ClusterCtaRank
            );
        }
        for target in ["sm_120", "sm_120a", "sm_121", "sm_129"] {
            assert_eq!(
                placeholder_config(Some(target))
                    .unwrap()
                    .shared_address_mode,
                SharedAddressMode::CtaLocal
            );
        }
    }

    #[test]
    fn sm12x_never_mentions_cluster_state() {
        let ptx = build_placeholder_ptx(
            placeholder_config(Some("sm_120")).unwrap(),
            InstrumentMethod::ExtendedNativeDump,
            64,
            IketPayloadKindAttr::None,
        )
        .unwrap();
        assert!(!ptx.contains("cluster"));
        assert!(ptx.contains("mov.b32 %r, 0x20"));
    }

    #[test]
    fn extended_dump_keeps_timestamp_and_event_id_in_separate_words() {
        let ptx = build_placeholder_ptx(
            placeholder_config(Some("sm_100")).unwrap(),
            InstrumentMethod::ExtendedNativeDump,
            64,
            IketPayloadKindAttr::None,
        )
        .unwrap();
        assert!(!ptx.contains("or.b32"));
        assert!(ptx.contains("mov.b64 %ts_evtid, {%t, %evtid}"));
        assert!(ptx.contains("st.weak.shared.u64"));
    }

    #[test]
    fn event_id_reservations_are_method_specific() {
        let config = placeholder_config(Some("sm_90")).unwrap();
        assert!(
            build_placeholder_ptx(
                config,
                InstrumentMethod::NativeDump,
                30,
                IketPayloadKindAttr::None
            )
            .is_ok()
        );
        assert!(
            build_placeholder_ptx(
                config,
                InstrumentMethod::NativeDump,
                64,
                IketPayloadKindAttr::None
            )
            .is_err()
        );
        assert!(
            build_placeholder_ptx(
                config,
                InstrumentMethod::ExtendedNativeDump,
                31,
                IketPayloadKindAttr::None
            )
            .is_ok()
        );
    }
}
