/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IKET CUBIN metadata encoding.

use crate::{EncodedEventName, InstrumentMethod};
use dialect_iket::attributes::IketPayloadKindAttr;
use std::collections::BTreeMap;
use thiserror::Error;

const IKET_META_INFO_SYMBOL: &str = "__iket_meta_info";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum EventPosition {
    NotInRange = 0,
    RangeStart = 1,
    RangeStartEnd = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RangeType {
    StartEnd = 1,
    PushPop = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventMetadata {
    pub event_id: u32,
    pub method: InstrumentMethod,
    pub payload: IketPayloadKindAttr,
    pub position: EventPosition,
    pub range_id: u32,
    pub name: EncodedEventName,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeMetadata {
    pub range_id: u32,
    pub range_type: RangeType,
    pub name: EncodedEventName,
}

/// One initialized device global consumed by the IKET CUBIN parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataObject {
    pub symbol: String,
    pub bytes: Vec<u8>,
    pub alignment: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IketMetadataError {
    #[error("IKET metadata symbol {symbol:?} is emitted more than once")]
    DuplicateSymbol { symbol: String },
    #[error("IKET range ID zero is reserved for discrete events")]
    ZeroRangeId,
}

/// Encode the global objects parsed by IKET.
pub fn encode_metadata_objects(
    method: InstrumentMethod,
    events: &[EventMetadata],
    ranges: &[RangeMetadata],
) -> Result<Vec<MetadataObject>, IketMetadataError> {
    let mut objects = vec![MetadataObject {
        symbol: IKET_META_INFO_SYMBOL.to_owned(),
        bytes: encode_meta_info(method),
        alignment: 8,
    }];

    for event in events {
        objects.push(MetadataObject {
            symbol: format!("__iket_evt_decl_id_{}_attrs", event.event_id),
            bytes: encode_event(event),
            alignment: 4,
        });
        objects.push(MetadataObject {
            symbol: event.name.string_symbol.clone(),
            bytes: nul_terminated(event.name.full_name.as_bytes()),
            alignment: 1,
        });
    }
    for range in ranges {
        if range.range_id == 0 {
            return Err(IketMetadataError::ZeroRangeId);
        }
        objects.push(MetadataObject {
            symbol: format!("__iket_range_decl_id_{}_attrs", range.range_id),
            bytes: encode_range(range),
            alignment: 8,
        });
        // Event and range names can be identical. Give the range string a
        // kind-specific symbol, like the CUDA C++ IKET declarations do.
        let hash = crate::event_name::fnv1a_64(range.name.full_name.as_bytes());
        objects.push(MetadataObject {
            symbol: format!("__iket_string_decl_rng_h{hash:016x}_str"),
            bytes: nul_terminated(range.name.full_name.as_bytes()),
            alignment: 1,
        });
    }

    let mut unique = Vec::<MetadataObject>::new();
    let mut symbol_indices = BTreeMap::<String, usize>::new();
    for object in objects {
        if let Some(index) = symbol_indices.get(&object.symbol) {
            let previous = &unique[*index];
            // Repeated uses of one event name intentionally share one string
            // table object. Any other symbol collision is malformed metadata.
            if previous.bytes != object.bytes || previous.alignment != object.alignment {
                return Err(IketMetadataError::DuplicateSymbol {
                    symbol: object.symbol,
                });
            }
        } else {
            symbol_indices.insert(object.symbol.clone(), unique.len());
            unique.push(object);
        }
    }
    Ok(unique)
}

/// CUDA C++ IKET's range ID hash.
pub fn fnv1a_32(bytes: &[u8]) -> u32 {
    bytes.iter().fold(2_166_136_261, |hash, byte| {
        (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
    })
}

fn encode_meta_info(method: InstrumentMethod) -> Vec<u8> {
    let max_event_id = match method {
        InstrumentMethod::NativeDump => 31,
        InstrumentMethod::ExtendedNativeDump => 4095,
    };
    let mut bytes = Vec::with_capacity(48);
    for value in [48, 0, 7, max_event_id, u32::MAX, 60, 0xbabe_f19d, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    // kBasic | kPayload in the IKET feature bitset.
    bytes.extend_from_slice(&3u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(bytes.len(), 48);
    bytes
}

fn encode_event(event: &EventMetadata) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(60);
    for value in [
        60,
        event.event_id,
        method_id(event.method),
        payload_id(event.payload),
        event.position as u32,
        event.range_id,
        event.name.full_name_size,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&event.name.inline);
    debug_assert_eq!(bytes.len(), 60);
    bytes
}

fn encode_range(range: &RangeMetadata) -> Vec<u8> {
    let pair_mode = match range.range_type {
        RangeType::StartEnd => 1, // kNameOnly
        RangeType::PushPop => 0,  // kInvalid
    };
    let mut bytes = Vec::with_capacity(72);
    for value in [
        72,
        0, // kIntraWarp
        range.range_id,
        u32::MAX, // kAuto color
        range.range_type as u32,
        pair_mode,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&range.name.full_name_size.to_le_bytes());
    bytes.extend_from_slice(&range.name.inline);
    bytes.extend_from_slice(&0u32.to_le_bytes()); // marker_count
    debug_assert_eq!(bytes.len(), 72);
    bytes
}

fn method_id(method: InstrumentMethod) -> u32 {
    match method {
        InstrumentMethod::NativeDump => 3,
        InstrumentMethod::ExtendedNativeDump => 5,
    }
}

fn payload_id(payload: IketPayloadKindAttr) -> u32 {
    match payload {
        IketPayloadKindAttr::None => 0,
        IketPayloadKindAttr::I8 => 1,
        IketPayloadKindAttr::U8 => 2,
        IketPayloadKindAttr::I16 => 3,
        IketPayloadKindAttr::U16 => 4,
        IketPayloadKindAttr::I32 => 5,
        IketPayloadKindAttr::U32 => 6,
        IketPayloadKindAttr::I64 => 7,
        IketPayloadKindAttr::F32 => 13,
        IketPayloadKindAttr::F64 => 14,
        IketPayloadKindAttr::Pointer => 15,
        IketPayloadKindAttr::U64 => 16,
    }
}

fn nul_terminated(bytes: &[u8]) -> Vec<u8> {
    let mut result = bytes.to_vec();
    result.push(0);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IKET_COMPATIBILITY_PROFILE, encode_event_name};

    fn name(value: &str) -> EncodedEventName {
        encode_event_name(IKET_COMPATIBILITY_PROFILE, value).unwrap()
    }

    #[test]
    fn meta_info_tracks_the_selected_method_budget() {
        let native = encode_metadata_objects(InstrumentMethod::NativeDump, &[], &[]).unwrap();
        let extended =
            encode_metadata_objects(InstrumentMethod::ExtendedNativeDump, &[], &[]).unwrap();
        assert_eq!(&native[0].bytes[12..16], &31u32.to_le_bytes());
        assert_eq!(&extended[0].bytes[12..16], &4095u32.to_le_bytes());
    }

    #[test]
    fn event_attributes_match_the_cuda_cpp_layout() {
        let event = EventMetadata {
            event_id: 64,
            method: InstrumentMethod::ExtendedNativeDump,
            payload: IketPayloadKindAttr::U64,
            position: EventPosition::NotInRange,
            range_id: 0,
            name: name("mma"),
        };
        let objects =
            encode_metadata_objects(InstrumentMethod::ExtendedNativeDump, &[event], &[]).unwrap();
        let attrs = &objects[1].bytes;
        assert_eq!(attrs.len(), 60);
        assert_eq!(&attrs[4..8], &64u32.to_le_bytes());
        assert_eq!(&attrs[8..12], &5u32.to_le_bytes());
        assert_eq!(&attrs[12..16], &16u32.to_le_bytes());
    }

    #[test]
    fn long_names_emit_hash_placeholders_and_full_string_objects() {
        let full_name = "x".repeat(80);
        let event = EventMetadata {
            event_id: 1,
            method: InstrumentMethod::NativeDump,
            payload: IketPayloadKindAttr::None,
            position: EventPosition::NotInRange,
            range_id: 0,
            name: name(&full_name),
        };
        let objects = encode_metadata_objects(InstrumentMethod::NativeDump, &[event], &[]).unwrap();
        assert_eq!(objects[1].bytes[28], b'h');
        assert_eq!(objects[2].bytes, nul_terminated(full_name.as_bytes()));
    }

    #[test]
    fn range_hash_matches_cuda_cpp_fnv1a() {
        assert_eq!(fnv1a_32(b"hello"), 0x4f9f_2cab);
    }
}
