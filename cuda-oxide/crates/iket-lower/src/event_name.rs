/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA C++ IKET-compatible event-name encoding.

use crate::method::IketCompatibilityProfile;
use std::collections::BTreeMap;
use thiserror::Error;

const FNV1A_64_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV1A_64_PRIME: u64 = 1_099_511_628_211;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedEventName {
    /// Fixed-size `event_name` / `range_name` field stored in IKET metadata.
    pub inline: Vec<u8>,
    /// Full source name, excluding the trailing NUL.
    pub full_name: String,
    /// Full source name size including its trailing NUL, as stored by IKET.
    pub full_name_size: u32,
    /// Collision-resistant, ELF-safe symbol understood by IKET.
    pub string_symbol: String,
    /// Whether `inline` contains an `h<16hex>` placeholder.
    pub uses_hash_placeholder: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EventNameError {
    #[error("IKET event name must not be empty")]
    Empty,
    #[error("IKET event name must not contain NUL")]
    ContainsNul,
    #[error("IKET event name is too large to encode its byte length in u32")]
    TooLarge,
    #[error("IKET event-name hash collision between {first:?} and {second:?}")]
    HashCollision { first: String, second: String },
}

pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV1A_64_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV1A_64_PRIME)
    })
}

pub fn encode_event_name(
    profile: IketCompatibilityProfile,
    name: &str,
) -> Result<EncodedEventName, EventNameError> {
    if name.is_empty() {
        return Err(EventNameError::Empty);
    }
    if name.as_bytes().contains(&0) {
        return Err(EventNameError::ContainsNul);
    }
    let full_name_size =
        u32::try_from(name.len().saturating_add(1)).map_err(|_| EventNameError::TooLarge)?;
    let hash = fnv1a_64(name.as_bytes());
    let mut inline = vec![0; profile.event_name_inline_bytes];
    let uses_hash_placeholder = name.len().saturating_add(1) > inline.len();
    if uses_hash_placeholder {
        let placeholder = format!("h{hash:016x}");
        inline[..placeholder.len()].copy_from_slice(placeholder.as_bytes());
    } else {
        inline[..name.len()].copy_from_slice(name.as_bytes());
    }

    Ok(EncodedEventName {
        inline,
        full_name: name.to_string(),
        full_name_size,
        // The host parser keys on the prefix and hashes the symbol contents;
        // the suffix is intentionally independent of source characters.
        string_symbol: format!("__iket_string_decl_evt_h{hash:016x}_str"),
        uses_hash_placeholder,
    })
}

/// Deduplicated CUBIN string-table entries with explicit collision checking.
#[derive(Debug)]
pub struct EventNameTable {
    profile: IketCompatibilityProfile,
    names_by_hash: BTreeMap<u64, String>,
}

impl EventNameTable {
    pub fn new(profile: IketCompatibilityProfile) -> Self {
        Self {
            profile,
            names_by_hash: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, name: &str) -> Result<EncodedEventName, EventNameError> {
        let encoded = encode_event_name(self.profile, name)?;
        let hash = fnv1a_64(name.as_bytes());
        if let Some(existing) = self.names_by_hash.get(&hash) {
            if existing != name {
                return Err(EventNameError::HashCollision {
                    first: existing.clone(),
                    second: name.to_string(),
                });
            }
        } else {
            self.names_by_hash.insert(hash, name.to_string());
        }
        Ok(encoded)
    }

    pub fn len(&self) -> usize {
        self.names_by_hash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names_by_hash.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::IKET_COMPATIBILITY_PROFILE;

    #[test]
    fn short_name_is_inline_and_nul_padded() {
        let encoded = encode_event_name(IKET_COMPATIBILITY_PROFILE, "tma.issue").unwrap();
        assert!(!encoded.uses_hash_placeholder);
        assert_eq!(&encoded.inline[..10], b"tma.issue\0");
        assert_eq!(encoded.inline.len(), 32);
        assert_eq!(encoded.full_name_size, 10);
    }

    #[test]
    fn thirty_one_bytes_fit_but_thirty_two_use_the_long_name_path() {
        let inline = encode_event_name(IKET_COMPATIBILITY_PROFILE, &"a".repeat(31)).unwrap();
        assert!(!inline.uses_hash_placeholder);

        let long = encode_event_name(IKET_COMPATIBILITY_PROFILE, &"b".repeat(32)).unwrap();
        assert!(long.uses_hash_placeholder);
        assert_eq!(long.inline[0], b'h');
        assert_eq!(long.inline[17], 0);
        assert_eq!(long.full_name_size, 33);
    }

    #[test]
    fn inline_limit_is_measured_in_utf8_bytes_not_characters() {
        // Sixteen CJK code points occupy 48 UTF-8 bytes and therefore cannot
        // fit in the 32-byte metadata field despite having fewer characters.
        let encoded = encode_event_name(IKET_COMPATIBILITY_PROFILE, &"事".repeat(16)).unwrap();
        assert!(encoded.uses_hash_placeholder);
        assert_eq!(encoded.full_name_size, 49);
    }

    #[test]
    fn hash_matches_cuda_cpp_iket_fnv1a() {
        assert_eq!(fnv1a_64(b"hello"), 0xa430_d846_80aa_bd0b);
    }

    #[test]
    fn symbol_is_safe_for_names_with_spaces_and_punctuation() {
        let encoded = encode_event_name(
            IKET_COMPATIBILITY_PROFILE,
            "producer mainloop / tma.issue [stage=0]",
        )
        .unwrap();
        assert!(
            encoded
                .string_symbol
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
        );
    }

    #[test]
    fn table_deduplicates_repeated_names() {
        let mut table = EventNameTable::new(IKET_COMPATIBILITY_PROFILE);
        table.insert("mma").unwrap();
        table.insert("mma").unwrap();
        assert_eq!(table.len(), 1);
    }
}
