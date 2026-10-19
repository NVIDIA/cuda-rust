/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Embedded device artifact bundles.
//!
//! The wire format is intentionally independent of a particular accelerator
//! backend. A bundle names a producer, records the device target it was built
//! for, and carries one or more generated device-code payloads.

use core::fmt;

pub const ARTIFACT_SECTION_NAME: &str = ".oxart";
#[cfg(feature = "object-write")]
const ARTIFACT_ANCHOR_SECTION_NAME: &str = ".oxlink";
pub const ARTIFACT_MAGIC: [u8; 8] = *b"OXIDEART";
pub const ARTIFACT_VERSION: u16 = 2;
const LEGACY_ARTIFACT_VERSION: u16 = 1;

const HEADER_BYTES: usize = 32;
const PAYLOAD_RECORD_BYTES: usize = 24;
const ENTRY_RECORD_BYTES: usize = 24;

const OPTION_NO_FMA_CONTRACTION: u64 = 1 << 0;
const OPTION_DEBUG_LINE_TABLES: u64 = 1 << 1;
const OPTION_DEBUG_FULL: u64 = 1 << 2;
const OPTION_DEBUG_MASK: u64 = OPTION_DEBUG_LINE_TABLES | OPTION_DEBUG_FULL;
const KNOWN_COMPILE_OPTIONS: u64 = OPTION_NO_FMA_CONTRACTION | OPTION_DEBUG_MASK;

/// Marker written on the second line of a versioned NVVM/LTOIR `.target`
/// sidecar. Its presence makes older one-line readers reject the artifact
/// instead of silently ignoring required compile policy. The marker versions
/// the target/options handshake; the `.options` file carries its own v1/v2
/// format header.
pub const COMPILE_OPTIONS_TARGET_MARKER: &str = "compile-options=v1";

const COMPILE_OPTIONS_SIDECAR_HEADER: &str = "cuda-oxide-compile-options-v1";
const COMPILE_OPTIONS_FMA_ON: &str = "cuda-oxide-compile-options-v1\nfma-contraction=on\n";
const COMPILE_OPTIONS_FMA_OFF: &str = "cuda-oxide-compile-options-v1\nfma-contraction=off\n";
const COMPILE_OPTIONS_FMA_ON_LINE_TABLES: &str =
    "cuda-oxide-compile-options-v2\nfma-contraction=on\ndebug=line-tables\n";
const COMPILE_OPTIONS_FMA_OFF_LINE_TABLES: &str =
    "cuda-oxide-compile-options-v2\nfma-contraction=off\ndebug=line-tables\n";
const COMPILE_OPTIONS_FMA_ON_FULL_DEBUG: &str =
    "cuda-oxide-compile-options-v2\nfma-contraction=on\ndebug=full\n";
const COMPILE_OPTIONS_FMA_OFF_FULL_DEBUG: &str =
    "cuda-oxide-compile-options-v2\nfma-contraction=off\ndebug=full\n";

/// Debug information that later CUDA compilation/linking stages must retain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ArtifactDebugPolicy {
    /// No device debug output is requested.
    #[default]
    None,
    /// Preserve source line mappings without disabling optimization.
    LineTables,
    /// Preserve full debug information; libNVVM finalization runs unoptimized.
    Full,
}

/// Compilation policy that must remain attached to a device artifact until
/// its final machine-code generation step.
///
/// A zero value preserves the historical defaults, which keeps version-1
/// bundles emitted before this field was used fully compatible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArtifactCompileOptions(u64);

impl ArtifactCompileOptions {
    /// Construct the historical default policy.
    pub const fn new() -> Self {
        Self(0)
    }

    /// Record whether ordinary floating-point multiply/add expressions may
    /// contract into fused operations.
    pub const fn with_fma_contraction(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 &= !OPTION_NO_FMA_CONTRACTION;
        } else {
            self.0 |= OPTION_NO_FMA_CONTRACTION;
        }
        self
    }

    /// Whether ordinary floating-point multiply/add expressions may contract.
    pub const fn fma_contraction_enabled(self) -> bool {
        self.0 & OPTION_NO_FMA_CONTRACTION == 0
    }

    /// Record the debug policy that all remaining CUDA compiler stages must
    /// use.
    pub const fn with_debug_policy(mut self, policy: ArtifactDebugPolicy) -> Self {
        self.0 &= !OPTION_DEBUG_MASK;
        self.0 |= match policy {
            ArtifactDebugPolicy::None => 0,
            ArtifactDebugPolicy::LineTables => OPTION_DEBUG_LINE_TABLES,
            ArtifactDebugPolicy::Full => OPTION_DEBUG_FULL,
        };
        self
    }

    /// Debug policy attached to this artifact.
    pub const fn debug_policy(self) -> ArtifactDebugPolicy {
        match self.0 & OPTION_DEBUG_MASK {
            OPTION_DEBUG_LINE_TABLES => ArtifactDebugPolicy::LineTables,
            OPTION_DEBUG_FULL => ArtifactDebugPolicy::Full,
            _ => ArtifactDebugPolicy::None,
        }
    }

    fn from_bits(bits: u64) -> Result<Self, ArtifactError> {
        if bits & !KNOWN_COMPILE_OPTIONS != 0 || bits & OPTION_DEBUG_MASK == OPTION_DEBUG_MASK {
            return Err(ArtifactError::UnsupportedCompileOptions(bits));
        }
        Ok(Self(bits))
    }

    const fn bits(self) -> u64 {
        self.0
    }

    /// Encode this policy for a sibling `.options` file.
    pub const fn sidecar_text(self) -> &'static str {
        match (self.fma_contraction_enabled(), self.debug_policy()) {
            (true, ArtifactDebugPolicy::None) => COMPILE_OPTIONS_FMA_ON,
            (false, ArtifactDebugPolicy::None) => COMPILE_OPTIONS_FMA_OFF,
            (true, ArtifactDebugPolicy::LineTables) => COMPILE_OPTIONS_FMA_ON_LINE_TABLES,
            (false, ArtifactDebugPolicy::LineTables) => COMPILE_OPTIONS_FMA_OFF_LINE_TABLES,
            (true, ArtifactDebugPolicy::Full) => COMPILE_OPTIONS_FMA_ON_FULL_DEBUG,
            (false, ArtifactDebugPolicy::Full) => COMPILE_OPTIONS_FMA_OFF_FULL_DEBUG,
        }
    }

    /// Parse a complete `.options` file. Version 1 defaults to no debug policy;
    /// version 2 records line-table or full-debug preservation explicitly.
    pub fn from_sidecar_text(value: &str) -> Result<Self, ArtifactError> {
        match value {
            COMPILE_OPTIONS_FMA_ON => Ok(Self::new()),
            COMPILE_OPTIONS_FMA_OFF => Ok(Self::new().with_fma_contraction(false)),
            COMPILE_OPTIONS_FMA_ON_LINE_TABLES => {
                Ok(Self::new().with_debug_policy(ArtifactDebugPolicy::LineTables))
            }
            COMPILE_OPTIONS_FMA_OFF_LINE_TABLES => Ok(Self::new()
                .with_fma_contraction(false)
                .with_debug_policy(ArtifactDebugPolicy::LineTables)),
            COMPILE_OPTIONS_FMA_ON_FULL_DEBUG => {
                Ok(Self::new().with_debug_policy(ArtifactDebugPolicy::Full))
            }
            COMPILE_OPTIONS_FMA_OFF_FULL_DEBUG => Ok(Self::new()
                .with_fma_contraction(false)
                .with_debug_policy(ArtifactDebugPolicy::Full)),
            _ => Err(ArtifactError::MalformedCompileOptions(format!(
                "expected `{COMPILE_OPTIONS_SIDECAR_HEADER}` or `cuda-oxide-compile-options-v2` with canonical fma/debug settings"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArtifactPayloadKind {
    Ptx,
    NvvmIr,
    Ltoir,
    Cubin,
}

impl ArtifactPayloadKind {
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Ptx => 0x100,
            Self::NvvmIr => 0x110,
            Self::Ltoir => 0x120,
            Self::Cubin => 0x200,
        }
    }

    pub const fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x100 => Some(Self::Ptx),
            0x110 => Some(Self::NvvmIr),
            0x120 => Some(Self::Ltoir),
            0x200 => Some(Self::Cubin),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArtifactEntryKind {
    Kernel,
    DeviceFunction,
}

impl ArtifactEntryKind {
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Kernel => 1,
            Self::DeviceFunction => 2,
        }
    }

    pub const fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Kernel),
            2 => Some(Self::DeviceFunction),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactPayloadSpec<'a> {
    pub kind: ArtifactPayloadKind,
    pub name: &'a str,
    pub bytes: &'a [u8],
}

impl<'a> ArtifactPayloadSpec<'a> {
    pub const fn new(kind: ArtifactPayloadKind, name: &'a str, bytes: &'a [u8]) -> Self {
        Self { kind, name, bytes }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactEntrySpec<'a> {
    pub symbol: &'a str,
    pub kind: ArtifactEntryKind,
    pub metadata: Option<u64>,
}

impl<'a> ArtifactEntrySpec<'a> {
    pub const fn new(symbol: &'a str, kind: ArtifactEntryKind) -> Self {
        Self {
            symbol,
            kind,
            metadata: None,
        }
    }

    pub const fn with_metadata(mut self, metadata: u64) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactBundleSpec<'a> {
    pub name: &'a str,
    pub target: &'a str,
    pub compile_options: ArtifactCompileOptions,
    pub payloads: Vec<ArtifactPayloadSpec<'a>>,
    pub entries: Vec<ArtifactEntrySpec<'a>>,
}

impl<'a> ArtifactBundleSpec<'a> {
    pub fn new(name: &'a str, target: &'a str) -> Self {
        Self {
            name,
            target,
            compile_options: ArtifactCompileOptions::new(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn with_payload(mut self, payload: ArtifactPayloadSpec<'a>) -> Self {
        self.payloads.push(payload);
        self
    }

    pub fn with_compile_options(mut self, options: ArtifactCompileOptions) -> Self {
        self.compile_options = options;
        self
    }

    pub fn with_entry(mut self, entry: ArtifactEntrySpec<'a>) -> Self {
        self.entries.push(entry);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactPayload<'a> {
    pub kind: ArtifactPayloadKind,
    pub name: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactEntry<'a> {
    pub symbol: &'a str,
    pub kind: ArtifactEntryKind,
    pub metadata: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactBundle<'a> {
    pub name: &'a str,
    pub target: &'a str,
    pub compile_options: ArtifactCompileOptions,
    pub payloads: Vec<ArtifactPayload<'a>>,
    pub entries: Vec<ArtifactEntry<'a>>,
}

impl<'a> ArtifactBundle<'a> {
    pub fn payload(&self, kind: ArtifactPayloadKind) -> Option<&'a [u8]> {
        self.payloads
            .iter()
            .find(|payload| payload.kind == kind)
            .map(|payload| payload.bytes)
    }

    pub fn entry(&self, symbol: &str) -> Option<&ArtifactEntry<'a>> {
        self.entries.iter().find(|entry| entry.symbol == symbol)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedArtifactPayload {
    pub kind: ArtifactPayloadKind,
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedArtifactEntry {
    pub symbol: String,
    pub kind: ArtifactEntryKind,
    pub metadata: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedArtifactBundle {
    pub name: String,
    pub target: String,
    pub compile_options: ArtifactCompileOptions,
    pub payloads: Vec<OwnedArtifactPayload>,
    pub entries: Vec<OwnedArtifactEntry>,
}

impl OwnedArtifactBundle {
    pub fn payload(&self, kind: ArtifactPayloadKind) -> Option<&[u8]> {
        self.payloads
            .iter()
            .find(|payload| payload.kind == kind)
            .map(|payload| payload.bytes.as_slice())
    }

    pub fn entry(&self, symbol: &str) -> Option<&OwnedArtifactEntry> {
        self.entries.iter().find(|entry| entry.symbol == symbol)
    }
}

impl<'a> From<ArtifactBundle<'a>> for OwnedArtifactBundle {
    fn from(bundle: ArtifactBundle<'a>) -> Self {
        Self {
            name: bundle.name.to_string(),
            target: bundle.target.to_string(),
            compile_options: bundle.compile_options,
            payloads: bundle
                .payloads
                .into_iter()
                .map(|payload| OwnedArtifactPayload {
                    kind: payload.kind,
                    name: payload.name.to_string(),
                    bytes: payload.bytes.to_vec(),
                })
                .collect(),
            entries: bundle
                .entries
                .into_iter()
                .map(|entry| OwnedArtifactEntry {
                    symbol: entry.symbol.to_string(),
                    kind: entry.kind,
                    metadata: entry.metadata,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    TooLarge(&'static str),
    EmptyBundleName,
    EmptyTarget,
    EmptyPayloadName,
    EmptyPayload,
    EmptyEntrySymbol,
    Truncated(&'static str),
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedCompileOptions(u64),
    MalformedCompileOptions(String),
    UnsupportedPayloadKind(u16),
    UnsupportedEntryKind(u16),
    InvalidUtf8(&'static str),
    UnsupportedHostTarget(String),
    Object(String),
    Malformed(String),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge(field) => write!(f, "embedded artifact {field} is too large"),
            Self::EmptyBundleName => f.write_str("embedded artifact bundle name is empty"),
            Self::EmptyTarget => f.write_str("embedded artifact target is empty"),
            Self::EmptyPayloadName => f.write_str("embedded artifact payload name is empty"),
            Self::EmptyPayload => f.write_str("embedded artifact payload is empty"),
            Self::EmptyEntrySymbol => f.write_str("embedded artifact entry symbol is empty"),
            Self::Truncated(field) => write!(f, "embedded artifact is truncated in {field}"),
            Self::BadMagic => f.write_str("embedded artifact has bad magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported embedded artifact version {version}")
            }
            Self::UnsupportedCompileOptions(bits) => {
                write!(
                    f,
                    "unsupported embedded artifact compile-options bits {bits:#x}"
                )
            }
            Self::MalformedCompileOptions(message) => {
                write!(f, "malformed cuda-oxide compile options: {message}")
            }
            Self::UnsupportedPayloadKind(kind) => {
                write!(f, "unsupported embedded artifact payload kind {kind}")
            }
            Self::UnsupportedEntryKind(kind) => {
                write!(f, "unsupported embedded artifact entry kind {kind}")
            }
            Self::InvalidUtf8(field) => write!(f, "embedded artifact {field} is not utf-8"),
            Self::UnsupportedHostTarget(target) => {
                write!(f, "unsupported host object target '{target}'")
            }
            Self::Object(message) | Self::Malformed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ArtifactError {}

pub fn build_artifact_blob(spec: &ArtifactBundleSpec<'_>) -> Result<Vec<u8>, ArtifactError> {
    validate_spec(spec)?;

    let mut out = vec![0; HEADER_BYTES];
    push_bytes(&mut out, spec.name.as_bytes());
    push_bytes(&mut out, spec.target.as_bytes());

    let payload_record_start = out.len();
    out.resize(out.len() + spec.payloads.len() * PAYLOAD_RECORD_BYTES, 0);
    let entry_record_start = out.len();
    out.resize(out.len() + spec.entries.len() * ENTRY_RECORD_BYTES, 0);

    for (index, payload) in spec.payloads.iter().enumerate() {
        let name_offset = checked_u32(out.len(), "payload name offset")?;
        push_bytes(&mut out, payload.name.as_bytes());
        align_vec(&mut out, 8);
        let data_offset = checked_u32(out.len(), "payload data offset")?;
        push_bytes(&mut out, payload.bytes);
        align_vec(&mut out, 8);

        let record = payload_record_start + index * PAYLOAD_RECORD_BYTES;
        write_u16(&mut out, record, payload.kind.to_u16());
        write_u16(&mut out, record + 2, 0);
        write_u32(&mut out, record + 4, data_offset);
        write_u32(
            &mut out,
            record + 8,
            checked_u32(payload.bytes.len(), "payload length")?,
        );
        write_u32(&mut out, record + 12, name_offset);
        write_u16(
            &mut out,
            record + 16,
            checked_u16(payload.name.len(), "payload name length")?,
        );
    }

    for (index, entry) in spec.entries.iter().enumerate() {
        let symbol_offset = checked_u32(out.len(), "entry symbol offset")?;
        push_bytes(&mut out, entry.symbol.as_bytes());
        align_vec(&mut out, 8);

        let record = entry_record_start + index * ENTRY_RECORD_BYTES;
        write_u16(&mut out, record, entry.kind.to_u16());
        write_u16(&mut out, record + 2, u16::from(entry.metadata.is_some()));
        write_u64(&mut out, record + 4, entry.metadata.unwrap_or(0));
        write_u32(&mut out, record + 12, symbol_offset);
        write_u16(
            &mut out,
            record + 16,
            checked_u16(entry.symbol.len(), "entry symbol length")?,
        );
    }

    let total_len = checked_u32(out.len(), "total length")?;
    out[0..8].copy_from_slice(&ARTIFACT_MAGIC);
    // Keep default-policy bundles on v1 for backward compatibility. A bundle
    // that carries required compile policy uses v2 so an older reader rejects
    // it instead of silently ignoring the semantic flag.
    let version = if spec.compile_options == ArtifactCompileOptions::new() {
        LEGACY_ARTIFACT_VERSION
    } else {
        ARTIFACT_VERSION
    };
    write_u16(&mut out, 8, version);
    write_u16(&mut out, 10, HEADER_BYTES as u16);
    write_u32(&mut out, 12, total_len);
    write_u16(&mut out, 16, checked_u16(spec.name.len(), "name length")?);
    write_u16(
        &mut out,
        18,
        checked_u16(spec.target.len(), "target length")?,
    );
    write_u16(
        &mut out,
        20,
        checked_u16(spec.payloads.len(), "payload count")?,
    );
    write_u16(
        &mut out,
        22,
        checked_u16(spec.entries.len(), "entry count")?,
    );
    write_u64(&mut out, 24, spec.compile_options.bits());

    Ok(out)
}

pub fn parse_artifact_section(mut bytes: &[u8]) -> Result<Vec<ArtifactBundle<'_>>, ArtifactError> {
    let mut bundles = Vec::new();
    while !bytes.is_empty() {
        if bytes.iter().all(|byte| *byte == 0) {
            break;
        }
        let total_len = artifact_blob_total_len(bytes)?;
        let (blob, rest) = bytes.split_at(total_len);
        bundles.push(parse_artifact_blob(blob)?);
        bytes = rest;
    }
    Ok(bundles)
}

pub fn parse_artifact_blob(bytes: &[u8]) -> Result<ArtifactBundle<'_>, ArtifactError> {
    require_len(bytes, HEADER_BYTES, "header")?;
    if bytes[0..8] != ARTIFACT_MAGIC {
        return Err(ArtifactError::BadMagic);
    }
    let version = read_u16(bytes, 8)?;
    if !matches!(version, LEGACY_ARTIFACT_VERSION | ARTIFACT_VERSION) {
        return Err(ArtifactError::UnsupportedVersion(version));
    }
    let header_len = read_u16(bytes, 10)? as usize;
    if header_len != HEADER_BYTES {
        return Err(ArtifactError::Malformed(format!(
            "unsupported embedded artifact header length {header_len}"
        )));
    }
    let total_len = read_u32(bytes, 12)? as usize;
    if total_len < HEADER_BYTES {
        return Err(ArtifactError::Malformed(format!(
            "embedded artifact length {total_len} is smaller than the header"
        )));
    }
    if total_len > bytes.len() {
        return Err(ArtifactError::Truncated("blob"));
    }
    let bytes = &bytes[..total_len];
    let name_len = read_u16(bytes, 16)? as usize;
    let target_len = read_u16(bytes, 18)? as usize;
    let payload_count = read_u16(bytes, 20)? as usize;
    let entry_count = read_u16(bytes, 22)? as usize;
    let compile_options = if version == LEGACY_ARTIFACT_VERSION {
        ArtifactCompileOptions::new()
    } else {
        ArtifactCompileOptions::from_bits(read_u64(bytes, 24)?)?
    };

    let mut cursor = HEADER_BYTES;
    let name = read_str(bytes, cursor, name_len, "bundle name")?;
    cursor += name_len;
    let target = read_str(bytes, cursor, target_len, "target")?;
    cursor += target_len;

    let payload_records = cursor;
    cursor = cursor
        .checked_add(payload_count * PAYLOAD_RECORD_BYTES)
        .ok_or(ArtifactError::TooLarge("payload records"))?;
    require_len(bytes, cursor, "payload records")?;
    let entry_records = cursor;
    cursor = cursor
        .checked_add(entry_count * ENTRY_RECORD_BYTES)
        .ok_or(ArtifactError::TooLarge("entry records"))?;
    require_len(bytes, cursor, "entry records")?;

    let mut payloads = Vec::with_capacity(payload_count);
    for index in 0..payload_count {
        let record = payload_records + index * PAYLOAD_RECORD_BYTES;
        let kind_raw = read_u16(bytes, record)?;
        let kind = ArtifactPayloadKind::from_u16(kind_raw)
            .ok_or(ArtifactError::UnsupportedPayloadKind(kind_raw))?;
        let data_offset = read_u32(bytes, record + 4)? as usize;
        let data_len = read_u32(bytes, record + 8)? as usize;
        let name_offset = read_u32(bytes, record + 12)? as usize;
        let name_len = read_u16(bytes, record + 16)? as usize;
        let name = read_str(bytes, name_offset, name_len, "payload name")?;
        let data = read_slice(bytes, data_offset, data_len, "payload data")?;
        payloads.push(ArtifactPayload {
            kind,
            name,
            bytes: data,
        });
    }

    let mut entries = Vec::with_capacity(entry_count);
    for index in 0..entry_count {
        let record = entry_records + index * ENTRY_RECORD_BYTES;
        let kind_raw = read_u16(bytes, record)?;
        let kind = ArtifactEntryKind::from_u16(kind_raw)
            .ok_or(ArtifactError::UnsupportedEntryKind(kind_raw))?;
        let flags = read_u16(bytes, record + 2)?;
        let metadata = if flags & 1 != 0 {
            Some(read_u64(bytes, record + 4)?)
        } else {
            None
        };
        let symbol_offset = read_u32(bytes, record + 12)? as usize;
        let symbol_len = read_u16(bytes, record + 16)? as usize;
        let symbol = read_str(bytes, symbol_offset, symbol_len, "entry symbol")?;
        entries.push(ArtifactEntry {
            symbol,
            kind,
            metadata,
        });
    }

    Ok(ArtifactBundle {
        name,
        target,
        compile_options,
        payloads,
        entries,
    })
}

pub fn artifact_blob_total_len(bytes: &[u8]) -> Result<usize, ArtifactError> {
    require_len(bytes, HEADER_BYTES, "header")?;
    if bytes[0..8] != ARTIFACT_MAGIC {
        return Err(ArtifactError::BadMagic);
    }
    let total_len = read_u32(bytes, 12)? as usize;
    if total_len < HEADER_BYTES {
        return Err(ArtifactError::Malformed(format!(
            "embedded artifact length {total_len} is smaller than the header"
        )));
    }
    if total_len > bytes.len() {
        return Err(ArtifactError::Truncated("blob"));
    }
    Ok(total_len)
}

#[cfg(feature = "object-read")]
pub fn read_artifact_bundles_from_object_bytes(
    bytes: &[u8],
) -> Result<Vec<OwnedArtifactBundle>, ArtifactError> {
    use object::{Object, ObjectSection};

    let file = object::File::parse(bytes).map_err(|e| ArtifactError::Object(e.to_string()))?;
    let mut bundles = Vec::new();
    for section in file.sections() {
        let name = section
            .name()
            .map_err(|e| ArtifactError::Object(e.to_string()))?;
        if name != ARTIFACT_SECTION_NAME {
            continue;
        }
        let data = section
            .data()
            .map_err(|e| ArtifactError::Object(e.to_string()))?;
        bundles.extend(parse_artifact_section(data)?.into_iter().map(Into::into));
    }
    Ok(bundles)
}

/// Wrap an artifact section blob in a relocatable host object file.
///
/// The object contains a single `.oxart` data section. When
/// `anchor_symbol` is given, a global symbol with that name is defined at
/// the start of the section. The anchor matters for *library* crates:
/// their artifact object becomes a member of an `.rlib` archive, and a
/// linker only extracts an archive member when the member defines a
/// symbol that resolves an outstanding undefined reference. Without a
/// defined symbol the member is silently skipped and the bundle never
/// reaches the final binary. Host-side code (the `#[cuda_module]` macro)
/// emits a matching reference to the anchor to force the extraction.
/// `SHF_GNU_RETAIN` on the section additionally protects it from
/// `--gc-sections` once the member has been linked in.
#[cfg(feature = "object-write")]
pub fn build_host_object_for_target(
    section_data: &[u8],
    target: &str,
    anchor_symbol: Option<&str>,
) -> Result<Vec<u8>, ArtifactError> {
    if section_data.is_empty() {
        return Err(ArtifactError::EmptyPayload);
    }

    match anchor_symbol {
        Some(anchor_symbol) => build_host_object_with_section(
            ARTIFACT_SECTION_NAME,
            section_data,
            target,
            &[(anchor_symbol, false)],
        ),
        None => build_host_object_with_section(ARTIFACT_SECTION_NAME, section_data, target, &[]),
    }
}

/// Wrap an artifact blob with a target-specific anchor and a weak legacy alias.
///
/// New owner-filter-aware macros reference the target-specific symbol. The
/// weak package-level alias keeps older macro expansions link-compatible while
/// avoiding duplicate-symbol failures when one package has several targets.
#[cfg(feature = "object-write")]
pub fn build_host_object_for_target_with_legacy_anchor(
    section_data: &[u8],
    target: &str,
    anchor_symbol: &str,
    legacy_anchor_symbol: &str,
) -> Result<Vec<u8>, ArtifactError> {
    if section_data.is_empty() {
        return Err(ArtifactError::EmptyPayload);
    }
    build_host_object_with_section(
        ARTIFACT_SECTION_NAME,
        section_data,
        target,
        &[(anchor_symbol, false), (legacy_anchor_symbol, true)],
    )
}

/// Build a host object that only defines an artifact link-anchor symbol.
///
/// The CUDA backend uses this when an owner filter deliberately suppresses a
/// crate's device artifact. Older `#[cuda_module]` expansions can still
/// contain the legacy anchor reference, so the linker needs a matching weak
/// definition even though no `.oxart` section should be embedded. Keeping the
/// placeholder in a separate section means artifact discovery correctly sees
/// no device bundle.
#[cfg(feature = "object-write")]
pub fn build_host_anchor_object_for_target(
    target: &str,
    anchor_symbol: &str,
) -> Result<Vec<u8>, ArtifactError> {
    if anchor_symbol.is_empty() {
        return Err(ArtifactError::Malformed(
            "embedded artifact anchor symbol is empty".to_string(),
        ));
    }

    build_host_object_with_section(
        ARTIFACT_ANCHOR_SECTION_NAME,
        &[0],
        target,
        &[(anchor_symbol, true)],
    )
}

#[cfg(feature = "object-write")]
fn build_host_object_with_section(
    section_name: &str,
    section_data: &[u8],
    target: &str,
    anchor_symbols: &[(&str, bool)],
) -> Result<Vec<u8>, ArtifactError> {
    use object::write::{Object, Symbol, SymbolSection};
    use object::{SectionFlags, SectionKind, SymbolFlags, SymbolKind, SymbolScope};

    let target = HostObjectTarget::parse(target)?;
    let mut object = Object::new(target.format, target.architecture, target.endianness);
    let section_id = object.add_section(
        Vec::new(),
        section_name.as_bytes().to_vec(),
        SectionKind::Data,
    );
    let section = object.section_mut(section_id);
    section.set_data(section_data.to_vec(), 8);
    section.flags = SectionFlags::Elf {
        sh_flags: elf::SHF_ALLOC | elf::SHF_GNU_RETAIN,
    };

    for (anchor_symbol, weak) in anchor_symbols {
        // Global binding so the symbol can satisfy undefined references
        // from other objects (that is what triggers archive extraction);
        // `Linkage` scope so it stays hidden and never leaks into the
        // dynamic symbol table of the final binary.
        object.add_symbol(Symbol {
            name: anchor_symbol.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Data,
            scope: SymbolScope::Linkage,
            weak: *weak,
            section: SymbolSection::Section(section_id),
            flags: SymbolFlags::None,
        });
    }

    object
        .write()
        .map_err(|e| ArtifactError::Object(e.to_string()))
}

#[cfg(feature = "object-write")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostObjectTarget {
    format: object::BinaryFormat,
    architecture: object::Architecture,
    endianness: object::Endianness,
}

#[cfg(feature = "object-write")]
impl HostObjectTarget {
    fn parse(target: &str) -> Result<Self, ArtifactError> {
        let target = target.to_ascii_lowercase();
        let (architecture, endianness) = if target.starts_with("x86_64")
            || target.starts_with("amd64")
            || target.starts_with("x86-64")
        {
            (object::Architecture::X86_64, object::Endianness::Little)
        } else if target.starts_with("aarch64") || target.starts_with("arm64") {
            (object::Architecture::Aarch64, object::Endianness::Little)
        } else {
            return Err(ArtifactError::UnsupportedHostTarget(target));
        };

        let format = if target.contains("linux") {
            object::BinaryFormat::Elf
        } else {
            return Err(ArtifactError::UnsupportedHostTarget(target));
        };

        Ok(Self {
            format,
            architecture,
            endianness,
        })
    }
}

#[cfg(feature = "object-write")]
mod elf {
    pub const SHF_ALLOC: u64 = 0x2;
    pub const SHF_GNU_RETAIN: u64 = 0x20_0000;
}

fn validate_spec(spec: &ArtifactBundleSpec<'_>) -> Result<(), ArtifactError> {
    if spec.name.is_empty() {
        return Err(ArtifactError::EmptyBundleName);
    }
    if spec.target.is_empty() {
        return Err(ArtifactError::EmptyTarget);
    }
    if spec.payloads.is_empty() {
        return Err(ArtifactError::EmptyPayload);
    }
    for payload in &spec.payloads {
        if payload.name.is_empty() {
            return Err(ArtifactError::EmptyPayloadName);
        }
        if payload.bytes.is_empty() {
            return Err(ArtifactError::EmptyPayload);
        }
    }
    for entry in &spec.entries {
        if entry.symbol.is_empty() {
            return Err(ArtifactError::EmptyEntrySymbol);
        }
    }
    Ok(())
}

fn checked_u16(value: usize, field: &'static str) -> Result<u16, ArtifactError> {
    u16::try_from(value).map_err(|_| ArtifactError::TooLarge(field))
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, ArtifactError> {
    u32::try_from(value).map_err(|_| ArtifactError::TooLarge(field))
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(bytes);
}

fn align_vec(out: &mut Vec<u8>, alignment: usize) {
    let rem = out.len() % alignment;
    if rem != 0 {
        out.resize(out.len() + alignment - rem, 0);
    }
}

fn read_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: usize,
    field: &'static str,
) -> Result<&'a [u8], ArtifactError> {
    let end = offset
        .checked_add(len)
        .ok_or(ArtifactError::TooLarge(field))?;
    bytes
        .get(offset..end)
        .ok_or(ArtifactError::Truncated(field))
}

fn read_str<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: usize,
    field: &'static str,
) -> Result<&'a str, ArtifactError> {
    let bytes = read_slice(bytes, offset, len, field)?;
    core::str::from_utf8(bytes).map_err(|_| ArtifactError::InvalidUtf8(field))
}

fn require_len(bytes: &[u8], len: usize, field: &'static str) -> Result<(), ArtifactError> {
    if bytes.len() < len {
        Err(ArtifactError::Truncated(field))
    } else {
        Ok(())
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ArtifactError> {
    let bytes = read_slice(bytes, offset, 2, "u16")?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ArtifactError> {
    let bytes = read_slice(bytes, offset, 4, "u32")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ArtifactError> {
    let bytes = read_slice(bytes, offset, 8, "u64")?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn write_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_blob() -> Vec<u8> {
        build_artifact_blob(
            &ArtifactBundleSpec::new("demo", "sm_90")
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Ptx,
                    "demo.ptx",
                    b"ptx",
                ))
                .with_entry(ArtifactEntrySpec::new("hello", ArtifactEntryKind::Kernel)),
        )
        .unwrap()
    }

    fn sample_payload_record_start() -> usize {
        HEADER_BYTES + "demo".len() + "sm_90".len()
    }

    #[test]
    fn artifact_blob_round_trips_ptx_payload() {
        let blob = sample_blob();
        assert_eq!(read_u16(&blob, 8).unwrap(), LEGACY_ARTIFACT_VERSION);
        let bundles = parse_artifact_section(&blob).unwrap();

        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name, "demo");
        assert_eq!(bundles[0].target, "sm_90");
        assert!(bundles[0].compile_options.fma_contraction_enabled());
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::Ptx),
            Some(&b"ptx"[..])
        );
        assert_eq!(
            bundles[0].entry("hello").unwrap().kind,
            ArtifactEntryKind::Kernel
        );
    }

    #[test]
    fn artifact_blob_round_trips_non_ptx_payload_kinds() {
        let blob = build_artifact_blob(
            &ArtifactBundleSpec::new("demo", "sm_90")
                .with_compile_options(ArtifactCompileOptions::new().with_fma_contraction(false))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::NvvmIr,
                    "demo.ll",
                    b"nvvm ir",
                ))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Ltoir,
                    "demo.ltoir",
                    b"ltoir",
                ))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Cubin,
                    "demo.cubin",
                    b"cubin",
                )),
        )
        .unwrap();
        assert_eq!(read_u16(&blob, 8).unwrap(), ARTIFACT_VERSION);
        let bundles = parse_artifact_section(&blob).unwrap();

        assert_eq!(bundles.len(), 1);
        assert!(!bundles[0].compile_options.fma_contraction_enabled());
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::NvvmIr),
            Some(&b"nvvm ir"[..])
        );
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::Ltoir),
            Some(&b"ltoir"[..])
        );
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::Cubin),
            Some(&b"cubin"[..])
        );
    }

    #[test]
    fn legacy_v1_bundle_defaults_to_fma_contraction() {
        let mut blob = sample_blob();
        write_u16(&mut blob, 8, LEGACY_ARTIFACT_VERSION);
        // Version 1 reserved these bytes. A new reader must ignore them and
        // preserve the historical default rather than assigning new meaning.
        write_u64(&mut blob, 24, OPTION_NO_FMA_CONTRACTION);

        let bundle = parse_artifact_blob(&blob).unwrap();
        assert!(bundle.compile_options.fma_contraction_enabled());
        assert_eq!(
            bundle.compile_options.debug_policy(),
            ArtifactDebugPolicy::None
        );
    }

    #[test]
    fn version_2_rejects_unknown_compile_option_bits() {
        let mut blob = sample_blob();
        write_u16(&mut blob, 8, ARTIFACT_VERSION);
        write_u64(&mut blob, 24, 1 << 63);

        assert!(matches!(
            parse_artifact_blob(&blob),
            Err(ArtifactError::UnsupportedCompileOptions(bits)) if bits == 1 << 63
        ));
    }

    #[test]
    fn compile_options_sidecar_round_trips_fma_and_debug_policies() {
        for allow_fma_contraction in [true, false] {
            for debug in [
                ArtifactDebugPolicy::None,
                ArtifactDebugPolicy::LineTables,
                ArtifactDebugPolicy::Full,
            ] {
                let expected = ArtifactCompileOptions::new()
                    .with_fma_contraction(allow_fma_contraction)
                    .with_debug_policy(debug);
                let parsed =
                    ArtifactCompileOptions::from_sidecar_text(expected.sidecar_text()).unwrap();
                assert_eq!(parsed, expected);
            }
        }
        assert!(ArtifactCompileOptions::from_sidecar_text("fma-contraction=off\n").is_err());
    }

    #[test]
    fn artifact_blob_round_trips_each_debug_policy() {
        for debug in [ArtifactDebugPolicy::LineTables, ArtifactDebugPolicy::Full] {
            let expected = ArtifactCompileOptions::new()
                .with_fma_contraction(false)
                .with_debug_policy(debug);
            let blob = build_artifact_blob(
                &ArtifactBundleSpec::new("debug", "sm_90")
                    .with_compile_options(expected)
                    .with_payload(ArtifactPayloadSpec::new(
                        ArtifactPayloadKind::NvvmIr,
                        "debug.ll",
                        b"nvvm ir",
                    )),
            )
            .unwrap();

            assert_eq!(read_u16(&blob, 8).unwrap(), ARTIFACT_VERSION);
            assert_eq!(
                parse_artifact_blob(&blob).unwrap().compile_options,
                expected
            );
        }
    }

    #[test]
    fn version_2_rejects_conflicting_debug_bits() {
        let mut blob = sample_blob();
        write_u16(&mut blob, 8, ARTIFACT_VERSION);
        write_u64(&mut blob, 24, OPTION_DEBUG_MASK);

        assert!(matches!(
            parse_artifact_blob(&blob),
            Err(ArtifactError::UnsupportedCompileOptions(bits)) if bits == OPTION_DEBUG_MASK
        ));
    }

    #[test]
    fn artifact_section_ignores_trailing_zero_padding() {
        let mut section = sample_blob();
        section.extend_from_slice(&[0; HEADER_BYTES]);

        let bundles = parse_artifact_section(&section).unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name, "demo");
    }

    #[test]
    fn artifact_section_parses_concatenated_blobs() {
        let mut first = build_artifact_blob(&ArtifactBundleSpec::new("a", "sm_80").with_payload(
            ArtifactPayloadSpec::new(ArtifactPayloadKind::Ptx, "a.ptx", b"a"),
        ))
        .unwrap();
        write_u16(&mut first, 8, LEGACY_ARTIFACT_VERSION);
        let second = build_artifact_blob(&ArtifactBundleSpec::new("b", "sm_90").with_payload(
            ArtifactPayloadSpec::new(ArtifactPayloadKind::Ptx, "b.ptx", b"b"),
        ))
        .unwrap();

        let mut section = first;
        section.extend_from_slice(&second);

        let bundles = parse_artifact_section(&section).unwrap();
        assert_eq!(
            bundles.iter().map(|bundle| bundle.name).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn artifact_section_rejects_truncated_blob_without_panicking() {
        let mut blob = sample_blob();
        let oversized_len = (blob.len() + 1) as u32;
        write_u32(&mut blob, 12, oversized_len);

        let error = parse_artifact_section(&blob).unwrap_err();
        assert_eq!(error, ArtifactError::Truncated("blob"));
    }

    #[test]
    fn artifact_blob_rejects_total_len_smaller_than_header() {
        let mut blob = sample_blob();
        write_u32(&mut blob, 12, (HEADER_BYTES - 1) as u32);

        let error = parse_artifact_blob(&blob).unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::Malformed(message) if message.contains("smaller than the header")
        ));
    }

    #[test]
    fn artifact_blob_rejects_invalid_utf8_payload_name() {
        let mut blob = sample_blob();
        let payload_record = sample_payload_record_start();
        let payload_name_offset = read_u32(&blob, payload_record + 12).unwrap() as usize;
        blob[payload_name_offset] = 0xff;

        let error = parse_artifact_blob(&blob).unwrap_err();
        assert_eq!(error, ArtifactError::InvalidUtf8("payload name"));
    }

    #[test]
    fn artifact_blob_rejects_unknown_payload_kind() {
        let mut blob = sample_blob();
        write_u16(&mut blob, sample_payload_record_start(), 0xffff);

        let error = parse_artifact_blob(&blob).unwrap_err();
        assert_eq!(error, ArtifactError::UnsupportedPayloadKind(0xffff));
    }

    #[test]
    fn artifact_section_name_is_portable() {
        assert!(ARTIFACT_SECTION_NAME.len() <= 8);
    }

    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn host_object_round_trips_section_on_supported_formats() {
        let blob = build_artifact_blob(
            &ArtifactBundleSpec::new("demo", "sm_90")
                .with_compile_options(ArtifactCompileOptions::new().with_fma_contraction(false))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Ptx,
                    "demo.ptx",
                    b"ptx",
                )),
        )
        .unwrap();

        for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
            let object = build_host_object_for_target(&blob, target, None).unwrap();
            let bundles = read_artifact_bundles_from_object_bytes(&object).unwrap();
            assert_eq!(bundles.len(), 1);
            assert_eq!(
                bundles[0].payload(ArtifactPayloadKind::Ptx),
                Some(&b"ptx"[..])
            );
            assert!(!bundles[0].compile_options.fma_contraction_enabled());
        }
    }

    /// The anchor symbol must be a *defined* global pointing at the
    /// `.oxart` section. A linker only extracts an rlib archive member if
    /// the member defines a symbol someone references, so an undefined or
    /// missing anchor would reintroduce the dropped-bundle bug.
    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn host_object_defines_requested_anchor_symbol() {
        use object::{Object, ObjectSymbol};

        let blob = sample_blob();
        let bytes =
            build_host_object_for_target(&blob, "x86_64-unknown-linux-gnu", Some("demo_anchor"))
                .unwrap();

        let file = object::File::parse(bytes.as_slice()).unwrap();
        let anchor = file
            .symbols()
            .find(|symbol| symbol.name() == Ok("demo_anchor"))
            .expect("anchor symbol missing from artifact object");
        assert!(anchor.is_definition());
        assert!(anchor.is_global());
        assert_eq!(anchor.address(), 0);

        // The data must still round-trip with the symbol present.
        let bundles = read_artifact_bundles_from_object_bytes(&bytes).unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name, "demo");
    }

    /// Omitting the anchor must keep producing a symbol-free object (the
    /// shape used by tests and any non-rlib embedding).
    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn host_object_without_anchor_has_no_symbols() {
        use object::Object;

        let blob = sample_blob();
        let bytes = build_host_object_for_target(&blob, "x86_64-unknown-linux-gnu", None).unwrap();

        let file = object::File::parse(bytes.as_slice()).unwrap();
        assert_eq!(file.symbols().count(), 0);
    }

    /// A filtered crate still needs to satisfy the host macro's anchor
    /// reference, but it must not look like it contains a device artifact.
    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn anchor_only_object_defines_symbol_without_artifact_section() {
        use object::{Object, ObjectSymbol};

        let bytes = build_host_anchor_object_for_target(
            "x86_64-unknown-linux-gnu",
            "filtered_crate_anchor",
        )
        .unwrap();
        let file = object::File::parse(bytes.as_slice()).unwrap();
        let anchor = file
            .symbols()
            .find(|symbol| symbol.name() == Ok("filtered_crate_anchor"))
            .expect("anchor symbol missing from placeholder object");

        assert!(anchor.is_definition());
        assert!(anchor.is_global());
        assert!(anchor.is_weak());
        assert!(
            read_artifact_bundles_from_object_bytes(&bytes)
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(feature = "object-write")]
    #[test]
    fn anchor_only_object_rejects_empty_symbol() {
        assert!(matches!(
            build_host_anchor_object_for_target("x86_64-unknown-linux-gnu", ""),
            Err(ArtifactError::Malformed(_))
        ));
    }

    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn target_anchor_object_also_defines_weak_legacy_alias() {
        use object::{Object, ObjectSymbol};

        let blob = sample_blob();
        let bytes = build_host_object_for_target_with_legacy_anchor(
            &blob,
            "x86_64-unknown-linux-gnu",
            "target_anchor",
            "legacy_anchor",
        )
        .unwrap();
        let file = object::File::parse(bytes.as_slice()).unwrap();
        let target = file
            .symbols()
            .find(|symbol| symbol.name() == Ok("target_anchor"))
            .expect("target-specific anchor missing");
        let legacy = file
            .symbols()
            .find(|symbol| symbol.name() == Ok("legacy_anchor"))
            .expect("legacy anchor alias missing");

        assert!(target.is_definition());
        assert!(!target.is_weak());
        assert!(legacy.is_definition());
        assert!(legacy.is_weak());
        assert_eq!(
            read_artifact_bundles_from_object_bytes(&bytes)
                .unwrap()
                .len(),
            1
        );
    }

    /// A strong undefined reference must pull an archive member whose matching
    /// compatibility alias is weak; otherwise old macros could link but lose
    /// the actual `.oxart` payload.
    #[cfg(all(
        target_os = "linux",
        target_arch = "x86_64",
        feature = "object-read",
        feature = "object-write"
    ))]
    #[test]
    fn weak_legacy_alias_extracts_artifact_from_static_archive() {
        use std::process::Command;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "oxide_artifact_weak_archive_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();

        let object = build_host_object_for_target_with_legacy_anchor(
            &sample_blob(),
            "x86_64-unknown-linux-gnu",
            "target_anchor",
            "legacy_anchor",
        )
        .unwrap();
        let object_path = root.join("artifact.o");
        let archive_path = root.join("libartifact.a");
        let source_path = root.join("main.c");
        let binary_path = root.join("app");
        std::fs::write(&object_path, object).unwrap();
        std::fs::write(
            &source_path,
            b"extern const unsigned char legacy_anchor;\nint main(void) { return legacy_anchor; }\n",
        )
        .unwrap();

        let ar = Command::new("ar")
            .args(["crs"])
            .arg(&archive_path)
            .arg(&object_path)
            .status()
            .expect("`ar` is required for the static-archive anchor test");
        assert!(ar.success(), "failed to create static archive");
        let cc = Command::new("cc")
            .arg(&source_path)
            .arg(&archive_path)
            .arg("-Wl,-z,noexecstack")
            .arg("-o")
            .arg(&binary_path)
            .status()
            .expect("a C linker driver is required for the anchor test");
        assert!(
            cc.success(),
            "weak anchor did not resolve the host reference"
        );

        let executable = std::fs::read(&binary_path).unwrap();
        let bundles = read_artifact_bundles_from_object_bytes(&executable).unwrap();
        assert_eq!(bundles.len(), 1, "archive payload was not extracted");
        assert_eq!(bundles[0].name, "demo");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(feature = "object-write")]
    #[test]
    fn host_object_rejects_non_cuda_host_targets() {
        let blob = sample_blob();

        for target in ["powerpc64le-unknown-linux-gnu", "x86_64-apple-darwin"] {
            assert!(matches!(
                build_host_object_for_target(&blob, target, None),
                Err(ArtifactError::UnsupportedHostTarget(_))
            ));
        }
    }
}
