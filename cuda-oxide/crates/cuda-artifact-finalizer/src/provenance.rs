/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io;
#[cfg(not(unix))]
use std::io::{Read, Seek, SeekFrom};
use std::time::SystemTime;

use crate::FinalizerError;

const DIGEST_DOMAIN: &[u8] = b"cuda-oxide/artifact-finalizer/digest/v1";
// Bump this recipe version whenever tool invocation, option translation,
// input ordering, output validation, or other output-affecting semantics
// change. Cache keys and the cargo-oxide/backend handshake rely on it.
const RECIPE: &[u8] = b"cuda-oxide/artifact-finalizer/recipe/v2";

/// Exact compiler inputs discovered alongside the loaded CUDA tools.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolProvenance {
    /// SHA-256 of the exact loaded libNVVM file, if it can be proven.
    pub libnvvm_sha256: Option<[u8; 32]>,
    /// SHA-256 of the exact loaded nvJitLink file, if it can be proven.
    pub nvjitlink_sha256: Option<[u8; 32]>,
    /// SHA-256 of the exact libdevice bytes added to libNVVM.
    pub libdevice_sha256: [u8; 32],
}

/// Stable identity of the open file descriptor whose contents were hashed.
///
/// Named fields keep the parent/child handoff explicit and independently
/// inspectable. Optional Unix fields are absent on other platforms.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolFileIdentity {
    /// File length in bytes.
    pub length: u64,
    /// Whole seconds in the modification timestamp since the Unix epoch.
    pub modified_seconds: u64,
    /// Nanosecond component of the modification timestamp.
    pub modified_nanoseconds: u32,
    /// Unix device identifier, when available.
    pub device: Option<u64>,
    /// Unix inode number, when available.
    pub inode: Option<u64>,
    /// Unix change-time seconds, when available.
    pub change_time_seconds: Option<i64>,
    /// Unix change-time nanoseconds, when available.
    pub change_time_nanoseconds: Option<i64>,
}

impl ToolFileIdentity {
    pub(crate) fn capture(file: &File) -> Option<Self> {
        let metadata = file.metadata().ok()?;
        let modified = metadata
            .modified()
            .ok()?
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?;

        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        Some(Self {
            length: metadata.len(),
            modified_seconds: modified.as_secs(),
            modified_nanoseconds: modified.subsec_nanos(),
            #[cfg(unix)]
            device: Some(metadata.dev()),
            #[cfg(not(unix))]
            device: None,
            #[cfg(unix)]
            inode: Some(metadata.ino()),
            #[cfg(not(unix))]
            inode: None,
            #[cfg(unix)]
            change_time_seconds: Some(metadata.ctime()),
            #[cfg(not(unix))]
            change_time_seconds: None,
            #[cfg(unix)]
            change_time_nanoseconds: Some(metadata.ctime_nsec()),
            #[cfg(not(unix))]
            change_time_nanoseconds: None,
        })
    }

    pub(crate) fn matches_file(&self, file: &File) -> bool {
        Self::capture(file).as_ref() == Some(self)
    }

    /// Whether every Unix identity field (device, inode, change time) is
    /// present. Length and modification time alone are too weak to prove the
    /// descriptor is unchanged, so digest reuse must rehash without them.
    pub(crate) fn has_unix_identity(&self) -> bool {
        self.device.is_some()
            && self.inode.is_some()
            && self.change_time_seconds.is_some()
            && self.change_time_nanoseconds.is_some()
    }
}

/// A content digest bound to the exact open-file identity that produced it.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedToolProvenance {
    /// SHA-256 of the complete tool DSO.
    pub sha256: [u8; 32],
    /// Identity of the retained descriptor used to compute `sha256`.
    pub file: ToolFileIdentity,
}

/// Versioned cargo-oxide to codegen-backend materializer handoff.
///
/// The child still opens each CUDA DSO itself. Matching descriptor identities
/// allow it to reuse the parent's content digest without rereading the tool.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MaterializerHandshakeV1 {
    /// Wire-format version; must equal [`Self::VERSION`].
    pub version: u32,
    /// Combined semantic provenance used by Cargo's fingerprint.
    pub provenance_sha256: [u8; 32],
    /// Exact libNVVM content and retained-file identity.
    pub libnvvm: PinnedToolProvenance,
    /// Exact nvJitLink content and retained-file identity.
    pub nvjitlink: PinnedToolProvenance,
    /// SHA-256 of the libdevice bytes supplied to libNVVM.
    pub libdevice_sha256: [u8; 32],
}

impl MaterializerHandshakeV1 {
    /// Current handshake wire-format version.
    pub const VERSION: u32 = 1;

    /// Construct a self-consistent v1 handshake from named inputs.
    pub fn new(
        libnvvm: PinnedToolProvenance,
        nvjitlink: PinnedToolProvenance,
        libdevice_sha256: [u8; 32],
    ) -> Self {
        Self {
            version: Self::VERSION,
            provenance_sha256: common_provenance_digest(
                &libnvvm.sha256,
                &nvjitlink.sha256,
                &libdevice_sha256,
            ),
            libnvvm,
            nvjitlink,
            libdevice_sha256,
        }
    }

    /// Whether the version and combined digest agree with all named fields.
    pub fn has_consistent_provenance(&self) -> bool {
        self.version == Self::VERSION
            && self.provenance_sha256
                == common_provenance_digest(
                    &self.libnvvm.sha256,
                    &self.nvjitlink.sha256,
                    &self.libdevice_sha256,
                )
    }
}

/// Stable identity of the finalizer algorithm itself.
pub fn recipe_digest() -> [u8; 32] {
    Sha256::digest(RECIPE).into()
}

pub(crate) fn digest_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Run one CUDA-tool operation between exact checks of its retained DSO.
///
/// An unavailable initial digest is allowed for runtime fallback, whose cache
/// is disabled separately. When an exact digest exists (and therefore may be
/// part of Cargo's fingerprint or a cache key), the caller revalidates the
/// retained descriptor identity before and after the operation. The initial
/// digest remains bound to that descriptor, and the post-check runs even when
/// the tool call itself returned an error.
pub(crate) fn with_revalidated_tool_identity<T>(
    tool: &'static str,
    expected: Option<[u8; 32]>,
    mut current_digest: impl FnMut() -> Option<[u8; 32]>,
    operation: impl FnOnce() -> Result<T, FinalizerError>,
) -> Result<T, FinalizerError> {
    let Some(expected) = expected else {
        return operation();
    };
    if current_digest() != Some(expected) {
        return Err(FinalizerError::ToolIdentityChanged { tool });
    }

    let result = operation();
    if current_digest() != Some(expected) {
        return Err(FinalizerError::ToolIdentityChanged { tool });
    }
    result
}

/// Hash a retained CUDA-tool descriptor and reject a concurrent replacement.
pub(crate) fn digest_file_handle(file: &File) -> io::Result<[u8; 32]> {
    digest_file_handle_with_post_read(file, || {})
}

fn digest_file_handle_with_post_read(
    file: &File,
    post_read: impl FnOnce(),
) -> io::Result<[u8; 32]> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "tool fingerprint input is not a regular file",
        ));
    }
    let snapshot = FileSnapshot::capture(&metadata)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;

        let mut offset = 0_u64;
        loop {
            let read = match file.read_at(&mut buffer, offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            offset = offset
                .checked_add(read as u64)
                .ok_or_else(|| io::Error::other("tool file length overflow"))?;
        }
    }

    #[cfg(not(unix))]
    {
        let mut reader = file.try_clone()?;
        reader.seek(SeekFrom::Start(0))?;
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }

    let digest = hasher.finalize().into();
    post_read();
    if FileSnapshot::capture(&file.metadata()?)? != snapshot {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tool file changed while it was fingerprinted",
        ));
    }
    Ok(digest)
}

#[derive(Debug, Eq, PartialEq)]
struct FileSnapshot {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_time: (i64, i64),
}

impl FileSnapshot {
    fn capture(metadata: &fs::Metadata) -> io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            change_time: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

/// Unambiguous ordered digest used for recipes, provenance, and artifacts.
pub(crate) struct StableDigest {
    hasher: Sha256,
}

impl StableDigest {
    pub(crate) fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(DIGEST_DOMAIN);
        Self { hasher }
    }

    pub(crate) fn field(mut self, tag: &str, value: impl AsRef<[u8]>) -> Self {
        let tag = tag.as_bytes();
        let value = value.as_ref();
        self.hasher.update([1]);
        self.hasher.update(length_prefix(tag.len()));
        self.hasher.update(tag);
        self.hasher.update(length_prefix(value.len()));
        self.hasher.update(value);
        self
    }

    pub(crate) fn finish(mut self) -> [u8; 32] {
        self.hasher.update([0xff]);
        self.hasher.finalize().into()
    }
}

fn length_prefix(length: usize) -> [u8; 8] {
    u64::try_from(length)
        .expect("digest fields cannot exceed u64::MAX bytes")
        .to_be_bytes()
}

pub(crate) fn common_provenance_digest(
    libnvvm: &[u8; 32],
    nvjitlink: &[u8; 32],
    libdevice: &[u8; 32],
) -> [u8; 32] {
    StableDigest::new()
        .field("recipe", recipe_digest())
        .field("libnvvm-sha256", libnvvm)
        .field("libnvjitlink-sha256", nvjitlink)
        .field("libdevice-sha256", libdevice)
        .finish()
}

pub(crate) fn compiler_provenance_digest(libnvvm: &[u8; 32], libdevice: &[u8; 32]) -> [u8; 32] {
    StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"nvvm-ir-to-ltoir")
        .field("libnvvm-sha256", libnvvm)
        .field("libdevice-sha256", libdevice)
        .finish()
}

pub(crate) fn linker_provenance_digest(nvjitlink: &[u8; 32]) -> [u8; 32] {
    StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"ltoir-to-output")
        .field("libnvjitlink-sha256", nvjitlink)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn provenance_is_route_specific_and_content_sensitive() {
        let nvvm = [1; 32];
        let linker = [2; 32];
        let libdevice = [3; 32];
        assert_ne!(
            compiler_provenance_digest(&nvvm, &libdevice),
            linker_provenance_digest(&linker)
        );
        assert_ne!(
            common_provenance_digest(&nvvm, &linker, &libdevice),
            common_provenance_digest(&nvvm, &linker, &[4; 32])
        );
    }

    #[test]
    fn stable_digest_distinguishes_field_boundaries_and_order() {
        let left = StableDigest::new()
            .field("input", b"ab")
            .field("input", b"c")
            .finish();
        let different_boundaries = StableDigest::new()
            .field("input", b"a")
            .field("input", b"bc")
            .finish();
        let reversed = StableDigest::new()
            .field("input", b"c")
            .field("input", b"ab")
            .finish();
        assert_ne!(left, different_boundaries);
        assert_ne!(left, reversed);
    }

    fn example_file_identity() -> ToolFileIdentity {
        ToolFileIdentity {
            length: 123,
            modified_seconds: 456,
            modified_nanoseconds: 789,
            device: Some(10),
            inode: Some(11),
            change_time_seconds: Some(12),
            change_time_nanoseconds: Some(13),
        }
    }

    #[test]
    fn materializer_handshake_serializes_with_named_fields() {
        let file = example_file_identity();
        let handshake = MaterializerHandshakeV1::new(
            PinnedToolProvenance {
                sha256: [1; 32],
                file,
            },
            PinnedToolProvenance {
                sha256: [2; 32],
                file,
            },
            [3; 32],
        );
        let json = serde_json::to_string(&handshake).unwrap();
        for field in [
            "version",
            "provenance_sha256",
            "libnvvm",
            "nvjitlink",
            "libdevice_sha256",
            "sha256",
            "file",
            "length",
            "modified_seconds",
            "change_time_seconds",
        ] {
            assert!(
                json.contains(&format!("\"{field}\"")),
                "handshake JSON omitted named field {field}: {json}"
            );
        }
        assert_eq!(
            serde_json::from_str::<MaterializerHandshakeV1>(&json).unwrap(),
            handshake
        );
        let with_unknown = json.replacen('{', "{\"family.0\":0,", 1);
        assert!(serde_json::from_str::<MaterializerHandshakeV1>(&with_unknown).is_err());
    }

    #[test]
    fn materializer_handshake_rejects_changed_fields_and_versions() {
        let file = example_file_identity();
        let mut handshake = MaterializerHandshakeV1::new(
            PinnedToolProvenance {
                sha256: [1; 32],
                file,
            },
            PinnedToolProvenance {
                sha256: [2; 32],
                file,
            },
            [3; 32],
        );
        assert!(handshake.has_consistent_provenance());
        handshake.libnvvm.sha256[0] ^= 1;
        assert!(!handshake.has_consistent_provenance());
        handshake.libnvvm.sha256[0] ^= 1;
        handshake.version += 1;
        assert!(!handshake.has_consistent_provenance());
    }

    #[test]
    fn post_call_tool_change_rejects_the_operation_result() {
        let expected = [7; 32];
        let changed = [8; 32];
        let checks = Cell::new(0_u32);
        let operation_calls = Cell::new(0_u32);

        let error = with_revalidated_tool_identity(
            "test CUDA tool",
            Some(expected),
            || {
                let check = checks.get();
                checks.set(check + 1);
                Some(if check == 0 { expected } else { changed })
            },
            || {
                operation_calls.set(operation_calls.get() + 1);
                Ok(b"must not be accepted".to_vec())
            },
        )
        .expect_err("a post-call identity change must discard successful output");

        assert!(matches!(
            error,
            FinalizerError::ToolIdentityChanged {
                tool: "test CUDA tool"
            }
        ));
        assert_eq!(checks.get(), 2, "identity must be checked on both sides");
        assert_eq!(operation_calls.get(), 1, "the seam models a post-call race");
    }

    #[test]
    fn tool_digest_stays_bound_to_open_file_after_path_replacement() {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-artifact-finalizer-provenance-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tool.so");
        let replacement = directory.join("replacement.so");
        std::fs::write(&path, b"tool version one").unwrap();
        std::fs::write(&replacement, b"tool version two").unwrap();

        let opened = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(&replacement, &path).unwrap();

        assert_eq!(
            digest_file_handle(&opened).unwrap(),
            digest_bytes(b"tool version one")
        );
        assert_eq!(
            digest_file_handle(&File::open(&path).unwrap()).unwrap(),
            digest_bytes(b"tool version two")
        );

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn tool_digest_changes_after_in_place_content_change() {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-artifact-finalizer-in-place-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tool.so");
        std::fs::write(&path, b"tool bytes version one").unwrap();
        #[cfg(unix)]
        let original_inode = {
            use std::os::unix::fs::MetadataExt;
            path.metadata().unwrap().ino()
        };
        let original = digest_file_handle(&File::open(&path).unwrap()).unwrap();

        std::fs::write(&path, b"tool bytes version two").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(path.metadata().unwrap().ino(), original_inode);
        }
        let changed = digest_file_handle(&File::open(&path).unwrap()).unwrap();

        assert_eq!(original, digest_bytes(b"tool bytes version one"));
        assert_eq!(changed, digest_bytes(b"tool bytes version two"));
        assert_ne!(original, changed);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn metadata_change_between_read_and_validation_rejects_digest() {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-artifact-finalizer-mid-hash-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tool.so");
        std::fs::write(&path, b"tool bytes before hash").unwrap();
        let opened = File::open(&path).unwrap();

        let error = digest_file_handle_with_post_read(&opened, || {
            std::fs::write(&path, b"tool bytes changed during hash and now longer").unwrap();
        })
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "tool file changed while it was fingerprinted"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
