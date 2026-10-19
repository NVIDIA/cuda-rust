/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Locate an anchor without borrowing dynamic-loader metadata or reading memory
//! through an unbounded raw pointer. Procfs supplies the mapped file's identity;
//! metadata validation and artifact reads use the same open file description.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

#[derive(Debug)]
struct Mapping<'a> {
    range: &'a str,
    major: u64,
    minor: u64,
    inode: u64,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn mapping_at<'a>(maps: &'a [u8], address: usize) -> io::Result<Mapping<'a>> {
    for line in maps
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let mut fields = line
            .split(u8::is_ascii_whitespace)
            .filter(|field| !field.is_empty());
        let field = |value: Option<&'a [u8]>| -> io::Result<&'a str> {
            std::str::from_utf8(value.ok_or_else(|| invalid("missing mapping field"))?)
                .map_err(|_| invalid("invalid mapping metadata"))
        };
        let range = field(fields.next())?;
        let (start, end) = range
            .split_once('-')
            .ok_or_else(|| invalid("invalid mapping range"))?;
        let start =
            usize::from_str_radix(start, 16).map_err(|_| invalid("invalid mapping start"))?;
        let end = usize::from_str_radix(end, 16).map_err(|_| invalid("invalid mapping end"))?;
        if !(start..end).contains(&address) {
            continue;
        }
        let _permissions = fields
            .next()
            .ok_or_else(|| invalid("missing mapping permissions"))?;
        let _offset = fields
            .next()
            .ok_or_else(|| invalid("missing mapping offset"))?;
        let device = field(fields.next())?;
        let (major, minor) = device
            .split_once(':')
            .ok_or_else(|| invalid("invalid mapping device"))?;
        let major =
            u64::from_str_radix(major, 16).map_err(|_| invalid("invalid mapping device major"))?;
        let minor =
            u64::from_str_radix(minor, 16).map_err(|_| invalid("invalid mapping device minor"))?;
        let inode = field(fields.next())?
            .parse::<u64>()
            .map_err(|_| invalid("invalid mapping inode"))?;
        if inode == 0 {
            return Err(invalid("artifact anchor is not in a file-backed mapping"));
        }
        // Do not parse the displayed filename: procfs escapes some bytes and
        // filenames may contain whitespace. map_files gives the native path.
        return Ok(Mapping {
            range,
            major,
            minor,
            inode,
        });
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "artifact anchor has no mapped image",
    ))
}

fn mapped_identity_matches(file: &File, mapping: &Mapping<'_>) -> io::Result<bool> {
    // SAFETY: the descriptor stays open and no references to the mapping are created.
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            1,
            libc::PROT_NONE,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    if address == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let result = fs::read("/proc/self/maps").and_then(|maps| {
        let probe = mapping_at(&maps, address.addr())?;
        if (probe.major, probe.minor, probe.inode) != (mapping.major, mapping.minor, mapping.inode)
        {
            return Ok(false);
        }
        // Btrfs subvolumes can share procfs device and inode numbers.
        let map_files = Path::new("/proc/self/map_files");
        Ok(fs::read_link(map_files.join(probe.range))?
            == fs::read_link(map_files.join(mapping.range))?)
    });
    // SAFETY: this releases only the successful mapping above, including on read errors.
    unsafe { libc::munmap(address, 1) };
    result
}

fn read_verified(path: &Path, mapping: &Mapping<'_>) -> io::Result<Vec<u8>> {
    // A pathname can change after map_files was read. Avoid blocking on a
    // replaced FIFO before its type and identity can be checked below.
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
        .open(path)?;
    // Compare both file identities through procfs; stat can report different IDs.
    if !file.metadata()?.is_file() || !mapped_identity_matches(&file, mapping)? {
        return Err(invalid(
            "artifact image no longer identifies the mapped file",
        ));
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub(super) fn read(anchor: &u8) -> io::Result<Vec<u8>> {
    let maps = fs::read("/proc/self/maps")?;
    let mapping = mapping_at(&maps, std::ptr::from_ref(anchor).addr())?;
    let path = fs::read_link(Path::new("/proc/self/map_files").join(mapping.range))?;
    if !path.is_absolute() {
        return Err(invalid("mapped artifact image has no absolute file path"));
    }
    read_verified(&path, &mapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedded::artifact_bundles_containing;
    use oxide_artifacts::{ArtifactBundleSpec, ArtifactPayloadKind, ArtifactPayloadSpec};
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn mappings_require_the_address_and_a_file_identity() {
        let maps = b"1000-2000 r--p 00000000 08:02 37 /a path\\012with escapes\n2000-3000 rw-p 00000000 00:00 0 [heap]\n";
        let mapping = mapping_at(maps, 0x1000).unwrap();
        assert_eq!((mapping.major, mapping.minor, mapping.inode), (8, 2, 37));
        assert_eq!(
            mapping_at(maps, 0x2000).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            mapping_at(maps, 0x3000).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert!(mapping_at(b"1000-2000 r--p 0 not-a-device 1", 0x1000).is_err());
    }

    #[test]
    fn anonymous_anchor_is_an_error_without_executable_fallback() {
        let anchor = Box::new(0_u8);
        assert!(artifact_bundles_containing(&anchor).is_err());
    }

    #[test]
    fn image_read_rejects_a_different_file_identity() {
        static ANCHOR: u8 = 0;
        let maps = fs::read("/proc/self/maps").unwrap();
        let mut mapping = mapping_at(&maps, std::ptr::from_ref(&ANCHOR).addr()).unwrap();
        let path = std::env::current_exe().unwrap();
        assert!(read_verified(&path, &mapping).is_ok());
        mapping.inode ^= 1;
        assert_eq!(
            read_verified(&path, &mapping).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn opened_file_mapping_preserves_device_and_inode_checks() {
        static ANCHOR: u8 = 0;
        let maps = fs::read("/proc/self/maps").unwrap();
        let mut mapping = mapping_at(&maps, std::ptr::from_ref(&ANCHOR).addr()).unwrap();
        let path = std::env::current_exe().unwrap();
        let file = File::open(&path).unwrap();
        assert!(mapped_identity_matches(&file, &mapping).unwrap());
        mapping.major ^= 1;
        assert!(!mapped_identity_matches(&file, &mapping).unwrap());
        assert_eq!(
            read_verified(&path, &mapping).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        mapping.major ^= 1;
        mapping.minor ^= 1;
        assert!(!mapped_identity_matches(&file, &mapping).unwrap());
        mapping.minor ^= 1;
        mapping.inode ^= 1;
        assert!(!mapped_identity_matches(&file, &mapping).unwrap());
    }

    #[test]
    fn shared_library_discovery() {
        const CHILD: &str = "CUDA_OXIDE_MAPPED_IMAGE_TEST";
        if let Ok(mode) = std::env::var(CHILD) {
            let path = Path::new(OsStr::from_bytes(b"./plugin space\\name\nline\xff.so"));
            // SAFETY: the fixture contains only constant artifact data. Keep
            // the library alive for every borrowed symbol and artifact read.
            let library = unsafe { libloading::Library::new(path) }.unwrap();
            let symbol = unsafe { library.get::<*const u8>(b"artifact_anchor\0") }.unwrap();
            // SAFETY: this exported initialized byte remains mapped while
            // `library` lives, and the reference never escapes this scope.
            let anchor = unsafe { &**symbol };
            match mode.as_str() {
                "changed-directory" => std::env::set_current_dir("..").unwrap(),
                "renamed" => fs::rename(path, "renamed.so").unwrap(),
                "deleted" => fs::remove_file(path).unwrap(),
                "replaced" => {
                    let maps = fs::read("/proc/self/maps").unwrap();
                    let mut mapping = mapping_at(&maps, std::ptr::from_ref(anchor).addr()).unwrap();
                    fs::rename(path, "replaced.so").unwrap();
                    fs::copy("fixture.so", path).unwrap();
                    assert_eq!(
                        read_verified(path, &mapping).unwrap_err().kind(),
                        io::ErrorKind::InvalidData
                    );
                    let replacement = File::open(path).unwrap();
                    // Simulate equal inode numbers in distinct Btrfs subvolumes.
                    mapping.inode = replacement.metadata().unwrap().ino();
                    assert!(!mapped_identity_matches(&replacement, &mapping).unwrap());
                    assert!(matches!(
                        read_verified(path, &mapping),
                        Err(error) if error.kind() == io::ErrorKind::InvalidData
                    ));
                }
                "relative" => {}
                _ => panic!("unknown test mode"),
            }
            let result = artifact_bundles_containing(anchor);
            if mode == "deleted" {
                assert!(
                    result.is_err(),
                    "deleted image must not select another binary"
                );
            } else {
                let bundles = result.unwrap();
                assert_eq!(
                    bundles
                        .iter()
                        .map(|bundle| bundle.name.as_str())
                        .collect::<Vec<_>>(),
                    ["plugin", "other"]
                );
                assert_eq!(
                    bundles[0].payload(ArtifactPayloadKind::Ptx),
                    Some(b"fixture ptx".as_slice())
                );
            }
            return;
        }
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "cuda-oxide-mapped-image-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        let mut bytes = Vec::new();
        for name in ["plugin", "other"] {
            bytes.extend(
                oxide_artifacts::build_artifact_blob(
                    &ArtifactBundleSpec::new(name, "sm_80").with_payload(ArtifactPayloadSpec::new(
                        ArtifactPayloadKind::Ptx,
                        "test",
                        b"fixture ptx",
                    )),
                )
                .unwrap(),
            );
        }
        let source = format!(
            "__attribute__((section(\".oxart\"),used)) const unsigned char artifact_anchor[] = {{{}}};\n",
            bytes
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        fs::write(dir.join("plugin.c"), source).unwrap();
        let compiler = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
        let built = Command::new(compiler)
            .current_dir(&dir)
            .args(["-shared", "-fPIC", "plugin.c", "-o", "fixture.so"])
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        for mode in [
            "relative",
            "changed-directory",
            "renamed",
            "deleted",
            "replaced",
        ] {
            fs::copy(
                dir.join("fixture.so"),
                dir.join(OsStr::from_bytes(b"plugin space\\name\nline\xff.so")),
            )
            .unwrap();
            let child = Command::new(std::env::current_exe().unwrap())
                .current_dir(&dir)
                .env(CHILD, mode)
                .args([
                    "--exact",
                    "embedded::mapped_image::tests::shared_library_discovery",
                    "--nocapture",
                ])
                .output()
                .unwrap();
            assert!(
                child.status.success(),
                "{mode}:\n{}\n{}",
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
