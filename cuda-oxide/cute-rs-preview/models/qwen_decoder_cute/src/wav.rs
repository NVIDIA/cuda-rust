/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Mono IEEE-float WAV output without quantizing the decoded waveform.
use crate::operator::Result;
use std::{
    io::{BufWriter, Write},
    path::Path,
};

/// Build RIFF, format, fact, and data headers for mono 32-bit IEEE-float samples.
///
/// `sample_rate` is in Hz. Rejects zero rate, empty/nonfinite samples, and
/// lengths or byte rates exceeding the WAV fields' 32-bit bounds before writing.
fn header(sample_rate: u32, samples: &[f32]) -> Result<[u8; 56]> {
    if sample_rate == 0 || samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
        return Err("WAV requires a positive sample rate and nonempty finite samples".into());
    }
    let byte_rate = sample_rate
        .checked_mul(4)
        .ok_or("WAV sample rate overflows byte rate")?;
    let sample_count = u32::try_from(samples.len())?;
    let data_bytes = sample_count
        .checked_mul(4)
        .ok_or("WAV data exceeds RIFF limit")?;
    let riff_bytes = data_bytes.checked_add(48).ok_or("WAV exceeds RIFF limit")?;
    let mut bytes = [0; 56];
    bytes[..4].copy_from_slice(b"RIFF");
    bytes[4..8].copy_from_slice(&riff_bytes.to_le_bytes());
    bytes[8..16].copy_from_slice(b"WAVEfmt ");
    bytes[16..20].copy_from_slice(&16u32.to_le_bytes());
    bytes[20..22].copy_from_slice(&3u16.to_le_bytes());
    bytes[22..24].copy_from_slice(&1u16.to_le_bytes());
    bytes[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    bytes[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    bytes[32..34].copy_from_slice(&4u16.to_le_bytes());
    bytes[34..36].copy_from_slice(&32u16.to_le_bytes());
    bytes[36..40].copy_from_slice(b"fact");
    bytes[40..44].copy_from_slice(&4u32.to_le_bytes());
    bytes[44..48].copy_from_slice(&sample_count.to_le_bytes());
    bytes[48..52].copy_from_slice(b"data");
    bytes[52..56].copy_from_slice(&data_bytes.to_le_bytes());
    Ok(bytes)
}

/// Validate, write little-endian sample bits, and flush the destination.
///
/// Samples are neither clipped nor quantized. Validation leaves the writer
/// untouched; later I/O failures can leave a partial WAV stream.
fn encode(output: &mut impl Write, sample_rate: u32, samples: &[f32]) -> Result<()> {
    output.write_all(&header(sample_rate, samples)?)?;
    for sample in samples {
        output.write_all(&sample.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}

/// Write a mono float WAV at `sample_rate` Hz, preserving each finite f32 sample.
///
/// Validates the waveform before creating or truncating `path`. Filesystem and
/// write errors propagate; a failure after creation can leave a partial file.
pub fn write(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<()> {
    header(sample_rate, samples)?;
    encode(
        &mut BufWriter::new(std::fs::File::create(path)?),
        sample_rate,
        samples,
    )
}

/// In-memory WAV format and validation checks, requiring no audio device.
#[cfg(test)]
mod tests {
    use super::*;
    /// Check RIFF chunk sizes, mono float metadata, and exact sample bit preservation.
    #[test]
    fn float_wav_preserves_sample_bits_and_contains_consistent_chunk_lengths() {
        let mut bytes = Vec::new();
        encode(&mut bytes, 24_000, &[0.0, -0.5, 1.0]).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize,
            bytes.len() - 8
        );
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 3);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            24_000
        );
        assert_eq!(&bytes[36..40], b"fact");
        assert_eq!(u32::from_le_bytes(bytes[44..48].try_into().unwrap()), 3);
        assert_eq!(&bytes[48..52], b"data");
        assert_eq!(u32::from_le_bytes(bytes[52..56].try_into().unwrap()), 12);
        assert_eq!(
            &bytes[56..],
            &[0u32, (-0.5f32).to_bits(), 1.0f32.to_bits()]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
    /// Verify invalid waveforms fail before emitting header or sample bytes.
    #[test]
    fn rejects_empty_nonfinite_and_invalid_sample_rate_before_writing() {
        let mut bytes = Vec::new();
        for (rate, samples) in [
            (0, &[0.0][..]),
            (24_000, &[][..]),
            (24_000, &[f32::NAN][..]),
            (u32::MAX, &[0.0][..]),
        ] {
            assert!(encode(&mut bytes, rate, samples).is_err());
            assert!(bytes.is_empty());
        }
    }
}
