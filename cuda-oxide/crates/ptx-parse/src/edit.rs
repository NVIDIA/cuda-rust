/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fmt;
use std::ops::Range;

/// Which side of replaced or inserted text owns an ambiguous source boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapBias {
    Left,
    Right,
}

/// A monotone byte-offset map produced while applying an [`EditScript`].
///
/// Unedited bytes map one-to-one. Offsets inside replaced text are mapped to
/// either edge of the replacement according to [`MapBias`], which lets callers
/// conservatively project complete source ranges through normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditMap {
    original_len: usize,
    output_len: usize,
    edits: Vec<MappedEdit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MappedEdit {
    original: Range<usize>,
    output: Range<usize>,
}

/// Text and source mapping returned by [`EditScript::apply_with_map`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedEdits {
    pub text: String,
    pub map: EditMap,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditScript {
    edits: Vec<Edit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Edit {
    range: Range<usize>,
    replacement: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditError {
    InvalidRange {
        range: Range<usize>,
    },
    Overlap {
        first: Range<usize>,
        second: Range<usize>,
    },
    OutOfBounds {
        range: Range<usize>,
        source_len: usize,
    },
    NonCharacterBoundary {
        offset: usize,
    },
}

impl fmt::Display for EditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { range } => write!(
                formatter,
                "PTX edit range {}..{} is reversed",
                range.start, range.end
            ),
            Self::Overlap { first, second } => write!(
                formatter,
                "PTX edits {}..{} and {}..{} conflict",
                first.start, first.end, second.start, second.end
            ),
            Self::OutOfBounds { range, source_len } => write!(
                formatter,
                "PTX edit range {}..{} exceeds source length {source_len}",
                range.start, range.end
            ),
            Self::NonCharacterBoundary { offset } => {
                write!(
                    formatter,
                    "PTX edit offset {offset} is not a UTF-8 boundary"
                )
            }
        }
    }
}

impl std::error::Error for EditError {}

impl EditScript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn insert(&mut self, offset: usize, text: impl Into<String>) -> Result<(), EditError> {
        self.replace(offset..offset, text)
    }

    pub fn delete(&mut self, range: Range<usize>) -> Result<(), EditError> {
        self.replace(range, "")
    }

    pub fn replace(
        &mut self,
        range: Range<usize>,
        replacement: impl Into<String>,
    ) -> Result<(), EditError> {
        if range.start > range.end {
            return Err(EditError::InvalidRange { range });
        }
        if let Some(existing) = self
            .edits
            .iter()
            .find(|edit| ranges_conflict(&edit.range, &range))
        {
            return Err(EditError::Overlap {
                first: existing.range.clone(),
                second: range,
            });
        }
        self.edits.push(Edit {
            range,
            replacement: replacement.into(),
        });
        Ok(())
    }

    pub fn apply(&self, source: &str) -> Result<String, EditError> {
        Ok(self.apply_with_map(source)?.text)
    }

    /// Apply all edits and retain a bidirectional map between the input and
    /// output byte coordinate spaces.
    pub fn apply_with_map(&self, source: &str) -> Result<AppliedEdits, EditError> {
        let mut edits: Vec<&Edit> = self.edits.iter().collect();
        edits.sort_by_key(|edit| (edit.range.start, edit.range.end));

        for edit in &edits {
            if edit.range.end > source.len() {
                return Err(EditError::OutOfBounds {
                    range: edit.range.clone(),
                    source_len: source.len(),
                });
            }
            for offset in [edit.range.start, edit.range.end] {
                if !source.is_char_boundary(offset) {
                    return Err(EditError::NonCharacterBoundary { offset });
                }
            }
        }

        let replacement_bytes = edits
            .iter()
            .map(|edit| edit.replacement.len())
            .sum::<usize>();
        let removed_bytes = edits
            .iter()
            .map(|edit| edit.range.end - edit.range.start)
            .sum::<usize>();
        let mut output = String::with_capacity(source.len() + replacement_bytes - removed_bytes);
        let mut mapped_edits = Vec::with_capacity(edits.len());
        let mut cursor = 0usize;
        for edit in edits {
            output.push_str(&source[cursor..edit.range.start]);
            let output_start = output.len();
            output.push_str(&edit.replacement);
            let output_end = output.len();
            mapped_edits.push(MappedEdit {
                original: edit.range.clone(),
                output: output_start..output_end,
            });
            cursor = edit.range.end;
        }
        output.push_str(&source[cursor..]);
        let output_len = output.len();
        Ok(AppliedEdits {
            text: output,
            map: EditMap {
                original_len: source.len(),
                output_len,
                edits: mapped_edits,
            },
        })
    }
}

impl EditMap {
    pub fn original_len(&self) -> usize {
        self.original_len
    }

    pub fn output_len(&self) -> usize {
        self.output_len
    }

    pub fn original_to_output(&self, offset: usize, bias: MapBias) -> Option<usize> {
        (offset <= self.original_len)
            .then(|| map_offset(offset, bias, &self.edits, Direction::Forward))
    }

    pub fn output_to_original(&self, offset: usize, bias: MapBias) -> Option<usize> {
        (offset <= self.output_len)
            .then(|| map_offset(offset, bias, &self.edits, Direction::Reverse))
    }

    /// Smallest output range that conservatively covers `original`.
    pub fn original_range_to_output(&self, original: Range<usize>) -> Option<Range<usize>> {
        (original.start <= original.end && original.end <= self.original_len).then(|| {
            self.original_to_output(original.start, MapBias::Left)
                .expect("validated original offset")
                ..self
                    .original_to_output(original.end, MapBias::Right)
                    .expect("validated original offset")
        })
    }

    /// Smallest original range that conservatively covers `output`.
    pub fn output_range_to_original(&self, output: Range<usize>) -> Option<Range<usize>> {
        (output.start <= output.end && output.end <= self.output_len).then(|| {
            self.output_to_original(output.start, MapBias::Left)
                .expect("validated output offset")
                ..self
                    .output_to_original(output.end, MapBias::Right)
                    .expect("validated output offset")
        })
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Reverse,
}

fn map_offset(offset: usize, bias: MapBias, edits: &[MappedEdit], direction: Direction) -> usize {
    let index = edits.partition_point(|edit| source_range(edit, direction).end < offset);
    if let Some(edit) = edits.get(index) {
        let source = source_range(edit, direction);
        let destination = destination_range(edit, direction);
        if offset >= source.start && offset <= source.end {
            if source.is_empty() && offset == source.start {
                return match bias {
                    MapBias::Left => destination.start,
                    MapBias::Right => destination.end,
                };
            }
            if offset == source.start {
                return destination.start;
            }
            if offset == source.end {
                return destination.end;
            }
            return match bias {
                MapBias::Left => destination.start,
                MapBias::Right => destination.end,
            };
        }
    }
    let completed = edits.partition_point(|edit| source_range(edit, direction).end <= offset);
    let delta = edits.get(completed.wrapping_sub(1)).map_or(0, |edit| {
        let source = source_range(edit, direction);
        let destination = destination_range(edit, direction);
        destination.end as isize - source.end as isize
    });
    offset.saturating_add_signed(delta)
}

fn source_range(edit: &MappedEdit, direction: Direction) -> &Range<usize> {
    match direction {
        Direction::Forward => &edit.original,
        Direction::Reverse => &edit.output,
    }
}

fn destination_range(edit: &MappedEdit, direction: Direction) -> &Range<usize> {
    match direction {
        Direction::Forward => &edit.output,
        Direction::Reverse => &edit.original,
    }
}

fn ranges_conflict(first: &Range<usize>, second: &Range<usize>) -> bool {
    match (first.is_empty(), second.is_empty()) {
        (false, false) => first.start < second.end && second.start < first.end,
        (true, true) => first.start == second.start,
        (true, false) => first.start >= second.start && first.start <= second.end,
        (false, true) => second.start >= first.start && second.start <= first.end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_non_overlapping_edits_in_source_order() {
        let mut edits = EditScript::new();
        edits.replace(6..12, "PTX").unwrap();
        edits.insert(0, "lossless ").unwrap();
        edits.delete(12..13).unwrap();
        assert_eq!(edits.apply("hello source!").unwrap(), "lossless hello PTX");
    }

    #[test]
    fn rejects_ambiguous_or_invalid_edits() {
        let mut edits = EditScript::new();
        edits.delete(2..5).unwrap();
        assert!(matches!(
            edits.insert(5, "x"),
            Err(EditError::Overlap { .. })
        ));
        assert!(matches!(
            edits.replace(Range { start: 8, end: 7 }, "x"),
            Err(EditError::InvalidRange { .. })
        ));
        let mut duplicate_insert = EditScript::new();
        duplicate_insert.insert(1, "x").unwrap();
        assert!(matches!(
            duplicate_insert.insert(1, "y"),
            Err(EditError::Overlap { .. })
        ));
    }

    #[test]
    fn validates_source_boundaries_when_applying() {
        let mut out_of_bounds = EditScript::new();
        out_of_bounds.delete(2..20).unwrap();
        assert!(matches!(
            out_of_bounds.apply("short"),
            Err(EditError::OutOfBounds { .. })
        ));

        let mut inside_utf8 = EditScript::new();
        inside_utf8.insert(1, "x").unwrap();
        assert!(matches!(
            inside_utf8.apply("λ"),
            Err(EditError::NonCharacterBoundary { offset: 1 })
        ));
    }

    #[test]
    fn maps_offsets_and_ranges_across_insert_replace_and_delete() {
        let mut edits = EditScript::new();
        edits.insert(0, "pre-").unwrap();
        edits.replace(2..4, "replacement").unwrap();
        edits.delete(6..8).unwrap();
        let applied = edits.apply_with_map("abcdefgh").unwrap();
        assert_eq!(applied.text, "pre-abreplacementef");
        assert_eq!(applied.map.original_to_output(0, MapBias::Left), Some(0));
        assert_eq!(applied.map.original_to_output(0, MapBias::Right), Some(4));
        assert_eq!(applied.map.original_range_to_output(2..4), Some(6..17));
        assert_eq!(applied.map.output_range_to_original(6..17), Some(2..4));
        assert_eq!(applied.map.original_range_to_output(4..6), Some(17..19));
        assert_eq!(applied.map.original_range_to_output(6..8), Some(19..19));
        assert_eq!(applied.map.output_to_original(20, MapBias::Left), None);
    }
}
