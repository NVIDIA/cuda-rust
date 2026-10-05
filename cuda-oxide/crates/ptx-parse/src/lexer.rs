/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::ParseError;
use std::ops::Range;

/// The lossless lexical class of one PTX source token.
///
/// This deliberately stops short of assigning ISA semantics. In particular,
/// [`Word`](Self::Word) retains identifiers, instruction heads, numeric
/// literals, and implementation-defined spellings without rejecting syntax
/// introduced by a newer PTX version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Whitespace,
    LineComment,
    BlockComment,
    QuotedString,
    Preprocessor,
    Word,
    Punctuation,
    Unknown,
}

impl TokenKind {
    /// Whether this token carries no PTX syntax of its own.
    pub fn is_trivia(self) -> bool {
        matches!(
            self,
            Self::Whitespace | Self::LineComment | Self::BlockComment
        )
    }
}

/// One lossless token represented by its exact byte range in the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    kind: TokenKind,
    start: u32,
    end: u32,
}

impl Token {
    fn new(kind: TokenKind, span: Range<usize>) -> Self {
        Self {
            kind,
            start: span
                .start
                .try_into()
                .expect("the lexer rejects PTX sources larger than u32::MAX"),
            end: span
                .end
                .try_into()
                .expect("the lexer rejects PTX sources larger than u32::MAX"),
        }
    }

    pub fn kind(&self) -> TokenKind {
        self.kind
    }

    pub fn span(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    pub fn text<'source>(&self, source: &'source str) -> &'source str {
        &source[self.span()]
    }
}

pub(crate) fn lex(source: &str) -> Result<Vec<Token>, ParseError> {
    if source.len() > u32::MAX as usize {
        return Err(ParseError::SourceTooLarge {
            bytes: source.len(),
        });
    }
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        let start = cursor;
        let (kind, end) = if bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            (TokenKind::Whitespace, cursor)
        } else if bytes[cursor..].starts_with(b"//") {
            cursor += 2;
            while bytes.get(cursor).is_some_and(|byte| *byte != b'\n') {
                cursor += 1;
            }
            (TokenKind::LineComment, cursor)
        } else if bytes[cursor..].starts_with(b"/*") {
            cursor += 2;
            while cursor < bytes.len() && !bytes[cursor..].starts_with(b"*/") {
                cursor += source[cursor..]
                    .chars()
                    .next()
                    .expect("cursor is before the end of the source")
                    .len_utf8();
            }
            if cursor == bytes.len() {
                return Err(ParseError::UnterminatedBlockComment { offset: start });
            }
            cursor += 2;
            (TokenKind::BlockComment, cursor)
        } else if bytes[cursor] == b'"' {
            cursor += 1;
            let mut closed = false;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'\\' if cursor + 1 < bytes.len() => {
                        cursor += 1;
                        cursor += source[cursor..]
                            .chars()
                            .next()
                            .expect("an escape has a following character")
                            .len_utf8();
                    }
                    b'"' => {
                        cursor += 1;
                        closed = true;
                        break;
                    }
                    b'\n' | b'\r' => break,
                    _ => {
                        cursor += source[cursor..]
                            .chars()
                            .next()
                            .expect("cursor is before the end of the source")
                            .len_utf8();
                    }
                }
            }
            if !closed {
                return Err(ParseError::UnterminatedQuotedString { offset: start });
            }
            (TokenKind::QuotedString, cursor)
        } else if bytes[cursor] == b'#' && is_preprocessor_start(source, cursor) {
            cursor = preprocessor_end(source, cursor + 1);
            (TokenKind::Preprocessor, cursor)
        } else if is_word_byte(bytes[cursor]) {
            cursor += 1;
            while bytes.get(cursor).is_some_and(|byte| is_word_byte(*byte)) {
                cursor += 1;
            }
            (TokenKind::Word, cursor)
        } else if bytes[cursor..].starts_with(b"::") {
            cursor += 2;
            (TokenKind::Punctuation, cursor)
        } else if bytes[cursor].is_ascii() && !bytes[cursor].is_ascii_control() {
            cursor += 1;
            (TokenKind::Punctuation, cursor)
        } else {
            cursor += source[cursor..]
                .chars()
                .next()
                .expect("cursor is before the end of the source")
                .len_utf8();
            (TokenKind::Unknown, cursor)
        };
        tokens.push(Token::new(kind, start..end));
    }

    debug_assert_eq!(tokens.first().map_or(0, |token| token.start), 0);
    debug_assert_eq!(
        tokens.last().map_or(0, |token| token.end),
        source.len() as u32
    );
    debug_assert!(
        tokens
            .windows(2)
            .all(|tokens| tokens[0].end == tokens[1].start)
    );
    Ok(tokens)
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'%' | b'.')
}

fn is_preprocessor_start(source: &str, offset: usize) -> bool {
    let line_start = source[..offset]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    source[line_start..offset]
        .bytes()
        .all(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\x0c'))
}

fn preprocessor_end(source: &str, mut cursor: usize) -> usize {
    let bytes = source.as_bytes();
    loop {
        let Some(relative_newline) = bytes[cursor..].iter().position(|byte| *byte == b'\n') else {
            return bytes.len();
        };
        let newline = cursor + relative_newline;
        let before_newline = newline
            .checked_sub(1)
            .filter(|offset| bytes[*offset] == b'\r');
        let last = before_newline.unwrap_or(newline).checked_sub(1);
        if last.is_some_and(|offset| bytes[offset] == b'\\') {
            cursor = newline + 1;
        } else {
            return newline;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_every_source_byte_without_normalizing() {
        let source = "  .file 1 \"a;{b}\" // λ\n/* x */ mov.u32 %r1, 0;\n";
        let tokens = lex(source).unwrap();
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text(source))
                .collect::<String>(),
            source
        );
        assert!(
            tokens
                .windows(2)
                .all(|tokens| tokens[0].end == tokens[1].start)
        );
        assert!(
            tokens
                .iter()
                .all(|token| source.is_char_boundary(token.start as usize)
                    && source.is_char_boundary(token.end as usize))
        );
        assert_eq!(std::mem::size_of::<Token>(), 12);
    }

    #[test]
    fn keeps_preprocessor_logical_lines_opaque() {
        let source = " #define FOO(x) \\\n  x; { }\nmov.u32 %r1, 0;";
        let tokens = lex(source).unwrap();
        let preprocessors = tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Preprocessor)
            .collect::<Vec<_>>();
        assert_eq!(preprocessors.len(), 1);
        assert_eq!(preprocessors[0].text(source), "#define FOO(x) \\\n  x; { }");
    }

    #[test]
    fn reports_the_start_of_unterminated_regions() {
        assert_eq!(
            lex("x /* no end").unwrap_err(),
            ParseError::UnterminatedBlockComment { offset: 2 }
        );
        assert_eq!(
            lex(".file \"no end").unwrap_err(),
            ParseError::UnterminatedQuotedString { offset: 6 }
        );
    }

    #[test]
    fn distinguishes_double_colons_from_label_terminators() {
        let source = "L0: tcgen05.wait::ld.sync.aligned;";
        let tokens = lex(source).unwrap();
        assert_eq!(
            tokens
                .iter()
                .filter(|token| !token.kind().is_trivia())
                .map(|token| token.text(source))
                .collect::<Vec<_>>(),
            ["L0", ":", "tcgen05.wait", "::", "ld.sync.aligned", ";"]
        );
    }
}
