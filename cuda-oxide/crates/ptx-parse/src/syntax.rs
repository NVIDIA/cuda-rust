/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::{Token, TokenKind};
use std::ops::Range;

/// Stable index of a PTX statement in [`crate::Document::statements`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StatementId(usize);

impl StatementId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Stable index of a lexical PTX scope in [`crate::Document::scopes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeId(usize);

impl ScopeId {
    pub const ROOT: Self = Self(0);

    pub fn index(self) -> usize {
        self.0
    }
}

/// The structural class of one PTX statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementKind {
    Directive,
    Instruction,
    CallableHeader,
    Label,
    Preprocessor,
    Unknown,
}

/// One PTX statement backed by exact source and token ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    id: StatementId,
    kind: StatementKind,
    scope: ScopeId,
    span: Range<usize>,
    token_range: Range<usize>,
}

impl Statement {
    pub fn id(&self) -> StatementId {
        self.id
    }

    pub fn kind(&self) -> StatementKind {
        self.kind
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn text<'source>(&self, source: &'source str) -> &'source str {
        &source[self.span.clone()]
    }

    pub(crate) fn token_range(&self) -> Range<usize> {
        self.token_range.clone()
    }
}

/// One lexical scope delimited by structural braces.
///
/// The module root is always [`ScopeId::ROOT`] and has no opening or closing
/// brace. An unclosed scope has no `close_span` and is accompanied by a
/// diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    id: ScopeId,
    parent: Option<ScopeId>,
    header: Option<StatementId>,
    span: Range<usize>,
    open_span: Option<Range<usize>>,
    close_span: Option<Range<usize>>,
}

impl Scope {
    pub fn id(&self) -> ScopeId {
        self.id
    }

    pub fn parent(&self) -> Option<ScopeId> {
        self.parent
    }

    /// The callable or section header which immediately opened this scope.
    pub fn header(&self) -> Option<StatementId> {
        self.header
    }

    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn open_span(&self) -> Option<Range<usize>> {
        self.open_span.clone()
    }

    pub fn close_span(&self) -> Option<Range<usize>> {
        self.close_span.clone()
    }

    pub fn body_span(&self) -> Range<usize> {
        self.open_span.as_ref().map_or(self.span.clone(), |open| {
            open.end
                ..self
                    .close_span
                    .as_ref()
                    .map_or(self.span.end, |close| close.start)
        })
    }
}

/// A recoverable structural problem which did not make later byte offsets
/// ambiguous.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    kind: DiagnosticKind,
    span: Range<usize>,
}

impl Diagnostic {
    pub(crate) fn new(kind: DiagnosticKind, span: Range<usize>) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> DiagnosticKind {
        self.kind
    }

    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticKind {
    UnknownToken,
    UnrecognizedSyntax,
    UnmatchedClosingDelimiter,
    UnterminatedDelimiter,
    UnterminatedStatement,
    MalformedDirective,
    MalformedCallable,
    MalformedInstruction,
}

/// Byte-accounting proof for a parsed document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coverage {
    source_bytes: usize,
    token_bytes: usize,
    non_trivia_bytes: usize,
    recognized_bytes: usize,
    unknown_bytes: usize,
    diagnostic_count: usize,
}

impl Coverage {
    pub(crate) fn add_diagnostics(&mut self, count: usize) {
        self.diagnostic_count += count;
    }

    pub fn source_bytes(self) -> usize {
        self.source_bytes
    }

    pub fn token_bytes(self) -> usize {
        self.token_bytes
    }

    pub fn non_trivia_bytes(self) -> usize {
        self.non_trivia_bytes
    }

    pub fn recognized_bytes(self) -> usize {
        self.recognized_bytes
    }

    pub fn unknown_bytes(self) -> usize {
        self.unknown_bytes
    }

    pub fn diagnostic_count(self) -> usize {
        self.diagnostic_count
    }

    pub fn is_lossless(self) -> bool {
        self.token_bytes == self.source_bytes
    }

    pub fn is_complete(self) -> bool {
        self.is_lossless()
            && self.recognized_bytes == self.non_trivia_bytes
            && self.unknown_bytes == 0
            && self.diagnostic_count == 0
    }
}

pub(crate) struct ParsedSyntax {
    pub statements: Vec<Statement>,
    pub scopes: Vec<Scope>,
    pub diagnostics: Vec<Diagnostic>,
    pub coverage: Coverage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ownership {
    Unowned,
    Recognized,
    Unknown,
}

pub(crate) fn parse(source: &str, tokens: &[Token]) -> ParsedSyntax {
    let mut statements = Vec::new();
    let mut scopes = vec![Scope {
        id: ScopeId::ROOT,
        parent: None,
        header: None,
        span: 0..source.len(),
        open_span: None,
        close_span: None,
    }];
    let mut scope_stack = vec![ScopeId::ROOT];
    let mut diagnostics = Vec::new();
    let mut ownership = vec![Ownership::Unowned; tokens.len()];
    let mut next_scope_header = None;
    let mut cursor = 0usize;

    while let Some(start) = next_significant(tokens, cursor) {
        let token = &tokens[start];
        match token.kind() {
            TokenKind::Preprocessor => {
                next_scope_header = None;
                push_statement(
                    source,
                    tokens,
                    &mut statements,
                    &mut ownership,
                    StatementKind::Preprocessor,
                    *scope_stack.last().expect("the root scope remains open"),
                    start..start + 1,
                );
                cursor = start + 1;
                continue;
            }
            TokenKind::Unknown => {
                next_scope_header = None;
                push_statement(
                    source,
                    tokens,
                    &mut statements,
                    &mut ownership,
                    StatementKind::Unknown,
                    *scope_stack.last().expect("the root scope remains open"),
                    start..start + 1,
                );
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::UnknownToken,
                    span: token.span(),
                });
                cursor = start + 1;
                continue;
            }
            TokenKind::Punctuation if token.text(source) == "{" => {
                ownership[start] = Ownership::Recognized;
                let id = ScopeId(scopes.len());
                let open_span = token.span();
                scopes.push(Scope {
                    id,
                    parent: scope_stack.last().copied(),
                    header: next_scope_header.take(),
                    span: open_span.start..source.len(),
                    open_span: Some(open_span),
                    close_span: None,
                });
                scope_stack.push(id);
                cursor = start + 1;
                continue;
            }
            TokenKind::Punctuation if token.text(source) == "}" => {
                next_scope_header = None;
                if scope_stack.len() == 1 {
                    push_statement(
                        source,
                        tokens,
                        &mut statements,
                        &mut ownership,
                        StatementKind::Unknown,
                        ScopeId::ROOT,
                        start..start + 1,
                    );
                    diagnostics.push(Diagnostic {
                        kind: DiagnosticKind::UnmatchedClosingDelimiter,
                        span: token.span(),
                    });
                } else {
                    ownership[start] = Ownership::Recognized;
                    let scope = scope_stack.pop().expect("a non-root scope is open");
                    let close_span = token.span();
                    scopes[scope.index()].span.end = close_span.end;
                    scopes[scope.index()].close_span = Some(close_span);
                }
                cursor = start + 1;
                continue;
            }
            _ => {}
        }

        let outcome = scan_item(source, tokens, start);
        let token_range = if outcome.end == start {
            start..start + 1
        } else {
            start..outcome.end
        };
        let kind = if outcome.end == start {
            StatementKind::Unknown
        } else {
            classify(source, tokens, token_range.clone(), outcome.terminator)
        };
        let id = push_statement(
            source,
            tokens,
            &mut statements,
            &mut ownership,
            kind,
            *scope_stack.last().expect("the root scope remains open"),
            token_range,
        );
        let span = statements[id.index()].span();
        if kind == StatementKind::Unknown {
            diagnostics.push(Diagnostic {
                kind: outcome
                    .diagnostic
                    .unwrap_or(DiagnosticKind::UnrecognizedSyntax),
                span,
            });
        } else if let Some(kind) = outcome.diagnostic {
            diagnostics.push(Diagnostic { kind, span });
        }
        next_scope_header = (outcome.terminator == Terminator::Brace).then_some(id);
        cursor = outcome.end.max(start + 1);
    }

    for scope in scope_stack.into_iter().skip(1) {
        let open = scopes[scope.index()]
            .open_span
            .as_ref()
            .expect("only non-root scopes remain unclosed");
        diagnostics.push(Diagnostic {
            kind: DiagnosticKind::UnterminatedDelimiter,
            span: open.start..source.len(),
        });
    }

    let token_bytes = tokens.iter().map(|token| token.span().len()).sum();
    let non_trivia_bytes = tokens
        .iter()
        .filter(|token| !token.kind().is_trivia())
        .map(|token| token.span().len())
        .sum();
    let recognized_bytes = tokens
        .iter()
        .zip(&ownership)
        .filter(|(token, owner)| !token.kind().is_trivia() && **owner == Ownership::Recognized)
        .map(|(token, _)| token.span().len())
        .sum();
    let unknown_bytes = non_trivia_bytes - recognized_bytes;
    let diagnostic_count = diagnostics.len();

    debug_assert!(
        tokens
            .iter()
            .zip(&ownership)
            .all(|(token, owner)| { token.kind().is_trivia() || *owner != Ownership::Unowned })
    );

    ParsedSyntax {
        statements,
        scopes,
        diagnostics,
        coverage: Coverage {
            source_bytes: source.len(),
            token_bytes,
            non_trivia_bytes,
            recognized_bytes,
            unknown_bytes,
            diagnostic_count,
        },
    }
}

fn push_statement(
    source: &str,
    tokens: &[Token],
    statements: &mut Vec<Statement>,
    ownership: &mut [Ownership],
    kind: StatementKind,
    scope: ScopeId,
    token_range: Range<usize>,
) -> StatementId {
    let id = StatementId(statements.len());
    let owner = if kind == StatementKind::Unknown {
        Ownership::Unknown
    } else {
        Ownership::Recognized
    };
    for index in token_range.clone() {
        if !tokens[index].kind().is_trivia() {
            debug_assert_eq!(ownership[index], Ownership::Unowned);
            ownership[index] = owner;
        }
    }
    let span = tokens[token_range.start].span().start..tokens[token_range.end - 1].span().end;
    debug_assert!(source.is_char_boundary(span.start) && source.is_char_boundary(span.end));
    statements.push(Statement {
        id,
        kind,
        scope,
        span,
        token_range,
    });
    id
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Terminator {
    Semicolon,
    Line,
    Brace,
    ScopeClose,
    EndOfFile,
    Recovery,
}

struct ScanOutcome {
    end: usize,
    terminator: Terminator,
    diagnostic: Option<DiagnosticKind>,
}

fn scan_item(source: &str, tokens: &[Token], start: usize) -> ScanOutcome {
    let mut cursor = start;
    let mut delimiters = Vec::new();
    let mut last_significant = start;

    while cursor < tokens.len() {
        let token = &tokens[cursor];
        if token.kind().is_trivia() {
            if token.text(source).contains('\n') {
                if delimiters.is_empty() && should_end_at_line(source, tokens, start..cursor) {
                    return ScanOutcome {
                        end: last_significant + 1,
                        terminator: Terminator::Line,
                        diagnostic: None,
                    };
                }
                if !delimiters.is_empty()
                    && next_significant(tokens, cursor + 1)
                        .is_some_and(|next| starts_callable_header(source, tokens, next))
                {
                    return ScanOutcome {
                        end: last_significant + 1,
                        terminator: Terminator::Recovery,
                        diagnostic: Some(DiagnosticKind::UnterminatedDelimiter),
                    };
                }
            }
            cursor += 1;
            continue;
        }
        last_significant = cursor;

        if token.kind() == TokenKind::Preprocessor && cursor != start {
            return ScanOutcome {
                end: cursor,
                terminator: Terminator::Recovery,
                diagnostic: Some(DiagnosticKind::UnterminatedStatement),
            };
        }
        if token.kind() == TokenKind::Unknown {
            return ScanOutcome {
                end: cursor + 1,
                terminator: Terminator::Recovery,
                diagnostic: Some(DiagnosticKind::UnknownToken),
            };
        }

        if token.kind() == TokenKind::Punctuation {
            match token.text(source) {
                ";" => {
                    return ScanOutcome {
                        end: cursor + 1,
                        terminator: Terminator::Semicolon,
                        diagnostic: (!delimiters.is_empty())
                            .then_some(DiagnosticKind::UnterminatedDelimiter),
                    };
                }
                "{" if delimiters.is_empty() && is_braced_header(source, tokens, start..cursor) => {
                    return ScanOutcome {
                        end: cursor,
                        terminator: Terminator::Brace,
                        diagnostic: None,
                    };
                }
                "{" => delimiters.push("}"),
                "[" => delimiters.push("]"),
                "(" => delimiters.push(")"),
                "}" | "]" | ")" => {
                    if delimiters.last().copied() == Some(token.text(source)) {
                        delimiters.pop();
                    } else if delimiters.is_empty() && token.text(source) == "}" {
                        return ScanOutcome {
                            end: cursor,
                            terminator: Terminator::ScopeClose,
                            diagnostic: None,
                        };
                    } else {
                        return ScanOutcome {
                            end: cursor + 1,
                            terminator: Terminator::Recovery,
                            diagnostic: Some(DiagnosticKind::UnmatchedClosingDelimiter),
                        };
                    }
                }
                _ => {}
            }
        }
        cursor += 1;
    }

    ScanOutcome {
        end: last_significant + 1,
        terminator: Terminator::EndOfFile,
        diagnostic: (!delimiters.is_empty()).then_some(DiagnosticKind::UnterminatedDelimiter),
    }
}

fn classify(
    source: &str,
    tokens: &[Token],
    range: Range<usize>,
    terminator: Terminator,
) -> StatementKind {
    let significant = significant_indices(tokens, range.clone());
    let Some(mut cursor) = skip_labels(source, tokens, &significant, 0) else {
        return StatementKind::Unknown;
    };
    if cursor == significant.len() {
        return if matches!(terminator, Terminator::Line | Terminator::ScopeClose) {
            StatementKind::Label
        } else {
            StatementKind::Unknown
        };
    }

    if token_is(source, tokens, significant[cursor], "@") {
        cursor += 1;
        if cursor < significant.len() && token_is(source, tokens, significant[cursor], "!") {
            cursor += 1;
        }
        if cursor >= significant.len() || tokens[significant[cursor]].kind() != TokenKind::Word {
            return StatementKind::Unknown;
        }
        cursor += 1;
    }
    let Some(head) = significant.get(cursor).map(|index| &tokens[*index]) else {
        return StatementKind::Unknown;
    };
    if head.kind() != TokenKind::Word {
        return StatementKind::Unknown;
    }
    let head = head.text(source);
    if head.starts_with('.') {
        if contains_callable_keyword(source, tokens, range) {
            StatementKind::CallableHeader
        } else {
            StatementKind::Directive
        }
    } else if terminator == Terminator::Semicolon && is_instruction_head(head) {
        StatementKind::Instruction
    } else {
        StatementKind::Unknown
    }
}

fn should_end_at_line(source: &str, tokens: &[Token], range: Range<usize>) -> bool {
    let significant = significant_indices(tokens, range.clone());
    if significant.is_empty() {
        return false;
    }
    if skip_labels(source, tokens, &significant, 0) == Some(significant.len()) {
        return true;
    }
    if contains_scope_header_keyword(source, tokens, range) {
        return false;
    }
    let words = significant
        .iter()
        .filter(|index| tokens[**index].kind() == TokenKind::Word)
        .map(|index| tokens[*index].text(source))
        .collect::<Vec<_>>();
    let Some(first) = words.first() else {
        return false;
    };
    if !first.starts_with('.') {
        return false;
    }
    if words.iter().all(|word| is_linkage_prefix(word)) {
        return false;
    }
    !words.iter().any(|word| requires_semicolon(word))
}

fn is_braced_header(source: &str, tokens: &[Token], range: Range<usize>) -> bool {
    contains_scope_header_keyword(source, tokens, range)
}

fn contains_scope_header_keyword(source: &str, tokens: &[Token], range: Range<usize>) -> bool {
    significant_indices(tokens, range).into_iter().any(|index| {
        tokens[index].kind() == TokenKind::Word
            && matches!(tokens[index].text(source), ".entry" | ".func" | ".section")
    })
}

fn contains_callable_keyword(source: &str, tokens: &[Token], range: Range<usize>) -> bool {
    significant_indices(tokens, range).into_iter().any(|index| {
        tokens[index].kind() == TokenKind::Word
            && matches!(tokens[index].text(source), ".entry" | ".func")
    })
}

fn starts_callable_header(source: &str, tokens: &[Token], start: usize) -> bool {
    let mut cursor = start;
    while let Some(token) = tokens.get(cursor) {
        if token.kind().is_trivia() {
            if token.text(source).contains('\n') {
                return false;
            }
            cursor += 1;
            continue;
        }
        if token.kind() != TokenKind::Word {
            return false;
        }
        match token.text(source) {
            ".entry" | ".func" => return true,
            word if is_linkage_prefix(word) => cursor += 1,
            _ => return false,
        }
    }
    false
}

fn is_linkage_prefix(word: &str) -> bool {
    matches!(word, ".visible" | ".extern" | ".weak" | ".common")
}

fn requires_semicolon(word: &str) -> bool {
    matches!(
        word,
        ".reg"
            | ".sreg"
            | ".const"
            | ".global"
            | ".local"
            | ".param"
            | ".shared"
            | ".tex"
            | ".surf"
            | ".branchtargets"
            | ".calltargets"
            | ".callprototype"
            | ".pragma"
    )
}

fn skip_labels(
    source: &str,
    tokens: &[Token],
    significant: &[usize],
    mut cursor: usize,
) -> Option<usize> {
    while cursor + 1 < significant.len()
        && tokens[significant[cursor]].kind() == TokenKind::Word
        && token_is(source, tokens, significant[cursor + 1], ":")
        && (cursor + 2 == significant.len()
            || !token_is(source, tokens, significant[cursor + 2], ":"))
    {
        cursor += 2;
    }
    if cursor < significant.len() && token_is(source, tokens, significant[cursor], ":") {
        None
    } else {
        Some(cursor)
    }
}

fn significant_indices(tokens: &[Token], range: Range<usize>) -> Vec<usize> {
    range
        .filter(|index| !tokens[*index].kind().is_trivia())
        .collect()
}

fn next_significant(tokens: &[Token], cursor: usize) -> Option<usize> {
    (cursor..tokens.len()).find(|index| !tokens[*index].kind().is_trivia())
}

fn token_is(source: &str, tokens: &[Token], index: usize, expected: &str) -> bool {
    tokens[index].text(source) == expected
}

fn is_instruction_head(head: &str) -> bool {
    head.as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;

    fn parse_source(source: &str) -> ParsedSyntax {
        let tokens = lexer::lex(source).unwrap();
        parse(source, &tokens)
    }

    #[test]
    fn classifies_multiline_headers_statements_and_scopes() {
        let parsed = parse_source(
            ".version 8.7\n.target sm_90\n.visible\n.entry kernel(\n.param .u64 p\n)\n{\nL0:\n@%p0 bra L0;\nret;\n}\n",
        );
        assert_eq!(
            parsed
                .statements
                .iter()
                .map(|statement| statement.kind)
                .collect::<Vec<_>>(),
            [
                StatementKind::Directive,
                StatementKind::Directive,
                StatementKind::CallableHeader,
                StatementKind::Label,
                StatementKind::Instruction,
                StatementKind::Instruction,
            ]
        );
        assert_eq!(parsed.scopes.len(), 2);
        let body = &parsed.scopes[1];
        assert_eq!(body.parent(), Some(ScopeId::ROOT));
        assert_eq!(body.header(), Some(StatementId(2)));
        assert_eq!(
            parsed.statements[3..]
                .iter()
                .map(Statement::scope)
                .collect::<Vec<_>>(),
            [ScopeId(1), ScopeId(1), ScopeId(1)]
        );
        assert!(parsed.diagnostics.is_empty());
        assert!(parsed.coverage.is_complete());
    }

    #[test]
    fn keeps_initializer_and_operand_braces_inside_statements() {
        let parsed =
            parse_source(".global .u32 table[2] = {\n1, 2\n};\nmov.u32 {%r1, %r2}, {%r3, %r4};");
        assert_eq!(
            parsed
                .statements
                .iter()
                .map(|statement| statement.kind)
                .collect::<Vec<_>>(),
            [StatementKind::Directive, StatementKind::Instruction]
        );
        assert_eq!(parsed.scopes.len(), 1);
        assert!(parsed.coverage.is_complete());
    }

    #[test]
    fn associates_a_multiline_section_header_with_its_scope() {
        let parsed = parse_source(".section .debug_info\n{\n.b8 1\n}\n");
        assert_eq!(parsed.statements[0].kind(), StatementKind::Directive);
        assert_eq!(parsed.scopes[1].header(), Some(parsed.statements[0].id()));
        assert_eq!(parsed.statements[1].scope(), parsed.scopes[1].id());
        assert!(parsed.coverage.is_complete());
    }

    #[test]
    fn makes_recovery_and_unknown_coverage_observable() {
        let parsed = parse_source("mov.u32 %r1, [oops;\nret;");
        assert_eq!(parsed.statements[0].kind, StatementKind::Instruction);
        assert_eq!(parsed.statements[1].kind, StatementKind::Instruction);
        assert_eq!(
            parsed.diagnostics[0].kind,
            DiagnosticKind::UnterminatedDelimiter
        );
        assert!(parsed.coverage.is_lossless());
        assert!(!parsed.coverage.is_complete());
        assert_eq!(parsed.coverage.unknown_bytes(), 0);
        assert_eq!(parsed.coverage.diagnostic_count(), 1);
    }

    #[test]
    fn recovers_labels_after_same_line_semicolons() {
        let parsed = parse_source("mov.u32 %r0, 1; L1: add.u32 %r1, %r0, 1;");
        assert_eq!(
            parsed
                .statements
                .iter()
                .map(|statement| statement.kind)
                .collect::<Vec<_>>(),
            [StatementKind::Instruction, StatementKind::Instruction]
        );
        assert!(parsed.coverage.is_complete());
    }

    #[test]
    fn recovers_a_truncated_callable_at_the_next_header() {
        let parsed = parse_source(
            ".visible .entry kernel(\n.extern .func helper(\n.extern .func final();\n",
        );
        assert_eq!(
            parsed
                .statements
                .iter()
                .map(Statement::kind)
                .collect::<Vec<_>>(),
            [
                StatementKind::CallableHeader,
                StatementKind::CallableHeader,
                StatementKind::CallableHeader,
            ]
        );
        assert_eq!(
            parsed
                .diagnostics
                .iter()
                .map(Diagnostic::kind)
                .collect::<Vec<_>>(),
            [
                DiagnosticKind::UnterminatedDelimiter,
                DiagnosticKind::UnterminatedDelimiter,
            ]
        );
    }

    #[test]
    fn accepts_a_label_immediately_before_a_scope_close() {
        let parsed = parse_source("{ bra DONE; DONE: }");
        assert_eq!(
            parsed
                .statements
                .iter()
                .map(|statement| statement.kind)
                .collect::<Vec<_>>(),
            [StatementKind::Instruction, StatementKind::Label]
        );
        assert_eq!(parsed.scopes.len(), 2);
        assert!(parsed.coverage.is_complete());
    }

    #[test]
    fn does_not_confuse_double_colon_modifiers_with_labels() {
        let parsed = parse_source(
            "L0: tcgen05.wait::ld.sync.aligned;\nmapa.shared::cluster.u64 %rd1, %rd2, %r3;",
        );
        assert_eq!(
            parsed
                .statements
                .iter()
                .map(|statement| statement.kind)
                .collect::<Vec<_>>(),
            [StatementKind::Instruction, StatementKind::Instruction]
        );
        assert!(parsed.coverage.is_complete());
    }

    #[test]
    fn diagnoses_unbalanced_structural_scopes() {
        let unmatched = parse_source("}\nret;");
        assert_eq!(unmatched.statements[0].kind, StatementKind::Unknown);
        assert_eq!(
            unmatched.diagnostics[0].kind,
            DiagnosticKind::UnmatchedClosingDelimiter
        );
        assert!(unmatched.coverage.unknown_bytes() > 0);
        assert!(!unmatched.coverage.is_complete());

        let unterminated = parse_source(".entry kernel() {\nret;");
        assert_eq!(
            unterminated.diagnostics[0].kind,
            DiagnosticKind::UnterminatedDelimiter
        );
        assert_eq!(unterminated.scopes.len(), 2);
        assert!(unterminated.scopes[1].close_span().is_none());
        assert!(!unterminated.coverage.is_complete());
    }
}
