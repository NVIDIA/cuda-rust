/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lossless structural views over PTX source text.
//!
//! This crate deliberately does not type-check the PTX ISA. Instructions are
//! discovered structurally, so an opcode introduced by a newer PTX version is
//! retained with the same source spans as a known opcode. Consumers which need
//! ISA semantics can layer that policy over [`Instruction::head`].

mod edit;
mod lexer;
mod syntax;

pub use edit::{AppliedEdits, EditError, EditMap, EditScript, MapBias};
pub use lexer::{Token, TokenKind};
pub use syntax::{
    Coverage, Diagnostic, DiagnosticKind, Scope, ScopeId, Statement, StatementId, StatementKind,
};

use std::fmt;
use std::ops::Range;

/// A parsed PTX document which borrows its source and owns only structural
/// indices into it.
#[derive(Clone, Debug)]
pub struct Document<'source> {
    source: &'source str,
    tokens: Vec<Token>,
    statements: Vec<Statement>,
    scopes: Vec<Scope>,
    diagnostics: Vec<Diagnostic>,
    coverage: Coverage,
    labels: Vec<Label<'source>>,
    directives: Vec<Directive<'source>>,
    callables: Vec<Callable<'source>>,
    instructions: Vec<Instruction<'source>>,
    statements_by_scope: Vec<Vec<StatementId>>,
    labels_by_statement: Vec<Vec<LabelId>>,
    directive_by_statement: Vec<Option<usize>>,
    callable_by_statement: Vec<Option<usize>>,
    instruction_by_statement: Vec<Option<usize>>,
}

/// Stable index of a projected label in [`Document::labels`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelId(usize);

impl LabelId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// One PTX statement label, including a label prefixed to another statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label<'source> {
    source: &'source str,
    id: LabelId,
    statement: StatementId,
    scope: ScopeId,
    span: Range<usize>,
    name_span: Range<usize>,
}

/// A typed view of one PTX directive statement.
///
/// The view retains the original spelling and is projected from a
/// [`StatementKind::Directive`] node; it does not independently rescan source
/// lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Directive<'source> {
    source: &'source str,
    statement: StatementId,
    scope: ScopeId,
    span: Range<usize>,
    line_span: Range<usize>,
    name_span: Range<usize>,
    arguments_span: Range<usize>,
    label_name_spans: Vec<Range<usize>>,
}

/// A typed, lossless view of one `.reg` directive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterDeclaration<'source> {
    source: &'source str,
    statement: StatementId,
    scope: ScopeId,
    span: Range<usize>,
    qualifier_spans: Vec<Range<usize>>,
    bindings: Vec<RegisterBinding<'source>>,
}

/// One scalar register or register-bank binding in a `.reg` declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterBinding<'source> {
    source: &'source str,
    span: Range<usize>,
    name_span: Range<usize>,
    bank_size: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterDeclarationErrorKind {
    MissingQualifier,
    MissingBinding,
    UnexpectedToken,
    InvalidBankSize,
}

/// A `.reg` directive that cannot be interpreted without guessing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterDeclarationError {
    statement: StatementId,
    offset: usize,
    kind: RegisterDeclarationErrorKind,
}

/// The two callable forms defined by PTX.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallableKind {
    Entry,
    Function,
}

/// A typed view of one PTX callable declaration or definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Callable<'source> {
    source: &'source str,
    statement: StatementId,
    scope: ScopeId,
    definition_scope: Option<ScopeId>,
    kind: CallableKind,
    span: Range<usize>,
    header_span: Range<usize>,
    body_span: Option<Range<usize>>,
    name_span: Range<usize>,
    is_extern: bool,
}

/// A closed callable definition bound to its parsed document.
///
/// Binding the two prevents body-scoped queries from accepting a callable
/// projected from another document.
#[derive(Clone, Copy, Debug)]
pub struct CallableDefinition<'document, 'source> {
    document: &'document Document<'source>,
    callable: &'document Callable<'source>,
}

/// One semicolon-terminated PTX instruction.
///
/// The source text is not normalized. Predicates and same-line labels are
/// exposed through [`Self::prefix`], while [`Self::head`] retains the exact
/// opcode and modifier spelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction<'source> {
    source: &'source str,
    statement: StatementId,
    scope: ScopeId,
    span: Range<usize>,
    prefix_span: Range<usize>,
    head_span: Range<usize>,
    operand_spans: Vec<Range<usize>>,
    label_name_spans: Vec<Range<usize>>,
    predicate: Option<Predicate<'source>>,
}

/// A guard predicate recovered from an instruction statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Predicate<'source> {
    source: &'source str,
    start: u32,
    register: &'source str,
    register_start: u32,
    register_end: u32,
    negated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LabelSpans {
    span: Range<usize>,
    name_span: Range<usize>,
}

/// A lexical error which prevents reliable source-span recovery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    SourceTooLarge { bytes: usize },
    UnterminatedBlockComment { offset: usize },
    UnterminatedQuotedString { offset: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceTooLarge { bytes } => {
                write!(formatter, "PTX source is {bytes} bytes; maximum is 4 GiB")
            }
            Self::UnterminatedBlockComment { offset } => {
                write!(formatter, "unterminated PTX block comment at byte {offset}")
            }
            Self::UnterminatedQuotedString { offset } => {
                write!(formatter, "unterminated PTX quoted string at byte {offset}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl<'source> Document<'source> {
    /// Parse a lossless structural view of `source`.
    ///
    /// Tokens losslessly partition the original source. Unknown instruction
    /// heads are accepted, while unterminated comments and strings fail the
    /// document because every following source span would be ambiguous.
    pub fn parse(source: &'source str) -> Result<Self, ParseError> {
        let tokens = lexer::lex(source)?;
        let masked = mask_non_code(source, &tokens);
        let mut parsed = syntax::parse(source, &tokens);
        let labels = discover_labels(source, &tokens, &parsed.statements);
        let (directives, directive_diagnostics) =
            discover_directives(source, &tokens, &parsed.statements);
        let (callables, callable_diagnostics) =
            discover_callables(source, &tokens, &parsed.statements, &parsed.scopes);
        let (instructions, instruction_diagnostics) = discover_instructions(
            source,
            &masked,
            &tokens,
            &parsed.statements,
            &parsed.diagnostics,
        );
        parsed.coverage.add_diagnostics(
            directive_diagnostics.len()
                + callable_diagnostics.len()
                + instruction_diagnostics.len(),
        );
        parsed.diagnostics.extend(directive_diagnostics);
        parsed.diagnostics.extend(callable_diagnostics);
        parsed.diagnostics.extend(instruction_diagnostics);
        let mut statements_by_scope = vec![Vec::new(); parsed.scopes.len()];
        for statement in &parsed.statements {
            statements_by_scope[statement.scope().index()].push(statement.id());
        }
        let mut labels_by_statement = vec![Vec::new(); parsed.statements.len()];
        for label in &labels {
            labels_by_statement[label.statement().index()].push(label.id());
        }
        let mut directive_by_statement = vec![None; parsed.statements.len()];
        for (index, directive) in directives.iter().enumerate() {
            directive_by_statement[directive.statement().index()] = Some(index);
        }
        let mut callable_by_statement = vec![None; parsed.statements.len()];
        for (index, callable) in callables.iter().enumerate() {
            callable_by_statement[callable.statement().index()] = Some(index);
        }
        let mut instruction_by_statement = vec![None; parsed.statements.len()];
        for (index, instruction) in instructions.iter().enumerate() {
            instruction_by_statement[instruction.statement().index()] = Some(index);
        }
        Ok(Self {
            source,
            tokens,
            statements: parsed.statements,
            scopes: parsed.scopes,
            diagnostics: parsed.diagnostics,
            coverage: parsed.coverage,
            labels,
            directives,
            callables,
            instructions,
            statements_by_scope,
            labels_by_statement,
            directive_by_statement,
            callable_by_statement,
            instruction_by_statement,
        })
    }

    pub fn source(&self) -> &'source str {
        self.source
    }

    /// Lossless lexical tokens in source order.
    ///
    /// Their spans are contiguous and cover exactly [`Self::source`].
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Structural statements in source order. Every non-trivia token is owned
    /// by one statement or a lexical scope delimiter. Unrecognized input is
    /// retained as [`StatementKind::Unknown`].
    pub fn statements(&self) -> &[Statement] {
        &self.statements
    }

    /// Lexical scopes in source order, beginning with [`ScopeId::ROOT`].
    pub fn scopes(&self) -> &[Scope] {
        &self.scopes
    }

    pub fn statement(&self, id: StatementId) -> Option<&Statement> {
        self.statements.get(id.index())
    }

    pub fn statements_in_scope(&self, scope: ScopeId) -> impl Iterator<Item = &Statement> {
        self.statements_by_scope
            .get(scope.index())
            .into_iter()
            .flatten()
            .map(|statement| &self.statements[statement.index()])
    }

    pub fn scope(&self, id: ScopeId) -> Option<&Scope> {
        self.scopes.get(id.index())
    }

    /// Recoverable structural problems found while retaining later nodes.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn coverage(&self) -> Coverage {
        self.coverage
    }

    pub fn labels(&self) -> &[Label<'source>] {
        &self.labels
    }

    pub fn label(&self, id: LabelId) -> Option<&Label<'source>> {
        self.labels.get(id.index())
    }

    pub fn labels_for_statement(
        &self,
        statement: StatementId,
    ) -> impl Iterator<Item = &Label<'source>> {
        self.labels_by_statement
            .get(statement.index())
            .into_iter()
            .flatten()
            .map(|label| &self.labels[label.index()])
    }

    pub fn labels_in(&self, span: Range<usize>) -> impl Iterator<Item = &Label<'source>> {
        let start = self
            .labels
            .partition_point(|label| label.span.start < span.start);
        self.labels[start..]
            .iter()
            .take_while(move |label| label.span.start < span.end)
            .filter(move |label| label.span.end <= span.end)
    }

    pub fn directives(&self) -> &[Directive<'source>] {
        &self.directives
    }

    pub fn directive_for_statement(&self, statement: StatementId) -> Option<&Directive<'source>> {
        self.directive_by_statement
            .get(statement.index())
            .copied()
            .flatten()
            .map(|index| &self.directives[index])
    }

    pub fn directives_in(&self, span: Range<usize>) -> impl Iterator<Item = &Directive<'source>> {
        let start = self
            .directives
            .partition_point(|directive| directive.span.start < span.start);
        self.directives[start..]
            .iter()
            .take_while(move |directive| directive.span.start < span.end)
            .filter(move |directive| directive.span.end <= span.end)
    }

    /// Parse every `.reg` directive in source order.
    ///
    /// Structural parsing remains forward-compatible; consumers which need
    /// register semantics receive an explicit error for an unfamiliar binding
    /// grammar rather than a partial declaration.
    pub fn register_declarations(
        &self,
    ) -> impl Iterator<Item = Result<RegisterDeclaration<'source>, RegisterDeclarationError>> + '_
    {
        self.directives
            .iter()
            .filter(|directive| directive.name() == ".reg")
            .map(|directive| {
                register_declaration_from_directive(self.source, &self.tokens, directive)
            })
    }

    pub fn callables(&self) -> &[Callable<'source>] {
        &self.callables
    }

    pub fn callable_for_statement(&self, statement: StatementId) -> Option<&Callable<'source>> {
        self.callable_by_statement
            .get(statement.index())
            .copied()
            .flatten()
            .map(|index| &self.callables[index])
    }

    /// Return every callable whose symbol exactly matches `name`.
    ///
    /// PTX may contain both a declaration and a definition for a symbol, so
    /// this deliberately exposes an iterator instead of choosing one match.
    pub fn callables_named<'document: 'query, 'query>(
        &'document self,
        name: &'query str,
    ) -> impl Iterator<Item = &'document Callable<'source>> + 'query {
        self.callables
            .iter()
            .filter(move |callable| callable.name() == name)
    }

    /// Return every closed callable definition in source order.
    pub fn definitions(&self) -> impl Iterator<Item = CallableDefinition<'_, 'source>> {
        self.callables
            .iter()
            .filter(|callable| callable.body_span.is_some())
            .map(|callable| CallableDefinition {
                document: self,
                callable,
            })
    }

    pub fn definitions_named<'document: 'query, 'query>(
        &'document self,
        name: &'query str,
    ) -> impl Iterator<Item = CallableDefinition<'document, 'source>> + 'query {
        self.definitions()
            .filter(move |definition| definition.callable.name() == name)
    }

    pub fn instructions(&self) -> &[Instruction<'source>] {
        &self.instructions
    }

    pub fn instruction_for_statement(
        &self,
        statement: StatementId,
    ) -> Option<&Instruction<'source>> {
        self.instruction_by_statement
            .get(statement.index())
            .copied()
            .flatten()
            .map(|index| &self.instructions[index])
    }

    /// Return instructions fully contained by `span` in source order.
    ///
    /// Instructions are source-ordered, so the query finds its first possible
    /// result with a binary search and visits only the overlapping window.
    pub fn instructions_in(
        &self,
        span: Range<usize>,
    ) -> impl Iterator<Item = &Instruction<'source>> {
        let start = self
            .instructions
            .partition_point(|instruction| instruction.span.start < span.start);
        self.instructions[start..]
            .iter()
            .take_while(move |instruction| instruction.span.start < span.end)
            .filter(move |instruction| instruction.span.end <= span.end)
    }
}

impl<'source> Label<'source> {
    pub fn id(&self) -> LabelId {
        self.id
    }

    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Byte range covering the label name and terminal colon.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn name(&self) -> &'source str {
        &self.source[self.name_span.clone()]
    }

    pub fn name_span(&self) -> Range<usize> {
        self.name_span.clone()
    }
}

impl<'source> Directive<'source> {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Byte range covering optional labels and the directive without leading
    /// indentation, trailing comment, or newline.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Byte range covering the physical source line where the directive
    /// begins, including its newline when present.
    pub fn line_span(&self) -> Range<usize> {
        self.line_span.clone()
    }

    pub fn name(&self) -> &'source str {
        &self.source[self.name_span.clone()]
    }

    pub fn name_span(&self) -> Range<usize> {
        self.name_span.clone()
    }

    pub fn labels(&self) -> impl ExactSizeIterator<Item = &'source str> + '_ {
        self.label_name_spans
            .iter()
            .map(|span| &self.source[span.clone()])
    }

    pub fn arguments(&self) -> &'source str {
        &self.source[self.arguments_span.clone()]
    }

    pub fn arguments_span(&self) -> Range<usize> {
        self.arguments_span.clone()
    }

    pub fn text(&self) -> &'source str {
        &self.source[self.span.clone()]
    }
}

impl RegisterDeclaration<'_> {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn qualifiers(&self) -> impl ExactSizeIterator<Item = &str> {
        self.qualifier_spans
            .iter()
            .map(|span| &self.source[span.clone()])
    }

    pub fn bindings(&self) -> &[RegisterBinding<'_>] {
        &self.bindings
    }
}

impl RegisterBinding<'_> {
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Scalar name, or the base name of a register bank.
    pub fn name(&self) -> &str {
        &self.source[self.name_span.clone()]
    }

    pub fn name_span(&self) -> Range<usize> {
        self.name_span.clone()
    }

    /// Number of concrete registers in a bank such as `%r<4>`.
    pub fn bank_size(&self) -> Option<u32> {
        self.bank_size
    }
}

impl RegisterDeclarationError {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn kind(&self) -> RegisterDeclarationErrorKind {
        self.kind
    }
}

impl fmt::Display for RegisterDeclarationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "malformed PTX .reg declaration at byte {} ({:?})",
            self.offset, self.kind
        )
    }
}

impl std::error::Error for RegisterDeclarationError {}

impl<'source> Callable<'source> {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Scope containing the callable body, or `None` for a declaration.
    pub fn definition_scope(&self) -> Option<ScopeId> {
        self.definition_scope
    }

    pub fn kind(&self) -> CallableKind {
        self.kind
    }

    /// Byte range covering the complete declaration or definition.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Byte range from the first linkage directive through the callable name.
    pub fn header_span(&self) -> Range<usize> {
        self.header_span.clone()
    }

    /// Source range inside a definition's outer braces.
    pub fn body_span(&self) -> Option<Range<usize>> {
        self.body_span.clone()
    }

    pub fn name(&self) -> &'source str {
        &self.source[self.name_span.clone()]
    }

    pub fn name_span(&self) -> Range<usize> {
        self.name_span.clone()
    }

    pub fn is_extern(&self) -> bool {
        self.is_extern
    }

    /// Original source covering the complete declaration or definition when
    /// its extent was recovered, otherwise the structural header.
    pub fn text(&self) -> &'source str {
        &self.source[self.span.clone()]
    }

    /// Original source inside a definition's outer braces.
    pub fn body_text(&self) -> Option<&'source str> {
        self.body_span
            .as_ref()
            .map(|span| &self.source[span.clone()])
    }

    /// Original source preceding a definition's opening brace.
    ///
    /// This includes parameter lists and callable directives such as
    /// `.maxntid`. Declarations and incomplete definitions return `None`.
    pub fn definition_header_text(&self) -> Option<&'source str> {
        self.definition_header_span().map(|span| &self.source[span])
    }

    pub fn definition_header_span(&self) -> Option<Range<usize>> {
        self.body_span.as_ref().map(|body| {
            debug_assert_eq!(self.source.as_bytes()[body.start - 1], b'{');
            self.span.start..body.start - 1
        })
    }
}

impl<'document, 'source> CallableDefinition<'document, 'source> {
    pub fn callable(self) -> &'document Callable<'source> {
        self.callable
    }

    pub fn scope(self) -> ScopeId {
        self.callable
            .definition_scope()
            .expect("CallableDefinition always has a scope")
    }

    pub fn text(self) -> &'source str {
        self.callable.text()
    }

    pub fn header_text(self) -> &'source str {
        self.callable
            .definition_header_text()
            .expect("CallableDefinition always has a closed body")
    }

    pub fn body_text(self) -> &'source str {
        self.callable
            .body_text()
            .expect("CallableDefinition always has a closed body")
    }

    pub fn instructions(self) -> impl Iterator<Item = &'document Instruction<'source>> + 'document {
        self.document.instructions_in(
            self.callable
                .body_span()
                .expect("CallableDefinition always has a closed body"),
        )
    }
}

impl<'source> Instruction<'source> {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Byte range covering the optional predicate/labels through the terminal
    /// semicolon. Leading indentation and trailing comments are excluded.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Byte offset at which the opcode begins.
    pub fn head_offset(&self) -> usize {
        self.head_span.start
    }

    pub fn head_span(&self) -> Range<usize> {
        self.head_span.clone()
    }

    /// Byte offset immediately after the terminal semicolon.
    pub fn end_offset(&self) -> usize {
        self.span.end
    }

    /// Optional predicate and same-line labels preceding the opcode.
    pub fn prefix(&self) -> &'source str {
        &self.source[self.prefix_span.clone()]
    }

    pub fn labels(&self) -> impl ExactSizeIterator<Item = &'source str> + '_ {
        self.label_name_spans
            .iter()
            .map(|span| &self.source[span.clone()])
    }

    pub fn predicate(&self) -> Option<Predicate<'source>> {
        self.predicate
    }

    /// Exact opcode and ordered modifier spelling.
    pub fn head(&self) -> &'source str {
        &self.source[self.head_span.clone()]
    }

    /// Instruction opcode without ordered modifier suffixes.
    pub fn base_opcode(&self) -> &'source str {
        self.head()
            .split_once('.')
            .map_or(self.head(), |(base, _)| base)
    }

    /// Top-level operands in source order.
    pub fn operands(&self) -> impl ExactSizeIterator<Item = &'source str> + '_ {
        self.operand_spans
            .iter()
            .map(|span| &self.source[span.clone()])
    }

    pub fn operand_spans(&self) -> impl ExactSizeIterator<Item = Range<usize>> + '_ {
        self.operand_spans.iter().cloned()
    }

    pub fn text(&self) -> &'source str {
        &self.source[self.span.clone()]
    }
}

impl<'source> Predicate<'source> {
    pub fn text(self) -> &'source str {
        &self.source[self.span()]
    }

    pub fn span(self) -> Range<usize> {
        self.start as usize..self.register_end as usize
    }

    pub fn register(self) -> &'source str {
        self.register
    }

    pub fn is_negated(self) -> bool {
        self.negated
    }

    pub fn register_span(self) -> Range<usize> {
        self.register_start as usize..self.register_end as usize
    }
}

/// Split a comma-separated PTX operand list without splitting nested register
/// lists, addresses, or parameter tuples.
pub fn split_top_level(source: &str) -> Option<Vec<&str>> {
    split_top_level_spans(source, 0)
        .map(|spans| spans.into_iter().map(|span| &source[span]).collect())
}

fn discover_labels<'source>(
    source: &'source str,
    tokens: &[Token],
    statements: &[Statement],
) -> Vec<Label<'source>> {
    let mut labels = Vec::new();
    for statement in statements {
        let significant = significant_token_indices(tokens, statement);
        let (_, spans) = leading_label_spans(source, tokens, &significant);
        for spans in spans {
            labels.push(Label {
                source,
                id: LabelId(labels.len()),
                statement: statement.id(),
                scope: statement.scope(),
                span: spans.span,
                name_span: spans.name_span,
            });
        }
    }
    labels
}

fn discover_directives<'source>(
    source: &'source str,
    tokens: &[Token],
    statements: &[Statement],
) -> (Vec<Directive<'source>>, Vec<Diagnostic>) {
    let mut directives = Vec::new();
    let mut diagnostics = Vec::new();
    for statement in statements
        .iter()
        .filter(|statement| statement.kind() == StatementKind::Directive)
    {
        let significant = significant_token_indices(tokens, statement);
        let (cursor, label_spans) = leading_label_spans(source, tokens, &significant);
        let Some(name) = significant
            .get(cursor)
            .map(|index| &tokens[*index])
            .filter(|token| token.kind() == TokenKind::Word && token.text(source).starts_with('.'))
        else {
            diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedDirective,
                statement.span(),
            ));
            continue;
        };
        let span = statement.span();
        directives.push(Directive {
            source,
            statement: statement.id(),
            scope: statement.scope(),
            line_span: physical_line_span(source, span.start),
            arguments_span: trim_span(source, name.span().end..span.end),
            name_span: name.span(),
            label_name_spans: label_spans
                .into_iter()
                .map(|spans| spans.name_span)
                .collect(),
            span,
        });
    }
    (directives, diagnostics)
}

fn register_declaration_from_directive<'source>(
    source: &'source str,
    tokens: &[Token],
    directive: &Directive<'source>,
) -> Result<RegisterDeclaration<'source>, RegisterDeclarationError> {
    let error = |offset, kind| RegisterDeclarationError {
        statement: directive.statement,
        offset,
        kind,
    };
    let first_token =
        tokens.partition_point(|token| token.span().end <= directive.arguments_span.start);
    let significant = tokens[first_token..]
        .iter()
        .take_while(|token| token.span().start < directive.arguments_span.end)
        .filter(|token| {
            let span = token.span();
            span.start >= directive.arguments_span.start
                && span.end <= directive.arguments_span.end
                && !token.kind().is_trivia()
        })
        .collect::<Vec<_>>();
    let Some(semicolon) = significant.last().filter(|token| token.text(source) == ";") else {
        return Err(error(
            directive.arguments_span.end,
            RegisterDeclarationErrorKind::UnexpectedToken,
        ));
    };
    let body = &significant[..significant.len() - 1];
    let mut cursor = 0;
    let mut qualifier_spans = Vec::new();
    while let Some(token) = body
        .get(cursor)
        .filter(|token| token.kind() == TokenKind::Word && token.text(source).starts_with('.'))
    {
        qualifier_spans.push(token.span());
        cursor += 1;
    }
    if qualifier_spans.is_empty() {
        return Err(error(
            body.first()
                .map_or(semicolon.span().start, |token| token.span().start),
            RegisterDeclarationErrorKind::MissingQualifier,
        ));
    }
    if cursor == body.len() {
        return Err(error(
            semicolon.span().start,
            RegisterDeclarationErrorKind::MissingBinding,
        ));
    }

    let mut bindings = Vec::new();
    loop {
        let Some(name) = body.get(cursor).filter(|token| {
            token.kind() == TokenKind::Word && !token.text(source).starts_with('.')
        }) else {
            let offset = body
                .get(cursor)
                .map_or(semicolon.span().start, |token| token.span().start);
            return Err(error(offset, RegisterDeclarationErrorKind::MissingBinding));
        };
        cursor += 1;
        let mut binding_end = name.span().end;
        let bank_size = if body
            .get(cursor)
            .is_some_and(|token| token.text(source) == "<")
        {
            let Some(size) = body.get(cursor + 1) else {
                return Err(error(
                    binding_end,
                    RegisterDeclarationErrorKind::InvalidBankSize,
                ));
            };
            let Some(close) = body
                .get(cursor + 2)
                .filter(|token| token.text(source) == ">")
            else {
                return Err(error(
                    size.span().start,
                    RegisterDeclarationErrorKind::InvalidBankSize,
                ));
            };
            let size = size
                .text(source)
                .parse::<u32>()
                .ok()
                .filter(|size| *size > 0)
                .ok_or_else(|| {
                    error(
                        size.span().start,
                        RegisterDeclarationErrorKind::InvalidBankSize,
                    )
                })?;
            cursor += 3;
            binding_end = close.span().end;
            Some(size)
        } else {
            None
        };
        bindings.push(RegisterBinding {
            source,
            span: name.span().start..binding_end,
            name_span: name.span(),
            bank_size,
        });

        match body.get(cursor).map(|token| token.text(source)) {
            None => break,
            Some(",") if cursor + 1 < body.len() => cursor += 1,
            Some(",") => {
                return Err(error(
                    body[cursor].span().end,
                    RegisterDeclarationErrorKind::MissingBinding,
                ));
            }
            Some(_) => {
                return Err(error(
                    body[cursor].span().start,
                    RegisterDeclarationErrorKind::UnexpectedToken,
                ));
            }
        }
    }

    Ok(RegisterDeclaration {
        source,
        statement: directive.statement,
        scope: directive.scope,
        span: directive.span(),
        qualifier_spans,
        bindings,
    })
}

fn discover_callables<'source>(
    source: &'source str,
    tokens: &[Token],
    statements: &[Statement],
    scopes: &[Scope],
) -> (Vec<Callable<'source>>, Vec<Diagnostic>) {
    let mut callables = Vec::new();
    let mut diagnostics = Vec::new();
    for statement in statements
        .iter()
        .filter(|statement| statement.kind() == StatementKind::CallableHeader)
    {
        let significant = significant_token_indices(tokens, statement);
        let Some((keyword_cursor, kind)) =
            significant.iter().enumerate().find_map(|(cursor, index)| {
                match tokens[*index].text(source) {
                    ".entry" => Some((cursor, CallableKind::Entry)),
                    ".func" => Some((cursor, CallableKind::Function)),
                    _ => None,
                }
            })
        else {
            diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedCallable,
                statement.span(),
            ));
            continue;
        };

        let mut name_cursor = keyword_cursor + 1;
        if kind == CallableKind::Function
            && significant
                .get(name_cursor)
                .is_some_and(|index| tokens[*index].text(source) == "(")
        {
            let Some(after_parameters) =
                skip_balanced_tokens(source, tokens, &significant, name_cursor, "(", ")")
            else {
                diagnostics.push(Diagnostic::new(
                    DiagnosticKind::MalformedCallable,
                    statement.span(),
                ));
                continue;
            };
            name_cursor = after_parameters;
        }
        let Some(name) = significant
            .get(name_cursor)
            .map(|index| &tokens[*index])
            .filter(|token| token.kind() == TokenKind::Word)
        else {
            diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedCallable,
                statement.span(),
            ));
            continue;
        };
        let definition_scope = scopes
            .iter()
            .find(|scope| scope.header() == Some(statement.id()))
            .map(Scope::id);
        let definition = definition_scope.and_then(|scope| scopes.get(scope.index()));
        let closed_definition = definition.filter(|scope| scope.close_span().is_some());
        let span = closed_definition.map_or_else(
            || statement.span(),
            |scope| statement.span().start..scope.span().end,
        );
        let body_span = closed_definition.map(Scope::body_span);
        callables.push(Callable {
            source,
            statement: statement.id(),
            scope: statement.scope(),
            definition_scope,
            kind,
            span,
            header_span: statement.span().start..name.span().end,
            body_span,
            name_span: name.span(),
            is_extern: significant[..keyword_cursor]
                .iter()
                .any(|index| tokens[*index].text(source) == ".extern"),
        });
    }
    (callables, diagnostics)
}

fn significant_token_indices(tokens: &[Token], statement: &Statement) -> Vec<usize> {
    statement
        .token_range()
        .filter(|index| !tokens[*index].kind().is_trivia())
        .collect()
}

fn leading_label_spans(
    source: &str,
    tokens: &[Token],
    significant: &[usize],
) -> (usize, Vec<LabelSpans>) {
    let mut cursor = 0usize;
    let mut spans = Vec::new();
    while cursor + 1 < significant.len()
        && tokens[significant[cursor]].kind() == TokenKind::Word
        && tokens[significant[cursor + 1]].text(source) == ":"
        && (cursor + 2 == significant.len() || tokens[significant[cursor + 2]].text(source) != ":")
    {
        let name_span = tokens[significant[cursor]].span();
        let span = name_span.start..tokens[significant[cursor + 1]].span().end;
        spans.push(LabelSpans { span, name_span });
        cursor += 2;
    }
    (cursor, spans)
}

fn skip_balanced_tokens(
    source: &str,
    tokens: &[Token],
    significant: &[usize],
    start: usize,
    open: &str,
    close: &str,
) -> Option<usize> {
    let mut depth = 0usize;
    for (cursor, index) in significant.iter().enumerate().skip(start) {
        match tokens[*index].text(source) {
            token if token == open => depth += 1,
            token if token == close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(cursor + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn physical_line_span(source: &str, offset: usize) -> Range<usize> {
    let start = source[..offset]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    let end = source[offset..]
        .find('\n')
        .map_or(source.len(), |newline| offset + newline + 1);
    start..end
}

fn discover_instructions<'source>(
    source: &'source str,
    masked: &str,
    tokens: &[Token],
    statements: &[Statement],
    diagnostics: &[Diagnostic],
) -> (Vec<Instruction<'source>>, Vec<Diagnostic>) {
    let mut instructions = Vec::new();
    let mut projection_diagnostics = Vec::new();
    for statement in statements
        .iter()
        .filter(|statement| statement.kind() == StatementKind::Instruction)
    {
        if let Some(instruction) = instruction_from_statement(source, masked, tokens, statement) {
            instructions.push(instruction);
        } else if !diagnostics
            .iter()
            .any(|diagnostic| ranges_overlap(diagnostic.span(), statement.span()))
        {
            projection_diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedInstruction,
                statement.span(),
            ));
        }
    }
    (instructions, projection_diagnostics)
}

fn ranges_overlap(left: Range<usize>, right: Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn instruction_from_statement<'source>(
    source: &'source str,
    masked: &str,
    tokens: &[Token],
    statement: &Statement,
) -> Option<Instruction<'source>> {
    let significant = significant_token_indices(tokens, statement);
    let (mut cursor, label_spans) = leading_label_spans(source, tokens, &significant);
    let mut predicate = None;
    if significant
        .get(cursor)
        .is_some_and(|index| tokens[*index].text(source) == "@")
    {
        let predicate_start = tokens[significant[cursor]].span().start;
        cursor += 1;
        if significant
            .get(cursor)
            .is_some_and(|index| tokens[*index].text(source) == "!")
        {
            cursor += 1;
        }
        let register = tokens.get(*significant.get(cursor)?)?;
        if register.kind() != TokenKind::Word {
            return None;
        }
        predicate = Some(Predicate {
            source,
            start: predicate_start
                .try_into()
                .expect("the lexer rejects sources larger than u32::MAX"),
            register: register.text(source),
            register_start: register
                .span()
                .start
                .try_into()
                .expect("the lexer rejects sources larger than u32::MAX"),
            register_end: register
                .span()
                .end
                .try_into()
                .expect("the lexer rejects sources larger than u32::MAX"),
            negated: significant
                .get(cursor.wrapping_sub(1))
                .is_some_and(|index| tokens[*index].text(source) == "!"),
        });
        cursor += 1;
    }
    let head_start = tokens.get(*significant.get(cursor)?)?;
    let mut head_end = head_start.span().end;
    while cursor + 2 < significant.len()
        && tokens[significant[cursor + 1]].text(source) == "::"
        && tokens[significant[cursor + 2]].kind() == TokenKind::Word
    {
        cursor += 2;
        head_end = tokens[significant[cursor]].span().end;
    }
    let semicolon = tokens.get(*significant.last()?)?;
    if head_start.kind() != TokenKind::Word || semicolon.text(source) != ";" {
        return None;
    }
    let head_span = head_start.span().start..head_end;
    let prefix_span = trim_span(masked, statement.span().start..head_span.start);
    let operands = trim_span(masked, head_span.end..semicolon.span().start);
    let operand_spans = if operands.is_empty() {
        Vec::new()
    } else {
        split_top_level_spans(&masked[operands.clone()], operands.start)?
    };
    Some(Instruction {
        source,
        statement: statement.id(),
        scope: statement.scope(),
        span: statement.span(),
        prefix_span,
        head_span,
        operand_spans,
        label_name_spans: label_spans
            .into_iter()
            .map(|spans| spans.name_span)
            .collect(),
        predicate,
    })
}

fn split_top_level_spans(source: &str, base: usize) -> Option<Vec<Range<usize>>> {
    let leading = source.len() - source.trim_start().len();
    let source = source.trim();
    let base = base + leading;
    if source.is_empty() {
        return Some(Vec::new());
    }

    let mut operands = Vec::new();
    let mut delimiters = Vec::new();
    let mut operand_start = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        match byte {
            b'{' => delimiters.push(b'}'),
            b'[' => delimiters.push(b']'),
            b'(' => delimiters.push(b')'),
            b'}' | b']' | b')' if delimiters.pop() != Some(byte) => return None,
            b'}' | b']' | b')' => {}
            b',' if delimiters.is_empty() => {
                let span = trim_span(source, operand_start..index);
                if span.is_empty() {
                    return None;
                }
                operands.push(base + span.start..base + span.end);
                operand_start = index + 1;
            }
            _ => {}
        }
    }
    if !delimiters.is_empty() {
        return None;
    }
    let span = trim_span(source, operand_start..source.len());
    if span.is_empty() {
        return None;
    }
    operands.push(base + span.start..base + span.end);
    Some(operands)
}

fn trim_span(source: &str, span: Range<usize>) -> Range<usize> {
    let text = &source[span.clone()];
    if text.trim().is_empty() {
        return span.end..span.end;
    }
    let leading = text.len() - text.trim_start().len();
    let trailing = text.trim_end().len();
    span.start + leading..span.start + trailing
}

fn mask_non_code(source: &str, tokens: &[Token]) -> String {
    let mut masked = source.as_bytes().to_vec();
    for token in tokens.iter().filter(|token| {
        matches!(
            token.kind(),
            TokenKind::LineComment
                | TokenKind::BlockComment
                | TokenKind::QuotedString
                | TokenKind::Preprocessor
        )
    }) {
        for byte in &mut masked[token.span()] {
            if *byte != b'\n' && *byte != b'\r' {
                *byte = b' ';
            }
        }
    }
    String::from_utf8(masked).expect("masking tokens preserves UTF-8 boundaries")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_unknown_predicated_and_multiline_instructions() {
        let source = ".visible .entry kernel() {\n\
                      L0: @!%p1 future.op.u32\n\
                          {%r1, %r2}, [%rd3, {%r4, %r5}]; // tail\n\
                      ret;\n\
                      }\n";
        let document = Document::parse(source).unwrap();
        assert_eq!(document.instructions().len(), 2);
        let future = &document.instructions()[0];
        assert_eq!(future.prefix(), "L0: @!%p1");
        assert_eq!(future.labels().collect::<Vec<_>>(), ["L0"]);
        assert_eq!(future.predicate().unwrap().register(), "%p1");
        assert!(future.predicate().unwrap().is_negated());
        assert_eq!(
            &source[future.predicate().unwrap().register_span()],
            future.predicate().unwrap().register()
        );
        assert_eq!(future.head(), "future.op.u32");
        assert_eq!(&source[future.head_span()], future.head());
        assert_eq!(
            future.operands().collect::<Vec<_>>(),
            ["{%r1, %r2}", "[%rd3, {%r4, %r5}]"]
        );
        assert_eq!(
            future
                .operand_spans()
                .map(|span| &source[span])
                .collect::<Vec<_>>(),
            future.operands().collect::<Vec<_>>()
        );
        assert_eq!(
            future.text(),
            "L0: @!%p1 future.op.u32\n{%r1, %r2}, [%rd3, {%r4, %r5}];"
        );
        assert_eq!(document.instructions()[1].head(), "ret");
    }

    #[test]
    fn ignores_comments_strings_directives_and_operand_symbols() {
        let source = "// fake.u32 %r1;\n\
                      .file 1 \"quoted.u32 %r2;\"\n\
                      .target sm_90\n\
                      .visible .entry kernel() {\n\
                      call.uni (%r1), helper, (%r2);\n\
                      /* fake2.u32 %r3; */ ret;\n\
                      }";
        let document = Document::parse(source).unwrap();
        assert_eq!(
            document
                .instructions()
                .iter()
                .map(Instruction::head)
                .collect::<Vec<_>>(),
            ["call.uni", "ret"]
        );
    }

    #[test]
    fn projects_directives_from_statement_nodes() {
        let source = "  .version 8.9\n.target sm_120a, debug // generated\n";
        let document = Document::parse(source).unwrap();
        assert_eq!(document.directives().len(), 2);
        let target = &document.directives()[1];
        assert_eq!(target.name(), ".target");
        assert_eq!(target.arguments(), "sm_120a, debug");
        assert_eq!(target.text(), ".target sm_120a, debug");
        assert_eq!(
            &source[target.line_span()],
            ".target sm_120a, debug // generated\n"
        );
        assert_eq!(
            document.statement(target.statement()).unwrap().kind(),
            StatementKind::Directive
        );
        assert_eq!(target.scope(), ScopeId::ROOT);
    }

    #[test]
    fn projects_prefixed_and_standalone_labels_with_lineage() {
        let source = "L0:\nL1: L2: @%p0 bra L0;\nts: .branchtargets L0, L1;\n";
        let document = Document::parse(source).unwrap();
        assert_eq!(
            document
                .labels()
                .iter()
                .map(Label::name)
                .collect::<Vec<_>>(),
            ["L0", "L1", "L2", "ts"]
        );
        assert_eq!(
            document.instructions()[0].labels().collect::<Vec<_>>(),
            ["L1", "L2"]
        );
        assert_eq!(
            document.directives()[0].labels().collect::<Vec<_>>(),
            ["ts"]
        );
        assert_eq!(document.directives()[0].name(), ".branchtargets");
        assert_eq!(document.directives()[0].arguments(), "L0, L1;");
        for label in document.labels() {
            assert_eq!(document.label(label.id()), Some(label));
            assert_eq!(
                document.statement(label.statement()).unwrap().scope(),
                label.scope()
            );
        }
    }

    #[test]
    fn projects_multiline_callable_headers_and_definition_scopes() {
        let source = "\
.visible
.entry kernel() { ret; }
.extern .func (.param .b32 result)
    __nv_helper(.param .b32 input);
.weak .func local_helper() { ret; }
";
        let document = Document::parse(source).unwrap();
        assert_eq!(document.callables().len(), 3);
        assert_eq!(document.callables()[0].name(), "kernel");
        assert_eq!(document.callables()[0].kind(), CallableKind::Entry);
        assert!(!document.callables()[0].is_extern());
        assert!(document.callables()[0].definition_scope().is_some());
        assert!(document.callables()[0].body_span().is_some());
        assert_eq!(
            &source[document.callables()[0].span()],
            ".visible\n.entry kernel() { ret; }"
        );
        assert_eq!(document.callables()[1].name(), "__nv_helper");
        assert_eq!(document.callables()[1].kind(), CallableKind::Function);
        assert!(document.callables()[1].is_extern());
        assert!(document.callables()[1].definition_scope().is_none());
        assert_eq!(
            document.callables()[1].span(),
            document
                .statement(document.callables()[1].statement())
                .unwrap()
                .span()
        );
        assert_eq!(document.callables()[2].name(), "local_helper");
        assert!(!document.callables()[2].is_extern());
        assert!(document.callables()[2].definition_scope().is_some());
        assert!(document.callables().iter().all(|callable| {
            document
                .statement(callable.statement())
                .is_some_and(|statement| statement.kind() == StatementKind::CallableHeader)
        }));
    }

    #[test]
    fn does_not_claim_an_unclosed_callable_body() {
        let document = Document::parse(".entry incomplete() {\nret;\n").unwrap();
        let callable = &document.callables()[0];
        assert!(callable.definition_scope().is_some());
        assert!(callable.body_span().is_none());
        assert_eq!(
            callable.span(),
            callable.header_span().start
                ..document.statement(callable.statement()).unwrap().span().end
        );
        assert!(
            document
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.kind() == DiagnosticKind::UnterminatedDelimiter)
        );
    }

    #[test]
    fn retains_quoted_directive_arguments() {
        let document = Document::parse(".file 1 \"kernel.cu\"\n").unwrap();
        assert_eq!(document.directives()[0].arguments(), "1 \"kernel.cu\"");
    }

    #[test]
    fn queries_callable_source_without_reconstructing_syntax() {
        let source = "\
.extern .func helper();
.visible .entry kernel(
    .param .u64 output
)
.maxntid 128, 1, 1
{
    ret;
}
.extern .func helper();
";
        let document = Document::parse(source).unwrap();
        let helpers = document.callables_named("helper").collect::<Vec<_>>();
        assert_eq!(helpers.len(), 2);

        let kernel = document.callables_named("kernel").next().unwrap();
        assert_eq!(
            kernel.definition_header_text().unwrap(),
            ".visible .entry kernel(\n    .param .u64 output\n)\n.maxntid 128, 1, 1\n"
        );
        assert_eq!(kernel.body_text().unwrap(), "\n    ret;\n");
        assert_eq!(
            kernel.text(),
            ".visible .entry kernel(\n    .param .u64 output\n)\n.maxntid 128, 1, 1\n{\n    ret;\n}"
        );
        assert!(helpers[0].definition_header_text().is_none());
        assert!(helpers[0].body_text().is_none());

        let definition = document.definitions_named("kernel").next().unwrap();
        assert_eq!(definition.callable(), kernel);
        assert_eq!(definition.scope(), kernel.definition_scope().unwrap());
        assert_eq!(definition.text(), kernel.text());
        assert_eq!(
            definition.header_text(),
            kernel.definition_header_text().unwrap()
        );
        assert_eq!(definition.body_text(), kernel.body_text().unwrap());
        assert_eq!(
            definition
                .instructions()
                .map(Instruction::base_opcode)
                .collect::<Vec<_>>(),
            ["ret"]
        );
        assert_eq!(document.definitions_named("helper").count(), 0);
        assert_eq!(document.definitions().count(), 1);
    }

    #[test]
    fn restricts_instruction_queries_to_source_ranges() {
        let source = ".visible .entry first() { mov.u32 %r1, 1; ret; }\n\
.visible .entry second() { @!%p1 bra.uni done; done: exit; }\n";
        let document = Document::parse(source).unwrap();
        let second = document.callables_named("second").next().unwrap();
        let instructions = document
            .instructions_in(second.body_span().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .map(|instruction| instruction.head())
                .collect::<Vec<_>>(),
            ["bra.uni", "exit"]
        );
        assert_eq!(instructions[0].base_opcode(), "bra");
        assert_eq!(instructions[1].base_opcode(), "exit");
        assert!(document.instructions_in(10..10).next().is_none());
    }

    #[test]
    fn preserves_utf8_byte_offsets_while_masking_non_code() {
        let source = "// λλ\nmov.u32 %r1, %tid.x;";
        let document = Document::parse(source).unwrap();
        let instruction = &document.instructions()[0];
        assert_eq!(&source[instruction.span()], instruction.text());
        assert_eq!(instruction.head_offset(), source.find("mov.u32").unwrap());
    }

    #[test]
    fn supports_multiple_instructions_on_one_line() {
        let document = Document::parse("mov.u32 %r1, 1; add.u32 %r2, %r1, 2;").unwrap();
        assert_eq!(
            document
                .instructions()
                .iter()
                .map(Instruction::head)
                .collect::<Vec<_>>(),
            ["mov.u32", "add.u32"]
        );
    }

    #[test]
    fn keeps_valid_instructions_after_a_recoverable_statement_error() {
        let document = Document::parse("mov.u32 %r1, [oops;\nret;").unwrap();
        assert_eq!(
            document
                .instructions()
                .iter()
                .map(Instruction::head)
                .collect::<Vec<_>>(),
            ["ret"]
        );
        assert_eq!(
            document.diagnostics()[0].kind(),
            DiagnosticKind::UnterminatedDelimiter
        );
    }

    #[test]
    fn retains_double_colon_instruction_modifiers_in_the_head() {
        let document = Document::parse(
            "L0: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r1];",
        )
        .unwrap();
        let instruction = &document.instructions()[0];
        assert_eq!(
            instruction.head(),
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
        );
        assert_eq!(instruction.prefix(), "L0:");
        assert_eq!(instruction.operands().collect::<Vec<_>>(), ["[%r1]"]);
    }

    #[test]
    fn rejects_unterminated_non_code_regions() {
        assert_eq!(
            Document::parse("/* no end").unwrap_err(),
            ParseError::UnterminatedBlockComment { offset: 0 }
        );
        assert_eq!(
            Document::parse(".file 1 \"no end").unwrap_err(),
            ParseError::UnterminatedQuotedString { offset: 8 }
        );
    }

    #[test]
    fn splits_nested_top_level_operands() {
        assert_eq!(
            split_top_level("{%r1, %r2}, [%rd1, {%r3, %r4}], (%r5, %r6)").unwrap(),
            ["{%r1, %r2}", "[%rd1, {%r3, %r4}]", "(%r5, %r6)"]
        );
        assert!(split_top_level("%r1, [%r2").is_none());
    }

    #[test]
    fn projects_scalar_multi_binding_and_bank_register_declarations() {
        let source = "\
.reg .pred %p0;
{
    .reg .u64 dst64, mbar64;
    .reg .v2 .b32 %pair<4>;
}
";
        let document = Document::parse(source).unwrap();
        let declarations = document
            .register_declarations()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(declarations.len(), 3);
        assert_eq!(declarations[0].qualifiers().collect::<Vec<_>>(), [".pred"]);
        assert_eq!(declarations[0].bindings()[0].name(), "%p0");
        assert_eq!(declarations[0].bindings()[0].bank_size(), None);
        assert_eq!(
            declarations[1]
                .bindings()
                .iter()
                .map(RegisterBinding::name)
                .collect::<Vec<_>>(),
            ["dst64", "mbar64"]
        );
        assert_eq!(
            declarations[2].qualifiers().collect::<Vec<_>>(),
            [".v2", ".b32"]
        );
        assert_eq!(declarations[2].bindings()[0].name(), "%pair");
        assert_eq!(declarations[2].bindings()[0].bank_size(), Some(4));
        assert_eq!(&source[declarations[2].bindings()[0].span()], "%pair<4>");
        assert_eq!(&source[declarations[2].bindings()[0].name_span()], "%pair");
        assert_ne!(declarations[0].scope(), declarations[1].scope());
    }

    #[test]
    fn rejects_partial_register_declaration_semantics() {
        let source = ".reg %r0;\n.reg .b32;\n.reg .b32 %r<0>;\n.reg .b32 %r0 = 1;\n";
        let document = Document::parse(source).unwrap();
        let errors = document
            .register_declarations()
            .map(Result::unwrap_err)
            .collect::<Vec<_>>();
        assert_eq!(errors.len(), 4);
        assert_eq!(
            errors[0].kind(),
            RegisterDeclarationErrorKind::MissingQualifier
        );
        assert_eq!(
            errors[1].kind(),
            RegisterDeclarationErrorKind::MissingBinding
        );
        assert_eq!(
            errors[2].kind(),
            RegisterDeclarationErrorKind::InvalidBankSize
        );
        assert_eq!(
            errors[3].kind(),
            RegisterDeclarationErrorKind::UnexpectedToken
        );
        for error in errors {
            assert_eq!(
                document.statement(error.statement()).unwrap().kind(),
                StatementKind::Directive
            );
            assert!(error.offset() < source.len());
        }
    }

    #[test]
    fn arbitrary_ascii_never_panics_or_loses_successfully_lexed_input() {
        const ALPHABET: &[u8] = b" abcXYZ019_.$%@!,:;{}[]()/*\\\"#\n\t+-|=";
        let mut state = 0x9e37_79b9_u32;
        for length in 0..512 {
            let mut source = String::with_capacity(length);
            for _ in 0..length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                source.push(ALPHABET[(state as usize) % ALPHABET.len()] as char);
            }
            let Ok(document) = Document::parse(&source) else {
                continue;
            };
            assert_eq!(
                document
                    .tokens()
                    .iter()
                    .map(|token| token.text(&source))
                    .collect::<String>(),
                source
            );
            assert!(document.coverage().is_lossless());
            assert!(
                document
                    .statements()
                    .iter()
                    .all(|statement| document.scope(statement.scope()).is_some())
            );
            assert!(document.instructions().iter().all(|instruction| {
                document
                    .statement(instruction.statement())
                    .is_some_and(|statement| statement.kind() == StatementKind::Instruction)
            }));
        }
    }
}
