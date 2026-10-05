/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! AST module types and source-location tracking for the JIT compiler.
//!
//! ## Source Location Recovery
//!
//! At proc-macro expansion time we have access to real file / line / column
//! information through `proc_macro2::Span`.  That information is lost when the
//! AST is round-tripped through `quote!` → `syn::parse2` because `proc_macro2`
//! runs in *fallback* mode at runtime and every token gets
//! `Span::call_site()` (= `1:0`).
//!
//! To recover accurate positions we use the following scheme:
//!
//! 1. **At proc-macro time** we call `Span::source_text()` on the whole
//!    `ItemMod` to obtain the *verbatim* original source text (whitespace,
//!    newlines and all).  We also record the **span base** – the `(file,
//!    line, column)` of the module's opening token – via `Span::file()` and
//!    `Span::start()`.
//!
//! 2. **At runtime** we feed that source text to `syn::parse_str` instead of
//!    re-quoting.  Because the string is character-for-character identical to
//!    the original source, `syn::parse_str` produces spans whose line/column
//!    numbers are offsets *within that string* that correspond 1-to-1 with the
//!    original file layout.
//!
//! 3. Any runtime span can then be mapped to an absolute `SourceLocation` via:
//!
//!    ```text
//!    absolute_line = base_line + (span_line − 1)
//!    absolute_col  = if span_line == 1 { base_col + span_col }
//!                    else               { span_col }
//!    ```
//!
//! This gives exact file / line / column for *every* node in the syn AST –
//! statements, expressions, sub-expressions, individual tokens – without
//! requiring any up-front walk or key-based lookup table.

pub use syn;

use syn::ItemMod;

// ---------------------------------------------------------------------------
// SourceLocation
// ---------------------------------------------------------------------------

/// A concrete source location: file path + 1-indexed line + 0-indexed column.
///
/// This is independent of `proc_macro2::Span` and can be stored, displayed and
/// attached to Tile IR locations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    /// Source file path as reported by the compiler (`Span::file()`).
    pub file: String,
    /// 1-indexed line number.
    pub line: usize,
    /// 0-indexed column number (UTF-8 characters).
    pub column: usize,
}

impl SourceLocation {
    pub fn new(file: String, line: usize, column: usize) -> Self {
        Self { file, line, column }
    }

    /// Sentinel for "no location available".
    pub fn unknown() -> Self {
        Self {
            file: String::new(),
            line: 0,
            column: 0,
        }
    }

    /// `true` when this location carries meaningful source information.
    pub fn is_known(&self) -> bool {
        self.line > 0 && !self.file.is_empty()
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_known() {
            write!(f, "{}:{}:{}", self.file, self.line, self.column)
        } else {
            write!(f, "<unknown>")
        }
    }
}

// ---------------------------------------------------------------------------
// SpanBase
// ---------------------------------------------------------------------------

/// The "anchor" needed to turn a runtime `proc_macro2::Span` (whose
/// line/column numbers are relative to the re-parsed source-text string) into
/// an absolute [`SourceLocation`].
///
/// Captured once per module at proc-macro expansion time.
#[derive(Debug, Clone)]
pub struct SpanBase {
    /// The source file that contains this module.
    pub file: String,
    /// 1-indexed line of the module's first token in the original file.
    pub base_line: usize,
    /// 0-indexed column of the module's first token in the original file.
    pub base_col: usize,
}

impl SpanBase {
    pub fn new(file: String, base_line: usize, base_col: usize) -> Self {
        Self {
            file,
            base_line,
            base_col,
        }
    }

    /// A sentinel for when no span information was captured.
    pub fn unknown() -> Self {
        Self {
            file: String::new(),
            base_line: 0,
            base_col: 0,
        }
    }

    pub fn is_known(&self) -> bool {
        self.base_line > 0 && !self.file.is_empty()
    }

    /// Convert a *string-relative* `(line, column)` – as obtained from a
    /// `proc_macro2::Span` produced by `syn::parse_str` on the verbatim
    /// source text – into an absolute [`SourceLocation`].
    ///
    /// The arithmetic is:
    /// ```text
    /// abs_line = base_line + (str_line - 1)
    /// abs_col  = if str_line == 1 { base_col + str_col } else { str_col }
    /// ```
    ///
    /// The column adjustment on the first line accounts for the fact that the
    /// source text may not start at column 0 of its line in the original file
    /// (e.g. the module body is indented).  Subsequent lines in
    /// `source_text()` start at column 0 of their respective file lines, so
    /// no adjustment is needed.
    pub fn resolve(&self, str_line: usize, str_col: usize) -> SourceLocation {
        if !self.is_known() || str_line == 0 {
            return SourceLocation::unknown();
        }
        let abs_line = self.base_line + (str_line - 1);
        let abs_col = if str_line == 1 {
            self.base_col + str_col
        } else {
            str_col
        };
        SourceLocation::new(self.file.clone(), abs_line, abs_col)
    }

    /// Convenience: resolve a `proc_macro2::Span`'s start position.
    pub fn resolve_span(&self, span: &proc_macro2::Span) -> SourceLocation {
        let start = span.start();
        self.resolve(start.line, start.column)
    }

    /// Return the module-level source location (first token).
    pub fn module_location(&self) -> SourceLocation {
        if self.is_known() {
            SourceLocation::new(self.file.clone(), self.base_line, self.base_col)
        } else {
            SourceLocation::unknown()
        }
    }
}

impl Default for SpanBase {
    fn default() -> Self {
        Self::unknown()
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// A parsed module AST together with the information needed to recover source
/// locations for any node inside it.
#[derive(Clone)]
pub struct Module {
    /// Short name of the module (e.g. `"core"`, `"my_kernels"`).
    name: &'static str,
    /// Fully-qualified Rust module path (e.g. `"cutile::core"`, `"my_app::activations"`).
    /// Set from `module_path!()` at the definition site. Used for deduplication
    /// when multiple modules import the same dependency.
    absolute_path: String,
    /// The syn AST.  When constructed via [`Module::with_span_base`] the AST
    /// was produced by `syn::parse_str` on the original source text, so every
    /// token span is string-relative and can be mapped to an absolute position
    /// through [`span_base`].
    ast: ItemMod,
    /// Anchor for converting runtime spans → absolute source locations.
    span_base: SpanBase,
}

impl Module {
    /// Create a module **without** source-location tracking.
    ///
    /// This is the backwards-compatible constructor used by modules that do
    /// not (yet) capture source text (e.g. the `core` DSL module).
    pub fn new(name: &'static str, ast: ItemMod) -> Self {
        Self {
            name,
            absolute_path: String::new(),
            ast,
            span_base: SpanBase::unknown(),
        }
    }

    /// Create a module **with** a [`SpanBase`] so that every runtime span in
    /// `ast` can be resolved to an absolute [`SourceLocation`].
    pub fn with_span_base(name: &'static str, ast: ItemMod, span_base: SpanBase) -> Self {
        Self {
            name,
            absolute_path: String::new(),
            ast,
            span_base,
        }
    }

    /// Set the fully-qualified module path (from `module_path!()` at the definition site).
    pub fn set_absolute_path(&mut self, path: String) {
        self.absolute_path = path;
    }

    /// Returns the fully-qualified module path, or the short name if not set.
    pub fn absolute_path(&self) -> &str {
        if self.absolute_path.is_empty() {
            self.name
        } else {
            &self.absolute_path
        }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn ast(&self) -> &ItemMod {
        &self.ast
    }

    pub fn span_base(&self) -> &SpanBase {
        &self.span_base
    }

    /// Resolve any `proc_macro2::Span` (from a node inside `self.ast()`) to
    /// an absolute [`SourceLocation`].
    ///
    /// This is the primary entry point used by the JIT compiler: given *any*
    /// syn AST node obtained from this module, call
    /// `module.resolve_span(&node.span())` to get a file:line:col triple that
    /// points back to the user's source code.
    pub fn resolve_span(&self, span: &proc_macro2::Span) -> SourceLocation {
        self.span_base.resolve_span(span)
    }
}
