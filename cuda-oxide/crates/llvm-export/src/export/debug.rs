/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug metadata emission.
//!
//! Line-table mode emits just enough metadata to map machine instructions back
//! to source lines. Full mode builds on that with the first variable/type slice:
//! simple source locals are described with `llvm.dbg.declare` and compact DWARF
//! type nodes.

use std::{
    fmt::Write,
    path::{Path, PathBuf},
};

use combine::stream::position::SourcePosition;
use pliron::{
    location::{Location, Source},
    uniqued_any,
};

use crate::ops::{
    DebugGlobalVariableInfo, DebugLocalTypeKind, DebugLocalVariableInfo,
    DebugProjectedVariableInfo, DebugSourcePosition, DebugWholeVariableInfo,
};

use super::config::FunctionLocalStaticPlacement;
use super::state::{ModuleExportState, ResolvedDebugScope};

impl<'a> ModuleExportState<'a> {
    pub(super) fn has_debug_metadata(&self) -> bool {
        self.debug_compile_unit.is_some()
    }

    pub(super) fn debug_subprogram_for_function(
        &mut self,
        name: &str,
        linkage_name: &str,
        loc: &Location,
    ) -> Option<usize> {
        if !self.debug_kind.line_tables_enabled() {
            return None;
        }
        if let Some(existing) = self.debug_function_subprograms.get(linkage_name) {
            return Some(*existing);
        }

        let (path, pos) = self.source_position_from_location(loc)?;
        let cu_id = self.ensure_debug_compile_unit(&path);
        let file_id = self.ensure_debug_file(&path);
        let subroutine_type_id = self.ensure_debug_subroutine_type();
        // Shared-static owners are scoped to their source namespace so the
        // owning DISubprogram and the AS3 DIGlobalVariable agree on identity.
        // The scope map and the subprogram cache are both keyed by the final
        // exported (linkage) name, which module indexing guarantees is unique.
        let owner_scope = self.debug_shared_function_scopes.get(linkage_name).cloned();
        let (debug_name, scope_id) = if let Some(owner_scope) = owner_scope {
            let scope_id = self
                .ensure_debug_namespace(&owner_scope.namespace)
                .expect("validated shared owner namespace must yield a scope");
            (owner_scope.name, scope_id)
        } else {
            (name.to_string(), file_id)
        };
        let debug_name = escape_debug_string(&debug_name);
        let escaped_linkage_name = escape_debug_string(linkage_name);
        let line = pos.line;
        let id = self.alloc_metadata_id();

        self.debug_nodes.push((
            id,
            format!(
                "distinct !DISubprogram(name: \"{debug_name}\", \
                 linkageName: \"{escaped_linkage_name}\", \
                 scope: !{scope_id}, file: !{file_id}, \
                 line: {line}, type: !{subroutine_type_id}, scopeLine: {line}, \
                 spFlags: DISPFlagDefinition, unit: !{cu_id}, retainedNodes: !{{}})"
            ),
        ));
        self.debug_function_subprograms
            .insert(linkage_name.to_string(), id);
        self.debug_subprogram_files.insert(id, path);
        self.debug_subprogram_fallbacks
            .insert(id, (pos.line, pos.column));

        Some(id)
    }

    pub(super) fn register_debug_source_scopes_for_function(
        &mut self,
        scope: usize,
        op: pliron::context::Ptr<pliron::operation::Operation>,
    ) {
        let Some(map) = crate::ops::debug_source_scope_map(self.ctx, op) else {
            return;
        };
        self.debug_source_scope_maps.insert(scope, map);
    }

    pub(super) fn attach_debug_to_last_line(
        &mut self,
        output: &mut String,
        output_before: usize,
        scope: Option<usize>,
        loc: &Location,
        allow_scope_fallback: bool,
    ) {
        if output.len() == output_before {
            return;
        }

        let Some(scope) = scope else {
            return;
        };
        let location_id = if crate::is_artificial_debug_location(loc) {
            Some(self.ensure_artificial_debug_location(scope))
        } else {
            self.debug_location_for_scope(scope, loc).or_else(|| {
                if allow_scope_fallback {
                    // LLVM rejects inlinable calls inside a debug-scoped function
                    // unless the call itself has a location. When rustc/pliron did
                    // not give the call one, point it at the function line instead
                    // of letting opt discard the whole debug graph.
                    self.debug_fallback_location_for_scope(scope)
                } else {
                    None
                }
            })
        };
        let Some(location_id) = location_id else {
            return;
        };

        if output.ends_with('\n') {
            output.pop();
            writeln!(output, ", !dbg !{location_id}").unwrap();
        }
    }

    pub(super) fn emit_debug_intrinsic_declarations(&self, output: &mut String) {
        if self.debug_declare_used {
            writeln!(
                output,
                "declare void @llvm.dbg.declare(metadata, metadata, metadata)"
            )
            .unwrap();
        }
        if self.debug_value_used {
            writeln!(
                output,
                "declare void @llvm.dbg.value(metadata, metadata, metadata)"
            )
            .unwrap();
        }
    }

    pub(super) fn emit_debug_metadata(&mut self, output: &mut String) {
        let Some(cu_id) = self.debug_compile_unit else {
            return;
        };

        self.finalize_debug_globals(cu_id);

        let dwarf_version_id = self.alloc_metadata_id();
        let debug_info_version_id = self.alloc_metadata_id();

        writeln!(output, "!llvm.dbg.cu = !{{!{cu_id}}}").unwrap();
        writeln!(
            output,
            "!llvm.module.flags = !{{!{dwarf_version_id}, !{debug_info_version_id}}}"
        )
        .unwrap();
        writeln!(
            output,
            "!{dwarf_version_id} = !{{i32 2, !\"Dwarf Version\", i32 2}}"
        )
        .unwrap();
        writeln!(
            output,
            "!{debug_info_version_id} = !{{i32 2, !\"Debug Info Version\", i32 3}}"
        )
        .unwrap();

        for (id, node) in &self.debug_nodes {
            writeln!(output, "!{id} = {node}").unwrap();
        }
    }

    /// Create a global-variable expression for one source Rust static.
    ///
    /// `linkage_name` is the actual generated LLVM symbol, while `info.ty` is
    /// the semantic Rust type. Keeping those independent is required for
    /// initialized globals whose physical storage is `[N x i8]`.
    pub(super) fn debug_global_variable(
        &mut self,
        linkage_name: &str,
        alignment_bytes: Option<u64>,
        address_space: u32,
        owner_function: Option<&str>,
        info: &DebugGlobalVariableInfo,
    ) -> Result<Option<usize>, String> {
        if !self.debug_kind.variables_enabled() {
            return Ok(None);
        }
        if info.name.is_empty()
            || info.namespace.is_empty()
            || info.namespace.iter().any(String::is_empty)
            || info.declaration.file.as_os_str().is_empty()
            || info.declaration.line <= 0
            || info.declaration.column <= 0
        {
            return Ok(None);
        }
        if self.debug_globals_finalized {
            return Err(format!(
                "cannot add debug global `{linkage_name}` after compile-unit finalization"
            ));
        }
        let owner_function = owner_function.map(ToOwned::to_owned);
        if let Some((previous, previous_address_space, previous_owner, expression_id)) =
            self.debug_global_variables.get(linkage_name)
        {
            if previous == info
                && *previous_address_space == address_space
                && *previous_owner == owner_function
            {
                return Ok(Some(*expression_id));
            }
            return Err(format!(
                "conflicting debug identities for LLVM global `@{linkage_name}`: {:?} in address space {} owned by {:?} versus {:?} in address space {} owned by {:?}",
                previous,
                previous_address_space,
                previous_owner,
                info,
                address_space,
                owner_function
            ));
        }

        self.ensure_debug_compile_unit(&info.declaration.file);
        let file_id = self.ensure_debug_file(&info.declaration.file);
        let type_id = self.ensure_debug_type(&info.ty);
        let scope_id = if info.is_function_local {
            let Some(owner) = owner_function.as_deref() else {
                return Ok(None);
            };
            let Some(scope) = self.debug_function_subprograms.get(owner) else {
                return Ok(None);
            };
            *scope
        } else {
            self.ensure_debug_namespace(&info.namespace)
                .expect("validated non-empty namespace must yield a scope")
        };
        let name = escape_debug_string(&info.name);
        let escaped_linkage_name = escape_debug_string(linkage_name);
        let alignment = alignment_bytes
            .filter(|alignment| *alignment != 0)
            .and_then(|alignment| alignment.checked_mul(8))
            .map(|alignment_bits| format!(", align: {alignment_bits}"))
            .unwrap_or_default();

        let variable_id = self.alloc_metadata_id();
        self.debug_nodes.push((
            variable_id,
            format!(
                "distinct !DIGlobalVariable(name: \"{name}\", linkageName: \"{escaped_linkage_name}\", \
                 scope: !{scope_id}, file: !{file_id}, line: {}, type: !{type_id}, \
                 isLocal: {}, isDefinition: true{alignment})",
                info.declaration.line, info.is_local_to_unit
            ),
        ));

        let expression_id = self.alloc_metadata_id();
        // Clang's NVPTX frontend represents a shared variable's CUDA DWARF
        // address class with this target expression. NVPTXDwarfDebug consumes
        // the sequence and emits DW_AT_address_class = 8 (shared space) for
        // cuda-gdb. AS1 global and AS4 constant variables use the established
        // empty expression; NVPTX derives class 4 for AS4 from the physical
        // global's address space.
        let expression = if address_space == crate::types::address_space::SHARED {
            "!DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef)"
        } else {
            "!DIExpression()"
        };
        self.debug_nodes.push((
            expression_id,
            format!("!DIGlobalVariableExpression(var: !{variable_id}, expr: {expression})"),
        ));
        // A function-local static has exactly one verifier-accepted home,
        // and which one depends on the consuming LLVM (see
        // [`FunctionLocalStaticPlacement`]). Module globals always belong to
        // the compile unit.
        if info.is_function_local
            && self.debug_function_local_static_placement
                == FunctionLocalStaticPlacement::SubprogramRetainedNodes
        {
            self.debug_subprogram_retained_globals
                .entry(scope_id)
                .or_default()
                .push(expression_id);
        } else {
            self.debug_global_expressions.push(expression_id);
        }
        self.debug_global_variables.insert(
            linkage_name.to_string(),
            (info.clone(), address_space, owner_function, expression_id),
        );
        Ok(Some(expression_id))
    }

    fn ensure_debug_namespace(&mut self, segments: &[String]) -> Option<usize> {
        let mut parent = None;
        for segment in segments {
            let key = (parent, segment.clone());
            if let Some(existing) = self.debug_namespaces.get(&key) {
                parent = Some(*existing);
                continue;
            }

            let id = self.alloc_metadata_id();
            let scope = parent
                .map(|parent| format!("!{parent}"))
                .unwrap_or_else(|| "null".to_string());
            let name = escape_debug_string(segment);
            self.debug_nodes.push((
                id,
                format!("!DINamespace(name: \"{name}\", scope: {scope})"),
            ));
            self.debug_namespaces.insert(key, id);
            parent = Some(id);
        }
        parent
    }

    /// Add the global-expression tuple to the already-reserved compile unit.
    ///
    /// Compile units are allocated as soon as the first source object is seen,
    /// but the complete global list is only known after top-level export. LLVM
    /// metadata permits the resulting forward reference.
    fn finalize_debug_globals(&mut self, cu_id: usize) {
        if self.debug_globals_finalized {
            return;
        }
        self.debug_globals_finalized = true;
        self.finalize_debug_retained_globals();
        if self.debug_global_expressions.is_empty() {
            return;
        }

        let expressions = self
            .debug_global_expressions
            .iter()
            .map(|id| format!("!{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        let globals_id = self.alloc_metadata_id();
        self.debug_nodes
            .push((globals_id, format!("!{{{expressions}}}")));

        let Some((_, compile_unit)) = self.debug_nodes.iter_mut().find(|(id, _)| *id == cu_id)
        else {
            debug_assert!(false, "reserved debug compile unit must exist");
            return;
        };
        let Some(prefix) = compile_unit.strip_suffix(')') else {
            debug_assert!(false, "debug compile unit must end in ')'");
            return;
        };
        *compile_unit = format!("{prefix}, globals: !{globals_id})");
    }

    /// Splice function-local static expressions into their owning
    /// `DISubprogram`'s `retainedNodes` tuple.
    ///
    /// Owners are written with an empty tuple when the function is exported,
    /// because AS3 globals are emitted before functions and the complete
    /// retained set is only known after top-level export (the same forward
    /// dependency `finalize_debug_globals` resolves for the compile unit).
    /// Owners are visited in metadata-id order so the output is
    /// deterministic; the map is drained, so a repeated call is a no-op.
    fn finalize_debug_retained_globals(&mut self) {
        let mut owners: Vec<(usize, Vec<usize>)> =
            self.debug_subprogram_retained_globals.drain().collect();
        owners.sort_by_key(|(owner, _)| *owner);
        for (owner, expressions) in owners {
            let Some((_, subprogram)) = self.debug_nodes.iter_mut().find(|(id, _)| *id == owner)
            else {
                debug_assert!(false, "retained-node owner DISubprogram must exist");
                continue;
            };
            let Some(prefix) = subprogram.strip_suffix("retainedNodes: !{})") else {
                debug_assert!(
                    false,
                    "retained-node owner must be a DISubprogram with an empty retainedNodes tuple"
                );
                continue;
            };
            let retained = expressions
                .iter()
                .map(|id| format!("!{id}"))
                .collect::<Vec<_>>()
                .join(", ");
            *subprogram = format!("{prefix}retainedNodes: !{{{retained}}})");
        }
    }

    pub(super) fn debug_local_variable_for_scope(
        &mut self,
        scope: usize,
        loc: &Location,
        op: pliron::context::Ptr<pliron::operation::Operation>,
        info: &DebugLocalVariableInfo,
    ) -> Option<(usize, usize)> {
        let source_scope = crate::ops::debug_local_source_scope(self.ctx, op);
        let declaration = crate::ops::debug_local_declaration_location(self.ctx, op);
        self.debug_local_variable_for_source(scope, loc, info, source_scope, declaration)
    }

    pub(super) fn debug_projected_variable_for_scope(
        &mut self,
        scope: usize,
        loc: &Location,
        projected: &DebugProjectedVariableInfo,
    ) -> Option<(usize, usize)> {
        let declaration = projected.declaration.as_ref().map(|declaration| {
            (
                declaration.file.clone(),
                SourcePosition {
                    line: declaration.line,
                    column: declaration.column,
                },
            )
        });
        self.debug_local_variable_for_source(
            scope,
            loc,
            &projected.variable,
            projected.source_scope,
            declaration,
        )
    }

    pub(super) fn debug_whole_variable_for_scope(
        &mut self,
        scope: usize,
        loc: &Location,
        whole: &DebugWholeVariableInfo,
    ) -> Option<(usize, usize)> {
        let declaration = whole.declaration.as_ref().map(|declaration| {
            (
                declaration.file.clone(),
                SourcePosition {
                    line: declaration.line,
                    column: declaration.column,
                },
            )
        });
        self.debug_local_variable_for_source(
            scope,
            loc,
            &whole.variable,
            whole.source_scope,
            declaration,
        )
    }

    fn debug_local_variable_for_source(
        &mut self,
        scope: usize,
        loc: &Location,
        info: &DebugLocalVariableInfo,
        source_scope: Option<u32>,
        declaration: Option<(PathBuf, SourcePosition)>,
    ) -> Option<(usize, usize)> {
        if !self.debug_kind.variables_enabled() {
            return None;
        }

        let (path, pos) =
            declaration.or_else(|| self.local_variable_position_from_location(loc))?;
        let file_id = self.ensure_debug_file(&path);
        let resolved_scope = source_scope
            .and_then(|source_scope| self.resolve_debug_source_scope(scope, source_scope))
            .unwrap_or_else(|| ResolvedDebugScope {
                scope: self.debug_scope_for_file(scope, &path).unwrap_or(scope),
                inlined_at: None,
            });
        let variable_scope = resolved_scope.scope;
        let location_id = self
            .debug_location_for_resolved_scope(resolved_scope, loc)
            .or_else(|| {
                let location_scope = self.debug_scope_for_file(resolved_scope.scope, &path)?;
                self.ensure_debug_location(location_scope, pos, resolved_scope.inlined_at)
            })?;
        let key = (variable_scope, path, pos.line, info.clone());
        if let Some(var_id) = self.debug_local_variables.get(&key).copied() {
            return Some((var_id, location_id));
        }

        let type_id = self.ensure_debug_type(&info.ty);
        let name = escape_debug_string(&info.name);
        let arg = info
            .argument_index
            .map(|idx| format!("arg: {idx}, "))
            .unwrap_or_default();
        let id = self.alloc_metadata_id();

        self.debug_nodes.push((
            id,
            format!(
                "!DILocalVariable(name: \"{name}\", {arg}scope: !{variable_scope}, file: !{file_id}, \
                 line: {}, type: !{type_id})",
                pos.line
            ),
        ));
        self.debug_local_variables.insert(key, id);

        Some((id, location_id))
    }

    fn resolve_debug_source_scope(
        &mut self,
        function_scope: usize,
        source_scope: u32,
    ) -> Option<ResolvedDebugScope> {
        if let Some(scope) = self
            .debug_resolved_source_scopes
            .get(&(function_scope, source_scope))
            .copied()
        {
            return Some(scope);
        }

        let source_scope_data = {
            let map = self.debug_source_scope_maps.get(&function_scope)?;
            map.scopes
                .iter()
                .find(|candidate| candidate.id == source_scope)?
                .clone()
        };
        let is_root_scope = source_scope_data.parent.is_none();

        let parent = source_scope_data
            .parent
            // rustc emits SourceScopes as a tree in which every parent precedes
            // its child (parent id < child id). Requiring that here matches the
            // real invariant and guarantees termination: a malformed map with a
            // cyclic or forward parent link degrades to the function scope
            // instead of recursing without bound.
            .filter(|&parent| parent < source_scope)
            .and_then(|parent| self.resolve_debug_source_scope(function_scope, parent))
            .unwrap_or(ResolvedDebugScope {
                scope: function_scope,
                inlined_at: None,
            });

        let resolved = if is_root_scope && source_scope_data.inlined.is_none() {
            parent
        } else if let Some(inlined) = source_scope_data.inlined {
            let span = source_scope_data.span.as_ref();
            let callee_scope = self.ensure_debug_inlined_subprogram(&inlined.callee_name, span)?;
            let inlined_at = inlined
                .callsite
                .as_ref()
                .and_then(|callsite| self.ensure_debug_location_from_position(parent, callsite))
                .or(parent.inlined_at);

            ResolvedDebugScope {
                scope: callee_scope,
                inlined_at,
            }
        } else if let Some(span) = source_scope_data.span.as_ref() {
            ResolvedDebugScope {
                scope: self.ensure_debug_lexical_block(parent.scope, span)?,
                inlined_at: parent.inlined_at,
            }
        } else {
            parent
        };

        self.debug_resolved_source_scopes
            .insert((function_scope, source_scope), resolved);
        Some(resolved)
    }

    fn ensure_debug_inlined_subprogram(
        &mut self,
        name: &str,
        span: Option<&DebugSourcePosition>,
    ) -> Option<usize> {
        let span = span?;
        let key = (name.to_string(), span.file.clone(), span.line);
        if let Some(id) = self.debug_inlined_subprograms.get(&key).copied() {
            return Some(id);
        }

        let cu_id = self.ensure_debug_compile_unit(&span.file);
        let file_id = self.ensure_debug_file(&span.file);
        let subroutine_type_id = self.ensure_debug_subroutine_type();
        let name = escape_debug_string(name);
        let id = self.alloc_metadata_id();

        self.debug_nodes.push((
            id,
            format!(
                "distinct !DISubprogram(name: \"{name}\", scope: !{file_id}, file: !{file_id}, \
                 line: {}, type: !{subroutine_type_id}, scopeLine: {}, \
                 spFlags: DISPFlagDefinition, unit: !{cu_id}, retainedNodes: !{{}})",
                span.line, span.line
            ),
        ));
        self.debug_subprogram_files.insert(id, span.file.clone());
        self.debug_subprogram_fallbacks
            .insert(id, (span.line, span.column));
        self.debug_inlined_subprograms.insert(key, id);

        Some(id)
    }

    fn ensure_debug_lexical_block(
        &mut self,
        parent_scope: usize,
        span: &DebugSourcePosition,
    ) -> Option<usize> {
        if span.line <= 0 || span.column <= 0 {
            return Some(parent_scope);
        }

        let key = (parent_scope, span.file.clone(), span.line, span.column);
        if let Some(id) = self.debug_lexical_blocks.get(&key).copied() {
            return Some(id);
        }

        let file_id = self.ensure_debug_file(&span.file);
        let id = self.alloc_metadata_id();
        self.debug_nodes.push((
            id,
            format!(
                "!DILexicalBlock(scope: !{parent_scope}, file: !{file_id}, line: {}, column: {})",
                span.line, span.column
            ),
        ));
        self.debug_lexical_blocks.insert(key, id);
        self.debug_subprogram_files.insert(id, span.file.clone());

        Some(id)
    }

    fn ensure_debug_compile_unit(&mut self, path: &Path) -> usize {
        if let Some(id) = self.debug_compile_unit {
            return id;
        }

        let file_id = self.ensure_debug_file(path);
        let id = self.alloc_metadata_id();
        let is_optimized = if self.debug_kind.variables_enabled() {
            "false"
        } else {
            "true"
        };
        let emission_kind = if self.debug_kind.variables_enabled() {
            "FullDebug"
        } else {
            "LineTablesOnly"
        };
        self.debug_nodes.push((
            id,
            format!(
                "distinct !DICompileUnit(language: DW_LANG_Rust, file: !{file_id}, \
                 producer: \"cuda-oxide\", isOptimized: {is_optimized}, runtimeVersion: 0, \
                 emissionKind: {emission_kind})"
            ),
        ));
        self.debug_compile_unit = Some(id);
        id
    }

    fn ensure_debug_file(&mut self, path: &Path) -> usize {
        if let Some(id) = self.debug_files.get(path).copied() {
            return id;
        }

        let (filename, directory) = split_file_and_directory(path);
        let filename = escape_debug_string(&filename);
        let directory = escape_debug_string(&directory);
        let id = self.alloc_metadata_id();

        self.debug_nodes.push((
            id,
            format!("!DIFile(filename: \"{filename}\", directory: \"{directory}\")"),
        ));
        self.debug_files.insert(path.to_path_buf(), id);

        id
    }

    fn ensure_debug_subroutine_type(&mut self) -> usize {
        if let Some(id) = self.debug_subroutine_type {
            return id;
        }

        let id = self.alloc_metadata_id();
        self.debug_nodes
            .push((id, "!DISubroutineType(types: !{null})".to_string()));
        self.debug_subroutine_type = Some(id);

        id
    }

    fn ensure_debug_type(&mut self, ty: &DebugLocalTypeKind) -> usize {
        if let Some(id) = self.debug_types.get(ty).copied() {
            return id;
        }

        if matches!(ty, DebugLocalTypeKind::Enum { .. }) {
            return self.ensure_enum_debug_type(ty);
        }

        let node = match ty {
            DebugLocalTypeKind::Basic {
                name,
                size_bits,
                encoding,
            } => {
                let name = escape_debug_string(name);
                format!("!DIBasicType(name: \"{name}\", size: {size_bits}, encoding: {encoding})")
            }
            DebugLocalTypeKind::Pointer { name, size_bits } => {
                let name = escape_debug_string(name);
                format!(
                    "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"{name}\", \
                     baseType: null, size: {size_bits})"
                )
            }
            DebugLocalTypeKind::TypedPointer {
                name,
                size_bits,
                pointee,
            } => {
                assert!(
                    pointee.is_valid_typed_pointer_pointee(),
                    "typed pointer cannot export an opaque or composite pointee descendant"
                );
                let base = self.ensure_debug_type(pointee);
                let name = escape_debug_string(name);
                format!(
                    "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"{name}\", \
                     baseType: !{base}, size: {size_bits})"
                )
            }
            DebugLocalTypeKind::Struct {
                name,
                size_bits,
                members,
            } => {
                // Emit each member's base type (may recurse) and a DW_TAG_member
                // node, then the elements tuple, then the composite itself.
                let member_ids: Vec<usize> = members
                    .iter()
                    .map(|member| {
                        let base = self.ensure_debug_type(&member.ty);
                        let member_name = escape_debug_string(&member.name);
                        let member_size = member.ty.size_bits();
                        let id = self.alloc_metadata_id();
                        self.debug_nodes.push((
                            id,
                            format!(
                                "!DIDerivedType(tag: DW_TAG_member, name: \"{member_name}\", \
                                 baseType: !{base}, size: {member_size}, offset: {})",
                                member.offset_bits
                            ),
                        ));
                        id
                    })
                    .collect();
                let elements = member_ids
                    .iter()
                    .map(|id| format!("!{id}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let elements_id = self.alloc_metadata_id();
                self.debug_nodes
                    .push((elements_id, format!("!{{{elements}}}")));
                let name = escape_debug_string(name);
                format!(
                    "!DICompositeType(tag: DW_TAG_structure_type, name: \"{name}\", \
                     size: {size_bits}, elements: !{elements_id})"
                )
            }
            DebugLocalTypeKind::Array {
                size_bits,
                element,
                count,
                ..
            } => {
                let base = self.ensure_debug_type(element);
                let subrange_id = self.alloc_metadata_id();
                self.debug_nodes
                    .push((subrange_id, format!("!DISubrange(count: {count})")));
                let elements_id = self.alloc_metadata_id();
                self.debug_nodes
                    .push((elements_id, format!("!{{!{subrange_id}}}")));
                format!(
                    "!DICompositeType(tag: DW_TAG_array_type, baseType: !{base}, \
                     size: {size_bits}, elements: !{elements_id})"
                )
            }
            DebugLocalTypeKind::Enum { .. } => {
                unreachable!("enum debug types use ensure_enum_debug_type")
            }
        };

        let id = self.alloc_metadata_id();
        self.debug_nodes.push((id, node));
        self.debug_types.insert(ty.clone(), id);

        id
    }

    /// Emit a Rust enum using the same parent/child scope relationships rustc
    /// gives LLVM's native DWARF enum builder.
    ///
    /// Reserving the top-level and variant-part metadata IDs first lets child
    /// nodes reference their semantic parents even though LLVM metadata is
    /// serialized as a flat numbered list. This matters for discriminator
    /// members at non-zero offsets: the member must be tied to the enum object
    /// whose address `llvm.dbg.declare` describes, not emitted as an orphan.
    fn ensure_enum_debug_type(&mut self, ty: &DebugLocalTypeKind) -> usize {
        if let Some(id) = self.debug_types.get(ty).copied() {
            return id;
        }

        let DebugLocalTypeKind::Enum {
            name,
            size_bits,
            discriminant,
            variants,
        } = ty
        else {
            unreachable!("ensure_enum_debug_type requires an enum")
        };

        // rustc creates a recursive metadata graph. Reserve the parent IDs and
        // cache the top-level type before emitting children so forward metadata
        // references are well-defined and recursive type discovery terminates.
        let enum_type_id = self.alloc_metadata_id();
        let variant_part_id = self.alloc_metadata_id();
        self.debug_types.insert(ty.clone(), enum_type_id);

        let discriminator_member_id = discriminant.as_ref().map(|discriminant| {
            // LLVM MC's NVPTX text writer prints negative one-byte DWARF
            // constants as 64-bit hex literals, which ptxas rejects.
            // `DW_FORM_data1` is signless; the discriminator type determines
            // its interpretation, so an unsigned carrier preserves the tag
            // bits and comparison semantics. Remove this conversion once all
            // supported llc versions truncate the literal to the record width.
            let discriminator_ty = match discriminant.ty.as_ref() {
                DebugLocalTypeKind::Basic {
                    size_bits,
                    encoding,
                    ..
                } if *encoding == "DW_ATE_signed" => DebugLocalTypeKind::Basic {
                    name: format!("u{size_bits}"),
                    size_bits: *size_bits,
                    encoding: "DW_ATE_unsigned",
                },
                ty => ty.clone(),
            };
            let base = self.ensure_debug_type(&discriminator_ty);
            let id = self.alloc_metadata_id();
            self.debug_nodes.push((
                id,
                format!(
                    "!DIDerivedType(tag: DW_TAG_member, scope: !{enum_type_id}, \
                     baseType: !{base}, size: {}, offset: {}, flags: DIFlagArtificial)",
                    discriminant.ty.size_bits(),
                    discriminant.offset_bits
                ),
            ));
            id
        });
        let discriminant_width = discriminant
            .as_ref()
            .map(|discriminant| discriminant.ty.size_bits())
            .unwrap_or(64);

        let mut variant_member_ids = Vec::with_capacity(variants.len());
        for variant in variants {
            // The variant struct is a sibling of the variant part under the
            // enum type. Reserve its ID before its fields so each field can
            // carry the correct struct scope, matching rustc native debuginfo.
            let variant_struct_id = self.alloc_metadata_id();
            let mut payload_member_ids = Vec::with_capacity(variant.members.len());
            for member in &variant.members {
                let base = self.ensure_debug_type(&member.ty);
                let member_name = escape_debug_string(&member.name);
                let member_size = member.ty.size_bits();
                let id = self.alloc_metadata_id();
                self.debug_nodes.push((
                    id,
                    format!(
                        "!DIDerivedType(tag: DW_TAG_member, name: \"{member_name}\", \
                         scope: !{variant_struct_id}, baseType: !{base}, \
                         size: {member_size}, offset: {})",
                        member.offset_bits
                    ),
                ));
                payload_member_ids.push(id);
            }

            let payload_elements = payload_member_ids
                .iter()
                .map(|id| format!("!{id}"))
                .collect::<Vec<_>>()
                .join(", ");
            let payload_elements_id = self.alloc_metadata_id();
            self.debug_nodes
                .push((payload_elements_id, format!("!{{{payload_elements}}}")));

            let variant_name = escape_debug_string(&variant.name);
            self.debug_nodes.push((
                variant_struct_id,
                format!(
                    "!DICompositeType(tag: DW_TAG_structure_type, name: \"{variant_name}\", \
                     scope: !{enum_type_id}, size: {size_bits}, elements: !{payload_elements_id})"
                ),
            ));

            let extra_data = variant
                .discriminant
                .map(|value| format!(", extraData: i{discriminant_width} {value}"))
                .unwrap_or_default();
            let variant_member_id = self.alloc_metadata_id();
            self.debug_nodes.push((
                variant_member_id,
                format!(
                    "!DIDerivedType(tag: DW_TAG_member, name: \"{variant_name}\", \
                     scope: !{variant_part_id}, baseType: !{variant_struct_id}, \
                     size: {size_bits}, offset: 0{extra_data})"
                ),
            ));
            variant_member_ids.push(variant_member_id);
        }

        let variant_elements = variant_member_ids
            .iter()
            .map(|id| format!("!{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        let variant_elements_id = self.alloc_metadata_id();
        self.debug_nodes
            .push((variant_elements_id, format!("!{{{variant_elements}}}")));

        let discriminator = discriminator_member_id
            .map(|id| format!(", discriminator: !{id}"))
            .unwrap_or_default();
        self.debug_nodes.push((
            variant_part_id,
            format!(
                "!DICompositeType(tag: DW_TAG_variant_part, scope: !{enum_type_id}, \
                 size: {size_bits}, elements: !{variant_elements_id}{discriminator})"
            ),
        ));

        let enum_elements_id = self.alloc_metadata_id();
        self.debug_nodes
            .push((enum_elements_id, format!("!{{!{variant_part_id}}}")));
        let name = escape_debug_string(name);
        self.debug_nodes.push((
            enum_type_id,
            format!(
                "!DICompositeType(tag: DW_TAG_structure_type, name: \"{name}\", \
                 size: {size_bits}, elements: !{enum_elements_id})"
            ),
        ));

        enum_type_id
    }

    fn debug_location_for_scope(&mut self, scope: usize, loc: &Location) -> Option<usize> {
        if !self.debug_kind.line_tables_enabled() {
            return None;
        }

        match loc {
            Location::CallSite { callee, caller } => {
                return self.debug_call_site_location_for_scope(scope, callee, caller);
            }
            Location::Named { child_loc, .. } => {
                return self.debug_location_for_scope(scope, child_loc);
            }
            Location::Fused { locations, .. } => {
                return locations
                    .iter()
                    .find_map(|loc| self.debug_location_for_scope(scope, loc));
            }
            Location::SrcPos { .. } | Location::Unknown => {}
        }

        let (path, pos) = self.source_position_from_location(loc)?;
        if let Some(resolved) = self.resolved_debug_scope_for_position(scope, &path, pos) {
            return self.debug_location_for_path_position(resolved, &path, pos);
        }

        let location_scope = self.debug_scope_for_file(scope, &path)?;

        self.ensure_debug_location(location_scope, pos, None)
    }

    fn debug_location_for_resolved_scope(
        &mut self,
        resolved: ResolvedDebugScope,
        loc: &Location,
    ) -> Option<usize> {
        if !self.debug_kind.line_tables_enabled() {
            return None;
        }

        let (path, pos) = self.source_position_from_location(loc)?;
        self.debug_location_for_path_position(resolved, &path, pos)
    }

    fn debug_location_for_path_position(
        &mut self,
        resolved: ResolvedDebugScope,
        path: &Path,
        pos: SourcePosition,
    ) -> Option<usize> {
        let location_scope = self.debug_scope_for_file(resolved.scope, path)?;

        self.ensure_debug_location(location_scope, pos, resolved.inlined_at)
    }

    fn resolved_debug_scope_for_position(
        &mut self,
        function_scope: usize,
        path: &Path,
        pos: SourcePosition,
    ) -> Option<ResolvedDebugScope> {
        let source_scope = {
            let map = self.debug_source_scope_maps.get(&function_scope)?;
            map.locations
                .iter()
                .filter(|location| {
                    location.pos.file.as_path() == path
                        && location.pos.line == pos.line
                        && location.pos.column == pos.column
                })
                .max_by_key(|location| source_scope_depth(map, location.scope))
                .map(|location| location.scope)
        }?;

        self.resolve_debug_source_scope(function_scope, source_scope)
    }

    fn ensure_debug_location_from_position(
        &mut self,
        resolved: ResolvedDebugScope,
        pos: &DebugSourcePosition,
    ) -> Option<usize> {
        let scope = self.debug_scope_for_file(resolved.scope, &pos.file)?;
        self.ensure_debug_location(
            scope,
            SourcePosition {
                line: pos.line,
                column: pos.column,
            },
            resolved.inlined_at,
        )
    }

    fn debug_call_site_location_for_scope(
        &mut self,
        scope: usize,
        callee: &Location,
        caller: &Location,
    ) -> Option<usize> {
        let caller_location = self
            .source_position_from_location(caller)
            .and_then(|(path, pos)| {
                let caller_scope = self.debug_scope_for_file(scope, &path)?;
                self.ensure_debug_location(caller_scope, pos, None)
            });

        let Some((callee_path, callee_pos)) = self.source_position_from_location(callee) else {
            return caller_location;
        };
        let callee_scope = self.debug_scope_for_file(scope, &callee_path)?;

        self.ensure_debug_location(callee_scope, callee_pos, caller_location)
    }

    fn debug_scope_for_file(&mut self, scope: usize, path: &Path) -> Option<usize> {
        let scope_path = self.debug_subprogram_files.get(&scope)?;
        if scope_path.as_path() == path {
            return Some(scope);
        }

        let key = (scope, path.to_path_buf());
        if let Some(id) = self.debug_file_scopes.get(&key).copied() {
            return Some(id);
        }

        let file_id = self.ensure_debug_file(path);
        let id = self.alloc_metadata_id();
        self.debug_nodes.push((
            id,
            format!("!DILexicalBlockFile(scope: !{scope}, file: !{file_id}, discriminator: 0)"),
        ));
        self.debug_file_scopes.insert(key, id);
        self.debug_subprogram_files.insert(id, path.to_path_buf());

        Some(id)
    }

    fn ensure_debug_location(
        &mut self,
        scope: usize,
        pos: SourcePosition,
        inlined_at: Option<usize>,
    ) -> Option<usize> {
        if pos.line <= 0 || pos.column <= 0 {
            return None;
        }

        let key = (scope, pos.line, pos.column, inlined_at);
        if let Some(id) = self.debug_locations.get(&key).copied() {
            return Some(id);
        }

        let id = self.alloc_metadata_id();
        let inlined_at = inlined_at
            .map(|location_id| format!(", inlinedAt: !{location_id}"))
            .unwrap_or_default();
        self.debug_nodes.push((
            id,
            format!(
                "!DILocation(line: {}, column: {}, scope: !{}{inlined_at})",
                pos.line, pos.column, scope
            ),
        ));
        self.debug_locations.insert(key, id);

        Some(id)
    }

    fn debug_fallback_location_for_scope(&mut self, scope: usize) -> Option<usize> {
        let (line, column) = self.debug_subprogram_fallbacks.get(&scope).copied()?;
        self.ensure_debug_location(scope, SourcePosition { line, column }, None)
    }

    fn ensure_artificial_debug_location(&mut self, scope: usize) -> usize {
        let key = (scope, 0, 0, None);
        if let Some(id) = self.debug_locations.get(&key).copied() {
            return id;
        }

        let id = self.alloc_metadata_id();
        self.debug_nodes.push((
            id,
            format!("!DILocation(line: 0, column: 0, scope: !{scope})"),
        ));
        self.debug_locations.insert(key, id);
        id
    }

    fn local_variable_position_from_location(
        &self,
        loc: &Location,
    ) -> Option<(PathBuf, SourcePosition)> {
        match loc {
            Location::CallSite { callee, caller } => self
                .source_position_from_location(callee)
                .or_else(|| self.source_position_from_location(caller)),
            Location::Named { child_loc, .. } => {
                self.local_variable_position_from_location(child_loc)
            }
            Location::Fused { locations, .. } => locations
                .iter()
                .find_map(|loc| self.local_variable_position_from_location(loc)),
            Location::SrcPos { .. } | Location::Unknown => self.source_position_from_location(loc),
        }
    }

    fn source_position_from_location(&self, loc: &Location) -> Option<(PathBuf, SourcePosition)> {
        match loc {
            Location::SrcPos {
                src: Source::File(path_key),
                pos,
            } if pos.line > 0 && pos.column > 0 => Some((
                uniqued_any::get(self.ctx, *path_key).clone(),
                SourcePosition {
                    line: pos.line,
                    column: pos.column,
                },
            )),
            Location::SrcPos { .. } | Location::Unknown => None,
            Location::Named { child_loc, .. } => self.source_position_from_location(child_loc),
            Location::Fused { locations, .. } => locations
                .iter()
                .find_map(|loc| self.source_position_from_location(loc)),
            Location::CallSite { caller, callee } => self
                .source_position_from_location(caller)
                .or_else(|| self.source_position_from_location(callee)),
        }
    }
}

fn split_file_and_directory(path: &Path) -> (String, String) {
    let filename = path
        .file_name()
        .filter(|name| !name.is_empty())
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned();

    let directory = path
        .parent()
        .map(|parent| {
            let dir = parent.to_string_lossy();
            if dir.is_empty() {
                ".".to_string()
            } else {
                dir.into_owned()
            }
        })
        .unwrap_or_else(|| ".".to_string());

    (filename, directory)
}

fn source_scope_depth(map: &crate::ops::DebugSourceScopeMap, scope: u32) -> usize {
    let mut depth = 0;
    let mut current = Some(scope);

    while let Some(scope_id) = current {
        let Some(data) = map.scopes.iter().find(|candidate| candidate.id == scope_id) else {
            break;
        };
        depth += 1;
        // Parents always precede their child in a well-formed rustc scope tree;
        // only follow strictly-smaller ids so a malformed cyclic map cannot
        // spin here forever.
        current = data.parent.filter(|&parent| parent < scope_id);
    }

    depth
}

fn escape_debug_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\5C"),
            '"' => out.push_str("\\22"),
            '\n' => out.push_str("\\0A"),
            '\r' => out.push_str("\\0D"),
            '\t' => out.push_str("\\09"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use combine::stream::position::SourcePosition;
    use pliron::{context::Context, location::Source};

    fn global_info() -> DebugGlobalVariableInfo {
        DebugGlobalVariableInfo {
            name: "COUNTER".to_string(),
            namespace: vec!["collision_crate".to_string(), "left".to_string()],
            ty: DebugLocalTypeKind::Basic {
                name: "u64".to_string(),
                size_bits: 64,
                encoding: "DW_ATE_unsigned",
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/collision.rs"),
                line: 7,
                column: 5,
            },
            is_local_to_unit: true,
            is_function_local: false,
        }
    }

    #[test]
    fn debug_strings_use_llvm_metadata_escapes() {
        assert_eq!(escape_debug_string("a\\b\"c\n\t"), "a\\5Cb\\22c\\0A\\09");
    }

    #[test]
    fn split_file_and_directory_handles_bare_and_nested_paths() {
        assert_eq!(
            split_file_and_directory(Path::new("kernel.rs")),
            ("kernel.rs".to_string(), ".".to_string())
        );
        assert_eq!(
            split_file_and_directory(Path::new("/tmp/cuda-oxide/kernel.rs")),
            ("kernel.rs".to_string(), "/tmp/cuda-oxide".to_string())
        );
    }

    #[test]
    fn source_position_from_location_unwraps_named_locations() {
        let mut ctx = Context::new();
        let loc = Location::Named {
            name: "lowered".to_string(),
            child_loc: Box::new(Location::SrcPos {
                src: Source::new_from_file(&mut ctx, PathBuf::from("/tmp/kernel.rs")),
                pos: SourcePosition {
                    line: 12,
                    column: 4,
                },
            }),
        };
        let state = ModuleExportState::new(
            &ctx,
            true,
            super::super::config::DebugKind::LineTables,
            None,
            super::super::config::FunctionLocalStaticPlacement::CompileUnitGlobals,
        );

        let (path, pos) = state
            .source_position_from_location(&loc)
            .expect("location should unwrap");

        assert_eq!(path, PathBuf::from("/tmp/kernel.rs"));
        assert_eq!(pos.line, 12);
        assert_eq!(pos.column, 4);
    }

    #[test]
    fn global_expressions_and_namespaces_are_uniqued_and_conflicts_rejected() {
        let ctx = Context::new();
        let mut state = ModuleExportState::new(
            &ctx,
            true,
            super::super::config::DebugKind::Full,
            None,
            super::super::config::FunctionLocalStaticPlacement::CompileUnitGlobals,
        );
        let info = global_info();

        let first = state
            .debug_global_variable("__device_global_0", Some(8), 1, None, &info)
            .expect("first identity is valid")
            .expect("full debug emits an expression");
        let repeated = state
            .debug_global_variable("__device_global_0", Some(8), 1, None, &info)
            .expect("identical repeated identity is valid")
            .expect("full debug emits an expression");
        assert_eq!(first, repeated);
        assert_eq!(state.debug_global_expressions, vec![first]);
        assert_eq!(state.debug_namespaces.len(), 2);

        let mut conflict = info;
        conflict.is_local_to_unit = false;
        let error = state
            .debug_global_variable("__device_global_0", Some(8), 1, None, &conflict)
            .expect_err("one linkage name cannot describe two source identities");
        assert!(error.contains("conflicting debug identities"), "{error}");

        let cu_id = state.debug_compile_unit.expect("compile unit reserved");
        state.finalize_debug_globals(cu_id);
        let node_count = state.debug_nodes.len();
        state.finalize_debug_globals(cu_id);
        assert_eq!(state.debug_nodes.len(), node_count);
    }
}
