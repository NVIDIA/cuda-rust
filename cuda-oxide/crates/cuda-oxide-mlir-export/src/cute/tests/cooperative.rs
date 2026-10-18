/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Independent address evaluation for cooperative-copy and matrix-load mappings.

use crate::cute::{
    copy::emit::integer,
    mma::ldmatrix::{matrix_address_plan, matrix_offset_fits_i32},
};

#[cfg(test)]
mod tests {
    //! Evaluate exported copy and matrix-load address arithmetic against independent layout expectations.

    use crate::render_module;
    use crate::{CutlassFullCuteMlir22, MlirConsumerProfile};
    use cute_layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
    use dialect_cute::{
        attributes::CuteMatrixRoleAttr,
        ops::{CuteCopyG2SOp, CuteLdmatrixOp},
    };
    use dialect_mir::{
        ops::{MirFuncOp, MirReturnOp},
        types::{MirFP16Type, MirPtrType, address_space},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::TypeAttr,
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::TypeHandle,
    };

    /// Construct the canonical swizzled shared tile used by cooperative-copy fixtures.
    fn shared_layout() -> ComposedLayout {
        ComposedLayout::new(
            Swizzle::new(3, 3, 3),
            0,
            "(64,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    /// Append a typed source operation to the fixture block.
    fn append<O: Op>(ctx: &Context, block: Ptr<BasicBlock>, op: O) {
        op.get_operation().insert_at_back(block, ctx);
    }

    /// Build the default copy or matrix-load fixture for the selected matrix role.
    fn module(ctx: &mut Context, copy: bool, role: CuteMatrixRoleAttr) -> ModuleOp {
        copy_module(ctx, copy, role, 64)
    }

    /// Build a copy or matrix-load fixture with the selected row extent and default layout.
    fn copy_module(
        ctx: &mut Context,
        copy: bool,
        role: CuteMatrixRoleAttr,
        rows: usize,
    ) -> ModuleOp {
        module_with_layout(ctx, copy, role, rows, shared_layout())
    }

    /// Build a copy or matrix-load fixture using the supplied shared layout.
    fn module_with_layout(
        ctx: &mut Context,
        copy: bool,
        role: CuteMatrixRoleAttr,
        rows: usize,
        layout: ComposedLayout,
    ) -> ModuleOp {
        module_with_copy_plan(ctx, copy, role, rows, layout, None)
    }

    /// Atom bytes, thread/value/tile layouts, and composed shared layout for one copy fixture.
    type CopyPlanFixture = (u32, Layout, Layout, Layout, ComposedLayout);

    /// Construct the typed test function for an optional explicit cooperative-copy plan.
    fn module_with_copy_plan(
        ctx: &mut Context,
        copy: bool,
        role: CuteMatrixRoleAttr,
        rows: usize,
        layout: ComposedLayout,
        copy_plan: Option<CopyPlanFixture>,
    ) -> ModuleOp {
        module_with_copy_storage(ctx, copy, role, rows, layout, copy_plan, None)
    }

    /// Select integer storage while keeping pointer pointees and copy element identity aligned.
    fn module_with_copy_storage(
        ctx: &mut Context,
        copy: bool,
        role: CuteMatrixRoleAttr,
        rows: usize,
        layout: ComposedLayout,
        copy_plan: Option<CopyPlanFixture>,
        integer_storage: Option<(u32, Signedness)>,
    ) -> ModuleOp {
        dialect_mir::register(ctx);
        dialect_cute::register(ctx);
        let module = ModuleOp::new(ctx, Identifier::try_from("attention_copy").unwrap());
        let element: TypeHandle = if let Some((width, signedness)) = integer_storage {
            IntegerType::get(ctx, width, signedness).into()
        } else {
            MirFP16Type::get(ctx).into()
        };
        let i32_: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let i64_: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let gmem: TypeHandle = MirPtrType::get(ctx, element, false, address_space::GENERIC).into();
        let smem: TypeHandle = MirPtrType::get(ctx, element, true, address_space::GENERIC).into();
        let slot: TypeHandle = MirPtrType::get(ctx, i32_, true, address_space::GENERIC).into();
        let types = if copy {
            vec![gmem, i64_, i64_, i64_, i64_, i64_, smem, i64_, i32_]
        } else {
            vec![smem, i64_, i64_, i32_, slot]
        };
        let ty = FunctionType::get(ctx, types.clone(), vec![]);
        let op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = MirFuncOp::new(ctx, op, TypeAttr::new(ty.into()));
        func.set_symbol_name(ctx, Identifier::try_from("attention_copy").unwrap());
        module.append_operation(ctx, op, 0);
        let block = BasicBlock::new(ctx, None, types);
        block.insert_at_back(op.deref(ctx).get_region(0), ctx);
        let args = block.deref(ctx).arguments().collect::<Vec<_>>();
        if copy {
            let (atom, thread, value, tile, shared) = copy_plan.unwrap_or_else(|| {
                (
                    16,
                    "(16,8):(8,1)".parse().unwrap(),
                    format!("({},8):(8,1)", rows / 16).parse().unwrap(),
                    format!("({rows},64):(64,1)").parse().unwrap(),
                    ComposedLayout::new(
                        Swizzle::new(3, 3, 3),
                        0,
                        format!("({rows},64):(64,1)").parse().unwrap(),
                        OffsetUnit::Elements,
                    )
                    .unwrap(),
                )
            });
            let copy = CuteCopyG2SOp::new(
                ctx,
                args.try_into().unwrap(),
                atom,
                thread,
                value,
                tile,
                shared,
                8,
                element,
            );
            append(ctx, block, copy);
        } else {
            let load = CuteLdmatrixOp::new(ctx, args.try_into().unwrap(), role, layout, element);
            append(ctx, block, load);
        }
        let ret = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        append(ctx, block, MirReturnOp::new(ret));
        module
    }

    /// Check the canonical copy emits four atoms with runtime source-byte counts for edge zero filling.
    #[test]
    fn attention_copy_repeats_four_atoms_with_dynamic_zero_fill() {
        let mut ctx = Context::new();
        let source = module(&mut ctx, true, CuteMatrixRoleAttr::A);
        let target = CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&ctx, &source)
            .unwrap();
        let text = render_module(&target).unwrap();
        assert_eq!(
            text.matches("\"nvvm.cp.async.shared.global\"").count(),
            4,
            "{text}"
        );
        assert!(text.contains("#nvvm<load_cache_modifier cg>"), "{text}");
        assert!(text.contains("arith.select"), "{text}");
        assert!(text.contains("arith.xori"), "{text}");
        assert!(!text.contains("cp.async.commit"), "{text}");
        assert!(!text.contains("cp.async.wait"), "{text}");
        assert!(!text.contains("cute.copy_g2s"), "{text}");
    }

    /// Check A/B matrix roles select the reviewed normal or transposed ldmatrix instruction.
    #[test]
    fn attention_matrix_roles_select_normal_and_transposed_loads() {
        for (role, number, layout) in [
            (CuteMatrixRoleAttr::A, 4, "row"),
            (CuteMatrixRoleAttr::B, 2, "col"),
            (CuteMatrixRoleAttr::BPair, 4, "col"),
            (CuteMatrixRoleAttr::BPairKMajor, 4, "row"),
        ] {
            let mut ctx = Context::new();
            let source = module(&mut ctx, false, role);
            let target = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &source)
                .unwrap();
            let text = render_module(&target).unwrap();
            assert!(text.contains(&format!("num = {number} : i32")), "{text}");
            assert!(
                text.contains(&format!("#nvvm.mma_layout<{layout}>")),
                "{text}"
            );
            assert_eq!(text.matches("\"llvm.store\"").count(), number, "{text}");
            assert!(!text.contains("cute.ldmatrix"), "{text}");
        }
    }

    /// Find the device function's block through the executable GPU envelope.
    fn function_block(op: &crate::MlirOperation) -> &crate::MlirBlock {
        if op.name == "func.func" {
            return &op.regions[0].blocks[0];
        }
        op.regions
            .iter()
            .flat_map(|r| &r.blocks)
            .flat_map(|b| &b.operations)
            .find_map(|op| {
                /// Recognize a nested target subtree containing a translated function.
                fn has_function(op: &crate::MlirOperation) -> bool {
                    op.name == "func.func"
                        || op
                            .regions
                            .iter()
                            .flat_map(|r| &r.blocks)
                            .flat_map(|b| &b.operations)
                            .any(has_function)
                }
                has_function(op).then(|| function_block(op))
            })
            .expect("translated function")
    }

    // Execute emitted scalar MLIR, observing the addresses and source byte
    // counts handed to hardware. This checks the actual lowering, not a second
    // copy of its address formula. No kernel or GPU runtime is involved.
    /// Interpret exported integer address arithmetic and collect each memory instruction's runtime arguments.
    fn memory_arguments(block: &crate::MlirBlock, args: &[u64]) -> Vec<Vec<u64>> {
        use crate::{MlirAttribute as A, MlirFloatType, MlirType as T};
        let mut values = block
            .arguments
            .iter()
            .zip(args)
            .map(|(arg, value)| (arg.id, *value))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut memory = vec![];
        for op in &block.operations {
            let a = op
                .operands
                .iter()
                .map(|v| values[&v.id])
                .collect::<Vec<_>>();
            let property = |name: &str| match op.properties[name] {
                A::Integer { value, .. } => value as u64,
                _ => panic!("integer property {name}"),
            };
            let result = match op.name.as_str() {
                "arith.constant" => property("value"),
                "arith.extui" | "llvm.addrspacecast" => a[0],
                "arith.trunci" => u64::from(a[0] as u32),
                "arith.addi" => a[0].wrapping_add(a[1]),
                "arith.subi" => a[0].wrapping_sub(a[1]),
                "arith.muli" => a[0].wrapping_mul(a[1]),
                "arith.divui" => a[0] / a[1],
                "arith.remui" => a[0] % a[1],
                "arith.minui" => a[0].min(a[1]),
                "arith.andi" => a[0] & a[1],
                "arith.xori" => a[0] ^ a[1],
                "arith.shrui" => a[0] >> a[1],
                "arith.shli" => a[0] << a[1],
                "arith.cmpi" => match property("predicate") {
                    6 => u64::from(a[0] < a[1]),
                    7 => u64::from(a[0] <= a[1]),
                    p => panic!("unexpected predicate {p}"),
                },
                "arith.select" => {
                    if a[0] != 0 {
                        a[1]
                    } else {
                        a[2]
                    }
                }
                "llvm.getelementptr" => {
                    let width = match op.properties["elem_type"] {
                        A::Type(T::Integer(8)) => 1,
                        A::Type(T::Integer(16)) => 2,
                        A::Type(T::Integer(32)) => 4,
                        A::Type(T::Float(MlirFloatType::F16)) => 2,
                        ref ty => panic!("unexpected GEP element {ty:?}"),
                    };
                    a[0].wrapping_add(a[1].wrapping_mul(width))
                }
                "nvvm.cp.async.shared.global" => {
                    memory.push(a);
                    continue;
                }
                "nvvm.ldmatrix" => {
                    memory.push(a);
                    break;
                }
                "func.return" => break,
                name => panic!("unexpected operation {name}"),
            };
            // Arithmetic has the result's bit width, including wraparound.
            // Pointers retain all 64 bits even when their offsets were i32.
            let result = match op.results[0].ty {
                T::Integer(width) if width < 64 => result & ((1u64 << width) - 1),
                _ => result,
            };
            values.insert(op.results[0].id, result);
        }
        memory
    }

    /// Integer copy carriers preserve D128 byte addresses and every edge atom's source-byte count.
    #[test]
    fn attention_copy_integer16_preserves_storage_and_zero_fills_d128_edges() {
        use crate::{MlirAttribute as A, MlirType as T};
        for signedness in [Signedness::Unsigned, Signedness::Signed] {
            for atom in [4, 8, 16] {
                let mut ctx = Context::new();
                let tile: Layout = "(8,128):(128,1)".parse().unwrap();
                let shared = ComposedLayout::new(
                    Swizzle::new(3, 3, 3),
                    0,
                    tile.clone(),
                    OffsetUnit::Elements,
                )
                .unwrap();
                let source = module_with_copy_storage(
                    &mut ctx,
                    true,
                    CuteMatrixRoleAttr::A,
                    8,
                    shared_layout(),
                    Some((
                        atom,
                        "(8,16):(16,1)".parse().unwrap(),
                        "(1,8):(8,1)".parse().unwrap(),
                        tile,
                        shared,
                    )),
                    Some((16, signedness)),
                );
                let target = CutlassFullCuteMlir22::new("sm_120a")
                    .unwrap()
                    .translate_module(&ctx, &source)
                    .unwrap();
                let block = function_block(&target.root);
                assert!(block.operations.iter().any(|op| {
                    op.name == "llvm.getelementptr"
                        && op.properties["elem_type"] == A::Type(T::Integer(16))
                }));
                for (rows, columns, tile_row, tile_column) in [
                    (64, 128, 0, 0),
                    (19, 125, 2, 0),
                    (9, 131, 1, 1),
                    (0, 0, 0, 0),
                    (64, 128, u64::MAX, u64::MAX),
                ] {
                    for thread in 0..128 {
                        let copies = memory_arguments(
                            block,
                            &[
                                0x10000,
                                rows,
                                columns,
                                256,
                                tile_row,
                                tile_column,
                                0x20000,
                                1024,
                                thread,
                            ],
                        );
                        let per_atom = u64::from(atom / 2);
                        assert_eq!(copies.len(), (8 / per_atom) as usize);
                        for (index, copy) in copies.iter().enumerate() {
                            let row = thread / 16;
                            let column = (thread % 16) * 8 + index as u64 * per_atom;
                            let bytes = (row * 128 + column) * 2;
                            let destination = 0x20000 + (bytes ^ ((bytes & 0x380) >> 3));
                            let global_row = u128::from(tile_row) * 8 + u128::from(row);
                            let global_col = u128::from(tile_column) * 128 + u128::from(column);
                            let valid =
                                global_row < u128::from(rows) && global_col < u128::from(columns);
                            let source = if valid {
                                0x10000 + (global_row as u64 * 256 + global_col as u64) * 2
                            } else {
                                0x10000
                            };
                            let valid_bytes = if valid {
                                (columns - global_col as u64).min(per_atom) * 2
                            } else {
                                0
                            };
                            assert_eq!(
                                *copy,
                                [destination, source, valid_bytes],
                                "signedness={signedness:?}, atom={atom}, thread={thread}, shape=({rows},{columns}), tile=({tile_row},{tile_column})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The physical two-byte address calculation must reject other integer widths.
    #[test]
    fn attention_copy_rejects_other_integer_storage_widths() {
        for width in [8, 32, 64] {
            let mut ctx = Context::new();
            let source = module_with_copy_storage(
                &mut ctx,
                true,
                CuteMatrixRoleAttr::A,
                16,
                shared_layout(),
                Some((
                    if width == 8 { 4 } else { 16 },
                    "(16,8):(8,1)".parse().unwrap(),
                    "(1,8):(8,1)".parse().unwrap(),
                    "(16,64):(64,1)".parse().unwrap(),
                    ComposedLayout::new(
                        Swizzle::new(3, 3, 3),
                        0,
                        "(16,64):(64,1)".parse().unwrap(),
                        OffsetUnit::Elements,
                    )
                    .unwrap(),
                )),
                Some((width, Signedness::Unsigned)),
            );
            let error = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &source)
                .unwrap_err();
            assert!(error.to_string().contains("16-bit"), "{error}");
        }
    }

    /// Compare emitted destinations and valid source-byte counts with the verified thread/value assignment.
    #[test]
    fn attention_copy_matches_thread_layout_and_zero_fills_edge_bytes() {
        let mut ctx = Context::new();
        let source = copy_module(&mut ctx, true, CuteMatrixRoleAttr::A, 16);
        let target = CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&ctx, &source)
            .unwrap();
        let block = function_block(&target.root);
        for (rows, columns, tile_row, tile_column) in [
            (64, 64, 0, 0),
            (19, 61, 1, 0),
            (19, 67, 1, 1),
            (0, 0, 0, 0),
            (19, 61, 2, 1),
            (64, 64, u64::MAX, u64::MAX),
        ] {
            for thread in 0..128 {
                let copies = memory_arguments(
                    block,
                    &[
                        0x10000,
                        rows,
                        columns,
                        128,
                        tile_row,
                        tile_column,
                        0x20000,
                        1024,
                        thread,
                    ],
                );
                assert_eq!(copies.len(), 1);
                let row = thread / 8;
                let column = thread % 8 * 8;
                let plain_bytes = (row * 64 + column) * 2;
                let expected_shared = 0x20000 + (plain_bytes ^ ((plain_bytes & 0x380) >> 3));
                let global_row = u128::from(tile_row) * 16 + u128::from(row);
                let global_column = u128::from(tile_column) * 64 + u128::from(column);
                let in_bounds =
                    global_row < u128::from(rows) && global_column < u128::from(columns);
                let expected_bytes = if in_bounds {
                    (columns - global_column as u64).min(8) * 2
                } else {
                    0
                };
                let expected_global = if in_bounds {
                    0x10000 + (global_row as u64 * 128 + global_column as u64) * 2
                } else {
                    0x10000
                };
                assert_eq!(
                    copies[0],
                    vec![expected_shared, expected_global, expected_bytes],
                    "thread={thread}, rows={rows}, columns={columns}, tile=({tile_row},{tile_column})"
                );
            }
        }
    }

    /// Check bounded thread IDs cover every copy atom and oversized tile coordinates still zero-fill safely.
    #[test]
    fn attention_copy_bounded_threads_preserve_all_atoms_and_u64_bounds() {
        // A mask instead of remainder corrupts the six-thread fixtures; a
        // warp-sized bound corrupts IDs32..1023. Each valid thread must keep
        // its three atoms, including partial source sizes and zero-fill.
        for (thread_rows, thread_columns, column_first) in [
            (1u64, 1u64, false),
            (6, 1, false),
            (8, 1, false),
            (16, 8, false),
            (1024, 1, false),
            (2, 3, true),
        ] {
            for atom_bytes in [4u32, 8, 16] {
                let values_per_atom = u64::from(atom_bytes) / 2;
                let values_per_thread = 3 * values_per_atom;
                let tile_columns = thread_columns * values_per_thread;
                let thread: Layout = if column_first {
                    format!("({thread_rows},{thread_columns}):(1,{thread_rows})")
                } else {
                    format!("({thread_rows},{thread_columns}):({thread_columns},1)")
                }
                .parse()
                .unwrap();
                let value: Layout = format!("(1,{values_per_thread}):(0,1)").parse().unwrap();
                let tile: Layout = format!("({thread_rows},{tile_columns}):({tile_columns},1)")
                    .parse()
                    .unwrap();
                let shared = ComposedLayout::new(
                    Swizzle::new(3, 3, 3),
                    8,
                    tile.clone(),
                    OffsetUnit::Elements,
                )
                .unwrap();
                let shared_bytes = shared.to_byte_offsets(2).unwrap();
                let mut ctx = Context::new();
                let source = module_with_copy_plan(
                    &mut ctx,
                    true,
                    CuteMatrixRoleAttr::A,
                    0,
                    shared_layout(),
                    Some((atom_bytes, thread, value, tile, shared)),
                );
                let target = CutlassFullCuteMlir22::new("sm_120a")
                    .unwrap()
                    .translate_module(&ctx, &source)
                    .unwrap();
                let block = function_block(&target.root);
                let cp_ops = block
                    .operations
                    .iter()
                    .filter(|op| op.name == "nvvm.cp.async.shared.global")
                    .collect::<Vec<_>>();
                assert_eq!(cp_ops.len(), 3);
                for op in cp_ops {
                    assert_eq!(
                        op.properties["size"],
                        super::integer(i128::from(atom_bytes), 32)
                    );
                }
                for (rows, columns, tile_row, tile_column) in [
                    (thread_rows, tile_columns, 0, 0),
                    (thread_rows - 1, tile_columns - 1, 0, 0),
                    (thread_rows + 1, tile_columns + values_per_atom - 1, 1, 1),
                    (0, 0, 0, 0),
                    (thread_rows, tile_columns, u64::MAX, u64::MAX),
                    (
                        (1u64 << 32) + thread_rows,
                        tile_columns,
                        (1u64 << 32) / thread_rows,
                        0,
                    ),
                    (
                        thread_rows,
                        (1u64 << 32) + tile_columns,
                        0,
                        (1u64 << 32) / tile_columns,
                    ),
                ] {
                    let leading_dim = columns.div_ceil(8) * 8;
                    for thread in 0..thread_rows * thread_columns {
                        // Independent inversion of the two fixture thread maps.
                        let (local_row, column_group) = if column_first {
                            (thread % thread_rows, thread / thread_rows)
                        } else {
                            (thread / thread_columns, thread % thread_columns)
                        };
                        let copies = memory_arguments(
                            block,
                            &[
                                0x1234_0001_0000,
                                rows,
                                columns,
                                leading_dim,
                                tile_row,
                                tile_column,
                                0x5678_0002_0000,
                                1 << 20,
                                thread,
                            ],
                        );
                        assert_eq!(copies.len(), 3);
                        for (atom, copy) in copies.iter().enumerate() {
                            let local_column =
                                column_group * values_per_thread + atom as u64 * values_per_atom;
                            let destination = shared_bytes
                                .checked_call(&cute_layout::IntTuple::Leaf(
                                    (local_row + thread_rows * local_column) as i64,
                                ))
                                .unwrap() as u64;
                            let global_row = u128::from(tile_row) * u128::from(thread_rows)
                                + u128::from(local_row);
                            let global_column = u128::from(tile_column) * u128::from(tile_columns)
                                + u128::from(local_column);
                            let valid = global_row < u128::from(rows)
                                && global_column < u128::from(columns);
                            let valid_bytes = if valid {
                                (columns - global_column as u64).min(values_per_atom) * 2
                            } else {
                                0
                            };
                            let source_offset = if valid {
                                (global_row as u64 * leading_dim + global_column as u64) * 2
                            } else {
                                0
                            };
                            assert_eq!(
                                copy,
                                &[
                                    0x5678_0002_0000 + destination,
                                    0x1234_0001_0000 + source_offset,
                                    valid_bytes
                                ],
                                "threads=({thread_rows},{thread_columns}), column_first={column_first}, atom_bytes={atom_bytes}, thread={thread}, atom={atom}, shape=({rows},{columns}), tile=({tile_row},{tile_column})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Compare each emitted lane address with the independent byte-domain shared-layout swizzle.
    #[test]
    fn attention_matrix_lane_addresses_follow_byte_swizzle() {
        for role in [
            CuteMatrixRoleAttr::A,
            CuteMatrixRoleAttr::B,
            CuteMatrixRoleAttr::BPair,
        ] {
            let mut ctx = Context::new();
            let source = module(&mut ctx, false, role);
            let target = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &source)
                .unwrap();
            let block = function_block(&target.root);
            for lane in 0..32 {
                let addresses = memory_arguments(block, &[0x20000, 1, 1, lane, 0x30000]);
                let row = 16 + lane % 16;
                let column = if role != CuteMatrixRoleAttr::B {
                    16 + lane / 16 * 8
                } else {
                    8
                };
                let plain_bytes = (row * 64 + column) * 2;
                assert_eq!(
                    addresses[0],
                    [0x20000 + (plain_bytes ^ ((plain_bytes & 0x380) >> 3))]
                );
                assert_eq!(addresses[0][0] % 16, 0);
            }
        }
    }

    /// Check a proved shared-address range permits narrow arithmetic without changing byte offsets.
    #[test]
    fn attention_matrix_uses_i32_for_proven_shared_addresses() {
        let mut ctx = Context::new();
        let source = module(&mut ctx, false, CuteMatrixRoleAttr::A);
        let target = CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&ctx, &source)
            .unwrap();
        let block = function_block(&target.root);
        for op in block
            .operations
            .iter()
            .take_while(|op| op.name != "nvvm.ldmatrix")
            .filter(|op| {
                matches!(
                    op.name.as_str(),
                    "arith.addi"
                        | "arith.muli"
                        | "arith.divui"
                        | "arith.remui"
                        | "arith.andi"
                        | "arith.xori"
                )
            })
        {
            assert_eq!(
                op.results[0].ty,
                crate::MlirType::Integer(32),
                "{}",
                op.name
            );
        }
        let source_gep = block
            .operations
            .iter()
            .find(|op| op.name == "llvm.getelementptr")
            .unwrap();
        assert_eq!(source_gep.operands[1].ty, crate::MlirType::Integer(64));
    }

    /// Check proved carry-free window/lane splitting preserves the full matrix-load address calculation.
    #[test]
    fn attention_matrix_factors_window_and_lane_pointer_offsets() {
        for role in [
            CuteMatrixRoleAttr::A,
            CuteMatrixRoleAttr::B,
            CuteMatrixRoleAttr::BPair,
            CuteMatrixRoleAttr::BPairKMajor,
        ] {
            let mut ctx = Context::new();
            let source = module(&mut ctx, false, role);
            let target = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &source)
                .unwrap();
            let block = function_block(&target.root);
            let geps = block
                .operations
                .iter()
                .take_while(|op| op.name != "nvvm.ldmatrix")
                .filter(|op| op.name == "llvm.getelementptr")
                .collect::<Vec<_>>();
            assert_eq!(geps.len(), 2, "keep the uniform high offset additive");
            assert_eq!(geps[1].operands[0].id, geps[0].results[0].id);
            let mut lane_dependent = std::collections::BTreeSet::from([block.arguments[3].id]);
            for op in block
                .operations
                .iter()
                .take_while(|op| op.name != "nvvm.ldmatrix")
            {
                if op.operands.iter().any(|v| lane_dependent.contains(&v.id)) {
                    lane_dependent.extend(op.results.iter().map(|v| v.id));
                }
            }
            assert!(!lane_dependent.contains(&geps[0].operands[1].id));
            assert!(lane_dependent.contains(&geps[1].operands[1].id));
        }
    }

    /// Reject narrow-address proofs whose intermediate terms overflow despite cancellation in the final sum.
    #[test]
    fn attention_matrix_i32_proof_checks_each_bound_before_narrowing() {
        let max = i64::from(i32::MAX);
        for (inner, offset, expected) in [
            ("(16,16):(32,2)".to_owned(), max - 510, true),
            ("(16,16):(32,2)".to_owned(), max - 509, false),
            ("(16,16):(32,2)".to_owned(), -1, false),
            ("(16,16):(-32,2)".to_owned(), 480, false),
            (format!("(1,1):({max},0)"), 0, true),
            (format!("(1,1):({},0)", max + 1), 0, false),
            (format!("({max},1):(0,0)"), 0, true),
            (format!("({},1):(0,0)", max + 1), 0, false),
            ("(65536,65536):(0,0)".to_owned(), 0, false),
            (format!("({},2):(0,0)", i64::MAX), 0, false),
            ("(0,16):(32,2)".to_owned(), 0, false),
            ("(-16,16):(32,2)".to_owned(), 0, false),
            ("16:2".to_owned(), 0, false),
            ("((4,4),(8,2)):((32,128),(2,16))".to_owned(), 16, true),
        ] {
            let layout = ComposedLayout::new(
                Swizzle::IDENTITY,
                offset,
                inner.parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap();
            assert_eq!(
                super::matrix_offset_fits_i32(&layout),
                expected,
                "{layout:?}"
            );
        }
        for (swizzle, expected) in [
            (Swizzle::new(1, 29, 1), true),
            (Swizzle::new(1, 29, -1), true),
            (Swizzle::new(1, 30, 1), false),
            (Swizzle::new(1, 30, -1), false),
            (Swizzle::new(0, 32, 0), true),
        ] {
            let layout = ComposedLayout::new(
                swizzle,
                0,
                "(16,16):(32,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap();
            assert_eq!(
                super::matrix_offset_fits_i32(&layout),
                expected,
                "{layout:?}"
            );
        }
        assert!(!super::matrix_offset_fits_i32(&shared_layout()));
    }

    /// Enumerate legal matrix windows and lanes and compare emitted addresses with checked layout evaluation.
    #[test]
    fn attention_matrix_addresses_match_checked_layout_for_every_window_and_lane() {
        use cute_layout::IntTuple;
        let layouts = [
            shared_layout(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                16,
                "(32,32):(32,1)".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                -16,
                "(32,32):(32,1)".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                1024,
                "(32,32):(-32,1)".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(2, 3, -2),
                32,
                "((4,8),(8,4)):((32,128),(1,8))".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                64,
                "((4,8),(8,4)):((32,128),(1,-8))".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                1i64 << 32,
                "(32,32):(64,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(1, 30, 1),
                0,
                "(32,32):(64,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(1, 30, -1),
                0,
                "(32,32):(64,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                8192,
                "((8,8),64):((128,1024),2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                8192,
                "((8,8),64):((1024,128),2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, -3),
                0,
                "(64,64):(128,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                16,
                "(64,64):(128,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                0,
                "((3,16),64):((128,512),2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                i64::from(i32::MAX) - 8191,
                "(64,64):(128,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
        ];
        for (layout_index, layout) in layouts.into_iter().enumerate() {
            let bytes = layout.to_byte_offsets(2).unwrap();
            let modes = layout.inner().modes();
            let rows = modes[0].checked_size().unwrap() as u64;
            let columns = modes[1].checked_size().unwrap() as u64;
            for role in [
                CuteMatrixRoleAttr::A,
                CuteMatrixRoleAttr::B,
                CuteMatrixRoleAttr::BPair,
                CuteMatrixRoleAttr::BPairKMajor,
            ] {
                let mut ctx = Context::new();
                let source = module_with_layout(&mut ctx, false, role, 0, layout.clone());
                let target = CutlassFullCuteMlir22::new("sm_120a")
                    .unwrap()
                    .translate_module(&ctx, &source)
                    .unwrap();
                let block = function_block(&target.root);
                let source_geps = block
                    .operations
                    .iter()
                    .take_while(|op| op.name != "nvvm.ldmatrix")
                    .filter(|op| op.name == "llvm.getelementptr")
                    .count();
                let expected_geps = if [0, 9, 10, 11, 14].contains(&layout_index) {
                    2
                } else {
                    1
                };
                assert_eq!(
                    source_geps, expected_geps,
                    "factoring selection: {layout:?}, {role:?}"
                );
                let expected_width = if [0, 1, 4].contains(&layout_index) || layout_index >= 9 {
                    32
                } else {
                    64
                };
                for op in block
                    .operations
                    .iter()
                    .take_while(|op| op.name != "nvvm.ldmatrix")
                    .filter(|op| {
                        matches!(
                            op.name.as_str(),
                            "arith.addi"
                                | "arith.muli"
                                | "arith.divui"
                                | "arith.remui"
                                | "arith.andi"
                                | "arith.xori"
                        )
                    })
                {
                    assert_eq!(
                        op.results[0].ty,
                        crate::MlirType::Integer(expected_width),
                        "layout={layout:?}, op={}",
                        op.name
                    );
                }
                let tile_columns = if role == CuteMatrixRoleAttr::B { 8 } else { 16 };
                // Hardware matrix order, independent of the emitter's arithmetic.
                let quadrants = if role == CuteMatrixRoleAttr::BPairKMajor {
                    [(0, 0), (0, 8), (8, 0), (8, 8)]
                } else if role == CuteMatrixRoleAttr::B {
                    [(0, 0), (8, 0), (0, 0), (8, 0)]
                } else {
                    [(0, 0), (8, 0), (0, 8), (8, 8)]
                };
                for window_row in 0..rows / 16 {
                    for window_column in 0..columns / tile_columns {
                        for lane in 0..32 {
                            let (row_half, column_half) = quadrants[lane as usize / 8];
                            let row = window_row * 16 + row_half + lane % 8;
                            let column = window_column * tile_columns + column_half;
                            let offset = bytes
                                .checked_call(&IntTuple::Leaf((row + rows * column) as i64))
                                .unwrap();
                            let base = 0x1234_0002_0000u64;
                            let actual = memory_arguments(
                                block,
                                &[base, window_row, window_column, lane, 0x30000],
                            );
                            assert_eq!(
                                actual,
                                [vec![base.wrapping_add(offset as u64)]],
                                "layout={layout:?}, role={role:?}, window=({window_row},{window_column}), lane={lane}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Reject factoring when proof work exceeds its cap or the selected matrix window is incomplete.
    #[test]
    fn attention_matrix_factoring_caps_proof_and_requires_a_complete_window() {
        for inner in ["(1024,2048):(4096,2)", "(8,64):(128,2)", "(16,8):(16,2)"] {
            let layout = ComposedLayout::new(
                Swizzle::IDENTITY,
                0,
                inner.parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap();
            assert!(super::matrix_offset_fits_i32(&layout));
            assert!(
                super::matrix_address_plan(&layout, CuteMatrixRoleAttr::A).is_none(),
                "{inner}"
            );
        }
    }

    /// Check K-major B loads arrange paired registers in the order expected by the following MMA.
    #[test]
    fn attention_k_major_b_pair_loads_contiguous_mma_register_pairs() {
        let mut ctx = Context::new();
        let source = module(&mut ctx, false, CuteMatrixRoleAttr::BPairKMajor);
        let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
        let target = profile.translate_module(&ctx, &source).unwrap();
        let new_block = function_block(&target.root);
        let old_source = module(&mut ctx, false, CuteMatrixRoleAttr::A);
        let old_target = profile.translate_module(&ctx, &old_source).unwrap();
        let old_block = function_block(&old_target.root);

        // ldmatrix.x4 gets one row address from each lane: groups of eight
        // lanes supply the four 8x8 submatrices. For K-major B, adjacent
        // registers must cover K=0..7 and K=8..15 at the same N coordinate.
        let quadrants = [(0, 0), (0, 8), (8, 0), (8, 8)];
        for window_row in 0..4 {
            for window_column in 0..4 {
                for (matrix, (row_half, column_half)) in quadrants.into_iter().enumerate() {
                    for matrix_row in 0..8 {
                        let lane = matrix as u64 * 8 + matrix_row;
                        let args = [0x20000, window_row, window_column, lane, 0x30000];
                        let addresses = memory_arguments(new_block, &args);
                        assert_eq!(addresses.len(), 1);
                        let row = window_row * 16 + row_half + matrix_row;
                        let column = window_column * 16 + column_half;
                        let bytes = (row * 64 + column) * 2;
                        let expected = 0x20000 + (bytes ^ ((bytes & 0x380) >> 3));
                        assert_eq!(
                            addresses[0],
                            [expected],
                            "window=({window_row},{window_column}), lane={lane}"
                        );
                        assert_eq!(expected % 16, 0);

                        // Independent compatibility check: the hardware's new
                        // register order equals the existing A-to-B permutation.
                        let old_matrix = [0, 2, 1, 3][matrix];
                        let mut old_args = args;
                        old_args[3] = old_matrix * 8 + matrix_row;
                        assert_eq!(addresses, memory_arguments(old_block, &old_args));
                    }
                }
            }
        }
    }
}
