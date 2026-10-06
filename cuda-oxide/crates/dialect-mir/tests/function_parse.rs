/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use combine::Parser;
use pliron::{
    context::Context,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    printable::Printable,
};

#[test]
fn parsed_function_has_one_body_and_survives_printing() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let source = r"mir.func @identity: builtin.function <(builtin.integer ui64) -> (builtin.integer ui64)> [] {
        ^entry(value: builtin.integer ui64):
            mir.return (value) [] []: <(builtin.integer ui64) -> ()>
    }";
    let function = parse_from_str(
        spaced(Operation::top_level_parser()).skip(combine::eof()),
        &mut ctx,
        source,
    )
    .unwrap();
    assert_eq!(function.deref(&ctx).num_regions(), 1);
    let body = function.deref(&ctx).get_region(0);
    assert_eq!(body.deref(&ctx).iter(&ctx).count(), 1);
    verify_operation(function, &ctx).unwrap();

    let printed = function.deref(&ctx).disp(&ctx).to_string();
    let mut fresh = Context::new();
    dialect_mir::register(&mut fresh);
    let reparsed = parse_from_str(
        spaced(Operation::top_level_parser()).skip(combine::eof()),
        &mut fresh,
        &printed,
    )
    .unwrap();
    assert_eq!(reparsed.deref(&fresh).num_regions(), 1);
    verify_operation(reparsed, &fresh).unwrap();
}

#[test]
fn declarations_have_one_empty_region_with_or_without_braces() {
    for suffix in ["", " []", " {}", " [] {}"] {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let source = format!(
            "mir.func @declaration: builtin.function <(builtin.integer ui32) -> (builtin.integer ui32)>{suffix}"
        );
        let function = parse_from_str(
            spaced(Operation::top_level_parser()).skip(combine::eof()),
            &mut ctx,
            &source,
        )
        .unwrap();
        assert_eq!(function.deref(&ctx).num_regions(), 1);
        let region = function.deref(&ctx).get_region(0);
        assert_eq!(region.deref(&ctx).iter(&ctx).count(), 0);
        verify_operation(function, &ctx).unwrap();
    }
}
