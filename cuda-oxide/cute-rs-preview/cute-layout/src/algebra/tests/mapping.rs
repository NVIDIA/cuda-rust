/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Contract tests for the mapping algebra family.

use super::*;

/// Coalescing must preserve the flat offset sequence across singleton, nested and strided modes.
#[test]
fn coalesce_examples() {
    assert_eq!(coalesce(&l("(2,4):(1,2)")).to_string(), "8:1");
    assert_eq!(coalesce(&l("(2,1,4):(1,7,2)")).to_string(), "8:1");
    assert_eq!(coalesce(&l("(2,4):(4,1)")).to_string(), "(2,4):(4,1)");
    assert_eq!(coalesce(&l("1:0")).to_string(), "1:0");
}

/// Flattening must keep every flat coordinate's offset and expose one mode per
/// integer of the original shape.
#[test]
fn flatten_is_definitional() {
    let nested = [l("((2,3),4):((12,4),1)"), l("(2,(3,4)):(1,(2,6))")];
    for a in corpus().into_iter().chain(nested) {
        let flat = flatten(&a);
        assert_eq!(flat.rank(), a.shape.flatten().len(), "{a}");
        assert_eq!(flat.shape.flatten(), a.shape.flatten(), "{a}");
        assert_eq!(flat.stride.flatten(), a.stride.flatten(), "{a}");
        for coord in 0..a.size() {
            let coord = IntTuple::Leaf(coord);
            assert_eq!(flat.call(&coord), a.call(&coord), "{a} at {coord}");
        }
    }
    assert_eq!(
        flatten(&l("((2,3),4):((12,4),1)")).to_string(),
        "(2,3,4):(12,4,1)"
    );
    assert_eq!(flatten(&l("(4):(1)")).to_string(), "4:1");
}

/// Checks the definition of composition at every coordinate:
/// `result(i) == a(b(i))`.
///
/// This test uses a one-mode `b`. Its stride must line up with a boundary
/// in flattened `a`. For shape modes `[s0, s1, ...]`, those boundaries are
/// `[1, s0, s0*s1, ...]`: the number of coordinates covered before the
/// next mode changes. The end of `b` must also divide one of those
/// boundaries exactly.
///
/// Other tests cover multi-mode composition through divide, product, and
/// TV construction. Imperfect final tiles are deliberately excluded here
/// because composition may describe their overhang; predication, rather
/// than the pointwise identity, makes those tiles safe.
#[test]
fn composition_is_definitional() {
    let mut checked = 0;
    for a in corpus() {
        let prefix: Vec<i64> = {
            let mut p = vec![1i64];
            for (s, _) in super::flat_modes(&super::coalesce(&a)) {
                p.push(p.last().unwrap() * s);
            }
            p
        };
        let total = a.size();
        for n in [1i64, 2, 3, 4, 6, 8] {
            for d in [0i64, 1, 2, 3, 4, 6, 8] {
                let b = Layout {
                    shape: IntTuple::Leaf(n),
                    stride: IntTuple::Leaf(d),
                };
                if b.cosize() > total {
                    continue;
                }
                let admissible = d == 0
                    || (prefix.contains(&d)
                        && prefix.iter().any(|&p| p >= n * d && p % (n * d) == 0));
                if !admissible {
                    continue;
                }
                let Ok(r) = composition(&a, &b) else {
                    continue; // divisibility precondition not met: fine
                };
                assert_eq!(r.size(), b.size(), "size mismatch for {a} o {b}");
                for i in 0..b.size() {
                    let expect = a.call(&IntTuple::Leaf(b.call(&IntTuple::Leaf(i))));
                    let got = r.call(&IntTuple::Leaf(i));
                    assert_eq!(got, expect, "({a}) o ({b}) at {i}");
                }
                checked += 1;
            }
        }
    }
    assert!(
        checked > 60,
        "corpus too weak: only {checked} pairs checked"
    );
}

/// Composed layouts must match known offset maps and reject unsupported mode walks.
#[test]
fn composition_known_examples() {
    // A documented nested example: select a column-like walk through a
    // row-major matrix, then preserve the resulting submode structure.
    assert_eq!(
        composition(&l("(6,2):(8,2)"), &l("(4,3):(3,1)"))
            .unwrap()
            .to_string(),
        "((2,2),3):((24,2),8)"
    );
    // Every top-level mode on the right remains a top-level result mode.
    let r = composition(&l("(8,8):(8,1)"), &l("(4,2):(2,1)")).unwrap();
    for i in 0..8 {
        assert_eq!(
            r.call(&IntTuple::Leaf(i)),
            l("(8,8):(8,1)").call(&IntTuple::Leaf(l("(4,2):(2,1)").call(&IntTuple::Leaf(i))))
        );
    }
}

/// Joining a layout and its complement must reach each offset in `[0, M)`
/// exactly once.
#[test]
fn complement_is_definitional() {
    let cases = [
        ("4:1", 24),
        ("4:2", 24),
        ("6:4", 24),
        ("(2,3):(3,1)", 12),
        ("(2,2):(1,8)", 16),
        ("(4,4):(4,1)", 32),
    ];
    for (text, m) in cases {
        let a = l(text);
        let c = complement(&a, m).unwrap();
        let combined = Layout::from_modes(vec![a.clone(), c.clone()]);
        let total = combined.size();
        assert_eq!(total, m, "complement of {text} in {m} has wrong total size");
        let mut seen: Vec<i64> = (0..total)
            .map(|i| combined.call(&IntTuple::Leaf(i)))
            .collect();
        seen.sort_unstable();
        let expect: Vec<i64> = (0..m).collect();
        assert_eq!(
            seen, expect,
            "complement of {text} in {m} is not a bijection"
        );
    }
    // Even offsets plus a parity coordinate cover all 24 offsets.
    assert_eq!(
        complement(&l("4:2"), 24).unwrap().to_string(),
        "(2,3):(1,8)"
    );
}

/// A right inverse must turn every reachable consecutive offset back into
/// a coordinate that produces that offset.
#[test]
fn right_inverse_is_definitional() {
    for a in corpus() {
        let r = right_inverse(&a).unwrap();
        for i in 0..r.size() {
            assert_eq!(
                a.call(&IntTuple::Leaf(r.call(&IntTuple::Leaf(i)))),
                i,
                "right_inverse of {a} fails at {i}"
            );
        }
        // A compact one-to-one layout has no gap, so its inverse covers
        // the full logical cell count.
        if a.cosize() == a.size() {
            assert_eq!(r.size(), a.size(), "right_inverse of compact {a} not full");
        }
    }
}

/// A left inverse must recover every original coordinate for a layout with
/// no duplicate offsets.
#[test]
fn left_inverse_is_definitional() {
    for a in corpus() {
        // Keep only layouts where every coordinate reaches a distinct
        // offset; duplicate offsets cannot be undone.
        let mut image: Vec<i64> = (0..a.size()).map(|i| a.call(&IntTuple::Leaf(i))).collect();
        image.sort_unstable();
        image.dedup();
        if image.len() as i64 != a.size() {
            continue;
        }
        let r = left_inverse(&a).unwrap();
        for i in 0..a.size() {
            assert_eq!(
                r.call(&IntTuple::Leaf(a.call(&IntTuple::Leaf(i)))),
                i,
                "left_inverse of {a} fails at {i}"
            );
        }
    }
}
