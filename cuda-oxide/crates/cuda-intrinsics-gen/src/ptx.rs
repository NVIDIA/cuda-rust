/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use ptx_parse::{Document, ParseError, split_top_level};
use serde::{Deserialize, Serialize};
use std::fmt;

/// The shape of one operand in a PTX instruction.
///
/// Register operands accept both LLVM TableGen placeholders such as `$dst`
/// and registers emitted by LLVM such as `%r12`. Exact operands are useful for
/// literals and special registers, whose spelling is part of the instruction
/// contract. Register-or-immediate operands also accept integer literals.
/// Register-predicate pairs model PTX destinations such as `d|p`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperandPattern {
    Register,
    Immediate,
    RegisterOrImmediate,
    RegisterPredicatePair,
    Exact { value: String },
    RegisterList { length: usize },
    Address,
}

impl<'de> Deserialize<'de> for OperandPattern {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Kind {
            Register,
            Immediate,
            RegisterOrImmediate,
            RegisterPredicatePair,
            Exact,
            RegisterList,
            Address,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Representation {
            kind: Kind,
            value: Option<String>,
            length: Option<usize>,
        }

        let representation = Representation::deserialize(deserializer)?;
        match (
            representation.kind,
            representation.value,
            representation.length,
        ) {
            (Kind::Register, None, None) => Ok(Self::Register),
            (Kind::Immediate, None, None) => Ok(Self::Immediate),
            (Kind::RegisterOrImmediate, None, None) => Ok(Self::RegisterOrImmediate),
            (Kind::RegisterPredicatePair, None, None) => Ok(Self::RegisterPredicatePair),
            (Kind::Exact, Some(value), None) => Ok(Self::Exact { value }),
            (Kind::RegisterList, None, Some(length)) => Ok(Self::RegisterList { length }),
            (Kind::Address, None, None) => Ok(Self::Address),
            (Kind::Register, _, _) => Err(serde::de::Error::custom(
                "register operand accepts only the `kind` field",
            )),
            (Kind::Immediate, _, _) => Err(serde::de::Error::custom(
                "immediate operand accepts only the `kind` field",
            )),
            (Kind::RegisterOrImmediate, _, _) => Err(serde::de::Error::custom(
                "register_or_immediate operand accepts only the `kind` field",
            )),
            (Kind::RegisterPredicatePair, _, _) => Err(serde::de::Error::custom(
                "register_predicate_pair operand accepts only the `kind` field",
            )),
            (Kind::Exact, _, _) => Err(serde::de::Error::custom(
                "exact operand requires only a `value` field",
            )),
            (Kind::RegisterList, _, _) => Err(serde::de::Error::custom(
                "register_list operand requires only a `length` field",
            )),
            (Kind::Address, _, _) => Err(serde::de::Error::custom(
                "address operand accepts only the `kind` field",
            )),
        }
    }
}

/// A PTX instruction shape with an exact mnemonic and ordered modifier list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionPattern {
    pub mnemonic: String,
    pub modifiers: Vec<String>,
    pub operands: Vec<OperandPattern>,
}

impl InstructionPattern {
    #[cfg(test)]
    pub fn new(
        mnemonic: impl Into<String>,
        modifiers: &[&str],
        operands: Vec<OperandPattern>,
    ) -> Self {
        Self {
            mnemonic: mnemonic.into(),
            modifiers: modifiers
                .iter()
                .map(|modifier| (*modifier).into())
                .collect(),
            operands,
        }
    }

    /// Reject malformed policy before matching imported or emitted PTX.
    pub fn validate(&self) -> Result<(), String> {
        if !is_head_component(&self.mnemonic) {
            return Err(format!("invalid mnemonic {:?}", self.mnemonic));
        }
        for modifier in &self.modifiers {
            if !is_head_component(modifier) {
                return Err(format!("invalid modifier {modifier:?}"));
            }
        }
        for operand in &self.operands {
            match operand {
                OperandPattern::Register
                | OperandPattern::Immediate
                | OperandPattern::RegisterOrImmediate
                | OperandPattern::RegisterPredicatePair
                | OperandPattern::Address => {}
                OperandPattern::Exact { value } => {
                    if value.is_empty() || value.trim() != value {
                        return Err(format!("invalid exact operand {value:?}"));
                    }
                }
                OperandPattern::RegisterList { length } => {
                    if *length == 0 {
                        return Err("register-list operand length must be positive".into());
                    }
                }
            }
        }
        Ok(())
    }

    /// Return true when `source` contains an instruction with exactly this
    /// shape. Comments and quoted directive strings are not searched.
    ///
    /// Fails when `source` is not lexically well-formed PTX, so a broken
    /// artifact reads as an error rather than as "no match".
    pub fn matches(&self, source: &str) -> Result<bool, ParseError> {
        contains_matching_instruction(source, self)
    }
}

/// One instruction that matched a reviewed PTX shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InstructionMatch {
    /// Byte offset in the masked PTX source.
    pub offset: usize,
    /// Byte offset immediately after the instruction semicolon.
    pub end: usize,
    /// Non-whitespace text before the instruction in the same statement.
    pub prefix: String,
    /// Trimmed operands in source order.
    pub operands: Vec<String>,
}

impl fmt::Display for InstructionPattern {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.mnemonic)?;
        for modifier in &self.modifiers {
            write!(formatter, ".{modifier}")?;
        }
        if !self.operands.is_empty() {
            formatter.write_str(" ")?;
        }
        for (index, operand) in self.operands.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            match operand {
                OperandPattern::Register => formatter.write_str("<register>")?,
                OperandPattern::Immediate => formatter.write_str("<immediate>")?,
                OperandPattern::RegisterOrImmediate => {
                    formatter.write_str("<register-or-immediate>")?
                }
                OperandPattern::RegisterPredicatePair => {
                    formatter.write_str("<register|predicate>")?
                }
                OperandPattern::Exact { value } => formatter.write_str(value)?,
                OperandPattern::RegisterList { length } => {
                    write!(formatter, "<register-list:{length}>")?
                }
                OperandPattern::Address => formatter.write_str("[<address>]")?,
            }
        }
        formatter.write_str(";")
    }
}

/// Search emitted PTX or a TableGen assembly string for an exact instruction
/// shape.
pub fn contains_matching_instruction(
    source: &str,
    pattern: &InstructionPattern,
) -> Result<bool, ParseError> {
    Ok(!matching_instructions(source, pattern)?.is_empty())
}

/// Return every instruction matching `pattern` without treating comments or
/// quoted directive strings as PTX code.
pub(crate) fn matching_instructions(
    source: &str,
    pattern: &InstructionPattern,
) -> Result<Vec<InstructionMatch>, ParseError> {
    Ok(instructions_with_matching_head(source, pattern)?
        .into_iter()
        .filter(|instruction| {
            instruction.operands.len() == pattern.operands.len()
                && instruction
                    .operands
                    .iter()
                    .zip(&pattern.operands)
                    .all(|(operand, expected)| operand_matches(operand, expected))
        })
        .collect())
}

/// Return every instruction with the exact mnemonic and modifier sequence.
///
/// A source which does not lex as PTX is an error, not an empty match list:
/// callers gate generated intrinsics on these matches, so unparseable PTX
/// must fail loudly instead of reading as "no match".
pub(crate) fn instructions_with_matching_head(
    source: &str,
    pattern: &InstructionPattern,
) -> Result<Vec<InstructionMatch>, ParseError> {
    if pattern.mnemonic.is_empty() {
        return Ok(Vec::new());
    }
    let document = Document::parse(source)?;
    Ok(document
        .instructions()
        .iter()
        .filter(|instruction| instruction_head_matches(instruction.head(), pattern))
        .map(|instruction| InstructionMatch {
            offset: instruction.head_offset(),
            end: instruction.end_offset(),
            prefix: instruction.prefix().to_owned(),
            operands: instruction.operands().map(str::to_owned).collect(),
        })
        .collect())
}

fn instruction_head_matches(head: &str, pattern: &InstructionPattern) -> bool {
    let mut parts = head.split('.');
    parts.next() == Some(pattern.mnemonic.as_str())
        && parts.eq(pattern.modifiers.iter().map(String::as_str))
}

fn operand_matches(operand: &str, pattern: &OperandPattern) -> bool {
    match pattern {
        OperandPattern::Register => is_register(operand),
        OperandPattern::Immediate => is_integer_literal(operand),
        OperandPattern::RegisterOrImmediate => is_register(operand) || is_integer_literal(operand),
        OperandPattern::RegisterPredicatePair => is_register_predicate_pair(operand),
        OperandPattern::Exact { value } => operand.trim() == value,
        OperandPattern::RegisterList { length } => enclosed_body(operand, b'{', b'}')
            // TableGen assembly strings escape a literal register-list brace
            // pair as `{{...}}`; emitted PTX contains the usual `{...}`.
            .map(|body| enclosed_body(body, b'{', b'}').unwrap_or(body))
            .and_then(split_top_level)
            .is_some_and(|registers| {
                registers.len() == *length && registers.iter().all(|register| is_register(register))
            }),
        OperandPattern::Address => {
            enclosed_body(operand, b'[', b']').is_some_and(|address| !address.trim().is_empty())
        }
    }
}

fn is_head_component(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':'))
}

fn is_register(operand: &str) -> bool {
    let operand = operand.trim();
    if let Some(name) = operand.strip_prefix('$') {
        return is_identifier(name);
    }
    let Some(name) = operand.strip_prefix('%') else {
        return false;
    };
    let Some(first_digit) = name.find(|character: char| character.is_ascii_digit()) else {
        return false;
    };
    first_digit > 0
        && name[..first_digit]
            .bytes()
            .all(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && name[first_digit..]
            .bytes()
            .all(|byte| byte.is_ascii_digit())
}

fn is_register_predicate_pair(operand: &str) -> bool {
    let mut parts = operand.split('|');
    let Some(register) = parts.next() else {
        return false;
    };
    let Some(predicate) = parts.next() else {
        return false;
    };

    parts.next().is_none() && is_register(register) && is_predicate_register(predicate)
}

fn is_predicate_register(operand: &str) -> bool {
    let operand = operand.trim();
    if let Some(name) = operand.strip_prefix('$') {
        return name == "pred";
    }

    operand.strip_prefix("%p").is_some_and(|digits| {
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    })
}

fn is_integer_literal(operand: &str) -> bool {
    let operand = operand.trim();
    let digits = operand
        .strip_prefix('+')
        .or_else(|| operand.strip_prefix('-'))
        .unwrap_or(operand);

    if let Some(hex_digits) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        return !hex_digits.is_empty() && hex_digits.bytes().all(|byte| byte.is_ascii_hexdigit());
    }

    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn enclosed_body(source: &str, open: u8, close: u8) -> Option<&str> {
    let source = source.trim();
    if source.as_bytes().first() != Some(&open) {
        return None;
    }

    let mut delimiters = Vec::new();
    for (index, byte) in source.bytes().enumerate() {
        match byte {
            b'{' => delimiters.push(b'}'),
            b'[' => delimiters.push(b']'),
            b'(' => delimiters.push(b')'),
            b'}' | b']' | b')' => {
                if delimiters.pop() != Some(byte) {
                    return None;
                }
                if delimiters.is_empty() {
                    return (byte == close && index + 1 == source.len())
                        .then_some(&source[1..index]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const LDMATRIX_X4_MODIFIERS: &[&str] = &["sync", "aligned", "m8n8", "x4", "shared", "b16"];

    fn ldmatrix_x4() -> InstructionPattern {
        InstructionPattern::new(
            "ldmatrix",
            LDMATRIX_X4_MODIFIERS,
            vec![
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::Address,
            ],
        )
    }

    #[test]
    fn matches_emitted_and_tablegen_registers() {
        assert!(
            ldmatrix_x4()
                .matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];")
                .unwrap()
        );
        assert!(ldmatrix_x4().matches(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$dst0, $dst1, $dst2, $dst3}, [$addr];"
        ).unwrap());
    }

    #[test]
    fn requires_exact_mnemonic_and_ordered_modifiers() {
        assert!(
            !ldmatrix_x4()
                .matches("loadmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];")
                .unwrap()
        );
        assert!(
            !ldmatrix_x4()
                .matches(
                    "ldmatrix_extra.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];"
                )
                .unwrap()
        );
        assert!(
            !ldmatrix_x4()
                .matches("ldmatrix.aligned.sync.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];")
                .unwrap()
        );
        assert!(
            !ldmatrix_x4()
                .matches("ldmatrix.sync.aligned.m8n8.x4.shared {%r1, %r2, %r3, %r4}, [%rd5];")
                .unwrap()
        );
        assert!(
            !ldmatrix_x4()
                .matches(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16.relaxed {%r1, %r2, %r3, %r4}, [%rd5];"
                )
                .unwrap()
        );
    }

    #[test]
    fn ordered_sparse_mma_qualifier_is_one_exact_modifier() {
        let pattern =
            InstructionPattern::new("mma", &["sp::ordered_metadata", "sync", "aligned"], vec![]);
        pattern.validate().unwrap();
        assert_eq!(
            pattern.to_string(),
            "mma.sp::ordered_metadata.sync.aligned;"
        );
        assert!(
            pattern
                .matches("mma.sp::ordered_metadata.sync.aligned;")
                .unwrap()
        );
        for invalid in [
            "mma.sp.sync.aligned;",
            "mma.sp.ordered_metadata.sync.aligned;",
            "mma.sp::ordered_metadata.sync.aligned.extra;",
        ] {
            assert!(!pattern.matches(invalid).unwrap(), "{invalid}");
        }
    }

    #[test]
    fn rejects_missing_shared_and_transposed_variant() {
        assert!(
            !ldmatrix_x4()
                .matches("ldmatrix.sync.aligned.m8n8.x4.b16 {%r1, %r2, %r3, %r4}, [%rd5];")
                .unwrap()
        );
        assert!(
            !ldmatrix_x4()
                .matches(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];"
                )
                .unwrap()
        );
    }

    #[test]
    fn requires_exact_top_level_operand_arity() {
        let pattern = InstructionPattern::new(
            "mov",
            &["u32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: "%tid.x".into(),
                },
            ],
        );
        assert!(pattern.matches("mov.u32 %r1, %tid.x;").unwrap());
        assert!(!pattern.matches("mov.u32 %r1;").unwrap());
        assert!(!pattern.matches("mov.u32 %r1, %tid.x, 0;").unwrap());
        assert!(!pattern.matches("mov.u32 %r1, %tid.y;").unwrap());
    }

    #[test]
    fn distinguishes_x2_and_x4_register_lists() {
        assert!(
            !ldmatrix_x4()
                .matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2}, [%rd5];")
                .unwrap()
        );

        let x2 = InstructionPattern::new(
            "ldmatrix",
            &["sync", "aligned", "m8n8", "x2", "shared", "b16"],
            vec![
                OperandPattern::RegisterList { length: 2 },
                OperandPattern::Address,
            ],
        );
        assert!(
            x2.matches("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r1, %r2}, [%rd5];")
                .unwrap()
        );
        assert!(
            !x2.matches("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];")
                .unwrap()
        );
    }

    #[test]
    fn accepts_tablegen_escaped_register_list_braces() {
        assert!(ldmatrix_x4().matches(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$rx40, $rx41, $rx42, $rx43}}, [$src];"
        ).unwrap());
    }

    #[test]
    fn block_comments_of_odd_or_even_length_do_not_mask_following_instruction() {
        let pattern = InstructionPattern::new(
            "mov",
            &["u32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: "%tid.x".into(),
                },
            ],
        );
        assert!(pattern.matches("/*x*/\nmov.u32 %r1, %tid.x;").unwrap());
        assert!(pattern.matches("/*xy*/\nmov.u32 %r1, %tid.x;").unwrap());
    }

    #[test]
    fn nested_commas_do_not_change_top_level_arity() {
        let pattern = InstructionPattern::new(
            "cp",
            &["async", "bulk", "tensor", "shared"],
            vec![
                OperandPattern::Address,
                OperandPattern::Address,
                OperandPattern::Address,
            ],
        );
        assert!(
            pattern
                .matches("cp.async.bulk.tensor.shared [%rd1], [%rd2, {%r1, %r2}], [%rd3];")
                .unwrap()
        );
    }

    #[test]
    fn exact_literals_and_addresses_are_typed() {
        let barrier = InstructionPattern::new(
            "bar",
            &["sync"],
            vec![OperandPattern::Exact { value: "0".into() }],
        );
        assert!(barrier.matches("bar.sync 0;").unwrap());
        assert!(!barrier.matches("bar.sync %r0;").unwrap());

        let load = InstructionPattern::new("ld", &["shared", "u32"], vec![OperandPattern::Address]);
        assert!(load.matches("ld.shared.u32 [%rd1 + 16];").unwrap());
        assert!(!load.matches("ld.shared.u32 %rd1;").unwrap());
        assert!(!load.matches("ld.shared.u32 [];").unwrap());
    }

    #[test]
    fn register_or_immediate_accepts_registers_and_integer_literals() {
        let vote = InstructionPattern::new(
            "vote",
            &["sync", "ballot", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
            ],
        );

        for member_mask in ["$mask", "%r3", "0", "-1", "+42", "0xFF", "-0X2a"] {
            assert!(
                vote.matches(&format!("vote.sync.ballot.b32 %r1, %p2, {member_mask};"))
                    .unwrap(),
                "member mask {member_mask:?}"
            );
        }
    }

    #[test]
    fn register_or_immediate_rejects_malformed_integer_literals() {
        let vote = InstructionPattern::new(
            "vote",
            &["sync", "ballot", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
            ],
        );

        for member_mask in [
            "+", "-", "0x", "-0x", "0xGG", "1.0", "1u", "0x1_0", "--1", "0b11",
        ] {
            assert!(
                !vote
                    .matches(&format!("vote.sync.ballot.b32 %r1, %p2, {member_mask};"))
                    .unwrap(),
                "member mask {member_mask:?}"
            );
        }
    }

    #[test]
    fn register_or_immediate_has_a_closed_policy_shape() {
        let pattern = InstructionPattern::new(
            "vote",
            &["sync", "all", "pred"],
            vec![OperandPattern::RegisterOrImmediate],
        );
        assert_eq!(
            pattern.to_string(),
            "vote.sync.all.pred <register-or-immediate>;"
        );

        let encoded = serde_json::to_string(&pattern).unwrap();
        assert!(encoded.contains(r#""kind":"register_or_immediate""#));
        assert_eq!(
            serde_json::from_str::<InstructionPattern>(&encoded).unwrap(),
            pattern
        );
        assert!(
            serde_json::from_str::<InstructionPattern>(
                r#"{"mnemonic":"vote","modifiers":[],"operands":[{"kind":"register_or_immediate","value":"-1"}]}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn immediate_accepts_only_integer_literals() {
        let wait_group = InstructionPattern::new(
            "cp",
            &["async", "wait_group"],
            vec![OperandPattern::Immediate],
        );

        for operand in ["0", "3", "-1", "+42", "0x7"] {
            assert!(
                wait_group
                    .matches(&format!("cp.async.wait_group {operand};"))
                    .unwrap(),
                "immediate {operand:?}"
            );
        }
        for operand in ["$n", "%r1", "1.0", "0x", "-"] {
            assert!(
                !wait_group
                    .matches(&format!("cp.async.wait_group {operand};"))
                    .unwrap(),
                "non-immediate {operand:?}"
            );
        }

        assert_eq!(wait_group.to_string(), "cp.async.wait_group <immediate>;");
        let encoded = serde_json::to_string(&wait_group).unwrap();
        assert!(encoded.contains(r#""kind":"immediate""#));
        assert_eq!(
            serde_json::from_str::<InstructionPattern>(&encoded).unwrap(),
            wait_group
        );
        assert!(
            serde_json::from_str::<InstructionPattern>(
                r#"{"mnemonic":"cp","modifiers":[],"operands":[{"kind":"immediate","value":"3"}]}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn register_predicate_pair_matches_emitted_and_tablegen_ptx() {
        let match_all = InstructionPattern::new(
            "match",
            &["all", "sync", "b32"],
            vec![
                OperandPattern::RegisterPredicatePair,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::RegisterOrImmediate,
            ],
        );

        assert!(
            match_all
                .matches("match.all.sync.b32 %r1|%p2, %r3, %r4;")
                .unwrap()
        );
        assert!(
            match_all
                .matches("match.all.sync.b32 $dest|$pred, $value, $mask;")
                .unwrap()
        );
        assert!(
            match_all
                .matches("match.all.sync.b32 %r1 | %p2, 7, -1;")
                .unwrap()
        );
    }

    #[test]
    fn register_predicate_pair_rejects_partial_or_malformed_pairs() {
        let match_all = InstructionPattern::new(
            "match",
            &["all", "sync", "b32"],
            vec![OperandPattern::RegisterPredicatePair],
        );

        for destination in [
            "%r1",
            "%r1|",
            "|%p2",
            "%r1|%p2|%p3",
            "%r1|%r2",
            "%r1|1",
            "1|%p2",
            "$dest|bad",
            "$dest|$value",
            "{%r1, %p2}",
        ] {
            assert!(
                !match_all
                    .matches(&format!("match.all.sync.b32 {destination};"))
                    .unwrap(),
                "destination {destination:?}"
            );
        }
    }

    #[test]
    fn register_predicate_pair_has_a_closed_policy_shape() {
        let pattern = InstructionPattern::new(
            "match",
            &["all", "sync", "b64"],
            vec![OperandPattern::RegisterPredicatePair],
        );
        assert_eq!(
            pattern.to_string(),
            "match.all.sync.b64 <register|predicate>;"
        );

        let encoded = serde_json::to_string(&pattern).unwrap();
        assert!(encoded.contains(r#""kind":"register_predicate_pair""#));
        assert_eq!(
            serde_json::from_str::<InstructionPattern>(&encoded).unwrap(),
            pattern
        );
        assert_eq!(
            toml::from_str::<InstructionPattern>(&toml::to_string(&pattern).unwrap()).unwrap(),
            pattern
        );

        for extra_field in [r#","value":"%r1|%p2""#, r#","length":2"#] {
            let source = format!(
                r#"{{"mnemonic":"match","modifiers":[],"operands":[{{"kind":"register_predicate_pair"{extra_field}}}]}}"#
            );
            assert!(serde_json::from_str::<InstructionPattern>(&source).is_err());
        }
    }

    #[test]
    fn comments_and_quoted_directives_never_supply_a_match() {
        let line_comment =
            "// ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];";
        let block_comment =
            "/* ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5]; */";
        let quoted =
            ".file 1 \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5];\"";
        assert!(!ldmatrix_x4().matches(line_comment).unwrap());
        assert!(!ldmatrix_x4().matches(block_comment).unwrap());
        assert!(!ldmatrix_x4().matches(quoted).unwrap());

        let real_instruction = format!(
            "{line_comment}\nldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r1, %r2, %r3, %r4}}, [%rd5]; // real"
        );
        assert!(ldmatrix_x4().matches(&real_instruction).unwrap());
    }

    #[test]
    fn matching_instructions_preserve_order_and_operands() {
        let pattern = InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Register,
                OperandPattern::Exact { value: "31".into() },
                OperandPattern::Register,
            ],
        );
        let source = r#"
            // shfl.sync.idx.b32 lo, lo, %r99, 31, %r98;
            shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;
            .file 1 "shfl.sync.idx.b32 lo, lo, %r97, 31, %r96;"
            shfl.sync.idx.b32 lo, lo, %r3, 31, %r4;
        "#;

        let matches = matching_instructions(source, &pattern).unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches[0].offset < matches[1].offset);
        assert!(
            matches
                .iter()
                .all(|instruction| instruction.end > instruction.offset)
        );
        assert_eq!(matches[0].operands, ["lo", "lo", "%r1", "31", "%r2"]);
        assert_eq!(matches[1].operands, ["lo", "lo", "%r3", "31", "%r4"]);

        let source = format!("{source}\nshfl.sync.idx.b32 hi, hi, %r5, 31, %r6;");
        let head_matches = instructions_with_matching_head(&source, &pattern).unwrap();
        assert_eq!(head_matches.len(), 3);
        assert_eq!(head_matches[2].operands[0], "hi");
    }

    #[test]
    fn matching_instructions_expose_same_statement_prefixes() {
        let pattern = InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Register,
                OperandPattern::Exact { value: "31".into() },
                OperandPattern::Register,
            ],
        );
        let matches =
            matching_instructions("@%p1 shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;", &pattern)
                .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].prefix, "@%p1");
    }

    #[test]
    fn malformed_delimiters_do_not_match() {
        assert!(
            !ldmatrix_x4()
                .matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4], [%rd5];")
                .unwrap()
        );
        assert!(
            !ldmatrix_x4()
                .matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd5;")
                .unwrap()
        );
    }

    #[test]
    fn an_empty_mnemonic_never_matches() {
        let pattern = InstructionPattern::new("", &[], vec![]);
        assert!(!pattern.matches("ret;").unwrap());
    }

    #[test]
    fn policy_shape_round_trips_and_rejects_unknown_fields() {
        let pattern = InstructionPattern::new(
            "mov",
            &["u32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: "%tid.x".into(),
                },
            ],
        );
        let encoded = serde_json::to_string(&pattern).unwrap();
        assert_eq!(
            serde_json::from_str::<InstructionPattern>(&encoded).unwrap(),
            pattern
        );
        let encoded = toml::to_string(&pattern).unwrap();
        assert_eq!(
            toml::from_str::<InstructionPattern>(&encoded).unwrap(),
            pattern
        );
        assert!(
            serde_json::from_str::<InstructionPattern>(
                r#"{"mnemonic":"mov","modifiers":["u32"],"operands":[],"extra":true}"#,
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<InstructionPattern>(
                r#"{"mnemonic":"mov","modifiers":["u32"],"operands":[{"kind":"register","extra":true}]}"#,
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<InstructionPattern>(
                r#"{"mnemonic":"mov","modifiers":["u32"],"operands":[{"kind":"wildcard"}]}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn malformed_policy_is_rejected_before_matching() {
        for pattern in [
            InstructionPattern::new("", &[], vec![]),
            InstructionPattern::new("mov.u32", &[], vec![]),
            InstructionPattern::new("mov", &[""], vec![]),
            InstructionPattern::new(
                "mov",
                &["u32"],
                vec![OperandPattern::Exact {
                    value: " %tid.x".into(),
                }],
            ),
            InstructionPattern::new(
                "ldmatrix",
                &["x1"],
                vec![OperandPattern::RegisterList { length: 0 }],
            ),
        ] {
            assert!(pattern.validate().is_err(), "{pattern:?}");
        }
    }
}
