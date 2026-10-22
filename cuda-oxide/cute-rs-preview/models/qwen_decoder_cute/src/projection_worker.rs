/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Diagnostic-only projection service using the production typed projection path.
//!
//! `--projection-worker MODEL_DIR` emits one ready JSON object, then accepts one
//! request per line: `{id, layer, stage, rows, input_bf16}`. Rows are one or
//! two. Predictor stages are q/k/v/projected/gate/up/down with layers `0..5`
//! and `adapter` (the biased 2048-to-1024 input projection, any layer value);
//! Talker stages are talker_q/talker_k/talker_v/talker_projected/talker_gate/
//! talker_up/talker_down with layers `0..28` and `talker_head`. Input and
//! output arrays contain exact unsigned BF16 bits in dense row-major order.
//! Hashes cover those bits serialized as little-endian u16. Each response is
//! completed before publication. Ordinary validation errors allow another
//! request; oversized frames, I/O failures, and GPU errors terminate the process.
//!
//! This uses standalone `Runner::project` calls, which dispatch by the same
//! per-shape reduction plan as generation. Generation groups Q/K/V and gate/up
//! submission through the same entries. These forced-input results are neither
//! autonomous generation nor performance evidence.

use crate::operator::Result;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{BufRead, Read, Write};

/// Maximum encoded request size, including its optional final newline, in bytes.
const MAX_REQUEST_BYTES: usize = 131_072;
/// Maximum UTF-8 byte length of a nonempty string request identifier.
const MAX_ID_BYTES: usize = 256;

/// A projection with compile-time widths at the execution boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Hidden width 1024 to sixteen 128-wide query heads.
    Query,
    /// Hidden width 1024 to eight 128-wide key heads.
    Key,
    /// Hidden width 1024 to eight 128-wide value heads.
    Value,
    /// Concatenated query-head attention width 2048 to hidden width 1024.
    Projected,
    /// Hidden width 1024 to feed-forward gate width 3072.
    Gate,
    /// Hidden width 1024 to feed-forward up width 3072.
    Up,
    /// Feed-forward product width 3072 to hidden width 1024.
    Down,
    /// Biased predictor input projection from width 2048 to hidden width 1024.
    Adapter,
    /// Talker hidden width 2048 to sixteen 128-wide query heads.
    TalkerQuery,
    /// Talker hidden width 2048 to eight 128-wide key heads.
    TalkerKey,
    /// Talker hidden width 2048 to eight 128-wide value heads.
    TalkerValue,
    /// Talker attention width 2048 to hidden width 2048.
    TalkerProjected,
    /// Talker hidden width 2048 to feed-forward gate width 6144.
    TalkerGate,
    /// Talker hidden width 2048 to feed-forward up width 6144.
    TalkerUp,
    /// Talker feed-forward product width 6144 to hidden width 2048.
    TalkerDown,
    /// Talker final hidden width 2048 to 3072 code-zero logits.
    TalkerHead,
}

impl Stage {
    /// Decode the deliberately closed protocol stage vocabulary.
    fn parse(value: &Value) -> Result<Self> {
        match value.as_str() {
            Some("q") => Ok(Self::Query),
            Some("k") => Ok(Self::Key),
            Some("v") => Ok(Self::Value),
            Some("projected") => Ok(Self::Projected),
            Some("gate") => Ok(Self::Gate),
            Some("up") => Ok(Self::Up),
            Some("down") => Ok(Self::Down),
            Some("adapter") => Ok(Self::Adapter),
            Some("talker_q") => Ok(Self::TalkerQuery),
            Some("talker_k") => Ok(Self::TalkerKey),
            Some("talker_v") => Ok(Self::TalkerValue),
            Some("talker_projected") => Ok(Self::TalkerProjected),
            Some("talker_gate") => Ok(Self::TalkerGate),
            Some("talker_up") => Ok(Self::TalkerUp),
            Some("talker_down") => Ok(Self::TalkerDown),
            Some("talker_head") => Ok(Self::TalkerHead),
            _ => Err("stage must be a predictor stage, adapter, or a talker_* stage".into()),
        }
    }

    /// Return the stable protocol spelling, including for echoed responses.
    fn name(self) -> &'static str {
        match self {
            Self::Query => "q",
            Self::Key => "k",
            Self::Value => "v",
            Self::Projected => "projected",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
            Self::Adapter => "adapter",
            Self::TalkerQuery => "talker_q",
            Self::TalkerKey => "talker_k",
            Self::TalkerValue => "talker_v",
            Self::TalkerProjected => "talker_projected",
            Self::TalkerGate => "talker_gate",
            Self::TalkerUp => "talker_up",
            Self::TalkerDown => "talker_down",
            Self::TalkerHead => "talker_head",
        }
    }

    /// Return `(input_width, output_width)` for dense row-major validation.
    fn geometry(self) -> (usize, usize) {
        match self {
            Self::Query => (1024, 2048),
            Self::Key | Self::Value => (1024, 1024),
            Self::Projected => (2048, 1024),
            Self::Gate | Self::Up => (1024, 3072),
            Self::Down => (3072, 1024),
            Self::Adapter => (2048, 1024),
            Self::TalkerQuery | Self::TalkerProjected => (2048, 2048),
            Self::TalkerKey | Self::TalkerValue => (2048, 1024),
            Self::TalkerGate | Self::TalkerUp => (2048, 6144),
            Self::TalkerDown => (6144, 2048),
            Self::TalkerHead => (2048, 3072),
        }
    }

    /// Number of decoder layers this stage may address; the heads ignore the layer.
    fn layers(self) -> usize {
        match self {
            Self::Query
            | Self::Key
            | Self::Value
            | Self::Projected
            | Self::Gate
            | Self::Up
            | Self::Down => 5,
            Self::Adapter | Self::TalkerHead => 1,
            _ => 28,
        }
    }
}

/// A completely checked request, with no retained borrowed protocol storage.
struct Request {
    /// Caller correlation ID, preserved as a string or unsigned JSON integer.
    id: Value,
    /// Zero-based predictor layer index, bounded to the five loaded layers.
    layer: usize,
    /// Projection role determining both typed weight identity and dimensions.
    stage: Stage,
    /// Dense row count, restricted to the existing one/two-row decode kernels.
    rows: usize,
    /// Finite BF16 values preserved exactly as their original unsigned bits.
    input_bf16: Vec<u16>,
}

/// Return a safe correlation ID or null for malformed request error responses.
fn response_id(value: &Value) -> Value {
    let id = &value["id"];
    if id.as_u64().is_some()
        || id
            .as_str()
            .is_some_and(|s| !s.is_empty() && s.len() <= MAX_ID_BYTES)
    {
        id.clone()
    } else {
        Value::Null
    }
}

/// Reject malformed dimensions and nonfinite BF16 values before any GPU work.
fn parse_request(value: &Value) -> Result<Request> {
    let object = value.as_object().ok_or("request must be a JSON object")?;
    if object.keys().any(|key| {
        !matches!(
            key.as_str(),
            "id" | "layer" | "stage" | "rows" | "input_bf16"
        )
    }) {
        return Err("request contains an unknown field".into());
    }
    let id = response_id(value);
    if id.is_null() {
        return Err(
            "id must be an unsigned integer or a nonempty string of at most 256 bytes".into(),
        );
    }
    let stage = Stage::parse(&value["stage"])?;
    let layer = value["layer"]
        .as_u64()
        .filter(|&n| n < stage.layers() as u64)
        .ok_or("layer must be an integer below the stage's layer count")? as usize;
    let rows = value["rows"]
        .as_u64()
        .filter(|&n| matches!(n, 1 | 2))
        .ok_or("rows must be one or two")? as usize;
    let input = value["input_bf16"]
        .as_array()
        .ok_or("input_bf16 must be an unsigned integer array")?;
    let expected = rows * stage.geometry().0;
    if input.len() != expected {
        return Err(format!("input_bf16 requires exactly {expected} elements").into());
    }
    let input_bf16 = input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let bits = value
                .as_u64()
                .filter(|&bits| bits <= u16::MAX as u64)
                .ok_or_else(|| format!("input_bf16[{index}] is not an unsigned 16-bit integer"))?
                as u16;
            if bits & 0x7f80 == 0x7f80 {
                return Err(format!("input_bf16[{index}] is nonfinite").into());
            }
            Ok(bits)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Request {
        id,
        layer,
        stage,
        rows,
        input_bf16,
    })
}

/// Read a single bounded line; EOF is clean, while overflow requires termination.
///
/// A final line without a newline is accepted. The one-byte overflow sentinel
/// bounds allocation and detects overlong frames without draining arbitrary data.
fn read_request_line(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut bytes = Vec::new();
    let count = reader
        .take((MAX_REQUEST_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if count > MAX_REQUEST_BYTES {
        return Err(format!("request exceeds {MAX_REQUEST_BYTES} bytes").into());
    }
    Ok((count != 0).then_some(bytes))
}

/// Hash dense BF16 values without depending on host byte order or FP conversion.
fn bits_sha256(bits: &[u16]) -> String {
    let mut hash = Sha256::new();
    for value in bits {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// Publish one compact JSON line and flush it before accepting another request.
fn write_response(output: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

/// Construct the common error envelope; fatal errors end the worker immediately.
fn error_response(id: Value, error: &dyn std::fmt::Display, fatal: bool) -> Value {
    json!({"type":"error", "schema_version":1, "id":id,
        "error":error.to_string(), "fatal":fatal})
}

// GPU boundary: host-only protocol tests below need no CUDA context or runtime.
use crate::{
    checkpoint::QwenWeights,
    generation::{PredictorWeights, TalkerWeights},
    model::{Batch, Runner},
};
use cuda_core::CudaContext;
use cute_rs::{
    BFloat16,
    decoder::BiasWeights,
    ffn::{ProjectionRole, ProjectionWeights},
};
use std::{fs, path::Path};

/// Upload owned input, execute the production role-typed projection, and read back.
///
/// `Batch` retains allocations through the existing readiness/completion checks;
/// no raw device pointer or launch is introduced. Output is completed before its
/// exact bits are exposed.
fn project_values<Role: ProjectionRole, const IN: usize, const OUT: usize>(
    runner: &Runner,
    request: &Request,
    weights: &ProjectionWeights<'_, Role, IN, OUT>,
    bias: Option<&BiasWeights<'_, Role, OUT>>,
) -> Result<Vec<u16>> {
    let values: Vec<_> = request
        .input_bf16
        .iter()
        .copied()
        .map(BFloat16::from_bits)
        .collect();
    let input = Batch::<IN>::from_host(runner.stream(), request.rows, &values)?;
    let output = runner.project(&input, weights, bias)?;
    Ok(output
        .read(runner.stream())?
        .iter()
        .map(|value| value.to_bits())
        .collect())
}

/// Select the exact typed weight role for the requested stage and layer.
fn execute(
    runner: &Runner,
    predictor: &PredictorWeights<'_>,
    talker: &TalkerWeights<'_>,
    request: &Request,
) -> Result<Value> {
    let output_bf16 = match request.stage {
        Stage::Adapter => project_values(
            runner,
            request,
            &predictor.projection,
            Some(&predictor.bias),
        )?,
        Stage::TalkerHead => project_values(runner, request, &talker.codec_head, None)?,
        Stage::Query
        | Stage::Key
        | Stage::Value
        | Stage::Projected
        | Stage::Gate
        | Stage::Up
        | Stage::Down => {
            let layer = predictor
                .layers
                .get(request.layer)
                .ok_or("loaded predictor layer is absent")?;
            match request.stage {
                Stage::Query => project_values(runner, request, layer.attention().query(), None)?,
                Stage::Key => project_values(runner, request, layer.attention().key(), None)?,
                Stage::Value => project_values(runner, request, layer.attention().value(), None)?,
                Stage::Projected => {
                    project_values(runner, request, layer.attention().output(), None)?
                }
                Stage::Gate => project_values(runner, request, layer.ffn().gate(), None)?,
                Stage::Up => project_values(runner, request, layer.ffn().up(), None)?,
                _ => project_values(runner, request, layer.ffn().down(), None)?,
            }
        }
        _ => {
            let layer = talker
                .layers
                .get(request.layer)
                .ok_or("loaded Talker layer is absent")?;
            match request.stage {
                Stage::TalkerQuery => {
                    project_values(runner, request, layer.attention().query(), None)?
                }
                Stage::TalkerKey => project_values(runner, request, layer.attention().key(), None)?,
                Stage::TalkerValue => {
                    project_values(runner, request, layer.attention().value(), None)?
                }
                Stage::TalkerProjected => {
                    project_values(runner, request, layer.attention().output(), None)?
                }
                Stage::TalkerGate => project_values(runner, request, layer.ffn().gate(), None)?,
                Stage::TalkerUp => project_values(runner, request, layer.ffn().up(), None)?,
                _ => project_values(runner, request, layer.ffn().down(), None)?,
            }
        }
    };
    let width = request.stage.geometry().1;
    if output_bf16.len() != request.rows * width {
        return Err("projection returned an unexpected output extent".into());
    }
    if output_bf16.iter().any(|bits| bits & 0x7f80 == 0x7f80) {
        return Err("projection returned a nonfinite output".into());
    }
    Ok(
        json!({"schema_version":1, "id":request.id, "layer":request.layer,
        "stage":request.stage.name(), "rows":request.rows, "width":width,
        "input_bf16_sha256":bits_sha256(&request.input_bf16),
        "output_bf16_sha256":bits_sha256(&output_bf16), "output_bf16":output_bf16}),
    )
}

/// Stream the executable hash so embedded device artifacts are content-pinned.
fn binary_sha256() -> Result<String> {
    let mut file = fs::File::open(std::env::current_exe()?)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65_536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Verify pinned model artifacts, initialize one worker, and serve until EOF.
///
/// Only protocol JSON is written to stdout. Startup failures propagate to main's
/// stderr diagnostic before readiness; fatal service errors emit an error object
/// and then propagate. Model, stream, and runner ownership spans the whole loop.
pub fn run(model_dir: &Path) -> Result<()> {
    let actual_artifacts = crate::checkpoint_validation::verify_pinned_artifacts(model_dir)?;
    let binary_sha256 = binary_sha256()?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let weights = QwenWeights::load(&stream, model_dir.join("model.safetensors"))?;
    let runner = Runner::new(stream)?;
    let predictor = weights.predictor()?;
    let talker = weights.talker()?;
    if predictor.layers.len() != 5 || talker.layers.len() != 28 {
        return Err("projection worker requires five predictor and 28 Talker layers".into());
    }
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    write_response(
        &mut output,
        &json!({
            "type":"ready", "schema_version":1,
            "mode":"DIAGNOSTIC production projection path, predictor and Talker decode shapes",
            "binary_sha256":binary_sha256, "actual_artifacts":actual_artifacts,
            "gpu":context.device_name()?,
            "cuda_visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            "source_sha256":{
                "worker":format!("{:x}",Sha256::digest(include_bytes!("projection_worker.rs"))),
                "decode_linear":format!("{:x}",Sha256::digest(include_bytes!("decode_linear.rs"))),
                "grouped_decode":format!("{:x}",Sha256::digest(include_bytes!("decode_linear_grouped.rs"))),
                "projection_kernel":format!("{:x}",Sha256::digest(include_bytes!("../../../examples/ffn_cute/src/kernel.rs"))),
                "reference_kernel":format!("{:x}",Sha256::digest(include_bytes!("reference_linear_kernel.rs"))),
                "reference_plan":format!("{:x}",Sha256::digest(include_bytes!("reference_linear.rs")))
            },
            "embedded_kernel_artifact_identity":"covered by binary_sha256",
            "max_request_bytes":MAX_REQUEST_BYTES,
            "grouped_equivalence_qualified":false,
            "qualification_required":"exact equality against every saved Rust stage fixture before intervention",
            "not_autonomous":true, "not_performance":true
        }),
    )?;
    loop {
        let bytes = match read_request_line(&mut input) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Ok(()),
            Err(error) => {
                write_response(&mut output, &error_response(Value::Null, &error, true))?;
                return Err(error);
            }
        };
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                write_response(&mut output, &error_response(Value::Null, &error, false))?;
                continue;
            }
        };
        let request = match parse_request(&value) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    &mut output,
                    &error_response(response_id(&value), &error, false),
                )?;
                continue;
            }
        };
        match execute(&runner, &predictor, &talker, &request) {
            Ok(response) => write_response(&mut output, &response)?,
            Err(error) => {
                write_response(&mut output, &error_response(request.id, &error, true))?;
                return Err(error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid request with a role-dependent dense input extent.
    fn request(stage: &str, rows: usize) -> Value {
        let width = Stage::parse(&json!(stage)).unwrap().geometry().0;
        json!({"id":"fixture", "layer":0, "stage":stage, "rows":rows,
            "input_bf16":vec![0u16;rows*width]})
    }

    #[test]
    fn all_projection_geometries_validate_exact_extents() {
        for (name, input, output) in [
            ("q", 1024, 2048),
            ("k", 1024, 1024),
            ("v", 1024, 1024),
            ("projected", 2048, 1024),
            ("gate", 1024, 3072),
            ("up", 1024, 3072),
            ("down", 3072, 1024),
        ] {
            for rows in [1, 2] {
                let mut value = request(name, rows);
                let parsed = parse_request(&value).unwrap();
                assert_eq!(parsed.stage.geometry(), (input, output));
                assert_eq!(parsed.stage.name(), name);
                assert_eq!(parsed.rows, rows);
                assert_eq!(parsed.layer, 0);
                assert_eq!(parsed.id, "fixture");
                assert_eq!(parsed.input_bf16.len(), rows * input);
                value["input_bf16"].as_array_mut().unwrap().pop();
                assert!(parse_request(&value).is_err());
                value["input_bf16"] = json!(vec![0; rows * input + 1]);
                assert!(parse_request(&value).is_err());
            }
        }
    }

    #[test]
    fn validates_layer_rows_stage_and_identifier_without_narrowing() {
        let base = request("q", 1);
        for (field, invalid) in [
            ("layer", json!(-1)),
            ("layer", json!(5)),
            ("layer", json!(u64::MAX)),
            ("layer", json!(1.5)),
            ("rows", json!(0)),
            ("rows", json!(3)),
            ("rows", json!(1.0)),
            ("stage", json!("output")),
            ("stage", Value::Null),
            ("id", json!("")),
            ("id", json!("x".repeat(257))),
            ("id", json!(-1)),
            ("id", json!(1.5)),
            ("id", Value::Null),
        ] {
            let mut value = base.clone();
            value[field] = invalid;
            assert!(parse_request(&value).is_err(), "field {field}");
        }
        for id in [json!(0), json!(u64::MAX), json!("x".repeat(256))] {
            let mut value = base.clone();
            value["id"] = id.clone();
            value["layer"] = json!(4);
            assert_eq!(parse_request(&value).unwrap().id, id);
        }
        let mut unknown = base.clone();
        unknown["typo"] = json!(true);
        assert!(parse_request(&unknown).is_err());
        for field in ["id", "layer", "stage", "rows", "input_bf16"] {
            let mut value = base.clone();
            value.as_object_mut().unwrap().remove(field);
            assert!(parse_request(&value).is_err());
        }
        assert!(parse_request(&json!([])).is_err());
    }

    #[test]
    fn rejects_nonfinite_and_non_u16_bits_but_preserves_finite_bits() {
        for invalid in [
            json!(-1),
            json!(65536),
            json!(1.5),
            json!("1"),
            json!(0x7f80),
            json!(0xff80),
            json!(0x7f81),
            json!(0xffff),
        ] {
            let mut value = request("q", 1);
            value["input_bf16"][0] = invalid;
            assert!(parse_request(&value).is_err());
        }
        let mut value = request("q", 1);
        let bits = [0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f];
        for (index, bits) in bits.iter().enumerate() {
            value["input_bf16"][index] = json!(bits);
        }
        assert_eq!(
            &parse_request(&value).unwrap().input_bf16[..bits.len()],
            &bits
        );
    }

    #[test]
    fn framing_is_bounded_and_preserves_following_requests() {
        let mut reader = Cursor::new(b"first\nsecond\nlast");
        assert_eq!(read_request_line(&mut reader).unwrap().unwrap(), b"first\n");
        assert_eq!(
            read_request_line(&mut reader).unwrap().unwrap(),
            b"second\n"
        );
        assert_eq!(read_request_line(&mut reader).unwrap().unwrap(), b"last");
        assert!(read_request_line(&mut reader).unwrap().is_none());
        let mut exact = vec![b' '; MAX_REQUEST_BYTES];
        exact[MAX_REQUEST_BYTES - 1] = b'\n';
        assert_eq!(
            read_request_line(&mut Cursor::new(exact))
                .unwrap()
                .unwrap()
                .len(),
            MAX_REQUEST_BYTES
        );
        assert!(read_request_line(&mut Cursor::new(vec![b'x'; MAX_REQUEST_BYTES + 1])).is_err());
    }

    #[test]
    fn bit_hash_uses_little_endian_and_error_envelopes_are_json_lines() {
        assert_eq!(
            bits_sha256(&[0x1234, 0xabcd]),
            format!("{:x}", Sha256::digest([0x34, 0x12, 0xcd, 0xab]))
        );
        let value = error_response(json!("a"), &"invalid request", false);
        let mut output = Vec::new();
        write_response(&mut output, &value).unwrap();
        assert_eq!(output.last(), Some(&b'\n'));
        assert_eq!(serde_json::from_slice::<Value>(&output).unwrap(), value);
        assert_eq!(response_id(&json!({"id":-1})), Value::Null);
    }
}
