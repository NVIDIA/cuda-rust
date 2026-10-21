/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! CustomVoice prompt construction and B=1 speech generation with explicit selection policy.
//!
//! Prepared text IDs become Talker prompt rows; each selected code-zero token
//! seeds fifteen code-predictor selections. The sixteen embeddings feed the
//! next Talker step with its text-conditioning row. Completed codec frames are
//! decoded together into a host waveform, then optionally written with a report.
//! Greedy requests may use a persistent eager or captured predictor; seeded
//! sampling retains host selection and independently derived model RNG streams.
//! Prompt streaming layout does not imply incremental waveform delivery.
use crate::{
    checkpoint::QwenWeights,
    checkpoint_validation::Reference,
    generation::{self, TalkerWeights, TextWeights},
    model::{Batch, DecoderCache, Runner},
    operator::Result,
    sampling::{Sampler, SamplingConfig},
};
use cuda_core::CudaContext;
use cute_rs::Bf16;
use cute_rs::decoder::TokenIds;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path, sync::Arc, time::Instant};

/// Validated tokenized B=1 CustomVoice request, before checkpoint-dependent speaker lookup.
///
/// Text IDs include the official assistant wrapper; tokenization itself happens
/// outside this module. Streaming mode here controls prompt layout, not audio delivery.
struct Request {
    /// Assistant-wrapped text tokens in the 151936-entry vocabulary.
    text_ids: Vec<u32>,
    /// Optional already-tokenized instruction prefix, using the same text vocabulary.
    instruct_ids: Vec<u32>,
    /// Lowercase speaker key resolved against the checkpoint config.
    speaker: String,
    /// Lowercase language key or `auto`, subject to configured speaker dialect overrides.
    language: String,
    /// Use the full-content prompt layout instead of interleaving later text rows with frames.
    non_streaming: bool,
    /// Code-zero selection limit in 2..=4096, including the final uncompleted selection.
    max_new_tokens: usize,
    /// Per-model selection policies and request-owned seed specification.
    sampling: SamplingPlan,
}

/// Separate typed Talker/predictor sampling policies and an explicit request seed.
struct SamplingPlan {
    /// Checked selection policy over 3072 Talker scores before external tag suppression.
    talker: SamplingConfig<3072>,
    /// Checked selection policy over each 2048-entry predictor codebook.
    predictor: SamplingConfig<2048>,
    /// Explicit Talker seed; predictor state uses a versioned derived seed.
    seed: u64,
    /// Whether stochastic sampling was requested, requiring host selection.
    enabled: bool,
}
impl SamplingPlan {
    /// Parse optional seeded sampling; absence selects greedy decoding for both models.
    ///
    /// Present sampling requires a u64 seed and complete per-model settings. Numeric
    /// settings are checked before use, including top-p bounds before FP32 narrowing.
    fn parse(value: Option<&Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self {
                talker: SamplingConfig::greedy()?,
                predictor: SamplingConfig::greedy()?,
                seed: 0,
                enabled: false,
            });
        };
        let seed = value["seed"]
            .as_u64()
            .ok_or("sampling requires an unsigned 64-bit seed")?;
        /// Parse one model's temperature/top-k/top-p against its typed vocabulary size.
        fn config<const V: usize>(value: &Value) -> Result<SamplingConfig<V>> {
            let temperature = value["temperature"]
                .as_f64()
                .ok_or("sampling requires temperature")? as f32;
            let top_k = usize::try_from(
                value["top_k"]
                    .as_u64()
                    .ok_or("sampling requires integer top_k")?,
            )?;
            let top_p = value["top_p"].as_f64().ok_or("sampling requires top_p")?;
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err("sampling top_p must be in (0,1] before FP32 conversion".into());
            }
            SamplingConfig::sample(temperature, top_k, top_p as f32)
        }
        Ok(Self {
            talker: config(&value["talker"])?,
            predictor: config(&value["predictor"])?,
            seed,
            enabled: true,
        })
    }
}

/// Derive a stable predictor RNG seed from SHA-256 of the versioned domain and seed bytes.
///
/// The first eight digest bytes are interpreted little-endian, separating predictor
/// draws from the Talker stream without claiming compatibility with Torch RNG.
fn predictor_seed(seed: u64) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"cuda-oxide/qwen-predictor/splitmix64-v1");
    digest.update(seed.to_le_bytes());
    let digest = digest.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}
impl Request {
    /// Check assistant-wrapped text IDs, optional instruction IDs, and request bounds.
    ///
    /// Normalizes speaker/language names to lowercase, defaults language to `auto`,
    /// and limits generation to 2..=4096 tokens. Speaker/language existence is checked
    /// later against the loaded model config; instruction IDs need no wrapper here.
    fn parse(value: &Value) -> Result<Self> {
        let parse_ids = |name: &str, required: bool| -> Result<Vec<u32>> {
            let Some(ids) = value.get(name) else {
                if required {
                    return Err(format!("request requires {name}").into());
                } else {
                    return Ok(Vec::new());
                }
            };
            let ids = ids.as_array().ok_or("token IDs must be an array")?;
            if ids.len() > 32760 {
                return Err("tokenized input exceeds the Talker position limit".into());
            }
            ids.iter()
                .map(|id| {
                    let id = u32::try_from(
                        id.as_u64()
                            .ok_or("token IDs must be nonnegative integers")?,
                    )?;
                    if id >= 151936 {
                        return Err("text token exceeds Qwen vocabulary".into());
                    }
                    Ok(id)
                })
                .collect()
        };
        let text_ids = parse_ids("text_ids", true)?;
        if text_ids.len() < 9
            || !text_ids.starts_with(&[151644, 77091, 198])
            || !text_ids.ends_with(&[151645, 198, 151644, 77091, 198])
        {
            return Err("text_ids must include the official assistant wrapper and at least one content token".into());
        }
        let speaker = value["speaker"]
            .as_str()
            .ok_or("request requires a speaker")?
            .to_lowercase();
        let language = value
            .get("language")
            .map(|language| language.as_str().ok_or("language must be a string"))
            .transpose()?
            .unwrap_or("auto")
            .to_lowercase();
        if speaker.is_empty() || language.is_empty() {
            return Err("CustomVoice speaker and language cannot be empty".into());
        }
        let non_streaming = value
            .get("non_streaming_mode")
            .map(|v| v.as_bool().ok_or("non_streaming_mode must be boolean"))
            .transpose()?
            .unwrap_or(false);
        let max_new_tokens = usize::try_from(
            value
                .get("max_new_tokens")
                .map(|v| v.as_u64().ok_or("max_new_tokens must be an integer"))
                .transpose()?
                .unwrap_or(128),
        )?;
        if !(2..=4096).contains(&max_new_tokens) {
            return Err("max_new_tokens must be between 2 and 4096".into());
        }
        Ok(Self {
            text_ids,
            instruct_ids: parse_ids("instruct_ids", false)?,
            speaker,
            language,
            non_streaming,
            max_new_tokens,
            sampling: SamplingPlan::parse(value.get("sampling"))?,
        })
    }
}

/// Checkpoint-configured token tags needed to assemble the CustomVoice prompt.
struct PromptPlan {
    /// Ordered think/language/speaker/pad/BOS tags in the 3072-entry codec vocabulary.
    codec_ids: Vec<u32>,
    /// Text BOS, EOS, and PAD IDs, in that fixed order.
    text_special: [u32; 3],
}
/// Resolve speaker, language/dialect overrides, and text/codec tags from CustomVoice config.
///
/// Auto or Chinese requests use a speaker-specific dialect when configured.
/// Unknown names, missing IDs, wrong model type, and out-of-vocabulary tags fail.
fn prompt_plan(request: &Request, config: &Value) -> Result<PromptPlan> {
    if config["tts_model_type"] != "custom_voice" {
        return Err("this runner requires the pinned CustomVoice model".into());
    }
    let id = |value: &Value, vocab: u32| -> Result<u32> {
        let id = u32::try_from(
            value
                .as_u64()
                .ok_or("configured token ID missing or invalid")?,
        )?;
        if id >= vocab {
            return Err("configured token exceeds vocabulary".into());
        }
        Ok(id)
    };
    let talker = &config["talker_config"];
    let speaker = id(&talker["spk_id"][&request.speaker], 3072)?;
    let mut language = if request.language == "auto" {
        None
    } else {
        Some(id(&talker["codec_language_id"][&request.language], 3072)?)
    };
    if matches!(request.language.as_str(), "auto" | "chinese")
        && let Some(dialect) = talker["spk_is_dialect"][&request.speaker].as_str()
    {
        language = Some(id(&talker["codec_language_id"][dialect], 3072)?);
    }
    let mut codec_ids = vec![
        id(
            &talker[if language.is_some() {
                "codec_think_id"
            } else {
                "codec_nothink_id"
            }],
            3072,
        )?,
        id(&talker["codec_think_bos_id"], 3072)?,
    ];
    if let Some(language) = language {
        codec_ids.push(language);
    }
    codec_ids.extend([
        id(&talker["codec_think_eos_id"], 3072)?,
        speaker,
        id(&talker["codec_pad_id"], 3072)?,
        id(&talker["codec_bos_id"], 3072)?,
    ]);
    Ok(PromptPlan {
        codec_ids,
        text_special: [
            id(&config["tts_bos_token_id"], 151936)?,
            id(&config["tts_eos_token_id"], 151936)?,
            id(&config["tts_pad_token_id"], 151936)?,
        ],
    })
}

/// Validate 3072 raw logits, suppress codec tags, then apply repetition penalty and selection.
///
/// Only codes `0..2048` and EOS 2150 remain eligible. EOS is suppressed for
/// the first two selections; previous codes receive the sign-aware 1.05 penalty
/// in FP32 before the request's greedy or stochastic policy is applied.
fn next_code_with(logits: &[Bf16], previous: &[u32], sampler: &mut Sampler<3072>) -> Result<u32> {
    if logits.len() != 3072 {
        return Err("Talker logits require 3072 values".into());
    }
    let mut scores = vec![f32::NEG_INFINITY; 3072];
    for (index, logit) in logits.iter().enumerate() {
        let mut score = logit.to_f32();
        if !score.is_finite() {
            return Err(format!("nonfinite Talker logit at {index}").into());
        }
        if (index >= 2048 && index != 2150) || (index == 2150 && previous.len() < 2) {
            continue;
        }
        if previous.contains(&(index as u32)) {
            score = if score < 0.0 {
                score * 1.05
            } else {
                score / 1.05
            };
        }
        scores[index] = score;
    }
    sampler.select(&scores)
}

/// Use greedy selection for CPU fixtures of tag masking, EOS gating, and repetition penalty.
#[cfg(test)]
fn next_code(logits: &[Bf16], previous: &[u32]) -> Result<u32> {
    next_code_with(
        logits,
        previous,
        &mut Sampler::new(SamplingConfig::greedy()?, 0),
    )
}

/// Owned dense BF16 rows for initial Talker input and per-frame text conditioning.
struct PreparedPrompt {
    /// Initial dense Talker prompt, with instructions and codec/text tags already combined.
    input: Batch<2048>,
    /// Text-conditioning rows consumed in order after each completed codec frame.
    trailing: Batch<2048>,
    /// One 2048-wide text PAD row reused after trailing text is exhausted.
    pad: Batch<2048>,
}

/// Assemble role, instruction, text, and codec embeddings using the pinned prompt schedule.
///
/// Non-streaming prompt layout includes all content up front; streaming layout
/// keeps later text rows for one-at-a-time frame conditioning. This controls
/// model inputs only: waveform delivery remains complete-output. Rejects invalid
/// config/IDs, CUDA failures, or prompt plus generation exceeding 32768 positions.
fn prepare(
    runner: &Runner,
    talker: &TalkerWeights<'_>,
    text: &TextWeights<'_>,
    request: &Request,
    config: &Value,
) -> Result<PreparedPrompt> {
    let plan = prompt_plan(request, config)?;
    let projected = generation::text_input(runner, text, TokenIds::new(&request.text_ids)?)?;
    let special = generation::text_input(runner, text, TokenIds::new(&plan.text_special)?)?;
    let codec = runner.embed(&talker.codec_embedding, TokenIds::new(&plan.codec_ids)?)?;
    let pad = runner.select(&special, &[2])?;
    let role = runner.select(&projected, &[0, 1, 2])?;
    let mut pad_bos = vec![2; plan.codec_ids.len() - 2];
    pad_bos.push(0);
    let text_tags = runner.select(&special, &pad_bos)?;
    let tag_indices = (0..u32::try_from(plan.codec_ids.len() - 1)?).collect::<Vec<_>>();
    let codec_tags = runner.select(&codec, &tag_indices)?;
    let tags = runner.add(&text_tags, &codec_tags)?;
    let mut input = runner.concat(&role, &tags)?;
    let codec_bos = runner.select(&codec, &[u32::try_from(plan.codec_ids.len() - 1)?])?;
    let trailing;
    if request.non_streaming {
        let content = (3..u32::try_from(request.text_ids.len() - 5)?).collect::<Vec<_>>();
        let content = runner.select(&projected, &content)?;
        let eos = runner.select(&special, &[1])?;
        let content = runner.concat(&content, &eos)?;
        let codec_pad = runner.select(&codec, &[u32::try_from(plan.codec_ids.len() - 2)?])?;
        let codec_pad = runner.select(&codec_pad, &vec![0; content.rows()])?;
        let content = runner.add(&content, &codec_pad)?;
        input = runner.concat(&input, &content)?;
        input = runner.concat(&input, &runner.add(&pad, &codec_bos)?)?;
        trailing = runner.select(&pad, &[0])?;
    } else {
        let first = runner.select(&projected, &[3])?;
        input = runner.concat(&input, &runner.add(&first, &codec_bos)?)?;
        let rest = (4..u32::try_from(request.text_ids.len() - 5)?).collect::<Vec<_>>();
        let rest = runner.select(&projected, &rest)?;
        trailing = runner.concat(&rest, &runner.select(&special, &[1])?)?;
    }
    if !request.instruct_ids.is_empty() {
        let instruction =
            generation::text_input(runner, text, TokenIds::new(&request.instruct_ids)?)?;
        input = runner.concat(&instruction, &input)?;
    }
    let capacity = input
        .rows()
        .checked_add(request.max_new_tokens - 1)
        .ok_or("Talker capacity overflow")?;
    if capacity > 32768 {
        return Err("prompt plus generation exceeds the Talker position limit".into());
    }
    Ok(PreparedPrompt {
        input,
        trailing,
        pad,
    })
}

/// Read and parse request/config JSON after rejecting files larger than 1 MiB.
///
/// Filesystem and JSON errors propagate; the size check is on pre-read metadata.
fn load_json(path: &Path) -> Result<Value> {
    if fs::metadata(path)?.len() > 1024 * 1024 {
        return Err("request/config JSON exceeds 1 MiB".into());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Compare the Rust-assembled prompt against the independently exported official
/// prompt. Existing runner and weights avoid another full checkpoint upload.
pub fn validate_prompt_with(
    runner: &Runner,
    weights: &QwenWeights,
    reference: &Reference,
    model_dir: &Path,
) -> Result<()> {
    let shape = reference.manifest()["tensors"]["prompt/text_ids"]["shape"]
        .as_array()
        .ok_or("reference text shape missing")?;
    let count = usize::try_from(
        shape
            .get(1)
            .and_then(Value::as_u64)
            .ok_or("reference text length missing")?,
    )?;
    let ids = reference
        .i64("prompt/text_ids", &[1, count])?
        .into_iter()
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let value = json!({"text_ids":ids,"speaker":reference.manifest()["speaker"],"language":reference.manifest()["language"],"max_new_tokens":reference.manifest()["generation"]["max_new_tokens"]});
    let request = Request::parse(&value)?;
    let prepared = prepare(
        runner,
        &weights.talker()?,
        &weights.text()?,
        &request,
        &load_json(&model_dir.join("config.json"))?,
    )?;
    for (name, batch) in [
        ("inputs_embeds", &prepared.input),
        ("trailing_text_hidden", &prepared.trailing),
        ("tts_pad_embed", &prepared.pad),
    ] {
        let actual = batch.read(runner.stream())?;
        let expected = reference.bf16(&format!("prompt/{name}"), &[1, batch.rows(), 2048])?;
        let (mut error, mut norm, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
        for (a, b) in actual.iter().zip(&expected) {
            let (a, b) = (f64::from(a.to_f32()), f64::from(b.to_f32()));
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite prompt comparison".into());
            }
            error += (a - b) * (a - b);
            norm += b * b;
            max_abs = max_abs.max((a - b).abs());
        }
        let relative_l2 = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
        println!("Rust prompt {name}: relative L2={relative_l2:.6}, max abs={max_abs:.6}");
        if relative_l2 > 0.01 {
            return Err(format!("Rust prompt {name} exceeds relative L2 0.01").into());
        }
    }
    Ok(())
}

/// Greedy predictor execution policy; sampled requests retain host selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PredictorMode {
    /// Use ordinary synchronous model operators and host selection for every code group.
    Synchronous,
    /// Reuse persistent private storage while submitting the fixed greedy schedule directly.
    Eager,
    /// Replay the captured fixed greedy schedule over persistent private storage.
    Graph,
}
impl PredictorMode {
    /// Parse `sync`, `eager`, or `graph`, defaulting to graph for greedy requests.
    ///
    /// Sampling always selects synchronous execution after validating the mode string,
    /// preserving host RNG ordering; even sampled requests reject unknown modes.
    fn parse(value: Option<&str>, sampled: bool) -> Result<Self> {
        let selected = match value {
            None | Some("graph") => Self::Graph,
            Some("eager") => Self::Eager,
            Some("sync") => Self::Synchronous,
            _ => return Err("QWEN_PREDICTOR_MODE must be sync, eager, or graph".into()),
        };
        // The captured selector implements greedy decoding. Seeded sampling
        // retains its original host RNG and draw order.
        Ok(if sampled { Self::Synchronous } else { selected })
    }

    /// Read `QWEN_PREDICTOR_MODE` and resolve a request-compatible execution policy.
    ///
    /// An absent variable uses defaults; invalid Unicode or unknown values fail.
    fn for_request(request: &Request) -> Result<Self> {
        let value = std::env::var("QWEN_PREDICTOR_MODE");
        match value {
            Ok(value) => Self::parse(Some(&value), request.sampling.enabled),
            Err(std::env::VarError::NotPresent) => Self::parse(None, request.sampling.enabled),
            Err(error) => Err(error.into()),
        }
    }

    /// Return the human-readable predictor execution label used in reports.
    fn label(self) -> &'static str {
        match self {
            Self::Synchronous => "synchronous",
            Self::Eager => "persistent eager predictor",
            Self::Graph => "CUDA Graph predictor",
        }
    }
}

/// Talker submission policy; both paths retain host token selection and sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TalkerMode {
    /// Complete every public model operator, retaining the numerical reference path.
    Synchronous,
    /// Submit all decoder layers and the head over private storage before completion.
    Eager,
}
impl TalkerMode {
    /// Parse the explicit comparison switch; unknown modes are errors.
    fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("eager") => Ok(Self::Eager),
            Some("sync") => Ok(Self::Synchronous),
            _ => Err("QWEN_TALKER_MODE must be sync or eager".into()),
        }
    }
    /// Read the selected mode once when loading the engine; reject invalid Unicode.
    fn from_env() -> Result<Self> {
        match std::env::var("QWEN_TALKER_MODE") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(error.into()),
        }
    }
    /// Stable report label for paired benchmark provenance.
    fn label(self) -> &'static str {
        match self {
            Self::Synchronous => "synchronous",
            Self::Eager => "persistent eager Talker",
        }
    }
}

/// Exclusive ownership of either the completed public cache or the private plan.
enum TalkerExecution {
    /// Ordinary operators own and publish their existing synchronous cache.
    Synchronous(DecoderCache<2048, 6144, 8>),
    /// One completed region per input row; boxed to keep this dispatch value small.
    Eager(Box<crate::talker_decode::TalkerDecodePlan>),
}
impl TalkerExecution {
    /// Report the common completed prefix used for the next rotary position.
    fn len(&self) -> usize {
        match self {
            Self::Synchronous(cache) => cache.len(),
            Self::Eager(plan) => plan.len(),
        }
    }
    /// Execute one decode row using the selected schedule and identical weight owner.
    fn step(
        &mut self,
        runner: &Runner,
        weights: &generation::TalkerWeights<'_>,
        input: &Batch<2048>,
        positions: &[[u32; 3]],
    ) -> Result<generation::TalkerStep> {
        match self {
            Self::Synchronous(cache) => {
                generation::talker_step(runner, weights, cache, input, positions)
            }
            Self::Eager(plan) => plan.run(input, positions),
        }
    }
}

/// Persistent model/codec owners and optional greedy predictor plan reused across requests.
struct Engine {
    /// Synchronous language-model operators and bounded output allocation pool.
    runner: Runner,
    /// Owned checkpoint shared with any persistent predictor plan.
    weights: Arc<QwenWeights>,
    /// Optional private eager/graph schedule; sampled requests bypass it.
    predictor_plan: Option<crate::predictor_graph::PredictorPlan>,
    /// Selected greedy execution policy reported for reproducibility.
    predictor_mode: PredictorMode,
    /// Selected Talker submission policy; host sampling is independent of this mode.
    talker_mode: TalkerMode,
    /// Wall seconds spent preparing optional predictor storage and graph.
    predictor_prepare_seconds: f64,
    /// Loaded speech-tokenizer decoder returning complete 24 kHz host audio.
    codec: crate::codec_model::Runner,
    /// Parsed model JSON used for speaker/language and prompt tags.
    config: Value,
    /// Wall seconds for config, CUDA setup, weights, predictor plan, and codec loading.
    load_seconds: f64,
    /// Verified pinned artifact identities included in every run report.
    actual_artifacts: Value,
    /// Hexadecimal SHA-256 of the running executable, computed before model-load timing.
    binary_sha256: String,
    /// CUDA device name reported by the GPU-0 context.
    gpu: String,
}
impl Engine {
    /// Verify pinned artifact hashes and load GPU-0 models plus the selected predictor plan.
    ///
    /// Hashes the running executable for provenance and records setup durations in
    /// seconds. Checkpoint, configuration, CUDA, and graph preparation errors propagate.
    fn load(model_dir: &Path, predictor_mode: PredictorMode) -> Result<Self> {
        let talker_mode = TalkerMode::from_env()?;
        let actual_artifacts = crate::checkpoint_validation::verify_pinned_artifacts(model_dir)?;
        let mut binary = fs::File::open(std::env::current_exe()?)?;
        let mut digest = Sha256::new();
        let mut bytes = [0u8; 65536];
        loop {
            let count = binary.read(&mut bytes)?;
            if count == 0 {
                break;
            }
            digest.update(&bytes[..count]);
        }
        let binary_sha256 = format!("{:x}", digest.finalize());
        let started = Instant::now();
        let config = load_json(&model_dir.join("config.json"))?;
        let context = CudaContext::new(0)?;
        let gpu = context.device_name()?;
        let stream = context.default_stream();
        let runner = Runner::new(stream.clone())?;
        let weights = Arc::new(QwenWeights::load(
            &stream,
            model_dir.join("model.safetensors"),
        )?);
        let predictor_started = Instant::now();
        let predictor_plan = match predictor_mode {
            PredictorMode::Synchronous => None,
            PredictorMode::Eager | PredictorMode::Graph => {
                Some(crate::predictor_graph::PredictorPlan::new(
                    stream.clone(),
                    weights.clone(),
                    predictor_mode == PredictorMode::Graph,
                )?)
            }
        };
        let predictor_prepare_seconds = predictor_started.elapsed().as_secs_f64();
        let codec = crate::codec_model::Runner::load(
            stream,
            &model_dir.join("speech_tokenizer/model.safetensors"),
        )?;
        Ok(Self {
            runner,
            weights,
            predictor_plan,
            predictor_mode,
            talker_mode,
            predictor_prepare_seconds,
            codec,
            config,
            load_seconds: started.elapsed().as_secs_f64(),
            actual_artifacts,
            binary_sha256,
            gpu,
        })
    }
    /// Generate complete codec frames and decode a complete host waveform for one request.
    ///
    /// Each invocation owns fresh Talker cache and sampler state; persistent predictor
    /// storage is reused only for greedy requests. A predicted code zero is completed
    /// by the predictor on the following iteration, so at most `max_new_tokens - 1`
    /// frames are decoded. Returns finite nonempty audio plus timing/provenance JSON;
    /// no WAV or report file is written here. Inference errors abort the request.
    fn synthesize(&mut self, request_value: &Value) -> Result<(Vec<f32>, Value)> {
        let request = Request::parse(request_value)?;
        // Separate RNG streams keep Talker choices independent of the number
        // of predictor draws. Equal seeds do not reproduce PyTorch's RNG.
        let mut talker_sampler = Sampler::new(request.sampling.talker, request.sampling.seed);
        let mut predictor_sampler = Sampler::new(
            request.sampling.predictor,
            predictor_seed(request.sampling.seed),
        );
        let (runner, weights, codec, config) =
            (&self.runner, &self.weights, &self.codec, &self.config);
        let stream = runner.stream();
        let synthesis = Instant::now();
        let prompt_started = Instant::now();
        let talker = weights.talker()?;
        let predictor = weights.predictor()?;
        let prompt = prepare(runner, &talker, &weights.text()?, &request, config)?;
        let prompt_seconds = prompt_started.elapsed().as_secs_f64();
        let capacity = prompt.input.rows() + request.max_new_tokens - 1;
        let mut cache = DecoderCache::<2048, 6144, 8>::new(stream, 28, capacity)?;
        let positions = (0..u32::try_from(prompt.input.rows())?)
            .map(|p| [p; 3])
            .collect::<Vec<_>>();
        let timer = Instant::now();
        let mut step =
            generation::talker_step(runner, &talker, &mut cache, &prompt.input, &positions)?;
        let mut execution = match self.talker_mode {
            TalkerMode::Synchronous => TalkerExecution::Synchronous(cache),
            TalkerMode::Eager => {
                TalkerExecution::Eager(Box::new(crate::talker_decode::TalkerDecodePlan::new(
                    stream.clone(),
                    weights.clone(),
                    cache,
                )?))
            }
        };
        // Per-request plan setup is included, just like the ordinary cache setup.
        let mut talker_seconds = timer.elapsed().as_secs_f64();
        let mut predictor_seconds = 0.0;
        let mut sequences = vec![next_code_with(
            &step.logits.read(stream)?,
            &[],
            &mut talker_sampler,
        )?];
        let mut frames = Vec::<[u32; 16]>::new();
        let mut first_frame_seconds = None;
        let mut stop_reason = "max_new_tokens";
        // Preserve the official generate() schedule: each subsequent Talker call
        // completes the previously predicted code-zero token, then predicts another.
        for _ in 1..request.max_new_tokens {
            let code = sequences[sequences.len() - 1];
            if code == 2150 {
                stop_reason = "eos";
                break;
            }
            let timer = Instant::now();
            let code_zero = generation::code_zero_input(runner, &talker, code)?;
            let frame = if let Some(plan) = self
                .predictor_plan
                .as_mut()
                .filter(|_| !request.sampling.enabled)
            {
                plan.run(&step.last_hidden, &code_zero)?
            } else {
                generation::predictor_frame(
                    runner,
                    &predictor,
                    &step.last_hidden,
                    &code_zero,
                    &mut predictor_sampler,
                )?
            };
            predictor_seconds += timer.elapsed().as_secs_f64();
            let mut codes = [0; 16];
            codes[0] = code;
            codes[1..].copy_from_slice(&frame.remaining_codes);
            let frame_index = frames.len();
            frames.push(codes);
            if first_frame_seconds.is_none() {
                first_frame_seconds = Some(synthesis.elapsed().as_secs_f64());
            }
            let text = if frame_index < prompt.trailing.rows() {
                runner.select(&prompt.trailing, &[u32::try_from(frame_index)?])?
            } else {
                runner.select(&prompt.pad, &[0])?
            };
            let input = runner.add(&frame.embedding_sum, &text)?;
            let position = u32::try_from(execution.len())?;
            let timer = Instant::now();
            step = execution.step(runner, &talker, &input, &[[position; 3]])?;
            talker_seconds += timer.elapsed().as_secs_f64();
            let code = next_code_with(&step.logits.read(stream)?, &sequences, &mut talker_sampler)?;
            sequences.push(code);
            if code == 2150 {
                stop_reason = "eos";
                break;
            }
        }
        if frames.is_empty() {
            return Err("generation produced no codec frames".into());
        }
        let generation_seconds = synthesis.elapsed().as_secs_f64();
        let timer = Instant::now();
        let waveform = codec.decode(&frames)?;
        let codec_seconds = timer.elapsed().as_secs_f64();
        let synthesis_seconds = synthesis.elapsed().as_secs_f64();

        let audio_seconds = waveform.len() as f64 / crate::codec_model::SAMPLE_RATE as f64;
        // Hashing is a validation/reporting cost, outside inference timing.
        let waveform_sha256 = waveform_digest(&waveform)?;
        let report = json!({"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","model_revision":"0c0e3051f131929182e2c023b9537f8b1c68adfe","actual_artifacts":self.actual_artifacts,"binary_sha256":self.binary_sha256,"gpu":self.gpu,"cuda_visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),"request":request_value,"mode":if request.sampling.enabled {"B=1 BF16 seeded sampling, host selection".to_owned()} else {format!("B=1 BF16 deterministic greedy, {}", self.predictor_mode.label())},"talker_execution":self.talker_mode.label(),"predictor_execution":if request.sampling.enabled {PredictorMode::Synchronous.label()} else {self.predictor_mode.label()},"predictor_prepare_seconds":self.predictor_prepare_seconds,"sampling_rng":if request.sampling.enabled {Some(crate::sampling::RNG_VERSION)} else {None},"talker_seed":request.sampling.seed,"predictor_seed":predictor_seed(request.sampling.seed),"seed_derivation":"predictor uses first 8 little-endian bytes of SHA256(cuda-oxide/qwen-predictor/splitmix64-v1 || seed_le_u64)","ready_for_speedup_claim":false,"delivery_mode":"complete_waveform","incremental_streaming":false,"sample_rate":24_000,"samples":waveform.len(),"audio_seconds":audio_seconds,"waveform_f32le_sha256":waveform_sha256,"stop_reason":stop_reason,"code_zero_sequence":sequences,"codec_frames":frames,"model_load_seconds":self.load_seconds,"prompt_seconds":prompt_seconds,"talker_seconds":talker_seconds,"predictor_seconds":predictor_seconds,"generation_seconds":generation_seconds,"codec_seconds":codec_seconds,"synthesis_seconds":synthesis_seconds,"first_codec_frame_seconds":first_frame_seconds,"first_playable_audio_seconds":synthesis_seconds,"first_playable_audio_scope":"first complete waveform returned to host; speech quality assessed separately; no incremental delivery","waveform_available_seconds":synthesis_seconds,"real_time_factor":synthesis_seconds/audio_seconds,"codec_frames_per_second":frames.len()as f64/synthesis_seconds,"generation_codec_frames_per_second":frames.len()as f64/generation_seconds});
        Ok((waveform, report))
    }
}

/// Hash exact little-endian FP32 waveform bytes after requiring finite nonempty audio.
///
/// The hexadecimal SHA-256 digest is computed outside inference timing.
fn waveform_digest(waveform: &[f32]) -> Result<String> {
    if waveform.is_empty() || waveform.iter().any(|sample| !sample.is_finite()) {
        return Err("benchmark requires a nonempty finite waveform".into());
    }
    let mut digest = Sha256::new();
    for sample in waveform {
        digest.update(sample.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Compare non-null codec frames, stopping reason, sample count, and waveform digest.
///
/// Timing and provenance fields deliberately do not affect deterministic output identity.
fn same_output(left: &Value, right: &Value) -> bool {
    [
        "codec_frames",
        "stop_reason",
        "samples",
        "waveform_f32le_sha256",
    ]
    .iter()
    .all(|key| !left[*key].is_null() && left[*key] == right[*key])
}

/// Load one tokenized request, synthesize audio, and write a mono float WAV plus JSON sidecar.
///
/// The sidecar replaces the output path's extension with `json`. Existing files
/// may be overwritten; validation, inference, and I/O errors propagate and outputs
/// are not committed atomically.
pub fn run(model_dir: &Path, request_json: &Path, output_wav: &Path) -> Result<()> {
    let request = load_json(request_json)?;
    let parsed = Request::parse(&request)?;
    let mut engine = Engine::load(model_dir, PredictorMode::for_request(&parsed)?)?;
    let (waveform, mut report) = engine.synthesize(&request)?;
    crate::wav::write(
        output_wav,
        crate::codec_model::SAMPLE_RATE as u32,
        &waveform,
    )?;
    report["waveform"] = json!(output_wav);
    let report_path = output_wav.with_extension("json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "Wrote {} ({} samples), report {}",
        output_wav.display(),
        waveform.len(),
        report_path.display()
    );
    Ok(())
}

/// Return the observed nearest-rank quantile for a fraction in `(0, 1]`.
///
/// Sorts a copy and uses `ceil(len * fraction) - 1`; rejects empty/nonfinite
/// samples and invalid fractions rather than interpolating between observations.
fn percentile(values: &[f64], fraction: f64) -> Result<f64> {
    if values.is_empty()
        || values.iter().any(|v| !v.is_finite())
        || !fraction.is_finite()
        || fraction <= 0.0
        || fraction > 1.0
    {
        return Err("percentile requires finite samples and fraction in (0,1]".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize).saturating_sub(1);
    Ok(sorted[index.min(sorted.len() - 1)])
}

/// Weights and the prepared predictor plan persist across warmups/repetitions.
/// Each sample includes prompt work, Talker cache allocation, decoding and codec;
/// predictor storage/capture preparation is reported separately as load work.
pub fn benchmark(
    model_dir: &Path,
    request_json: &Path,
    output_dir: &Path,
    repetitions: usize,
) -> Result<()> {
    if !(2..=100).contains(&repetitions) {
        return Err("benchmark repetitions must be 2..=100".into());
    }
    let request = load_json(request_json)?;
    let parsed = Request::parse(&request)?;
    let mut engine = Engine::load(model_dir, PredictorMode::for_request(&parsed)?)?;
    fs::create_dir_all(output_dir)?;
    for _ in 0..3 {
        engine.synthesize(&request)?;
    }
    let mut runs = Vec::new();
    for index in 0..repetitions {
        let (waveform, report) = engine.synthesize(&request)?;
        if runs
            .first()
            .is_some_and(|first| !same_output(first, &report))
        {
            return Err(
                "deterministic benchmark changed codes, stopping, or waveform between runs".into(),
            );
        }
        if index == 0 {
            crate::wav::write(&output_dir.join("output.wav"), 24_000, &waveform)?;
        }
        println!(
            "Benchmark {}/{repetitions}: {:.6}s",
            index + 1,
            report["synthesis_seconds"].as_f64().unwrap_or_default()
        );
        runs.push(report);
    }
    let latencies: Vec<_> = runs
        .iter()
        .map(|r| r["synthesis_seconds"].as_f64().unwrap())
        .collect();
    let rtfs: Vec<_> = runs
        .iter()
        .map(|r| r["real_time_factor"].as_f64().unwrap())
        .collect();
    let report = json!({"schema_version":2,"warmups":3,"repetitions":repetitions,"timing_boundary":"prepared token IDs to complete host waveform; includes prompt GPU work, Talker cache allocation, decoding, codec; excludes weights and predictor plan preparation, tokenization, WAV file I/O and output validation","delivery_mode":"complete_waveform","incremental_streaming":false,"outputs_bit_identical":true,"percentile_estimator":"nearest rank (observed sample)","first_playable_audio_p50_seconds":percentile(&latencies,0.5)?,"first_playable_audio_p95_seconds":percentile(&latencies,0.95)?,"peak_gpu_memory_bytes":null,"latency_p50_seconds":percentile(&latencies,0.5)?,"latency_p95_seconds":percentile(&latencies,0.95)?,"real_time_factor_p50":percentile(&rtfs,0.5)?,"real_time_factor_p95":percentile(&rtfs,0.95)?,"real_time_factor_definition":"wall seconds / audio seconds; lower is better","runs":runs});
    fs::write(
        output_dir.join("benchmark.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

/// CPU-only request, prompt-policy, sampling, and benchmark-report validation.
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    /// Build the smallest wrapped English CustomVoice request used by policy fixtures.
    fn request() -> Value {
        json!({"text_ids":[151644,77091,198,9707,151645,198,151644,77091,198],"speaker":"Ryan","language":"English","max_new_tokens":4})
    }
    /// Verify execution-mode parsing and the mandatory synchronous fallback for sampling.
    #[test]
    fn predictor_execution_preserves_sampling_and_rejects_unknown_modes() {
        assert_eq!(
            PredictorMode::parse(None, false).unwrap(),
            PredictorMode::Graph
        );
        assert_eq!(
            PredictorMode::parse(Some("eager"), false).unwrap(),
            PredictorMode::Eager
        );
        assert_eq!(
            PredictorMode::parse(Some("sync"), false).unwrap(),
            PredictorMode::Synchronous
        );
        for mode in [None, Some("sync"), Some("eager"), Some("graph")] {
            assert_eq!(
                PredictorMode::parse(mode, true).unwrap(),
                PredictorMode::Synchronous
            );
        }
        for sampled in [false, true] {
            assert!(PredictorMode::parse(Some("typo"), sampled).is_err());
        }
    }

    /// Build a minimal config with ordinary and dialect-specific speakers and token tags.
    fn config() -> Value {
        json!({"tts_model_type":"custom_voice","tts_bos_token_id":151672,"tts_eos_token_id":151673,"tts_pad_token_id":151671,"talker_config":{"codec_think_id":2154,"codec_nothink_id":2155,"codec_think_bos_id":2156,"codec_think_eos_id":2157,"codec_pad_id":2148,"codec_bos_id":2149,"spk_id":{"ryan":3061,"eric":2875},"spk_is_dialect":{"ryan":false,"eric":"sichuan_dialect"},"codec_language_id":{"english":2050,"chinese":2055,"sichuan_dialect":2062}}})
    }
    /// Reject invalid audio and detect changes in every deterministic output field.
    #[test]
    fn waveform_validation_and_run_stability_include_codes_and_audio() {
        assert!(waveform_digest(&[]).is_err());
        assert!(waveform_digest(&[f32::NAN]).is_err());
        assert!(waveform_digest(&[f32::INFINITY]).is_err());
        let first = json!({"codec_frames":[[1,2]],"stop_reason":"eos","samples":2,"waveform_f32le_sha256":waveform_digest(&[0., 1.]).unwrap()});
        assert!(same_output(&first, &first));
        for (key, value) in [
            ("codec_frames", json!([[2, 1]])),
            ("stop_reason", json!("cap")),
            ("samples", json!(3)),
            (
                "waveform_f32le_sha256",
                json!(waveform_digest(&[0., -1.]).unwrap()),
            ),
        ] {
            let mut changed = first.clone();
            changed[key] = value;
            assert!(!same_output(&first, &changed));
        }
        assert!(!same_output(&json!({}), &json!({})));
    }
    /// Verify nearest-rank selection and rejection of empty or nonfinite measurements.
    #[test]
    fn benchmark_percentiles_use_observed_nearest_rank_and_reject_nonfinite() {
        assert_eq!(percentile(&[3., 1., 4., 2.], 0.5).unwrap(), 2.);
        assert_eq!(percentile(&[3., 1., 4., 2.], 0.95).unwrap(), 4.);
        assert!(percentile(&[], 0.5).is_err());
        assert!(percentile(&[f64::NAN], 0.5).is_err());
        assert!(percentile(&[1.], 1.5).is_err());
    }
    /// Check wrapper, vocabulary, and generation-length validation before model execution.
    #[test]
    fn request_rejects_unwrapped_empty_out_of_vocab_and_unbounded_generation() {
        let good = request();
        assert_eq!(Request::parse(&good).unwrap().text_ids.len(), 9);
        for ids in [
            json!([]),
            json!([9707]),
            json!([151644, 77091, 198, 151936, 151645, 198, 151644, 77091, 198]),
        ] {
            let mut bad = good.clone();
            bad["text_ids"] = ids;
            assert!(Request::parse(&bad).is_err());
        }
        for limit in [0, 1, 4097] {
            let mut bad = good.clone();
            bad["max_new_tokens"] = json!(limit);
            assert!(Request::parse(&bad).is_err());
        }
    }
    /// Distinguish an absent language default from explicitly malformed language values.
    #[test]
    fn request_defaults_absent_language_but_rejects_nonstring_language() {
        let mut value = request();
        value.as_object_mut().unwrap().remove("language");
        assert_eq!(Request::parse(&value).unwrap().language, "auto");
        for bad in [json!(null), json!(123), json!([]), json!(true)] {
            value["language"] = bad;
            assert!(Request::parse(&value).is_err());
        }
    }
    /// Exercise per-model sampling bounds and the fixed predictor seed derivation.
    #[test]
    fn sampling_request_keeps_vocabularies_and_seed_checked() {
        let mut value = request();
        assert!(!Request::parse(&value).unwrap().sampling.enabled);
        value["sampling"] = json!({"seed":42,"talker":{"temperature":0.9,"top_k":50,"top_p":1.0},"predictor":{"temperature":0.9,"top_k":50,"top_p":1.0}});
        let parsed = Request::parse(&value).unwrap();
        assert!(parsed.sampling.enabled);
        assert_eq!(parsed.sampling.seed, 42);
        for field in ["seed", "talker", "predictor"] {
            let mut bad = value.clone();
            bad["sampling"][field] = Value::Null;
            assert!(Request::parse(&bad).is_err());
        }
        value["sampling"]["predictor"]["top_k"] = json!(2049);
        assert!(Request::parse(&value).is_err());
        value["sampling"]["predictor"]["top_k"] = json!(2048);
        value["sampling"]["talker"]["top_k"] = json!(3072);
        assert!(Request::parse(&value).is_ok());
        value["sampling"]["talker"]["top_p"] = json!(1.00000001);
        assert!(Request::parse(&value).is_err());
        // Domain separation must not shift one SplitMix stream by its increment,
        // including seed zero. The derivation is stable for reproducible runs.
        assert_ne!(predictor_seed(0), 0);
        assert_ne!(predictor_seed(0), 0x9e3779b97f4a7c15);
        assert_ne!(predictor_seed(0), predictor_seed(1));
        assert_eq!(predictor_seed(0), 1544247571153862893);
    }
    /// Verify language tags, auto mode, dialect overrides, and missing-speaker errors.
    #[test]
    fn prompt_tags_follow_language_speaker_and_dialect_policy() {
        let mut value = request();
        assert_eq!(
            prompt_plan(&Request::parse(&value).unwrap(), &config())
                .unwrap()
                .codec_ids,
            [2154, 2156, 2050, 2157, 3061, 2148, 2149]
        );
        value["language"] = json!("Auto");
        assert_eq!(
            prompt_plan(&Request::parse(&value).unwrap(), &config())
                .unwrap()
                .codec_ids,
            [2155, 2156, 2157, 3061, 2148, 2149]
        );
        value["speaker"] = json!("Eric");
        assert_eq!(
            prompt_plan(&Request::parse(&value).unwrap(), &config())
                .unwrap()
                .codec_ids,
            [2154, 2156, 2062, 2157, 2875, 2148, 2149]
        );
        value["speaker"] = json!("missing");
        assert!(prompt_plan(&Request::parse(&value).unwrap(), &config()).is_err());
    }
    /// Check codec-tag masking, two-step EOS suppression, and the repetition penalty.
    #[test]
    fn generation_policy_suppresses_tags_and_eos_then_applies_repetition_penalty() {
        let mut logits = vec![Bf16::from_f32(-10.0); 3072];
        logits[3061] = Bf16::from_f32(100.0);
        logits[2150] = Bf16::from_f32(99.0);
        logits[7] = Bf16::from_f32(1.0);
        logits[8] = Bf16::from_f32(0.98);
        assert_eq!(next_code(&logits, &[]).unwrap(), 7);
        assert_eq!(next_code(&logits, &[7]).unwrap(), 8);
        assert_eq!(next_code(&logits, &[7, 8]).unwrap(), 2150);
    }
}
