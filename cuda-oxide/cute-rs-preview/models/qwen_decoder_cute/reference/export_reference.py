#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export pinned official Qwen checkpoints/intermediates for Rust comparison.

This runner is an offline oracle, not a runtime dependency of the Rust model.
Tensor files are contiguous little-endian values. Shapes retain PyTorch's batch
axis; Q/K/V head tensors are exported in token-major order where noted.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

#: Pinned CustomVoice checkpoint revision; hashes below identify its local bytes.
MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
#: Official Qwen3-TTS implementation revision used by fixture generation.
SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
#: Digest of all relative Python paths and bytes under the pinned qwen_tts tree.
SOURCE_TREE_SHA256 = "340cf3d22ee52bcd96ca4cf1da1f2671f99d6f1b7abf62b1944eeb1a638dfe59"
#: Expected SHA-256 values for required local weights, configs and tokenizer files.
ARTIFACT_HASHES = {
    "config.json": "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
    "generation_config.json": "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
    "model.safetensors": "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137",
    "speech_tokenizer/config.json": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
    "speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    "tokenizer_config.json": "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
}
#: Named official implementation hashes recorded alongside the full-tree digest.
SOURCE_HASHES = {
    "qwen_tts/core/models/modeling_qwen3_tts.py": "25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0",
    "qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py": "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628",
}


def sha256(path):
    """Hash a local file in bounded chunks and return its hexadecimal SHA-256."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while data := source.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def source_tree_sha256(root):
    """Hash sorted relative Python paths and file bytes beneath qwen_tts.

    Bytecode caches are excluded; both a source rename and a content change alter
    the digest used to identify the pinned official implementation."""
    digest = hashlib.sha256()
    for path in sorted((root / "qwen_tts").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_artifacts(root, hashes):
    """Verify each named local file against its pinned SHA-256.

    Return byte counts and observed hashes for the manifest. Missing files raise
    an I/O error; changed bytes raise ValueError before model loading."""
    result = {}
    for name, expected in hashes.items():
        path = root / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Artifact differs from pinned revision: {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return result


class Writer:
    """Write raw tensor fixtures and retain their manifest entries.

    root is the output directory; tensors maps logical names to relative files,
    dtypes, shapes, byte counts and hashes. Callers supply trusted fixture names."""
    def __init__(self, root):
        """Create the fixture directory and an empty tensor manifest."""
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.tensors = {}

    def tensor(self, name, tensor):
        """Copy one tensor to contiguous host storage and write little-endian bytes.

        Supports BF16, FP32, I64 and I32. BF16 is serialized through its integer bit
        representation without numeric conversion. GPU readback completes before
        writing; the logical name selects both the output file and manifest entry."""
        import torch

        value = tensor.detach().contiguous().cpu()
        dtype = {torch.bfloat16: "BF16", torch.float32: "F32", torch.int64: "I64", torch.int32: "I32"}[value.dtype]
        if value.dtype == torch.bfloat16:
            data = value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
        else:
            data = value.numpy().astype({"F32": "<f4", "I64": "<i8", "I32": "<i4"}[dtype], copy=False).tobytes()
        target = self.root / (name + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.tensors[name] = {
            "file": str(target.relative_to(self.root)),
            "dtype": dtype,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }


class DecoderHooks:
    """Observe actual module inputs and outputs without replacing arithmetic.

    name and writer select the output namespace. first_call/max_calls bound the
    captured window; call tracks every observed invocation, including skipped
    ones. steps records positions/masks for selected calls, and handles retains
    the registrations that remove() later detaches.
    """

    def __init__(self, name, decoder, writer, max_calls, first_call=0):
        """Register input/output hooks for a bounded range of decoder calls.

        first_call skips earlier calls, and max_calls limits recording. Retained hook
        handles must later be removed with remove(); recorded steps are renumbered
        from zero within this selected range."""
        self.name, self.writer, self.max_calls = name, writer, max_calls
        self.first_call = first_call
        self.call = -1
        self.handles = []
        self.steps = []
        self.handles.append(decoder.register_forward_pre_hook(self.start, with_kwargs=True))
        self.handles.append(decoder.register_forward_hook(self.finish, with_kwargs=True))
        for index, layer in enumerate(decoder.layers):
            self.output(layer, f"layer_{index:02}/output")
        layer = decoder.layers[0]
        for path, stage in [
            ("input_layernorm", "normalized"), ("self_attn.q_proj", "q"),
            ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v"),
            ("self_attn.q_norm", "q_normed"), ("self_attn.k_norm", "k_normed"),
            ("self_attn.o_proj", "projected"), ("post_attention_layernorm", "post_normed"),
            ("mlp.gate_proj", "gate"), ("mlp.up_proj", "up"),
            ("mlp.act_fn", "silu"), ("mlp.down_proj", "down"),
        ]:
            self.output(layer.get_submodule(path), "layer_00/" + stage)
        self.input(layer.self_attn.o_proj, "layer_00/attention")
        self.input(layer.post_attention_layernorm, "layer_00/residual")
        self.input(layer.mlp.down_proj, "layer_00/product")
        self.handles.append(layer.self_attn.register_forward_pre_hook(self.attention, with_kwargs=True))
        self.output(decoder.norm, "hidden")

    @property
    def enabled(self):
        """Whether the current decoder call lies inside the requested capture window."""
        return self.first_call <= self.call < self.first_call + self.max_calls

    def save(self, name, value):
        """Write a tensor under the current relative step, if capture is enabled."""
        if self.enabled:
            self.writer.tensor(f"{self.name}/step_{self.call - self.first_call:03}/{name}", value)

    def start(self, _module, _args, kwargs):
        """Advance the call counter and capture input rows, positions and masks."""
        self.call += 1
        if not self.enabled:
            return
        self.save("input", kwargs["inputs_embeds"])
        item = {"call": self.call - self.first_call, "rows": kwargs["inputs_embeds"].shape[1]}
        for name in ["position_ids", "cache_position", "attention_mask"]:
            value = kwargs.get(name)
            if value is not None:
                item[name] = value.detach().cpu().tolist()
        self.steps.append(item)

    def finish(self, _module, _args, _kwargs, output):
        """Export completed key/value caches in [batch, token, head, lane] order."""
        if self.enabled and output.past_key_values is not None:
            cache = output.past_key_values
            # Decoder cache storage is [batch, head, token, lane]. Rust is token-major.
            for index, layer in enumerate(cache.layers):
                self.save(f"layer_{index:02}/cache_key", layer.keys.transpose(1, 2))
                self.save(f"layer_{index:02}/cache_value", layer.values.transpose(1, 2))

    def output(self, module, name):
        """Register a forward hook for a module tensor or the first tuple result."""
        def hook(_module, _args, output):
            """Save the observed output without replacing the module computation."""
            self.save(name, output[0] if isinstance(output, tuple) else output)
        self.handles.append(module.register_forward_hook(hook))

    def input(self, module, name):
        """Register a pre-hook that records the first positional tensor input."""
        def hook(_module, args):
            """Save the input before this module applies its arithmetic."""
            self.save(name, args[0])
        self.handles.append(module.register_forward_pre_hook(hook))

    def attention(self, _module, _args, kwargs):
        """Record the cosine and sine tables actually passed to attention."""
        cos, sin = kwargs["position_embeddings"]
        self.save("cos", cos)
        self.save("sin", sin)

    def remove(self):
        """Detach all retained hooks so later model calls stop writing fixtures."""
        for handle in self.handles:
            handle.remove()


def arguments():
    """Parse local source/model paths, capture window and optional codec fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text", default="Hello. This is a cuda-oxide acceptance test.")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--predictor-frame", type=int, default=0,
                        help="Zero-based generated codec frame whose fifteen predictor calls are captured")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--codec-fixtures", action="store_true")
    parser.add_argument("--codec-long-fixtures", action="store_true",
                        help="Also export 73/301-frame attention-window/chunk-boundary waveforms")
    return parser.parse_args()


def main():
    """Verify pinned inputs, run greedy reference generation and write fixtures.

    Hooks capture actual intermediate tensors and caches; their copies make this
    a correctness export, not an inference-speed measurement. Invalid capture
    windows or an incomplete predictor frame raise ValueError."""
    args = arguments()
    if args.max_new_tokens < 3:
        raise ValueError("At least three generation steps are required to capture prefill/decode")
    if not 0 <= args.predictor_frame < args.max_new_tokens - 1:
        raise ValueError("predictor-frame must select a frame before the generation cap")
    if sys.byteorder != "little":
        raise RuntimeError("Export is validated on little-endian hosts only")
    if source_tree_sha256(args.source_dir) != SOURCE_TREE_SHA256:
        raise ValueError("Official Python source differs from pinned revision")
    artifacts = verify_artifacts(args.model_dir, ARTIFACT_HASHES)
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from qwen_tts import Qwen3TTSModel

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    writer = Writer(args.output)
    print("Loading pinned model", flush=True)
    wrapper = Qwen3TTSModel.from_pretrained(
        str(args.model_dir), device_map=args.device, dtype=torch.bfloat16,
        attn_implementation=args.attention, local_files_only=True,
    )
    model = wrapper.model.eval()
    talker = model.talker
    predictor = talker.code_predictor
    codec = model.speech_tokenizer.model
    print("Loaded dtype:", next(talker.parameters()).dtype, "codec:", next(codec.decoder.parameters()).dtype, flush=True)
    # Observe exactly what the official prompt builder passes to generate().
    original_generate = talker.generate
    generation = {}
    def observed_generate(*positional, **kwargs):
        """Record the prepared prompt and selected sequence around the original call."""
        for name in ["inputs_embeds", "attention_mask", "trailing_text_hidden", "tts_pad_embed"]:
            writer.tensor("prompt/" + name, kwargs[name])
        generation.update({k: v for k, v in kwargs.items() if not torch.is_tensor(v)})
        result = original_generate(*positional, **kwargs)
        writer.tensor("generation/sequences", result.sequences)
        return result
    talker.generate = observed_generate
    hooks = [DecoderHooks("talker", talker.model, writer, args.max_new_tokens),
             DecoderHooks("predictor", predictor.model, writer, 15,
                          first_call=args.predictor_frame * 15)]
    handles = []
    hooks[0].output(talker.codec_head, "logits")
    # The input adapter runs before the predictor decoder pre-hook; record its
    # original 2048-wide input using a separate counter.
    adapter_calls = [0]
    def adapter_input(_module, inputs):
        """Capture the selected frame's fifteen pre-adapter inputs using a separate counter."""
        index = adapter_calls[0] - args.predictor_frame * 15
        adapter_calls[0] += 1
        if 0 <= index < 15:
            writer.tensor(f"predictor/step_{index:03}/adapter_input", inputs[0])
    handles.append(predictor.small_to_mtp_projection.register_forward_pre_hook(adapter_input))
    for index, head in enumerate(predictor.lm_head):
        hooks[1].output(head, "logits")
    ids = wrapper._tokenize_texts([wrapper._build_assistant_text(args.text)])
    writer.tensor("prompt/text_ids", ids[0])
    print("Generating reference traces", flush=True)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        codes, hidden = model.generate(
            input_ids=ids, instruct_ids=[None], languages=["English"], speakers=[args.speaker],
            non_streaming_mode=False, max_new_tokens=args.max_new_tokens,
            do_sample=False, subtalker_dosample=False, repetition_penalty=1.05,
        )
    writer.tensor("generation/codes", codes[0])
    writer.tensor("generation/talker_hidden", hidden[0])
    for hook in hooks:
        hook.remove()
    for handle in handles:
        handle.remove()
    talker.generate = original_generate
    if len(hooks[1].steps) != 15 or len(codes[0]) <= args.predictor_frame:
        raise ValueError("generation stopped before completing the requested predictor frame")
    print("Generated frames:", codes[0].tolist(), flush=True)
    if args.codec_fixtures or args.codec_long_fixtures:
        export_codec(codec, writer, args.device, codes[0])
    if args.codec_long_fixtures:
        export_codec_long(codec, writer, args.device)
    manifest = {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_HASHES, "source_tree_sha256": SOURCE_TREE_SHA256,
        "artifacts": artifacts,
        "environment": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "accelerate", "safetensors", "torchaudio", "numpy"]},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
        "attention": args.attention, "sdpa_backend": "math", "dtype": "BF16",
        "codec_dtype": str(next(codec.decoder.parameters()).dtype),
        "text": args.text, "speaker": args.speaker, "language": "English", "generation": generation,
        "talker_steps": hooks[0].steps, "predictor_steps": hooks[1].steps,
        "predictor_frame": args.predictor_frame,
        "codes": codes[0].cpu().tolist(), "tensors": writer.tensors,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote", len(writer.tensors), "tensors to", args.output, flush=True)


def export_codec(codec, writer, device, generated_codes):
    """Export short decoder stages for synthetic and generated audio-code frames.

    Inputs have [batch, frames, 16] layout; saved waveforms are widened to FP32.
    Temporary hooks observe intermediate stages and are removed after each case."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    decoder = codec.decoder
    names = ["pre_conv", "pre_transformer.input_proj", "pre_transformer.layers.0.input_layernorm",
             "pre_transformer.layers.0.self_attn.q_proj", "pre_transformer.layers.0.self_attn.k_proj",
             "pre_transformer.layers.0.self_attn.v_proj", "pre_transformer.layers.0", "pre_transformer",
             "upsample.0.1", "upsample.1.1", "decoder.0", "decoder.1", "decoder.2", "decoder.3", "decoder.4", "decoder.6"]
    for label, codes in [("two_frames", torch.arange(32, device=device).reshape(1, 2, 16) % 2048),
                         ("four_frames", (torch.arange(64, device=device).reshape(1, 4, 16) * 37 + 11) % 2048),
                         ("generated", generated_codes.unsqueeze(0))]:
        handles = []
        for name in names:
            def hook(_module, _inputs, output, name=name):
                """Unwrap this decoder stage's tensor result and save it under the fixture name."""
                if hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                if isinstance(output, tuple):
                    output = output[0]
                writer.tensor(f"codec/{label}/{name.replace('.', '/')}", output)
            handles.append(decoder.get_submodule(name).register_forward_hook(hook))
        # Quantizer.decode() and ModuleList containers do not invoke forward
        # hooks. Observe their actual output at the next operation's input.
        def quantizer_output(_module, inputs):
            """Record quantizer output at the next convolution's input boundary."""
            writer.tensor(f"codec/{label}/quantizer", inputs[0])
        handles.append(decoder.pre_conv.register_forward_pre_hook(quantizer_output))
        writer.tensor(f"codec/{label}/codes", codes)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            values = codec.decode(codes).audio_values[0]
        writer.tensor(f"codec/{label}/waveform", values.float())
        for handle in handles:
            handle.remove()
        print("Codec", label, "samples", values.numel(), flush=True)


def export_codec_long(codec, writer, device):
    """Exercise the 72-position attention window and official chunked_decode."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for label, frames in [("long_window", 73), ("chunk_boundary", 301)]:
        codes = (torch.arange(frames * 16, device=device).reshape(1, frames, 16) * 37 + 11) % 2048
        writer.tensor(f"codec/{label}/codes", codes)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            values = codec.decode(codes).audio_values[0]
        writer.tensor(f"codec/{label}/waveform", values.float())
        print("Codec", label, "samples", values.numel(), flush=True)


if __name__ == "__main__":
    main()
