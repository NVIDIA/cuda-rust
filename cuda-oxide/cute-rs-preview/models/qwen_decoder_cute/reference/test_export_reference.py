# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the reference artifact boundary."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from export_reference import DecoderHooks, Writer, source_tree_sha256, verify_artifacts


class ArtifactTests(unittest.TestCase):
    """Host regressions for capture windows, tensor bit patterns and pinned identity."""
    def test_predictor_capture_selects_second_frame_and_renumbers_its_steps(self):
        """Require the selected fifteen predictor calls to start at fixture step zero."""
        with tempfile.TemporaryDirectory() as directory:
            # Exercise the actual pre-hook/window writer without constructing a
            # large decoder. Module registration itself is PyTorch's contract.
            hook = DecoderHooks.__new__(DecoderHooks)
            hook.name = "predictor"
            hook.writer = Writer(Path(directory))
            hook.max_calls = 15
            hook.first_call = 15
            hook.call = -1
            hook.steps = []
            for call in range(31):
                hook.start(None, (), {"inputs_embeds": torch.full((1, 1, 2), call, dtype=torch.bfloat16)})
            self.assertEqual(len(hook.steps), 15)
            self.assertEqual(hook.steps[0]["call"], 0)
            first = Path(directory) / "predictor/step_000/input.bin"
            last = Path(directory) / "predictor/step_014/input.bin"
            self.assertEqual(first.read_bytes(), b"\x70\x41" * 2)  # BF16 15
            self.assertEqual(last.read_bytes(), b"\xe8\x41" * 2)  # BF16 29
            self.assertFalse((Path(directory) / "predictor/step_015/input.bin").exists())

    def test_bf16_payload_preserves_bits_and_token_major_cache_layout(self):
        """Preserve signed zero and NaN bits while exporting token-major cache storage."""
        with tempfile.TemporaryDirectory() as directory:
            writer = Writer(Path(directory))
            # [batch, head, token, lane] -> [batch, token, head, lane].
            bits = torch.tensor([0x0000, 0x8000, 0x3F80, 0x7FC1], dtype=torch.uint16)
            value = bits.view(torch.bfloat16).reshape(1, 2, 2, 1).transpose(1, 2)
            writer.tensor("cache/key", value)
            entry = writer.tensors["cache/key"]
            self.assertEqual(entry["shape"], [1, 2, 2, 1])
            self.assertEqual(entry["dtype"], "BF16")
            payload = (Path(directory) / entry["file"]).read_bytes()
            self.assertEqual(payload, b"\x00\x00\x80\x3f\x00\x80\xc1\x7f")
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())

    def test_integer_token_ids_are_not_converted_through_bf16(self):
        """Keep large vocabulary IDs exact in little-endian integer payloads."""
        with tempfile.TemporaryDirectory() as directory:
            writer = Writer(Path(directory))
            writer.tensor("ids", torch.tensor([151644, 2150], dtype=torch.int64))
            payload = (Path(directory) / "ids.bin").read_bytes()
            self.assertEqual(payload, (151644).to_bytes(8, "little") + (2150).to_bytes(8, "little"))

    def test_pinned_artifact_rejects_one_changed_byte(self):
        """Reject a same-length checkpoint mutation after verifying the original bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").write_bytes(b"pinned")
            hashes = {"model": hashlib.sha256(b"pinned").hexdigest()}
            self.assertEqual(verify_artifacts(root, hashes)["model"]["bytes"], 6)
            (root / "model").write_bytes(b"pinnEd")
            with self.assertRaisesRegex(ValueError, "differs from pinned revision"):
                verify_artifacts(root, hashes)

    def test_source_tree_hash_includes_paths_and_code_but_not_pycache(self):
        """Include Python filenames in source identity and exclude bytecode caches."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "qwen_tts"
            package.mkdir()
            (package / "first.py").write_text("x = 1\n")
            original = source_tree_sha256(root)
            (package / "ignored.pyc").write_bytes(b"bytecode")
            self.assertEqual(source_tree_sha256(root), original)
            (package / "first.py").rename(package / "second.py")
            self.assertNotEqual(source_tree_sha256(root), original)


if __name__ == "__main__":
    unittest.main()
