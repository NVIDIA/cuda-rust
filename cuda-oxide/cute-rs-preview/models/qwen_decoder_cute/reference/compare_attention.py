#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay the same saved predictor inputs with official eager and SDPA math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from export_reference import SOURCE_TREE_SHA256, source_tree_sha256


def main():
    """Replay saved predictor inputs using separate eager and SDPA-math caches.

    Verify official source identity and saved BF16 tensors, then report hidden
    state/logit differences and selected codes. This diagnostic holds inputs
    fixed; it does not compare independently generated speech or benchmark speed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if source_tree_sha256(args.source_dir) != SOURCE_TREE_SHA256:
        raise ValueError('Official source differs from the pinned reference')
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(args.source_dir.resolve()))
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from qwen_tts import Qwen3TTSModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    manifest = json.loads((args.reference_dir / 'manifest.json').read_text())

    def read(name):
        """Read a hash-checked BF16 fixture and restore its recorded shape on the GPU."""
        entry = manifest['tensors'][name]
        raw = (args.reference_dir / entry['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256'] or entry['dtype'] != 'BF16':
            raise ValueError('Invalid BF16 fixture: ' + name)
        return torch.from_numpy(np.frombuffer(raw, dtype='<u2').copy()).view(torch.bfloat16).reshape(entry['shape']).cuda()

    wrapper = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    predictor = wrapper.model.talker.code_predictor.eval()
    results = []
    for backend in ['sdpa', 'eager']:
        predictor.model.set_attn_implementation(backend)
        cache = None
        for index, step in enumerate(manifest['predictor_steps']):
            prefix = f'predictor/step_{index:03}'
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                x = predictor.small_to_mtp_projection(read(prefix + '/adapter_input'))
                output = predictor.model(inputs_embeds=x, attention_mask=torch.tensor(step['attention_mask'], device='cuda'), position_ids=torch.tensor(step['position_ids'], device='cuda'), cache_position=torch.tensor(step['cache_position'], device='cuda'), past_key_values=cache, use_cache=True, output_hidden_states=True)
                cache = output.past_key_values
                hidden = output.last_hidden_state
                logits = predictor.lm_head[index](hidden)
                for stage, actual in [('hidden', hidden), ('logits', logits)]:
                    expected = read(prefix + '/' + stage)
                    a, e = actual.double(), expected.double()
                    diff = a - e
                    item = dict(backend=backend, name=prefix+'/'+stage, max_abs=diff.abs().max().item(), relative_l2=(diff.norm()/e.norm()).item(), cosine=torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(),dim=0).item())
                    if stage == 'logits':
                        item.update(actual_code=actual[0,-1].argmax().item(), reference_code=expected[0,-1].argmax().item())
                    results.append(item)
                    print(item, flush=True)
    args.output.write_text(json.dumps(dict(mode='official predictor replay using identical saved adapter inputs; separate caches per backend', model_revision=manifest['model_revision'], source_revision=manifest['source_revision'], results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
