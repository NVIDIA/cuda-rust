#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate identical FP16 fixtures before timing Rust and pinned CUTLASS FMHA."""
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
COMMIT = 'dcf215af68a2d08d305076c152a06f201728cd53'
SOURCE = ROOT / 'reference/cute/blackwell/kernel/attention/fmha/fmha.py'
SOURCE_SHA256 = 'c11801367b5681e531b756238920124ba4d2ca7809bdb4f9aacd9aaa9ae3a955'
SOURCE_DIGESTS = {
    'cute/blackwell/kernel/attention/fmha/fmha.py': SOURCE_SHA256,
    'helpers/fmha_helpers.py': 'cbe090eb5567b7cdbd8b4944cbd579084fd60379c463bcfb011e6d4b28ef20e1',
    'helpers/__init__.py': '9cf1ae75e6473e5acf38b8ef7b498a14ff74fd54741a930708f6995fcacb7581',
}


SETUP_HINT = ('from cuda-oxide/ run `python3 -m venv cute-rs-preview/.venv && '
              'cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt`, '
              'then rerun; or set CUDA_OXIDE_CUTE_PYTHON to an environment that has them')


def require_reference_environment():
    """Stop before any GPU work when the shared comparison environment is missing."""
    missing = [name for name in ('numpy', 'torch', 'cutlass', 'cuda')
               if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(f'missing Python packages: {", ".join(missing)}; {SETUP_HINT}')


def verify_sources(reference=ROOT / 'reference'):
    for name, expected in SOURCE_DIGESTS.items():
        if hashlib.sha256((reference / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'vendored FMHA source digest mismatch: {name}')


def require_matched_optimization(rust_result):
    if rust_result.get('compiler_optimization_level') != 2:
        raise ValueError('timing requires both implementations at compiler optimization level 2; '
                         'rebuild with CUDA_OXIDE_CUTLASS_OPT_LEVEL=2')


@dataclass(frozen=True)
class Config:
    batch: int = 1
    query: int = 1024
    keys: int = 1024
    heads: int = 8
    kv_heads: int = 8
    head_dim: int = 128
    causal: bool = False

    def validate(self):
        if self.head_dim != 128:
            raise ValueError('this matched FP16 profile requires head_dim=128')
        for name in ['batch', 'query', 'keys', 'heads', 'kv_heads']:
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 < value < 2**31 - 256:
                raise ValueError(f'{name} must be positive and fit padded i32 coordinates')
        if self.heads % self.kv_heads:
            raise ValueError('query heads must be divisible by KV heads')
        if self.batch * self.heads * ((self.query + 255) // 256) > (2**31 - 1) // 2:
            raise ValueError('persistent work tile count exceeds the supported grid limit')
        for count in [self.batch * self.query * self.heads * 128,
                      self.batch * self.keys * self.kv_heads * 128]:
            if count > (sys.maxsize // 2):
                raise ValueError('FP16 buffer size exceeds host addressable memory')
        return self

    def kv_head(self, head):
        return head // (self.heads // self.kv_heads)

    def valid_keys(self, query_row):
        return min(self.keys, query_row + 1) if self.causal else self.keys

    def cli(self):
        args = ['--shape', ','.join(str(x) for x in
                [self.batch, self.query, self.keys, self.heads, self.kv_heads])]
        return args + (['--causal'] if self.causal else [])


def write_f16(path, values):
    with Path(path).open('wb') as output:
        for value in values:
            output.write(struct.pack('<e', value))


def read_f16(path, count):
    raw = Path(path).read_bytes()
    if len(raw) != count * 2:
        raise ValueError(f'{path}: expected {count * 2} bytes, found {len(raw)}')
    return [x[0] for x in struct.iter_unpack('<e', raw)]


def oracle_row(q, keys, values, valid_keys, scale):
    """Independent stable attention row; only valid keys enter the denominator."""
    logits = [sum(a * b for a, b in zip(q, key)) * scale for key in keys[:valid_keys]]
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    total = sum(weights)
    return [sum(weights[i] * values[i][d] for i in range(valid_keys)) / total
            for d in range(len(values[0]))]


def checked_errors(actual, expected, atol=0.1, rtol=1e-5):
    if len(actual) != len(expected) or not actual:
        raise ValueError('output/oracle lengths must be equal and nonempty')
    maximum = squared = 0.0
    failures = 0
    for a, e in zip(actual, expected):
        if not math.isfinite(a) or not math.isfinite(e):
            raise ValueError('nonfinite output or oracle; timing is forbidden')
        error = abs(a - e)
        maximum = max(maximum, error)
        squared += error * error
        failures += error > atol + rtol * abs(e)
    if failures:
        raise ValueError(f'{failures}/{len(actual)} values fail: max_abs={maximum}, atol={atol}')
    return {'max_abs': maximum, 'rms': math.sqrt(squared / len(actual)),
            'atol': atol, 'rtol': rtol, 'elements': len(actual)}


def checked_tensor_errors(actual, expected, atol):
    import torch
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('invalid output shape or nonfinite output; timing is forbidden')
    delta = (actual.float() - expected.float()).abs()
    if (delta > atol + expected.float().abs() * 1e-5).any():
        raise ValueError(f'output fails oracle: max_abs={delta.max().item()}, atol={atol}')
    return {'max_abs': delta.max().item(), 'rms': delta.square().mean().sqrt().item(),
            'atol': atol, 'rtol': 1e-5, 'elements': delta.numel()}


def fixture(config, directory, seed, pattern):
    import numpy as np
    rng = np.random.default_rng(seed)
    shapes = {'q': (config.batch, config.query, config.heads, 128),
              'k': (config.batch, config.keys, config.kv_heads, 128),
              'v': (config.batch, config.keys, config.kv_heads, 128)}
    arrays = {}
    for name, shape in shapes.items():
        if pattern == 'uniform':
            a = np.zeros(shape, dtype=np.float16) if name != 'v' else rng.integers(-2, 3, shape).astype(np.float16)
        else:
            a = (rng.standard_normal(shape) * 0.5).astype(np.float16)
        a.astype('<f2', copy=False).tofile(directory / f'{name}.f16')
        arrays[name] = a
    return arrays


def mathematical_oracle(config, arrays, device):
    """Chunked FP32 Torch reference independent of either kernel's online schedule."""
    import torch
    torch.set_float32_matmul_precision('highest')
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('the independent oracle requires full FP32 matmul with TF32 disabled')
    q, k, v = [torch.from_numpy(arrays[n]).to(device=device, dtype=torch.float32)
               for n in ['q', 'k', 'v']]
    output = torch.empty_like(q)
    scale = 1.0 / math.sqrt(128)
    for b in range(config.batch):
        for h in range(config.heads):
            kh = config.kv_head(h)
            for start in range(0, config.query, 128):
                end = min(start + 128, config.query)
                scores = q[b, start:end, h] @ k[b, :, kh].T * scale
                if config.causal:
                    rows = torch.arange(start, end, device=device)[:, None]
                    cols = torch.arange(config.keys, device=device)[None, :]
                    scores.masked_fill_(cols > rows, -torch.inf)
                output[b, start:end, h] = torch.softmax(scores, dim=1) @ v[b, :, kh]
    return output.cpu()


def sm100_fmha_prefill_cutedsl(config, arrays):
    """Compile the untouched public reference on the exact supplied FP16 bytes."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cuda.bindings.driver as cuda
    import torch
    from cutlass.cute.runtime import from_dlpack
    if cutlass.__version__ != '4.7.0':
        raise RuntimeError('reference requires pinned CUTLASS 4.7.0 runtime')
    verify_sources()
    sys.path.insert(0, str(ROOT / 'reference'))
    spec = importlib.util.spec_from_file_location('pinned_sm100_fmha', SOURCE)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    tensors = [torch.from_numpy(arrays[n]).cuda() for n in ['q', 'k', 'v']]
    output = torch.full_like(tensors[0], math.nan)
    hratio = config.heads // config.kv_heads
    views = [tensors[0].view(config.batch, config.query, config.kv_heads, hratio, 128),
             tensors[1].view(config.batch, config.keys, config.kv_heads, 1, 128),
             tensors[2].view(config.batch, config.keys, config.kv_heads, 1, 128),
             output.view(config.batch, config.query, config.kv_heads, hratio, 128)]
    cute_tensors = [from_dlpack(x, assumed_align=32).mark_layout_dynamic(leading_dim=4)
                    for x in views]
    mask = example.fmha_utils.MaskEnum.WINDOW_MASK
    if not config.causal and config.keys % 128:
        mask = example.fmha_utils.MaskEnum.RESIDUAL_MASK
    fmha = example.BlackwellFusedMultiHeadAttentionForward(
        cutlass.Float32, cutlass.Float32, (128, 128), 128, True, mask, False, False)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    problem = (config.batch, config.query, config.query, config.keys, config.heads,
               config.kv_heads, 128, 128)
    scale = 1.0 / math.sqrt(128)
    args = (*cute_tensors, problem, None, None, None, None,
            scale * math.log2(math.e), scale, 1.0, None, None,
            cutlass.Int32(0) if config.causal else None, None, None, stream, False)
    torch.cuda.synchronize()
    compiled = cute.compile(fmha, *args, options='--opt-level=2')
    def launch():
        compiled(*args)
    launch()
    torch_stream.synchronize()
    return launch, output, torch_stream, (compiled, args, tensors, views, cute_tensors)


def sample_reference(launch, stream, warmup, iterations, windows):
    import torch
    samples = []
    for _ in range(windows):
        for _ in range(warmup):
            launch()
        stream.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iterations):
            launch()
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return samples


def gpu_metadata():
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.current.memory,power.draw,power.limit,utilization.gpu', '--format=csv,noheader,nounits']
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {'query': command[1], 'output': result.stdout.strip(), 'error': result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust-bin', type=Path, required=True)
    parser.add_argument('--shape', default='1,1024,1024,8,8', help='B,Sq,Sk,Hq,Hkv; D=128')
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--pattern', choices=['gaussian', 'uniform'], default='gaussian')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--sanitizer', choices=['memcheck', 'synccheck'])
    args = parser.parse_args()
    require_reference_environment()
    if args.sanitizer:
        args.check_only = True
    shape = [int(v) for v in args.shape.split(',')]
    if len(shape) != 5:
        parser.error('--shape requires B,Sq,Sk,Hq,Hkv')
    config = Config(*shape, causal=args.causal).validate()
    if args.warmup < 0 or args.iters < 1 or args.windows < 1 or args.timeout < 1:
        parser.error('warmup must be nonnegative; iters, windows, timeout positive')
    if not args.rust_bin.is_file():
        parser.error(f'Rust executable does not exist: {args.rust_bin}')
    with redirect_stdout(sys.stderr):
        report = run_case(args, config)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    timing = report['timing']
    for label, key in [('CuTeDSL', 'sm100_fmha_prefill_cutedsl'),
                       ('CuTe-RS', 'sm100_fmha_prefill_cute_rs')]:
        if timing is None:
            print(f'{label}: PASS')
        else:
            print(f'{label}: {timing[key]["median_us"]:.3f} us')


def run_case(args, config):
    import numpy as np
    import torch
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError('this matched comparison requires SM100 (B200)')
    with tempfile.TemporaryDirectory(prefix='sm100-fmha-prefill-') as tmp:
        directory = Path(tmp)
        arrays = fixture(config, directory, args.seed, args.pattern)
        expected = mathematical_oracle(config, arrays, 'cuda')
        launch, output, stream, keepalive = sm100_fmha_prefill_cutedsl(config, arrays)
        atol = 0.005 if args.pattern == 'uniform' else 0.1
        reference_output = output.cpu().float()
        reference_errors = checked_tensor_errors(reference_output, expected, atol)
        expected.numpy().astype('<f4').tofile(directory / 'oracle.f32')
        binary = str(args.rust_bin.resolve())
        base = [binary, *config.cli(), '--input-dir', str(directory),
                '--output', str(directory / 'rust.f16'), '--json', str(directory / 'rust.json'),
                '--oracle', str(directory / 'oracle.f32'), '--atol', str(atol)]
        command = [*base, '--check-only']
        if args.sanitizer:
            command = ['compute-sanitizer', '--tool', args.sanitizer, '--error-exitcode', '99', *command]
        subprocess.run(command, check=True, timeout=args.timeout,
                       stdout=sys.stderr if args.sanitizer else subprocess.PIPE)
        rust_check = json.loads((directory / 'rust.json').read_text())
        rust_output = torch.from_numpy(np.fromfile(directory / 'rust.f16', dtype='<f2').reshape(expected.shape)).float()
        rust_errors = checked_tensor_errors(rust_output, expected, atol)
        cross_errors = checked_tensor_errors(rust_output, reference_output, atol)
        report = {'source_commit': COMMIT, 'source_sha256': SOURCE_SHA256, 'source_closure_sha256': SOURCE_DIGESTS,
                  'reference_compiler_optimization_level': 2, 'sanitizer': args.sanitizer,
                  'rust_executable_sha256': hashlib.sha256(args.rust_bin.read_bytes()).hexdigest(),
                  'config': asdict(config), 'seed': args.seed, 'pattern': args.pattern,
                  'oracle': {'dtype': 'float32', 'matmul_precision': torch.get_float32_matmul_precision(),
                             'allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'query_chunk': 128},
                  'rust_kernel': rust_check,
                  'validation': {'sm100_fmha_prefill_cute_rs': rust_errors,
                                 'sm100_fmha_prefill_cutedsl': reference_errors,
                                 'cross_implementation': cross_errors},
                  'timing': None}
        if not args.check_only:
            require_matched_optimization(rust_check)
            before = gpu_metadata()
            reference_samples = sample_reference(launch, stream, args.warmup, args.iters, args.windows)
            subprocess.run([*base, '--warmup', str(args.warmup), '--iters', str(args.iters),
                            '--windows', str(args.windows)], check=True, stdout=subprocess.PIPE, timeout=args.timeout)
            rust_result = json.loads((directory / 'rust.json').read_text())
            after = torch.from_numpy(np.fromfile(directory / 'rust.f16', dtype='<f2').reshape(expected.shape)).float()
            checked_tensor_errors(after, expected, atol)
            checked_tensor_errors(output.cpu().float(), expected, atol)
            report['timing'] = {'mode': 'direct-cuda-events', 'cache': 'hot',
                'measurement_scope': 'complete invocation throughput including host submission gaps',
                'warmup': args.warmup, 'iterations': args.iters, 'windows': args.windows,
                'sm100_fmha_prefill_cutedsl': {'samples_us': reference_samples,
                                             'median_us': statistics.median(reference_samples)},
                'sm100_fmha_prefill_cute_rs': rust_result,
                'gpu_before': before, 'gpu_after': gpu_metadata()}
        return report


if __name__ == '__main__':
    main()
