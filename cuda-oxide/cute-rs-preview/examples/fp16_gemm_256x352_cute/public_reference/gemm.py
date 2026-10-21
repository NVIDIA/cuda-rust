# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original SM100 FP16 GEMM reference for cuda-oxide.

The two CTAs compute 256x352 outputs using 256x192 and 256x160 MMAs.
This module uses the public, experimental CUTLASS Python DSL 4.7.0 APIs.
Inputs and output remain owned by the prepared callable until it is released.
See the sibling documentation for sources and the hardware validation status.
"""

import functools
import importlib.metadata
import operator

import cutlass
from cutlass import cute, utils
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor
from cutlass.experimental import cuda, primitives as prim
from cuda.bindings import driver

# Byte offsets describe disjoint regions in each CTA's shared allocation.
_STAGES = 5
_A_BYTES = 128 * 64 * 2
_B0_BYTES = 96 * 64 * 2
_B1_BYTES = 80 * 64 * 2
_B_OFFSET = _STAGES * _A_BYTES
_C_OFFSET = _B_OFFSET + _STAGES * (_B0_BYTES + _B1_BYTES)
_C_BYTES = 128 * 32 * 2
_CONTROL = _C_OFFSET + 2 * _C_BYTES
SMEM_BYTES = _CONTROL + 128
assert SMEM_BYTES == 211072
_GROUP = prim.CTAGroup.CTA_2


@cute.jit
def _raw(ptr: cute.Pointer):
    """Adapt a CuTe pointer through public APIs for experimental primitives."""
    return cutlass.Pointer(ptr.to_llvm_ptr(), dtype=ptr.dtype, space=ptr.memspace)


@cute.jit
def _wait(bar: cutlass.Pointer, phase: cutlass.Int32, cluster: cutlass.Constexpr = False):
    """Acquire a completed barrier phase; remote epilogue arrivals need cluster scope."""
    if cutlass.const_expr(cluster):
        while not prim.mbarrier_try_wait_parity(
            bar, phase, scope=prim.MBarrierScope.CLUSTER, order=prim.MemOrder.ACQUIRE
        ):
            pass
    else:
        while not prim.mbarrier_try_wait_parity(bar, phase):
            pass


@cute.jit
def _operand(base: cute.Pointer):
    """Describe a 64-wide K-major FP16 tile, with 128-byte swizzle and K16 steps."""
    return prim.Tcgen05SmemDesc.build(
        _raw(base), leading_byte_offset=16, stride_byte_offset=8 * 64 * 2,
        layout=prim.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )


class _Gemm:
    """Static launch configuration; descriptors derive their bounds from tensor views."""

    def __init__(self, has_bias, clusters):
        self.has_bias = has_bias
        self.clusters = clusters

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
                 bias: cute.Tensor, stream: driver.CUstream):
        """Build four checked tensor maps and enqueue exactly one clustered kernel."""
        ad = cuda.create_tensor_map_tiled_from_view(
            a, (128, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        bd0 = cuda.create_tensor_map_tiled_from_view(
            b, (96, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        bd1 = cuda.create_tensor_map_tiled_from_view(
            b, (80, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        cd = cuda.create_tensor_map_tiled_from_view(
            c, (128, 32), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s64b
        )
        assert ad.shared_storage_bytes == _A_BYTES
        assert bd0.shared_storage_bytes == _B0_BYTES
        assert bd1.shared_storage_bytes == _B1_BYTES
        assert cd.shared_storage_bytes == _C_BYTES
        self.kernel(ad, bd0, bd1, cd, bias, a.shape[0], c.shape[1], a.shape[1]).launch(
            grid=(2 * self.clusters, 1, 1), block=(192, 1, 1), cluster=(2, 1, 1),
            smem=SMEM_BYTES, stream=stream,
        )

    @cute.kernel
    def kernel(self, ad: cutlass.GridConstant[cuda.TensorMap],
               bd0: cutlass.GridConstant[cuda.TensorMap],
               bd1: cutlass.GridConstant[cuda.TensorMap],
               cd: cutlass.GridConstant[cuda.TensorMap], bias: cute.Tensor,
               m: cutlass.Constexpr, n: cutlass.Constexpr, k: cutlass.Constexpr):
        """Five-stage TMA/MMA ring with a separate, two-stage TMA store epilogue.

        Each role executes convergently within its assigned warps. Only CTA 0's
        elected MMA lane issues MMAs. Every epilogue warp participates in TMEM
        loads; all lanes of warp 0 participate in allocation and deallocation.
        """
        tid, _, _ = cute.arch.thread_idx()
        warp = tid // 32
        rank = prim.cluster_ctarank()
        base = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
        full = _raw(cute.recast_ptr(base + _CONTROL, dtype=cutlass.Int64))
        empty = full + _STAGES
        acc_empty, acc_full, dealloc = full + 10, full + 11, full + 12
        token = cutlass.Pointer((full + 13).to_llvm_ptr(), dtype=cutlass.Int32, space=cutlass.AddressSpace.smem)
        if tid == 0:
            for slot in cutlass.range_constexpr(_STAGES):
                prim.mbarrier_init(full + slot, 1)
                prim.mbarrier_init(empty + slot, 1)
            prim.mbarrier_init(acc_empty, 2)
            prim.mbarrier_init(acc_full, 1)
            prim.mbarrier_init(dealloc, 32)
        prim.fence_mbarrier_init()
        prim.barrier_cluster_arrive_relaxed_aligned()
        prim.barrier_cluster_wait_aligned()
        if warp == 4:
            if prim.elect_sync():
                prim.prefetch_tensormap(ad.get_ptr())
                prim.prefetch_tensormap(bd0.get_ptr())
                prim.prefetch_tensormap(bd1.get_ptr())
                prim.prefetch_tensormap(cd.get_ptr())

        first = cute.arch.block_idx()[0] // 2
        stride = cute.arch.grid_dim()[0] // 2
        mtiles = m // 256
        tile_count = mtiles * ((n + 351) // 352)
        ktiles = (k + 63) // 64

        if warp == 5:
            # Producers in both CTAs deliver distinct halves. CTA-group-2
            # redirects all six completion byte counts to CTA 0's full barrier.
            slot, phase, tile = cutlass.Int32(0), cutlass.Int32(1), first
            while tile < tile_count:
                row = (tile % mtiles) * 256 + rank * 128
                col = (tile // mtiles) * 352
                kt = cutlass.Int32(0)
                while kt < ktiles:
                    _wait(empty + slot, phase)
                    if prim.elect_sync():
                        if rank == 0:
                            transaction_bytes = 2 * (ad.global_tx_bytes() +
                                bd0.global_tx_bytes() + bd1.global_tx_bytes())
                            prim.mbarrier_arrive_expect_tx(full + slot, transaction_bytes)
                        prim.cp_async_bulk_tensor_shared_cluster_global(
                            _raw(base + slot * _A_BYTES), ad.get_ptr(), [kt * 64, row],
                            cutlass.Array(full + slot, (1,), dtype=cutlass.Int64), [], group=_GROUP,
                        )
                        bstage = base + _B_OFFSET + slot * (_B0_BYTES + _B1_BYTES)
                        prim.cp_async_bulk_tensor_shared_cluster_global(
                            _raw(bstage), bd0.get_ptr(), [kt * 64, col + rank * 96],
                            cutlass.Array(full + slot, (1,), dtype=cutlass.Int64), [], group=_GROUP,
                        )
                        prim.cp_async_bulk_tensor_shared_cluster_global(
                            _raw(bstage + _B0_BYTES), bd1.get_ptr(),
                            [kt * 64, col + 192 + rank * 80],
                            cutlass.Array(full + slot, (1,), dtype=cutlass.Int64), [], group=_GROUP,
                        )
                    slot += 1
                    if slot == _STAGES:
                        slot = 0
                        phase ^= 1
                    kt += 1
                tile += stride
            # Wait for the last used slot, accounting for a short first K tile
            # and a phase wrap. No producer can outlive its shared storage.
            for _ in cutlass.range_constexpr(_STAGES - 1):
                slot += 1
                if slot == _STAGES:
                    slot = 0
                    phase ^= 1
            _wait(empty + slot, phase)

        elif warp == 4:
            prim.barrier_cta_sync(2, thread_count=160)
            tmem = token.load()
            d0 = prim.make_tmem_ptr(tmem, cutlass.Float32)
            d1 = prim.make_tmem_ptr(tmem + 192, cutlass.Float32)
            id0 = prim.Tcgen05InstrDesc.build(
                a_dtype=cutlass.Float16, b_dtype=cutlass.Float16,
                c_dtype=cutlass.Float32, m_dim=256, n_dim=192,
            )
            id1 = prim.Tcgen05InstrDesc.build(
                a_dtype=cutlass.Float16, b_dtype=cutlass.Float16,
                c_dtype=cutlass.Float32, m_dim=256, n_dim=160,
            )
            slot, phase = cutlass.Int32(0), cutlass.Int32(0)
            released, tile = cutlass.Int32(1), first
            if rank == 0:
                while tile < tile_count:
                    _wait(acc_empty, released, True)
                    kt = cutlass.Int32(0)
                    while kt < ktiles:
                        _wait(full + slot, phase)
                        prim.tcgen05_fence(prim.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prim.elect_sync():
                            da = _operand(base + slot * _A_BYTES)
                            bstage = base + _B_OFFSET + slot * (_B0_BYTES + _B1_BYTES)
                            db0, db1 = _operand(bstage), _operand(bstage + _B0_BYTES)
                            for part in cutlass.range_constexpr(4):
                                use_d = kt > 0
                                if cutlass.const_expr(part > 0):
                                    use_d = cutlass.Boolean(True)
                                a16 = da.advance_start_address(part * 32)
                                prim.tcgen05_mma(
                                    prim.Tcgen05MMAKind.F16, _GROUP, d0, a16,
                                    db0.advance_start_address(part * 32), id0, use_d,
                                    collector_op=prim.Tcgen05MMACollectorOp.FILL,
                                )
                                prim.tcgen05_mma(
                                    prim.Tcgen05MMAKind.F16, _GROUP, d1, a16,
                                    db1.advance_start_address(part * 32), id1, use_d,
                                    collector_op=prim.Tcgen05MMACollectorOp.LASTUSE,
                                )
                            prim.tcgen05_commit(empty + slot, multicast_mask=3, group=_GROUP)
                        slot += 1
                        if slot == _STAGES:
                            slot = 0
                            phase ^= 1
                        kt += 1
                    if prim.elect_sync():
                        prim.tcgen05_commit(acc_full, multicast_mask=3, group=_GROUP)
                    released ^= 1
                    tile += stride
                _wait(acc_empty, released, True)
        else:
            if warp == 0:
                prim.tcgen05_alloc(token, 512, group=_GROUP)
                prim.tcgen05_relinquish_alloc_permit(group=_GROUP)
            prim.barrier_cta_sync(2, thread_count=160)
            tmem = token.load()
            # CuTe derives the physical shared addresses from the same B64
            # layout represented by the output tensor map, without XOR formulas.
            shared_layout = cute.make_composed_layout(
                cute.make_swizzle(2, 3, 3), 0,
                cute.make_layout((128, 32), stride=(32, 1)),
            )
            phase, output_slot, tile = cutlass.Int32(0), cutlass.Int32(0), first
            while tile < tile_count:
                row = (tile % mtiles) * 256 + rank * 128
                col = (tile // mtiles) * 352
                row_bias = cutlass.Float32(0)
                if cutlass.const_expr(self.has_bias):
                    row_bias = bias[row + tid].to(cutlass.Float32)
                _wait(acc_full, phase)
                prim.tcgen05_fence(prim.Tcgen05Fence.AFTER_THREAD_SYNC)
                for piece in range(11):
                    output_slot ^= 1
                    dst = base + _C_OFFSET + output_slot * _C_BYTES
                    # A 32x32b.x32 load gives each lane 32 columns of one row.
                    address = tmem + (warp * 32 << 16) + piece * 32
                    source = prim.make_tmem_ptr(address, cutlass.Float32)
                    values = prim.tcgen05_ld(prim.Tcgen05LdStShape.SHAPE_32X32B, source, num=32)
                    prim.tcgen05_wait(prim.Tcgen05Wait.LOAD)
                    narrowed = cute.make_rmem_tensor(32, cutlass.Float16)
                    for j in cutlass.range_constexpr(32):
                        narrowed[j] = (values[j] + row_bias).to(cutlass.Float16)
                    shared = cute.make_tensor(cute.recast_ptr(dst, dtype=cutlass.Float16), shared_layout)
                    row_view = shared[tid, None]
                    row_view = cute.make_tensor(row_view.iterator.align(16), row_view.layout)
                    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16,
                        num_bits_per_copy=128), narrowed, row_view)
                    prim.fence_proxy(prim.Proxy.ASYNC_SHARED, space=prim.SharedSpace.shared_cta)
                    prim.barrier_cta_sync(1, thread_count=128)
                    if tid == 0:
                        prim.cp_async_bulk_tensor_global_shared_cta(
                            cd.get_ptr(), _raw(dst), [col + piece * 32, row]
                        )
                        prim.cp_async_bulk_commit_group()
                        prim.cp_async_bulk_wait_group(1, read=True)
                    prim.barrier_cta_sync(1, thread_count=128)
                prim.tcgen05_fence(prim.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prim.barrier_cta_sync(1, thread_count=128)
                if tid == 0:
                    # One arrival per CTA, after all four epilogue warps drain.
                    peer_empty = prim.mapa(acc_empty, 0)
                    prim.mbarrier_arrive(peer_empty, scope=prim.MemScope.CLUSTER)
                phase ^= 1
                tile += stride
            if warp == 0:
                if tid == 0:
                    prim.cp_async_bulk_wait_group(0, read=True)
                prim.barrier_cta_sync(3, thread_count=32)
                peer_dealloc = prim.mapa(dealloc, rank ^ 1)
                prim.mbarrier_arrive(peer_dealloc, scope=prim.MemScope.CLUSTER)
                _wait(dealloc, cutlass.Int32(0), True)
                pointer = prim.make_tmem_ptr(tmem, cutlass.Float32)
                prim.tcgen05_dealloc(pointer, 512, group=_GROUP)


def _dimensions(m, n, k):
    """Check positive signed-32-bit dimensions and an overflow-safe tile schedule."""
    dims = tuple(operator.index(x) for x in (m, n, k))
    if any(isinstance(x, bool) for x in (m, n, k)) or any(x <= 0 or x >= 2**31 for x in dims):
        raise ValueError('M, N, and K must be positive signed-32-bit integers')
    m, n, k = dims
    if m % 256 or n % 8 or k % 8:
        raise ValueError('M must be a multiple of 256; N and K must be multiples of 8')
    if ((n + 351) // 352 - 1) * 352 + 320 >= 2**31:
        raise ValueError('padded N coordinates must fit signed 32 bits')
    if (m // 256) * ((n + 351) // 352) >= 2**31:
        raise ValueError('persistent tile count must fit signed 32 bits')
    return dims


def _version():
    """Reject unvalidated API versions rather than silently changing code generation."""
    if importlib.metadata.version('nvidia-cutlass-dsl') != '4.7.0':
        raise RuntimeError('this reference requires nvidia-cutlass-dsl==4.7.0')


@functools.lru_cache(maxsize=128)
def compile_kernel(m, n, k, has_bias=False, clusters=1):
    """Compile fake tensor contracts for SM100 without launching or needing SM100 hardware."""
    _version()
    m, n, k = _dimensions(m, n, k)
    clusters = operator.index(clusters)
    if clusters <= 0 or clusters > (2**31 - 1) // 2 or clusters > (m // 256) * ((n + 351) // 352):
        raise ValueError('clusters must be positive and no greater than the output tile count')
    if (m // 256) * ((n + 351) // 352) + clusters - 1 >= 2**31:
        raise ValueError('persistent tile advancement must fit signed 32 bits')
    args = (
        make_fake_tensor(cutlass.Float16, (m, k), (k, 1), assumed_align=16),
        make_fake_tensor(cutlass.Float16, (n, k), (k, 1), assumed_align=16),
        make_fake_tensor(cutlass.Float16, (m, n), (n, 1), assumed_align=16),
        make_fake_tensor(cutlass.Float16, (m,), (1,), assumed_align=16),
        make_fake_stream(),
    )
    return cute.compile(_Gemm(bool(has_bias), clusters), *args, options='--gpu-arch sm_100a')


def _sm100(device):
    """Check the actual launch target; offline compilation intentionally skips this check."""
    import torch
    if torch.device(device).type != 'cuda' or torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError('this reference requires an SM100 GPU')


def cluster_capacity():
    """Return the public hardware-query capacity for two-CTA clusters on the current SM100 GPU."""
    import torch
    device = torch.cuda.current_device()
    _sm100(device)
    return int(utils.HardwareInfo(device).get_max_active_clusters(2))


def _arguments(tensors, has_bias):
    """Validate Torch storage, exact strides, alignment, and common CUDA device."""
    import torch
    if len(tensors) != (4 if has_bias else 3):
        raise TypeError('expected A, B, bias, C' if has_bias else 'expected A, B, C')
    a, b, c = tensors[0], tensors[1], tensors[-1]
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError('arguments must be torch.Tensor objects')
    if a.ndim != 3 or b.ndim != 3 or c.ndim != 3 or a.shape[0] != 1:
        raise ValueError('expected A[1,M,K], B[1,K,N], and C[1,M,N]')
    m, n, k = _dimensions(a.shape[1], b.shape[2], a.shape[2])
    if tuple(b.shape) != (1, k, n) or tuple(c.shape) != (1, m, n):
        raise ValueError('matrix shapes do not match')
    expected = [(m * k, k, 1), (n * k, 1, k), (m * n, n, 1)]
    if any(tuple(t.stride()) != s for t, s in zip((a, b, c), expected)):
        raise ValueError('A/C must be contiguous; B must transpose contiguous [1,N,K]')
    if has_bias and (tuple(tensors[2].shape) != (m,) or tuple(tensors[2].stride()) != (1,)):
        raise ValueError('bias must be contiguous with shape [M]')
    if any(t.dtype != torch.float16 for t in tensors):
        raise ValueError('all tensors must be FP16 tensors')
    if any(t.device != a.device for t in tensors) or any(t.data_ptr() % 16 for t in tensors):
        raise ValueError('tensors must share a device and have 16-byte-aligned addresses')
    # Output must not overlap any input. Check the exact dense storage spans.
    clo, chi = c.data_ptr(), c.data_ptr() + c.numel() * 2
    for t in tensors[:-1]:
        if max(clo, t.data_ptr()) < min(chi, t.data_ptr() + t.numel() * 2):
            raise ValueError('output must not overlap an input')
    _sm100(a.device)
    return m, n, k


def prepare_callable(has_bias=False, *, inputs=None, output=None, clusters=None):
    """Prepare ``call(A, B, [bias,] C, stream=...)`` with owned, cached DLPack views.

    Supplying inputs/output performs validation, fake-tensor JIT compilation,
    and view conversion immediately, with zero GPU kernel launches. Repeated
    calls with the same tensor objects reuse those views during graph capture.
    ``stream`` accepts a Torch CUDA stream or an explicit driver CUstream.
    A raw CUstream must belong to the tensors' device; Torch streams are checked.
    """
    _version()
    cache = {}

    def fingerprint(tensors):
        """Snapshot CPU storage metadata so mutable Tensor objects cannot reuse stale views."""
        return tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                     for t in tensors)

    def bind(tensors):
        """Own tensors and their 2-D DSL views for the lifetime of this callable."""
        import torch
        key = tuple(id(t) for t in tensors)
        if key in cache:
            if fingerprint(tensors) != cache[key][3]:
                raise ValueError('a prepared tensor changed its storage, shape, strides, dtype, or device')
        else:
            m, n, k = _arguments(tensors, has_bias)
            with torch.cuda.device(tensors[0].device):
                capacity = int(utils.HardwareInfo(tensors[0].device.index).get_max_active_clusters(2))
            requested = capacity if clusters is None else operator.index(clusters)
            if requested <= 0:
                raise ValueError('clusters must be positive')
            count = min(requested, capacity, (m // 256) * ((n + 351) // 352))
            with torch.cuda.device(tensors[0].device):
                compiled = compile_kernel(m, n, k, has_bias, count)
            a, b, c = tensors[0], tensors[1], tensors[-1]
            bias = tensors[2] if has_bias else a.reshape(-1)[:m]
            views = tuple(from_dlpack(t, assumed_align=16) for t in (a[0], b[0].T, c[0], bias))
            cache[key] = (tensors, views, compiled, fingerprint(tensors))
        return cache[key]

    def call(*tensors, stream=None):
        """Enqueue one GEMM on the explicitly requested stream, or the device's current stream."""
        import torch
        owners, views, compiled, _ = bind(tensors)
        if stream is None:
            stream = torch.cuda.current_stream(owners[0].device)
        if isinstance(stream, torch.cuda.Stream):
            if stream.device != owners[0].device:
                raise ValueError('stream and tensors must share a CUDA device')
            stream = driver.CUstream(stream.cuda_stream)
        if not isinstance(stream, driver.CUstream):
            raise TypeError('stream must be torch.cuda.Stream or cuda.bindings.driver.CUstream')
        with torch.cuda.device(owners[0].device):
            compiled(*views, stream)
        return tensors[-1]

    if (inputs is None) != (output is None):
        raise ValueError('inputs and output must be supplied together')
    if inputs is not None:
        bind((*inputs, output))
    return call
