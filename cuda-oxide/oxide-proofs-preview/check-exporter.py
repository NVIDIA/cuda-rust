#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the MIR exporter and Lean together on scalar and vecadd MIR fixtures."""

from pathlib import Path
import subprocess
import sys
import tempfile

from check import ROOT, require_success, run_lake

REPO = ROOT.parent
SCALAR_FIXTURE = REPO / "crates/lean-exporter/fixtures/arithmetic.mir"
VECADD_FIXTURE = REPO / "crates/lean-exporter/fixtures/vecadd.mir"
SGEMM_TILED_FIXTURE = REPO / "crates/lean-exporter/fixtures/sgemm_tiled.mir"
WARP_REDUCE_FIXTURE = REPO / "crates/lean-exporter/fixtures/warp_reduce.mir"
SUBWARP_MULTIBUF_FIXTURE = REPO / "crates/lean-exporter/fixtures/subwarp_multibuf.mir"
ASYNC_COPY_FIXTURE = REPO / "crates/lean-exporter/fixtures/async_copy_stage.mir"
TCGEN05_MATMUL_FIXTURE = REPO / "crates/lean-exporter/fixtures/tcgen05_matmul.mir"
WARP_RING_FIXTURE = REPO / "crates/lean-exporter/fixtures/warp_ring_stage.mir"
CLUSTER_CLC_FIXTURE = REPO / "crates/lean-exporter/fixtures/cluster_clc_stage.mir"
PROOFS = """
namespace CudaOxide.Generated

theorem index_formula (inputs : Fin 3 → BitVec 64) :
    evaluate inputs =
      inputs ⟨0, by decide⟩ * inputs ⟨1, by decide⟩ + inputs ⟨2, by decide⟩ := by
  rfl

example : evaluate (fun i =>
    if i.val = 0 then 3 else if i.val = 1 then 256 else 255) =
    (1023 : BitVec 64) := by decide

-- The exported arithmetic must retain the source operation's wraparound.
example : evaluate (fun i =>
    if i.val = 0 then 18446744073709551615 else 1) =
    (0 : BitVec 64) := by decide

#print axioms index_formula
end CudaOxide.Generated
"""


def export(path, fn_name="global_index"):
    return subprocess.run(
        ["cargo", "run", "--quiet", "--package", "lean-exporter", "--",
         str(path), fn_name],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def require_export(path, fn_name="global_index"):
    result = export(path, fn_name)
    if result.returncode != 0:
        raise RuntimeError(f"Exporter failed for {path} ({fn_name}):\n{result.stderr}")
    return result.stdout


def main():
    require_success("build")
    source = SCALAR_FIXTURE.read_text()
    if source.count("mir.mul") != 1:
        raise RuntimeError("The arithmetic fixture must contain exactly one mir.mul")
    with tempfile.TemporaryDirectory(prefix="cuda-oxide-lean-export-") as directory:
        temporary = Path(directory)
        generated = require_export(SCALAR_FIXTURE)
        valid = temporary / "Valid.lean"
        valid.write_text(generated + PROOFS)
        require_success("env", "lean", str(valid))
        print("PASS: generated Lean proves the MIR arithmetic formula and wraparound.", flush=True)

        mutated_mir = temporary / "mutated.mir"
        mutated_mir.write_text(source.replace("mir.mul", "mir.add", 1))
        mutated = require_export(mutated_mir)
        if mutated == generated:
            raise RuntimeError("Changing a MIR operation did not change the generated Lean")
        invalid = temporary / "Mutated.lean"
        invalid.write_text(mutated + PROOFS.replace("#print axioms index_formula\n", ""))
        result = run_lake("env", "lean", str(invalid))
        if result.returncode == 0 or "Tactic `rfl` failed" not in result.stdout:
            raise RuntimeError(f"Expected the arithmetic formula proof to fail:\n{result.stdout}")
        print("REJECTED as expected: changed MIR no longer proves the original formula.", flush=True)

        unsupported_mir = temporary / "unsupported.mir"
        unsupported_mir.write_text(source.replace("mir.mul", "mir.div", 1))
        rejected = export(unsupported_mir)
        if rejected.returncode <= 0 or rejected.stdout or "mir.div" not in rejected.stderr:
            raise RuntimeError(f"Expected explicit rejection without Lean output:\n{rejected.stderr}")
        print("REJECTED as expected: unsupported MIR operation emits no Lean.", flush=True)

        # Verified vecadd kernel MIR checks
        vecadd_source = VECADD_FIXTURE.read_text()
        vecadd_lean = require_export(VECADD_FIXTURE, "vecadd")
        vecadd_valid = temporary / "VecaddValid.lean"
        vecadd_valid.write_text(vecadd_lean)
        require_success("env", "lean", str(vecadd_valid))
        print("PASS: exported vecadd MIR proves no_oob_idx, disjoint_writes, and writes_all_elems.", flush=True)

        # Mutate vecadd MIR by dropping the `0>=2` (`a.len() >= c.len()`) precondition:
        missing_req_mir = temporary / "vecadd_missing_req.mir"
        missing_req_mir.write_text(vecadd_source.replace("0>=2,1>=2", "1>=2", 1))
        missing_req_lean = require_export(missing_req_mir, "vecadd")
        missing_req_file = temporary / "VecaddMissingReq.lean"
        missing_req_file.write_text(
            "\n".join(line for line in missing_req_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_missing = run_lake("env", "lean", str(missing_req_file))
        if res_missing.returncode == 0 or "error: omega could not prove the goal:" not in res_missing.stdout:
            raise RuntimeError(f"Expected missing `a.len() >= c.len()` precondition to fail in Lean:\n{res_missing.stdout}")
        print("REJECTED as expected: removing `a.len() >= c.len()` breaks the `no_oob_idx(a)` proof.", flush=True)

        # Mutate vecadd MIR with an off-by-one neighbor access `a[idx + 1]` under `idx < c.len()`:
        off_by_one_mir = temporary / "vecadd_off_by_one.mir"
        off_by_one_mir.write_text(
            vecadd_source.replace(
                "a_elem = mir.ptr_offset (a_ptr, idx)",
                "one = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n"
                "    idx_plus_one = mir.add (idx, one) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n"
                "    a_elem = mir.ptr_offset (a_ptr, idx_plus_one)",
                1,
            )
        )
        off_by_one_lean = require_export(off_by_one_mir, "vecadd")
        off_by_one_file = temporary / "VecaddOffByOne.lean"
        off_by_one_file.write_text(
            "\n".join(line for line in off_by_one_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_oob = run_lake("env", "lean", str(off_by_one_file))
        if res_oob.returncode == 0 or "error: omega could not prove the goal:" not in res_oob.stdout:
            raise RuntimeError(f"Expected off-by-one `a[idx + 1]` access to fail in Lean:\n{res_oob.stdout}")
        print("REJECTED as expected: off-by-one `a[idx + 1]` access fails the `no_oob_idx(a)` proof.", flush=True)

        # Mutate vecadd MIR with an inclusive guard `mir.le (idx, c_len)` (`idx <= c.len()`):
        inclusive_mir = temporary / "vecadd_inclusive.mir"
        inclusive_mir.write_text(vecadd_source.replace("mir.lt (idx, c_len)", "mir.le (idx, c_len)", 1))
        rejected_inc = export(inclusive_mir, "vecadd")
        if rejected_inc.returncode == 0 or not rejected_inc.stderr:
            raise RuntimeError("Expected inclusive `idx <= c.len()` guard to be rejected by exporter")
        print("REJECTED as expected: inclusive `idx <= c.len()` guard is rejected before Lean.", flush=True)

        # Mutate vecadd MIR with an inner conditional branch that can skip writing `c` under `idx < c_len`:
        skip_write_mir = temporary / "vecadd_skip_write.mir"
        skip_write_mir.write_text(
            vecadd_source.replace(
                "    mir.store (c_elem, sum) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:RawMut>, builtin.fp32) -> ()>;\n    mir.goto () [^exit] []: <() -> ()>",
                "    skip_flag = mir.lt (c_len, idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n    mir.cond_br (skip_flag) [^do_write, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n  ^do_write():\n    mir.store (c_elem, sum) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:RawMut>, builtin.fp32) -> ()>;\n    mir.goto () [^exit] []: <() -> ()>",
                1,
            )
        )
        skip_write_lean = require_export(skip_write_mir, "vecadd")
        skip_write_file = temporary / "VecaddSkipWrite.lean"
        skip_write_file.write_text(
            "\n".join(line for line in skip_write_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_skip = run_lake("env", "lean", str(skip_write_file))
        if res_skip.returncode == 0:
            raise RuntimeError(f"Expected conditional skip under writes_all_elems to fail MirWritesAllElems in Lean:\n{res_skip.stdout}")
        print("REJECTED as expected: conditional branch skipping output write fails `MirWritesAllElems` in Lean 4.", flush=True)

        # Mutate vecadd MIR by returning early on thread 0 BEFORE the `< c.len()` guard:
        early_ret_source = vecadd_source.replace(
            "    in_bounds = mir.lt (idx, c_len)",
            "    zero_idx = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;\n"
            "    is_t0 = mir.eq (idx, zero_idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n"
            "    mir.cond_br (is_t0) [^exit, ^chk_bounds] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n"
            "  ^chk_bounds():\n"
            "    in_bounds = mir.lt (idx, c_len)",
            1,
        )
        early_ret_mir = temporary / "vecadd_early_ret.mir"
        early_ret_mir.write_text(early_ret_source)
        early_ret_lean = require_export(early_ret_mir, "vecadd")
        early_ret_file = temporary / "VecaddEarlyRet.lean"
        early_ret_file.write_text(
            "\n".join(line for line in early_ret_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_early = run_lake("env", "lean", str(early_ret_file))
        if res_early.returncode == 0:
            raise RuntimeError(f"Expected pre-guard early return on idx == 0 to fail MirWritesAllElems:\n{res_early.stdout}")
        print("REJECTED as expected: pre-guard `if idx == 0 { return; }` fails `MirWritesAllElems` in Lean 4.", flush=True)

        # Verified 2D phased shared-memory SGEMM checks
        sgemm_source = SGEMM_TILED_FIXTURE.read_text()
        sgemm_lean = require_export(SGEMM_TILED_FIXTURE, "sgemm_tiled_views")
        sgemm_valid = temporary / "SgemmValid.lean"
        sgemm_valid.write_text(sgemm_lean)
        require_success("env", "lean", str(sgemm_valid))
        print("PASS: exported 2D sgemm_tiled_views MIR proves no_oob_idx, no_data_race, and uniform_barriers.", flush=True)

        # Mutate sgemm_tiled_views MIR by removing the first `nvvm.barrier0` (`sync_threads()`):
        race_mir = temporary / "sgemm_missing_barrier.mir"
        race_mir.write_text(sgemm_source.replace("    nvvm.barrier0 () [] []: <() -> ()>;\n", "", 1))
        race_lean = require_export(race_mir, "sgemm_tiled_views")
        race_file = temporary / "SgemmRace.lean"
        race_file.write_text(
            "\n".join(line for line in race_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_race = run_lake("env", "lean", str(race_file))
        if res_race.returncode == 0 or "error: omega could not prove the goal:" not in res_race.stdout:
            raise RuntimeError(f"Expected missing `sync_threads()` to fail NoDataRace in Lean:\n{res_race.stdout}")
        print("REJECTED as expected: missing `sync_threads()` between write and read fails `no_data_race`.", flush=True)

        # Mutate sgemm_tiled_views MIR to add an init `nvvm.barrier0` before the loop while
        # dropping the second in-loop `nvvm.barrier0` (`^phase1_end`):
        init_one_bar_mir = temporary / "sgemm_init_one_loop_bar.mir"
        init_one_bar_mir.write_text(
            sgemm_source.replace(
                "mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
                "nvvm.barrier0 () [] []: <() -> ()>;\n        mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
                1,
            ).replace(
                "      ^phase1_end():\n        nvvm.barrier0 () [] []: <() -> ()>;",
                "      ^phase1_end():",
                1,
            )
        )
        init_one_bar_lean = require_export(init_one_bar_mir, "sgemm_tiled_views")
        init_one_bar_file = temporary / "SgemmInitOneBar.lean"
        init_one_bar_file.write_text(
            "\n".join(line for line in init_one_bar_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_init_one = run_lake("env", "lean", str(init_one_bar_file))
        if res_init_one.returncode == 0 or "error: omega could not prove the goal:" not in res_init_one.stdout:
            raise RuntimeError(f"Expected 1-barrier loop after init sync_threads() to fail NoDataRace in Lean:\n{res_init_one.stdout}")
        print("REJECTED as expected: 1-barrier loop after init `sync_threads()` fails `no_data_race`.", flush=True)

        # Verified 2D sgemm_tiled_raw global strip checks (RowStrip on `a`, ColStrip on `b`)
        sgemm_raw_lean = require_export(SGEMM_TILED_FIXTURE, "sgemm_tiled_raw")
        sgemm_raw_valid = temporary / "SgemmRawValid.lean"
        sgemm_raw_valid.write_text(sgemm_raw_lean)
        require_success("env", "lean", str(sgemm_raw_valid))
        print("PASS: exported 2D sgemm_tiled_raw MIR proves global RowStrip/ColStrip and shared-memory obligations.", flush=True)

        # Mutate sgemm_tiled_raw MIR to index `a` with an unguarded offset `a_len` (`a[a.len()]`):
        bad_strip_mir = temporary / "sgemm_bad_strip.mir"
        bad_strip_mir.write_text(
            sgemm_source.replace(
                "a_elem = mir.ptr_offset (a_ptr, a_idx)",
                "a_elem = mir.ptr_offset (a_ptr, a_len)",
                1,
            )
        )
        bad_strip_lean = require_export(bad_strip_mir, "sgemm_tiled_raw")
        bad_strip_file = temporary / "SgemmBadStrip.lean"
        bad_strip_file.write_text(
            "\n".join(line for line in bad_strip_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_bad_strip = run_lake("env", "lean", str(bad_strip_file))
        if res_bad_strip.returncode == 0 or "error: omega could not prove the goal:" not in res_bad_strip.stdout:
            raise RuntimeError(f"Expected OOB `a[a.len()]` 2D global slice access to fail in Lean:\n{res_bad_strip.stdout}")
        print("REJECTED as expected: OOB `a[a.len()]` 2D global slice access fails `MirNoOobGlobal` in Lean 4.", flush=True)

        # Mutate sgemm_tiled_raw MIR to check `a_col < k` instead of `a_col < a_eff` (ignoring band check):
        unguarded_strip_mir = temporary / "sgemm_unguarded_strip.mir"
        unguarded_strip_mir.write_text(
            SGEMM_TILED_FIXTURE.read_text().replace(
                "in_a = mir.lt (a_col, a_eff)",
                "in_a = mir.lt (a_col, k)",
                1,
            )
        )
        unguarded_strip_lean = require_export(unguarded_strip_mir, "sgemm_tiled_raw")
        unguarded_strip_file = temporary / "SgemmUnguardedStrip.lean"
        unguarded_strip_file.write_text(
            "\n".join(line for line in unguarded_strip_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_unguarded_strip = run_lake("env", "lean", str(unguarded_strip_file))
        if res_unguarded_strip.returncode == 0 or "error: omega could not prove the goal:" not in res_unguarded_strip.stdout:
            raise RuntimeError(f"Expected unguarded 2D global slice access (`a_col < k`) to fail in Lean:\n{res_unguarded_strip.stdout}")
        print("REJECTED as expected: unguarded 2D global slice access (`a_col < k`) fails `MirNoOobGlobal` in Lean 4.", flush=True)

        # Mutate sgemm_tiled_raw MIR with off-by-one `a[a_start + a_col + 1]`:
        oob_strip_mir = temporary / "sgemm_oob_strip.mir"
        oob_strip_mir.write_text(
            SGEMM_TILED_FIXTURE.read_text().replace(
                "a_elem = mir.ptr_offset (a_ptr, a_idx)",
                "a_idx_p1 = mir.add (a_idx, one_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        a_elem = mir.ptr_offset (a_ptr, a_idx_p1)",
                1,
            )
        )
        oob_strip_lean = require_export(oob_strip_mir, "sgemm_tiled_raw")
        oob_strip_file = temporary / "SgemmOobStrip.lean"
        oob_strip_file.write_text(
            "\n".join(line for line in oob_strip_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_oob_strip = run_lake("env", "lean", str(oob_strip_file))
        if res_oob_strip.returncode == 0 or "error: omega could not prove the goal:" not in res_oob_strip.stdout:
            raise RuntimeError(f"Expected off-by-one 2D strip `a[a_start + a_col + 1]` to fail NoOobRowStrip in Lean:\n{res_oob_strip.stdout}")
        print("REJECTED as expected: off-by-one 2D row-strip `a[a_start + a_col + 1]` fails `no_oob_idx`.", flush=True)

        # Stage 3: Verified warp & 2-phase block reductions
        warp_source = WARP_REDUCE_FIXTURE.read_text()
        warp_lean = require_export(WARP_REDUCE_FIXTURE, "warp_reduce_sum")
        warp_valid = temporary / "WarpReduceValid.lean"
        warp_valid.write_text(warp_lean)
        require_success("env", "lean", str(warp_valid))
        print("PASS: exported warp_reduce_sum MIR proves 1D warp butterfly reduction obligations.", flush=True)

        block_lean = require_export(WARP_REDUCE_FIXTURE, "block_reduce_sum")
        block_valid = temporary / "BlockReduceValid.lean"
        block_valid.write_text(block_lean)
        require_success("env", "lean", str(block_valid))
        print("PASS: exported block_reduce_sum MIR proves leader-guarded WARP_SUMS[tid / 32] and Phase 1 scan.", flush=True)

        # Mutate block_reduce_sum MIR by removing the `if lane == 0` guard:
        no_leader_mir = temporary / "block_reduce_no_leader.mir"
        no_leader_mir.write_text(
            warp_source.replace(
                "mir.cond_br (is_leader) [^leader_stage, ^barrier_sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
                "mir.goto () [^leader_stage] []: <() -> ()>",
                1,
            )
        )
        no_leader_lean = require_export(no_leader_mir, "block_reduce_sum")
        no_leader_file = temporary / "BlockReduceNoLeader.lean"
        no_leader_file.write_text(
            "\n".join(line for line in no_leader_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_no_leader = run_lake("env", "lean", str(no_leader_file))
        if res_no_leader.returncode == 0 or "error: omega could not prove the goal:" not in res_no_leader.stdout:
            raise RuntimeError(f"Expected missing `lane == 0` guard to fail NoDataRace in Lean:\n{res_no_leader.stdout}")
        print("REJECTED as expected: missing `lane == 0` guard before `WARP_SUMS[tid / 32]` fails `no_data_race`.", flush=True)

        # Mutate block_reduce_sum MIR to read `WARP_SUMS[w]` in `^exit` AFTER `while w < 8` exits:
        post_loop_mir = temporary / "block_reduce_post_loop.mir"
        post_loop_mir.write_text(
            warp_source.replace(
                "      ^exit():\n        mir.return () [] []: <() -> ()>\n    }\n}",
                "      ^exit():\n        bad_ws_raw = mir.shared_alloc () [] [elem_type: builtin.type builtin.fp32, size: builtin.integer <8: i64>, alloc_key: builtin.string \"WARP_SUMS\", source_name: builtin.string \"safe_warp_reduce::kernels::block_reduce_sum::WARP_SUMS\"]: <() -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;\n        bad_ws_ref = mir.cast (bad_ws_raw) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority Reborrow]: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>)>;\n        bad_post_ptr = mir.ptr_offset (bad_ws_ref, w) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>)>;\n        _bad_post_val = mir.load (bad_post_ptr) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>) -> (builtin.fp32)>;\n        mir.return () [] []: <() -> ()>\n    }\n}",
                1,
            )
        )
        post_loop_lean = require_export(post_loop_mir, "block_reduce_sum")
        post_loop_file = temporary / "BlockReducePostLoop.lean"
        post_loop_file.write_text(
            "\n".join(line for line in post_loop_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_post_loop = run_lake("env", "lean", str(post_loop_file))
        if res_post_loop.returncode == 0 or "error: omega could not prove the goal:" not in res_post_loop.stdout:
            raise RuntimeError(f"Expected post-loop `WARP_SUMS[w]` access (`!(w < 8)`) to fail in Lean:\n{res_post_loop.stdout}")
        print("REJECTED as expected: post-loop `WARP_SUMS[w]` access (`!(w < 8)`) fails `MirNoOobShared` in Lean 4.", flush=True)

        # Stage 4: Sub-warp bitwise coordinate splits & multi-buffer shared staging
        subwarp_source = SUBWARP_MULTIBUF_FIXTURE.read_text()
        split_lean = require_export(SUBWARP_MULTIBUF_FIXTURE, "subwarp_bit_split_stage")
        split_valid = temporary / "SubwarpBitSplitValid.lean"
        split_valid.write_text(split_lean)
        require_success("env", "lean", str(split_valid))
        print("PASS: exported subwarp_bit_split_stage MIR proves bit-split bijection and sub-warp lane scan.", flush=True)

        multibuf_lean = require_export(SUBWARP_MULTIBUF_FIXTURE, "multibuf_subwarp_stage")
        multibuf_valid = temporary / "MultibufSubwarpValid.lean"
        multibuf_valid.write_text(multibuf_lean)
        require_success("env", "lean", str(multibuf_valid))
        print("PASS: exported multibuf_subwarp_stage MIR proves 3-phase multi-buffer S_X_DIRECT/S_XS_DIRECT/S_ACT_DIRECT staging.", flush=True)

        # Mutate multibuf_subwarp_stage MIR by widening `if tid < 64` to `if tid < 128`:
        bad_guard_mir = temporary / "multibuf_bad_guard.mir"
        bad_guard_mir.write_text(
            subwarp_source.replace(
                "c64 = mir.constant () [] [value: builtin.integer <64: ui64>]",
                "c64 = mir.constant () [] [value: builtin.integer <128: ui64>]",
                1,
            )
        )
        bad_guard_lean = require_export(bad_guard_mir, "multibuf_subwarp_stage")
        bad_guard_file = temporary / "MultibufBadGuard.lean"
        bad_guard_file.write_text(
            "\n".join(line for line in bad_guard_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_bad_guard = run_lake("env", "lean", str(bad_guard_file))
        if res_bad_guard.returncode == 0 or "error: omega could not prove the goal:" not in res_bad_guard.stdout:
            raise RuntimeError(f"Expected widened `if tid < 128` guard to fail NoOobTileAccess in Lean:\n{res_bad_guard.stdout}")
        print("REJECTED as expected: widening `if tid < 64` to `if tid < 128` fails `no_oob_idx` on `S_XS_DIRECT` and `S_X_DIRECT`.", flush=True)

        # Mutate multibuf_subwarp_stage MIR by removing the second `nvvm.barrier0` (`^barrier_1`):
        missing_bar2_mir = temporary / "multibuf_missing_bar2.mir"
        missing_bar2_mir.write_text(
            subwarp_source.replace(
                "      ^barrier_1():\n        nvvm.barrier0 () [] []: <() -> ()>;",
                "      ^barrier_1():",
                1,
            )
        )
        missing_bar2_lean = require_export(missing_bar2_mir, "multibuf_subwarp_stage")
        missing_bar2_file = temporary / "MultibufMissingBar2.lean"
        missing_bar2_file.write_text(
            "\n".join(line for line in missing_bar2_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_missing_bar2 = run_lake("env", "lean", str(missing_bar2_file))
        if res_missing_bar2.returncode == 0 or "error: omega could not prove the goal:" not in res_missing_bar2.stdout:
            raise RuntimeError(f"Expected missing second barrier to fail NoDataRace on S_ACT_DIRECT in Lean:\n{res_missing_bar2.stdout}")
        print("REJECTED as expected: missing second `sync_threads()` before Phase 2 read of `S_ACT_DIRECT` fails `no_data_race`.", flush=True)

        # Mutate subwarp_bit_split_stage MIR with overlapping `tid | 1` (`mir.bitor`):
        bad_bitor_mir = temporary / "subwarp_bad_bitor.mir"
        bad_bitor_mir.write_text(
            subwarp_source.replace(
                "split_idx = mir.add (row_off, k_lane)",
                "c1_bad = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n        split_idx = mir.bitor (tid, c1_bad)",
                1,
            )
        )
        bad_bitor_lean = require_export(bad_bitor_mir, "subwarp_bit_split_stage")
        bad_bitor_file = temporary / "SubwarpBadBitor.lean"
        bad_bitor_file.write_text(
            "\n".join(line for line in bad_bitor_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_bad_bitor = run_lake("env", "lean", str(bad_bitor_file))
        if res_bad_bitor.returncode == 0 or "error: omega could not prove the goal:" not in res_bad_bitor.stdout:
            raise RuntimeError(f"Expected overlapping `tid | 1` bitor to fail MirNoSharedRace in Lean:\n{res_bad_bitor.stdout}")
        print("REJECTED as expected: overlapping `tid | 1` bitwise OR fails `MirNoSharedRace` in Lean 4.", flush=True)

        # Mutate subwarp_bit_split_stage MIR with narrowing `((tid * 2) as u8) as u64` wraparound:
        narrow_cast_mir = temporary / "subwarp_narrow_cast.mir"
        narrow_cast_mir.write_text(
            subwarp_source.replace(
                "size: builtin.integer <256: i64>",
                "size: builtin.integer <512: i64>",
                2,
            ).replace(
                "split_idx = mir.add (row_off, k_lane) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;",
                "c2_u64 = mir.constant () [] [value: builtin.integer <2: ui64>]: <() -> (builtin.integer ui64)>;\n        tid_x2 = mir.mul (tid, c2_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        trunc_u8 = mir.cast (tid_x2) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui64) -> (builtin.integer ui8)>;\n        split_idx = mir.cast (trunc_u8) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui8) -> (builtin.integer ui64)>;",
                1,
            )
        )
        narrow_cast_lean = require_export(narrow_cast_mir, "subwarp_bit_split_stage")
        narrow_cast_file = temporary / "SubwarpNarrowCast.lean"
        narrow_cast_file.write_text(
            "\n".join(line for line in narrow_cast_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_narrow = run_lake("env", "lean", str(narrow_cast_file))
        if res_narrow.returncode == 0 or "error: omega could not prove the goal:" not in res_narrow.stdout:
            raise RuntimeError(f"Expected narrowing `((tid * 2) as u8)` cast collision to fail NoDataRace in Lean:\n{res_narrow.stdout}")
        print("REJECTED as expected: narrowing `((tid * 2) as u8)` cast wraparound fails `no_data_race`.", flush=True)

        # Stage 5: Async TMA copy + mbarrier token ordering
        async_source = ASYNC_COPY_FIXTURE.read_text()
        async_lean = require_export(ASYNC_COPY_FIXTURE, "async_copy_stage")
        async_valid = temporary / "AsyncCopyValid.lean"
        async_valid.write_text(async_lean)
        require_success("env", "lean", str(async_valid))
        print("PASS: exported async_copy_stage MIR proves mbarrier token-ordered TMA copy and consumer read.", flush=True)

        # Mutate async_copy_stage MIR by bypassing `mbarrier_try_wait_parity`:
        no_wait_mir = temporary / "async_copy_no_wait.mir"
        no_wait_mir.write_text(
            async_source.replace(
                "mir.cond_br (not_ready) [^wait_spin, ^consume_tile] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
                "mir.goto () [^consume_tile] []: <() -> ()>",
                1,
            )
        )
        no_wait_lean = require_export(no_wait_mir, "async_copy_stage")
        no_wait_file = temporary / "AsyncCopyNoWait.lean"
        no_wait_file.write_text(
            "\n".join(line for line in no_wait_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_no_wait = run_lake("env", "lean", str(no_wait_file))
        if res_no_wait.returncode == 0 or "error: omega could not prove the goal:" not in res_no_wait.stdout:
            raise RuntimeError(f"Expected bypassing mbarrier_try_wait_parity to fail NoDataRace in Lean:\n{res_no_wait.stdout}")
        print("REJECTED as expected: reading `TILE` without `mbarrier_try_wait_parity` fails `no_data_race`.", flush=True)

        # Mutate async_copy_stage MIR by replacing the `mbarrier_try_wait_parity` loop with `sync_threads()`.
        # The copy and the read then sit in different barrier phases, so only `uniform_barriers` can refuse it:
        wait_loop = async_source[async_source.index("      ^wait_header():"):async_source.index("      ^consume_tile():")]
        sync_for_wait_mir = temporary / "async_copy_sync_for_wait.mir"
        sync_for_wait_mir.write_text(
            async_source.replace(
                wait_loop,
                "      ^wait_header():\n"
                "        nvvm.barrier0 () [] []: <() -> ()>;\n"
                "        mir.goto () [^consume_tile] []: <() -> ()>\n\n",
                1,
            )
        )
        sync_for_wait_lean = require_export(sync_for_wait_mir, "async_copy_stage")
        sync_for_wait_file = temporary / "AsyncCopySyncForWait.lean"
        sync_for_wait_file.write_text(
            "\n".join(line for line in sync_for_wait_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_sync_for_wait = run_lake("env", "lean", str(sync_for_wait_file))
        if (res_sync_for_wait.returncode == 0 or "checkUniformBarriersWith" not in res_sync_for_wait.stdout
                or "error: omega could not prove the goal:" in res_sync_for_wait.stdout):
            raise RuntimeError(f"Expected sync_threads in place of the mbarrier wait to fail only UniformBarriers in Lean:\n{res_sync_for_wait.stdout}")
        print("REJECTED as expected: `sync_threads()` in place of `mbarrier_try_wait_parity` before reading `TILE` fails `uniform_barriers`.", flush=True)

        # Mutate async_copy_stage MIR with oversized 32 KB TMA byte count into 16 KB `TILE`:
        oversized_tma_mir = temporary / "async_copy_oversized.mir"
        oversized_tma_mir.write_text(
            async_source.replace(
                "c16384_u32 = mir.constant () [] [value: builtin.integer <16384: ui32>]",
                "c16384_u32 = mir.constant () [] [value: builtin.integer <32768: ui32>]",
                1,
            )
        )
        oversized_tma_lean = require_export(oversized_tma_mir, "async_copy_stage")
        oversized_tma_file = temporary / "AsyncCopyOversized.lean"
        oversized_tma_file.write_text(
            "\n".join(line for line in oversized_tma_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_oversized = run_lake("env", "lean", str(oversized_tma_file))
        if res_oversized.returncode == 0 or "error: omega could not prove the goal:" not in res_oversized.stdout:
            raise RuntimeError(f"Expected oversized 32 KB TMA transfer into 16 KB TILE to fail NoOobTileAccess in Lean:\n{res_oversized.stdout}")
        print("REJECTED as expected: oversized TMA `expect_tx` byte count fails `no_oob_idx(TILE)`.", flush=True)

        # Stage 6: Blackwell tcgen05 + TMEM + PTX proxy coherence + mbarrier lifecycle
        tcgen05_source = TCGEN05_MATMUL_FIXTURE.read_text()
        tcgen05_lean = require_export(TCGEN05_MATMUL_FIXTURE, "tcgen05_matmul_verified")
        tcgen05_valid = temporary / "Tcgen05MatmulValid.lean"
        tcgen05_valid.write_text(tcgen05_lean)
        require_success("env", "lean", str(tcgen05_valid))
        print("PASS: exported tcgen05_matmul_verified MIR proves TMEM lifecycle, TMEM/SMEM bounds, warpgroup slices, and token ordering.", flush=True)

        # Stage 6 negative check 1: Missing `fence_proxy_async_shared_cta()` after `mbarrier_init` fails in Lean 4:
        no_proxy_mir = temporary / "tcgen05_no_proxy_fence.mir"
        no_proxy_mir.write_text(
            "\n".join(line for line in tcgen05_source.splitlines() if "fence_proxy_async_shared_cta" not in line)
        )
        no_proxy_lean = require_export(no_proxy_mir, "tcgen05_matmul_verified")
        no_proxy_file = temporary / "Tcgen05NoProxyFence.lean"
        no_proxy_file.write_text(
            "\n".join(line for line in no_proxy_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_no_proxy = run_lake("env", "lean", str(no_proxy_file))
        if res_no_proxy.returncode == 0:
            raise RuntimeError(f"Expected missing fence_proxy_async_shared_cta to fail TmemLifecycle in Lean:\n{res_no_proxy.stdout}")
        print("REJECTED as expected: missing `fence_proxy_async_shared_cta()` fails `TmemLifecycle` in Lean 4.", flush=True)

        # Stage 6 negative check 2: Out-of-order `tcgen05_commit` placed before `tcgen05_mma_f16` fails in Lean 4:
        commit_first_mir = temporary / "tcgen05_commit_before_mma.mir"
        commit_first_mir.write_text(
            tcgen05_source.replace(
                "        nvvm.tcgen05_mma_f16 (tmem_addr, a_desc, b_desc, idesc, false_i1) [] []: <(builtin.integer ui32, builtin.integer ui64, builtin.integer ui64, builtin.integer ui32, builtin.integer i1) -> ()>;\n        nvvm.tcgen05_commit_shared_cluster (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;",
                "        nvvm.tcgen05_commit_shared_cluster (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        nvvm.tcgen05_mma_f16 (tmem_addr, a_desc, b_desc, idesc, false_i1) [] []: <(builtin.integer ui32, builtin.integer ui64, builtin.integer ui64, builtin.integer ui32, builtin.integer i1) -> ()>;",
                1,
            )
        )
        commit_first_lean = require_export(commit_first_mir, "tcgen05_matmul_verified")
        commit_first_file = temporary / "Tcgen05CommitBeforeMma.lean"
        commit_first_file.write_text(
            "\n".join(line for line in commit_first_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_commit_first = run_lake("env", "lean", str(commit_first_file))
        if res_commit_first.returncode == 0:
            raise RuntimeError(f"Expected out-of-order tcgen05_commit before tcgen05_mma to fail in Lean:\n{res_commit_first.stdout}")
        print("REJECTED as expected: `tcgen05_commit` before `tcgen05_mma` fails `TmemLifecycle` in Lean 4.", flush=True)

        # Stage 6 negative check 3: 64-column TMEM allocation with 128-column MMA/ld overflows in Lean:
        tmem_overflow_mir = temporary / "tcgen05_col_overflow.mir"
        tmem_overflow_mir.write_text(
            tcgen05_source.replace(
                "c128_cols = mir.constant () [] [value: builtin.integer <128: ui32>]",
                "c128_cols = mir.constant () [] [value: builtin.integer <64: ui32>]",
            ).replace(
                "c128_dealloc = mir.constant () [] [value: builtin.integer <128: ui32>]",
                "c128_dealloc = mir.constant () [] [value: builtin.integer <64: ui32>]",
            )
        )
        tmem_overflow_lean = require_export(tmem_overflow_mir, "tcgen05_matmul_verified")
        tmem_overflow_file = temporary / "Tcgen05ColOverflow.lean"
        tmem_overflow_file.write_text(
            "\n".join(line for line in tmem_overflow_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_tmem_overflow = run_lake("env", "lean", str(tmem_overflow_file))
        if res_tmem_overflow.returncode == 0 or "error: omega could not prove the goal:" not in res_tmem_overflow.stdout:
            raise RuntimeError(f"Expected 128-col MMA/ld on 64-col TMEM allocation to fail NoOobTmemAccess in Lean:\n{res_tmem_overflow.stdout}")
        print("REJECTED as expected: 128-column `tcgen05` MMA/ld on 64-column `TMEM` fails `NoOobTmemAccess`.", flush=True)

        # `&mut [T]` positive + negative check: plain `c: &mut [f32]` with 64-bit hardware coordinates succeeds,
        # while racy `c[idx] = c[idx + 1]` on `c: &mut [f32]` fails `DisjointSliceAccessPair` in Lean 4:
        mut_slice_mir = temporary / "vecadd_mut_slice.mir"
        mut_slice_source = """mir.func @vecadd_mut_slice: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> ()>
  [contract_requires_len_ge: builtin.string "0>=2,1>=2", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "2", verify_no_oob_idx: builtin.string "0,1,2", verify_writes_all_elems: builtin.string "2"]
{
  ^entry(a: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, b: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, c: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>):
    bx32 = nvvm.read_ptx_sreg_ctaid_x () [] []: <() -> (builtin.integer ui32)>;
    bd32 = nvvm.read_ptx_sreg_ntid_x () [] []: <() -> (builtin.integer ui32)>;
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    bx = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    bd = mir.cast (bd32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    tx = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    base = mir.mul (bx, bd) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    idx = mir.add (base, tx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    c_len = mir.extract_field (c) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (builtin.integer ui64)>;
    in_bounds = mir.lt (idx, c_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_bounds) [^active, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^active():
    a_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    a_elem = mir.ptr_offset (a_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    a_val = mir.load (a_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    b_ptr = mir.extract_field (b) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    b_elem = mir.ptr_offset (b_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    b_val = mir.load (b_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;
    c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    c_elem = mir.ptr_offset (c_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    mir.store (c_elem, sum) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}
"""
        mut_slice_mir.write_text(mut_slice_source)
        mut_slice_lean = require_export(mut_slice_mir, "vecadd_mut_slice")
        mut_slice_file = temporary / "VecaddMutSlice.lean"
        mut_slice_file.write_text(mut_slice_lean)
        res_mut_slice = run_lake("env", "lean", str(mut_slice_file))
        if res_mut_slice.returncode != 0:
            raise RuntimeError(f"Expected plain &mut [f32] vecadd to verify in Lean:\n{res_mut_slice.stdout}")
        print("PASS: exported vecadd_mut_slice MIR (`c: &mut [f32]`) proves NoOobIdx, DisjointWrites, and WritesAllElems.", flush=True)

        racy_mut_mir = temporary / "vecadd_mut_racy.mir"
        racy_mut_mir.write_text(
            mut_slice_source.replace(
                "    c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;\n    c_elem = mir.ptr_offset (c_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;\n    mir.store (c_elem, sum) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;",
                "    c1_u64 = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n    idx_next = mir.add (idx, c1_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;\n    c_next_ptr = mir.ptr_offset (c_ptr, idx_next) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;\n    c_neighbor = mir.load (c_next_ptr) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>) -> (builtin.fp32)>;\n    sum2 = mir.add (sum, c_neighbor) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;\n    c_elem = mir.ptr_offset (c_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;\n    mir.store (c_elem, sum2) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;",
            )
        )
        racy_mut_lean = require_export(racy_mut_mir, "vecadd_mut_slice")
        racy_mut_file = temporary / "VecaddMutRacy.lean"
        racy_mut_file.write_text(
            "\n".join(line for line in racy_mut_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_racy_mut = run_lake("env", "lean", str(racy_mut_file))
        if res_racy_mut.returncode == 0:
            raise RuntimeError(f"Expected racy `c[idx] = ... + c[idx + 1]` on `&mut [f32]` to fail in Lean:\n{res_racy_mut.stdout}")
        print("REJECTED as expected: cross-thread `c[idx + 1]` read alongside `c[idx]` write on `&mut [f32]` fails in Lean 4.", flush=True)

        # Reverse-indexed `c[c.len() - 1 - idx]` positive + trapping `mir.assert` negative check:
        rev_mut_mir = temporary / "vecadd_mut_rev.mir"
        rev_mut_source = mut_slice_source.replace(
            "    c_elem = mir.ptr_offset (c_ptr, idx)",
            "    one_rev = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n    last_rev = mir.sub (c_len, one_rev) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    rev_idx = mir.sub (last_rev, idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    c_elem = mir.ptr_offset (c_ptr, rev_idx)",
            1,
        )
        rev_mut_mir.write_text(rev_mut_source)
        rev_mut_lean = require_export(rev_mut_mir, "vecadd_mut_slice")
        rev_mut_file = temporary / "VecaddMutRev.lean"
        rev_mut_file.write_text(rev_mut_lean)
        require_success("env", "lean", str(rev_mut_file))
        print("PASS: reverse-indexed `c[c.len() - 1 - idx]` proves MirNoOobGlobal, MirDisjointWrites, and MirWritesAllElems.", flush=True)

        trapping_wae_mir = temporary / "vecadd_trapping_wae.mir"
        trapping_wae_source = (
            rev_mut_source
            .replace('contract_requires_len_ge: builtin.string "0>=2,1>=2", ', 'contract_requires_len_ge: builtin.string "1>=2", ', 1)
            .replace('verify_no_oob_idx: builtin.string "0,1,2"', 'verify_no_oob_idx: builtin.string "1,2"', 1)
            .replace(
                "  ^active():\n",
                "  ^active():\n    a_len_chk = mir.extract_field (a) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;\n    a_ok_chk = mir.lt (idx, a_len_chk) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n    mir.assert (a_ok_chk) [] []: <(builtin.integer i1) -> ()>;\n",
                1,
            )
        )
        trapping_wae_mir.write_text(trapping_wae_source)
        trapping_wae_lean = require_export(trapping_wae_mir, "vecadd_mut_slice")
        trapping_wae_file = temporary / "VecaddTrappingWae.lean"
        trapping_wae_file.write_text(
            "\n".join(line for line in trapping_wae_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_trapping_wae = run_lake("env", "lean", str(trapping_wae_file))
        if res_trapping_wae.returncode == 0 or "error: omega could not prove the goal:" not in res_trapping_wae.stdout:
            raise RuntimeError(f"Expected trapping mir.assert before store to fail MirWritesAllElems in Lean:\n{res_trapping_wae.stdout}")
        print("REJECTED as expected: unproved `mir.assert` before store fails `MirWritesAllElems` in Lean 4.", flush=True)

        # Stage 8: Warp-specialized 4-stage TMA + metadata + SharedArray<Barrier, 4, 8> ring
        ring_source = WARP_RING_FIXTURE.read_text()
        ring_lean = require_export(WARP_RING_FIXTURE, "warp_ring_stage")
        ring_valid = temporary / "WarpRingValid.lean"
        ring_valid.write_text(ring_lean)
        require_success("env", "lean", str(ring_valid))
        print("PASS: exported warp_ring_stage MIR proves 4-stage SharedArray<Barrier, 4> bounds, slot-ordered TMA/metadata handoff, and cyclic back-pressure.", flush=True)

        # Stage 8 negative check 1: Off-by-one `p_inc == 5` wraparound on 4-stage ring fails `MirNoOobShared` in Lean 4:
        ring_wrap_oob_mir = temporary / "warp_ring_wrap_oob.mir"
        ring_wrap_oob_mir.write_text(
            ring_source.replace(
                "p_wrap = mir.eq (p_inc, c4_u64)",
                "c5_u64 = mir.constant () [] [value: builtin.integer <5: ui64>]: <() -> (builtin.integer ui64)>;\n        p_wrap = mir.eq (p_inc, c5_u64)",
                1,
            )
        )
        ring_wrap_oob_lean = require_export(ring_wrap_oob_mir, "warp_ring_stage")
        ring_wrap_oob_file = temporary / "WarpRingWrapOob.lean"
        ring_wrap_oob_file.write_text(
            "\n".join(line for line in ring_wrap_oob_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_ring_oob = run_lake("env", "lean", str(ring_wrap_oob_file))
        if res_ring_oob.returncode == 0 or "error: omega could not prove the goal:" not in res_ring_oob.stdout:
            raise RuntimeError(f"Expected off-by-one `p_inc == 5` ring wrap to fail MirNoOobShared in Lean:\n{res_ring_oob.stdout}")
        print("REJECTED as expected: off-by-one `stage + 1 == 5` wrap on 4-stage ring fails `no_oob_idx`.", flush=True)

        # Stage 8 negative check 2: Mismatched barrier slot (`FULL_BAR[c_stage]`) vs data slot (`RING_BUF[(c_stage + 1) & 3]`) fails `MirNoSharedRace`:
        ring_slot_mismatch_mir = temporary / "warp_ring_slot_mismatch.mir"
        ring_slot_mismatch_mir.write_text(
            ring_source.replace(
                "c_off = mir.mul (c_stage, c1024_u64)",
                "c3_mask = mir.constant () [] [value: builtin.integer <3: ui64>]: <() -> (builtin.integer ui64)>;\n        c_plus1 = mir.add (c_stage, one_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        c_wrong = mir.bitand (c_plus1, c3_mask) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        c_off = mir.mul (c_wrong, c1024_u64)",
                1,
            )
        )
        ring_slot_mismatch_lean = require_export(ring_slot_mismatch_mir, "warp_ring_stage")
        ring_slot_mismatch_file = temporary / "WarpRingSlotMismatch.lean"
        ring_slot_mismatch_file.write_text(
            "\n".join(line for line in ring_slot_mismatch_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_slot_mismatch = run_lake("env", "lean", str(ring_slot_mismatch_file))
        if res_slot_mismatch.returncode == 0 or "error: omega could not prove the goal:" not in res_slot_mismatch.stdout:
            raise RuntimeError(f"Expected mismatched barrier/data stage slot to fail MirNoSharedRace in Lean:\n{res_slot_mismatch.stdout}")
        print("REJECTED as expected: awaiting `FULL_BAR[c_stage]` while reading `RING_BUF[(c_stage + 1) & 3]` fails `no_data_race`.", flush=True)

        # Stage 8 negative check 3: Missing `EMPTY_BAR` wait in producer warp fails `UniformBarriers` (`NoUnbackpressuredLoop`) in Lean 4:
        ring_no_bp_mir = temporary / "warp_ring_no_backpressure.mir"
        ring_no_bp_mir.write_text(
            ring_source.replace(
                "mir.cond_br (empty_not_ready) [^prod_wait_spin, ^prod_issue] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
                "mir.goto () [^prod_issue] []: <() -> ()>",
                1,
            )
        )
        ring_no_bp_lean = require_export(ring_no_bp_mir, "warp_ring_stage")
        ring_no_bp_file = temporary / "WarpRingNoBackpressure.lean"
        ring_no_bp_file.write_text(
            "\n".join(line for line in ring_no_bp_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_ring_no_bp = run_lake("env", "lean", str(ring_no_bp_file))
        if res_ring_no_bp.returncode == 0:
            raise RuntimeError(f"Expected missing producer EMPTY_BAR wait to fail UniformBarriers in Lean:\n{res_ring_no_bp.stdout}")
        print("REJECTED as expected: warp-specialized TMA ring missing `EMPTY_BAR` wait fails `uniform_barriers`.", flush=True)

        # Stage 9: 2-CTA cluster + DSMEM (`map_shared_rank`) + remote `mbarrier_arrive_cluster` + Blackwell `CLC`
        clc_source = CLUSTER_CLC_FIXTURE.read_text()
        clc_lean = require_export(CLUSTER_CLC_FIXTURE, "cluster_clc_stage")
        clc_valid = temporary / "ClusterClcValid.lean"
        clc_valid.write_text(clc_lean)
        require_success("env", "lean", str(clc_valid))
        print("PASS: exported cluster_clc_stage MIR proves 2-CTA DSMEM rank/sync safety, remote mbarrier arrival, and 16-byte CLC response bounds/ordering.", flush=True)

        # Stage 9 negative check 1: OOB peer rank `my_rank + 1` on 2-CTA cluster fails `MirNoOobClusterRank` in Lean 4:
        clc_oob_rank_mir = temporary / "cluster_clc_oob_rank.mir"
        clc_oob_rank_mir.write_text(
            clc_source.replace(
                "peer_rank = mir.bitxor (my_rank, one_u32)",
                "peer_rank = mir.add (my_rank, one_u32)",
                1,
            )
        )
        clc_oob_rank_lean = require_export(clc_oob_rank_mir, "cluster_clc_stage")
        clc_oob_rank_file = temporary / "ClusterClcOobRank.lean"
        clc_oob_rank_file.write_text(
            "\n".join(line for line in clc_oob_rank_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_oob_rank = run_lake("env", "lean", str(clc_oob_rank_file))
        if res_oob_rank.returncode == 0 or "error: omega could not prove the goal:" not in res_oob_rank.stdout:
            raise RuntimeError(f"Expected OOB `my_rank + 1` on 2-CTA cluster to fail MirNoOobClusterRank in Lean:\n{res_oob_rank.stdout}")
        print("REJECTED as expected: `map_shared_rank(..., my_rank + 1)` on 2-CTA cluster fails `MirNoOobClusterRank`.", flush=True)

        # Stage 9 negative check 2: CTA-local `sync_threads()` instead of `cluster_sync()` before DSMEM read fails `UniformBarriers`:
        clc_local_sync_mir = temporary / "cluster_clc_local_sync.mir"
        clc_local_sync_mir.write_text(
            clc_source.replace(
                "nvvm.cluster_sync () [] []: <() -> ()>;",
                "nvvm.barrier0 () [] []: <() -> ()>;",
                1,
            )
        )
        clc_local_sync_lean = require_export(clc_local_sync_mir, "cluster_clc_stage")
        clc_local_sync_file = temporary / "ClusterClcLocalSync.lean"
        clc_local_sync_file.write_text(
            "\n".join(line for line in clc_local_sync_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_local_sync = run_lake("env", "lean", str(clc_local_sync_file))
        if res_local_sync.returncode == 0:
            raise RuntimeError(f"Expected CTA-local sync_threads() before DSMEM read to fail UniformBarriers in Lean:\n{res_local_sync.stdout}")
        print("REJECTED as expected: cross-CTA DSMEM read without `cluster_sync()` fails `uniform_barriers`.", flush=True)

        # Stage 10: Dual-commit `_cg2` (`MMA_BAR` + post-loop `ACCUM_FULL`) positive + unarmed `_cg2` TMA negative check:
        tcgen_source = TCGEN05_MATMUL_FIXTURE.read_text()
        stage10_cg2_source = (
            tcgen_source
            .replace(
                '[gpu_kernel: builtin.string "true", ',
                '[cluster_dim_x: builtin.integer <2: ui32>, cluster_dim_y: builtin.integer <1: ui32>, cluster_dim_z: builtin.integer <1: ui32>, gpu_kernel: builtin.string "true", ',
                1,
            )
            .replace(
                "      ^sync_after_init():\n        nvvm.barrier0 () [] []: <() -> ()>;",
                "      ^sync_after_init():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
                1,
            )
            .replace("nvvm.tcgen05_alloc (", "nvvm.tcgen05_alloc_cg2 (", 1)
            .replace(
                "      ^sync_after_alloc():\n        nvvm.barrier0 () [] []: <() -> ()>;",
                "      ^sync_after_alloc():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
                1,
            )
            .replace("nvvm.tcgen05_mma_f16 (", "nvvm.tcgen05_mma_f16_cg2 (", 1)
            .replace(
                "        nvvm.tcgen05_commit_shared_cluster (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        mir.goto () [^wait_mma_hdr] []: <() -> ()>",
                '        nvvm.tcgen05_commit_shared_cluster_cg2 (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        mir.goto () [^commit_accum] []: <() -> ()>\n\n      ^commit_accum():\n        mma_bar_p2 = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui64, size: builtin.integer <1: i64>, alloc_key: builtin.string "MMA_BAR", source_name: builtin.string "safe_tcgen05_matmul::kernels::tcgen05_matmul_verified::MMA_BAR"]: <() -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;\n        nvvm.tcgen05_commit_shared_cluster_cg2 (mma_bar_p2) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        mir.goto () [^wait_mma_hdr] []: <() -> ()>',
                1,
            )
            .replace(
                "ready_mma = nvvm.mbarrier_try_wait_parity_shared (mma_bar_w, zero_u32)",
                "ready_mma = nvvm.mbarrier_try_wait_parity_cluster (mma_bar_w, zero_u32)",
                1,
            )
            .replace(
                "      ^sync_before_dealloc():\n        nvvm.barrier0 () [] []: <() -> ()>;\n        mir.cond_br (is_warp0) [^dealloc_tmem, ^exit]",
                "      ^sync_before_dealloc():\n        mir.cond_br (is_warp0) [^relinquish_cg2, ^cluster_sync_dealloc] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^relinquish_cg2():\n        nvvm.tcgen05_relinquish_alloc_permit_cg2 () [] []: <() -> ()>;\n        mir.goto () [^cluster_sync_dealloc] []: <() -> ()>\n\n      ^cluster_sync_dealloc():\n        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.cond_br (is_warp0) [^dealloc_tmem, ^exit]",
                1,
            )
            .replace("nvvm.tcgen05_dealloc (", "nvvm.tcgen05_dealloc_cg2 (", 1)
            .replace(
                "      ^exit():\n        mir.return () [] []: <() -> ()>",
                "      ^exit():\n        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.return () [] []: <() -> ()>",
                1,
            )
        )
        stage10_valid_mir = temporary / "stage10_dual_commit_cg2.mir"
        stage10_valid_mir.write_text(stage10_cg2_source)
        stage10_valid_lean = require_export(stage10_valid_mir, "tcgen05_matmul_verified")
        stage10_valid_file = temporary / "Stage10DualCommitValid.lean"
        stage10_valid_file.write_text(stage10_valid_lean)
        require_success("env", "lean", str(stage10_valid_file))
        print("PASS: Stage 10 2-CTA `_cg2` dual-commit (`MMA_BAR` + `ACCUM_FULL`) proves `TmemLifecycle` and `UniformBarriers`.", flush=True)

        stage10_no_mma_mir = temporary / "stage10_no_mma.mir"
        stage10_no_mma_mir.write_text(
            stage10_cg2_source.replace(
                "        nvvm.tcgen05_mma_f16_cg2 (tmem_addr, a_desc, b_desc, idesc, false_i1) [] []: <(builtin.integer ui32, builtin.integer ui64, builtin.integer ui64, builtin.integer ui32, builtin.integer i1) -> ()>;\n",
                "",
                1,
            )
        )
        stage10_no_mma_lean = require_export(stage10_no_mma_mir, "tcgen05_matmul_verified")
        stage10_no_mma_file = temporary / "Stage10NoMma.lean"
        stage10_no_mma_file.write_text(
            "\n".join(line for line in stage10_no_mma_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_stage10_no_mma = run_lake("env", "lean", str(stage10_no_mma_file))
        if res_stage10_no_mma.returncode == 0:
            raise RuntimeError(f"Expected tcgen05_commit_cg2 without tcgen05_mma_cg2 to fail TmemLifecycle in Lean:\n{res_stage10_no_mma.stdout}")
        print("REJECTED as expected: `tcgen05_commit_cg2` without any `tcgen05_mma_cg2` fails `tmem_lifecycle`.", flush=True)

        # Stage 10 negative check 2: `.cta` `mbarrier_try_wait_parity_shared` on `_cg2` `MMA_BAR` fails `TmemLifecycle`:
        stage10_cta_wait_mir = temporary / "stage10_cta_wait.mir"
        stage10_cta_wait_mir.write_text(
            stage10_cg2_source.replace(
                "ready_mma = nvvm.mbarrier_try_wait_parity_cluster (mma_bar_w, zero_u32)",
                "ready_mma = nvvm.mbarrier_try_wait_parity_shared (mma_bar_w, zero_u32)",
                1,
            )
        )
        stage10_cta_wait_lean = require_export(stage10_cta_wait_mir, "tcgen05_matmul_verified")
        stage10_cta_wait_file = temporary / "Stage10CtaWait.lean"
        stage10_cta_wait_file.write_text(
            "\n".join(line for line in stage10_cta_wait_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_stage10_cta_wait = run_lake("env", "lean", str(stage10_cta_wait_file))
        if res_stage10_cta_wait.returncode == 0:
            raise RuntimeError(f"Expected .cta wait on _cg2 MMA_BAR to fail TmemLifecycle in Lean:\n{res_stage10_cta_wait.stdout}")
        print("REJECTED as expected: `.cta` `mbarrier_try_wait_parity` on `_cg2` barrier fails `tmem_lifecycle`.", flush=True)

        # Stage 10 negative check 3: Missing `tcgen05_fence_after_thread_sync` before `tcgen05_ld` fails `TmemLifecycle`:
        stage10_no_fence_mir = temporary / "stage10_no_fence.mir"
        stage10_no_fence_mir.write_text(
            stage10_cg2_source.replace(
                "      ^epilogue_init():\n        nvvm.tcgen05_fence_after_thread_sync () [] []: <() -> ()>;\n",
                "      ^epilogue_init():\n",
                1,
            )
        )
        stage10_no_fence_lean = require_export(stage10_no_fence_mir, "tcgen05_matmul_verified")
        stage10_no_fence_file = temporary / "Stage10NoFence.lean"
        stage10_no_fence_file.write_text(
            "\n".join(line for line in stage10_no_fence_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_stage10_no_fence = run_lake("env", "lean", str(stage10_no_fence_file))
        if res_stage10_no_fence.returncode == 0:
            raise RuntimeError(f"Expected missing tcgen05_fence_after_thread_sync before tcgen05_ld to fail TmemLifecycle in Lean:\n{res_stage10_no_fence.stdout}")
        print("REJECTED as expected: missing `tcgen05_fence_after_thread_sync` before `tcgen05_ld` fails `tmem_lifecycle`.", flush=True)

        # Pure float math intrinsic placeholder (`.sin()`) positive + un-inlined float helper negative check:
        vecadd_sin_mir = temporary / "vecadd_sin.mir"
        vecadd_sin_mir.write_text(
            vecadd_source.replace(
                "sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
                'a_sin = mir.call (a_val) [] [callee: builtin.string "__cuda_oxide_rust_intrinsic_sinf32"]: <(builtin.fp32) -> (builtin.fp32)>;\n    sum = mir.add (a_sin, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;',
                1,
            )
        )
        vecadd_sin_lean = require_export(vecadd_sin_mir, "vecadd")
        vecadd_sin_file = temporary / "VecaddSinValid.lean"
        vecadd_sin_file.write_text(vecadd_sin_lean)
        require_success("env", "lean", str(vecadd_sin_file))
        print("PASS: pure compiler math intrinsic placeholder (`sinf32`) exports as `.unk` and proves vecadd safety.", flush=True)

        vecadd_sneaky_float_mir = temporary / "vecadd_sneaky_float.mir"
        vecadd_sneaky_float_mir.write_text(
            vecadd_source.replace(
                "sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
                'a_tmp = mir.call (a_val) [] [callee: builtin.string "user_crate::sneaky_sync_helper"]: <(builtin.fp32) -> (builtin.fp32)>;\n    sum = mir.add (a_tmp, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;',
                1,
            )
        )
        sneaky_rejected = export(vecadd_sneaky_float_mir, "vecadd")
        if sneaky_rejected.returncode <= 0 or sneaky_rejected.stdout or "unmodeled function call" not in sneaky_rejected.stderr:
            raise RuntimeError(f"Expected un-inlined float helper to fail closed:\n{sneaky_rejected.stderr}")
        print("REJECTED as expected: un-inlined user `f32 -> f32` helper fails closed without emitting Lean.", flush=True)

        # Arithmetic `contract_requires` (`s0>=4*s1`, `s0>=p2`) + `nvvm.scalar_arithmetic` (`.rn`) positive & negative checks:
        scale_quad_source = """
mir.func @scale_quad: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>, builtin.integer ui32, builtin.fp32) -> ()>
  [contract_requires: builtin.string "s0>=4*s1,s1>=p2", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "1", verify_no_oob_idx: builtin.string "0,1"]
{
  ^entry(x: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, out: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>, n: builtin.integer ui32, scale: builtin.fp32):
    bx32 = nvvm.read_ptx_sreg_ctaid_x () [] []: <() -> (builtin.integer ui32)>;
    bd32 = nvvm.read_ptx_sreg_ntid_x () [] []: <() -> (builtin.integer ui32)>;
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    bx = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    bd = mir.cast (bd32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    tx = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    base = mir.mul (bx, bd) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    idx = mir.add (base, tx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    n64 = mir.cast (n) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    in_n = mir.lt (idx, n64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_n) [^active, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^active():
    c4 = mir.constant () [] [value: builtin.integer <4: ui64>]: <() -> (builtin.integer ui64)>;
    c3 = mir.constant () [] [value: builtin.integer <3: ui64>]: <() -> (builtin.integer ui64)>;
    idx4 = mir.mul (c4, idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    src_idx = mir.add (idx4, c3) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    x_ptr = mir.extract_field (x) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    x_elem = mir.ptr_offset (x_ptr, src_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    v = mir.load (x_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    scaled = nvvm.scalar_arithmetic (v, scale) [] [nvvm_scalar_arithmetic_format: nvvm.scalar_arithmetic_format F32, nvvm_scalar_arithmetic_operation: nvvm.scalar_arithmetic_operation Mul, nvvm_scalar_arithmetic_rounding: nvvm.scalar_arithmetic_rounding Rn, nvvm_scalar_arithmetic_saturation: nvvm.scalar_arithmetic_saturation None, nvvm_scalar_arithmetic_subnormal: nvvm.scalar_arithmetic_subnormal Preserve]: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;
    out_ptr = mir.extract_field (out) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    out_elem = mir.ptr_offset (out_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    mir.store (out_elem, scaled) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}
"""
        scale_quad_mir = temporary / "scale_quad.mir"
        scale_quad_mir.write_text(scale_quad_source)
        scale_quad_lean = require_export(scale_quad_mir, "scale_quad")
        scale_quad_file = temporary / "ScaleQuadValid.lean"
        scale_quad_file.write_text(scale_quad_lean)
        require_success("env", "lean", str(scale_quad_file))
        print("PASS: arithmetic `contract_requires` (`s0>=4*s1,s1>=p2`) + `nvvm.scalar_arithmetic` (`mul_rn_f32`) proves bounds and disjoint writes.", flush=True)

        scale_quad_oob_mir = temporary / "scale_quad_oob.mir"
        scale_quad_oob_mir.write_text(
            scale_quad_source.replace(
                "c3 = mir.constant () [] [value: builtin.integer <3: ui64>]",
                "c3 = mir.constant () [] [value: builtin.integer <4: ui64>]",
                1,
            )
        )
        scale_quad_oob_lean = require_export(scale_quad_oob_mir, "scale_quad")
        scale_quad_oob_file = temporary / "ScaleQuadOob.lean"
        scale_quad_oob_file.write_text(
            "\n".join(line for line in scale_quad_oob_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_scale_quad_oob = run_lake("env", "lean", str(scale_quad_oob_file))
        if res_scale_quad_oob.returncode == 0 or "error: omega could not prove the goal:" not in res_scale_quad_oob.stdout:
            raise RuntimeError(f"Expected `x[4 * idx + 4]` under `x.len() >= 4 * out.len()` to fail in Lean:\n{res_scale_quad_oob.stdout}")
        print("REJECTED as expected: off-by-one `x[4 * idx + 4]` under `x.len() >= 4 * out.len()` fails `no_oob_idx`.", flush=True)

        # CTA-uniform early `return` + strided multi-read pre-barrier `while` loop positive & divergent-return negative checks:
        early_ret_reduce_source = """
mir.func @strided_reduce_early_ret: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, builtin.integer ui32, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> ()>
  [contract_requires_len_ge: builtin.string "0>=2", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "2", verify_no_data_race: builtin.string "0", verify_no_oob_idx: builtin.string "0,2", verify_no_oob_shared: builtin.string "0", verify_uniform_barriers: builtin.string "true"]
{
  ^entry(x: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, m: builtin.integer ui32, y: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>):
    bx32 = nvvm.read_ptx_sreg_ctaid_x () [] []: <() -> (builtin.integer ui32)>;
    skip_blk = mir.ge (bx32, m) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
    mir.cond_br (skip_blk) [^exit, ^init] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^init():
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    tid = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    k = mir.extract_field (x) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
    t_bool = mir.constant () [] [value: builtin.integer <1: i1>]: <() -> (builtin.integer i1)>;
    mir.goto (tid, t_bool) [^loop_hdr] []: <(builtin.integer ui64, builtin.integer i1) -> ()>

  ^loop_hdr(it: builtin.integer ui64, flag: builtin.integer i1):
    c768 = mir.constant () [] [value: builtin.integer <768: ui64>]: <() -> (builtin.integer ui64)>;
    it_end = mir.add (it, c768) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    in_loop = mir.lt (it_end, k) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_loop) [^loop_body, ^post_loop] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^loop_body():
    x_ptr = mir.extract_field (x) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    p0 = mir.ptr_offset (x_ptr, it) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    _v0 = mir.load (p0) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    p3 = mir.ptr_offset (x_ptr, it_end) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    _v3 = mir.load (p3) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    c1024 = mir.constant () [] [value: builtin.integer <1024: ui64>]: <() -> (builtin.integer ui64)>;
    it_next = mir.add (it, c1024) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    f_next = mir.not (flag) [] []: <(builtin.integer i1) -> (builtin.integer i1)>;
    mir.goto (it_next, f_next) [^loop_hdr] []: <(builtin.integer ui64, builtin.integer i1) -> ()>

  ^post_loop():
    lane = nvvm.read_ptx_sreg_laneid () [] []: <() -> (builtin.integer ui32)>;
    z32 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
    is_l0 = mir.eq (lane, z32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
    mir.cond_br (is_l0) [^stage, ^sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^stage():
    c32 = mir.constant () [] [value: builtin.integer <32: ui64>]: <() -> (builtin.integer ui64)>;
    w_idx = mir.div (tid, c32) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    ws_base = mir.shared_alloc () [] [elem_type: builtin.type builtin.fp32, size: builtin.integer <8: i64>, alloc_key: builtin.string "WSUM", source_name: builtin.string "k::WSUM"]: <() -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    ws_slot = mir.ptr_offset (ws_base, w_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    zf = mir.float_constant () [] [float_value: builtin.single 0]: <() -> (builtin.fp32)>;
    mir.store (ws_slot, zf) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>, builtin.fp32) -> ()>;
    mir.goto () [^sync] []: <() -> ()>

  ^sync():
    nvvm.barrier0 () [] []: <() -> ()>;
    ws_r = mir.shared_alloc () [] [elem_type: builtin.type builtin.fp32, size: builtin.integer <8: i64>, alloc_key: builtin.string "WSUM", source_name: builtin.string "k::WSUM"]: <() -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    z64 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
    ws0 = mir.ptr_offset (ws_r, z64) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    s_val = mir.load (ws0) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>) -> (builtin.fp32)>;
    bx64 = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    c256 = mir.constant () [] [value: builtin.integer <256: ui64>]: <() -> (builtin.integer ui64)>;
    b_off = mir.mul (bx64, c256) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    gid = mir.add (b_off, tid) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    y_len = mir.extract_field (y) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (builtin.integer ui64)>;
    in_y = mir.lt (gid, y_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_y) [^write_y, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^write_y():
    y_ptr = mir.extract_field (y) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    y_elem = mir.ptr_offset (y_ptr, gid) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    mir.store (y_elem, s_val) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}
"""
        early_ret_mir = temporary / "early_ret_reduce.mir"
        early_ret_mir.write_text(early_ret_reduce_source)
        early_ret_lean = require_export(early_ret_mir, "strided_reduce_early_ret")
        early_ret_file = temporary / "EarlyRetReduceValid.lean"
        early_ret_file.write_text(early_ret_lean)
        require_success("env", "lean", str(early_ret_file))
        print("PASS: CTA-uniform early `return` + `bool`-carrying strided pre-barrier `while` loop proves `UniformBarriers` and `MirNoOobGlobal`.", flush=True)

        div_ret_mir = temporary / "div_ret_reduce.mir"
        div_ret_mir.write_text(
            early_ret_reduce_source.replace(
                "skip_blk = mir.ge (bx32, m)",
                "tx_early = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;\n    skip_blk = mir.ge (tx_early, m)",
                1,
            )
        )
        div_ret_lean = require_export(div_ret_mir, "strided_reduce_early_ret")
        div_ret_file = temporary / "DivRetReduceBad.lean"
        div_ret_file.write_text(
            "\n".join(line for line in div_ret_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_div_ret = run_lake("env", "lean", str(div_ret_file))
        if res_div_ret.returncode == 0:
            raise RuntimeError(f"Expected thread-divergent early return `tid_x >= m` before sync_threads to fail UniformBarriers:\n{res_div_ret.stdout}")
        print("REJECTED as expected: thread-divergent early `return` (`tid_x >= m`) before `sync_threads()` fails `uniform_barriers`.", flush=True)

        # Warp ballot (`nvvm.vote_sync_ballot`) + scoped atomics (`BlockAtomic` SMEM + `DeviceAtomic` global) checks:
        ballot_atomic_source = """
mir.func @ballot_atomic_check: builtin.function <(mir.slice <builtin.integer ui32,mutable:true,kind:UniqueRef>) -> ()>
  [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_data_race: builtin.string "0", verify_no_global_race: builtin.string "0", verify_no_oob_idx: builtin.string "0", verify_no_oob_shared: builtin.string "0", verify_uniform_barriers: builtin.string "true"]
{
  ^entry(done: mir.slice <builtin.integer ui32,mutable:true,kind:UniqueRef>):
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    tid = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    c8 = mir.constant () [] [value: builtin.integer <8: ui64>]: <() -> (builtin.integer ui64)>;
    in8 = mir.lt (tid, c8) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in8) [^init_h, ^sync1] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^init_h():
    sh0 = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui32, size: builtin.integer <8: i64>, alloc_key: builtin.string "HIST", source_name: builtin.string "k::HIST"]: <() -> (mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>)>;
    sp0 = mir.ptr_offset (sh0, tid) [] []: <(mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>)>;
    z32 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
    nvvm.atomic_store (z32, sp0) [] [nvvm_st_ordering: nvvm.atomic_ordering Relaxed, nvvm_st_scope: nvvm.atomic_scope Block]: <(builtin.integer ui32, mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>) -> ()>;
    mir.goto () [^sync1] []: <() -> ()>

  ^sync1():
    nvvm.barrier0 () [] []: <() -> ()>;
    fmask = mir.constant () [] [value: builtin.integer <4294967295: ui32>]: <() -> (builtin.integer ui32)>;
    pt = mir.constant () [] [value: builtin.integer <1: i1>]: <() -> (builtin.integer i1)>;
    bm = nvvm.vote_sync_ballot (fmask, pt) [] []: <(builtin.integer ui32, builtin.integer i1) -> (builtin.integer ui32)>;
    c7 = mir.constant () [] [value: builtin.integer <7: ui64>]: <() -> (builtin.integer ui64)>;
    bidx = mir.bitand (tid, c7) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    sh1 = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui32, size: builtin.integer <8: i64>, alloc_key: builtin.string "HIST", source_name: builtin.string "k::HIST"]: <() -> (mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>)>;
    sp1 = mir.ptr_offset (sh1, bidx) [] []: <(mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>)>;
    _old_s = nvvm.atomic_rmw (sp1, bm) [] [nvvm_rmw_kind: nvvm.atomic_rmw_kind Add, nvvm_rmw_ordering: nvvm.atomic_ordering Relaxed, nvvm_rmw_scope: nvvm.atomic_scope Block]: <(mir.ptr <builtin.integer ui32,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> (builtin.integer ui32)>;
    z64 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
    dlen = mir.extract_field (done) [] [index: mir.field_index 1]: <(mir.slice <builtin.integer ui32,mutable:true,kind:UniqueRef>) -> (builtin.integer ui64)>;
    ok_d = mir.lt (z64, dlen) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (ok_d) [^inc_d, ^ret_b] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^inc_d():
    dptr = mir.extract_field (done) [] [index: mir.field_index 0]: <(mir.slice <builtin.integer ui32,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.integer ui32,mutable:true,addrspace:0,kind:UniqueRef>)>;
    dp0 = mir.ptr_offset (dptr, z64) [] []: <(mir.ptr <builtin.integer ui32,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32,mutable:true,addrspace:0,kind:UniqueRef>)>;
    c1 = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;
    _old_g = nvvm.atomic_rmw (dp0, c1) [] [nvvm_rmw_kind: nvvm.atomic_rmw_kind Add, nvvm_rmw_ordering: nvvm.atomic_ordering AcqRel, nvvm_rmw_scope: nvvm.atomic_scope Device]: <(mir.ptr <builtin.integer ui32,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui32) -> (builtin.integer ui32)>;
    mir.goto () [^ret_b] []: <() -> ()>

  ^ret_b():
    mir.return () [] []: <() -> ()>
}
"""
        ba_mir = temporary / "ballot_atomic.mir"
        ba_mir.write_text(ballot_atomic_source)
        ba_lean = require_export(ba_mir, "ballot_atomic_check")
        ba_file = temporary / "BallotAtomicValid.lean"
        ba_file.write_text(ba_lean)
        require_success("env", "lean", str(ba_file))
        print("PASS: warp ballot + `.block` shared atomic + `.device` global atomic proves `MirNoSharedRace` and `MirNoGlobalRace`.", flush=True)

        bad_scope_mir = temporary / "bad_scope_atomic.mir"
        bad_scope_mir.write_text(
            ballot_atomic_source.replace(
                "nvvm_rmw_scope: nvvm.atomic_scope Device",
                "nvvm_rmw_scope: nvvm.atomic_scope Block",
                1,
            )
        )
        bad_scope_lean = require_export(bad_scope_mir, "ballot_atomic_check")
        bad_scope_file = temporary / "BadScopeAtomic.lean"
        bad_scope_file.write_text(
            "\n".join(line for line in bad_scope_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_bad_scope = run_lake("env", "lean", str(bad_scope_file))
        if res_bad_scope.returncode == 0:
            raise RuntimeError(f"Expected `.cta` BlockAtomic on global slice across blocks to fail MirNoGlobalRace:\n{res_bad_scope.stdout}")
        print("REJECTED as expected: `.cta`-scoped `BlockAtomic` on global slice across blocks fails `MirNoGlobalRace`.", flush=True)

    print("Exporter checks passed for scalar, 1D vecadd, 2D SGEMM, Stage 3 warp/block reductions, Stage 4 sub-warp multi-buffer, Stage 5 async mbarrier, Stage 6 Blackwell tcgen05/TMEM, Stage 8 warp-specialized ring, Stage 9 2-CTA cluster + CLC, and Stage 10 dual-commit `_cg2` fixtures.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
