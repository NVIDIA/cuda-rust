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
            raise RuntimeError(f"Expected conditional skip under writes_all_elems to fail WritesOnAllActivePaths in Lean:\n{res_skip.stdout}")
        print("REJECTED as expected: conditional branch skipping output write fails `WritesOnAllActivePaths` in Lean 4.", flush=True)

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

        # Mutate sgemm_tiled_raw MIR to index `a` with an unguarded/non-strip offset `a_len`:
        bad_strip_mir = temporary / "sgemm_bad_strip.mir"
        bad_strip_mir.write_text(
            sgemm_source.replace(
                "a_elem = mir.ptr_offset (a_ptr, a_idx)",
                "a_elem = mir.ptr_offset (a_ptr, a_len)",
                1,
            )
        )
        rejected_strip = export(bad_strip_mir, "sgemm_tiled_raw")
        if rejected_strip.returncode == 0 or "2D global slice pointer offset must use a row-strip" not in rejected_strip.stderr:
            raise RuntimeError(f"Expected non-strip 2D global slice access to be rejected:\n{rejected_strip.stderr}")
        print("REJECTED as expected: non-strip 2D global slice access is rejected before Lean.", flush=True)

        # Mutate sgemm_tiled_raw MIR to check `a_col < k` instead of `a_col < a_eff` (ignoring band check):
        unguarded_strip_mir = temporary / "sgemm_unguarded_strip.mir"
        unguarded_strip_mir.write_text(
            SGEMM_TILED_FIXTURE.read_text().replace(
                "in_a = mir.lt (a_col, a_eff)",
                "in_a = mir.lt (a_col, k)",
                1,
            )
        )
        rejected_unguarded_strip = export(unguarded_strip_mir, "sgemm_tiled_raw")
        if rejected_unguarded_strip.returncode == 0 or "2D global slice pointer offset must use a row-strip" not in rejected_unguarded_strip.stderr:
            raise RuntimeError(f"Expected unguarded 2D global slice strip access to be rejected:\n{rejected_unguarded_strip.stderr}")
        print("REJECTED as expected: unguarded 2D global slice strip access is rejected before Lean.", flush=True)

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
        rejected_post_loop = export(post_loop_mir, "block_reduce_sum")
        if rejected_post_loop.returncode == 0 or "shared memory pointer offset must use a recognized 2D tile coordinate" not in rejected_post_loop.stderr:
            raise RuntimeError(f"Expected post-loop `WARP_SUMS[w]` access to be rejected:\n{rejected_post_loop.stderr}")
        print("REJECTED as expected: post-loop `WARP_SUMS[w]` access (`w == 8`) is rejected before Lean.", flush=True)

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
        rejected_bitor = export(bad_bitor_mir, "subwarp_bit_split_stage")
        if rejected_bitor.returncode == 0 or "shared memory pointer offset must use a recognized 2D tile coordinate" not in rejected_bitor.stderr:
            raise RuntimeError(f"Expected overlapping `tid | 1` bitor to be rejected:\n{rejected_bitor.stderr}")
        print("REJECTED as expected: overlapping `tid | 1` bitwise OR is rejected before Lean.", flush=True)

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

    print("Exporter checks passed for scalar, 1D vecadd, 2D SGEMM, Stage 3 warp/block reductions, Stage 4 sub-warp multi-buffer, Stage 5 async mbarrier, and Stage 6 Blackwell tcgen05/TMEM fixtures.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
