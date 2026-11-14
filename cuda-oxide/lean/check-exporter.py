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

    print("Exporter checks passed for scalar, 1D vecadd, 2D SGEMM, Stage 3 warp/block reductions, Stage 4 sub-warp multi-buffer, and Stage 5 async mbarrier fixtures.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
