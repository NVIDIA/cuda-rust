; SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
; SPDX-FileCopyrightText: Copyright (c) 2026 ssoj13
; SPDX-License-Identifier: Apache-2.0
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
define void @probe_add_rm_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rm.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rm_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "add.rm.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_add_rm_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rm.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rm_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rm.sat.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rm_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rm.sat.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rn_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rn.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rn_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "add.rn.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_add_rn_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rn.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rn_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rn.sat.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rn_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rn.sat.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rp_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rp.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rp_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "add.rp.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_add_rp_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rp.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rp_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rp.sat.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rp_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rp.sat.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rz_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "add.rz.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_add_rz_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rz.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rz_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rz.sat.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_add_rz_sat_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "add.rz.sat.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rm_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rm.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rm_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "div.rm.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_div_rm_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rm.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rn_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rn.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rn_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "div.rn.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_div_rn_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rn.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rp_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rp.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rp_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "div.rp.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_div_rp_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rp.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_div_rz_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "div.rz.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_div_rz_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "div.rz.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rm_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rm.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rm_f64(double addrspace(1)* %out, double %a0, double %a1, double %a2) {
 %value = call double asm "fma.rm.f64 $0, $1, $2, $3;", "=d,d,d,d"(double %a0, double %a1, double %a2)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_fma_rm_ftz_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rm.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rm_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rm.ftz.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rm_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rm.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rn_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rn_f64(double addrspace(1)* %out, double %a0, double %a1, double %a2) {
 %value = call double asm "fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d"(double %a0, double %a1, double %a2)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_fma_rn_ftz_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rn.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rn_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rn.ftz.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rn_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rn.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rp_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rp.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rp_f64(double addrspace(1)* %out, double %a0, double %a1, double %a2) {
 %value = call double asm "fma.rp.f64 $0, $1, $2, $3;", "=d,d,d,d"(double %a0, double %a1, double %a2)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_fma_rp_ftz_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rp.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rp_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rp.ftz.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rp_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rp.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rz_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rz.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rz_f64(double addrspace(1)* %out, double %a0, double %a1, double %a2) {
 %value = call double asm "fma.rz.f64 $0, $1, $2, $3;", "=d,d,d,d"(double %a0, double %a1, double %a2)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_fma_rz_ftz_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rz.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rz_ftz_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rz.ftz.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_fma_rz_sat_f32(float addrspace(1)* %out, float %a0, float %a1, float %a2) {
 %value = call float asm "fma.rz.sat.f32 $0, $1, $2, $3;", "=f,f,f,f"(float %a0, float %a1, float %a2)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rm_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rm.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rm_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "mul.rm.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_mul_rm_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rm.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rn_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rn_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "mul.rn.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_mul_rn_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rn.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rp_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rp.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rp_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "mul.rp.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_mul_rp_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rp.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
define void @probe_mul_rz_f64(double addrspace(1)* %out, double %a0, double %a1) {
 %value = call double asm "mul.rz.f64 $0, $1, $2;", "=d,d,d"(double %a0, double %a1)
 store double %value, double addrspace(1)* %out
 ret void
}
define void @probe_mul_rz_ftz_f32(float addrspace(1)* %out, float %a0, float %a1) {
 %value = call float asm "mul.rz.ftz.f32 $0, $1, $2;", "=f,f,f"(float %a0, float %a1)
 store float %value, float addrspace(1)* %out
 ret void
}
@llvm.used = appending global [64 x i8*] [i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rm_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_add_rm_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rm_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rm_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rm_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rn_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_add_rn_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rn_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rn_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rn_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rp_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_add_rp_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rp_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rp_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rp_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rz_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_add_rz_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rz_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rz_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_add_rz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rm_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_div_rm_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rm_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rn_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_div_rn_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rn_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rp_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_div_rp_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rp_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rz_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_div_rz_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_div_rz_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rm_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double, double)* @probe_fma_rm_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rm_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rm_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rm_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rn_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double, double)* @probe_fma_rn_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rn_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rn_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rn_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rp_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double, double)* @probe_fma_rp_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rp_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rp_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rp_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rz_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double, double)* @probe_fma_rz_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rz_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rz_ftz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float, float)* @probe_fma_rz_sat_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rm_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_mul_rm_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rm_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rn_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_mul_rn_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rn_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rp_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_mul_rp_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rp_ftz_f32 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rz_f32 to i8*), i8* bitcast (void (double addrspace(1)*, double, double)* @probe_mul_rz_f64 to i8*), i8* bitcast (void (float addrspace(1)*, float, float)* @probe_mul_rz_ftz_f32 to i8*)], section "llvm.metadata"
!nvvm.annotations = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63}
!0 = !{void (float addrspace(1)*, float, float)* @probe_add_rm_f32, !"kernel", i32 1}
!1 = !{void (double addrspace(1)*, double, double)* @probe_add_rm_f64, !"kernel", i32 1}
!2 = !{void (float addrspace(1)*, float, float)* @probe_add_rm_ftz_f32, !"kernel", i32 1}
!3 = !{void (float addrspace(1)*, float, float)* @probe_add_rm_ftz_sat_f32, !"kernel", i32 1}
!4 = !{void (float addrspace(1)*, float, float)* @probe_add_rm_sat_f32, !"kernel", i32 1}
!5 = !{void (float addrspace(1)*, float, float)* @probe_add_rn_f32, !"kernel", i32 1}
!6 = !{void (double addrspace(1)*, double, double)* @probe_add_rn_f64, !"kernel", i32 1}
!7 = !{void (float addrspace(1)*, float, float)* @probe_add_rn_ftz_f32, !"kernel", i32 1}
!8 = !{void (float addrspace(1)*, float, float)* @probe_add_rn_ftz_sat_f32, !"kernel", i32 1}
!9 = !{void (float addrspace(1)*, float, float)* @probe_add_rn_sat_f32, !"kernel", i32 1}
!10 = !{void (float addrspace(1)*, float, float)* @probe_add_rp_f32, !"kernel", i32 1}
!11 = !{void (double addrspace(1)*, double, double)* @probe_add_rp_f64, !"kernel", i32 1}
!12 = !{void (float addrspace(1)*, float, float)* @probe_add_rp_ftz_f32, !"kernel", i32 1}
!13 = !{void (float addrspace(1)*, float, float)* @probe_add_rp_ftz_sat_f32, !"kernel", i32 1}
!14 = !{void (float addrspace(1)*, float, float)* @probe_add_rp_sat_f32, !"kernel", i32 1}
!15 = !{void (float addrspace(1)*, float, float)* @probe_add_rz_f32, !"kernel", i32 1}
!16 = !{void (double addrspace(1)*, double, double)* @probe_add_rz_f64, !"kernel", i32 1}
!17 = !{void (float addrspace(1)*, float, float)* @probe_add_rz_ftz_f32, !"kernel", i32 1}
!18 = !{void (float addrspace(1)*, float, float)* @probe_add_rz_ftz_sat_f32, !"kernel", i32 1}
!19 = !{void (float addrspace(1)*, float, float)* @probe_add_rz_sat_f32, !"kernel", i32 1}
!20 = !{void (float addrspace(1)*, float, float)* @probe_div_rm_f32, !"kernel", i32 1}
!21 = !{void (double addrspace(1)*, double, double)* @probe_div_rm_f64, !"kernel", i32 1}
!22 = !{void (float addrspace(1)*, float, float)* @probe_div_rm_ftz_f32, !"kernel", i32 1}
!23 = !{void (float addrspace(1)*, float, float)* @probe_div_rn_f32, !"kernel", i32 1}
!24 = !{void (double addrspace(1)*, double, double)* @probe_div_rn_f64, !"kernel", i32 1}
!25 = !{void (float addrspace(1)*, float, float)* @probe_div_rn_ftz_f32, !"kernel", i32 1}
!26 = !{void (float addrspace(1)*, float, float)* @probe_div_rp_f32, !"kernel", i32 1}
!27 = !{void (double addrspace(1)*, double, double)* @probe_div_rp_f64, !"kernel", i32 1}
!28 = !{void (float addrspace(1)*, float, float)* @probe_div_rp_ftz_f32, !"kernel", i32 1}
!29 = !{void (float addrspace(1)*, float, float)* @probe_div_rz_f32, !"kernel", i32 1}
!30 = !{void (double addrspace(1)*, double, double)* @probe_div_rz_f64, !"kernel", i32 1}
!31 = !{void (float addrspace(1)*, float, float)* @probe_div_rz_ftz_f32, !"kernel", i32 1}
!32 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rm_f32, !"kernel", i32 1}
!33 = !{void (double addrspace(1)*, double, double, double)* @probe_fma_rm_f64, !"kernel", i32 1}
!34 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rm_ftz_f32, !"kernel", i32 1}
!35 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rm_ftz_sat_f32, !"kernel", i32 1}
!36 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rm_sat_f32, !"kernel", i32 1}
!37 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rn_f32, !"kernel", i32 1}
!38 = !{void (double addrspace(1)*, double, double, double)* @probe_fma_rn_f64, !"kernel", i32 1}
!39 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rn_ftz_f32, !"kernel", i32 1}
!40 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rn_ftz_sat_f32, !"kernel", i32 1}
!41 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rn_sat_f32, !"kernel", i32 1}
!42 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rp_f32, !"kernel", i32 1}
!43 = !{void (double addrspace(1)*, double, double, double)* @probe_fma_rp_f64, !"kernel", i32 1}
!44 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rp_ftz_f32, !"kernel", i32 1}
!45 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rp_ftz_sat_f32, !"kernel", i32 1}
!46 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rp_sat_f32, !"kernel", i32 1}
!47 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rz_f32, !"kernel", i32 1}
!48 = !{void (double addrspace(1)*, double, double, double)* @probe_fma_rz_f64, !"kernel", i32 1}
!49 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rz_ftz_f32, !"kernel", i32 1}
!50 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rz_ftz_sat_f32, !"kernel", i32 1}
!51 = !{void (float addrspace(1)*, float, float, float)* @probe_fma_rz_sat_f32, !"kernel", i32 1}
!52 = !{void (float addrspace(1)*, float, float)* @probe_mul_rm_f32, !"kernel", i32 1}
!53 = !{void (double addrspace(1)*, double, double)* @probe_mul_rm_f64, !"kernel", i32 1}
!54 = !{void (float addrspace(1)*, float, float)* @probe_mul_rm_ftz_f32, !"kernel", i32 1}
!55 = !{void (float addrspace(1)*, float, float)* @probe_mul_rn_f32, !"kernel", i32 1}
!56 = !{void (double addrspace(1)*, double, double)* @probe_mul_rn_f64, !"kernel", i32 1}
!57 = !{void (float addrspace(1)*, float, float)* @probe_mul_rn_ftz_f32, !"kernel", i32 1}
!58 = !{void (float addrspace(1)*, float, float)* @probe_mul_rp_f32, !"kernel", i32 1}
!59 = !{void (double addrspace(1)*, double, double)* @probe_mul_rp_f64, !"kernel", i32 1}
!60 = !{void (float addrspace(1)*, float, float)* @probe_mul_rp_ftz_f32, !"kernel", i32 1}
!61 = !{void (float addrspace(1)*, float, float)* @probe_mul_rz_f32, !"kernel", i32 1}
!62 = !{void (double addrspace(1)*, double, double)* @probe_mul_rz_f64, !"kernel", i32 1}
!63 = !{void (float addrspace(1)*, float, float)* @probe_mul_rz_ftz_f32, !"kernel", i32 1}
!nvvmir.version = !{!64}
!64 = !{i32 2, i32 0, i32 3, i32 2}
