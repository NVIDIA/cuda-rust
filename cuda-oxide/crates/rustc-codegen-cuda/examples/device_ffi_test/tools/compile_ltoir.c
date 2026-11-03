/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Compile LLVM IR to LTOIR using libNVVM
 *
 * This tool takes cuda-oxide generated .ll files and compiles them to LTOIR
 * using libNVVM with the -gen-lto flag.
 *
 * Build:
 *   gcc -o compile_ltoir compile_ltoir.c \
 *       -I/usr/local/cuda/nvvm/include \
 *       -L/usr/local/cuda/nvvm/lib64 -lnvvm \
 *       -Wl,-rpath,/usr/local/cuda/nvvm/lib64
 *
 * Usage:
 *   ./compile_ltoir <input.ll> <arch> [output.ltoir] [--libdevice <path>]
 *
 * Examples:
 *   ./compile_ltoir device_ffi_test.ll sm_120 device_ffi_test.ltoir
 *   ./compile_ltoir kernel.ll sm_90 kernel.ltoir \
 *       --libdevice /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
 *
 * --libdevice <path> adds CUDA's `libdevice.10.bc` to the libNVVM program
 * before compiling, so any `__nv_*` math symbols in the input are resolved
 * during this step. The resulting LTOIR is self-contained and `link_ltoir`
 * does not need to add libdevice separately.
 *
 * A sibling <input>.options file selects -fma=0 or -fma=1. The tool writes
 * matching .options and versioned .target files beside the output LTOIR.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvvm.h>
#include "compile_options.h"

/**
 * Check an NVVM result and exit with error message if it failed.
 *
 * @param result  The nvvmResult to check
 * @param msg     Context message for the error
 * @param prog    Optional program handle for retrieving compilation log
 */
static void check_nvvm(nvvmResult result, const char* msg, nvvmProgram prog) {
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error: %s - %s\n", msg, nvvmGetErrorString(result));
        if (prog) {
            size_t logSize;
            if (nvvmGetProgramLogSize(prog, &logSize) == NVVM_SUCCESS && logSize > 1) {
                char* log = malloc(logSize);
                if (nvvmGetProgramLog(prog, log) == NVVM_SUCCESS) {
                    fprintf(stderr, "Log:\n%s\n", log);
                }
                free(log);
            }
        }
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <input.ll> <arch> [output.ltoir] [--libdevice <path>]\n",
                argv[0]);
        fprintf(stderr, "  arch: sm_100, sm_120, etc.\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s device_ffi_test.ll sm_120 device_ffi_test.ltoir\n", argv[0]);
        fprintf(stderr,
                "  %s kernel.ll sm_120 kernel.ltoir --libdevice "
                "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc\n",
                argv[0]);
        return 1;
    }

    const char* inputFile = argv[1];
    const char* arch = argv[2];
    const char* outputFile = NULL;
    const char* libdeviceFile = NULL;
    int allowFmaContraction = 1;

    // Positional output file is optional and must come before --libdevice.
    int argi = 3;
    if (argi < argc && argv[argi][0] != '-') {
        outputFile = argv[argi++];
    }
    while (argi < argc) {
        if (strcmp(argv[argi], "--libdevice") == 0 && argi + 1 < argc) {
            libdeviceFile = argv[argi + 1];
            argi += 2;
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", argv[argi]);
            return 1;
        }
    }

    if (cuda_oxide_read_fma_policy(inputFile, &allowFmaContraction) != 0) {
        return 1;
    }

    // Print libNVVM version info
    int major, minor;
    nvvmVersion(&major, &minor);
    printf("libNVVM version: %d.%d\n", major, minor);

    int irMajor, irMinor, dbgMajor, dbgMinor;
    nvvmIRVersion(&irMajor, &irMinor, &dbgMajor, &dbgMinor);
    printf("NVVM IR version: %d.%d (debug: %d.%d)\n", irMajor, irMinor, dbgMajor, dbgMinor);

    // Convert sm_XXX to compute_XXX
    char archOpt[64];
    snprintf(archOpt, sizeof(archOpt), "compute_%s", arch + 3);
    printf("Target architecture: %s\n", archOpt);

    // Read input file
    FILE* f = fopen(inputFile, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", inputFile);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(size + 1);
    if (!buffer || fread(buffer, 1, size, f) != size) {
        fprintf(stderr, "Error: Cannot read %s\n", inputFile);
        fclose(f);
        free(buffer);
        return 1;
    }
    buffer[size] = '\0';
    fclose(f);
    printf("Read %zu bytes from %s\n", size, inputFile);

    // Create program
    nvvmProgram prog;
    check_nvvm(nvvmCreateProgram(&prog), "nvvmCreateProgram", NULL);

    // Optionally add libdevice. libNVVM will inline the requested __nv_*
    // entry points during the -gen-lto compile, so the output LTOIR has
    // no dangling math symbols left for nvJitLink to resolve.
    char* libdeviceBuffer = NULL;
    size_t libdeviceSize = 0;
    if (libdeviceFile) {
        FILE* lf = fopen(libdeviceFile, "rb");
        if (!lf) {
            fprintf(stderr, "Error: Cannot open libdevice file %s\n", libdeviceFile);
            nvvmDestroyProgram(&prog);
            free(buffer);
            return 1;
        }
        fseek(lf, 0, SEEK_END);
        libdeviceSize = ftell(lf);
        fseek(lf, 0, SEEK_SET);
        libdeviceBuffer = malloc(libdeviceSize);
        if (!libdeviceBuffer || fread(libdeviceBuffer, 1, libdeviceSize, lf) != libdeviceSize) {
            fprintf(stderr, "Error: Cannot read libdevice file %s\n", libdeviceFile);
            fclose(lf);
            nvvmDestroyProgram(&prog);
            free(buffer);
            free(libdeviceBuffer);
            return 1;
        }
        fclose(lf);
        printf("Read %zu bytes of libdevice from %s\n", libdeviceSize, libdeviceFile);

        nvvmResult ldResult = nvvmAddModuleToProgram(prog, libdeviceBuffer, libdeviceSize,
                                                     "libdevice.10.bc");
        if (ldResult != NVVM_SUCCESS) {
            fprintf(stderr, "nvvmAddModuleToProgram(libdevice) failed: %s\n",
                    nvvmGetErrorString(ldResult));
            size_t logSize;
            if (nvvmGetProgramLogSize(prog, &logSize) == NVVM_SUCCESS && logSize > 1) {
                char* log = malloc(logSize);
                if (nvvmGetProgramLog(prog, log) == NVVM_SUCCESS) {
                    fprintf(stderr, "Log:\n%s\n", log);
                }
                free(log);
            }
            nvvmDestroyProgram(&prog);
            free(buffer);
            free(libdeviceBuffer);
            return 1;
        }
        printf("Libdevice added successfully\n");
    }

    // Add main module
    nvvmResult addResult = nvvmAddModuleToProgram(prog, buffer, size, inputFile);
    if (addResult != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmAddModuleToProgram failed: %s\n", nvvmGetErrorString(addResult));
        size_t logSize;
        if (nvvmGetProgramLogSize(prog, &logSize) == NVVM_SUCCESS && logSize > 1) {
            char* log = malloc(logSize);
            if (nvvmGetProgramLog(prog, log) == NVVM_SUCCESS) {
                fprintf(stderr, "Log:\n%s\n", log);
            }
            free(log);
        }
        nvvmDestroyProgram(&prog);
        free(buffer);
        free(libdeviceBuffer);
        return 1;
    }
    printf("Module added successfully\n");

    // Compile options - CRITICAL: -gen-lto generates LTOIR
    char archOption[128];
    snprintf(archOption, sizeof(archOption), "-arch=%s", archOpt);

    const char* options[] = {
        archOption,
        "-gen-lto",  // Generate LTOIR for link-time optimization
        allowFmaContraction ? "-fma=1" : "-fma=0"
    };
    int numOptions = 3;

    printf("Compiling with options: %s %s %s\n", options[0], options[1], options[2]);

    // Compile
    nvvmResult compileResult = nvvmCompileProgram(prog, numOptions, options);
    if (compileResult != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmCompileProgram failed: %s\n", nvvmGetErrorString(compileResult));
        size_t logSize;
        if (nvvmGetProgramLogSize(prog, &logSize) == NVVM_SUCCESS && logSize > 1) {
            char* log = malloc(logSize);
            if (nvvmGetProgramLog(prog, log) == NVVM_SUCCESS) {
                fprintf(stderr, "Log:\n%s\n", log);
            }
            free(log);
        }
        nvvmDestroyProgram(&prog);
        free(buffer);
        free(libdeviceBuffer);
        return 1;
    }
    printf("Compilation successful!\n");

    // Get result
    size_t resultSize;
    check_nvvm(nvvmGetCompiledResultSize(prog, &resultSize), "nvvmGetCompiledResultSize", prog);
    printf("LTOIR size: %zu bytes\n", resultSize);

    char* result = malloc(resultSize);
    check_nvvm(nvvmGetCompiledResult(prog, result), "nvvmGetCompiledResult", prog);

    // Save LTOIR
    char autoOutput[4096];
    if (!outputFile) {
        const char* dot = strrchr(inputFile, '.');
        size_t stemLength = dot ? (size_t)(dot - inputFile) : strlen(inputFile);
        if (stemLength + sizeof(".ltoir") > sizeof(autoOutput)) {
            fprintf(stderr, "Error: output path is too long\n");
            return 1;
        }
        memcpy(autoOutput, inputFile, stemLength);
        memcpy(autoOutput + stemLength, ".ltoir", sizeof(".ltoir"));
        outputFile = autoOutput;
    }
    if (cuda_oxide_clear_compile_metadata(outputFile) != 0) {
        return 1;
    }
    FILE* out = fopen(outputFile, "wb");
    if (!out || fwrite(result, 1, resultSize, out) != resultSize || fclose(out) != 0) {
        fprintf(stderr, "Error: Cannot write to %s\n", outputFile);
        return 1;
    }
    if (cuda_oxide_write_fma_policy(outputFile, allowFmaContraction) != 0) {
        return 1;
    }
    if (cuda_oxide_write_target_metadata(outputFile, arch) != 0) {
        return 1;
    }
    printf("Saved LTOIR to: %s\n", outputFile);

    // Cleanup
    free(result);
    free(buffer);
    free(libdeviceBuffer);
    nvvmDestroyProgram(&prog);

    printf("\n=== LLVM IR -> LTOIR compilation succeeded! ===\n");
    return 0;
}
