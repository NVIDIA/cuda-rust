/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Link multiple LTOIR files using nvJitLink
 *
 * This tool links cuda-oxide generated LTOIR with external LTOIR (e.g., CCCL)
 * to produce a final cubin.
 *
 * Build:
 *   gcc -o link_ltoir link_ltoir.c \
 *       -I/usr/local/cuda/include \
 *       -L/usr/local/cuda/lib64 -lnvJitLink \
 *       -Wl,-rpath,/usr/local/cuda/lib64
 *
 * Usage:
 *   ./link_ltoir -arch=sm_120 -o output.cubin input1.ltoir input2.ltoir ...
 *
 * Example:
 *   ./link_ltoir -arch=sm_120 -o merged.cubin \
 *       device_ffi_test.ltoir external_device_funcs.ltoir
 *
 * Sibling .options files preserve cuda-oxide's FMA policy. If any input
 * disables contraction, the complete nvJitLink LTO step uses -fma=0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvJitLink.h>
#include "compile_options.h"

#define MAX_INPUTS 32

/**
 * Convert nvJitLinkResult to human-readable string.
 */
const char* nvjitlink_result_str(nvJitLinkResult r) {
    switch (r) {
        case NVJITLINK_SUCCESS: return "SUCCESS";
        case NVJITLINK_ERROR_UNRECOGNIZED_OPTION: return "UNRECOGNIZED_OPTION";
        case NVJITLINK_ERROR_MISSING_ARCH: return "MISSING_ARCH";
        case NVJITLINK_ERROR_INVALID_INPUT: return "INVALID_INPUT";
        case NVJITLINK_ERROR_PTX_COMPILE: return "PTX_COMPILE";
        case NVJITLINK_ERROR_NVVM_COMPILE: return "NVVM_COMPILE";
        case NVJITLINK_ERROR_INTERNAL: return "INTERNAL";
        case NVJITLINK_ERROR_THREADPOOL: return "THREADPOOL";
        case NVJITLINK_ERROR_UNRECOGNIZED_INPUT: return "UNRECOGNIZED_INPUT";
        case NVJITLINK_ERROR_FINALIZE: return "FINALIZE";
        case NVJITLINK_ERROR_NULL_INPUT: return "NULL_INPUT";
        case NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS: return "INCOMPATIBLE_OPTIONS";
        case NVJITLINK_ERROR_INCORRECT_INPUT_TYPE: return "INCORRECT_INPUT_TYPE";
        case NVJITLINK_ERROR_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case NVJITLINK_ERROR_OUTDATED_LIBRARY: return "OUTDATED_LIBRARY";
        case NVJITLINK_ERROR_MISSING_FATBIN: return "MISSING_FATBIN";
        case NVJITLINK_ERROR_UNRECOGNIZED_ARCH: return "UNRECOGNIZED_ARCH";
        case NVJITLINK_ERROR_UNSUPPORTED_ARCH: return "UNSUPPORTED_ARCH";
        case NVJITLINK_ERROR_LTO_NOT_ENABLED: return "LTO_NOT_ENABLED";
        default: return "UNKNOWN";
    }
}

/**
 * Print the error log from nvJitLink (e.g., unresolved symbols).
 */
void print_error_log(nvJitLinkHandle handle) {
    size_t log_size;
    nvJitLinkGetErrorLogSize(handle, &log_size);
    if (log_size > 1) {
        char* log = (char*)malloc(log_size);
        nvJitLinkGetErrorLog(handle, log);
        fprintf(stderr, "Error log:\n%s\n", log);
        free(log);
    }
}

/**
 * Print the info log from nvJitLink (verbose linking info).
 */
void print_info_log(nvJitLinkHandle handle) {
    size_t log_size;
    nvJitLinkGetInfoLogSize(handle, &log_size);
    if (log_size > 1) {
        char* log = (char*)malloc(log_size);
        nvJitLinkGetInfoLog(handle, log);
        printf("Info log:\n%s\n", log);
        free(log);
    }
}

/**
 * Read an entire file into memory.
 *
 * @param path      Path to the file
 * @param size_out  Output: size of the file in bytes
 * @return          Allocated buffer containing file contents, or NULL on error
 */
char* read_file(const char* path, size_t* size_out) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* data = (char*)malloc(size);
    if (!data || fread(data, 1, size, f) != size) {
        fclose(f);
        free(data);
        return NULL;
    }
    fclose(f);

    *size_out = size;
    return data;
}

/**
 * Print usage information for the tool.
 */
void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s -arch=<arch> -o <output.cubin> <input1.ltoir> [input2.ltoir ...]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -arch=<arch>    Target architecture (e.g., sm_120)\n");
    fprintf(stderr, "  -o <file>       Output cubin file\n");
    fprintf(stderr, "  -v              Verbose output\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s -arch=sm_120 -o merged.cubin cuda_oxide.ltoir external.ltoir\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* arch = NULL;
    const char* output_file = NULL;
    int verbose = 0;
    const char* input_files[MAX_INPUTS];
    int num_inputs = 0;
    int allow_fma_contraction = 1;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-arch=", 6) == 0) {
            arch = argv[i] + 6;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (argv[i][0] != '-') {
            if (num_inputs < MAX_INPUTS) {
                input_files[num_inputs++] = argv[i];
            }
        }
    }

    if (!arch) {
        fprintf(stderr, "Error: -arch is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!output_file) {
        fprintf(stderr, "Error: -o is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (num_inputs == 0) {
        fprintf(stderr, "Error: At least one input file is required\n");
        print_usage(argv[0]);
        return 1;
    }
    for (int i = 0; i < num_inputs; i++) {
        int input_allows_fma = 1;
        if (cuda_oxide_read_fma_policy(input_files[i], &input_allows_fma) != 0) {
            return 1;
        }
        if (!input_allows_fma) {
            allow_fma_contraction = 0;
        }
    }

    printf("=== nvJitLink LTOIR Linker ===\n");
    printf("Architecture: %s\n", arch);
    printf("Output: %s\n", output_file);
    printf("Inputs: %d files\n", num_inputs);

    // Create nvJitLink handle
    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "-arch=%s", arch);

    const char* options[] = {
        arch_opt,
        "-lto",
        allow_fma_contraction ? "-fma=1" : "-fma=0"
    };
    int num_options = 3;

    printf("FMA contraction: %s\n", allow_fma_contraction ? "on" : "off");

    nvJitLinkHandle handle;
    nvJitLinkResult result = nvJitLinkCreate(&handle, num_options, options);
    if (result != NVJITLINK_SUCCESS) {
        fprintf(stderr, "nvJitLinkCreate failed: %s\n", nvjitlink_result_str(result));
        return 1;
    }

    // Add each LTOIR file
    char** file_data = (char**)malloc(num_inputs * sizeof(char*));
    for (int i = 0; i < num_inputs; i++) {
        size_t size;
        file_data[i] = read_file(input_files[i], &size);
        if (!file_data[i]) {
            fprintf(stderr, "Error: Cannot read %s\n", input_files[i]);
            nvJitLinkDestroy(&handle);
            return 1;
        }

        printf("  Adding: %s (%zu bytes)\n", input_files[i], size);

        result = nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR,
                                  file_data[i], size, input_files[i]);
        if (result != NVJITLINK_SUCCESS) {
            fprintf(stderr, "nvJitLinkAddData failed for %s: %s\n",
                    input_files[i], nvjitlink_result_str(result));
            print_error_log(handle);
            nvJitLinkDestroy(&handle);
            return 1;
        }
    }

    // Complete linking
    printf("\nLinking...\n");
    result = nvJitLinkComplete(handle);
    if (result != NVJITLINK_SUCCESS) {
        fprintf(stderr, "nvJitLinkComplete failed: %s\n", nvjitlink_result_str(result));
        print_error_log(handle);
        nvJitLinkDestroy(&handle);
        return 1;
    }

    if (verbose) {
        print_info_log(handle);
    }

    // Get linked cubin
    size_t cubin_size;
    result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
    if (result != NVJITLINK_SUCCESS) {
        fprintf(stderr, "nvJitLinkGetLinkedCubinSize failed: %s\n", nvjitlink_result_str(result));
        nvJitLinkDestroy(&handle);
        return 1;
    }

    void* cubin = malloc(cubin_size);
    result = nvJitLinkGetLinkedCubin(handle, cubin);
    if (result != NVJITLINK_SUCCESS) {
        fprintf(stderr, "nvJitLinkGetLinkedCubin failed: %s\n", nvjitlink_result_str(result));
        free(cubin);
        nvJitLinkDestroy(&handle);
        return 1;
    }

    // Save cubin
    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Error: Cannot write to %s\n", output_file);
        free(cubin);
        nvJitLinkDestroy(&handle);
        return 1;
    }
    fwrite(cubin, 1, cubin_size, out);
    fclose(out);

    printf("Linked cubin: %s (%zu bytes)\n", output_file, cubin_size);

    // Cleanup
    free(cubin);
    for (int i = 0; i < num_inputs; i++) {
        free(file_data[i]);
    }
    free(file_data);
    nvJitLinkDestroy(&handle);

    printf("\n=== Linking succeeded! ===\n");
    return 0;
}
