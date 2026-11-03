/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDA_OXIDE_COMPILE_OPTIONS_H
#define CUDA_OXIDE_COMPILE_OPTIONS_H

#include <errno.h>
#include <stdio.h>
#include <string.h>

#define CUDA_OXIDE_OPTIONS_HEADER "cuda-oxide-compile-options-v1\n"
#define CUDA_OXIDE_OPTIONS_FMA_ON CUDA_OXIDE_OPTIONS_HEADER "fma-contraction=on\n"
#define CUDA_OXIDE_OPTIONS_FMA_OFF CUDA_OXIDE_OPTIONS_HEADER "fma-contraction=off\n"

static inline int cuda_oxide_sibling_path(const char* artifact, const char* extension,
                                          char* output, size_t capacity) {
    const char* slash = strrchr(artifact, '/');
    const char* dot = strrchr(artifact, '.');
    size_t stem_length = strlen(artifact);
    if (dot && (!slash || dot > slash)) {
        stem_length = (size_t)(dot - artifact);
    }
    size_t extension_length = strlen(extension);
    if (stem_length + extension_length + 1 > capacity) {
        return -1;
    }
    memcpy(output, artifact, stem_length);
    memcpy(output + stem_length, extension, extension_length + 1);
    return 0;
}

/* Return 1 when .options is required, 0 for legacy/no target, and -1 on error. */
static inline int cuda_oxide_target_requires_options(const char* artifact) {
    char path[4096];
    if (cuda_oxide_sibling_path(artifact, ".target", path, sizeof(path)) != 0) {
        fprintf(stderr, "Error: target path is too long for %s\n", artifact);
        return -1;
    }

    FILE* file = fopen(path, "rb");
    if (!file) {
        if (errno == ENOENT) return 0;
        fprintf(stderr, "Error: Cannot open target metadata %s\n", path);
        return -1;
    }
    char value[256];
    size_t length = fread(value, 1, sizeof(value) - 1, file);
    int too_long = !feof(file);
    int read_failed = ferror(file);
    fclose(file);
    value[length] = '\0';
    if (read_failed || too_long) {
        fprintf(stderr, "Error: Cannot read target metadata %s\n", path);
        return -1;
    }

    char* newline = strchr(value, '\n');
    if (!newline || newline == value) {
        fprintf(stderr, "Error: Malformed target metadata %s\n", path);
        return -1;
    }
    const char* remainder = newline + 1;
    if (*remainder == '\0') return 0;
    if (strcmp(remainder, "compile-options=v1\n") == 0) return 1;

    fprintf(stderr, "Error: Unsupported target metadata in %s\n", path);
    return -1;
}

/* Missing metadata means a legacy artifact, whose historical default is on. */
static inline int cuda_oxide_read_fma_policy(const char* artifact, int* allow_fma_contraction) {
    char path[4096];
    if (cuda_oxide_sibling_path(artifact, ".options", path, sizeof(path)) != 0) {
        fprintf(stderr, "Error: compile-options path is too long for %s\n", artifact);
        return -1;
    }

    int options_required = cuda_oxide_target_requires_options(artifact);
    if (options_required < 0) return -1;

    FILE* file = fopen(path, "rb");
    if (!file) {
        if (errno == ENOENT) {
            if (options_required) {
                fprintf(stderr, "Error: Target metadata requires missing compile options %s\n",
                        path);
                return -1;
            }
            *allow_fma_contraction = 1;
            return 0;
        }
        fprintf(stderr, "Error: Cannot open compile options %s\n", path);
        return -1;
    }

    char value[128];
    size_t length = fread(value, 1, sizeof(value) - 1, file);
    int too_long = !feof(file);
    int read_failed = ferror(file);
    fclose(file);
    value[length] = '\0';
    if (read_failed || too_long) {
        fprintf(stderr, "Error: Cannot read compile options %s\n", path);
        return -1;
    }

    if (strcmp(value, CUDA_OXIDE_OPTIONS_FMA_ON) == 0) {
        *allow_fma_contraction = 1;
        return 0;
    }
    if (strcmp(value, CUDA_OXIDE_OPTIONS_FMA_OFF) == 0) {
        *allow_fma_contraction = 0;
        return 0;
    }

    fprintf(stderr, "Error: Unsupported cuda-oxide compile options in %s\n", path);
    return -1;
}

static inline int cuda_oxide_write_fma_policy(const char* artifact,
                                              int allow_fma_contraction) {
    char path[4096];
    if (cuda_oxide_sibling_path(artifact, ".options", path, sizeof(path)) != 0) {
        fprintf(stderr, "Error: compile-options path is too long for %s\n", artifact);
        return -1;
    }

    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot write compile options %s\n", path);
        return -1;
    }
    const char* value = allow_fma_contraction ? CUDA_OXIDE_OPTIONS_FMA_ON
                                               : CUDA_OXIDE_OPTIONS_FMA_OFF;
    size_t length = strlen(value);
    int failed = fwrite(value, 1, length, file) != length || fclose(file) != 0;
    if (failed) {
        fprintf(stderr, "Error: Cannot write compile options %s\n", path);
        return -1;
    }
    return 0;
}

static inline int cuda_oxide_clear_compile_metadata(const char* artifact) {
    const char* extensions[] = {".target", ".options"};
    for (size_t i = 0; i < sizeof(extensions) / sizeof(extensions[0]); i++) {
        char path[4096];
        if (cuda_oxide_sibling_path(artifact, extensions[i], path, sizeof(path)) != 0) {
            fprintf(stderr, "Error: metadata path is too long for %s\n", artifact);
            return -1;
        }
        if (remove(path) != 0 && errno != ENOENT) {
            fprintf(stderr, "Error: Cannot clear stale metadata %s\n", path);
            return -1;
        }
    }
    return 0;
}

/* Write this completion marker only after the sibling .options file exists. */
static inline int cuda_oxide_write_target_metadata(const char* artifact, const char* arch) {
    char path[4096];
    if (cuda_oxide_sibling_path(artifact, ".target", path, sizeof(path)) != 0) {
        fprintf(stderr, "Error: target path is too long for %s\n", artifact);
        return -1;
    }
    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot write target metadata %s\n", path);
        return -1;
    }
    int failed = fprintf(file, "%s\ncompile-options=v1\n", arch) < 0 || fclose(file) != 0;
    if (failed) {
        fprintf(stderr, "Error: Cannot write target metadata %s\n", path);
        return -1;
    }
    return 0;
}

#endif
