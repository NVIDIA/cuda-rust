/*
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
*/

{
  description = "Development shell for cuda-oxide";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      crane,
      ...
    }:
    let
      # Template flake for user projects. Extends cuda-oxide's devShell via
      # inputsFrom so users can add their own packages while inheriting the full
      # CUDA + Rust environment (including the shellHook that wires up the host
      # NVIDIA driver). nixpkgs and flake-utils are followed from cuda-oxide to
      # avoid duplicate closures.
      userFlakeContent = ''
        {
          description = "A cuda-oxide project";

          inputs = {
            cuda-oxide.url = "github:NVlabs/cuda-oxide";
            nixpkgs.follows = "cuda-oxide/nixpkgs";
            flake-utils.follows = "cuda-oxide/flake-utils";
          };

          outputs =
            {
              cuda-oxide,
              nixpkgs,
              flake-utils,
              ...
            }:
            flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
              system:
              let
                pkgs = nixpkgs.legacyPackages.''${system};
              in
              {
                devShells.default = pkgs.mkShell {
                  inputsFrom = [ cuda-oxide.devShells.''${system}.default ];
                  packages = [
                    # add project-specific packages here
                  ];
                };
              }
            );
        }
      '';

      userFlake = builtins.toFile "flake.nix" userFlakeContent;

      # Directory used by `nix flake init -t github:NVlabs/cuda-oxide`.
      # Content is system-independent; x86_64-linux is chosen arbitrarily.
      templateSrc = nixpkgs.legacyPackages.x86_64-linux.writeTextDir "flake.nix" userFlakeContent;
    in
    (flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfree = true;
          };
        };

        # LLVM
        llvmPkgs = pkgs.llvmPackages_22;

        # Nightly Rust
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # cuda
        cudaSymlinked = pkgs.symlinkJoin {
          name = "cuda-symlinked";
          paths = with pkgs.cudaPackages_13; [
            cuda_nvcc
            cuda_gdb.bin
            cuda_cudart
            # `cuda_cudart`'s `include/host_config.h` is a thin wrapper that
            # delegates to `include/crt/host_config.h`. nixpkgs ships those
            # `crt/*.h` headers in a separate `cuda_crt` package — without it,
            # plain C/C++ host code (e.g. the cublaslt bench) hits an infinite
            # `#include "crt/host_config.h"` -> `host_config.h` loop.
            cuda_crt
            # CUDA C++ Core Libraries — provides `<nv/target>` etc. that
            # `cuda_fp16.h` and other CTK headers reach into.
            cuda_cccl
            libnvvm
            libnvjitlink.lib
            # cuRAND headers: the shared cuda-bindings crate (cutile-rs) runs
            # bindgen over cuda.h and curand.h, so curand.h is a build input.
            libcurand.include
            # cuBLAS (incl. cuBLASLt) for the gemm_sol/bench/cublaslt_bench.c
            # baseline. Not a runtime dep of cuda-oxide itself; pulled in for
            # the bench tooling. Multi-output package: pick the .so output and
            # the headers explicitly (the default output is just LICENSE/src).
            libcublas.lib
            libcublas.include
          ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        cargoOxideCommonArgs = {
          src = ./.;
          cargoExtraArgs = "-p cargo-oxide";
          doCheck = false;

          nativeBuildInputs = [
            llvmPkgs.clang
            llvmPkgs.libclang
          ];

          CUDA_HOME = cudaSymlinked;
          LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
        };

        cargoOxideDeps = craneLib.buildDepsOnly (
          craneLib.crateNameFromCargoToml { cargoToml = ./cuda-oxide/crates/cargo-oxide/Cargo.toml; }
          // cargoOxideCommonArgs
        );

        new-project = pkgs.writeShellApplication {
          name = "cuda-oxide-new";
          runtimeInputs = [ cargo-oxide ];
          text = ''
            # `cargo oxide new <name> [--async]` takes a single positional
            # project name. Pick the first non-flag argument as the directory
            # to drop the template flake into, so flag ordering (e.g. a leading
            # `--async`) doesn't send the copy to the wrong path.
            project=""
            for arg in "$@"; do
              case "$arg" in
                -*) ;;
                *)
                  project="$arg"
                  break
                  ;;
              esac
            done

            output=$(cargo-oxide new "$@")

            if [ -n "$project" ] && [ -d "$project" ]; then
              cp ${userFlake} "$project/flake.nix"
              chmod +w "$project/flake.nix"
            fi

            echo "Note: run 'nix develop' inside the project directory before using cargo oxide."
            echo ""
            echo "$output"
          '';
        };

        # Packages the `cargo-oxide` CLI. Known impurity: `cargo oxide run`
        # still builds librustc_codegen_cuda.so on first use and caches it
        # outside the Nix store, so this derivation is not fully pure yet.
        cargo-oxide = craneLib.buildPackage (
          craneLib.crateNameFromCargoToml { cargoToml = ./cuda-oxide/crates/cargo-oxide/Cargo.toml; }
          // cargoOxideCommonArgs
          // {
            cargoArtifacts = cargoOxideDeps;
          }
        );
      in
      {
        formatter = pkgs.nixfmt;

        packages.cargo-oxide = cargo-oxide;

        apps.default = {
          type = "app";
          program = "${cargo-oxide}/bin/cargo-oxide";
        };

        apps.new = {
          type = "app";
          program = "${new-project}/bin/cuda-oxide-new";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain

            llvmPkgs.clang
            llvmPkgs.libclang

            cudaSymlinked

            cargo-oxide
          ];

          # Transitive native deps of librustc_driver / LLVM that rust-lld
          # resolves when linking the rustc_codegen_cuda cdylib (-lffi, -lxml2,
          # -lzstd, -lz). Putting them in buildInputs lets cc-wrapper add the
          # right -L paths via NIX_LDFLAGS.
          buildInputs = with pkgs; [
            libffi
            libxml2
            zstd
            zlib
          ];

          shellHook = ''
            export CUDA_HOME="${cudaSymlinked}"
            export LIBCLANG_PATH="${llvmPkgs.libclang.lib}/lib"

            # GPU driver setup borrowed from https://github.com/NVlabs/cutile-rs
            # NixOS provides /run/opengl-driver/lib
            if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            else
              # Non-NixOS Linux (Ubuntu, Arch, etc. running Nix)
              # Symlink only the NVIDIA driver to avoid host glibc pollution
              _nv_drv_dir=$(mktemp -d /tmp/nix-nvidia-driver.XXXXXX)

              for d in /usr/lib/x86_64-linux-gnu \
                       /lib/x86_64-linux-gnu \
                       /usr/lib/aarch64-linux-gnu \
                       /lib/aarch64-linux-gnu \
                       /usr/lib \
                       /usr/lib64; do
                if [ -e "$d/libcuda.so.1" ]; then
                  for lib in "$d"/libcuda.so* "$d"/libnvidia-ptxjitcompiler.so* "$d"/libnvidia-gpucomp.so*; do
                    [ -e "$lib" ] && ln -sf "$lib" "$_nv_drv_dir/"
                  done
                  break
              fi
              done

              if [ -n "$(ls -A "$_nv_drv_dir" 2>/dev/null)" ]; then
                export LD_LIBRARY_PATH="$_nv_drv_dir:$LD_LIBRARY_PATH"
                # cleanup when the user exits the nix shell
                trap 'rm -rf "$_nv_drv_dir"' EXIT
              else
                # Clean up immediately if no NVIDIA drivers were found
                rm -rf "$_nv_drv_dir"
              fi
            fi

            echo "🦀 cuda-oxide dev environment loaded"
            echo " ✓ CUDA $(nvcc  --version | grep 'release'      | awk '{print $6}' | cut -c 2-)"
            echo " ✓ Rust $(rustc --version |                       awk '{print $2}')"
          '';
        };
      }
    ))
    // {
      templates.default = {
        path = templateSrc;
        description = "Reproducible CUDA + Rust dev environment for cuda-oxide projects";
      };
    };
}
