/*
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
*/

{
  description = "A cuda-oxide project";

  inputs = {
    cuda-oxide.url = "github:NVIDIA/cuda-rust?dir=cuda-oxide";
    # Reuse the development shell's inputs to avoid duplicate closures.
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
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          # Inherit CUDA, Rust, and the shellHook that discovers the host driver.
          inputsFrom = [ cuda-oxide.devShells.${system}.default ];
          packages = [
            # add project-specific packages here
          ];
        };
      }
    );
}
