{
  description = "Development shell for cutile-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        isDarwin = pkgs.stdenv.isDarwin;

        # CUDA 13.3 fetched from nvidia (headers needed on all platforms for
        # bindgen type generation; runtime libraries only needed on Linux)
        cudaRedistBase = "https://developer.download.nvidia.com/compute/cuda/redist";
        # NVIDIA publishes per-architecture redist archives; aarch64 (server) uses
        # the "linux-sbsa" variant, everything else uses "linux-x86_64".
        cudaRedistArch = if pkgs.stdenv.hostPlatform.isAarch64 then "linux-sbsa" else "linux-x86_64";
        fetchCudaRedist =
          { name, version, sha256 }:
          pkgs.fetchurl {
            url = "${cudaRedistBase}/${name}/${cudaRedistArch}/${name}-${cudaRedistArch}-${version}-archive.tar.xz";
            sha256 = sha256.${cudaRedistArch};
          };
        # Hashes come from NVIDIA's manifest for this release:
        # https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json
        # (per package: .<name>.<arch>.sha256). Update from there on version bumps.
        #
        # ORDER MATTERS: archives are extracted with --skip-old-files
        # (first-wins), so components must come before other archives that
        # may carry stale copies of their files — libnvvm before cuda_nvcc,
        # since both can populate nvvm/.
        cudaRedistArchives = [
          (fetchCudaRedist {
            name = "cuda_crt";
            version = "13.3.33";
            sha256 = {
              linux-x86_64 = "4755d36d24c6ef7697a2d3e1dbb23c4562c9c0d97d48390d4cbd8ab32dec5b5f";
              linux-sbsa = "6f6194918c00b980d8fd2111bf0aa004977760855c6e1528e0653bf4c889fbef";
            };
          })
          (fetchCudaRedist {
            name = "libnvvm";
            version = "13.3.33";
            sha256 = {
              linux-x86_64 = "fc9c1fd5844e44c0e5eeb051378c1b13cf0e3bb3fe4966d5103c38885424f802";
              linux-sbsa = "5f8ca5c9a10c3c9804b045960ee6192281efec4c7d83d5f3245ec2de8612118e";
            };
          })
          (fetchCudaRedist {
            name = "cuda_nvcc";
            version = "13.3.33";
            sha256 = {
              linux-x86_64 = "93b098bda4a562ebf3541523ce82adc43f106a81dcf28bcbf8f0d8e093d1c66f";
              linux-sbsa = "b5dde44aadd52234af3944ae3b2e74e811ad8e71fb600bcc9dfe6d8540353499";
            };
          })
          (fetchCudaRedist {
            name = "cuda_cudart";
            version = "13.3.29";
            sha256 = {
              linux-x86_64 = "1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d";
              linux-sbsa = "0cdd73d11885062daf3aa98ad4d7b8bd84f89b398be11f7054edea9ed31f597d";
            };
          })
          (fetchCudaRedist {
            name = "libcurand";
            version = "10.4.3.29";
            sha256 = {
              linux-x86_64 = "0218e62ab413e435dcd0274ec8e63b62214e6aba8519201061d1597e73caadbb";
              linux-sbsa = "3c2245e848ff8948663646ad7870cc8451b7cf1726758bedff7c123011126e4b";
            };
          })
          (fetchCudaRedist {
            name = "cuda_tileiras";
            version = "13.3.36";
            sha256 = {
              linux-x86_64 = "1b055db199f806c746d53331200ccd8480bfdddd14638ed2911f30ee0cc4447b";
              linux-sbsa = "98d163bd49de3c06fc179e5534fe4d8d5e1ad65800bf8696f79d2aeccafa039e";
            };
          })
        ];
        # All archives must be extracted into one real directory tree, NOT
        # merged with symlinkJoin: tileiras locates libnvvm relative to its
        # own dereferenced path (/proc/self/exe), so through a symlink farm
        # it would look for nvvm/lib64/libnvvm.so inside the cuda_tileiras
        # store path, where it doesn't exist, and fail with exit status 5.
        cudaToolkit = pkgs.runCommand "cuda-toolkit-13.3" { } ''
          mkdir -p $out
          # --skip-old-files: cross-archive path collisions are first-wins
          # (see order note above) and show up in the build log instead of
          # silently taking whichever archive extracted last.
          ${pkgs.lib.concatMapStrings (src: ''
            tar xf ${src} --strip-components=1 -C $out --skip-old-files --warning=existing-file
          '') cudaRedistArchives}
          # cuda-bindings/build.rs also looks for libs in lib64/
          if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then
            ln -s lib "$out/lib64"
          fi
          # The exe-relative lookup above is the invariant that broke in
          # issue #172 — fail the build loudly if assembly ever loses it.
          if [ ! -x "$out/bin/tileiras" ]; then
            echo "cuda-toolkit assembly error: bin/tileiras missing or not executable" >&2
            exit 1
          fi
          if [ ! -f "$out/nvvm/lib64/libnvvm.so" ]; then
            echo "cuda-toolkit assembly error: nvvm/lib64/libnvvm.so missing (tileiras resolves it relative to its own binary)" >&2
            exit 1
          fi
        '';

        # bindgen uses libclang to parse CUDA headers.
        libclang = pkgs.llvmPackages.libclang;
        libcDev = pkgs.lib.getDev pkgs.stdenv.cc.libc;
        bindgenClangArgs = pkgs.lib.optionalString (!isDarwin) "-isystem ${libcDev}/include";

        # Nightly Rust
        rustToolchain =
          pkgs.rust-bin.nightly."2025-07-16".default.override
            {
              extensions = [
                "clippy"
                "rust-analyzer"
                "rust-src"
                "rustfmt"
              ];
            };
      in
      {
        devShells.default = pkgs.mkShell {
          hardeningDisable = [ "fortify" ];

          packages = [
            rustToolchain
            libclang
            pkgs.cmake
            pkgs.git
            pkgs.libffi
            pkgs.libxml2
            pkgs.ninja
            pkgs.pkg-config
            pkgs.python3
            pkgs.which
          ];

          CMAKE_GENERATOR = "Ninja";
          CUDA_TOOLKIT_PATH = "${cudaToolkit}";
          LIBCLANG_PATH = "${libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = bindgenClangArgs;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
            pkgs.libffi
            pkgs.libxml2
            libclang.lib
          ] ++ pkgs.lib.optionals (!isDarwin) [ cudaToolkit ]);

          shellHook = pkgs.lib.optionalString (!isDarwin) ''
            export PATH="${cudaToolkit}/bin:$PATH"
            # GPU driver libs: NixOS provides /run/opengl-driver/lib; on other
            # distros, symlink just the NVIDIA libs into a temp dir so we don't
            # pull in the host glibc.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            else
              _nv_drv_dir=$(mktemp -d /tmp/nix-nvidia-driver.XXXXXX)
              # Search the host's own Debian multiarch dir (uname -m matches
              # the triplet prefix on x86_64 and aarch64), then generic dirs.
              for d in /usr/lib/$(uname -m)-linux-gnu /lib/$(uname -m)-linux-gnu /usr/lib /usr/lib64; do
                if [ -e "$d/libcuda.so.1" ]; then
                  for lib in "$d"/libcuda.so* "$d"/libnvidia-ptxjitcompiler.so* "$d"/libnvidia-gpucomp.so*; do
                    [ -e "$lib" ] && ln -sf "$lib" "$_nv_drv_dir/"
                  done
                  break
                fi
              done
              if [ -n "$(ls -A "$_nv_drv_dir" 2>/dev/null)" ]; then
                export LD_LIBRARY_PATH="$_nv_drv_dir:$LD_LIBRARY_PATH"
              else
                rm -rf "$_nv_drv_dir"
              fi
            fi
          '' + ''
            echo ""
            echo "cutile-rs dev shell"
            echo " ${if isDarwin then "~" else "✓"} CUDA  ${if isDarwin then "(headers only — no GPU required)" else "$CUDA_TOOLKIT_PATH"}"
            echo " ✓ Rust  $(rustc --version 2>/dev/null | awk '{print $2}')"
          '' + pkgs.lib.optionalString isDarwin ''
            echo ""
            echo "macOS: compile-only mode. Run:"
            echo "  cargo run -p cutile-examples --example compile_only"
          '' + ''
            echo ""
          '';
        };
      }
    );
}
