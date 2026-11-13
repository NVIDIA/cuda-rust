/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit CPU compilation followed by foreground CUDA module installation.

use super::*;
use cutile_compiler::kernel_naming::KernelNaming;

type SharedBinary = compilation_flights::Artifact<'static, TileFunctionKey, KernelBinary>;

pub(super) fn compile_shared(
    key: &TileFunctionKey,
    control: Option<&CompileControl>,
    build: impl FnOnce() -> Result<KernelBinary, Error>,
) -> Result<SharedBinary, Error> {
    static FLIGHTS: OnceLock<compilation_flights::Registry<TileFunctionKey, KernelBinary>> =
        OnceLock::new();
    FLIGHTS
        .get_or_init(Default::default)
        .compile(key.clone(), control, build)
}

/// An owned compilation request with the target and specialization resolved.
///
/// Create it on the foreground thread using [`Specialization::into_compile_job`].
/// A generated module provider is a function pointer, so its job can be moved to
/// a CPU worker. [`Self::compile`] builds its AST on the winning compiler thread
/// (or waits for another compilation of the same key) and never loads
/// a CUDA module, executes input operations, or synchronizes a CUDA stream.
pub struct KernelCompileJob<F: Fn() -> Module = ModuleAstFn> {
    specialization: Specialization<F>,
    capabilities: Arc<ToolkitCapabilities>,
    effective_options: CompileOptions,
}

impl<F: Fn() -> Module> Specialization<F> {
    /// Captures target/toolchain metadata for CPU-only preparation.
    ///
    /// This method can query the CUDA device. Call it on the foreground thread,
    /// after resolving the launch's actual input metadata with `specialize_on`.
    /// Unlike `compile_on`, the returned job's `compile` method does no GPU work.
    pub fn into_compile_job(self) -> Result<KernelCompileJob<F>, Error> {
        let capabilities = ToolkitCapabilities::for_device(self.key.device_id)?;
        if capabilities.tileiras_version != self.key.tileiras_fingerprint
            || capabilities.target.architecture != self.key.gpu_name
        {
            return Err(
                anyhow::anyhow!("toolchain changed while resolving compilation job").into(),
            );
        }
        let effective_options = self.key.compile_options.with_jit_env_overrides()?;
        Ok(KernelCompileJob {
            specialization: self,
            capabilities,
            effective_options,
        })
    }
}

impl<F: Fn() -> Module> KernelCompileJob<F> {
    pub fn key(&self) -> &TileFunctionKey {
        &self.specialization.key
    }

    /// Compiles the resolved specialization to a host-owned CUBIN.
    ///
    /// Ordinary launches and other jobs share this compilation by exact key.
    /// Whichever starts first compiles; the other callers wait. Keep the returned
    /// artifact alive until launch or explicit installation so it remains reusable.
    /// A cancelled waiter does not cancel the owner; a live waiter retries if the
    /// owner is interrupted. Compiler failures are reported, and later calls may retry.
    ///
    /// Cancellation is checked between frontend stages; the assembler is
    /// killable. A frontend pass already running is allowed to finish. Changing
    /// compiler environment variables or tile libraries during tuning is an
    /// error, not permission to install an artifact under a stale identity.
    pub fn compile(self, control: &CompileControl) -> Result<PreparedKernel, Error> {
        control.check().map_err(anyhow::Error::from)?;
        self.check_environment()?;
        let specialization = self.specialization;
        let key = specialization.key;
        let binary = compile_shared(&key, Some(control), || {
            let binary = compile_kernel_binary(
                &key,
                || (specialization.module_ast_fn)(),
                &self.effective_options,
                &specialization.tile_libraries,
                Arc::clone(&self.capabilities),
                Some(control),
            )?;
            control.check().map_err(anyhow::Error::from)?;
            check_environment(&key, &self.effective_options)?;
            if binary.links_tile_libraries && !specialization.tile_libraries.unchanged() {
                return Err(
                    anyhow::anyhow!("tile libraries changed during kernel preparation").into(),
                );
            }
            Ok(binary)
        })?;
        let prepared = PreparedKernel {
            key,
            binary,
            tile_libraries: specialization.tile_libraries,
            effective_options: self.effective_options,
        };
        prepared.check_environment()?;
        Ok(prepared)
    }

    fn check_environment(&self) -> Result<(), Error> {
        check_environment(&self.specialization.key, &self.effective_options)?;
        Ok(())
    }
}

/// A CUBIN and its launch validator, owned by the caller rather than loaded into
/// the process-wide CUDA cache. Ordinary launches can find and load it while this
/// handle is alive; explicit installation is optional. Dropping an unused result
/// unregisters its CPU artifact when the last caller releases it, with no GPU work.
///
/// Ready means compiled, not numerically validated or benchmarked. The pipeline
/// should keep only a bounded number of these artifacts alive.
pub struct PreparedKernel {
    key: TileFunctionKey,
    binary: SharedBinary,
    tile_libraries: TileLibrarySnapshot,
    effective_options: CompileOptions,
}

impl PreparedKernel {
    pub fn key(&self) -> &TileFunctionKey {
        &self.key
    }

    pub fn disk_cache_hit(&self) -> bool {
        matches!(self.binary.stage2_source, Stage2Source::DiskCache { .. })
    }

    fn check_environment(&self) -> Result<(), Error> {
        check_environment(&self.key, &self.effective_options)?;
        if self.binary.links_tile_libraries && !self.tile_libraries.unchanged() {
            return Err(anyhow::anyhow!("tile libraries changed during kernel preparation").into());
        }
        Ok(())
    }

    /// Loads the binary and installs it under its exact specialization key.
    ///
    /// Call on the foreground GPU lane before correctness checks, or simply
    /// launch the same specialization while retaining this handle. Normal
    /// launches still enforce the compiler's parameter and hoisted bounds
    /// checks. Returns `true` if a concurrent caller already populated L1.
    /// Loaded modules retain the normal cache lifetime; this API never evicts a
    /// potentially executing kernel. A driver-rejected disk entry is recompiled
    /// using the same toolchain and control, as in the ordinary JIT path.
    pub fn install(self, control: &CompileControl) -> Result<bool, Error> {
        control.check().map_err(anyhow::Error::from)?;
        self.check_environment()?;
        let key = self.key;
        let slot = kernel_cache_slot(&key);
        let mut reused = true;
        let result = slot.get_or_try_init(|| {
            reused = false;
            let kernel = load_kernel_binary(
                &self.binary,
                key.device_id,
                &KernelNaming::new(&key.function_name).entry_name(),
                &key.module_name,
                &key.function_name,
                &key.function_generics,
                &key.display_hash(),
                Some(control),
            )?;
            if kernel.1 && !self.tile_libraries.unchanged() {
                return Err(anyhow::anyhow!(
                    "tile libraries changed while installing prepared kernel"
                )
                .into());
            }
            control.check().map_err(anyhow::Error::from)?;
            record_jit_compile();
            Ok::<_, Error>(kernel)
        });
        if let Err(error) = result {
            drop(slot);
            get_kernel_cache().remove_if(&key, |_, cell| {
                cell.get().is_none() && Arc::strong_count(cell) == 1
            });
            return Err(error);
        }
        control.check().map_err(anyhow::Error::from)?;
        Ok(reused)
    }
}

fn check_environment(key: &TileFunctionKey, effective: &CompileOptions) -> Result<(), Error> {
    if bytecode_override() != key.bytecode_override
        || tileiras_fingerprint() != key.tileiras_fingerprint
        || key.compile_options.with_jit_env_overrides()? != *effective
    {
        return Err(
            anyhow::anyhow!("compiler configuration changed during kernel preparation").into(),
        );
    }
    Ok(())
}
