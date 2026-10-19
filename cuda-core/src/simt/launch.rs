/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel launch configuration.
//!
//! [`LaunchConfig`] bundles raw grid dimensions, block dimensions, and dynamic
//! shared memory size. Constructing one is harmless, but submitting it without
//! proving that it matches a kernel is unsafe. Use a rank-preserving config and
//! [`PreparedLaunch`] for the checked path.

use crate::{CudaFunction, CudaStream, DriverError};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;

/// Grid and block dimensions plus dynamic shared memory size for a kernel
/// launch.
///
/// Each dimension tuple is `(x, y, z)`. This is inert configuration data: it
/// does not know which kernel will consume it and therefore cannot prove the
/// kernel's indexing, resource, or synchronization assumptions. Passing it to
/// a raw launch is an unsafe operation.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// Grid dimensions `(x, y, z)` in blocks.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)` in threads.
    pub block_dim: (u32, u32, u32),
    /// Bytes of dynamic shared memory allocated per block.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a 1-D launch configuration for `n` elements.
    ///
    /// Uses a block size of 256 threads and computes the grid size via
    /// ceiling division. No dynamic shared memory is requested.
    ///
    /// Suitable for simple element-wise kernels where thread index maps
    /// directly to element index. The helper does not inspect a kernel, so it
    /// does not by itself make a raw launch safe.
    pub fn for_num_elems(n: u32) -> Self {
        const DEFAULT_BLOCK_SIZE: u32 = 256;
        let grid_x = n.div_ceil(DEFAULT_BLOCK_SIZE);
        LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A rank-preserving launch configuration accepted by a typed kernel contract.
///
/// This trait is sealed. Use [`LaunchConfig1D`], [`LaunchConfig2D`], or
/// [`LaunchConfig3D`]; downstream crates cannot provide a configuration that
/// bypasses their fixed trailing dimensions.
pub trait KernelLaunchConfig: sealed::Sealed + Copy {
    #[doc(hidden)]
    fn __raw(self) -> LaunchConfig;
}

/// A one-dimensional launch configuration.
///
/// The hidden `y` and `z` grid and block dimensions are always `1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig1D {
    grid_x: u32,
    block_x: u32,
    shared_mem_bytes: u32,
}

impl LaunchConfig1D {
    /// Creates a one-dimensional launch configuration.
    ///
    /// Zero dimensions are reported when the configuration is prepared for a
    /// concrete kernel, where the error can include that kernel's name.
    pub const fn new(grid_x: u32, block_x: u32, shared_mem_bytes: u32) -> Self {
        Self {
            grid_x,
            block_x,
            shared_mem_bytes,
        }
    }
}

impl sealed::Sealed for LaunchConfig1D {}

impl KernelLaunchConfig for LaunchConfig1D {
    fn __raw(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.grid_x, 1, 1),
            block_dim: (self.block_x, 1, 1),
            shared_mem_bytes: self.shared_mem_bytes,
        }
    }
}

/// A two-dimensional launch configuration.
///
/// The hidden `z` grid and block dimensions are always `1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig2D {
    grid: (u32, u32),
    block: (u32, u32),
    shared_mem_bytes: u32,
}

impl LaunchConfig2D {
    /// Creates a two-dimensional launch configuration.
    pub const fn new(grid: (u32, u32), block: (u32, u32), shared_mem_bytes: u32) -> Self {
        Self {
            grid,
            block,
            shared_mem_bytes,
        }
    }
}

impl sealed::Sealed for LaunchConfig2D {}

impl KernelLaunchConfig for LaunchConfig2D {
    fn __raw(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.grid.0, self.grid.1, 1),
            block_dim: (self.block.0, self.block.1, 1),
            shared_mem_bytes: self.shared_mem_bytes,
        }
    }
}

/// A three-dimensional launch configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig3D {
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
}

impl LaunchConfig3D {
    /// Creates a three-dimensional launch configuration.
    pub const fn new(grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem_bytes: u32) -> Self {
        Self {
            grid,
            block,
            shared_mem_bytes,
        }
    }
}

impl sealed::Sealed for LaunchConfig3D {}

impl KernelLaunchConfig for LaunchConfig3D {
    fn __raw(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: self.grid,
            block_dim: self.block,
            shared_mem_bytes: self.shared_mem_bytes,
        }
    }
}

/// The block shapes accepted by a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockRequirement {
    /// The launch must use exactly these `(x, y, z)` block dimensions.
    ///
    /// A kernel compiled by cuda-oxide also carries this shape as PTX
    /// `.reqntid`, so the CUDA driver rejects a differing block on any axis
    /// independently of this check. Preparation still reports the mismatch
    /// first, with the kernel name and both shapes.
    Exact((u32, u32, u32)),
    /// The block may contain at most this many threads in total.
    ///
    /// This matches CUDA `__launch_bounds__` / PTX `.maxntid` semantics: the
    /// product `block.x * block.y * block.z` is bounded, not each axis.
    /// `MaxThreads(256)` therefore accepts both `(256, 1, 1)` and
    /// `(16, 16, 1)`.
    MaxThreads(u32),
}

/// The dynamic shared-memory sizes accepted by a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicSharedMemoryRequirement {
    /// The launch must allocate exactly `bytes` bytes.
    Exact {
        /// Required bytes per block.
        bytes: u32,
        /// Minimum alignment assumed by the generated device code.
        min_alignment: u32,
    },
    /// The launch may allocate any size in the inclusive range.
    ///
    /// Preparation requires the live function/device to support `max_bytes`,
    /// even when one prepared configuration chooses fewer bytes. This proves
    /// the function's dynamic-memory capacity for the full interval and lets
    /// concurrent preparations install one monotonic, race-safe maximum.
    /// Geometry-dependent cluster or cooperative occupancy is still checked
    /// for each concrete prepared configuration.
    Range {
        /// Smallest valid allocation in bytes.
        min_bytes: u32,
        /// Largest valid allocation in bytes.
        max_bytes: u32,
        /// Minimum alignment assumed by the generated device code.
        min_alignment: u32,
    },
}

/// Coordinate widths whose range is guaranteed by launch preparation.
///
/// `U32` does not change CUDA's grid or block dimensions. It proves that each
/// per-axis global coordinate fits in `u32`, allowing a contracted kernel to
/// keep coordinate arithmetic narrow until the final address calculation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoordinateRequirement {
    /// Make no narrower-coordinate promise.
    #[default]
    Native,
    /// Every active-axis coordinate is representable by `u32`.
    U32,
}

/// An immutable description of a kernel's launch-time assumptions.
///
/// Macro-generated contract marker types expose one of these through
/// [`KernelLaunchContract::SPEC`]. The name is retained in every validation
/// error so failures point at the kernel whose assumptions were violated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchContractSpec {
    kernel_name: &'static str,
    block: BlockRequirement,
    dynamic_shared_memory: DynamicSharedMemoryRequirement,
    cluster: Option<(u32, u32, u32)>,
    cooperative: bool,
    min_compute_capability: Option<(u32, u32)>,
    coordinates: CoordinateRequirement,
}

impl LaunchContractSpec {
    /// Creates a contract without cluster, cooperative, or architecture
    /// requirements.
    pub const fn new(
        kernel_name: &'static str,
        block: BlockRequirement,
        dynamic_shared_memory: DynamicSharedMemoryRequirement,
    ) -> Self {
        Self {
            kernel_name,
            block,
            dynamic_shared_memory,
            cluster: None,
            cooperative: false,
            min_compute_capability: None,
            coordinates: CoordinateRequirement::Native,
        }
    }

    /// Requires fixed thread-block cluster dimensions.
    #[must_use]
    pub const fn with_cluster(mut self, cluster: (u32, u32, u32)) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// Requires a cooperative launch.
    #[must_use]
    pub const fn with_cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    /// Requires at least the supplied CUDA compute capability.
    #[must_use]
    pub const fn with_min_compute_capability(mut self, major: u32, minor: u32) -> Self {
        self.min_compute_capability = Some((major, minor));
        self
    }

    /// Proves that every global coordinate fits in `u32`.
    ///
    /// Preparation checks `grid_axis * block_axis <= 2^32` independently for
    /// X, Y, and Z. Exactly `2^32` positions is valid because the largest
    /// zero-based coordinate is `u32::MAX`.
    #[must_use]
    pub const fn with_u32_coordinates(mut self) -> Self {
        self.coordinates = CoordinateRequirement::U32;
        self
    }

    /// Returns the diagnostic kernel name.
    pub const fn kernel_name(&self) -> &'static str {
        self.kernel_name
    }

    /// Returns the block-shape requirement.
    pub const fn block(&self) -> BlockRequirement {
        self.block
    }

    /// Returns the dynamic shared-memory requirement.
    pub const fn dynamic_shared_memory(&self) -> DynamicSharedMemoryRequirement {
        self.dynamic_shared_memory
    }

    /// Returns the required cluster dimensions, if any.
    pub const fn cluster(&self) -> Option<(u32, u32, u32)> {
        self.cluster
    }

    /// Returns whether the kernel requires a cooperative launch.
    pub const fn cooperative(&self) -> bool {
        self.cooperative
    }

    /// Returns the minimum compute capability, if one was declared.
    pub const fn min_compute_capability(&self) -> Option<(u32, u32)> {
        self.min_compute_capability
    }

    /// Returns the coordinate-width requirement.
    pub const fn coordinates(&self) -> CoordinateRequirement {
        self.coordinates
    }
}

/// A macro-generated kernel marker implements this trait to bind a launch
/// rank to one immutable contract.
pub trait KernelLaunchContract {
    /// Rank-preserving host launch configuration for this kernel.
    type Config: KernelLaunchConfig;

    /// Compile-time launch assumptions emitted by the kernel macro.
    const SPEC: LaunchContractSpec;
}

/// Device launch limits that apply independently of a particular kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLaunchLimits {
    pub(crate) max_threads_per_block: u32,
    pub(crate) max_block_dim: (u32, u32, u32),
    pub(crate) max_grid_dim: (u32, u32, u32),
    pub(crate) max_shared_memory_per_block: u32,
}

impl DeviceLaunchLimits {
    /// Maximum threads in one block.
    pub const fn max_threads_per_block(&self) -> u32 {
        self.max_threads_per_block
    }

    /// Maximum block dimensions `(x, y, z)`.
    pub const fn max_block_dim(&self) -> (u32, u32, u32) {
        self.max_block_dim
    }

    /// Maximum grid dimensions `(x, y, z)`.
    pub const fn max_grid_dim(&self) -> (u32, u32, u32) {
        self.max_grid_dim
    }

    /// Portable shared-memory limit per block, including static and dynamic
    /// shared memory.
    pub const fn max_shared_memory_per_block(&self) -> u32 {
        self.max_shared_memory_per_block
    }
}

/// Which launch dimension is named by a validation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchDimension {
    /// Grid dimensions, measured in blocks.
    Grid,
    /// Block dimensions, measured in threads.
    Block,
    /// Thread-block cluster dimensions, measured in blocks.
    Cluster,
}

/// Axis named by a dimension validation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchAxis {
    /// X axis.
    X,
    /// Y axis.
    Y,
    /// Z axis.
    Z,
}

/// Why preparation of a typed kernel launch failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum LaunchContractError {
    /// The contract omitted a useful kernel name.
    EmptyKernelName,
    /// A grid, block, or cluster dimension was zero.
    ZeroDimension {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Dimension group containing the zero.
        dimension: LaunchDimension,
        /// Axis containing the zero.
        axis: LaunchAxis,
    },
    /// Multiplying a three-dimensional shape exceeded the representable range.
    DimensionProductOverflow {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Shape whose product overflowed.
        dimension: LaunchDimension,
    },
    /// A per-axis global coordinate cannot be represented by `u32`.
    CoordinateRangeExceedsU32 {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Axis whose coordinate range is too large.
        axis: LaunchAxis,
        /// Number of blocks on this axis.
        grid: u32,
        /// Number of threads per block on this axis.
        block: u32,
        /// Number of coordinate values required on this axis.
        positions: u64,
    },
    /// A shared-memory alignment was not a nonzero power of two.
    InvalidSharedMemoryAlignment {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Invalid alignment.
        alignment: u32,
    },
    /// A shared-memory range had its endpoints reversed.
    InvalidSharedMemoryRange {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Inclusive lower bound.
        min_bytes: u32,
        /// Inclusive upper bound.
        max_bytes: u32,
    },
    /// The launch block did not equal the contract's exact block.
    BlockShapeMismatch {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Required block dimensions.
        required: (u32, u32, u32),
        /// Requested block dimensions.
        actual: (u32, u32, u32),
    },
    /// The block contained more threads than the contract permits.
    BlockThreadsExceedContract {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested threads per block.
        actual: u64,
        /// Contract maximum threads per block.
        max: u32,
    },
    /// Dynamic shared memory did not equal the contract's exact size.
    DynamicSharedMemoryExactMismatch {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Required byte count.
        required: u32,
        /// Requested byte count.
        actual: u32,
    },
    /// Dynamic shared memory fell outside the contract's inclusive range.
    DynamicSharedMemoryOutsideRange {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Inclusive lower bound.
        min: u32,
        /// Inclusive upper bound.
        max: u32,
        /// Requested byte count.
        actual: u32,
    },
    /// A cluster axis did not evenly divide the corresponding grid axis.
    ClusterDoesNotDivideGrid {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Axis that is not divisible.
        axis: LaunchAxis,
        /// Requested grid axis size.
        grid: u32,
        /// Required cluster axis size.
        cluster: u32,
    },
    /// A launch dimension exceeded a live device limit.
    DeviceDimensionExceeded {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Grid or block dimensions.
        dimension: LaunchDimension,
        /// Axis that exceeded its limit.
        axis: LaunchAxis,
        /// Requested size.
        actual: u32,
        /// Device maximum.
        max: u32,
    },
    /// The block contained more threads than the device permits.
    DeviceThreadsPerBlockExceeded {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested threads per block.
        actual: u64,
        /// Device maximum.
        max: u32,
    },
    /// The block contained more threads than this function permits.
    FunctionThreadsPerBlockExceeded {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested threads per block.
        actual: u64,
        /// Function maximum.
        max: u32,
    },
    /// Static plus dynamic shared-memory arithmetic overflowed.
    SharedMemoryTotalOverflow {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Function's static allocation.
        static_bytes: u32,
        /// Requested dynamic allocation.
        dynamic_bytes: u32,
    },
    /// Static plus dynamic shared memory exceeded the device limit.
    DeviceSharedMemoryExceeded {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested static plus dynamic bytes.
        total: u64,
        /// Device maximum.
        max: u32,
        /// Whether `max` is the device's opt-in limit.
        opt_in: bool,
    },
    /// The device compute capability is below the contract minimum.
    ComputeCapabilityTooLow {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Contract minimum.
        required: (u32, u32),
        /// Live device capability.
        actual: (u32, u32),
    },
    /// The live device does not support cooperative launch.
    CooperativeLaunchUnsupported {
        /// Kernel being prepared.
        kernel: &'static str,
    },
    /// The live device does not support thread-block clusters.
    ClusterLaunchUnsupported {
        /// Kernel being prepared.
        kernel: &'static str,
    },
    /// The requested cluster contains more blocks than this function and
    /// launch shape support on the live device.
    ClusterSizeExceeded {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested blocks per cluster.
        blocks: u64,
        /// Maximum blocks per cluster reported by occupancy.
        max: u32,
    },
    /// The contract disagrees with cluster dimensions compiled into the
    /// function.
    FunctionClusterShapeMismatch {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Cluster dimensions declared by the host contract.
        declared: (u32, u32, u32),
        /// Cluster dimensions required by the compiled function.
        required: (u32, u32, u32),
    },
    /// The host contract declares a fixed cluster but the compiled function
    /// does not contain matching required-cluster metadata.
    RequiredClusterDimensionsMissing {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Cluster dimensions declared by the host contract.
        declared: (u32, u32, u32),
    },
    /// CUDA rejected the concrete cluster shape for this function and launch
    /// configuration.
    ClusterShapeUnsupported {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested cluster dimensions.
        cluster: (u32, u32, u32),
    },
    /// CUDA reported that no cluster of the requested shape can be resident.
    ClusterHasNoResidency {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested cluster dimensions.
        cluster: (u32, u32, u32),
    },
    /// A cooperative grid cannot be fully resident on the device.
    CooperativeGridTooLarge {
        /// Kernel being prepared.
        kernel: &'static str,
        /// Requested number of blocks.
        blocks: u64,
        /// Maximum resident blocks for this function and launch shape.
        resident_capacity: u64,
    },
    /// A stream belongs to a different CUDA context from the prepared function.
    ContextMismatch {
        /// Kernel being launched.
        kernel: &'static str,
        /// Function device ordinal.
        function_device: usize,
        /// Stream device ordinal.
        stream_device: usize,
    },
    /// A declared `requires` size requirement over the kernel's
    /// parameters did not hold at launch time.
    SizeRequirementViolated {
        /// Kernel being launched.
        kernel: &'static str,
        /// Source text of the violated relation.
        relation: &'static str,
        /// Evaluated left-hand side of the comparison, widened to `u64`.
        lhs: u64,
        /// Evaluated right-hand side of the comparison, widened to `u64`.
        rhs: u64,
    },
    /// Arithmetic inside a `requires` size requirement left the `u64`
    /// range, so the relation could not be evaluated.
    SizeRequirementOverflow {
        /// Kernel being launched.
        kernel: &'static str,
        /// Source text of the relation whose arithmetic overflowed.
        relation: &'static str,
    },
    /// A CUDA driver query or one-time function configuration failed.
    Driver(DriverError),
}

impl Display for LaunchContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyKernelName => write!(f, "kernel launch contract has an empty name"),
            Self::ZeroDimension {
                kernel,
                dimension,
                axis,
            } => write!(f, "{kernel}: {dimension:?}.{axis:?} must be nonzero"),
            Self::DimensionProductOverflow { kernel, dimension } => {
                write!(f, "{kernel}: {dimension:?} dimension product overflowed")
            }
            Self::CoordinateRangeExceedsU32 {
                kernel,
                axis,
                grid,
                block,
                positions,
            } => write!(
                f,
                "{kernel}: Grid.{axis:?} {grid} * Block.{axis:?} {block} requires {positions} coordinates, exceeding the u32 range"
            ),
            Self::InvalidSharedMemoryAlignment { kernel, alignment } => write!(
                f,
                "{kernel}: dynamic shared-memory alignment {alignment} is not a nonzero power of two"
            ),
            Self::InvalidSharedMemoryRange {
                kernel,
                min_bytes,
                max_bytes,
            } => write!(
                f,
                "{kernel}: dynamic shared-memory range {min_bytes}..={max_bytes} is invalid"
            ),
            Self::BlockShapeMismatch {
                kernel,
                required,
                actual,
            } => write!(
                f,
                "{kernel}: block {actual:?} does not match required block {required:?}"
            ),
            Self::BlockThreadsExceedContract {
                kernel,
                actual,
                max,
            } => write!(
                f,
                "{kernel}: block has {actual} threads; contract maximum is {max}"
            ),
            Self::DynamicSharedMemoryExactMismatch {
                kernel,
                required,
                actual,
            } => write!(
                f,
                "{kernel}: dynamic shared memory is {actual} bytes; contract requires {required}"
            ),
            Self::DynamicSharedMemoryOutsideRange {
                kernel,
                min,
                max,
                actual,
            } => write!(
                f,
                "{kernel}: dynamic shared memory is {actual} bytes; contract permits {min}..={max}"
            ),
            Self::ClusterDoesNotDivideGrid {
                kernel,
                axis,
                grid,
                cluster,
            } => write!(
                f,
                "{kernel}: cluster {axis:?} size {cluster} does not divide grid size {grid}"
            ),
            Self::DeviceDimensionExceeded {
                kernel,
                dimension,
                axis,
                actual,
                max,
            } => write!(
                f,
                "{kernel}: {dimension:?}.{axis:?} size {actual} exceeds device maximum {max}"
            ),
            Self::DeviceThreadsPerBlockExceeded {
                kernel,
                actual,
                max,
            } => write!(
                f,
                "{kernel}: block has {actual} threads; device maximum is {max}"
            ),
            Self::FunctionThreadsPerBlockExceeded {
                kernel,
                actual,
                max,
            } => write!(
                f,
                "{kernel}: block has {actual} threads; function maximum is {max}"
            ),
            Self::SharedMemoryTotalOverflow {
                kernel,
                static_bytes,
                dynamic_bytes,
            } => write!(
                f,
                "{kernel}: {static_bytes} static + {dynamic_bytes} dynamic shared-memory bytes overflowed"
            ),
            Self::DeviceSharedMemoryExceeded {
                kernel,
                total,
                max,
                opt_in,
            } => write!(
                f,
                "{kernel}: {total} shared-memory bytes exceed the device {} limit {max}",
                if *opt_in { "opt-in" } else { "portable" }
            ),
            Self::ComputeCapabilityTooLow {
                kernel,
                required,
                actual,
            } => write!(
                f,
                "{kernel}: compute capability {}.{} is below required {}.{}",
                actual.0, actual.1, required.0, required.1
            ),
            Self::CooperativeLaunchUnsupported { kernel } => {
                write!(f, "{kernel}: device does not support cooperative launch")
            }
            Self::ClusterLaunchUnsupported { kernel } => {
                write!(f, "{kernel}: device does not support cluster launch")
            }
            Self::ClusterSizeExceeded {
                kernel,
                blocks,
                max,
            } => write!(
                f,
                "{kernel}: cluster has {blocks} blocks; live maximum is {max}"
            ),
            Self::FunctionClusterShapeMismatch {
                kernel,
                declared,
                required,
            } => write!(
                f,
                "{kernel}: declared cluster {declared:?} does not match function-required cluster {required:?}"
            ),
            Self::RequiredClusterDimensionsMissing { kernel, declared } => write!(
                f,
                "{kernel}: declared cluster {declared:?} is missing from the compiled function metadata"
            ),
            Self::ClusterShapeUnsupported { kernel, cluster } => write!(
                f,
                "{kernel}: cluster shape {cluster:?} is unsupported for this launch"
            ),
            Self::ClusterHasNoResidency { kernel, cluster } => write!(
                f,
                "{kernel}: no cluster with shape {cluster:?} can be resident"
            ),
            Self::CooperativeGridTooLarge {
                kernel,
                blocks,
                resident_capacity,
            } => write!(
                f,
                "{kernel}: cooperative grid has {blocks} blocks but only {resident_capacity} can be resident"
            ),
            Self::ContextMismatch {
                kernel,
                function_device,
                stream_device,
            } => write!(
                f,
                "{kernel}: function is on device {function_device}, stream is on device {stream_device}"
            ),
            Self::SizeRequirementViolated {
                kernel,
                relation,
                lhs,
                rhs,
            } => write!(
                f,
                "{kernel}: size requirement `{relation}` violated: left-hand side is {lhs}, right-hand side is {rhs}"
            ),
            Self::SizeRequirementOverflow { kernel, relation } => write!(
                f,
                "{kernel}: arithmetic in size requirement `{relation}` overflowed the u64 range"
            ),
            Self::Driver(error) => Display::fmt(error, f),
        }
    }
}

impl Error for LaunchContractError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DriverError> for LaunchContractError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

/// A kernel function and launch configuration whose contract has been checked.
///
/// Preparation performs all device/function resource queries. Reusing this
/// value performs no contract query; it only compares the stream's context
/// handle before the normal launch path binds that context.
pub struct PreparedLaunch<C: KernelLaunchContract> {
    function: CudaFunction,
    config: LaunchConfig,
    _contract: PhantomData<fn(C) -> C>,
}

impl<C: KernelLaunchContract> Clone for PreparedLaunch<C> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            config: self.config,
            _contract: PhantomData,
        }
    }
}

impl<C: KernelLaunchContract> PreparedLaunch<C> {
    /// Validates and prepares one generated kernel launch.
    ///
    /// This entry point is public only so `#[cuda_module]` expansions in
    /// downstream crates can call it. Applications should use the generated
    /// module constructor.
    ///
    /// # Safety
    ///
    /// `C::SPEC` must truthfully describe the compiled device function,
    /// including its rank, fixed block assumptions, shared-memory layout, and
    /// launch mode. cuda-oxide's macros generate that marker and uphold this
    /// relationship.
    #[doc(hidden)]
    pub unsafe fn __prepare(
        function: CudaFunction,
        config: C::Config,
    ) -> Result<Self, LaunchContractError> {
        let raw = config.__raw();
        validate_static(C::SPEC, raw)?;

        let context = function.context();
        let limits = context.launch_limits()?;
        let function_max_threads = function.max_threads_per_block()?;
        let static_shared = function.static_shared_memory_bytes()?;
        let function_max_dynamic = function.max_dynamic_shared_memory_bytes()?;

        validate_live_shape(C::SPEC, raw, limits, function_max_threads)?;

        // Configure the immutable contract maximum, not this particular
        // launch's chosen size. Concurrent preparations of two range values
        // therefore write the same function attribute and cannot lower the
        // maximum underneath an already prepared launch.
        let contract_dynamic_max = dynamic_shared_memory_max(C::SPEC.dynamic_shared_memory);
        let total_shared =
            shared_memory_total(C::SPEC.kernel_name, static_shared, contract_dynamic_max)?;
        if validate_shared_memory_limit(
            C::SPEC.kernel_name,
            total_shared,
            limits.max_shared_memory_per_block,
            false,
        )
        .is_err()
        {
            let opt_in_max = context.max_opt_in_shared_memory_per_block()?;
            validate_shared_memory_limit(C::SPEC.kernel_name, total_shared, opt_in_max, true)?;
        }

        if let Some(required) = C::SPEC.min_compute_capability {
            let (major, minor) = context.compute_capability()?;
            let actual = (
                u32::try_from(major).map_err(|_| {
                    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE)
                })?,
                u32::try_from(minor).map_err(|_| {
                    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE)
                })?,
            );
            validate_compute_capability(C::SPEC.kernel_name, required, actual)?;
        }

        if let Some(cluster) = C::SPEC.cluster {
            validate_cluster_support(C::SPEC.kernel_name, context.supports_cluster_launch()?)?;
            validate_required_cluster(
                C::SPEC.kernel_name,
                cluster,
                function.required_cluster_dimensions()?,
            )?;
        }

        if C::SPEC.cooperative {
            validate_cooperative_support(
                C::SPEC.kernel_name,
                context.supports_cooperative_launch()?,
            )?;
        }

        // Perform the only persistent function mutation after all checks that
        // do not themselves depend on the opted-in maximum. Cluster occupancy
        // is checked next and may leave this monotonic increase in place when
        // the concrete cluster shape is rejected.
        if contract_dynamic_max > function_max_dynamic {
            function.set_max_dynamic_shared_memory_bytes(contract_dynamic_max)?;
        }

        // Cluster occupancy is reused for cooperative residency when both
        // modes are declared: a cooperative clustered grid must fit entirely
        // in the concurrent cluster capacity reported by
        // `cuOccupancyMaxActiveClusters`.
        let mut clustered_resident_blocks: Option<u64> = None;

        if let Some(cluster) = C::SPEC.cluster {
            let max_cluster_size = function.max_potential_cluster_size(
                raw.grid_dim,
                raw.block_dim,
                raw.shared_mem_bytes,
            )?;
            let cluster_blocks =
                shape_product(C::SPEC.kernel_name, LaunchDimension::Cluster, cluster)?;
            validate_cluster_size(C::SPEC.kernel_name, cluster_blocks, max_cluster_size)?;

            let active_clusters = match function.max_active_clusters(
                raw.grid_dim,
                raw.block_dim,
                raw.shared_mem_bytes,
                cluster,
            ) {
                Ok(active_clusters) => active_clusters,
                Err(error)
                    if error.0 == cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CLUSTER_SIZE =>
                {
                    return Err(LaunchContractError::ClusterShapeUnsupported {
                        kernel: C::SPEC.kernel_name,
                        cluster,
                    });
                }
                Err(error) => return Err(error.into()),
            };
            validate_cluster_residency(C::SPEC.kernel_name, cluster, active_clusters)?;
            // Deliberate error reuse: `active_clusters * cluster_blocks` is a
            // device-capacity product over the cluster dimension, and
            // `DimensionProductOverflow { Cluster }` is the closest existing
            // diagnostic. The arm is unreachable in practice: both factors fit
            // in u32 (`active_clusters` is driver-reported, `cluster_blocks`
            // was validated against the u32 `max_cluster_size` above), so the
            // product always fits in u64. `checked_mul` keeps the math honest.
            clustered_resident_blocks = Some(
                u64::from(active_clusters)
                    .checked_mul(cluster_blocks)
                    .ok_or(LaunchContractError::DimensionProductOverflow {
                        kernel: C::SPEC.kernel_name,
                        dimension: LaunchDimension::Cluster,
                    })?,
            );
        }

        if C::SPEC.cooperative {
            let blocks = shape_product(C::SPEC.kernel_name, LaunchDimension::Grid, raw.grid_dim)?;
            let resident_capacity = if let Some(capacity) = clustered_resident_blocks {
                capacity
            } else {
                let block_threads =
                    shape_product(C::SPEC.kernel_name, LaunchDimension::Block, raw.block_dim)?;
                let active_per_sm = function.max_active_blocks_per_multiprocessor(
                    block_threads as u32,
                    raw.shared_mem_bytes,
                )?;
                let multiprocessors = context.multiprocessor_count()?;
                u64::from(active_per_sm) * u64::from(multiprocessors)
            };
            validate_cooperative_residency(C::SPEC.kernel_name, blocks, resident_capacity)?;
        }

        Ok(Self {
            function,
            config: raw,
            _contract: PhantomData,
        })
    }

    /// Returns the prepared CUDA function.
    pub fn function(&self) -> &CudaFunction {
        &self.function
    }

    /// Returns a copy of the validated raw launch configuration.
    ///
    /// This is hidden because generated launch methods are its intended
    /// consumer. Mutating the copy cannot change the prepared launch.
    #[doc(hidden)]
    pub fn __raw_config(&self) -> LaunchConfig {
        self.config
    }

    /// Rejects a stream from a different context without making a driver call.
    pub fn validate_stream(&self, stream: &CudaStream) -> Result<(), LaunchContractError> {
        let function_context = self.function.context();
        let stream_context = stream.context();
        if function_context.cu_ctx() == stream_context.cu_ctx() {
            Ok(())
        } else {
            Err(LaunchContractError::ContextMismatch {
                kernel: C::SPEC.kernel_name,
                function_device: function_context.ordinal(),
                stream_device: stream_context.ordinal(),
            })
        }
    }
}

fn validate_static(
    spec: LaunchContractSpec,
    config: LaunchConfig,
) -> Result<(), LaunchContractError> {
    if spec.kernel_name.trim().is_empty() {
        return Err(LaunchContractError::EmptyKernelName);
    }

    validate_shape(spec.kernel_name, LaunchDimension::Grid, config.grid_dim)?;
    validate_shape(spec.kernel_name, LaunchDimension::Block, config.block_dim)?;

    if spec.coordinates == CoordinateRequirement::U32 {
        validate_u32_coordinates(spec.kernel_name, config.grid_dim, config.block_dim)?;
    }

    match spec.block {
        BlockRequirement::Exact(required) => {
            validate_shape(spec.kernel_name, LaunchDimension::Block, required)?;
            if config.block_dim != required {
                return Err(LaunchContractError::BlockShapeMismatch {
                    kernel: spec.kernel_name,
                    required,
                    actual: config.block_dim,
                });
            }
        }
        BlockRequirement::MaxThreads(max) => {
            let actual = shape_product(spec.kernel_name, LaunchDimension::Block, config.block_dim)?;
            if actual > u64::from(max) {
                return Err(LaunchContractError::BlockThreadsExceedContract {
                    kernel: spec.kernel_name,
                    actual,
                    max,
                });
            }
        }
    }

    match spec.dynamic_shared_memory {
        DynamicSharedMemoryRequirement::Exact {
            bytes,
            min_alignment,
        } => {
            validate_alignment(spec.kernel_name, min_alignment)?;
            if config.shared_mem_bytes != bytes {
                return Err(LaunchContractError::DynamicSharedMemoryExactMismatch {
                    kernel: spec.kernel_name,
                    required: bytes,
                    actual: config.shared_mem_bytes,
                });
            }
        }
        DynamicSharedMemoryRequirement::Range {
            min_bytes,
            max_bytes,
            min_alignment,
        } => {
            validate_alignment(spec.kernel_name, min_alignment)?;
            if min_bytes > max_bytes {
                return Err(LaunchContractError::InvalidSharedMemoryRange {
                    kernel: spec.kernel_name,
                    min_bytes,
                    max_bytes,
                });
            }
            if !(min_bytes..=max_bytes).contains(&config.shared_mem_bytes) {
                return Err(LaunchContractError::DynamicSharedMemoryOutsideRange {
                    kernel: spec.kernel_name,
                    min: min_bytes,
                    max: max_bytes,
                    actual: config.shared_mem_bytes,
                });
            }
        }
    }

    if let Some(cluster) = spec.cluster {
        validate_shape(spec.kernel_name, LaunchDimension::Cluster, cluster)?;
        for (axis, grid, cluster) in axes(config.grid_dim, cluster) {
            if grid % cluster != 0 {
                return Err(LaunchContractError::ClusterDoesNotDivideGrid {
                    kernel: spec.kernel_name,
                    axis,
                    grid,
                    cluster,
                });
            }
        }
    }

    Ok(())
}

fn validate_u32_coordinates(
    kernel: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
) -> Result<(), LaunchContractError> {
    const U32_COORDINATE_COUNT: u64 = u32::MAX as u64 + 1;

    for (axis, grid, block) in axes(grid, block) {
        let positions = u64::from(grid) * u64::from(block);
        if positions > U32_COORDINATE_COUNT {
            return Err(LaunchContractError::CoordinateRangeExceedsU32 {
                kernel,
                axis,
                grid,
                block,
                positions,
            });
        }
    }
    Ok(())
}

fn validate_live_shape(
    spec: LaunchContractSpec,
    config: LaunchConfig,
    limits: DeviceLaunchLimits,
    function_max_threads: u32,
) -> Result<(), LaunchContractError> {
    validate_axes(config.grid_dim, limits.max_grid_dim, |axis, actual, max| {
        LaunchContractError::DeviceDimensionExceeded {
            kernel: spec.kernel_name,
            dimension: LaunchDimension::Grid,
            axis,
            actual,
            max,
        }
    })?;
    validate_axes(
        config.block_dim,
        limits.max_block_dim,
        |axis, actual, max| LaunchContractError::DeviceDimensionExceeded {
            kernel: spec.kernel_name,
            dimension: LaunchDimension::Block,
            axis,
            actual,
            max,
        },
    )?;

    let threads = shape_product(spec.kernel_name, LaunchDimension::Block, config.block_dim)?;
    if threads > u64::from(limits.max_threads_per_block) {
        return Err(LaunchContractError::DeviceThreadsPerBlockExceeded {
            kernel: spec.kernel_name,
            actual: threads,
            max: limits.max_threads_per_block,
        });
    }
    if threads > u64::from(function_max_threads) {
        return Err(LaunchContractError::FunctionThreadsPerBlockExceeded {
            kernel: spec.kernel_name,
            actual: threads,
            max: function_max_threads,
        });
    }

    Ok(())
}

fn shared_memory_total(
    kernel: &'static str,
    static_bytes: u32,
    dynamic_bytes: u32,
) -> Result<u64, LaunchContractError> {
    static_bytes
        .checked_add(dynamic_bytes)
        .map(u64::from)
        .ok_or(LaunchContractError::SharedMemoryTotalOverflow {
            kernel,
            static_bytes,
            dynamic_bytes,
        })
}

const fn dynamic_shared_memory_max(requirement: DynamicSharedMemoryRequirement) -> u32 {
    match requirement {
        DynamicSharedMemoryRequirement::Exact { bytes, .. } => bytes,
        DynamicSharedMemoryRequirement::Range { max_bytes, .. } => max_bytes,
    }
}

fn validate_shared_memory_limit(
    kernel: &'static str,
    total: u64,
    max: u32,
    opt_in: bool,
) -> Result<(), LaunchContractError> {
    if total <= u64::from(max) {
        Ok(())
    } else {
        Err(LaunchContractError::DeviceSharedMemoryExceeded {
            kernel,
            total,
            max,
            opt_in,
        })
    }
}

fn validate_compute_capability(
    kernel: &'static str,
    required: (u32, u32),
    actual: (u32, u32),
) -> Result<(), LaunchContractError> {
    if actual >= required {
        Ok(())
    } else {
        Err(LaunchContractError::ComputeCapabilityTooLow {
            kernel,
            required,
            actual,
        })
    }
}

fn validate_cluster_support(
    kernel: &'static str,
    supported: bool,
) -> Result<(), LaunchContractError> {
    if supported {
        Ok(())
    } else {
        Err(LaunchContractError::ClusterLaunchUnsupported { kernel })
    }
}

fn validate_cluster_size(
    kernel: &'static str,
    blocks: u64,
    max: u32,
) -> Result<(), LaunchContractError> {
    if blocks <= u64::from(max) {
        Ok(())
    } else {
        Err(LaunchContractError::ClusterSizeExceeded {
            kernel,
            blocks,
            max,
        })
    }
}

fn validate_required_cluster(
    kernel: &'static str,
    declared: (u32, u32, u32),
    required: Option<(u32, u32, u32)>,
) -> Result<(), LaunchContractError> {
    match required {
        Some(required) if declared == required => Ok(()),
        Some(required) => Err(LaunchContractError::FunctionClusterShapeMismatch {
            kernel,
            declared,
            required,
        }),
        None => Err(LaunchContractError::RequiredClusterDimensionsMissing { kernel, declared }),
    }
}

fn validate_cluster_residency(
    kernel: &'static str,
    cluster: (u32, u32, u32),
    active_clusters: u32,
) -> Result<(), LaunchContractError> {
    if active_clusters != 0 {
        Ok(())
    } else {
        Err(LaunchContractError::ClusterHasNoResidency { kernel, cluster })
    }
}

fn validate_cooperative_support(
    kernel: &'static str,
    supported: bool,
) -> Result<(), LaunchContractError> {
    if supported {
        Ok(())
    } else {
        Err(LaunchContractError::CooperativeLaunchUnsupported { kernel })
    }
}

fn validate_cooperative_residency(
    kernel: &'static str,
    blocks: u64,
    resident_capacity: u64,
) -> Result<(), LaunchContractError> {
    if blocks <= resident_capacity {
        Ok(())
    } else {
        Err(LaunchContractError::CooperativeGridTooLarge {
            kernel,
            blocks,
            resident_capacity,
        })
    }
}

fn validate_alignment(kernel: &'static str, alignment: u32) -> Result<(), LaunchContractError> {
    if alignment.is_power_of_two() {
        Ok(())
    } else {
        Err(LaunchContractError::InvalidSharedMemoryAlignment { kernel, alignment })
    }
}

fn validate_shape(
    kernel: &'static str,
    dimension: LaunchDimension,
    shape: (u32, u32, u32),
) -> Result<(), LaunchContractError> {
    for (axis, value, _) in axes(shape, shape) {
        if value == 0 {
            return Err(LaunchContractError::ZeroDimension {
                kernel,
                dimension,
                axis,
            });
        }
    }
    shape_product(kernel, dimension, shape)?;
    Ok(())
}

fn shape_product(
    kernel: &'static str,
    dimension: LaunchDimension,
    shape: (u32, u32, u32),
) -> Result<u64, LaunchContractError> {
    u64::from(shape.0)
        .checked_mul(u64::from(shape.1))
        .and_then(|xy| xy.checked_mul(u64::from(shape.2)))
        .ok_or(LaunchContractError::DimensionProductOverflow { kernel, dimension })
}

fn axes(actual: (u32, u32, u32), limit: (u32, u32, u32)) -> [(LaunchAxis, u32, u32); 3] {
    [
        (LaunchAxis::X, actual.0, limit.0),
        (LaunchAxis::Y, actual.1, limit.1),
        (LaunchAxis::Z, actual.2, limit.2),
    ]
}

fn validate_axes(
    actual: (u32, u32, u32),
    limit: (u32, u32, u32),
    error: impl Fn(LaunchAxis, u32, u32) -> LaunchContractError,
) -> Result<(), LaunchContractError> {
    for (axis, actual, max) in axes(actual, limit) {
        if actual > max {
            return Err(error(axis, actual, max));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "test_kernel";

    struct NonCloneContract;

    impl KernelLaunchContract for NonCloneContract {
        type Config = LaunchConfig1D;

        const SPEC: LaunchContractSpec = LaunchContractSpec::new(
            "non_clone",
            BlockRequirement::Exact((1, 1, 1)),
            DynamicSharedMemoryRequirement::Exact {
                bytes: 0,
                min_alignment: 1,
            },
        );
    }

    fn exact_spec(block: (u32, u32, u32)) -> LaunchContractSpec {
        LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::Exact(block),
            DynamicSharedMemoryRequirement::Exact {
                bytes: 0,
                min_alignment: 1,
            },
        )
    }

    fn raw(
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
    ) -> LaunchConfig {
        LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes,
        }
    }

    fn generous_limits() -> DeviceLaunchLimits {
        DeviceLaunchLimits {
            max_threads_per_block: 1024,
            max_block_dim: (1024, 1024, 64),
            max_grid_dim: (u32::MAX, 65_535, 65_535),
            max_shared_memory_per_block: 48 * 1024,
        }
    }

    #[test]
    fn typed_configs_fix_trailing_dimensions() {
        let one = LaunchConfig1D::new(7, 64, 16).__raw();
        assert_eq!(one.grid_dim, (7, 1, 1));
        assert_eq!(one.block_dim, (64, 1, 1));
        assert_eq!(one.shared_mem_bytes, 16);

        let two = LaunchConfig2D::new((7, 5), (16, 8), 32).__raw();
        assert_eq!(two.grid_dim, (7, 5, 1));
        assert_eq!(two.block_dim, (16, 8, 1));

        let three = LaunchConfig3D::new((7, 5, 3), (16, 8, 2), 64).__raw();
        assert_eq!(three.grid_dim, (7, 5, 3));
        assert_eq!(three.block_dim, (16, 8, 2));
    }

    #[test]
    fn prepared_launch_clone_does_not_require_clone_brand() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<PreparedLaunch<NonCloneContract>>();
    }

    #[test]
    fn rejects_zero_and_overflowing_shapes() {
        let zero = validate_static(exact_spec((32, 1, 1)), raw((0, 1, 1), (32, 1, 1), 0));
        assert!(matches!(
            zero,
            Err(LaunchContractError::ZeroDimension {
                dimension: LaunchDimension::Grid,
                axis: LaunchAxis::X,
                ..
            })
        ));

        let overflow = validate_static(
            exact_spec((1, 1, 1)),
            raw((u32::MAX, u32::MAX, 2), (1, 1, 1), 0),
        );
        assert!(matches!(
            overflow,
            Err(LaunchContractError::DimensionProductOverflow {
                dimension: LaunchDimension::Grid,
                ..
            })
        ));
    }

    #[test]
    fn exact_block_requires_the_whole_shape() {
        let result = validate_static(exact_spec((32, 4, 1)), raw((1, 1, 1), (64, 2, 1), 0));
        assert!(matches!(
            result,
            Err(LaunchContractError::BlockShapeMismatch {
                required: (32, 4, 1),
                actual: (64, 2, 1),
                ..
            })
        ));
    }

    #[test]
    fn max_threads_applies_to_the_block_product_and_checks_overflow() {
        let spec = LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::MaxThreads(256),
            DynamicSharedMemoryRequirement::Exact {
                bytes: 0,
                min_alignment: 1,
            },
        );
        assert!(validate_static(spec, raw((1, 1, 1), (128, 1, 1), 0)).is_ok());
        assert!(validate_static(spec, raw((1, 1, 1), (16, 16, 1), 0)).is_ok());
        assert!(matches!(
            validate_static(spec, raw((1, 1, 1), (17, 16, 1), 0)),
            Err(LaunchContractError::BlockThreadsExceedContract {
                actual: 272,
                max: 256,
                ..
            })
        ));

        let overflowing_spec = LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::MaxThreads(u32::MAX),
            DynamicSharedMemoryRequirement::Exact {
                bytes: 0,
                min_alignment: 1,
            },
        );
        assert!(matches!(
            validate_static(overflowing_spec, raw((1, 1, 1), (u32::MAX, u32::MAX, 2), 0),),
            Err(LaunchContractError::DimensionProductOverflow {
                dimension: LaunchDimension::Block,
                ..
            })
        ));
    }

    #[test]
    fn u32_coordinate_contract_accepts_exact_range_and_rejects_larger_axis() {
        let spec = exact_spec((2, 1, 1)).with_u32_coordinates();

        // 2^31 blocks * 2 threads gives exactly 2^32 zero-based coordinates,
        // whose largest value is u32::MAX.
        assert!(validate_static(spec, raw((1 << 31, 1, 1), (2, 1, 1), 0)).is_ok());

        assert!(matches!(
            validate_static(spec, raw(((1 << 31) + 1, 1, 1), (2, 1, 1), 0)),
            Err(LaunchContractError::CoordinateRangeExceedsU32 {
                axis: LaunchAxis::X,
                positions: 4_294_967_298,
                ..
            })
        ));
    }

    #[test]
    fn validates_dynamic_shared_memory_exact_range_and_alignment() {
        let exact = LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::Exact((32, 1, 1)),
            DynamicSharedMemoryRequirement::Exact {
                bytes: 1024,
                min_alignment: 16,
            },
        );
        assert!(validate_static(exact, raw((1, 1, 1), (32, 1, 1), 1024)).is_ok());
        assert!(matches!(
            validate_static(exact, raw((1, 1, 1), (32, 1, 1), 512)),
            Err(LaunchContractError::DynamicSharedMemoryExactMismatch { .. })
        ));

        let range = LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::Exact((32, 1, 1)),
            DynamicSharedMemoryRequirement::Range {
                min_bytes: 512,
                max_bytes: 2048,
                min_alignment: 32,
            },
        );
        assert!(validate_static(range, raw((1, 1, 1), (32, 1, 1), 512)).is_ok());
        assert!(validate_static(range, raw((1, 1, 1), (32, 1, 1), 2048)).is_ok());
        assert!(matches!(
            validate_static(range, raw((1, 1, 1), (32, 1, 1), 256)),
            Err(LaunchContractError::DynamicSharedMemoryOutsideRange { .. })
        ));

        let reversed = LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::Exact((32, 1, 1)),
            DynamicSharedMemoryRequirement::Range {
                min_bytes: 2,
                max_bytes: 1,
                min_alignment: 1,
            },
        );
        assert!(matches!(
            validate_static(reversed, raw((1, 1, 1), (32, 1, 1), 1)),
            Err(LaunchContractError::InvalidSharedMemoryRange { .. })
        ));

        let bad_alignment = LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::Exact((32, 1, 1)),
            DynamicSharedMemoryRequirement::Exact {
                bytes: 0,
                min_alignment: 3,
            },
        );
        assert!(matches!(
            validate_static(bad_alignment, raw((1, 1, 1), (32, 1, 1), 0)),
            Err(LaunchContractError::InvalidSharedMemoryAlignment { alignment: 3, .. })
        ));
    }

    #[test]
    fn cluster_dimensions_must_be_nonzero_and_divide_grid() {
        let spec = exact_spec((32, 1, 1)).with_cluster((2, 2, 1));
        assert!(validate_static(spec, raw((8, 4, 1), (32, 1, 1), 0)).is_ok());
        assert!(matches!(
            validate_static(spec, raw((7, 4, 1), (32, 1, 1), 0)),
            Err(LaunchContractError::ClusterDoesNotDivideGrid {
                axis: LaunchAxis::X,
                grid: 7,
                cluster: 2,
                ..
            })
        ));

        let zero = exact_spec((32, 1, 1)).with_cluster((2, 0, 1));
        assert!(matches!(
            validate_static(zero, raw((8, 4, 1), (32, 1, 1), 0)),
            Err(LaunchContractError::ZeroDimension {
                dimension: LaunchDimension::Cluster,
                axis: LaunchAxis::Y,
                ..
            })
        ));
    }

    #[test]
    fn validates_device_and_function_block_limits() {
        let spec = LaunchContractSpec::new(
            KERNEL,
            BlockRequirement::MaxThreads(2048),
            DynamicSharedMemoryRequirement::Exact {
                bytes: 0,
                min_alignment: 1,
            },
        );
        let limits = generous_limits();

        assert!(matches!(
            validate_live_shape(spec, raw((1, 1, 1), (1025, 1, 1), 0), limits, 1024),
            Err(LaunchContractError::DeviceDimensionExceeded {
                dimension: LaunchDimension::Block,
                axis: LaunchAxis::X,
                ..
            })
        ));
        assert!(matches!(
            validate_live_shape(spec, raw((1, 1, 1), (33, 33, 1), 0), limits, 2048),
            Err(LaunchContractError::DeviceThreadsPerBlockExceeded { actual: 1089, .. })
        ));
        assert!(matches!(
            validate_live_shape(spec, raw((1, 1, 1), (32, 16, 1), 0), limits, 256),
            Err(LaunchContractError::FunctionThreadsPerBlockExceeded {
                actual: 512,
                max: 256,
                ..
            })
        ));
        assert!(matches!(
            validate_live_shape(
                spec,
                raw((u32::MAX, 65_536, 1), (32, 1, 1), 0),
                limits,
                1024,
            ),
            Err(LaunchContractError::DeviceDimensionExceeded {
                dimension: LaunchDimension::Grid,
                axis: LaunchAxis::Y,
                ..
            })
        ));
    }

    #[test]
    fn validates_static_plus_dynamic_shared_resources() {
        let total = shared_memory_total(KERNEL, 16 * 1024, 40 * 1024).unwrap();
        assert!(matches!(
            validate_shared_memory_limit(KERNEL, total, 48 * 1024, false),
            Err(LaunchContractError::DeviceSharedMemoryExceeded {
                total: 57_344,
                max: 49_152,
                opt_in: false,
                ..
            })
        ));
        assert!(validate_shared_memory_limit(KERNEL, total, 96 * 1024, true).is_ok());
        assert!(matches!(
            validate_shared_memory_limit(KERNEL, 100 * 1024, 96 * 1024, true),
            Err(LaunchContractError::DeviceSharedMemoryExceeded { opt_in: true, .. })
        ));
        assert!(matches!(
            shared_memory_total(KERNEL, u32::MAX, 1),
            Err(LaunchContractError::SharedMemoryTotalOverflow { .. })
        ));
        assert_eq!(
            dynamic_shared_memory_max(DynamicSharedMemoryRequirement::Range {
                min_bytes: 1024,
                max_bytes: 8192,
                min_alignment: 16,
            }),
            8192
        );
    }

    #[test]
    fn shared_memory_range_requires_support_for_its_advertised_maximum() {
        let requirement = DynamicSharedMemoryRequirement::Range {
            min_bytes: 1024,
            max_bytes: 96 * 1024,
            min_alignment: 16,
        };
        let spec =
            LaunchContractSpec::new(KERNEL, BlockRequirement::Exact((32, 1, 1)), requirement);

        // The concrete 32 KiB choice belongs to the range.
        assert!(validate_static(spec, raw((1, 1, 1), (32, 1, 1), 32 * 1024)).is_ok());

        // Preparation nevertheless validates the immutable contract maximum,
        // so every value advertised by the range is safe on this device.
        let total = shared_memory_total(KERNEL, 0, dynamic_shared_memory_max(requirement)).unwrap();
        assert!(matches!(
            validate_shared_memory_limit(KERNEL, total, 64 * 1024, true),
            Err(LaunchContractError::DeviceSharedMemoryExceeded {
                total: 98_304,
                max: 65_536,
                opt_in: true,
                ..
            })
        ));
    }

    #[test]
    fn validates_architecture_and_optional_capabilities() {
        assert!(validate_compute_capability(KERNEL, (9, 0), (10, 0)).is_ok());
        assert!(matches!(
            validate_compute_capability(KERNEL, (9, 0), (8, 9)),
            Err(LaunchContractError::ComputeCapabilityTooLow { .. })
        ));
        assert!(matches!(
            validate_cluster_support(KERNEL, false),
            Err(LaunchContractError::ClusterLaunchUnsupported { .. })
        ));
        assert!(validate_cluster_size(KERNEL, 8, 8).is_ok());
        assert!(matches!(
            validate_cluster_size(KERNEL, 16, 8),
            Err(LaunchContractError::ClusterSizeExceeded {
                blocks: 16,
                max: 8,
                ..
            })
        ));
        assert!(validate_required_cluster(KERNEL, (2, 1, 1), Some((2, 1, 1))).is_ok());
        assert!(matches!(
            validate_required_cluster(KERNEL, (2, 1, 1), Some((4, 1, 1))),
            Err(LaunchContractError::FunctionClusterShapeMismatch { .. })
        ));
        assert!(matches!(
            validate_required_cluster(KERNEL, (2, 1, 1), None),
            Err(LaunchContractError::RequiredClusterDimensionsMissing { .. })
        ));
        assert!(validate_cluster_residency(KERNEL, (2, 1, 1), 1).is_ok());
        assert!(matches!(
            validate_cluster_residency(KERNEL, (2, 1, 1), 0),
            Err(LaunchContractError::ClusterHasNoResidency { .. })
        ));
        assert!(matches!(
            validate_cooperative_support(KERNEL, false),
            Err(LaunchContractError::CooperativeLaunchUnsupported { .. })
        ));
    }

    #[test]
    fn validates_cooperative_residency_capacity() {
        assert!(validate_cooperative_residency(KERNEL, 80, 80).is_ok());
        assert!(matches!(
            validate_cooperative_residency(KERNEL, 81, 80),
            Err(LaunchContractError::CooperativeGridTooLarge {
                blocks: 81,
                resident_capacity: 80,
                ..
            })
        ));
    }

    #[test]
    fn clustered_cooperative_residency_uses_active_cluster_capacity() {
        // A (2,1,1) cluster with 4 concurrent clusters can host 8 blocks.
        // A fully-resident cooperative grid of 8 blocks therefore passes, while
        // 10 blocks (5 clusters) must fail with CooperativeGridTooLarge.
        let cluster_blocks = 2u64;
        let active_clusters = 4u64;
        let resident_capacity = active_clusters * cluster_blocks;

        assert!(validate_cooperative_residency(KERNEL, 8, resident_capacity).is_ok());
        assert!(matches!(
            validate_cooperative_residency(KERNEL, 10, resident_capacity),
            Err(LaunchContractError::CooperativeGridTooLarge {
                blocks: 10,
                resident_capacity: 8,
                ..
            })
        ));
    }

    #[test]
    fn combined_cluster_cooperative_spec_is_expressible() {
        let spec = exact_spec((128, 1, 1))
            .with_cluster((2, 1, 1))
            .with_cooperative();
        assert_eq!(spec.cluster(), Some((2, 1, 1)));
        assert!(spec.cooperative());

        // Static geometry still requires the grid to be an integer number of
        // clusters even when cooperative residency is also required.
        assert!(validate_static(spec, raw((4, 1, 1), (128, 1, 1), 0)).is_ok());
        assert!(matches!(
            validate_static(spec, raw((3, 1, 1), (128, 1, 1), 0)),
            Err(LaunchContractError::ClusterDoesNotDivideGrid { .. })
        ));
    }

    #[test]
    fn spec_builders_preserve_diagnostic_metadata() {
        let spec = exact_spec((32, 1, 1))
            .with_cluster((2, 1, 1))
            .with_cooperative()
            .with_min_compute_capability(9, 0)
            .with_u32_coordinates();
        assert_eq!(spec.kernel_name(), KERNEL);
        assert_eq!(spec.block(), BlockRequirement::Exact((32, 1, 1)));
        assert_eq!(spec.cluster(), Some((2, 1, 1)));
        assert!(spec.cooperative());
        assert_eq!(spec.min_compute_capability(), Some((9, 0)));
        assert_eq!(spec.coordinates(), CoordinateRequirement::U32);
    }
}
