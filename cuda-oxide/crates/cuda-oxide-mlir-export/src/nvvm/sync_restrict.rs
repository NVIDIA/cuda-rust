// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native proxy fences retaining the catalog's restricted shared-memory ordering.

use crate::{
    MlirAttribute, MlirOperation, OperationInput, OperationTranslation, TranslationError,
    TranslationRegistry, TranslationSession,
};
use dialect_nvvm::ops::{
    FenceProxyAsyncGenericAcquireSharedClusterClusterOp,
    FenceProxyAsyncGenericReleaseSharedCtaClusterOp,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};

/// Register cluster-scoped acquire/release ordering between generic and async proxies.
pub fn register_nvvm_sync_restrict_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    /// Register a proxy-fence operation with its canonical identity and memory order.
    macro_rules! register {
        ($op:ty, $marker:literal, $order:literal) => {
            registry.register_operation::<$op>(ProxyFenceTranslation {
                marker: $marker,
                order: $order,
                verify: |ctx, source| {
                    <$op>::new(source)
                        .verify(ctx)
                        .map_err(|error| error.to_string())
                },
            })?;
        };
    }
    register!(
        FenceProxyAsyncGenericAcquireSharedClusterClusterOp,
        "v1:i0315",
        "acquire"
    );
    register!(
        FenceProxyAsyncGenericReleaseSharedCtaClusterOp,
        "v1:i0314",
        "release"
    );
    Ok(())
}

/// Keep proxy direction and restricted shared-memory ordering on a native fence.
struct ProxyFenceTranslation {
    /// Canonical catalog identity required before emitting this intrinsic.
    marker: &'static str,
    /// Acquire or release ordering spelling retained by the proxy fence.
    order: &'static str,
    /// Check the source operation's selectors and operand contract.
    verify: fn(&Context, Ptr<Operation>) -> Result<(), String>,
}

impl OperationTranslation for ProxyFenceTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        (self.verify)(ctx, source)?;
        if input.attributes.remove("cuda_oxide_intrinsic_marker")
            != Some(MlirAttribute::String(self.marker.into()))
        {
            return Err(format!(
                "sync-restrict proxy fence expected intrinsic marker {}",
                self.marker
            ));
        }
        if !input.operands.is_empty()
            || !input.results.is_empty()
            || !input.attributes.is_empty()
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err(
                "sync-restrict proxy fence requires no operands, results, residual attributes, or control flow"
                    .into(),
            );
        }
        let mut op = MlirOperation::new("nvvm.fence.proxy.sync_restrict")?;
        op.location = input.location;
        op.properties.insert(
            "order".into(),
            MlirAttribute::dialect(format!("#nvvm.mem_order<{}>", self.order))?,
        );
        // The native operation uses the same proxy properties for both orders;
        // acquire restricts shared::cluster and release restricts shared::cta.
        op.properties.insert(
            "fromProxy".into(),
            MlirAttribute::dialect("#nvvm.proxy_kind<generic>")?,
        );
        op.properties.insert(
            "toProxy".into(),
            MlirAttribute::dialect("#nvvm.proxy_kind<async>")?,
        );
        Ok(vec![op])
    }
}
