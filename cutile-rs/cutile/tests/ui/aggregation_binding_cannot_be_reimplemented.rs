// `KernelAtomic` is sealed. A downstream wrapper that delegated to
// `Arc<Tensor<T>>`'s launch plumbing would retain the storage as a *read*,
// bypassing the exclusive host authority an aggregation binding promises, so
// no type outside the crate may implement the trait.
use cutile::prelude::*;
use std::sync::Arc;

struct SharedWrapper(Arc<Tensor<f32>>);

impl cutile::tensor::KernelAtomic<f32> for SharedWrapper {}

fn main() {}
