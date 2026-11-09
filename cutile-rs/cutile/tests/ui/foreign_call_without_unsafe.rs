// Calling a foreign function needs an `unsafe` block: the caller vouches
// that the declaration matches the library.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches the library's `add_one`.
    #[cutile::foreign]
    pub unsafe fn add_one(#[inout] x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;

    #[cutile::entry()]
    fn add_one_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(add_one(t));
    }
}

fn main() {}
