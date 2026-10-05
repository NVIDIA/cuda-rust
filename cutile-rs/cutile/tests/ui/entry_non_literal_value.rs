// Entry options are read as literals by the JIT at first launch, where a
// non-literal value used to panic ("print_ir is not a literal"). It is now a
// compile error pointing at the value.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    const VERBOSE: bool = true;

    #[cutile::entry(print_ir = VERBOSE)]
    fn copy(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [4] }> = load_tile_like(x, z);
        z.store(t);
    }
}

fn main() {}
