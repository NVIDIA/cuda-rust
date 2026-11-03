// A misspelled `#[cutile::entry(..)]` key used to be silently ignored — the
// JIT only looks keys up by name — so `prnt_ir` never printed anything. It is
// now a compile error pointing at the key.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(prnt_ir = true)]
    fn copy(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [4] }> = load_tile_like(x, z);
        z.store(t);
    }
}

fn main() {}
