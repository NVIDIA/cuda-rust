// The capability is shared by construction: every program in the launch holds
// it and the atomics resolve the collisions. `&mut` would promise the
// disjointness the operation deliberately does not have.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn accumulate(out: &mut AddTensor<f32, { [-1, -1] }>) {
        let ones: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([0i32, 0i32], ones);
    }
}

fn main() {}
