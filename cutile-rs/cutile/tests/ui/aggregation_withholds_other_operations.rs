// One capability is one operation. An `AddPartition` offers `atomic_add` and
// nothing else; combining by maximum into the same storage takes a
// `MaxTensor` binding, and the launcher rejects binding both to one
// allocation. The wrong-operation method must not exist.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn accumulate(out: &AddTensor<i32, { [-1, -1] }>) {
        let ones: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: AddPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_max([0i32, 0i32], ones);
    }
}

fn main() {}
