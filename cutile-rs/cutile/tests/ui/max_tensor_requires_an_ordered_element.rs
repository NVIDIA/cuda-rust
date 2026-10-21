// `atomic::AtomicOrd` admits the element types Tile IR can compare
// atomically: `i32`, `u32`, `i64`, `u64`. There is no atomic floating-point
// maximum, so `MaxTensor<f32, _>` must fail at the type level rather than
// reach the JIT.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn running_max(out: &MaxTensor<f32, { [-1, -1] }>) {
        let ones: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: MaxPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_max([0i32, 0i32], ones);
    }
}

fn main() {}
