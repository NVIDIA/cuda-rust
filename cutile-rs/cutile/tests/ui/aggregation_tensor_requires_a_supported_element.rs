// `atomic::AtomicAdd` admits the element types Tile IR can add atomically:
// `f16`, `bf16`, `f32`, `f64` through the `AddF` mode, and `i32`, `u32`,
// `i64`, `u64` through `Add`. `i8` is an `ElementType` the IR cannot reduce
// atomically at all, so the pairing must fail at the type level rather than
// reach the JIT.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn accumulate(out: &AddTensor<i8, { [-1, -1] }>) {
        let ones: Tile<i8, { [16, 16] }> = constant(1i8, shape![16, 16]);
        let acc: AddPartition<i8, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([0i32, 0i32], ones);
    }
}

fn main() {}
