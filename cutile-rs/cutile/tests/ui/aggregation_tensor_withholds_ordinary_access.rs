// The aggregation capability exposes its reductions and nothing else. An
// ordinary load or store through the same storage would race the other
// programs' reductions, so neither the tensor nor the view it hands out
// offers one — and `partition` yields an `AddPartition`, never an
// ordinary `Partition` or `PartitionMut`.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn accumulate(out: &AddTensor<f32, { [-1, -1] }>) {
        let zero: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);

        // Nothing ordinary on the capability itself. `shape()` is internal
        // too: it exists only because the view constructor's
        // `output_type_meta` evaluates it, and tile counts come from
        // `num_tiles(&partition, axis)` instead.
        let _shape: Shape<{ [-1, -1] }> = out.shape();
        let _loaded: Tile<f32, { [16, 16] }> = out.load([0i32, 0i32]);
        out.store(zero);
        let _mut_view: PartitionMut<f32, { [16, 16] }> = out.partition_mut(shape![16, 16]);

        // `partition` exists, but it cannot produce an ordinary view.
        let _ordinary: Partition<f32, { [16, 16] }> = out.partition(shape![16, 16]);

        // Nor does the view it does produce permit an ordinary access.
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _tile: Tile<f32, { [16, 16] }> = acc.load([0i32, 0i32]);
        let _tok: Token = acc.store(zero, [0i32, 0i32]);
    }
}

fn main() {}
