// Binding an atomic destination requires exclusive mutable authority over the
// backing tensor for the launch. A shared `Arc<Tensor<_>>` is not that: another
// holder could read the storage while the atomics are in flight.
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn accumulate(out: &AddTensor<f32, { [-1, -1] }>, src: &Tensor<f32, { [-1, -1] }>) {
        let part: Partition<f32, { [16, 16] }> = src.partition(shape![16, 16]);
        let bid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, { [16, 16] }> = part.load([bid.0, bid.1]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([bid.0, bid.1], tile);
    }
}

fn shared(out: Arc<Tensor<f32>>, src: Arc<Tensor<f32>>) {
    let _ = kernels::accumulate(out, src);
}

fn main() {}
