// A kernel launcher is a `GraphNode` only when its argument op is: inputs
// recorded into a CUDA graph must be pre-allocated, because an allocation
// node recorded during capture returns a different address on replay while
// the kernel node keeps the capture-time pointer. `api::zeros(..)` allocates,
// so recording a launcher fed by it must not compile.
use cutile::cuda_async::cuda_graph::CudaGraph;
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn copy(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [4] }> = load_tile_like(x, z);
        z.store(t);
    }
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let x = api::ones::<f32>(&[4]).sync_on(&stream)?;
    let _graph = CudaGraph::scope(&stream, |s| {
        s.record(kernels::copy(api::zeros::<f32>(&[4]).partition([4]), &x))?;
        Ok(())
    })?;
    Ok(())
}
