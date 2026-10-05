// Replacing a stored tensor's token with an unordered one lets a later safe
// Partition::load race the store. The raw view constructor must require unsafe.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn reset(out: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let x: Tile<f32, { [4] }> = input.load_like(out);
        out.store(x);
        let shape = out.shape();
        let view: Partition<f32, { [4] }> =
            make_partition_view(out, shape, padding::Zero, dim_map::Identity, new_token_unordered());
        let y = view.load([program_id(0)]);
        out.store(y + y);
    }
}

fn main() {}
