// A `#[cutile::foreign]` function declares a tile-library function; the
// library supplies the body, so a body is an error.
#[cutile::module]
mod decls {
    use cutile::core::*;

    #[cutile::foreign]
    pub unsafe fn fft<const N: i32>(#[inout] x: Tile<f32, { [N] }>) -> Tile<f32, { [N] }> {
        x
    }
}

fn main() {}
