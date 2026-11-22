// The compiler cannot check a tile library's implementation, so a foreign
// function is declared `unsafe fn` and callers uphold its contract.
#[cutile::module]
mod decls {
    use cutile::core::*;

    #[cutile::foreign]
    pub fn fft<const N: i32>(#[inout] x: Tile<f32, { [N] }>) -> Tile<f32, { [N] }>;
}

fn main() {}
