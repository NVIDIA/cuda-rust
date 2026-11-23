// A foreign function cannot return a pointer.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches the library's `scratch`.
    #[cutile::foreign]
    pub unsafe fn scratch(x: Tile<f32, { [32] }>) -> *mut u8;
}

fn main() {}
