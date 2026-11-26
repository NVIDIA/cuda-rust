// A pointer operand is never written back: the callee writes through it,
// so `#[inout]` is rejected.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches the library's `fill`.
    #[cutile::foreign]
    pub unsafe fn fill(#[inout] dst: *mut f32) -> *mut f32;
}

fn main() {}
