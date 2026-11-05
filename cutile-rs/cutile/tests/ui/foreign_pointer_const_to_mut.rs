// A `*mut` pointer operand states that the library writes through it:
// rustc rejects the `*const` address of a shared tensor borrow.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches the library's `fill`, which writes 32 `f32` to `dst`.
    #[cutile::foreign]
    pub unsafe fn fill(dst: *mut f32);

    #[cutile::entry()]
    fn fill_kernel(x: &Tensor<f32, { [32] }>) {
        unsafe { fill(x.as_ptr()) };
    }
}

fn main() {}
