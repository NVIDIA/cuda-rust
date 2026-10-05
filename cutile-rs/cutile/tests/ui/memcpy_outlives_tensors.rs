// `Memcpy` holds the device pointers of `dst` and `src`. Executing it after
// both tensors were dropped would copy between freed allocations, so the op
// must borrow them: this program must fail to compile.
use cutile::api;
use cutile::tile_kernel::DeviceOp;

fn main() {
    let op = {
        let mut dst = api::zeros::<f32>(&[4]).sync().unwrap();
        let src = api::ones::<f32>(&[4]).sync().unwrap();
        api::memcpy(&mut dst, &src)
    };
    op.sync().unwrap();
}
