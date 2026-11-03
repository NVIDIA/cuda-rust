#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use core::intrinsics::mir::*;

static SHARED: i32 = 21;
static mut MUTABLE: i32 = 5;

#[inline(never)]
fn read_shared(value: &'static i32) -> i32 {
    *value
}

#[inline(never)]
fn update_mutable(value: &'static mut i32) -> i32 {
    *value += 1;
    *value
}

#[custom_mir(dialect = "runtime", phase = "initial")]
fn shared_reference_flow() -> i32 {
    mir! {
        type RET = i32;

        let _1: &'static i32;
        let _2: &'static i32;
        let _3: &'static i32;

        {
            _1 = Static(SHARED);
            _2 = _1;
            _3 = Move(_2);

            Call(
                RET = read_shared(Move(_3)),
                ReturnTo(bb1),
                UnwindUnreachable()
            )
        }

        bb1 = {
            Return()
        }
    }
}

#[custom_mir(dialect = "runtime", phase = "initial")]
fn mutable_reference_flow() -> i32 {
    mir! {
        type RET = i32;

        let _1: *mut i32;
        let _2: &'static mut i32;
        let _3: &'static mut i32;

        {
            _1 = StaticMut(MUTABLE);
            _2 = &mut (*_1);
            _3 = Move(_2);

            Call(
                RET = update_mutable(Move(_3)),
                ReturnTo(bb1),
                UnwindUnreachable()
            )
        }

        bb1 = {
            Return()
        }
    }
}

fn main() {
    assert_eq!(shared_reference_flow(), 21);
    assert_eq!(mutable_reference_flow(), 6);
}
