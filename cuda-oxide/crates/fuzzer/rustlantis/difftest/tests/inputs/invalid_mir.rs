#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn simple(x: i32) -> i32 {
    mir!(
        let temp1: i32;
        let temp2: f32;

        {
            temp1 = x;
            Goto(exit)
        }

        exit = {
            temp2 = Move(temp1);
            RET = temp2;
            Return()
        }
    )
}

pub fn main() {
    println!("{}", simple(5));
}