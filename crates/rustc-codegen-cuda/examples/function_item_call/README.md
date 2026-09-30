# function_item_call

Regression for passing a function item through a generic `FnOnce` helper.

Rust lowers `apply_once(plus_seven, x)` through a callable trait method such as
`<fn item as FnOnce>::call_once`. The device collector must enqueue the concrete
function-item body, and MIR import must emit a direct call name that matches the
collector/export naming policy. Otherwise the generated device IR references a
callee symbol that was never emitted.

The diverging-call regression also covers an ordinary direct `#[device]`
function returning `!`. Its body performs an observable store before looping
forever. A MIR `Call` with no normal successor must therefore keep the call and
end the caller block with `unreachable`; replacing the call itself with a trap
would erase the store.

`verify-code-shape.sh` pins that contract in the pre-optimization LLVM IR,
before the middle-end can inline the diverging helper. The divergent branches
are not launched; the host run exercises only the returning path.
