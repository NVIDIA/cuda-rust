# function_item_call

Regression for passing a function item through a generic `FnOnce` helper.

Rust lowers `apply_once(plus_seven, x)` through a callable trait method such as
`<fn item as FnOnce>::call_once`. The device collector must enqueue the concrete
function-item body, and MIR import must emit a direct call name that matches the
collector/export naming policy. Otherwise the generated device IR references a
callee symbol that was never emitted.
