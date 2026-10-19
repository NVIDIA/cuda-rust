#!/bin/bash
cargo build --release

export RUST_LOG=info

mkdir -p out
for seed in {0..10}; do
    target/release/generate $seed > out/$seed.rs
    target/release/difftest out/$seed.rs
done