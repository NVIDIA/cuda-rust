/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let dump = cutile_ir::bytecode::decoder::decode_bytecode_file(&path).unwrap();
    println!("{dump}");
}
