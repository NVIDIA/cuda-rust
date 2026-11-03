// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! libclang 22 returns a forward declaration where older versions returned the
//! definition. Bindgen must follow it to the definition before emitting fields.

#[test]
fn typedef_chain_preserves_forward_declared_struct_fields() {
    let bindings = bindgen::builder()
        .header_contents(
            "forward_declared_struct.h",
            r#"
                typedef struct Packet_st Packet_v1;
                typedef Packet_v1 Packet;
                struct Packet_st {
                    unsigned long long address;
                    unsigned long long count;
                };
            "#,
        )
        .generate()
        .expect("generate bindings for the forward-declared struct");
    let file = syn::parse_file(&bindings.to_string()).expect("parse generated bindings");
    let packet = file
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Struct(item) if item.ident == "Packet_st" => Some(item),
            _ => None,
        })
        .expect("emit the complete struct definition");
    let fields: Vec<_> = packet
        .fields
        .iter()
        .map(|field| field.ident.as_ref().expect("named field").to_string())
        .collect();
    assert_eq!(fields, ["address", "count"]);
}
