//! Delta-debugging tool: rewrites per-op debug-attr ids in a bytecode file.
//! Usage: patch_debug_ids <in.bc> <out.bc> <keep_n> <fill_id>
//! Keeps the first keep_n per-op ids, replaces the rest with fill_id.
fn read_varint(d: &[u8], p: &mut usize) -> u64 {
    let (mut r, mut s) = (0u64, 0);
    loop {
        let b = d[*p];
        *p += 1;
        r |= ((b & 0x7f) as u64) << s;
        if b & 0x80 == 0 {
            return r;
        }
        s += 7;
    }
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut d = std::fs::read(&a[1]).unwrap();
    let keep_n: usize = a[3].parse().unwrap();
    let fill: u64 = a[4].parse().unwrap();
    let mut p = 12; // header
    loop {
        let id = d[p] & 0x7f;
        let aligned = d[p] & 0x80 != 0;
        p += 1;
        if id == 0 {
            break;
        } // EndOfBytecode
        let len = read_varint(&d, &mut p) as usize;
        if aligned {
            let al = read_varint(&d, &mut p) as usize;
            let pad = (al - (p % al)) % al;
            p += pad;
        }
        if id == 3 {
            // Debug section: numFunctions varint, pad4, u32*, numIndices varint, pad8, u64*
            let start = p;
            let mut q = p;
            let nf = read_varint(&d, &mut q) as usize;
            q += (4 - ((q - start) % 4)) % 4;
            // Each function's flattened list starts with ITS OWN location
            // attr; the layout is [func0, ops0.., func1, ops1..], so the
            // offset table marks every list head to protect — not just the
            // first nf slots.
            let mut heads = Vec::with_capacity(nf);
            for _ in 0..nf {
                heads.push(u32::from_le_bytes(d[q..q + 4].try_into().unwrap()) as usize);
                q += 4;
            }
            let ni = read_varint(&d, &mut q) as usize;
            q += (8 - ((q - start) % 8)) % 8;
            let single = std::env::var("PATCH_SINGLE").is_ok();
            // Op counter that skips each function's head slot.
            let mut op_idx = 0usize;
            for i in 0..ni {
                if heads.contains(&i) {
                    continue; // a function's own attr, never patched
                }
                let hit = if single {
                    op_idx == keep_n
                } else {
                    op_idx >= keep_n
                };
                if hit {
                    let off = q + i * 8;
                    d[off..off + 8].copy_from_slice(&fill.to_le_bytes());
                }
                op_idx += 1;
            }
        }
        p += len;
    }
    std::fs::write(&a[2], &d).unwrap();
}
