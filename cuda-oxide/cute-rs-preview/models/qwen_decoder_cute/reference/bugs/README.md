# Bugs

One note per resolved bug: what went wrong, the picture that shows it, how it was fixed, what the fix costs.

| Date | Bug | Note |
| --- | --- | --- |
| 2026-10-17 | Vivian repeated under greedy decoding: tensor-core dot products rounded toward zero, the reference's cuBLAS FMA chains do not | [decode-projection-rounding.md](decode-projection-rounding.md) |
