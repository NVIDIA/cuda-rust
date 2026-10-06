## Summary

<!-- One paragraph: what does this change and why? -->

## Changes

<!-- Bullet list of the main code changes. -->

## Testing

<!-- How did you verify this? Paste relevant `cargo oxide run` /
`just -f cuda-oxide/Justfile check` output, or smoketest results. -->
- [ ] `just -f cuda-oxide/Justfile check` passes (the local mirror of CI: fmt, clippy, tests, guards, docs)
- [ ] `cargo oxide run <example>` passes, or `cuda-oxide/scripts/smoketest.sh -o '^<example>$'`
- [ ] New example added (if applicable)

## Checklist

- [ ] All commits signed off (`git commit -s`)
- [ ] SPDX headers on new source files

---
> Questions about the review? Ping us in [#contributors on Discord](https://discord.gg/ZUEr4AhH5C).
