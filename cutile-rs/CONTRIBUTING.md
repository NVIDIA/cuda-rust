# Contributing to cuTile Rust

Thank you for your interest in contributing to cuTile Rust! Based on the type of contribution, it will fall into two categories:

1. You want to report a bug, feature request, or documentation issue:
   - File an [issue](https://github.com/NVlabs/cutile-rs/issues/new/choose) describing what you encountered or what you want to see changed.
   - For bug reports, please include the following information:
     - Your OS (e.g., Ubuntu 22.04)
     - Your GPU and GPU architecture (e.g., NVIDIA A100, sm_80)
     - The error you're seeing (full error message / stack trace)
     - A minimal example to reproduce the error (if possible)
2. You want to implement a feature, improvement, or bug fix:
   - Please ensure that your commits are signed [following GitHub’s instruction](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification).
     - Before marking your PR as ready for review, please make sure all tests pass and all examples are running via `bash scripts/run_all.sh`.

# Branch and PR Naming

Use a prefix that describes the type of change.

| Type | Use for |
|---|---|
| `feat` | New features |
| `fix` | Bug fixes |
| `doc` | Documentation changes |
| `refactor` | Code refactoring (no behavior change) |
| `perf` | Performance improvements (no functional change) |
| `test` | Test additions or changes |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks |

**Branches** use `/` as a separator: `feat/warp-interop`, `fix/kernel-launch-sync`, `doc/contributing-guide`

**PR titles** use `:` as a separator, lowercase: `feat: add warp interop support`, `fix: resolve kernel launch sync race`, `doc: update contributing guide`

# Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

# Developer Certificate of Origin (DCO)
```
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    
Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
  
  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
  
  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
  
  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
