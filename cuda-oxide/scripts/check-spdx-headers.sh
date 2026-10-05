#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Require the standard SPDX header on every NVIDIA-authored source file.
#
# The header policy is two lines, in a comment near the top of the file:
#
#   SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
#
# Nothing enforced it, so it drifted: the OSRB review fixed 32 files in #812,
# and the very next scripts to land (#819, #835) arrived without headers and
# had to be fixed again. New files fail here now instead of in the next
# license audit.
#
# Scope: tracked *.rs *.sh *.py *.cu *.c *.h *.ll *.js *.css *.html *.nix
# files -- 1803 of them, and the header always sits within the first four
# lines; the check reads the first 15 so a longer shebang/attribute preamble
# cannot push a real header out of view, while a stray SPDX string in the body
# of a generator or test still cannot satisfy it.
#
# The last four extensions were outside the glob until this guard was extended,
# and the eight files they cover -- the book's `_static` CSS and JS, its two
# `_templates` HTML fragments, and `flake.nix` -- all carry the standard header
# already. That is the point: someone wrote them correctly, and nothing was
# checking, so the next one could arrive without a header exactly as #819 and
# #835 did for shell scripts. Every one of the eight passes unchanged, so this
# only closes the hole.
#
# Three kinds of files are deliberately not held to the standard header:
#
#   * cuda-oxide/crates/fuzzer/rustlantis/** is embedded third-party code (attributed in
#     THIRD_PARTY_NOTICES). Its files keep their upstream form; adding an
#     NVIDIA copyright line to code we do not own would be wrong, so the whole
#     subtree is out of scope.
#   * cutile-rs/** is a nested import of a separate product. Its sources keep
#     the headers that tree already uses; do not rewrite them to the Oxide
#     Apache-2.0 pair. Drop the `cutile-rs/` grep below when Oxide should
#     require that header on this path again.
#   * The two CUTLASS-derived benchmark helpers below carry a plain copyright
#     line plus `SPDX-License-Identifier: BSD-3-Clause`, reviewed and accepted
#     as-is by the OSRB. They are exempt from the Apache-2.0 header but must
#     still carry their BSD-3-Clause identifier, and each entry is checked
#     against the tree so a rename fails the run instead of quietly exempting
#     nothing.
#
# A new BSD-derived (or otherwise non-Apache) file is supposed to fail here:
# adding it to the exemption list is the explicit, reviewable act that
# replaces silence.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "${GIT_ROOT}"

copyright_re='SPDX-FileCopyrightText: Copyright \(c\) [0-9]{4}( ?- ?[0-9]{4})? NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.'
license_re='SPDX-License-Identifier: Apache-2\.0'
HEADER_WINDOW=15

# CUTLASS-derived files accepted by the OSRB with their upstream BSD-3-Clause
# terms. Path plus the identifier each must still carry.
BSD_EXEMPT_FILES=(
    cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_sol/bench/tutorial_gemm_utils.py
    cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_sol_final/bench/tutorial_gemm_utils.py
)

# `head | grep -q` under pipefail can report a match as failure when grep
# exits before head does (head then dies of SIGPIPE), so every check below
# reads the window into a variable first and greps that.
window() {
    head -n "${HEADER_WINDOW}" "$1"
}

# Self-test. The failure mode this guard has to survive is "silently stops
# matching anything", so prove both regexes still accept a known-good header
# and reject a missing one before a clean result is believed.
canary="$(mktemp -d)"
trap 'rm -rf "${canary}"' EXIT
printf '// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n// SPDX-License-Identifier: Apache-2.0\n' \
    >"${canary}/good.rs"
printf '// no header here\n' >"${canary}/bad.rs"
good="$(window "${canary}/good.rs")"
bad="$(window "${canary}/bad.rs")"
if ! grep -Eq "${copyright_re}" <<<"${good}" ||
    ! grep -Eq "${license_re}" <<<"${good}" ||
    grep -Eq "${copyright_re}" <<<"${bad}"; then
    echo "error: SPDX header guard self-test failed: the patterns no longer" >&2
    echo "       separate a compliant header from a missing one, so a clean" >&2
    echo "       result means nothing" >&2
    exit 1
fi

# Exemptions are verified, not trusted: each exempt file must exist and must
# still carry the BSD-3-Clause identifier it was exempted for.
for f in "${BSD_EXEMPT_FILES[@]}"; do
    if [ ! -f "${f}" ]; then
        echo "error: BSD exemption lists '${f}' which does not exist;" >&2
        echo "       update BSD_EXEMPT_FILES in $0" >&2
        exit 1
    fi
    if ! grep -q 'SPDX-License-Identifier: BSD-3-Clause' <<<"$(window "${f}")"; then
        echo "error: '${f}' is exempted as BSD-3-Clause but no longer carries" >&2
        echo "       that identifier in its first ${HEADER_WINDOW} lines" >&2
        exit 1
    fi
done

violations=''
while IFS= read -r f; do
    w="$(window "${f}")"
    if ! grep -Eq "${copyright_re}" <<<"${w}"; then
        violations="${violations}  ${f}: missing SPDX-FileCopyrightText line"$'\n'
    elif ! grep -Eq "${license_re}" <<<"${w}"; then
        violations="${violations}  ${f}: missing 'SPDX-License-Identifier: Apache-2.0' line"$'\n'
    fi
done < <(git ls-files -- '*.rs' '*.sh' '*.py' '*.cu' '*.c' '*.h' '*.ll' \
    '*.js' '*.css' '*.html' '*.nix' |
    grep -v '^cutile-rs/' |
    grep -v '^cuda-oxide/crates/fuzzer/rustlantis/' |
    grep -vxF -f <(printf '%s\n' "${BSD_EXEMPT_FILES[@]}"))

if [ -n "${violations}" ]; then
    echo "error: source files missing the standard SPDX header:" >&2
    printf '%s' "${violations}" >&2
    echo >&2
    echo "Add these two lines in a comment at the top of each file" >&2
    echo "(after the shebang, if any):" >&2
    echo >&2
    echo "  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved." >&2
    echo "  SPDX-License-Identifier: Apache-2.0" >&2
    exit 1
fi

echo "OK: every in-scope source file carries the standard SPDX header."
