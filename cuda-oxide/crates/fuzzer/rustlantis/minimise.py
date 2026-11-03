#!/usr/bin/env python3

import os
import subprocess
import string

end_bb = "Return()"

def check(file: os.PathLike) -> bool:
    out = subprocess.run(["timeout", "5", "target/release/difftest", str(file)], capture_output=True)
    if out.returncode == 124:
        return False
    err = out.stderr.decode(encoding = 'utf-8')
    return "didn't pass" in err and "stderr" not in err

def mutate(orig: str) -> str:
    if len(orig) == 0:
        return orig

    if orig[0] in string.ascii_uppercase and orig.endswith(")") and orig != end_bb:
        return end_bb
    else:
        return f"//{orig}"


if __name__ == "__main__":
    with open("minimised.rs", "w", encoding='utf-8') as working:
        with open("repro.rs", "r", encoding='utf-8') as orig:
            source = orig.readlines()

        progress = True
        while progress:
            progress = False
            limit = len(source)-1
            for line in reversed(range(limit)):
                saved = source[line].strip()
                if saved.startswith("//"):
                    continue
                elif len(saved) == 0:
                    continue
                elif saved.endswith("{"):
                    continue
                elif saved == "}":
                    continue
                elif saved.startswith("#"):
                    continue

                print(line, end='\r')

                source[line] = mutate(saved) + "\n"

                working.seek(0)
                working.writelines(source)
                working.truncate()
                working.flush()

                if check(working.name):
                    progress = True
                else:
                    source[line] = saved + "\n"

            working.seek(0)
            working.writelines(source)
            working.truncate()
            working.flush()
            print(f"done pass")