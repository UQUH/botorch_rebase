#!/usr/bin/env python3


def fix_line_232():
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        lines = f.readlines()

    # Replace line 232 (0-indexed 231) and surrounding lines
    lines[230] = "    return _gen_fourier_features(\n"
    lines[231] = "        kernel=kernel,\n"
    lines[232] = "        weight_generator=_weight_generator,\n"
    lines[233] = "        **kwargs,\n"
    lines[234] = "    )\n"

    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.writelines(lines)

    print("Fixed indentation in line 232 and surrounding lines")


if __name__ == "__main__":
    fix_line_232()
