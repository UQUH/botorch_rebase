#!/usr/bin/env python3

def fix_whitespace(filename):
    """Fix whitespace issues in a file."""
    with open(filename, "r") as f:
        lines = f.readlines()

    # Strip trailing whitespace from all lines
    lines = [line.rstrip() + "\n" for line in lines]

    with open(filename, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    fix_whitespace("botorch/sampling/pathwise/utils/transforms.py") 