#!/usr/bin/env python3


def fix_file(filename, line_num, new_content):
    """Fix a specific line in a file."""
    with open(filename, "r") as f:
        lines = f.readlines()

    if line_num < len(lines):
        lines[line_num] = new_content + "\n"

    # Also fix indentation issues by ensuring consistent whitespace
    for i in range(len(lines)):
        # Ensure that if a line has a tab character, it's converted to spaces
        if "\t" in lines[i]:
            # Count leading tabs
            leading_tabs = len(lines[i]) - len(lines[i].lstrip("\t"))
            # Replace each tab with 4 spaces (common convention)
            lines[i] = " " * (4 * leading_tabs) + lines[i].lstrip("\t")
        
        # Remove trailing whitespace
        lines[i] = lines[i].rstrip() + "\n"

    with open(filename, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    # Fix the specific line causing issues
    fix_file(
        "botorch/sampling/pathwise/features/generators.py",
        216,  # Line 217 (0-indexed)
        "    # smoothness parameter nu. The spectral density guides weight sampling.",
    ) 