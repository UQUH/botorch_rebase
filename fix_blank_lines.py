#!/usr/bin/env python3


def fix_blank_lines():
    """Fix blank line issues in generators.py."""
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        lines = f.readlines()

    # Fix the missing blank lines before functions
    for i in range(len(lines) - 1, 0, -1):
        # Check for function definitions
        if lines[i].startswith("def ") or lines[i].startswith("@"):
            # Ensure there are exactly two blank lines before a function definition
            # except at the start of the file
            if i > 1 and not lines[i - 1].strip() == "":
                # Insert two blank lines
                lines.insert(i, "\n")
                lines.insert(i, "\n")
            elif (
                i > 2 and lines[i - 1].strip() == "" and not lines[i - 2].strip() == ""
            ):
                # We have one blank line, need one more
                lines.insert(i, "\n")

    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.writelines(lines)

    print("Fixed blank line issues in generators.py")


if __name__ == "__main__":
    fix_blank_lines()
