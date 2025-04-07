#!/usr/bin/env python3


def fix_decorator_blank_lines():
    """Fix blank lines after function decorators in generators.py."""
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        content = f.read()

    # Replace problematic patterns: blank line between decorator and function definition
    content = content.replace(
        "@GenKernelFeatureMap.register", "\n@GenKernelFeatureMap.register"
    )
    content = content.replace(".register(", ".register(")
    content = content.replace(")\n\ndef", ")\ndef")

    # Write the corrected content back
    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.write(content)

    # Now fix the specific lines with the decorator blank line issue
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        lines = f.readlines()

    # Process the file line by line to find and fix decorator-function pairs
    i = 0
    while i < len(lines) - 1:
        if lines[i].strip().startswith("@") and lines[i + 1].strip() == "":
            # Remove the blank line after the decorator
            lines.pop(i + 1)
        else:
            i += 1

    # Write the fixed content back
    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.writelines(lines)

    print("Fixed blank line issues after decorators in generators.py")


if __name__ == "__main__":
    fix_decorator_blank_lines()
