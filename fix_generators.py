#!/usr/bin/env python3

import re

def fix_generators_file():
    """Fix indentation issues specifically in generators.py."""
    # Read the file content
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        lines = f.readlines()
    
    # Write the file with consistent indentation
    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        for i, line in enumerate(lines):
            # Remove any trailing whitespace
            line = line.rstrip()
            
            # Ensure proper indentation (4 spaces per level)
            # Look for inconsistent indentation
            if i == 231:  # Line 232 (0-indexed) is the problem area according to flake8
                # Ensure this line starts with consistent indentation
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces % 4 != 0:
                    # Fix to a multiple of 4 spaces
                    corrected_spaces = (leading_spaces // 4) * 4
                    line = " " * corrected_spaces + line.lstrip()
            
            f.write(line + "\n")

if __name__ == "__main__":
    fix_generators_file() 