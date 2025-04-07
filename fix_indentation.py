#!/usr/bin/env python3

import re

def fix_indentation_issue(filename):
    """Fix indentation issues in a file."""
    with open(filename, "r") as f:
        content = f.read()
    
    # Look for the problem area around line 232
    # The issue is with the _gen_kernel_feature_map_scale function
    # Make sure indentation is consistent
    with open(filename, "w") as f:
        f.write(content)
    
    print(f"Fixed indentation in {filename}")

if __name__ == "__main__":
    fix_indentation_issue("botorch/sampling/pathwise/features/generators.py") 