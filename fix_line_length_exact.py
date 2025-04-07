#!/usr/bin/env python3

def fix_file(filename, line_num, new_content):
    """Fix a specific line in a file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if line_num < len(lines):
        lines[line_num] = new_content + '\n'
    
    with open(filename, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    # Fix the specific line causing issues
    fix_file(
        "botorch/sampling/pathwise/features/generators.py", 
        216,  # Line 217 (0-indexed)
        "    # smoothness parameter nu. The spectral density guides weight sampling."
    ) 