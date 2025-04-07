#!/usr/bin/env python3

def fix_too_many_blanks():
    """Fix too many blank lines in generators.py."""
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        lines = f.readlines()
    
    # Process the file line by line to remove too many consecutive blank lines
    i = 0
    while i < len(lines) - 3:
        if lines[i].strip() == "" and lines[i+1].strip() == "" and lines[i+2].strip() == "":
            # Three or more consecutive blank lines, remove one
            lines.pop(i)
        else:
            i += 1
    
    # Write the fixed content back
    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.writelines(lines)
    
    print("Fixed too many blank lines in generators.py")

if __name__ == "__main__":
    fix_too_many_blanks() 