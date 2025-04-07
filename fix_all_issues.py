#!/usr/bin/env python3


def fix_generators_indentation():
    """Fix indentation issues in generators.py by completely rewriting
    the problematic section."""
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        lines = f.readlines()

    # The problematic section is around line 232
    # Let's rewrite the entire function to ensure consistency
    start_line = 0
    end_line = len(lines)

    for i, line in enumerate(lines):
        if "@GenKernelFeatureMap.register(kernels.ScaleKernel)" in line:
            start_line = i
            break

    for i in range(start_line, len(lines)):
        if (
            "@GenKernelFeatureMap.register" in lines[i]
            and "ScaleKernel" not in lines[i]
        ):
            end_line = i
            break

    # Rewrite the function with correct indentation
    new_section = """@GenKernelFeatureMap.register(kernels.ScaleKernel)
def _gen_kernel_feature_map_scale(
    kernel: kernels.ScaleKernel,
    *,
    num_ambient_inputs: Optional[int] = None,
    **kwargs: Any,
) -> KernelFeatureMap:
    active_dims = kernel.active_dims
    num_scale_kernel_inputs = get_kernel_num_inputs(
        kernel=kernel,
        num_ambient_inputs=num_ambient_inputs,
        default=None,
    )
    kwargs_copy = kwargs.copy()
    kwargs_copy["num_ambient_inputs"] = num_scale_kernel_inputs
    feature_map = gen_kernel_feature_map(
        kernel.base_kernel,
        **kwargs_copy,
    )

    if active_dims is not None and active_dims is not kernel.base_kernel.active_dims:
        feature_map.input_transform = transforms.ChainedTransform(
            feature_map.input_transform, transforms.FeatureSelector(indices=active_dims)
        )

    feature_map.output_transform = transforms.ChainedTransform(
        transforms.OutputscaleTransform(kernel), feature_map.output_transform
    )
    return feature_map

"""

    # Replace the problematic section
    new_lines = lines[:start_line] + [new_section] + lines[end_line:]

    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        for line in new_lines:
            if isinstance(line, list):
                f.writelines(line)
            else:
                f.write(line)


def fix_mixins_long_line():
    """Fix the long line in mixins.py."""
    with open("botorch/sampling/pathwise/utils/mixins.py", "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Call forward instead of super()" in line:
            lines[i] = (
                "        # Call forward() - bypassing super().__call__ to implement "
                "interface\n"
            )

    with open("botorch/sampling/pathwise/utils/mixins.py", "w") as f:
        f.writelines(lines)


def fix_transforms_whitespace():
    """Fix whitespace issues in transforms.py."""
    with open("botorch/sampling/pathwise/utils/transforms.py", "r") as f:
        lines = f.readlines()

    # Remove trailing whitespace from all lines
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip() + "\n"

    with open("botorch/sampling/pathwise/utils/transforms.py", "w") as f:
        f.writelines(lines)


def fix_generator_comment():
    """Fix the comment line in generators.py."""
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        lines = f.readlines()

    lines[216] = (
        "    # smoothness parameter nu. The spectral density guides weight sampling.\n"
    )

    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    print("Fixing generator comment...")
    fix_generator_comment()

    print("Fixing mixins long line...")
    fix_mixins_long_line()

    print("Fixing transforms whitespace...")
    fix_transforms_whitespace()

    print("Fixing generators indentation...")
    fix_generators_indentation()

    print("All fixes applied successfully.")
