#!/usr/bin/env python3


def fix_generators_file():
    """Fix the ScaleKernel section in generators.py."""

    # Replacement with correct indentation
    replacement = """@GenKernelFeatureMap.register(kernels.ScaleKernel)
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
    return feature_map"""

    # Read the current file
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        content = f.read()

    # Find the ScaleKernel section
    start_marker = "@GenKernelFeatureMap.register(kernels.ScaleKernel)"
    end_marker = "@GenKernelFeatureMap.register(kernels.ProductKernel)"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)

    if start_idx == -1 or end_idx == -1:
        print("Could not find the section to replace.")
        return False

    # Build the new content with the replacement
    new_content = content[:start_idx] + replacement + "\n\n\n" + content[end_idx:]

    # Write the corrected content directly back to the file
    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.write(new_content)

    print("Successfully fixed the ScaleKernel section in generators.py")
    return True


if __name__ == "__main__":
    fix_generators_file()
