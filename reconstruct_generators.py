#!/usr/bin/env python3


def fix_generators_file():
    # Create the corrected content for the ScaleKernel function
    # with proper indentation

    # Replacement with consistent indentation
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

    # Write the file with the corrected content
    with open("fix_scale_kernel.txt", "w") as f:
        f.write(replacement)

    print("Created fix_scale_kernel.txt with corrected content.")
    print("You'll need to manually insert this into generators.py.")


if __name__ == "__main__":
    fix_generators_file()
