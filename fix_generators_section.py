#!/usr/bin/env python3

def fix_generators_section():
    """Replace the matern kernel function section to fix indentation issues."""
    with open("botorch/sampling/pathwise/features/generators.py", "r") as f:
        content = f.read()
    
    # Find the start of the section (function definition)
    start_marker = "def _gen_kernel_feature_map_matern("
    end_marker = "@GenKernelFeatureMap.register(kernels.ScaleKernel)"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find the section to replace.")
        return
    
    # Define the replacement code with proper indentation
    replacement = """def _gen_kernel_feature_map_matern(
    kernel: kernels.MaternKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    # smoothness parameter nu. The spectral density guides weight sampling.
    # For Matern kernels, we use a different weight generator that incorporates the
    # smoothness parameter nu. Weights follow a distribution based on nu.
    # This follows the Matern kernel's spectral density.
    def _weight_generator(shape: Size) -> Tensor:
        try:
            n, d = shape
        except ValueError:
            raise UnsupportedError(
                f"Expected `shape` to be 2-dimensional, but {len(shape)=}."
            )

        dtype = kernel.dtype
        device = kernel.device
        nu = torch.tensor(kernel.nu, device=device, dtype=dtype)
        normals = draw_sobol_normal_samples(n=n, d=d, device=device, dtype=dtype)
        # For Matern kernels, we sample from a Gamma distribution based on nu
        return Gamma(nu, nu).rsample((n, 1)).rsqrt() * normals

    return _gen_fourier_features(
        kernel=kernel,
        weight_generator=_weight_generator,
        **kwargs,
    )

"""
    
    # Replace the section
    new_content = content[:start_idx] + replacement + content[end_idx:]
    
    with open("botorch/sampling/pathwise/features/generators.py", "w") as f:
        f.write(new_content)
    
    print("Successfully replaced the problematic section.")

if __name__ == "__main__":
    fix_generators_section() 