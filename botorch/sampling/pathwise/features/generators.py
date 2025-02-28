#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
.. [rahimi2007random]
    A. Rahimi and B. Recht. Random features for large-scale kernel machines.
    Advances in Neural Information Processing Systems 20 (2007).

.. [sutherland2015error]
    D. J. Sutherland and J. Schneider. On the error of random Fourier features.
    arXiv preprint arXiv:1506.02785 (2015).
"""

from __future__ import annotations

from collections.abc import Callable

from math import pi
from typing import Any, Callable, Iterable, Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features import maps
from botorch.sampling.pathwise.utils import (
    append_transform,
    get_kernel_num_inputs,
    is_finite_dimensional,
    prepend_transform,
    transforms,
)
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.utils.types import DEFAULT
from gpytorch import kernels
from torch import Size, Tensor
from torch.distributions import Gamma

TKernelFeatureMapGenerator = Callable[[kernels.Kernel, int, int], maps.KernelFeatureMap]
GenKernelFeatureMap = Dispatcher("gen_kernel_feature_map")


def gen_kernel_feature_map(
    kernel: kernels.Kernel,
    num_random_features: int = 1024,
    num_ambient_inputs: Optional[int] = None,
    **kwargs: Any,
) -> maps.KernelFeatureMap:
    r"""Generates a feature map :math:`\phi: \mathcal{X} \to \mathbb{R}^{n}` such that
    :math:`k(x, x') ≈ \phi(x)^{T} \phi(x')`. For stationary kernels :math:`k`, defaults
    to the method of random Fourier features. For more details, see [rahimi2007random]_
    and [sutherland2015error]_.

    Args:
        kernel: The kernel :math:`k` to be represented via a feature map.
        num_random_features: The number of random features used to estimate kernels
            that cannot be exactly represented as finite-dimensional feature maps.
        num_ambient_inputs: The number of ambient input features. Typically acts as a
            required argument for kernels with lengthscales whose :code:`active_dims`
            and :code:`ard_num_dims` attributes are both None.
        **kwargs: Additional keyword arguments are passed to subroutines.
    """
    return GenKernelFeatureMap(
        kernel,
        num_ambient_inputs=num_ambient_inputs,
        num_random_features=num_random_features,
        **kwargs,
    )

@GenKernelFeatureMap.register(kernels.ScaleKernel)
def _gen_kernel_feature_map_scale(
    kernel: kernels.ScaleKernel,
    *,
    num_ambient_inputs: Optional[int] = None,
    **kwargs: Any,
) -> maps.KernelFeatureMap:

    active_dims = kernel.active_dims
    num_scale_kernel_inputs = get_kernel_num_inputs(
        kernel=kernel,
        num_ambient_inputs=num_ambient_inputs,
        default=None,
    )
    feature_map = gen_kernel_feature_map(
        kernel.base_kernel, num_ambient_inputs=num_scale_kernel_inputs, **kwargs
    )

    # Maybe include a transform that extract relevant input features
    if active_dims is not None and active_dims is not kernel.base_kernel.active_dims:
        append_transform(
            module=feature_map,
            attr_name="input_transform",
            transform=transforms.FeatureSelector(indices=active_dims),
        )

    # Include a transform that multiplies by the square root of the kernel's outputscale
    prepend_transform(
        module=feature_map,
        attr_name="output_transform",
        transform=transforms.OutputscaleTransform(kernel),
    )
    return feature_map


@GenKernelFeatureMap.register(kernels.AdditiveKernel)
def _gen_kernel_feature_map_additive(
    kernel: kernels.AdditiveKernel,
    sub_kernels: Optional[Iterable[kernels.Kernel]] = None,
    **kwargs: Any,
) -> maps.DirectSumFeatureMap:
    feature_maps = [
        gen_kernel_feature_map(kernel=sub_kernel, **kwargs)
        for sub_kernel in (kernel.kernels if sub_kernels is None else sub_kernels)
    ]
    # Return direct sum `concat([f(x) for f in feature_maps], -1)`
    # Note: Direct sums only translate to concatenations for vector-valued feature maps
    return maps.DirectSumFeatureMap(feature_maps=feature_maps)


@GenKernelFeatureMap.register(kernels.ProductKernel)
def _gen_kernel_feature_map_product(
    kernel: kernels.ProductKernel,
    sub_kernels: Optional[Iterable[kernels.Kernel]] = None,
    cosine_only: Optional[bool] = DEFAULT,
    num_random_features: Optional[int] = None,
    **kwargs: Any,
) -> maps.OuterProductFeatureMap:
    sub_kernels = kernel.kernels if sub_kernels is None else sub_kernels
    if cosine_only is DEFAULT:
        # Note: We need to set `cosine_only=True` here in order to take the element-wise
        # product of features below. Otherwise, we would need to take the tensor product
        # of each pair of sine and cosine features.
        cosine_only = sum(not is_finite_dimensional(k) for k in sub_kernels) > 1

    # Generate feature maps for each sub-kernel
    sub_maps = []
    random_maps = []
    for sub_kernel in sub_kernels:
        sub_map = gen_kernel_feature_map(
            kernel=sub_kernel,
            cosine_only=cosine_only,
            num_random_features=num_random_features,
            random_feature_scale=1.0,  # we rescale once at the end
            **kwargs,
        )
        if is_finite_dimensional(sub_kernel):
            sub_maps.append(sub_map)
        else:
            random_maps.append(sub_map)

    # Define element-wise product of random feature maps
    if random_maps:
        random_map = (
            next(iter(random_maps))
            if len(random_maps) == 1
            else maps.HadamardProductFeatureMap(feature_maps=random_maps)
        )
        constant = torch.tensor(
            num_random_features**-0.5, device=kernel.device, dtype=kernel.dtype
        )
        prepend_transform(
            module=random_map,
            attr_name="output_transform",
            transform=transforms.ConstantMulTransform(constant),
        )
        sub_maps.append(random_map)

    # Return outer product `einsum("i,j,k->ijk", ...).view(-1)`
    return maps.OuterProductFeatureMap(feature_maps=sub_maps)


@GenKernelFeatureMap.register(kernels.IndexKernel)
def _gen_kernel_feature_map_index(
    kernel: kernels.IndexKernel, **ignore: Any
) -> maps.IndexKernelFeatureMap:
    return maps.IndexKernelFeatureMap(kernel=kernel)


@GenKernelFeatureMap.register(kernels.LinearKernel)
def _gen_kernel_feature_map_linear(
    kernel: kernels.LinearKernel,
    *,
    num_inputs: Optional[int] = None,
    **ignore: Any,
) -> maps.LinearKernelFeatureMap:
    num_features = get_kernel_num_inputs(kernel=kernel, num_ambient_inputs=num_inputs)
    return maps.LinearKernelFeatureMap(
        kernel=kernel, raw_output_shape=Size([num_features])
    )


@GenKernelFeatureMap.register(kernels.MultitaskKernel)
def _gen_kernel_feature_map_multitask(
    kernel: kernels.MultitaskKernel, **kwargs: Any
) -> maps.MultitaskKernelFeatureMap:
    # Generate a feature map for the data kernel
    data_feature_map = gen_kernel_feature_map(kernel.data_covar_module, **kwargs)
    if len(data_feature_map.output_shape) != 1:
        raise NotImplementedError  # TODO: Test what still works and give a hint

    return maps.MultitaskKernelFeatureMap(
        kernel=kernel, data_feature_map=data_feature_map
    )


@GenKernelFeatureMap.register(kernels.LCMKernel)
def _gen_kernel_feature_map_lcm(
    kernel: kernels.LCMKernel, **kwargs: Any
) -> maps.DirectSumFeatureMap:
    return _gen_kernel_feature_map_additive(
        kernel=kernel, sub_kernels=kernel.covar_module_list, **kwargs
    )



def _gen_fourier_features(
    kernel: kernels.Kernel,
    weight_generator: Callable[[Size], Tensor],
    num_random_features: int,
    num_inputs: Optional[int] = None,
    random_feature_scale: Optional[float] = None,
    cosine_only: bool = False,
    **ignore: Any,
) -> maps.FourierFeatureMap:
    r"""Generate a feature map :math:`\phi: \mathcal{X} \to \mathbb{R}^{2l}` that
    approximates a stationary kernel so that :math:`k(x, x') ≈ \phi(x)^\top \phi(x')`.

    Following [sutherland2015error]_, we default to representing complex exponentials
    by pairs of basis functions :math:`\phi_{i}(x) = \sin(x^\top w_{i})` and
    :math:`\phi_{i + l} = \cos(x^\top w_{i}).

    Args:
        kernel: A stationary kernel :math:`k(x, x') = k(x - x')`.
        weight_generator: A callable used to generate weight vectors :math:`w`.
        num_inputs: The number of ambient input features.
        num_random_features: The number of random Fourier features.
        random_feature_scale: Multiplicative constant for the feature map :math:`\phi`.
            Defaults to :code:`num_random_features ** -0.5` so that
            :math:`\phi(x)^\top \phi(x') ≈ k(x, x')`.
        cosine_only: Specifies whether or not to use cosine features with a random
            phase instead of paired sine and cosine features.
    """

    tkwargs = {"device": kernel.device, "dtype": kernel.dtype}
    num_inputs = get_kernel_num_inputs(kernel, num_ambient_inputs=num_inputs)
    input_transform = transforms.InverseLengthscaleTransform(kernel)
    if kernel.active_dims is not None:
        num_inputs = len(kernel.active_dims)

    constant = torch.tensor(
        2**0.5 * (random_feature_scale or num_random_features**-0.5), **tkwargs
    )
    output_transforms = [transforms.ConstantMulTransform(constant)]
    if cosine_only:
        bias = 2 * pi * torch.rand(num_random_features, **tkwargs)
        num_raw_features = num_random_features
        output_transforms.append(transforms.CosineTransform())
    elif num_random_features % 2:
        raise UnsupportedError(
            f"Expected an even number of random features, but {num_random_features=}."
         )
    else:
        bias = None
        num_raw_features = num_random_features // 2
        output_transforms.append(transforms.SineCosineTransform())
        

    weight = weight_generator(
        Size([kernel.batch_shape.numel() * num_raw_features, num_inputs])
    ).reshape(*kernel.batch_shape, num_raw_features, num_inputs)


    return maps.FourierFeatureMap(
        kernel=kernel,
        weight=weight,
        bias=bias,
        input_transform=input_transform,
        output_transform=transforms.ChainedTransform(*output_transforms),
    )


@GenKernelFeatureMap.register(kernels.RBFKernel)
def _gen_kernel_feature_map_rbf(
    kernel: kernels.RBFKernel, **kwargs: Any
) -> maps.FourierFeatureMap:
    def _weight_generator(shape: Size) -> Tensor:
        try:
            n, d = shape
        except ValueError:
            raise UnsupportedError(
                f"Expected `shape` to be 2-dimensional, but {len(shape)=}."
            )

        return draw_sobol_normal_samples(
            n=n,
            d=d,
            device=kernel.device,
            dtype=kernel.dtype,
        )

    return _gen_fourier_features(
        kernel=kernel, weight_generator=_weight_generator, **kwargs
    )


@GenKernelFeatureMap.register(kernels.MaternKernel)
def _gen_kernel_feature_map_matern(
    kernel: kernels.MaternKernel, **kwargs: Any
) -> maps.FourierFeatureMap:
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
        return Gamma(nu, nu).rsample((n, 1)).rsqrt() * normals

    return _gen_fourier_features(
        kernel=kernel, weight_generator=_weight_generator, **kwargs
    )