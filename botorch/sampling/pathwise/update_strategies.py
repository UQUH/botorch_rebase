#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy

from collections.abc import Callable

from types import NoneType

from typing import Any

import torch
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.sampling.pathwise.features import KernelEvaluationMap
from botorch.sampling.pathwise.paths import GeneralizedLinearPath, SamplePath
from botorch.sampling.pathwise.utils import (
    get_input_transform,
    get_train_inputs,
    get_train_targets,
    TInputTransform,
)
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.types import DEFAULT
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods import _GaussianLikelihoodBase, Likelihood
from gpytorch.models import ApproximateGP, ExactGP, GP
from gpytorch.variational import VariationalStrategy
from linear_operator.operators import (
    LinearOperator,
    SumLinearOperator,
    ZeroLinearOperator,
)
from torch import Tensor

TPathwiseUpdate = Callable[[GP, Tensor], SamplePath]
GaussianUpdate = Dispatcher("gaussian_update")


def gaussian_update(
    model: GP,
    sample_values: Tensor,
    likelihood: Likelihood | None = DEFAULT,
    **kwargs: Any,
) -> GeneralizedLinearPath:
    r"""Computes a Gaussian pathwise update in exact arithmetic:

    .. code-block:: text

        (f | y)(·) = f(·) + Cov(f(·), y) Cov(y, y)^{-1} (y - f(X) - ε),
                            \_______________________________________/
                                                V
                                    "Gaussian pathwise update"

    where `=` denotes equality in distribution, :math:`f \sim GP(0, k)`,
    :math:`y \sim N(f(X), \Sigma)`, and :math:`\epsilon \sim N(0, \Sigma)`.
    For more information, see [wilson2020sampling]_ and [wilson2021pathwise]_.

    Args:
        model: A Gaussian process prior together with a likelihood.
        sample_values: Assumed values for :math:`f(X)`.
        likelihood: An optional likelihood used to help define the desired
            update. Defaults to `model.likelihood` if it exists else None.
        **kwargs: Additional keyword arguments are passed to subroutines.
    """
    if likelihood is DEFAULT:
        likelihood = getattr(model, "likelihood", None)

    return GaussianUpdate(model, likelihood, sample_values=sample_values, **kwargs)


def _gaussian_update_exact(
    kernel: Kernel,
    points: Tensor,
    target_values: Tensor,
    sample_values: Tensor,
    noise_covariance: Tensor | LinearOperator | None = None,
    scale_tril: Tensor | LinearOperator | None = None,
    input_transform: TInputTransform | None = None,
) -> GeneralizedLinearPath:
    # Prepare Cholesky factor of `Cov(y, y)` and noise sample values as needed
    if isinstance(noise_covariance, (NoneType, ZeroLinearOperator)):
        scale_tril = kernel(points).cholesky() if scale_tril is None else scale_tril
    else:
        noise_values = torch.randn_like(sample_values).unsqueeze(-1)
        noise_values = noise_covariance.cholesky() @ noise_values
        sample_values = sample_values + noise_values.squeeze(-1)
        scale_tril = (
            SumLinearOperator(kernel(points), noise_covariance).cholesky()
            if scale_tril is None
            else scale_tril
        )

    # Solve for `Cov(y, y)^{-1}(y - f(X) - ε)`
    errors = target_values - sample_values
    weight = torch.cholesky_solve(errors.unsqueeze(-1), scale_tril.to_dense())

    # Define update feature map and paths
    feature_map = KernelEvaluationMap(
        kernel=kernel,
        points=points,
        input_transform=input_transform,
    )
    return GeneralizedLinearPath(feature_map=feature_map, weight=weight.squeeze(-1))


@GaussianUpdate.register(ExactGP, _GaussianLikelihoodBase)
def _gaussian_update_ExactGP(
    model: ExactGP,
    likelihood: _GaussianLikelihoodBase,
    *,
    sample_values: Tensor,
    target_values: Tensor | None = None,
    points: Tensor | None = None,
    noise_covariance: Tensor | LinearOperator | None = None,
    scale_tril: Tensor | LinearOperator | None = None,
) -> GeneralizedLinearPath:
    if points is None:
        (points,) = get_train_inputs(model, transformed=True)

    if target_values is None:
        target_values = get_train_targets(model, transformed=True)

    if noise_covariance is None:
        noise_covariance = likelihood.noise_covar(shape=points.shape[:-1])

    return _gaussian_update_exact(
        kernel=model.covar_module,
        points=points,
        target_values=target_values,
        sample_values=sample_values,
        noise_covariance=noise_covariance,
        scale_tril=scale_tril,
        input_transform=get_input_transform(model),
    )

@GaussianUpdate.register(MultiTaskGP, _GaussianLikelihoodBase)
def _draw_kernel_feature_paths_MultiTaskGP(
    model: MultiTaskGP,
    likelihood: _GaussianLikelihoodBase,
    *,
    sample_values: Tensor,
    target_values: Optional[Tensor] = None,
    points: Optional[Tensor] = None,
    noise_covariance: Optional[Union[Tensor, LinearOperator]] = None,
    **ignore: Any,
) -> GeneralizedLinearPath:
    if points is None:
        (points,) = get_train_inputs(model, transformed=True)

    if target_values is None:
        target_values = get_train_targets(model, transformed=True)

    if noise_covariance is None:
        noise_covariance = likelihood.noise_covar(shape=points.shape[:-1])

    # Prepare product kernel
    num_inputs = points.shape[-1]
    task_index = model._task_feature
    base_kernel = deepcopy(model.covar_module)
    base_kernel.active_dims = torch.LongTensor(
        [index for index in range(num_inputs) if index != task_index],
        device=base_kernel.device,
    )
    task_kernel = deepcopy(model.task_covar_module)
    task_kernel.active_dims = torch.LongTensor([task_index], device=base_kernel.device)

    # Return exact update using product kernel
    return _gaussian_update_exact(
        kernel=base_kernel * task_kernel,
        points=points,
        target_values=target_values,
        sample_values=sample_values,
        noise_covariance=noise_covariance,
        input_transform=get_input_transform(model),
    )


@GaussianUpdate.register(ApproximateGPyTorchModel, (Likelihood, NoneType))
def _gaussian_update_ApproximateGPyTorchModel(
    model: ApproximateGPyTorchModel,
    likelihood: Likelihood | None,
    **kwargs: Any,
) -> GeneralizedLinearPath:
    return GaussianUpdate(
        model.model, likelihood, input_transform=get_input_transform(model), **kwargs
    )


@GaussianUpdate.register(ApproximateGP, (Likelihood, NoneType))
def _gaussian_update_ApproximateGP(
    model: ApproximateGP, likelihood: Likelihood | None, **kwargs: Any
) -> GeneralizedLinearPath:
    return GaussianUpdate(model, model.variational_strategy, **kwargs)


@GaussianUpdate.register(ApproximateGP, VariationalStrategy)
def _gaussian_update_ApproximateGP_VariationalStrategy(
    model: ApproximateGP,
    variational_strategy: VariationalStrategy,
    *,
    sample_values: Tensor,
    target_values: Tensor | None = None,
    noise_covariance: Tensor | LinearOperator | None = None,
    input_transform: InputTransform | None = None,
    **ignore: Any,
) -> GeneralizedLinearPath:
    # TODO: Account for jitter added by `psd_safe_cholesky`
    if not isinstance(noise_covariance, (NoneType, ZeroLinearOperator)):
        raise NotImplementedError(
            f"`noise_covariance` argument not yet supported for {type(model)}."
        )

    # Inducing points `Z` are assumed to live in transformed space
    batch_shape = model.covar_module.batch_shape
    Z = variational_strategy.inducing_points
    L = variational_strategy._cholesky_factor(
        variational_strategy(Z, prior=True).lazy_covariance_matrix
    ).to(dtype=sample_values.dtype)

    # Generate whitened inducing variables `u`, then location-scale transform
    if target_values is None:
        base_values = variational_strategy.variational_distribution.rsample(
            sample_values.shape[: sample_values.ndim - len(batch_shape) - 1],
        )
        # target_values = model.mean_module(Z) + (u @ L.transpose(-1, -2))
        target_values = model.mean_module(Z) + (L @ base_values.unsqueeze(-1)).squeeze(
            -1
        )

    return _gaussian_update_exact(
        kernel=model.covar_module,
        points=Z,
        target_values=target_values,
        sample_values=sample_values,
        scale_tril=L,
        input_transform=input_transform,
    )
