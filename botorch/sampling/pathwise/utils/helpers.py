#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, overload, Tuple, Type, Union

import torch
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model, ModelList
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.sampling.pathwise.utils.transforms import OutcomeUntransformer
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.types import MISSING
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from torch import Size, Tensor

GetTrainInputs = Dispatcher("get_train_inputs")
GetTrainTargets = Dispatcher("get_train_targets")


def kernel_instancecheck(kernel: Kernel, kernel_type: Type[Kernel]) -> bool:
    """Check if a kernel is an instance of a kernel type, ignoring ScaleKernel wrappers."""
    if isinstance(kernel, kernel_type):
        return True
    if isinstance(kernel, ScaleKernel):
        return kernel_instancecheck(kernel.base_kernel, kernel_type)
    return False


def is_finite_dimensional(kernel: Kernel, max_depth: int = -1) -> bool:
    """Check if a kernel has a finite-dimensional feature map.

    Args:
        kernel: The kernel to check.
        max_depth: The maximum depth to search for finite-dimensional kernels.
            If -1, search all the way down. If 0, only check the top level.

    Returns:
        True if the kernel has a finite-dimensional feature map, False otherwise.
    """
    # TODO: Add support for more kernels
    if max_depth == 0:
        return not kernel_instancecheck(kernel, RBFKernel)
    
    if isinstance(kernel, ScaleKernel):
        return is_finite_dimensional(kernel.base_kernel, max_depth=max_depth-1 if max_depth > 0 else -1)
    
    return not kernel_instancecheck(kernel, RBFKernel)


def get_kernel_num_inputs(
    kernel: Kernel,
    num_ambient_inputs: Optional[int] = None,
    default: Optional[Optional[int]] = MISSING,
) -> Optional[int]:
    if kernel.active_dims is not None:
        return len(kernel.active_dims)

    if kernel.ard_num_dims is not None:
        return kernel.ard_num_dims

    if num_ambient_inputs is None:
        if default is MISSING:
            raise ValueError(
                "`num_ambient_inputs` must be passed when `kernel.active_dims` and "
                "`kernel.ard_num_dims` are both None and no `default` has been defined."
            )
        return default
    return num_ambient_inputs


def get_input_transform(model: GPyTorchModel) -> Optional[InputTransform]:
    r"""Returns a model's input_transform or None."""
    return getattr(model, "input_transform", None)


def get_output_transform(model: GPyTorchModel) -> Optional[OutcomeUntransformer]:
    r"""Returns a wrapped version of a model's outcome_transform or None."""
    transform = getattr(model, "outcome_transform", None)
    if transform is None:
        return None

    return OutcomeUntransformer(transform=transform, num_outputs=model.num_outputs)


@overload
def get_train_inputs(model: Model, transformed: bool = False) -> Tuple[Tensor, ...]:
    pass  # pragma: no cover


@overload
def get_train_inputs(model: ModelList, transformed: bool = False) -> List[...]:
    pass  # pragma: no cover


def get_train_inputs(model: Model, transformed: bool = False):
    return GetTrainInputs(model, transformed=transformed)


@GetTrainInputs.register(Model)
def _get_train_inputs_Model(model: Model, transformed: bool = False) -> Tuple[Tensor]:
    if not transformed:
        original_train_input = getattr(model, "_original_train_inputs", None)
        if torch.is_tensor(original_train_input):
            return (original_train_input,)

    (X,) = model.train_inputs
    transform = get_input_transform(model)
    if transform is None:
        return (X,)

    if model.training:
        return (transform.forward(X) if transformed else X,)
    return (X if transformed else transform.untransform(X),)


@GetTrainInputs.register(SingleTaskVariationalGP)
def _get_train_inputs_SingleTaskVariationalGP(
    model: SingleTaskVariationalGP, transformed: bool = False
) -> Tuple[Tensor]:
    (X,) = model.model.train_inputs
    if model.training != transformed:
        return (X,)

    transform = get_input_transform(model)
    if transform is None:
        return (X,)

    return (transform.forward(X) if model.training else transform.untransform(X),)


@GetTrainInputs.register(ModelList)
def _get_train_inputs_ModelList(
    model: ModelList, transformed: bool = False
) -> List[...]:
    return [get_train_inputs(m, transformed=transformed) for m in model.models]


@overload
def get_train_targets(model: Model, transformed: bool = False) -> Tensor:
    pass  # pragma: no cover


@overload
def get_train_targets(model: ModelList, transformed: bool = False) -> List[...]:
    pass  # pragma: no cover


def get_train_targets(model: Model, transformed: bool = False):
    return GetTrainTargets(model, transformed=transformed)


@GetTrainTargets.register(Model)
def _get_train_targets_Model(model: Model, transformed: bool = False) -> Tensor:
    Y = model.train_targets

    # Note: Avoid using `get_output_transform` here since it creates a Module
    transform = getattr(model, "outcome_transform", None)
    if transformed or transform is None:
        return Y

    if model.num_outputs == 1:
        return transform.untransform(Y.unsqueeze(-1))[0].squeeze(-1)
    return transform.untransform(Y.transpose(-2, -1))[0].transpose(-2, -1)


@GetTrainTargets.register(SingleTaskVariationalGP)
def _get_train_targets_SingleTaskVariationalGP(
    model: Model, transformed: bool = False
) -> Tensor:
    Y = model.model.train_targets
    transform = getattr(model, "outcome_transform", None)
    if transformed or transform is None:
        return Y

    if model.num_outputs == 1:
        return transform.untransform(Y.unsqueeze(-1))[0].squeeze(-1)

    # SingleTaskVariationalGP.__init__ doesn't bring the multitoutpout dimension inside
    return transform.untransform(Y)[0]


@GetTrainTargets.register(ModelList)
def _get_train_targets_ModelList(
    model: ModelList, transformed: bool = False
) -> List[...]:
    return [get_train_targets(m, transformed=transformed) for m in model.models]


def untransform_shape(
    transform: Union[InputTransform, OutcomeTransform],
    shape: Size,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Size:
    if transform is None:
        return shape

    test_case = torch.empty(shape, device=device, dtype=dtype)
    if isinstance(transform, OutcomeTransform):
        result, _ = transform.untransform(test_case)
    elif isinstance(transform, InputTransform):
        result = transform.untransform(test_case)
    else:
        result = transform(test_case)

    # TODO: This function assumes that dimensionality remains unchanged
    return result.shape[-test_case.ndim :] 