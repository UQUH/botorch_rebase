#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
from botorch import models
from botorch.models.model import Model
from botorch.sampling.pathwise.features.generators import gen_kernel_feature_map
from botorch.sampling.pathwise.utils import get_train_inputs
from gpytorch import kernels
from torch import Size

# TestCaseConfig: Configuration dataclass for test setup
# - Provides consistent test parameters across different test cases
# - Includes device, dtype, dimensions, and other key parameters
@dataclass(frozen=True)
class TestCaseConfig:
    device: torch.device
    dtype: torch.dtype = torch.float64
    seed: int = 0
    num_inputs: int = 2
    num_tasks: int = 2
    num_train: int = 5
    batch_shape: Size = Size([])
    num_random_features: int = 2048

# gen_random_inputs: Generates random input tensors for testing
# - Handles both single-task and multi-task models
# - Supports transformed/untransformed inputs
# - Manages task indices for multi-task models
def gen_random_inputs(
    model: Model,
    batch_shape: list[int],
    transformed: bool = False,
    task_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate random inputs for testing.
    
    Args:
        model: Model to generate inputs for
        batch_shape: Shape of batch dimension
        transformed: Whether to return transformed inputs
        task_id: Optional task ID for multi-task models
        seed: Optional random seed
        
    Returns:
        Tensor: Random input tensor
    """
    with (torch.random.fork_rng() if seed is not None else nullcontext()):
        if seed:
            torch.random.manual_seed(seed)

        (train_X,) = get_train_inputs(model, transformed=True)
        tkwargs = {"device": train_X.device, "dtype": train_X.dtype}
        
        # Get input dimension excluding task feature for MultiTaskGP
        input_dim = train_X.shape[-1]
        if isinstance(model, models.MultiTaskGP):
            input_dim -= 1  # Subtract task feature dimension
            
        # Generate random inputs
        X = torch.rand((*batch_shape, input_dim), **tkwargs)
        
        # Add task feature for MultiTaskGP
        if isinstance(model, models.MultiTaskGP):
            task_indices = torch.randint(0, model.num_tasks, (*batch_shape,), **tkwargs)  # Start from 0
            task_indices = task_indices.unsqueeze(-1)
            X = torch.cat([X, task_indices], dim=-1)
            if task_id is not None:
                X[..., -1] = task_id
                
        # Apply transforms if needed
        if not transformed and hasattr(model, "input_transform"):
            X = model.input_transform.untransform(X)
        return X

# gen_module: Factory function for generating test modules
# - Creates and configures various model types for testing
# - Handles SingleTaskGP, MultiTaskGP, ModelListGP, etc.
# - Sets up appropriate kernels and transforms
def gen_module(typ: Any, config: TestCaseConfig, **kwargs: Any) -> Any:
    """Generate a test module instance with given config."""
    if issubclass(typ, kernels.Kernel):
        kwargs.setdefault("batch_shape", config.batch_shape)
        if typ == kernels.RBFKernel:
            kwargs.setdefault("ard_num_dims", config.num_inputs)
        elif typ == kernels.LinearKernel:
            kwargs.setdefault("active_dims", [0])
        elif typ == kernels.IndexKernel:
            kwargs.setdefault("num_tasks", config.num_tasks)
            kwargs.setdefault("rank", kwargs["num_tasks"])
            kwargs.setdefault("active_dims", [0])
        elif typ == kernels.MultitaskKernel:
            kwargs.setdefault("num_tasks", config.num_tasks)
            kwargs.setdefault("rank", kwargs["num_tasks"])
            if "data_covar_module" not in kwargs:
                kwargs["data_covar_module"] = gen_module(kernels.RBFKernel, config)
    elif issubclass(typ, Model):
        # Generate training data
        tkwargs = {"device": config.device, "dtype": config.dtype}
        X = torch.rand(config.num_train, config.num_inputs, **tkwargs)
        Y = torch.randn(config.num_train, 1, **tkwargs)
        if typ == models.SingleTaskGP:
            kwargs.update(train_X=X, train_Y=Y)
        elif typ == models.MultiTaskGP:
            # Add task feature column and handle task indices
            task_feature = -1
            task_values = torch.arange(config.num_tasks, **tkwargs)  # Start from 0
            task_indices = torch.randint(0, config.num_tasks, (config.num_train,), **tkwargs)  # Start from 0
            X = torch.cat([X, task_indices.unsqueeze(-1)], dim=-1)
            kwargs.update(train_X=X, train_Y=Y, task_feature=task_feature)
        elif typ == models.SingleTaskVariationalGP:
            kwargs.update(train_X=X, train_Y=Y)
        elif typ == models.ModelListGP:
            # Create a list of models to combine
            models_list = [
                gen_module(models.SingleTaskGP, config),
                gen_module(models.SingleTaskGP, config),
            ]
            return typ(*models_list)

    module = typ(**kwargs)
    if hasattr(module, "raw_lengthscale"):
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            shape = module.raw_lengthscale.shape if module.ard_num_dims is None else Size([*module.batch_shape, 1, module.ard_num_dims])
            module.raw_lengthscale = torch.nn.Parameter(torch.zeros(shape, dtype=config.dtype, device=config.device))
            module.raw_lengthscale.data.add_(torch.rand_like(module.raw_lengthscale) * 0.2 - 2.0)

    return module.to(device=config.device, dtype=config.dtype) 