#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch
from botorch import models
from botorch.models import SingleTaskVariationalGP
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    draw_matheron_paths,
    MatheronPath,
    PathList,
)
from botorch.sampling.pathwise.utils import is_finite_dimensional
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Size

from .helpers import gen_module, gen_random_inputs, TestCaseConfig


class TestDrawMatheronPaths(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Set up test configurations
        config = TestCaseConfig(seed=0, device=self.device)
        batch_config = replace(config, batch_shape=Size([2]))

        # Initialize test models with different configurations
        self.base_models = [
            (batch_config, gen_module(models.SingleTaskGP, batch_config)),
            (batch_config, gen_module(models.MultiTaskGP, batch_config)),
            (config, gen_module(models.SingleTaskVariationalGP, config)),
        ]
        self.model_lists = [
            (batch_config, gen_module(models.ModelListGP, batch_config))
        ]

    def test_base_models(self, slack: float = 3.0):
        """Test Matheron path sampling for base model types."""
        sample_shape = Size([32, 32])
        for config, model in self.base_models:
            # Get appropriate kernel based on model type
            kernel = (
                model.model.covar_module
                if isinstance(model, models.SingleTaskVariationalGP)
                else model.covar_module
            )
            # Handle feature indices for multi-task models
            base_features = list(range(config.num_inputs))
            if isinstance(model, models.MultiTaskGP) and hasattr(model, "_task_feature"):
                if model._task_feature < len(base_features):
                    del base_features[model._task_feature]

            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                # Draw paths using kernel features
                paths = draw_matheron_paths(
                    model=model,
                    sample_shape=sample_shape,
                    prior_sampler=partial(
                        draw_kernel_feature_paths,
                        num_random_features=config.num_random_features,
                    ),
                )
                self.assertIsInstance(paths, MatheronPath)
                
                # Generate test inputs
                n = 16
                Z = gen_random_inputs(
                    model,
                    batch_shape=[n],
                    transformed=True,
                    task_id=0,  # only used by multi-task models
                )
                X = (
                    model.input_transform.untransform(Z)
                    if hasattr(model, "input_transform")
                    else Z
                )

                # Sample paths and get model posterior
                samples = paths(X)
                model.eval()
                with delattr_ctx(model, "outcome_transform"):
                    posterior = (
                        model.posterior(X[..., base_features], output_indices=[0])
                        if isinstance(model, models.MultiTaskGP)
                        else model.posterior(X)
                    )
                    mvn = posterior.mvn

                # Handle multi-task vs single-task outputs
                if isinstance(mvn, MultitaskMultivariateNormal):
                    num_tasks = kernel.batch_shape[0]
                    exact_mean = mvn.mean.transpose(-2, -1)
                    exact_covar = mvn.covariance_matrix.view(num_tasks, n, num_tasks, n)
                    exact_covar = torch.stack(
                        [exact_covar[..., i, :, i, :] for i in range(num_tasks)], dim=-3
                    )
                else:
                    exact_mean = mvn.mean
                    exact_covar = mvn.covariance_matrix

                # Scale by prior standard deviations for comparison
                if isinstance(model, SingleTaskVariationalGP):
                    try:
                        prior = model.forward(Z, prior=True)
                    except TypeError:
                        prior = model.forward(Z)
                else:
                    prior = model.forward(Z)

                istd = prior.covariance_matrix.diagonal(dim1=-2, dim2=-1).rsqrt()
                exact_mean = istd * exact_mean
                exact_covar = istd.unsqueeze(-1) * exact_covar * istd.unsqueeze(-2)

                # Apply outcome transform if present
                if hasattr(model, "outcome_transform"):
                    if kernel.batch_shape:
                        samples, _ = model.outcome_transform(samples.transpose(-2, -1))
                        samples = samples.transpose(-2, -1)
                    else:
                        samples, _ = model.outcome_transform(samples.unsqueeze(-1))
                        samples = samples.squeeze(-1)

                # Compute sample statistics
                samples = istd * samples.view(-1, *samples.shape[len(sample_shape):])
                sample_mean = samples.mean(dim=0)
                sample_covar = (samples - sample_mean).permute(*range(1, samples.ndim), 0)
                sample_covar = torch.divide(
                    sample_covar @ sample_covar.transpose(-2, -1), sample_shape.numel()
                )

                # Set tolerances based on feature dimensionality
                allclose_kwargs = {"atol": slack * sample_shape.numel() ** -0.5}
                if not is_finite_dimensional(kernel):
                    num_random_features_per_map = config.num_random_features / (
                        1
                        if not is_finite_dimensional(kernel, max_depth=0)
                        else sum(
                            not is_finite_dimensional(k)
                            for k in kernel.modules()
                            if k is not kernel
                        )
                    )
                    allclose_kwargs["atol"] += slack * num_random_features_per_map**-0.5

                # Verify mean and covariance matching
                self.assertTrue(exact_mean.allclose(sample_mean, **allclose_kwargs))
                self.assertTrue(exact_covar.allclose(sample_covar, **allclose_kwargs))

    def test_model_lists(self, tol: float = 3.0):
        """Test Matheron path sampling for model lists."""
        sample_shape = Size([32, 32])
        for config, model_list in self.model_lists:
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                path_list = draw_matheron_paths(
                    model=model_list,
                    sample_shape=sample_shape,
                )
                self.assertIsInstance(path_list, PathList)

                # Verify path list operations
                X = gen_random_inputs(model_list.models[0], batch_shape=[4])
                sample_list = path_list(X)
                self.assertIsInstance(sample_list, list)
                self.assertEqual(len(sample_list), len(model_list.models))
                for path, sample in zip(path_list, sample_list):
                    self.assertTrue(path(X).equal(sample))
