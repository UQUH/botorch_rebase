#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from itertools import repeat
from math import prod
from string import ascii_letters
from typing import Any, Iterable, List, Optional, Union

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.utils import (
    ChainedTransform,
    FeatureSelector,
    ModuleListMixin,
    TInputTransform,
    TOutputTransform,
    TransformedModuleMixin,
    untransform_shape,
)
from gpytorch import kernels
from linear_operator.operators import (
    InterpolatedLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
)
from torch import Size, Tensor
from torch.nn import Module


class FeatureMap(TransformedModuleMixin, Module):
    raw_output_shape: Size
    batch_shape: Size
    input_transform: Optional[TInputTransform]
    output_transform: Optional[TOutputTransform]
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    @abstractmethod
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        pass

    @property
    def output_shape(self) -> Size:
        if self.output_transform is None:
            return self.raw_output_shape

        return untransform_shape(
            self.output_transform,
            self.raw_output_shape,
            device=self.device,
            dtype=self.dtype,
        )


class FeatureMapList(Module, ModuleListMixin[FeatureMap]):
    def __init__(self, feature_maps: Iterable[FeatureMap]):
        Module.__init__(self)
        ModuleListMixin.__init__(self, attr_name="feature_maps", modules=feature_maps)

    def forward(self, x: Tensor, **kwargs: Any) -> List[Union[Tensor, LinearOperator]]:
        return [feature_map(x, **kwargs) for feature_map in self]

    @property
    def device(self) -> Optional[torch.device]:
        devices = {feature_map.device for feature_map in self}
        devices.discard(None)
        if len(devices) > 1:
            raise UnsupportedError(f"Feature maps must be colocated, but {devices=}.")
        return next(iter(devices)) if devices else None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        dtypes = {feature_map.dtype for feature_map in self}
        dtypes.discard(None)
        if len(dtypes) > 1:
            raise UnsupportedError(
                f"Feature maps must have the same data type, but {dtypes=}."
            )
        return next(iter(dtypes)) if dtypes else None


class DirectSumFeatureMap(FeatureMap, FeatureMapList):
    r"""Direct sums of features."""

    def __init__(
        self,
        feature_maps: Iterable[FeatureMap],
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
    ):
        FeatureMap.__init__(self)
        FeatureMapList.__init__(self, feature_maps=feature_maps)
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        blocks = []
        shape = self.raw_output_shape
        ndim = len(shape)
        for feature_map in self:
            block = feature_map(x, **kwargs).to_dense()
            block_ndim = len(feature_map.output_shape)
            if block_ndim < ndim:
                tile_shape = shape[-ndim:-block_ndim]
                num_copies = prod(tile_shape)
                if num_copies > 1:
                    block = block * (num_copies**-0.5)

                multi_index = (
                    ...,
                    *repeat(None, ndim - block_ndim),
                    *repeat(slice(None), block_ndim),
                )
                block = block[multi_index].expand(
                    *block.shape[:-block_ndim], *tile_shape, *block.shape[-block_ndim:]
                )
            blocks.append(block)

        return torch.concat(blocks, dim=-1)

    @property
    def raw_output_shape(self) -> Size:
        map_iter = iter(self)
        feature_map = next(map_iter)
        concat_size = feature_map.output_shape[-1]
        batch_shape = feature_map.output_shape[:-1]
        for feature_map in map_iter:
            concat_size += feature_map.output_shape[-1]
            batch_shape = torch.broadcast_shapes(
                batch_shape, feature_map.output_shape[:-1]
            )
        return Size((*batch_shape, concat_size))

    @property
    def batch_shape(self) -> Size:
        batch_shapes = {feature_map.batch_shape for feature_map in self}
        if len(batch_shapes) > 1:
            raise ValueError(
                f"Component maps have the same batch shapes, but {batch_shapes=}."
            )
        return next(iter(batch_shapes))


class OuterProductFeatureMap(FeatureMap, FeatureMapList):
    r"""Outer product of vector-valued features."""

    def __init__(
        self,
        feature_maps: Iterable[FeatureMap],
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
    ):
        FeatureMap.__init__(self)
        FeatureMapList.__init__(self, feature_maps=feature_maps)
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        num_maps = len(self)
        lhs = (f"...{ascii_letters[i]}" for i in range(num_maps))
        rhs = f"...{ascii_letters[:num_maps]}"
        eqn = f"{','.join(lhs)}->{rhs}"

        outputs_iter = (feature_map(x, **kwargs).to_dense() for feature_map in self)
        output = torch.einsum(eqn, *outputs_iter)
        return output.view(*output.shape[:-num_maps], -1)

    @property
    def raw_output_shape(self) -> Size:
        outer_size = 1
        batch_shapes = []
        for feature_map in self:
            *batch_shape, size = feature_map.output_shape
            outer_size *= size
            batch_shapes.append(batch_shape)
        return Size((*torch.broadcast_shapes(*batch_shapes), outer_size))

    @property
    def batch_shape(self) -> Size:
        batch_shapes = (feature_map.batch_shape for feature_map in self)
        return torch.broadcast_shapes(*batch_shapes)


class KernelFeatureMap(FeatureMap):
    r"""Base class for FeatureMap subclasses that represent kernels."""

    def __init__(
        self,
        kernel: kernels.Kernel,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
        ignore_active_dims: bool = False,
    ) -> None:
        r"""Initializes a KernelFeatureMap instance.

        Args:
            kernel: The kernel :math:`k` used to define the feature map.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
            ignore_active_dims: Whether to ignore the kernel's active_dims.
        """
        if not ignore_active_dims and kernel.active_dims is not None:
            feature_selector = FeatureSelector(kernel.active_dims)
            if input_transform is None:
                input_transform = feature_selector
            else:
                input_transform = ChainedTransform(input_transform, feature_selector)

        super().__init__()
        self.kernel = kernel
        self.input_transform = input_transform
        self.output_transform = output_transform

    @property
    def batch_shape(self) -> Size:
        return self.kernel.batch_shape

    @property
    def device(self) -> Optional[torch.device]:
        return self.kernel.device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self.kernel.dtype


class KernelEvaluationMap(KernelFeatureMap):
    r"""A feature map defined by centering a kernel at a set of points."""

    def __init__(
        self,
        kernel: kernels.Kernel,
        points: Tensor,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
    ) -> None:
        r"""Initializes a KernelEvaluationMap instance.

        Args:
            kernel: The kernel :math:`k` used to define the feature map.
            points: A tensor passed as the kernel's second argument.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """
        if not 1 < points.ndim < len(kernel.batch_shape) + 3:
            raise RuntimeError(
                f"Dimension mismatch: {points.ndim=}, but {len(kernel.batch_shape)=}."
            )

        try:
            torch.broadcast_shapes(points.shape[:-2], kernel.batch_shape)
        except RuntimeError:
            raise RuntimeError(
                f"Shape mismatch: {points.shape=}, but {kernel.batch_shape=}."
            )

        super().__init__(
            kernel=kernel,
            input_transform=input_transform,
            output_transform=output_transform,
        )
        self.points = points

    def forward(self, x: Tensor) -> Union[Tensor, LinearOperator]:
        return self.kernel(x, self.points)

    @property
    def raw_output_shape(self) -> Size:
        return self.points.shape[-2:-1]


class FourierFeatureMap(KernelFeatureMap):
    r"""Representation of a kernel :math:`k: \mathcal{X}^2 \to \mathbb{R}` as an
    n-dimensional feature map :math:`\phi: \mathcal{X} \to \mathbb{R}^n` satisfying:
    :math:`k(x, x') ≈ \phi(x)^\top \phi(x')`.
    """

    def __init__(
        self,
        kernel: kernels.Kernel,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
    ) -> None:
        r"""Initializes a FourierFeatureMap instance:

        .. code-block:: text

            feature_map(x) = output_transform(input_transform(x)^{T} weight + bias).

        Args:
            kernel: The kernel :math:`k` used to define the feature map.
            weight: A tensor of weights used to linearly combine the module's inputs.
            bias: A tensor of biases to be added to the linearly combined inputs.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """
        super().__init__(
            kernel=kernel,
            input_transform=input_transform,
            output_transform=output_transform,
        )
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose(-2, -1)
        return out if self.bias is None else out + self.bias.unsqueeze(-2)

    @property
    def raw_output_shape(self) -> Size:
        return self.weight.shape[-2:-1]


class IndexKernelFeatureMap(KernelFeatureMap):
    def __init__(
        self,
        kernel: kernels.IndexKernel,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
        ignore_active_dims: bool = False,
    ) -> None:
        r"""Initializes an IndexKernelFeatureMap instance:

        Args:
            kernel: IndexKernel whose features are to be returned.
            input_transform: An optional input transform for the module.
                For kernels with `active_dims`, defaults to a FeatureSelector
                instance that extracts the relevant input features.
            output_transform: An optional output transform for the module.
        """
        if not isinstance(kernel, kernels.IndexKernel):
            raise ValueError(f"Expected {kernels.IndexKernel}, but {type(kernel)=}.")

        super().__init__(
            kernel=kernel,
            input_transform=input_transform,
            output_transform=output_transform,
            ignore_active_dims=ignore_active_dims,
        )

    def forward(self, x: Optional[Tensor]) -> LinearOperator:
        if x is None:
            return self.kernel.covar_matrix.cholesky()

        i = x.long()
        j = torch.arange(self.kernel.covar_factor.shape[-1], device=x.device)[..., None]
        batch = torch.broadcast_shapes(self.batch_shape, i.shape[:-2], j.shape[:-2])
        return InterpolatedLinearOperator(
            base_linear_op=self.kernel.covar_matrix.cholesky(),
            left_interp_indices=i.expand(batch + i.shape[-2:]),
            right_interp_indices=j.expand(batch + j.shape[-2:]),
        ).to_dense()

    @property
    def raw_output_shape(self) -> Size:
        return self.kernel.raw_var.shape[-1:]


class LinearKernelFeatureMap(KernelFeatureMap):
    def __init__(
        self,
        kernel: kernels.LinearKernel,
        raw_output_shape: Size,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
        ignore_active_dims: bool = False,
    ) -> None:
        r"""Initializes a LinearKernelFeatureMap instance.

        Args:
            kernel: LinearKernel whose features are to be returned.
            raw_output_shape: The shape of the raw output features.
            input_transform: An optional input transform for the module.
                For kernels with `active_dims`, defaults to a FeatureSelector
                instance that extracts the relevant input features.
            output_transform: An optional output transform for the module.
        """
        if not isinstance(kernel, kernels.LinearKernel):
            raise ValueError(f"Expected {kernels.LinearKernel}, but {type(kernel)=}.")

        super().__init__(
            kernel=kernel,
            input_transform=input_transform,
            output_transform=output_transform,
            ignore_active_dims=ignore_active_dims,
        )
        self.raw_output_shape = raw_output_shape

    def forward(self, x: Tensor) -> Tensor:
        return self.kernel.variance.sqrt() * x


class MultitaskKernelFeatureMap(KernelFeatureMap):
    r"""Representation of a MultitaskKernel as a feature map."""

    def __init__(
        self,
        kernel: kernels.MultitaskKernel,
        data_feature_map: FeatureMap,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
        ignore_active_dims: bool = False,
    ) -> None:
        r"""Initializes a MultitaskKernelFeatureMap instance.

        Args:
            kernel: MultitaskKernel whose features are to be returned.
            data_feature_map: Representation of the multitask kernel's
                `data_covar_module` as a FeatureMap.
            input_transform: An optional input transform for the module.
                For kernels with `active_dims`, defaults to a FeatureSelector
                instance that extracts the relevant input features.
            output_transform: An optional output transform for the module.
        """
        if not isinstance(kernel, kernels.MultitaskKernel):
            raise ValueError(
                f"Expected {kernels.MultitaskKernel}, but {type(kernel)=}."
            )

        super().__init__(
            kernel=kernel,
            input_transform=input_transform,
            output_transform=output_transform,
            ignore_active_dims=ignore_active_dims,
        )
        self.data_feature_map = data_feature_map

    def forward(self, x: Tensor) -> Union[KroneckerProductLinearOperator, Tensor]:
        r"""Returns the Kronecker product of the square root task covariance matrix
        and a feature-map-based representation of :code:`data_covar_module`.
        """
        data_features = self.data_feature_map(x)
        task_features = self.kernel.task_covar_module.covar_matrix.cholesky()
        task_features = task_features.expand(
            *data_features.shape[: max(0, data_features.ndim - task_features.ndim)],
            *task_features.shape,
        )
        return KroneckerProductLinearOperator(data_features, task_features)

    @property
    def num_tasks(self) -> int:
        return self.kernel.num_tasks

    @property
    def raw_output_shape(self) -> Size:
        size0, *sizes = self.data_feature_map.output_shape
        return Size((self.num_tasks * size0, *sizes))
