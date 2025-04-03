#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC
from string import ascii_letters
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features import FeatureMap
from botorch.sampling.pathwise.utils import (
    ModuleDictMixin,
    ModuleListMixin,
    TInputTransform,
    TOutputTransform,
    TransformedModuleMixin,
)
from torch import einsum, Tensor
from torch.nn import Module, ModuleDict, ModuleList, Parameter


class SamplePath(ABC, TransformedModuleMixin, Module):
    r"""Abstract base class for Botorch sample paths."""


class PathDict(SamplePath, ModuleDictMixin[SamplePath]):
    r"""A dictionary of SamplePaths."""

    def __init__(
        self,
        paths: Mapping[str, SamplePath] | None = None,
        reducer: Callable[[list[Tensor]], Tensor] | None = None,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ) -> None:
        r"""Initializes a PathDict instance.

        Args:
            paths: An optional mapping of strings to sample paths.
            reducer: An optional callable used to combine each path's outputs.
                Must be provided if output_transform is specified.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
                Can only be specified if reducer is provided.
        """
        if reducer is None and output_transform is not None:
            raise UnsupportedError(
                "`output_transform` must be preceded by a `reducer`."
            )

        SamplePath.__init__(self)
        self.reducer = reducer
        self.input_transform = input_transform
        self.output_transform = output_transform
        
        # Initialize paths dictionary - reuse ModuleDict if provided
        self._paths_dict = (
            paths if isinstance(paths, ModuleDict) else ModuleDict({} if paths is None else paths)
        )
        self.register_module("_paths_dict", self._paths_dict)

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor | dict[str, Tensor]:
        outputs = [path(x, **kwargs) for path in self._paths_dict.values()]
        return dict(zip(self._paths_dict, outputs)) if self.reducer is None else self.reducer(outputs)

    def items(self) -> Iterable[tuple[str, SamplePath]]:
        return self._paths_dict.items()

    def keys(self) -> Iterable[str]:
        return self._paths_dict.keys()

    def values(self) -> Iterable[SamplePath]:
        return self._paths_dict.values()

    def __len__(self) -> int:
        return len(self._paths_dict)

    def __iter__(self) -> Iterator[str]:
        yield from self._paths_dict

    def __delitem__(self, key: str) -> None:
        del self._paths_dict[key]

    def __getitem__(self, key: str) -> SamplePath:
        return self._paths_dict[key]

    def __setitem__(self, key: str, val: SamplePath) -> None:
        self._paths_dict[key] = val


class PathList(SamplePath, ModuleListMixin[SamplePath]):
    r"""A list of SamplePaths."""

    def __init__(
        self,
        paths: Iterable[SamplePath] | None = None,
        reducer: Callable[[list[Tensor]], Tensor] | None = None,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ) -> None:
        r"""Initializes a PathList instance.

        Args:
            paths: An optional iterable of sample paths.
            reducer: An optional callable used to combine each path's outputs.
                Must be provided if output_transform is specified.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
                Can only be specified if reducer is provided.
        """
        if reducer is None and output_transform is not None:
            raise UnsupportedError(
                "`output_transform` must be preceded by a `reducer`."
            )

        SamplePath.__init__(self)
        self.reducer = reducer
        self.input_transform = input_transform
        self.output_transform = output_transform
        
        # Initialize paths list - reuse ModuleList if provided
        self._paths_list = (
            paths if isinstance(paths, ModuleList) else ModuleList([] if paths is None else paths)
        )
        self.register_module("_paths_list", self._paths_list)

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor | list[Tensor]:
        outputs = [path(x, **kwargs) for path in self._paths_list]
        return outputs if self.reducer is None else self.reducer(outputs)

    def __len__(self) -> int:
        return len(self._paths_list)

    def __iter__(self) -> Iterator[SamplePath]:
        yield from self._paths_list

    def __delitem__(self, key: int) -> None:
        del self._paths_list[key]

    def __getitem__(self, key: int) -> SamplePath:
        return self._paths_list[key]

    def __setitem__(self, key: int, val: SamplePath) -> None:
        self._paths_list[key] = val


class GeneralizedLinearPath(SamplePath):
    r"""A sample path in the form of a generalized linear model."""

    def __init__(
        self,
        feature_map: FeatureMap,
        weight: Parameter | Tensor,
        bias_module: Module | None = None,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ):
        r"""Initializes a GeneralizedLinearPath instance.

        .. code-block:: text

            path(x) = output_transform(bias_module(z) + feature_map(z)^T weight),
            where z = input_transform(x).

        Args:
            feature_map: A map used to featurize the module's inputs.
            weight: A tensor of weights used to combine input features.
            bias_module: An optional module used to define additive offsets.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """
        super().__init__()
        self.feature_map = feature_map
        # Register weight as buffer if not a Parameter
        if not isinstance(weight, Parameter):
            self.register_buffer("weight", weight)
        self.weight = weight
        self.bias_module = bias_module
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        features = self.feature_map(x, **kwargs)
        output = (features @ self.weight.unsqueeze(-1)).squeeze(-1)
        ndim = len(self.feature_map.output_shape)
        if ndim > 1:  # sum over the remaining feature dimensions
            output = einsum(f"...{ascii_letters[:ndim - 1]}->...", output)

        return output if self.bias_module is None else output + self.bias_module(x)
