#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import cast
from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList

T = TypeVar("T")
TModule = TypeVar("TModule", bound=Module)
TInputTransform = Union[InputTransform, Callable[[Tensor], Tensor]]
TOutputTransform = Union[OutcomeTransform, Callable[[Tensor], Tensor]]


class TransformedModuleMixin:
    r"""Mixin that wraps a module's __call__ method with optional transforms."""
    input_transform: Optional[TInputTransform]
    output_transform: Optional[TOutputTransform]

    def __call__(self, values: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        input_transform = getattr(self, "input_transform", None)
        if input_transform is not None:
            values = (
                input_transform.forward(values)
                if isinstance(input_transform, InputTransform)
                else input_transform(values)
            )

        output = super().__call__(values, *args, **kwargs)
        output_transform = getattr(self, "output_transform", None)
        if output_transform is None:
            return output

        return (
            output_transform.untransform(output)[0]
            if isinstance(output_transform, OutcomeTransform)
            else output_transform(output)
        )


class ModuleDictMixin(ABC, Generic[TModule]):
    def __init__(self, attr_name: str, modules: Optional[Mapping[str, TModule]] = None):
        self.__module_dict_name = attr_name
        self.register_module(self.__module_dict_name, cast(ModuleDict, modules))

    @property
    def __module_dict(self) -> ModuleDict:
        return getattr(self, self.__module_dict_name)

    def items(self) -> Iterable[Tuple[str, TModule]]:
        return self.__module_dict.items()

    def keys(self) -> Iterable[str]:
        return self.__module_dict.keys()

    def values(self) -> Iterable[TModule]:
        return self.__module_dict.values()

    def update(self, modules: Mapping[str, TModule]) -> None:
        self.__module_dict.update(modules)

    def __len__(self) -> int:
        return len(self.__module_dict)

    def __iter__(self) -> Iterator[str]:
        yield from self.__module_dict

    def __delitem__(self, key: str) -> None:
        del self.__module_dict[key]

    def __getitem__(self, key: str) -> TModule:
        return self.__module_dict[key]

    def __setitem__(self, key: str, val: TModule) -> None:
        self.__module_dict[key] = val


class ModuleListMixin(ABC, Generic[TModule]):
    def __init__(self, attr_name: str, modules: Optional[Iterable[TModule]] = None):
        self.__module_list_name = attr_name
        self.register_module(self.__module_list_name, cast(ModuleList, modules))

    @property
    def __module_list(self) -> ModuleList:
        return getattr(self, self.__module_list_name)

    def __len__(self) -> int:
        return len(self.__module_list)

    def __iter__(self) -> Iterator[TModule]:
        yield from self.__module_list

    def __delitem__(self, key: int) -> None:
        del self.__module_list[key]

    def __getitem__(self, key: int) -> TModule:
        return self.__module_list[key]

    def __setitem__(self, key: int, val: TModule) -> None:
        self.__module_list[key] = val