#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
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

# Generic type variable for module types
T = TypeVar("T")  # generic type variable
TModule = TypeVar("TModule", bound=Module)  # must be a Module subclass
TInputTransform = Union[InputTransform, Callable[[Tensor], Tensor]]
TOutputTransform = Union[OutcomeTransform, Callable[[Tensor], Tensor]]


class TransformedModuleMixin(Module):
    r"""Mixin that wraps a module's __call__ method with optional transforms.

    This mixin provides functionality to transform inputs before processing and outputs
    after processing. It inherits from Module to ensure proper PyTorch module behavior
    and requires subclasses to implement the forward method.

    Attributes:
        input_transform: Optional transform applied to input values before forward pass
        output_transform: Optional transform applied to output values after forward pass
    """

    input_transform: Optional[TInputTransform]
    output_transform: Optional[TOutputTransform]

    def __init__(self):
        # Initialize Module first to ensure proper PyTorch behavior
        super().__init__()
        self.input_transform = None
        self.output_transform = None

    def __call__(self, values: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # Apply input transform if present
        input_transform = getattr(self, "input_transform", None)
        if input_transform is not None:
            values = (
                input_transform.forward(values)
                if isinstance(input_transform, InputTransform)
                else input_transform(values)
            )

        # Call forward instead of super().__call__ since we're implementing the interface
        output = self.forward(values, *args, **kwargs)

        # Apply output transform if present
        output_transform = getattr(self, "output_transform", None)
        if output_transform is None:
            return output

        return (
            output_transform.untransform(output)[0]
            if isinstance(output_transform, OutcomeTransform)
            else output_transform(output)
        )

    @abstractmethod
    def forward(self, values: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Abstract method that must be implemented by subclasses.

        This enforces the PyTorch pattern of implementing computation in forward().
        """
        pass


class ModuleDictMixin(ABC, Generic[TModule]):
    r"""Mixin that provides dictionary-like access to a ModuleDict.

    This mixin allows a class to behave like a dictionary of modules while ensuring
    proper PyTorch module registration and parameter tracking. It uses a unique name
    for the underlying ModuleDict to avoid attribute conflicts.

    Type Args:
        TModule: The type of modules stored in the dictionary (must be Module subclass)
    """

    def __init__(self, attr_name: str, modules: Optional[Mapping[str, TModule]] = None):
        r"""Initialize ModuleDictMixin.

        Args:
            attr_name: Base name for the ModuleDict attribute
            modules: Optional initial mapping of module names to modules
        """
        # Use a unique name to avoid conflicts with existing attributes
        self.__module_dict_name = f"_{attr_name}_dict"
        # Create and register the ModuleDict
        self.register_module(
            self.__module_dict_name, ModuleDict({} if modules is None else modules)
        )

    @property
    def __module_dict(self) -> ModuleDict:
        """Access the underlying ModuleDict using the unique name."""
        return getattr(self, self.__module_dict_name)

    # Dictionary interface methods
    def items(self) -> Iterable[Tuple[str, TModule]]:
        """Return (key, value) pairs of the dictionary."""
        return self.__module_dict.items()

    def keys(self) -> Iterable[str]:
        """Return keys of the dictionary."""
        return self.__module_dict.keys()

    def values(self) -> Iterable[TModule]:
        """Return values of the dictionary."""
        return self.__module_dict.values()

    def update(self, modules: Mapping[str, TModule]) -> None:
        """Update the dictionary with new modules."""
        self.__module_dict.update(modules)

    def __len__(self) -> int:
        """Return number of modules in the dictionary."""
        return len(self.__module_dict)

    def __iter__(self) -> Iterator[str]:
        """Iterate over module names."""
        yield from self.__module_dict

    def __delitem__(self, key: str) -> None:
        """Delete a module by name."""
        del self.__module_dict[key]

    def __getitem__(self, key: str) -> TModule:
        """Get a module by name."""
        return self.__module_dict[key]

    def __setitem__(self, key: str, val: TModule) -> None:
        """Set a module by name."""
        self.__module_dict[key] = val


class ModuleListMixin(ABC, Generic[TModule]):
    r"""Mixin that provides list-like access to a ModuleList.

    This mixin allows a class to behave like a list of modules while ensuring
    proper PyTorch module registration and parameter tracking. It uses a unique name
    for the underlying ModuleList to avoid attribute conflicts.

    Type Args:
        TModule: The type of modules stored in the list (must be Module subclass)
    """

    def __init__(self, attr_name: str, modules: Optional[Iterable[TModule]] = None):
        r"""Initialize ModuleListMixin.

        Args:
            attr_name: Base name for the ModuleList attribute
            modules: Optional initial iterable of modules
        """
        # Use a unique name to avoid conflicts with existing attributes
        self.__module_list_name = f"_{attr_name}_list"
        # Create and register the ModuleList
        self.register_module(
            self.__module_list_name, ModuleList([] if modules is None else modules)
        )

    @property
    def __module_list(self) -> ModuleList:
        """Access the underlying ModuleList using the unique name."""
        return getattr(self, self.__module_list_name)

    # List interface methods
    def __len__(self) -> int:
        """Return number of modules in the list."""
        return len(self.__module_list)

    def __iter__(self) -> Iterator[TModule]:
        """Iterate over modules."""
        yield from self.__module_list

    def __delitem__(self, key: int) -> None:
        """Delete a module by index."""
        del self.__module_list[key]

    def __getitem__(self, key: int) -> TModule:
        """Get a module by index."""
        return self.__module_list[key]

    def __setitem__(self, key: int, val: TModule) -> None:
        """Set a module by index."""
        self.__module_list[key] = val
