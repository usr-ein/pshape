import numpy as np
from typing import Iterable, List, Optional, Any, Type, Dict
from pshape.identify_backend import (
    NDArrayLike,
    identify_backend,
    BackendType,
    PYTORCH_ENABLED,
    TF_ENABLED,
)


class ArrayMetric:
    name: str
    _default_value: Any = "?"
    _value: Optional[str] = None

    def __init__(self, arr: NDArrayLike):
        self.arr = arr
        self._value = None

    def is_compatible(self) -> bool:
        return hasattr(self.arr, self.name)

    def __len__(self) -> int:
        return len(str(self))

    def __str__(self) -> str:
        return str(self.value)

    @property
    def value(self):
        if self._value is None:
            if self.is_compatible():
                self._value = self._get_value()
            else:
                self._value = self._default_value

        return self._value

    def _get_value(self):
        return getattr(self.arr, self.name)


class NameMetric(ArrayMetric):
    name = "name"

    @property
    def value(self):
        return super().value

    @value.setter
    def value(self, val):
        self._value = val

    def is_compatible(self):
        # We don't want to have _get_value kick in

        return False


class ShapeMetric(ArrayMetric):
    name = "shape"

    def _get_value(self):
        return tuple(super()._get_value())


class DeviceMetric(ArrayMetric):
    """Specific to PyTorch"""

    name = "device"
    _default_value = "N/A"


class DtypeMetric(ArrayMetric):
    name = "dtype"

    def __str__(self) -> str:
        if identify_backend(self.arr) is BackendType.TENSORFLOW:
            return repr(self.value)
        else:
            return str(self.value)


class NumericMetric(ArrayMetric):
    name: str
    precision: int = 4
    _default_value = np.nan

    def __str__(self) -> str:
        return f"{self.value:.{self.precision}f}"


class CallableMetric(ArrayMetric):
    def _get_value(self):
        return super()._get_value()()


class NumericCallableMetric(NumericMetric, CallableMetric):
    def _get_value(self):
        backend = identify_backend(self.arr)

        if backend is BackendType.TENSORFLOW:
            import tensorflow as tf

            return getattr(tf.math, "reduce_" + self.name)(self.arr).numpy()
        else:
            return super()._get_value()

    def is_compatible(self):
        return super().is_compatible() or (
            identify_backend(self.arr) is BackendType.TENSORFLOW
        )


class MinMetric(NumericCallableMetric):
    name = "min"


class MaxMetric(NumericCallableMetric):
    name = "max"


class MeanMetric(NumericCallableMetric):
    name = "mean"

    def _get_value(self):
        backend = identify_backend(self.arr)

        if backend is BackendType.PYTORCH:
            import torch

            # PyTorch can't compute mean of integer-like types
            self.arr = self.arr.to(torch.float)

        return super()._get_value()


# NameMetric is the default first metric and can't be used elsewhere
DEFAULT_METRICS = [ShapeMetric]

if PYTORCH_ENABLED:
    DEFAULT_METRICS += [DeviceMetric]
DEFAULT_METRICS += [DtypeMetric]
DEFAULT_METRICS += [MinMetric]
DEFAULT_METRICS += [MeanMetric]
DEFAULT_METRICS += [MaxMetric]
