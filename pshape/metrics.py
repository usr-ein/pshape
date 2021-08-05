from typing import Iterable, List, Optional, Any, Type, Dict
from pshape.identify_backend import NDArrayLike, identify_backend, BackendType


class ArrayMetric:
    name: str
    _default_value: str = "?"
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


class DeviceMetric(ArrayMetric):
    """Specific to PyTorch"""

    name = "device"


class DtypeMetric(ArrayMetric):
    name = "dtype"


class NumericMetric(ArrayMetric):
    name: str
    precision: int = 4

    def __str__(self) -> str:
        if self.value is None or self.precision is None:
            import pdb

            pdb.set_trace()

        return f"{self.value:.{self.precision}f}"


class CallableMetric(ArrayMetric):
    def _get_value(self):
        return super()._get_value()()


class NumericCallableMetric(NumericMetric, CallableMetric):
    def _get_value(self):
        backend = identify_backend(self.arr)

        if backend is BackendType.TENSORFLOW:
            import tensorflow as tf

            return getattr(tf.math, "reduce_" + name)(self.arr).numpy()
        else:
            return super()._get_value()


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
DEFAULT_METRICS = [
    ShapeMetric,
    DeviceMetric,
    DtypeMetric,
    MinMetric,
    MeanMetric,
    MaxMetric,
]
