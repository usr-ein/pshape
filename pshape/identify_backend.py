from typing import Any
from enum import Enum
import numpy as np

NDArrayLike = Any

try:
    import torch

    __PYTORCH_ENABLED = True
except ImportError:
    __PYTORCH_ENABLED = False

try:
    import tensorflow as tf

    __TF_ENABLED = True
except ImportError:
    __TF_ENABLED = False


class BackendType(Enum):
    UNKNOWN = 0
    NUMPY = 1
    PYTORCH = 2
    TENSORFLOW = 3


def identify_backend(arr: NDArrayLike) -> BackendType:
    if isinstance(arr, np.ndarray):
        return BackendType.NUMPY
    elif __PYTORCH_ENABLED and isinstance(arr, torch.Tensor):
        return BackendType.PYTORCH
    elif __TF_ENABLED and isinstance(arr, tf.Tensor):
        return BackendType.TENSORFLOW
    else:
        return BackendType.UNKNOWN
