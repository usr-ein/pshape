from typing import Any
from enum import Enum
import numpy as np

NDArrayLike = Any

try:
    import torch

    PYTORCH_ENABLED = True
except ImportError:
    PYTORCH_ENABLED = False

try:
    import tensorflow as tf

    TF_ENABLED = True
except ImportError:
    TF_ENABLED = False


class BackendType(Enum):
    UNKNOWN = 0
    NUMPY = 1
    PYTORCH = 2
    TENSORFLOW = 3


def identify_backend(arr: NDArrayLike) -> BackendType:
    if isinstance(arr, np.ndarray):
        return BackendType.NUMPY
    elif PYTORCH_ENABLED and isinstance(arr, torch.Tensor):
        return BackendType.PYTORCH
    elif TF_ENABLED and isinstance(arr, tf.Tensor):
        return BackendType.TENSORFLOW
    else:
        return BackendType.UNKNOWN
