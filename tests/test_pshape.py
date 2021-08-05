#!/usr/bin/env python3
"""Module doc"""
import unittest
from pshape import pshape
import numpy as np

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


class TestPshape(unittest.TestCase):
    def test_doesnt_crash(self):
        pshape(np.eye(4), np.arange(10).reshape(5, 2, 1), heading=True)

    @unittest.skipUnless(PYTORCH_ENABLED, "torch not installed")
    def test_pytorch(self):
        pshape(torch.eye(4), torch.arange(10).view(5, 2, 1), heading=True)

    @unittest.skipUnless(TF_ENABLED, "tensorflow not installed")
    def test_tensorflow(self):
        pshape(tf.eye(4), tf.reshape(tf.range(10), (5, 2, 1)), heading=True)

    @unittest.skipUnless(PYTORCH_ENABLED, "torch not installed")
    def test_pytorch_np(self):
        pshape(np.eye(4), torch.arange(10).view(5, 2, 1), heading=True)

    @unittest.skipUnless(TF_ENABLED, "tensorflow not installed")
    def test_tensorflow_np(self):
        pshape(np.eye(4), tf.reshape(tf.range(10), (5, 2, 1)), heading=True)

    @unittest.skipUnless(
        TF_ENABLED and PYTORCH_ENABLED, "pytorch and tensorflow need to be installed"
    )
    def test_tensorflow_pytorch(self):
        pshape(
            torch.arange(10).view(5, 2, 1),
            tf.reshape(tf.range(10), (5, 2, 1)),
            heading=True,
        )

    @unittest.skipUnless(
        TF_ENABLED and PYTORCH_ENABLED, "pytorch and tensorflow need to be installed"
    )
    def test_tensorflow_pytorch_np(self):
        pshape(
            np.arange(10).reshape(5, 2, 1),
            torch.arange(10).view(5, 2, 1),
            tf.reshape(tf.range(10), (5, 2, 1)),
            heading=True,
        )


if __name__ == "__main__":
    unittest.main()
