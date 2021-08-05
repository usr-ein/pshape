# Pshape
[![PyPI version](https://badge.fury.io/py/pshape.svg)](https://badge.fury.io/py/pshape)

Prints shapes of NumPy-like arrays, as well as many more useful debugging metrics, along with the names of the input variable outside the functions' scope.

This comes in very handy when debugging complex programs that manipulate huge `ndarray`s (aka Tensors) where shape (dimensions) and value ranges vary
widely and are hard to inspect.

I got tired of writing tons of `print("my_tensor", my_tensor.shape, my_tensor.min(), my_tensor.max())` over and over, so I made that utility, but then
I got tired of copy/pasting it into every new projects from my Gist of it, so here I finally made it a library that I can pip install everywhere.

## Getting started

```sh
pip3 install pshape
```

`pshape` works with Numpy, PyTorch and Tensorflow, though TF is the least stable of the three since I don't test it extensively, but feel free to contribute ! You can mix and match `pshape` calls containing Numpy, Pytorch and Tensorflow arrays.

I try to make pshape as "safe" as possible, so that it never crashes your code, because it's not crucial that it work as a tool and it'd be silly to risk the whole program execution over a stupid bug in a debugging tool.

## Caveats
Because `pshape` uses the `inpect` built-in Python module, it can't analyse source code from the REPL interpreter, and hence it won't be able to see the variable names if you try something like:
```sh
$ python3
>>> import numpy as np
>>> from pshape import pshape
>>> pshape(np.arange(10))
arr_1 (10,) 0 4.5 9
```
As you can see the name used is not `np.arange(10)` as it should be, and defaults instead to `arr_1`, `arr_2`, etc.

> The only way that the inspect module can display source code is if the code came from a file that it can access. Source typed at an interactive prompt is discarded as soon as it is parsed, there's simply no way for inspect to access it. – @jasonharper

## Usage
```python3
>>> from pshape import pshape
>>> import numpy as np
>>> import torch
>>> 
>>> pshape(np.arange(10).reshape(5,2,1), heading=True)

name                         shape     dtype min    mean   max
np.arange(10).reshape(5,2,1) (5, 2, 1) int64 0.0000 4.5000 9.0000

>>> pshape(np.eye(4), np.arange(10).reshape(5,2,1), heading=True)

name                         shape     dtype   min    mean   max
np.eye(4)                    (4, 4)    float64 0.0000 0.2500 1.0000
np.arange(10).reshape(5,2,1) (5, 2, 1) int64   0.0000 4.5000 9.0000

>>> cool_arr1 = np.random.rand(123,4,2,1)
>>> cool_arr2 = np.random.rand(123,4,2,2)
>>> cool_arr3 = np.random.rand(123,4,2,3)
>>> pshape(cool_arr1, cool_arr2, cool_arr3)

cool_arr1 (123, 4, 2, 1) float64 0.0004 0.4961 1.0000
cool_arr2 (123, 4, 2, 2) float64 0.0006 0.4947 1.0000
cool_arr3 (123, 4, 2, 3) float64 0.0017 0.4997 0.9996

>>> pshape(cool_arr1, np.arange(12).reshape(3,4), cool_arr3)

cool_arr1                  (123, 4, 2, 1) float64 0.0004 0.4961 1.0000
np.arange(12).reshape(3,4) (3, 4)         int64   0.0000 5.5000 11.0000
cool_arr3                  (123, 4, 2, 3) float64 0.0017 0.4997 0.9996

>>> pshape(torch.arange(12).view(3,4), np.arange(12).reshape(3,4), cool_arr3)

name                       shape          device dtype       min    mean   max
torch.arange(12).view(3,4) (3, 4)         cpu    torch.int64 0.0000 5.5000 11.0000
np.arange(12).reshape(3,4) (3, 4)         N/A    int64       0.0000 5.5000 11.0000
cool_arr3                  (123, 4, 2, 3) N/A    float64     0.0017 0.4997 0.9996
```
To get a similar display, you can run the `demo_pshape.py` script at the root of this repo.

## Development

To install the latest version from Github, run:

```
git clone git@github.com:sam1902/pshape.git pshape
cd pshape
pip3 install -U .
```
