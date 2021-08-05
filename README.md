# Pshape
[![PyPI version](https://badge.fury.io/py/pshape.svg)](https://badge.fury.io/py/pshape)

Prints NumPy-like arrays' shapes, mins, means, and maxes, as well as the names of the input variable outside the functions' scope.

This comes in very handy when debugging complex programs that manipulate huge `ndarray`s (aka Tensors) where shape (dimensions) and value ranges vary
widely and are hard to inspect.

I got tired of writing tons of `print("my_tensor", my_tensor.shape, my_tensor.min(), my_tensor.max())` over and over, so I made that utility, but then
I got tired of copy/pasting it into every new projects from my Gist of it, so here I finally made it a library that I can pip install everywhere.

## Getting started

```sh
pip3 install pshape
```

## Caveats
Because `pshape` uses the `inpect` built-in Python module, it can't analyse source code from the REPL interpreter, and hence it won't be able to see the variable names if you try something like:
```sh
$ python3
>>> import numpy as np
>>> from pshape import pshape
>>> pshape(np.arange(10))
WARNING: ....
arr1 (10,) 0 4.5 9
```
As you can see that the name used is not `np.arange(10)` as it should be, and defaults instead to `arr1`, `arr2`, etc.

> The only way that the inspect module can display source code is if the code came from a file that it can access. Source typed at an interactive prompt is discarded as soon as it is parsed, there's simply no way for inspect to access it. – @jasonharper

## Usage
```
from pshape import pshape
import numpy as np

# Declare some arrays
cool_arr1 = np.random.rand(123,4,2,1)
cool_arr2 = np.random.rand(123,4,2,2)
cool_arr3 = np.random.rand(123,4,2,3)

pshape(cool_arr1, cool_arr2, cool_arr3)
# this prints:
#
# cool_arr1 (123, 4, 2, 1) 0.0014 0.5091 0.9973
# cool_arr2 (123, 4, 2, 2) 0.0004 0.5058 0.9990
# cool_arr3 (123, 4, 2, 3) 0.0002 0.4975 0.9992
#
# The printing format is {array name} {shape} {min} {mean} {max}

pshape(cool_arr1, np.arange(12).reshape(3,4), cool_arr3)
# this prints:
#
# cool_arr1                  (123, 4, 2, 1) 0.0014 0.5091 0.9973
# np.arange(12).reshape(3,4) (3, 4)         0.0000 5.5000 11.0000
# cool_arr3                  (123, 4, 2, 3) 0.0002 0.4975 0.9992
#
# Note that the statistics are neatly left justified to have a pleasant debugging experience
```

## Development

To install the latest version from Github, run:

```
git clone git@github.com:sam1902/pshape.git pshape
cd pshape
pip3 install -U .
```
