#!/usr/bin/env python3
"""Prints NumPy-like arrays' shapes, mins, means, and maxes, as well as the names of the input variable outside the functions' scope """

from typing import Iterable
import __main__ as main
import warnings
import inspect, re
import numpy as np


class InteractiveSourceError(Exception):
    """The only way that the inspect module can display source code is if the code came from a file that it can access.
    Source typed at an interactive prompt is discarded as soon as it is parsed, there's simply no way for inspect to access it. – @jasonharper"""

    pass


def split_cfg_comma(s):
    """The simplest and dumbest Context-Free Grammar parser.
    Just cares about commas and parenthesis depth."""
    elems = [""]
    depth = 0

    for c in s:
        if depth == 0 and c == ",":
            elems.append("")
        else:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elems[-1] += c
    elems = list(map(str.lstrip, elems))

    return elems


def pshape(*arrs: Iterable[np.ndarray], precision: int = 4) -> None:
    """Prints NumPy-like arrays' shapes, mins, means, and maxes, as well as the names of the input variable outside the functions' scope.
    Print format is '{name} {shape} {min} {mean} {max}'
    :param arrs: numpy-like arrays to print
    :param precision: decimal precision to round floats down to

    :return None
    """

    if len(arrs) == 0:
        return

    try:
        frame = inspect.currentframe()
        previous_frame = inspect.getframeinfo(frame.f_back)
        func_name = inspect.getframeinfo(frame).function

        if previous_frame.code_context is None:
            raise InteractiveSourceError
        call_line = previous_frame.code_context[0].strip()

        if not call_line.startswith("pshape("):
            warnings.warn(
                "Please don't call pshape in a compounded function like my_func(pshape(...)), this makes parsing a nightmare.",
                category=UserWarning,
            )

            return
        args_splitted = split_cfg_comma(
            re.search(func_name + "\((.*)\)", call_line).group(1)
        )
    except InteractiveSourceError:
        warnings.warn(
            "pshape only works in non-REPL environment because source typed at an interactive prompt is discarded as soon as it is parsed, so there's simply no way for inspect to access it. "
            "We'll give generic names to arrays instead like arr1, arr2, arr3...",
            category=UserWarning,
        )
        args_splitted = []

    if len(args_splitted) > len(arrs):
        args_splitted = args_splitted[: len(arrs)]
    elif len(args_splitted) < len(arrs):
        args_splitted.extend(
            [f"arr_{i+1}" for i in range(len(args_splitted), len(arrs))]
        )

    # Used for left justifying
    max_sizes = {
        "name": max(map(str.__len__, args_splitted)),
        "shape": -1,
        "min": -1,
        "max": -1,
        "mean": -1,
    }

    try_get_attr = (
        lambda arr, attr: getattr(arr, attr)()
        if hasattr(arr, attr) and callable(getattr(arr, attr))
        else "?"
    )

    # Computes the string sizes and finds the max size of each kind to see how to left justify that column correctly

    for name, arr in zip(args_splitted, arrs):
        max_sizes["shape"] = max(
            max_sizes["shape"],
            len(str(tuple(arr.shape)) if hasattr(arr, "shape") else "?"),
        )
        max_sizes["min"] = max(
            max_sizes["min"], len(f"{try_get_attr(arr, 'min'):.{precision}f}")
        )
        max_sizes["max"] = max(
            max_sizes["max"], len(f"{try_get_attr(arr, 'max'):.{precision}f}")
        )
        max_sizes["mean"] = max(
            max_sizes["mean"], len(f"{try_get_attr(arr, 'mean'):.{precision}f}")
        )

    for name, arr in zip(args_splitted, arrs):
        shape = str(tuple(arr.shape)) if hasattr(arr, "shape") else "?"
        min_v = f"{try_get_attr(arr, 'min'):.{precision}f}"
        max_v = f"{try_get_attr(arr, 'max'):.{precision}f}"
        mean_v = f"{try_get_attr(arr, 'mean'):.{precision}f}"
        print(
            f"{name.ljust(max_sizes['name'])} {shape.ljust(max_sizes['shape'])} {min_v.ljust(max_sizes['min'])} {mean_v.ljust(max_sizes['mean'])} {max_v.ljust(max_sizes['max'])}"
        )


if __name__ == "__main__":
    import numpy as np

    cool_arr1 = np.random.rand(123, 4, 2, 1)
    cool_arr2 = np.random.rand(123, 4, 2, 2)
    cool_arr3 = np.random.rand(123, 4, 2, 3)
    pshape(cool_arr1, cool_arr2, cool_arr3)
    # this prints:
    #
    # cool_arr1 (123, 4, 2, 1) 0.0014 0.5091 0.9973
    # cool_arr2 (123, 4, 2, 2) 0.0004 0.5058 0.9990
    # cool_arr3 (123, 4, 2, 3) 0.0002 0.4975 0.9992

    pshape(cool_arr1, np.arange(12).reshape(3, 4), cool_arr3)
    # this prints:
    #
    # cool_arr1                  (123, 4, 2, 1) 0.0014 0.5091 0.9973
    # np.arange(12).reshape(3,4) (3, 4)         0.0000 5.5000 11.0000
    # cool_arr3                  (123, 4, 2, 3) 0.0002 0.4975 0.9992
    #
    # Note that the statistics are neatly left justified to have a pleasant debugging experience
