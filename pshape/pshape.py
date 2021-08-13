#!/usr/bin/env python3
"""Prints NumPy-like arrays' shapes, mins, means, and maxes, as well as the names of the input variable outside the functions' scope """

from typing import Iterable, List, Optional, Any, Type, Dict
import sys
from copy import deepcopy
import warnings
import inspect, re
import numpy as np

from pshape.exceptions import InteractiveSourceError, ParsingError
from pshape.metrics import ArrayMetric, NameMetric, DEFAULT_METRICS, DeviceMetric
from pshape.identify_backend import (
    NDArrayLike,
    identify_backend,
    BackendType,
)


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


def pshape(
    *arrs: Iterable[NDArrayLike],
    precision: int = 4,
    metrics: List[Type[ArrayMetric]] = None,
    heading: bool = True,
    out=sys.stdout,
) -> None:
    """Prints shapes (and other metrics) of NumPy-like arrays, as well the variable names of the input as passed to this function.
    :param arrs: numpy-like arrays to print
    :param precision: decimal precision to round floats down to
    :param metrics: list of classes which inherit from ArrayMetric and which will be computed for each array
    :param heading: whether to print a heading row displaying each metric's name
    :param out: where to output the things printed, defaults to stdout

    :return None
    """
    if metrics is None:
        metrics = deepcopy(DEFAULT_METRICS)

    if len(arrs) == 0:
        return

    if DeviceMetric in metrics and all(
        identify_backend(arr) != BackendType.PYTORCH for arr in arrs
    ):
        metrics.remove(DeviceMetric)

    try:
        frame = inspect.currentframe()
        previous_frame = inspect.getframeinfo(frame.f_back)
        func_name = inspect.getframeinfo(frame).function

        if previous_frame.code_context is None:
            raise InteractiveSourceError
        call_line = previous_frame.code_context[0].strip()

        if not call_line.startswith(func_name + "("):
            warnings.warn(
                "Please don't call pshape in a compounded function like my_func(pshape(...)), this makes parsing a nightmare.",
                category=UserWarning,
            )

            return
        regex_search = re.search(func_name + "\((.*)\)", call_line)

        if regex_search is None:
            raise ParsingError
        else:
            func_args = regex_search.group(1)
        names = split_cfg_comma(func_args)
    except (InteractiveSourceError, ParsingError):
        # warnings.warn(
        #    "pshape only works in non-REPL environment because source typed at an interactive prompt is discarded as soon as it is parsed, so there's simply no way for inspect to access it. "
        #    "We'll give generic names to arrays instead like arr1, arr2, arr3...",
        #    category=UserWarning,
        # )
        names = []
    except Exception:
        # We should avoid crashing the calling program if possible
        warnings.warn(
            "pshape crashed when parsing argument names ! Continuing without them...",
            category=UserWarning,
        )
        names = []

    try:
        if len(names) > len(arrs):
            names = names[: len(arrs)]
        elif len(names) < len(arrs):
            names.extend([f"arr_{i+1}" for i in range(len(names), len(arrs))])

        metrics_arrs: List[List[ArrayMetric]] = []

        for name, arr in zip(names, arrs):
            # We have to add this metric manually because it's not possible to deduce the "name" (variable name)
            # from just the array object itself
            name_metric = NameMetric(arr)
            name_metric.value = name
            metrics_arrs.append([name_metric] + [M(arr) for M in metrics])

        metrics_max_lengths: Dict[Type[ArrayMetric], int] = {
            M: len(M.name) for M in [NameMetric] + metrics
        }

        for metrics_arr in metrics_arrs:
            for metric in metrics_arr:
                metrics_max_lengths[metric.__class__] = max(
                    metrics_max_lengths[metric.__class__], len(metric)
                )

        if heading:
            print(
                " ".join(
                    map(
                        lambda M: str(M.name).ljust(metrics_max_lengths[M]),
                        [NameMetric] + metrics,
                    )
                ),
                file=out,
            )

        for metrics_arr in metrics_arrs:
            print(
                " ".join(
                    map(
                        lambda m: str(m).ljust(metrics_max_lengths[m.__class__]),
                        metrics_arr,
                    )
                ),
                file=out,
            )
    except Exception as e:
        # We should avoid crashing the calling program if possible
        import traceback as tb

        trace = "".join(tb.format_exception(None, e, e.__traceback__))
        warnings.warn(
            "pshape totally crashed ! This is not fatal, it's just a bug in pshape, please report it at https://github.com/sam1902/pshape/issues/new \n Please make sure to include this: \n"
            + str(e)
            + "\n"
            + trace,
            category=UserWarning,
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
