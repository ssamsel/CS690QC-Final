import itertools as it
import re
from collections.abc import Iterable
from functools import wraps
from typing import overload

import numpy as np


def wrap_to_np(func):
    """Decorator that wraps iterable parameters to NDarrays

    Args:
        func (callable): function to decorate/wrap

    Returns:
        callable: decorated/wrapped function
    """

    @wraps(func)
    def outer(*args, **kwargs):
        args = list(args)
        for x in range(len(args)):
            if isinstance(args[x], Iterable) and not isinstance(args[x], str):
                args[x] = np.array(args[x])
        for k in kwargs:
            if isinstance(kwargs[k], Iterable) and not isinstance(args[x], str):
                kwargs[k] = np.array(kwargs[k])
        return func(*args, **kwargs)

    return outer


def all_matchings(N):
    """Calculate all possible parallel BSMs.

    Args:
        N (int): number of peers

    Returns:
        frozenset[frozenset[set[int]]]: set of set of edges
    """

    def _all_matchings(vertices, incompatible, chosen):
        if vertices == incompatible:
            return [chosen]
        result = []
        if len(vertices - incompatible) < 2:
            return [chosen]
        for edge in it.combinations(vertices - incompatible, 2):
            result += _all_matchings(
                vertices, incompatible | frozenset(edge), [*chosen, frozenset(edge)]
            )

        return result

    return frozenset(
        map(
            lambda x: frozenset(x),
            _all_matchings(set(range(N)), set(), list()),
        )
    )


def flatten(*args):
    """Flatten an iterable to 1d

    Returns:
        list[Any]: flattened array
    """
    if len(args) == 0:
        return []
    if isinstance(args[0], Iterable):
        return [*flatten(*args[0]), *flatten(*args[1:])]
    return [args[0], *flatten(*args[1:])]


def flatten2d(arr):
    """Flatten a symmetric 2d iterable to 1d.

    Args:
        arr (Iterable[Iterable[T]]): iterable to flatten

    Returns:
        list[T]: flattened array
    """
    ret = []
    for a in arr:
        ret += a
    return ret


@overload
def convert_to_hz(s: Iterable):
    pass


def convert_to_hz(s: str) -> float:
    """Convert human readable string to numeric frequency in hz

    Args:
        s (str): human readable frequency

    Returns:
        float: numeric value in hz
    """
    if isinstance(s, Iterable) and not isinstance(s, str):
        return [convert_to_hz(a) for a in s]
    if not isinstance(s, str):
        return s
    exponent = {None: 0, "k": 1, "m": 2, "g": 3}
    m = re.search(r"(?P<value>(\d|_|\.)*)\s*(?P<exp>k|m|g)?hz", s.lower())
    return float(m.group("value")) * 1000 ** exponent[m.group("exp")]


def canonical_name(c):
    """Get human readable name of simulation class.

    Args:
        c (type[Simulation]): Simulation class

    Returns:
        str: human readable name
    """
    return re.search(r"(.*\.)*(?P<name>\w*)", str(c)).group("name")


class Queue:
    """Basic lightweight implementation of a FIFO queue."""

    def __init__(self):
        self.q = list()

    def get(self):
        return self.q.pop(0)

    def put(self, x):
        return self.q.append(x)

    def peek(self):
        return self.q[0]

    def qsize(self):
        return len(self.q)

    def empty(self):
        return self.qsize() == 0
