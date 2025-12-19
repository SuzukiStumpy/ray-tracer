"""Various utility functions used elsewhere"""

import cProfile
import pstats
from typing import Callable, TypeVar


def clamp(value: float, lower: float, upper: float) -> float:
    """Restricts a floating point value to between two bounds."""
    return max(min(value, upper), lower)


"""The @profileit() decorator below has been adapted from the code presented here:
   https://www.paulbutcher.space/blog/2018/08/08/profileit
   ... And updated to add in typing so that pyrefly doesn't complain
"""


P = TypeVar("P")
R = TypeVar("R")


def profileit(limit: int = 30) -> Callable[[Callable[..., R]], Callable[..., R]]:
    def inner_profileit(func: Callable[..., R]) -> Callable[..., R]:
        def wrapper(*args: object, **kwargs: object) -> R:
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            prof.create_stats()
            pstats.Stats(prof).sort_stats("cumtime").print_stats(limit)
            return retval

        return wrapper

    return inner_profileit
