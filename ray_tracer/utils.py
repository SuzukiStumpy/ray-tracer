"""Various utility functions used elsewhere"""

import cProfile
import math
import pstats
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from ray_tracer.objects.abstract_object import Bounds


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


def calculate_camera_distance(
    bounds: "Bounds", field_of_view: float, padding: float = 1.1
) -> float:
    """Calculate the camera distance needed to frame the entire object.

    Args:
        bounds: The object's bounding box
        field_of_view: Camera's field of view in radians
        padding: Multiplier to add extra space (1.1 = 10% padding)

    Returns:
        The distance from the object center to position the camera
    """
    # Calculate object dimensions
    width = bounds.high.x - bounds.low.x
    height = bounds.high.y - bounds.low.y

    # Find the larger dimension
    max_dimension = max(width, height)

    # Calculate required distance: distance = (size/2) / tan(fov/2)
    required_distance = (max_dimension / 2) / math.tan(field_of_view / 2)

    return required_distance * padding
