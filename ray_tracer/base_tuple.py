"""Defines the base tuple class from which points and vectors are derived"""

import math

from .constants import EPSILON


class Tuple:
    """Base tuple class used by points and vectors"""

    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __new__(cls, x: float, y: float, z: float, w: float) -> object:
        # Import Point/Vector lazily to avoid circular imports during module
        # initialization (point.py and vector.py import this module).
        from .point import Point
        from .vector import Vector

        if math.isclose(w, 1.0, rel_tol=EPSILON):
            new_obj = object.__new__(Point)
        elif math.isclose(w, 0.0, rel_tol=EPSILON):
            new_obj = object.__new__(Vector)
        else:
            new_obj = object.__new__(Tuple)

        return new_obj

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z}"
        s += f", w={self.w})" if type(self) is Tuple else ")"
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return (
            self.x == other.x
            and self.y == other.y
            and self.z == other.z
            and self.w == other.w
        )
