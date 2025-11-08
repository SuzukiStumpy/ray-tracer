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
            return False

        return (
            math.isclose(self.x, other.x, abs_tol=EPSILON)
            and math.isclose(self.y, other.y, abs_tol=EPSILON)
            and math.isclose(self.z, other.z, abs_tol=EPSILON)
            and math.isclose(self.w, other.w, abs_tol=EPSILON)
        )

    def __add__(self, other: Tuple) -> object:
        if other.__class__.__name__ not in ["Point", "Vector"]:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__.__name__}'"
                f" and '{other.__class__.__name__}'"
            )

        if self.w + other.w > 1.0:
            raise TypeError("adding two points is unsupported")

        return Tuple(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )

    def __sub__(self, other: Tuple) -> object:
        if other.__class__.__name__ not in ["Point", "Vector"]:
            raise TypeError(
                f"unsupported operand type(s) for -: '{self.__class__.__name__}'"
                f" and '{other.__class__.__name__}'"
            )

        if self.w - other.w < 0.0:
            raise TypeError(
                f"unsupported operation '{self.__class__.__name__}' -"
                f" '{other.__class__.__name__}'"
            )

        return Tuple(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )

    def __neg__(self) -> object:
        return Tuple(
            -self.x,
            -self.y,
            -self.z,
            -self.w if self.__class__.__name__ != "Point" else self.w,
        )

    def __mul__(self, other: float) -> object:
        return Tuple(
            self.x * other,
            self.y * other,
            self.z * other,
            self.w * other,
        )

    def __rmul__(self, other: float) -> object:
        return self * other

    def __truediv__(self, other: float) -> object:
        if math.isclose(other, 0.0, rel_tol=EPSILON):
            raise ZeroDivisionError

        return Tuple(
            self.x / other,
            self.y / other,
            self.z / other,
            self.w / other,
        )
