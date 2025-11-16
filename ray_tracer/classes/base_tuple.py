"""Defines the base tuple class from which points and vectors are derived"""

import math
from typing import Self, cast, overload

from ..constants import EPSILON
from .abstract_tuple import AbstractTuple


class Tuple(AbstractTuple):
    """Base tuple class used by points and vectors"""

    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __new__(cls, x: float, y: float, z: float, w: float | None = None) -> Tuple:
        # Import Point/Vector lazily to avoid circular imports during module
        # initialization (point.py and vector.py import this module).
        from .point import Point
        from .vector import Vector

        # If the caller omitted the `w` value (common when subclass
        # constructors call into Tuple), return an instance of the requested
        # runtime class directly.
        if w is None:
            return object.__new__(cls)  # type: ignore

        if math.isclose(w, 1.0, rel_tol=EPSILON):
            new_obj = object.__new__(Point)
        elif math.isclose(w, 0.0, rel_tol=EPSILON):
            new_obj = object.__new__(Vector)
        else:
            new_obj = object.__new__(Tuple)

        return new_obj  # type: ignore

    def __repr__(self: Self) -> str:
        return super(self).__repr__()

    def __add__(self, other: AbstractTuple) -> AbstractTuple:
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

    def __sub__(self, other: AbstractTuple) -> AbstractTuple:
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

    def __neg__(self: Self) -> Self:
        # Preserve the runtime subclass (Point/Vector/Tuple) when negating.
        # Point/Vector constructors accept only (x, y, z) at runtime, so
        # call the runtime class with three args for those types and use
        # Tuple(...) for any plain Tuple instances.
        from .point import Point
        from .vector import Vector

        if isinstance(self, (Point, Vector)):
            # Call the concrete constructors explicitly so the type checker
            # sees a matching signature.
            if isinstance(self, Point):
                return Point(-self.x, -self.y, -self.z)
            return Vector(-self.x, -self.y, -self.z)

        # Explicitly cast to Self for the type checker; at runtime this
        # constructs a plain Tuple which is compatible with Self when the
        # actual instance is Tuple.
        return cast(
            Self,
            Tuple(
                -self.x,
                -self.y,
                -self.z,
                -self.w,
            ),
        )

    @overload
    def __mul__(self: Self, other: float) -> Self: ...

    @overload
    def __mul__(self: Self, other: float | AbstractTuple) -> Self: ...

    def __mul__(self: Self, other: float | AbstractTuple) -> Self:
        # Preserve the runtime subclass when multiplying by a scalar.
        from .point import Point
        from .vector import Vector

        # It's only legal to multiply points and vectors by scalars in this method
        if not isinstance(other, float):
            raise NotImplementedError

        if isinstance(self, (Point, Vector)):
            if isinstance(self, Point):
                return Point(self.x * other, self.y * other, self.z * other)
            return Vector(self.x * other, self.y * other, self.z * other)

        return cast(
            Self,
            Tuple(
                self.x * other,
                self.y * other,
                self.z * other,
                self.w * other,
            ),
        )

    def __rmul__(self: Self, other: float) -> Self:
        return self * other

    def __truediv__(self: Self, other: float) -> Self:
        if math.isclose(other, 0.0, rel_tol=EPSILON):
            raise ZeroDivisionError
        from .point import Point
        from .vector import Vector

        if isinstance(self, (Point, Vector)):
            if isinstance(self, Point):
                return Point(self.x / other, self.y / other, self.z / other)
            return Vector(self.x / other, self.y / other, self.z / other)

        return cast(
            Self,
            Tuple(
                self.x / other,
                self.y / other,
                self.z / other,
                self.w / other,
            ),
        )
