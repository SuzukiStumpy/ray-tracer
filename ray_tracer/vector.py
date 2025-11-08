import inspect
import math
from typing import cast, overload

from .base_tuple import Tuple
from .constants import EPSILON


class Vector(Tuple):
    """Defines a 3D vector"""

    # Overloads for static type checkers: external callers should see
    # only (x, y, z). Implementation accepts an optional `w`.
    @overload
    def __init__(self, x: float, y: float, z: float) -> None: ...

    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...

    def __init__(self, x: float, y: float, z: float, w: float | None = None) -> None:
        # Accept an optional fourth parameter for compatibility with the
        # Tuple(...) factory. Ignore the provided `w` and always initialize
        # vectors with w == 0.0.
        super().__init__(x, y, z, 0.0)

    def __new__(cls, x: float, y: float, z: float) -> Vector:
        return cast(cls, super().__new__(cls, x, y, z, 0))

    def __abs__(self) -> float:
        """Returns the magnitude of the vector"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def normalize(self) -> Vector:
        magnitude = abs(self)

        if math.isclose(magnitude, 0.0, rel_tol=EPSILON):
            raise ZeroDivisionError("vector has magnitude of zero")

        return Vector(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude,
        )


# Hide the optional `w` parameter from runtime introspection (inspect.signature
# and help()) by setting a custom Signature that shows only (x, y, z).
Vector.__signature__ = inspect.Signature(
    parameters=[
        inspect.Parameter(
            "self",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ),
        inspect.Parameter(
            "x",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=float,
        ),
        inspect.Parameter(
            "y",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=float,
        ),
        inspect.Parameter(
            "z",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=float,
        ),
    ],
    return_annotation=None,
)
