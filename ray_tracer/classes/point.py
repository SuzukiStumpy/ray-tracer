import inspect
from typing import cast, overload

from .base_tuple import Tuple


class Point(Tuple):
    """Defines a point in 3D space"""

    # Provide overloads for static type checkers: external callers should
    # consider only the (x, y, z) signature. The implementation accepts an
    # optional `w` for compatibility with the Tuple factory.
    @overload
    def __init__(self, x: float, y: float, z: float) -> None: ...

    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...

    def __init__(self, x: float, y: float, z: float, w: float | None = None) -> None:
        # Accept an optional fourth parameter so Tuple(...) factory calls
        # (which pass the w component) won't raise a TypeError when the
        # returned object's class is Point. We ignore the passed `w` and
        # always initialize a Point with w == 1.0.
        super().__init__(x, y, z, 1.0)

    def __new__(cls, x: float, y: float, z: float) -> Point:
        return cast(cls, super().__new__(cls, x, y, z, 1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})"

    def __reduce__(self) -> tuple:
        """Support for pickling Point objects for multiprocessing"""
        return (self.__class__, (self.x, self.y, self.z))


# Hide the optional `w` parameter from runtime introspection (inspect.signature
# and help()) by setting a custom Signature that shows only (x, y, z).
Point.__signature__ = inspect.Signature(
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
