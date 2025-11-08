import inspect
from typing import overload

from .base_tuple import Tuple


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
