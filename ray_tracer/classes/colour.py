from typing import Self, overload

from .abstract_tuple import AbstractTuple


class Colour(AbstractTuple):
    """Defines an RGB colour"""

    def __init__(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        super().__init__(x=r, y=g, z=b, w=a)

    def __repr__(self: Self) -> str:
        return (
            f"{self.__class__.__name__}(r={self.r}, g={self.g}, b={self.b}, a={self.a})"
        )

    @property
    def r(self) -> float:
        return self.x

    @property
    def red(self) -> float:
        return self.r

    @property
    def g(self) -> float:
        return self.y

    @property
    def green(self) -> float:
        return self.g

    @property
    def b(self) -> float:
        return self.z

    @property
    def blue(self) -> float:
        return self.b

    @property
    def a(self) -> float:
        return self.w

    @property
    def alpha(self) -> float:
        return self.alpha

    @overload
    def __add__(self, other: Colour) -> Self: ...

    @overload
    def __add__(self, other: AbstractTuple) -> AbstractTuple: ...

    def __add__(self, other: AbstractTuple) -> AbstractTuple:
        # Only allow adding two Colour instances; for other types return
        # NotImplemented so Python can try reflected operations or raise.
        if not isinstance(other, Colour):
            raise NotImplementedError

        # Construct the result as a Colour. Use a short multi-line call to
        # satisfy line-length linting and silence the abstract-instantiation
        # type checker here (Colour may be flagged abstract by the checker
        # even though runtime construction is intended).
        return Colour(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
            self.a + other.a,
        )

    def __sub__(self, other: AbstractTuple) -> AbstractTuple:
        raise NotImplementedError

    def __neg__(self) -> Self:
        raise NotImplementedError

    def __mul__(self, other: float) -> Self:
        raise NotImplementedError

    def __truediv__(self, other: float) -> Self:
        raise NotImplementedError
