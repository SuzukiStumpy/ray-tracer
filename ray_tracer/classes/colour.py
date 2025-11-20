from typing import Self, overload

from ray_tracer.utils import clamp

from .abstract_tuple import AbstractTuple


class Colour(AbstractTuple):
    """Defines an RGB colour"""

    def __init__(self, r: float, g: float, b: float) -> None:
        super().__init__(x=r, y=g, z=b, w=1.0)

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}(r={self.r}, g={self.g}, b={self.b})"

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

    @overload
    def __add__(self, other: Colour) -> Self: ...

    @overload
    def __add__(self, other: AbstractTuple) -> AbstractTuple: ...

    def __add__(self, other: AbstractTuple) -> AbstractTuple:
        if not isinstance(other, Colour):
            raise NotImplementedError

        return Colour(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
        )

    def __sub__(self, other: AbstractTuple) -> AbstractTuple:
        if not isinstance(other, Colour):
            raise NotImplementedError

        return Colour(
            self.r - other.r,
            self.g - other.g,
            self.b - other.b,
        )

    def __neg__(self) -> Self:
        raise NotImplementedError

    @overload
    def __mul__(self, other: float) -> Self: ...

    @overload
    def __mul__(self, other: Colour) -> Self: ...

    @overload
    def __mul__(self, other: AbstractTuple | float) -> Self: ...

    def __mul__(self, other: AbstractTuple | float) -> Self:
        if not isinstance(other, (Colour, float, int)):
            raise NotImplementedError(
                "You can multiply colours by scalars or other Colours only"
            )

        return (
            Colour(
                self.r * other.r,
                self.g * other.g,
                self.b * other.b,
            )
            if isinstance(other, Colour)
            else Colour(
                self.r * other,
                self.g * other,
                self.b * other,
            )
        )

    def __truediv__(self, other: float) -> Self:
        raise NotImplementedError

    def clamp(self) -> Self:
        return Colour(
            clamp(self.x, 0, 1),
            clamp(self.y, 0, 1),
            clamp(self.z, 0, 1),
        )


class Colours:
    """Define some basic named colours"""

    BLACK = Colour(0.0, 0.0, 0.0)
    RED = Colour(1.0, 0.0, 0.0)
    GREEN = Colour(0.0, 1.0, 0.0)
    BLUE = Colour(0.0, 0.0, 1.0)
    WHITE = Colour(1.0, 1.0, 1.0)

    @classmethod
    def values(cls) -> list[Colour]:
        return [
            v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
        ]
