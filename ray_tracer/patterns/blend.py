from sys import float_repr_style
from typing import cast, override

from PIL.GifImagePlugin import TYPE_CHECKING

from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.point import Point
from ray_tracer.patterns.abstract_pattern import AbstractPattern

if TYPE_CHECKING:
    from ray_tracer.objects.abstract_object import AbstractObject


class Blend(AbstractPattern):
    """Blends two patterns together"""

    def __init__(
        self, a: AbstractPattern, b: AbstractPattern, bias: float = 0.5
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.__dict__["bias"] = bias

    def colour_at(self, p: Point) -> Colour:
        """Intentionally not used, but required by base class"""
        return Colours.BLACK

    @override
    def colour_at_object(self, obj: "AbstractObject", p: Point) -> Colour:
        # Convert point to object space
        object_point = cast(Point, obj.inverse_transform * p)

        # Convert object point to pattern space
        pattern_point = cast(Point, self.inverse_transform * object_point)

        colour_a = self.a.colour_at_object(obj, pattern_point) * (1.0 - self.bias)
        colour_b = self.b.colour_at_object(obj, pattern_point) * self.bias

        computed_colour = colour_a + colour_b

        return computed_colour

    @property
    def bias(self) -> float:
        return self.__dict__["bias"]

    @bias.setter
    def bias(self, v: float) -> None:
        if v < 0.0:
            self.__dict__["bias"] = 0.0
        elif v > 1.0:
            self.__dict__["bias"] = 1.0
