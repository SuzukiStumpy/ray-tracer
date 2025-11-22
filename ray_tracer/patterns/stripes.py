import math
from typing import TYPE_CHECKING, cast

from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point

if TYPE_CHECKING:
    from ray_tracer.objects.abstract_object import AbstractObject

from ray_tracer.patterns.abstract_pattern import AbstractPattern


class Stripes(AbstractPattern):
    def __init__(self, a: Colour, b: Colour) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def colour_at(self, p: Point) -> Colour:
        return self.a if math.floor(p.x) % 2 == 0 else self.b

    def colour_at_object(self, obj: "AbstractObject", p: Point) -> Colour:
        # Convert point to object space
        object_point = cast(Point, obj.inverse_transform * p)

        # Convert object point to pattern space
        pattern_point = cast(Point, self.inverse_transform * object_point)

        return self.colour_at(pattern_point)
