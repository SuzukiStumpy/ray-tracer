import math

from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.constants import EPSILON
from ray_tracer.patterns.abstract_pattern import AbstractPattern


class Checkerboard(AbstractPattern):
    def __init__(
        self, a: Colour | AbstractPattern, b: Colour | AbstractPattern
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def colour_at(self, p: Point) -> Colour | AbstractPattern:
        # Adding EPSILON fixes random pixel acne around +/- 0.000
        return (
            self.a
            if (
                math.floor(p.x + EPSILON)
                + math.floor(p.y + EPSILON)
                + math.floor(p.z + EPSILON)
            )
            % 2
            == 0
            else self.b
        )
