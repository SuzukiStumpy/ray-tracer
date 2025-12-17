import math

from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.constants import EPSILON
from ray_tracer.patterns.abstract_pattern import AbstractPattern


class Stripes(AbstractPattern):
    def __init__(
        self, a: Colour | AbstractPattern, b: Colour | AbstractPattern
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def colour_at(self, p: Point) -> Colour | AbstractPattern:
        return self.a if math.floor(p.x + EPSILON) % 2 == 0 else self.b
