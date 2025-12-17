import math

from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.patterns.abstract_pattern import AbstractPattern


class Rings(AbstractPattern):
    def __init__(
        self, a: Colour | AbstractPattern, b: Colour | AbstractPattern
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def colour_at(self, p: Point) -> Colour | AbstractPattern:
        return self.a if math.floor(math.sqrt(p.x**2 + p.z**2)) % 2 == 0 else self.b
