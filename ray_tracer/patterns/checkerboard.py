import math

from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.patterns.abstract_pattern import AbstractPattern


class Checkerboard(AbstractPattern):
    def __init__(self, a: Colour, b: Colour) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def colour_at(self, p: Point) -> Colour:
        return (
            self.a
            if (math.floor(p.x) + math.floor(p.y) + math.floor(p.z)) % 2 == 0
            else self.b
        )
