from typing import cast

from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.patterns.abstract_pattern import AbstractPattern


class Gradient(AbstractPattern):
    def __init__(self, a: Colour, b: Colour) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def colour_at(self, p: Point) -> Colour:
        distance = cast(Colour, self.b - self.a)
        # Map pattern space from [-1, 1] to [0, 1] so pattern doesn't repeat
        multiplier = (p.x + 1) / 2

        return self.a + distance * multiplier
