from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.patterns.abstract_pattern import AbstractPattern


class TestPattern(AbstractPattern):
    __test__ = False

    def colour_at(self, p: Point) -> Colour:
        return Colour(p.x, p.y, p.z)
