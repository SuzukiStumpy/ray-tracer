from typing import cast

from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector


class Ray:
    def __init__(self, origin: Point, direction: Vector) -> None:
        self.origin = origin
        self.direction = direction

    def position(self, time: float) -> Point:
        return cast(Point, self.origin + (time * self.direction))

    def transform(self, m: Matrix) -> Ray:
        p: Point = cast(Point, m * self.origin)
        d: Vector = cast(Vector, m * self.direction)
        return Ray(p, d)
