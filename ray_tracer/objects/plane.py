import math
from typing import override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON
from ray_tracer.objects.abstract_object import AbstractObject, Bounds


class Plane(AbstractObject):
    def __init__(self) -> None:
        super().__init__()
        self.bounds = Bounds(
            Point(-math.inf, 0, -math.inf), Point(math.inf, 0, math.inf)
        )

    @override
    def _normal_func(self, op: Point, i: Intersection | None = None) -> Vector:
        return Vector(0, 1, 0)

    @override
    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        if abs(ray.direction.y) < EPSILON:
            return []

        t = -ray.origin.y / ray.direction.y

        return [Intersection(t, self)]
