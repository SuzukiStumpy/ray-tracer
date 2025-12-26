import math
from typing import override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON
from ray_tracer.objects.abstract_object import AbstractObject, Bounds


class Cube(AbstractObject):
    def __init__(self) -> None:
        super().__init__()
        self.bounds = Bounds(Point(-1, -1, -1), Point(1, 1, 1))

    @override
    def _normal_func(self, op: Point) -> Vector:
        maxc = max(abs(op.x), abs(op.y), abs(op.z))

        if maxc == abs(op.x):
            return Vector(op.x, 0, 0)
        elif maxc == abs(op.y):
            return Vector(0, op.y, 0)
        else:
            return Vector(0, 0, op.z)

    @override
    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        (xtmin, xtmax) = self.check_axis(ray.origin.x, ray.direction.x)
        (ytmin, ytmax) = self.check_axis(ray.origin.y, ray.direction.y)
        (ztmin, ztmax) = self.check_axis(ray.origin.z, ray.direction.z)

        tmin = max(xtmin, ytmin, ztmin)
        tmax = min(xtmax, ytmax, ztmax)

        return (
            [] if tmin > tmax else [Intersection(tmin, self), Intersection(tmax, self)]
        )

    def check_axis(self, origin: float, direction: float) -> tuple[float, float]:
        """Helper function to get planar intersects for a specific axis"""
        tmin_numerator = -1 - origin
        tmax_numerator = 1 - origin

        if abs(direction) >= EPSILON:
            tmin = tmin_numerator / direction
            tmax = tmax_numerator / direction
        else:
            tmin = tmin_numerator * math.inf
            tmax = tmax_numerator * math.inf

        return (tmax, tmin) if tmin > tmax else (tmin, tmax)
