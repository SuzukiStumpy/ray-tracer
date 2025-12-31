import math
from typing import override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON
from ray_tracer.objects.abstract_object import AbstractObject, Bounds


class Cylinder(AbstractObject):
    @override
    def __init__(
        self,
        minimum: float = -math.inf,
        maximum: float = math.inf,
        closed: bool = False,
    ) -> None:
        super().__init__()

        self.__dict__["min"] = minimum
        self.__dict__["max"] = maximum
        self.closed = closed

        self.bounds = Bounds(Point(-1, minimum, -1), Point(1, maximum, 1))

    @property
    def min(self) -> float:
        return self.__dict__["min"]

    @min.setter
    def min(self, m: float) -> None:
        self.__dict__["min"] = m if m < self.max else self.max - EPSILON

    @property
    def max(self) -> float:
        return self.__dict__["max"]

    @max.setter
    def max(self, m: float) -> None:
        self.__dict__["max"] = m if m > self.min else self.min + EPSILON

    @override
    def _normal_func(self, op: Point, i: Intersection | None = None) -> Vector:
        # compute the square of the distance from the y axis
        distance = op.x**2 + op.z**2

        if distance < 1 and op.y >= self.max - EPSILON:
            return Vector(0, 1, 0)
        elif distance < 1 and op.y <= self.min + EPSILON:
            return Vector(0, -1, 0)
        else:
            return Vector(op.x, 0, op.z)

    @override
    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        a = ray.direction.x**2 + ray.direction.z**2

        xs = []

        # ray is parallel to y axis, so skip the sidewall intersection logic
        if not math.isclose(a, 0.0, abs_tol=EPSILON):
            b = 2 * ray.origin.x * ray.direction.x + 2 * ray.origin.z * ray.direction.z
            c = ray.origin.x**2 + ray.origin.z**2 - 1
            discriminant = b**2 - 4 * a * c

            # ray does not intersect the cylinder
            if discriminant < 0:
                return []

            discriminant_root = math.sqrt(discriminant)
            two_a = 2 * a

            t0 = (-b - discriminant_root) / two_a
            t1 = (-b + discriminant_root) / two_a

            if t0 > t1:
                t0, t1 = t1, t0

            y0 = ray.origin.y + t0 * ray.direction.y

            if self.min < y0 and y0 < self.max:
                xs.append(Intersection(t0, self))

            y1 = ray.origin.y + t1 * ray.direction.y

            if self.min < y1 and y1 < self.max:
                xs.append(Intersection(t1, self))

        # test for intersection with end caps
        if self.closed:
            xs.extend(self.intersect_caps(ray))

        return xs

    # helper function to reduce duplication
    # checks to see if the intersection at 't' is within a radius of
    # 1 (the radius of the untransformed cylinder) from the y axis
    def _check_cap(self, ray: Ray, t: float) -> bool:
        x = ray.origin.x + t * ray.direction.x
        z = ray.origin.z + t * ray.direction.z

        return (x**2 + z**2) <= 1

    def intersect_caps(self, ray: Ray) -> list[Intersection]:
        # Caps only matter if the cylinder is closed and might possibly be intersected
        # by the ray
        if not self.closed or math.isclose(ray.direction.y, 0, abs_tol=EPSILON):
            return []

        xs = []

        # check for an intersection with the lower end cap
        t = (self.min - ray.origin.y) / ray.direction.y
        if self._check_cap(ray, t):
            xs.append(Intersection(t, self))

        # check for an intersection with the upper end cap
        t = (self.max - ray.origin.y) / ray.direction.y
        if self._check_cap(ray, t):
            xs.append(Intersection(t, self))

        return xs
