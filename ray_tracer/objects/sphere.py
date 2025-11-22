import math
from typing import cast, override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject


class Sphere(AbstractObject):
    def __init__(self) -> None:
        super().__init__()

    @override
    def _normal_func(self, op: Point) -> Vector:
        """op is the point in object space"""
        return cast(Vector, op - Point(0, 0, 0))

    @override
    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        # the vector from the object's centre to the ray origin (the object
        # is always centered on the world origin)
        object_to_ray = cast(Vector, ray.origin - Point(0, 0, 0))

        a = ray.direction.dot(ray.direction)
        b = 2 * ray.direction.dot(object_to_ray)
        c = object_to_ray.dot(object_to_ray) - 1

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return []

        discriminant_root = math.sqrt(discriminant)
        return [
            Intersection((-b - discriminant_root) / (2 * a), self),
            Intersection((-b + discriminant_root) / (2 * a), self),
        ]
