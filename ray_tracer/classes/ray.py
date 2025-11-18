import math
from typing import cast

from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject


class Ray:
    def __init__(self, origin: Point, direction: Vector) -> None:
        self.origin = origin
        self.direction = direction

    def position(self, time: float) -> Point:
        return cast(Point, self.origin + (time * self.direction))

    def intersect(self, object: AbstractObject) -> list[float]:
        # the vector from the object's centre to the ray origin (the object
        # is always centered on the world origin)
        object_to_ray = cast(Vector, self.origin - Point(0, 0, 0))

        a = self.direction.dot(self.direction)
        b = 2 * self.direction.dot(object_to_ray)
        c = object_to_ray.dot(object_to_ray) - 1

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return []

        discriminant_root = math.sqrt(discriminant)
        return [(-b - discriminant_root) / (2 * a), (-b + discriminant_root) / (2 * a)]
