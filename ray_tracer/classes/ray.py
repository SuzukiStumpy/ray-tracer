import math
from typing import cast

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject


class Ray:
    def __init__(self, origin: Point, direction: Vector) -> None:
        self.origin = origin
        self.direction = direction

    def position(self, time: float) -> Point:
        return cast(Point, self.origin + (time * self.direction))

    def intersect(self, obj: AbstractObject) -> list[Intersection]:
        # Transform the ray by the inverse of the object's transformation matrix
        r2 = self.transform(obj.transform.inverse())

        # the vector from the object's centre to the ray origin (the object
        # is always centered on the world origin)
        object_to_ray = cast(Vector, r2.origin - Point(0, 0, 0))

        a = r2.direction.dot(r2.direction)
        b = 2 * r2.direction.dot(object_to_ray)
        c = object_to_ray.dot(object_to_ray) - 1

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return []

        discriminant_root = math.sqrt(discriminant)
        return [
            Intersection((-b - discriminant_root) / (2 * a), obj),
            Intersection((-b + discriminant_root) / (2 * a), obj),
        ]

    def transform(self, m: Matrix) -> Ray:
        p: Point = cast(Point, m * self.origin)
        d: Vector = cast(Vector, m * self.direction)
        return Ray(p, d)
