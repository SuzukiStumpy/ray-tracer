import math
from typing import cast, override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON
from ray_tracer.objects.abstract_object import AbstractObject, Bounds


class Triangle(AbstractObject):
    @override
    def __init__(self, p1: Point, p2: Point, p3: Point) -> None:
        super().__init__()

        self.verts: list[Point] = [p1, p2, p3]
        self.edges: list[Vector] = [cast(Vector, p2 - p1), cast(Vector, p3 - p1)]
        self.normal: Vector = self.edges[1].cross(self.edges[0]).normalize()

    @override
    def _normal_func(self, op: Point) -> Vector:
        return self.normal

    @override
    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        """Uses the Moller-Trumbore ray/triangle intersection algorithm
        see: https://www.tandfonline.com/doi/abs/10.1080/10867651.1997.10487468"""
        dir_cross_e2 = ray.direction.cross(self.edges[1])
        determinant = dir_cross_e2.dot(self.edges[0])

        if math.isclose(determinant, 0, abs_tol=EPSILON):
            return []

        f = 1.0 / determinant

        p1_to_origin = cast(Vector, ray.origin - self.verts[0])
        u = f * p1_to_origin.dot(dir_cross_e2)

        if u < 0 or u > 1:
            return []

        origin_cross_e1 = p1_to_origin.cross(self.edges[0])
        v = f * ray.direction.dot(origin_cross_e1)

        if v < 0 or (u + v) > 1:
            return []

        t = f * self.edges[1].dot(origin_cross_e1)

        return [Intersection(t, self)]
