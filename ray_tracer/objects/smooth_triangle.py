import math
from typing import cast, override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON
from ray_tracer.objects.abstract_object import AbstractObject, Bounds


class SmoothTriangle(AbstractObject):
    def __init__(
        self, p1: Point, p2: Point, p3: Point, n1: Vector, n2: Vector, n3: Vector
    ) -> None:
        super().__init__()

        self.verts = [p1, p2, p3]
        self.edges: list[Vector] = [cast(Vector, p2 - p1), cast(Vector, p3 - p1)]
        self.normals = [n1, n2, n3]

        self.bounds = Bounds(
            Point(
                min(p1.x, p2.x, p3.x),
                min(p1.y, p2.y, p3.y),
                min(p1.z, p2.z, p3.z),
            ),
            Point(
                max(p1.x, p2.x, p3.x),
                max(p1.y, p2.y, p3.y),
                max(p1.z, p2.z, p3.z),
            ),
        )

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

        return [Intersection(t, self, u, v)]

    def _normal_func(self, op: Point, i: Intersection | None = None) -> Vector:
        if i is None:
            raise ValueError(
                "Smooth triangle _normal_func requires the passing of an intersection"
            )

        if i.u is None or i.v is None:
            raise ValueError(
                "Invalid intersection object passed to _normal_func in SmoothTriangle"
            )

        return cast(
            Vector,
            self.normals[1] * i.u
            + self.normals[2] * i.v
            + self.normals[0] * (1 - i.u - i.v),
        )
