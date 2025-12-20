from dataclasses import dataclass
from typing import cast

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON
from ray_tracer.objects.abstract_object import AbstractObject


@dataclass
class Computation:
    """Helper class to precompute some values needed for calculating interactions
    between an intersection and a ray"""

    t: float
    obj: AbstractObject
    point: Point
    over_point: Point
    under_point: Point
    eyev: Vector
    normalv: Vector
    reflectv: Vector
    inside: bool = False

    def __init__(
        self, hit: Intersection, ray: Ray, xs: list[Intersection] | None = None
    ) -> None:
        """hit is the single intersection with the ray that we are computing for.
        xs is the full list of intersections with all objects in the world for this
        ray"""
        if xs is None:
            xs = [hit]

        self.t = hit.t
        self.obj = hit.obj
        self.point = ray.position(self.t)
        self.eyev = -ray.direction
        self.normalv = self.obj.normal_at(self.point)

        if self.normalv.dot(self.eyev) < 0:
            self.inside = True
            self.normalv = -self.normalv

        self.reflectv = ray.direction.reflect(self.normalv)

        self.over_point = cast(Point, self.point + self.normalv * EPSILON)
        self.under_point = cast(Point, self.point - self.normalv * EPSILON)

        # Compute for transparent/refractive surfaces
        containers: list[AbstractObject] = []

        for i in xs:
            if i == hit:
                self.n1 = (
                    1.0 if not containers else containers[-1].material.refractive_index
                )

            if i.obj in containers:
                containers.remove(i.obj)
            else:
                containers.append(i.obj)

            if i == hit:
                self.n2 = (
                    1.0 if not containers else containers[-1].material.refractive_index
                )
                break
