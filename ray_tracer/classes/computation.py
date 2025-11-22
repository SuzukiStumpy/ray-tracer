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
    eyev: Vector
    normalv: Vector
    inside: bool = False

    def __init__(self, intersection: Intersection, ray: Ray) -> None:
        self.t = intersection.t
        self.obj = intersection.obj
        self.point = ray.position(self.t)
        self.eyev = -ray.direction
        self.normalv = self.obj.normal_at(self.point)

        if self.normalv.dot(self.eyev) < 0:
            self.inside = True
            self.normalv = -self.normalv

        self.over_point = cast(Point, self.point + self.normalv * EPSILON)
