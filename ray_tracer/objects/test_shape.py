"""Not intended to be a useful class, but primarily for testing base behaviour"""

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject, Bounds


class TestShape(AbstractObject):
    __test__ = False

    def __init__(self) -> None:
        super().__init__()
        self.bounds = Bounds(Point(-1, -1, -1), Point(1, 1, 1))

    # Normal for a non-existant shape is simply a zero length vector
    def _normal_func(self, op: Point) -> Vector:
        return Vector(0, 0, 0)

    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        return []
