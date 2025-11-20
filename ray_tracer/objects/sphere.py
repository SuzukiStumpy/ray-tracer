from typing import cast

from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject


class Sphere(AbstractObject):
    def __init__(self) -> None:
        super().__init__()

    def normal_at(self, p: Point) -> Vector:
        object_point: Point = cast(Point, self.transform.inverse() * p)
        object_normal: Vector = cast(Vector, object_point - Point(0, 0, 0))
        world_normal: Vector = cast(
            Vector, self.transform.inverse().transpose() * object_normal
        )

        return world_normal.normalize()
