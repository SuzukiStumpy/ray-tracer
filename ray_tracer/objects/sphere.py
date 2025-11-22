from typing import cast, override

from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject


class Sphere(AbstractObject):
    def __init__(self) -> None:
        super().__init__()

    @override
    def __normal_func(self, op: Point) -> Vector:
        """op is the point in object space"""
        return cast(Vector, op - Point(0, 0, 0))

    def normal_at(self, p: Point) -> Vector:
        return super()._normal_at(p, self.__normal_func)
