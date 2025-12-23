from typing import override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject


class Group(AbstractObject):
    @override
    def __init__(self) -> None:
        super().__init__()

        self.children: list[AbstractObject] = []

    @override
    def _normal_func(self, op: Point) -> Vector:
        raise NotImplementedError

    @override
    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        xs = []

        for c in self.children:
            xs.extend(c.intersect(ray))

        return sorted(xs, key=lambda i: i.t)

    def add_child(self, child: AbstractObject) -> None:
        if child == self:
            raise ValueError("A group cannot contain itself")

        child.set_parent(self)  # type: ignore[arg-type]
        self.children.append(child)
