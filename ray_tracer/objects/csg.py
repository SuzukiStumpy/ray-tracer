from enum import Enum

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.abstract_object import AbstractObject


class CSGOperation(Enum):
    union = 1
    intersection = 2
    difference = 3


class CSG(AbstractObject):
    def __init__(
        self, op: CSGOperation, s1: AbstractObject, s2: AbstractObject
    ) -> None:
        super().__init__()

        self.operation = op
        self.left = s1
        self.right = s2

        s1.set_parent(self)  # type: ignore[arg-type]
        s2.set_parent(self)  # type: ignore[arg-type]

    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        xs = self.left.intersect(ray)
        xs.extend(self.right.intersect(ray))
        xs.sort(key=lambda x: x.t)
        return self.filter_intersections(xs)

    def _normal_func(self, op: Point, i: Intersection | None = None) -> Vector: ...

    def filter_intersections(self, xs: list[Intersection]) -> list[Intersection]:
        # Begin outside of either child
        inl = False
        inr = False

        result = []

        for i in xs:
            # If i.obj is part of the 'left' child, then lhit is true
            lhit = self.left in i.obj

            if CSG.intersection_allowed(self.operation, lhit, inl, inr):
                result.append(i)

            # depending on which object was hit, toggle either inl or inr
            if lhit:
                inl = not inl
            else:
                inr = not inr

        return result

    @staticmethod
    def intersection_allowed(
        op: CSGOperation, lhit: bool, inl: bool, inr: bool
    ) -> bool:
        match op:
            case CSGOperation.union:
                return (lhit and not inr) or (not lhit and not inl)

            case CSGOperation.intersection:
                return (lhit and inr) or (not lhit and inl)

            case CSGOperation.difference:
                return (lhit and not inr) or (not lhit and inl)

            case _:
                raise RuntimeError("Invalid CSG operation")
