from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector

if TYPE_CHECKING:
    from ray_tracer.objects.csg import CSG
    from ray_tracer.objects.group import Group


@dataclass
class Bounds:
    low: Point
    high: Point


class AbstractObject(ABC):
    def __init__(self) -> None:
        self.id = id(self)
        # store values in __dict__ so __eq__ behavior stays consistent
        self.__dict__["transform"] = Matrix.Identity()
        self.__dict__["inverse_transform"] = self.__dict__["transform"].inverse()
        self.material = Material()
        self.parent: Group | CSG | None = None

    @property
    def bounds(self) -> Bounds:
        return self.__dict__["bounds"]

    @bounds.setter
    def bounds(self, bounds: Bounds) -> None:
        self.__dict__["bounds"] = bounds

    def __eq__(self, other: object) -> bool:
        ignore_keys = {"id", "parent", "children"}

        if self.__class__.__name__ == "Group":
            ignore_keys.remove("id")

        if isinstance(other, self.__class__):
            return {k: v for k, v in self.__dict__.items() if k not in ignore_keys} == {
                k: v for k, v in other.__dict__.items() if k not in ignore_keys
            }
        else:
            return False

    def __contains__(self, other: object) -> bool:
        if not isinstance(other, AbstractObject):
            return False
        elif self.__class__.__name__ == "Group" and hasattr(self, "children"):
            for c in self.children:
                if other in c:
                    return True
        elif (
            self.__class__.__name__ == "CSG"
            and hasattr(self, "left")
            and hasattr(self, "right")
        ):
            return other in self.left or other in self.right
        else:
            return self == other

    def set_transform(self, m: Matrix) -> None:
        # Use the property to set the transform and update the cached inverse
        self.transform = m
        self.inverse_transform = m.inverse()

    def set_parent(self, p: Group | CSG) -> None:
        self.parent = p

    @property
    def transform(self) -> Matrix:
        return self.__dict__["transform"]

    @transform.setter
    def transform(self, m: Matrix) -> None:
        # store under the same key so external code inspecting __dict__ sees it
        self.__dict__["transform"] = m
        # clear cached inverse so it will be recomputed lazily
        self.__dict__["inverse_transform"] = m.inverse()

    @property
    def inverse_transform(self) -> Matrix:
        v = self.__dict__.get("inverse_transform")
        if v is None:
            v = self.__dict__["transform"].inverse()
            self.__dict__["inverse_transform"] = v
        return v

    @inverse_transform.setter
    def inverse_transform(self, m: Matrix) -> None:
        self.__dict__["inverse_transform"] = m

    """ Computation of Normals
        ----------------------
        Client code should call .normal_at(point).

        Specific objects should override _normal_func(self, point) -> Vector which
        calculates the normal _in object space_.  The normal_at function below
        handles all the conversion to/from object space and normalises the resulting
        world space vector before returning.
    """

    def normal_at(self, p: Point, i: Intersection | None = None) -> Vector:
        local_point = self.world_to_object(p)
        local_normal = self._normal_func(local_point, i)
        return self.normal_to_world(local_normal)

    @abstractmethod
    def _normal_func(self, op: Point, i: Intersection | None = None) -> Vector: ...

    def intersect(self, ray: Ray) -> list[Intersection]:
        # Convert ray to object space
        local_ray = ray.transform(self.inverse_transform)
        return self._local_intersect(local_ray)

    @abstractmethod
    def _local_intersect(self, ray: Ray) -> list[Intersection]: ...

    def world_to_object(self, point: Point) -> Point:
        if self.parent:
            point = self.parent.world_to_object(point)

        return cast(Point, self.transform.inverse() * point)

    def normal_to_world(self, normal: Vector) -> Vector:
        normal = cast(Vector, self.transform.inverse().transpose() * normal)
        normal.w = 0
        normal = normal.normalize()

        if self.parent:
            normal = self.parent.normal_to_world(normal)

        return normal

    def is_in(self, bounds: Bounds, full_containment: bool = False) -> bool:
        """Determines whether an object falls within a specific set of bounds
        the full_containment flag allows us to determine whether the object must be
        fully contained, or just partially contained (ie: just a two axes being
        within the tested bounds)"""
        in_bounds = [False, False, False]

        if (
            self.bounds.low.x >= bounds.low.x
            and self.bounds.low.x <= bounds.high.x
            and self.bounds.high.x <= bounds.high.x
            and self.bounds.high.x >= bounds.low.x
        ):
            in_bounds[0] = True

        if (
            self.bounds.low.y >= bounds.low.y
            and self.bounds.low.y <= bounds.high.y
            and self.bounds.high.y <= bounds.high.y
            and self.bounds.high.y >= bounds.low.y
        ):
            in_bounds[1] = True

        if (
            self.bounds.low.z >= bounds.low.z
            and self.bounds.low.z <= bounds.high.z
            and self.bounds.high.z <= bounds.high.z
            and self.bounds.high.z >= bounds.low.z
        ):
            in_bounds[2] = True

        return (
            in_bounds.count(True) > 2
            if full_containment is False
            else in_bounds.count(True) == 3
        )
