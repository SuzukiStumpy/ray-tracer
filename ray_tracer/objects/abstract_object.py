from abc import ABC, abstractmethod
from typing import Callable, cast

from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector


class AbstractObject(ABC):
    def __init__(self) -> None:
        self.id = id(self)
        # store values in __dict__ so __eq__ behavior stays consistent
        self.__dict__["transform"] = Matrix.Identity()
        self.__dict__["inverse_transform"] = self.__dict__["transform"].inverse()
        self.material = Material()

    def __eq__(self, other: object) -> bool:
        ignore_keys = "id"

        if isinstance(other, self.__class__):
            return {k: v for k, v in self.__dict__.items() if k not in ignore_keys} == {
                k: v for k, v in other.__dict__.items() if k not in ignore_keys
            }
        else:
            return False

    def set_transform(self, m: Matrix) -> None:
        # Use the property to set the transform and update the cached inverse
        self.transform = m
        self.inverse_transform = m.inverse()

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

        The objects themselves need to implement normal_at(p) as a simple call
        to super()._normal_at(p, self.__normal_func)

        Where they also implement __normal_func(self, point) -> Vector which
        calculates the normal _in object space_.  The _normal_at function below
        handles all the conversion to/from object space and normalises the resulting
        world space vector before returning.

        Whilst the machinery here is a little complex, it saves the objects themselves
        from having to implement all the boilerplate and should ensure consistency
        in behaviour between different object types.
    """

    def _normal_at(self, p: Point, f: Callable) -> Vector:
        # Convert point to object space
        object_point: Point = cast(Point, self.inverse_transform * p)

        # Compute the normal at the point in object space
        object_normal = f(object_point)

        # Convert the normal back to world space
        world_normal = cast(Vector, self.inverse_transform.transpose() * object_normal)

        # Return the normalised world normal
        return world_normal.normalize()

    @abstractmethod
    def _normal_func(self, op: Point) -> Vector: ...
