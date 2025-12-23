from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from ray_tracer.classes.colour import Colour
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point

if TYPE_CHECKING:
    from ray_tracer.objects.abstract_object import AbstractObject


class AbstractPattern(ABC):
    def __init__(self) -> None:
        self.__dict__["transform"] = Matrix.Identity()
        self.__dict__["inverse_transform"] = self.transform.inverse()

    @abstractmethod
    def colour_at(self, p: Point) -> Colour | AbstractPattern: ...

    def colour_at_object(self, obj: "AbstractObject", p: Point) -> Colour:
        # Convert point to object space
        object_point = obj.world_to_object(p)

        # Convert object point to pattern space
        pattern_point = cast(Point, self.inverse_transform * object_point)

        computed_colour: Colour | AbstractPattern = self.colour_at(pattern_point)

        while isinstance(computed_colour, AbstractPattern):
            computed_colour = computed_colour.colour_at_object(obj, pattern_point)

        return computed_colour

    def set_transform(self, m: Matrix) -> None:
        self.transform = m
        self.inverse_transform = m.inverse()

    @property
    def transform(self) -> Matrix:
        return self.__dict__["transform"]

    @transform.setter
    def transform(self, m: Matrix) -> None:
        self.__dict__["transform"] = m
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
