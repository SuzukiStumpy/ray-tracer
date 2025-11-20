from abc import ABC

from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector


class AbstractObject(ABC):
    def __init__(self) -> None:
        self.id = id(self)
        self.transform = Matrix.Identity()
        self.material = Material()

    def set_transform(self, m: Matrix) -> None:
        self.transform = m

    def normal_at(self, p: Point) -> Vector: ...
