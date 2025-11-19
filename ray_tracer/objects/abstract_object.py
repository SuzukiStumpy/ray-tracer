from abc import ABC

from ray_tracer.classes.matrix import Matrix


class AbstractObject(ABC):
    def __init__(self) -> None:
        self.id = id(self)
        self.transform = Matrix.Identity()

    def set_transform(self, m: Matrix) -> None:
        self.transform = m
