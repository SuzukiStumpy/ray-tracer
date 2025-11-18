from ray_tracer.objects.abstract_object import AbstractObject


class Sphere(AbstractObject):
    def __init__(self) -> None:
        self.id = id(self)
