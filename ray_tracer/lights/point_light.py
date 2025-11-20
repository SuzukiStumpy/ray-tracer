from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.lights.light import Light


class PointLight(Light):
    """A simple point light source"""

    def __init__(self, position: Point, intensity: Colour) -> None:
        self.position = position
        self.intensity = intensity
