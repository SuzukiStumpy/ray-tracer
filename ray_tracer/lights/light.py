from abc import ABC
from dataclasses import dataclass, field

from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.point import Point


@dataclass
class Light(ABC):
    position: Point = field(default_factory=lambda: Point(0, 0, 0))
    intensity: Colour = field(default_factory=lambda: Colours.WHITE)
