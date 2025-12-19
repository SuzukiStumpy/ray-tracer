from typing import TYPE_CHECKING, cast, override

from opensimplex import noise3, random_seed, seed

from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.point import Point
from ray_tracer.patterns.abstract_pattern import AbstractPattern

if TYPE_CHECKING:
    from ray_tracer.objects.abstract_object import AbstractObject


class Noise(AbstractPattern):
    """Blends two patterns or colours together based on the output of the
    noise function"""

    def __init__(
        self,
        a: AbstractPattern | Colour,
        b: AbstractPattern | Colour,
        seed_: int | None = None,
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        if seed_ is None:
            random_seed()
        else:
            seed(seed_)

    def colour_at(self, p: Point) -> Colour:
        """Like blend, this isn't used, but is required by base class"""
        return Colours.BLACK

    @override
    def colour_at_object(self, obj: "AbstractObject", p: Point) -> Colour:
        # Comvert point to object space
        object_point = cast(Point, obj.inverse_transform * p)

        # Convert object point to pattern space
        pattern_point = cast(Point, self.inverse_transform * object_point)

        # Determine the value of the noise function at pattern_point and map it to the
        # range [0.0 - 1.0]
        noise_value = noise3(pattern_point.x, pattern_point.y, pattern_point.z)
        noise_value = (noise_value + 1) / 2

        colour_a = (
            self.a
            if isinstance(self.a, Colour)
            else self.a.colour_at_object(obj, pattern_point)
        ) * (1.0 - noise_value)

        colour_b = (
            self.b
            if isinstance(self.b, Colour)
            else self.b.colour_at_object(obj, pattern_point)
        ) * noise_value

        computed_colour = colour_a + colour_b

        return computed_colour
