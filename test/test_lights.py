from ray_tracer.classes.colour import Colour
from ray_tracer.classes.point import Point
from ray_tracer.lights.point_light import PointLight


class TestLight:
    def test_a_light_has_position_and_intensity(self) -> None:
        intensity = Colour(1, 1, 1)
        position = Point(0, 0, 0)

        light = PointLight(position, intensity)

        assert light.position == position
        assert light.intensity == intensity
