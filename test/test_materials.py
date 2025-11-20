import math

from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight

root2 = math.sqrt(2)


class TestMaterials:
    def test_the_default_material(self) -> None:
        m = Material()

        assert m.colour == Colours.WHITE
        assert m.ambient == 0.1
        assert m.diffuse == 0.9
        assert m.specular == 0.9
        assert m.shininess == 200.0


class TestLighting:
    def test_with_the_eye_between_the_light_and_the_surface(self) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Colours.WHITE)
        result = m.lighting(light, position, eyev, normalv)

        assert result == Colour(1, 1, 1)

    def test_with_the_eye_between_the_light_and_the_surface_eye_offset_45_degrees(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, root2 / 2, -root2 / 2)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Colours.WHITE)
        result = m.lighting(light, position, eyev, normalv)

        assert result == Colour(1.0, 1.0, 1.0)

    def test_with_the_eye_opposite_surface_light_offset_45_degrees(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 10, -10), Colours.WHITE)
        result = m.lighting(light, position, eyev, normalv)

        assert result == Colour(0.7364, 0.7364, 0.7364)

    def test_with_the_eye_in_the_path_of_the_reflection_vector(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, -root2 / 2, -root2 / 2)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 10, -10), Colours.WHITE)
        result = m.lighting(light, position, eyev, normalv)

        assert result == Colour(1, 1, 1)

    def test_with_the_light_behind_the_surface(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, 10), Colours.WHITE)
        result = m.lighting(light, position, eyev, normalv)

        assert result == Colour(0.1, 0.1, 0.1)
