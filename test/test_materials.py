import ray_tracer.patterns as Patterns
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import ROOT2
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.sphere import Sphere


class TestMaterials:
    def test_the_default_material(self) -> None:
        m = Material()

        assert m.colour == Colours.WHITE
        assert m.ambient == 0.1
        assert m.diffuse == 0.9
        assert m.specular == 0.9
        assert m.shininess == 200.0
        assert m.reflective == 0.0
        assert m.transparency == 0.0
        assert m.refractive_index == 1.0


class TestLighting:
    def test_with_the_eye_between_the_light_and_the_surface(self) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Colours.WHITE)
        result = m.lighting(Sphere(), light, position, eyev, normalv)

        assert result == Colour(1, 1, 1)

    def test_with_the_eye_between_the_light_and_the_surface_eye_offset_45_degrees(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, ROOT2 / 2, -ROOT2 / 2)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Colours.WHITE)
        result = m.lighting(Sphere(), light, position, eyev, normalv)

        assert result == Colour(1.0, 1.0, 1.0)

    def test_with_the_eye_opposite_surface_light_offset_45_degrees(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 10, -10), Colours.WHITE)
        result = m.lighting(Sphere(), light, position, eyev, normalv)

        assert result == Colour(0.7364, 0.7364, 0.7364)

    def test_with_the_eye_in_the_path_of_the_reflection_vector(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, -ROOT2 / 2, -ROOT2 / 2)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 10, -10), Colours.WHITE)
        result = m.lighting(Sphere(), light, position, eyev, normalv)

        assert result == Colour(1, 1, 1)

    def test_with_the_light_behind_the_surface(
        self,
    ) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, 10), Colours.WHITE)
        result = m.lighting(Sphere(), light, position, eyev, normalv)

        assert result == Colour(0.1, 0.1, 0.1)

    def test_lighting_with_the_surface_in_shadow(self) -> None:
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Colours.WHITE)
        in_shadow = True

        result = m.lighting(Sphere(), light, position, eyev, normalv, in_shadow)

        assert result == Colour(0.1, 0.1, 0.1)

    def test_lighting_with_a_pattern_applied(self) -> None:
        m = Material(
            Patterns.Stripes(Colours.WHITE, Colours.BLACK),
            ambient=1,
            diffuse=0,
            specular=0,
        )
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Colour(1, 1, 1))

        assert m.lighting(
            Sphere(), light, Point(0.9, 0, 0), eyev, normalv, False
        ) == Colour(1, 1, 1)
        assert m.lighting(
            Sphere(), light, Point(1.1, 0, 0), eyev, normalv, False
        ) == Colour(0, 0, 0)
