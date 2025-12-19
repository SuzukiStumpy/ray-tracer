import math

import pytest

from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.computation import Computation
from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON, ROOT2
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.world import World


class TestWorld:
    def test_creating_a_world(self) -> None:
        world = World()

        assert len(world.objects) == 0
        assert len(world.lights) == 0

    def test_the_default_world(self) -> None:
        light = PointLight(Point(-10, 10, -10), Colours.WHITE)
        s1 = Sphere()
        s1.material = Material(Colour(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)

        s2 = Sphere()
        s2.transform = Transforms.scaling(0.5, 0.5, 0.5)

        w = World(True)

        assert light in w.lights
        assert s1 in w.objects
        assert s2 in w.objects

    def test_intersecting_a_world_with_a_ray(self) -> None:
        w = World(True)
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))

        xs = w.intersect(r)

        assert len(xs) == 4
        assert xs[0].t == 4
        assert xs[1].t == 4.5
        assert xs[2].t == 5.5
        assert xs[3].t == 6

    def test_shading_an_intersection(self) -> None:
        w = World(True)
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        shape = w.objects[0]
        i = Intersection(4, shape)

        comps = Computation(i, r)
        c = w.shade_hit(comps)

        assert c == Colour(0.38066, 0.47583, 0.2855)

    def test_shaading_an_intersection_from_the_inside(self) -> None:
        w = World(True)
        w.lights[0] = PointLight(Point(0, 0.25, 0), Colours.WHITE)
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        shape = w.objects[1]
        i = Intersection(0.5, shape)

        comps = Computation(i, r)
        c = w.shade_hit(comps)

        assert c == Colour(0.90498, 0.90498, 0.90498)

    @pytest.mark.parametrize(
        "ray,expected",
        [
            (Ray(Point(0, 0, -5), Vector(0, 1, 0)), Colours.BLACK),
            (Ray(Point(0, 0, -5), Vector(0, 0, 1)), Colour(0.38066, 0.47583, 0.2855)),
        ],
        ids=[
            "misses",
            "hits",
        ],
    )
    def test_the_colour_when_a_ray(self, ray: Ray, expected: Colour) -> None:
        w = World(True)
        r = ray

        assert w.colour_at(r) == expected

    def test_the_colour_with_an_intersection_behind_the_ray(self) -> None:
        w = World(True)
        outer = w.objects[0]
        outer.material.ambient = 1

        inner = w.objects[1]
        inner.material.ambient = 1

        r = Ray(Point(0, 0, 0.75), Vector(0, 0, -1))

        assert inner.material.colour == w.colour_at(r)

    def test_there_is_no_shadow_when_nothing_is_colinear_with_point_and_light(
        self,
    ) -> None:
        w = World(True)
        p = Point(0, 10, 0)

        assert w.is_shadowed(p) is False

    def test_the_shadow_when_an_object_is_between_the_point_and_the_light(self) -> None:
        w = World(True)
        p = Point(10, -10, 10)

        assert w.is_shadowed(p) is True

    def test_there_is_no_shadow_when_an_object_is_behind_the_light(self) -> None:
        w = World(True)
        p = Point(-20, 20, -20)

        assert w.is_shadowed(p) is False

    def test_there_is_no_shadow_when_an_object_is_behind_the_point(self) -> None:
        w = World(True)
        p = Point(-2, 2, -2)

        assert w.is_shadowed(p) is False

    def test_shade_hit_is_given_an_intersection_in_shadow(self) -> None:
        w = World()
        w.lights = [PointLight(Point(0, 0, -10), Colours.WHITE)]
        s1 = Sphere()
        s2 = Sphere()
        s2.set_transform(Transforms.translation(0, 0, 10))
        w.objects = [s1, s2]
        r = Ray(Point(0, 0, 5), Vector(0, 0, 1))
        i = Intersection(4, s2)

        comps = Computation(i, r)

        assert w.shade_hit(comps) == Colour(0.1, 0.1, 0.1)

    def test_a_non_reflective_surface_reflects_black(self) -> None:
        w = World(default=True)
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        shape = w.objects[1]
        shape.material.ambient = 1.0
        i = Intersection(1, shape)

        comps = Computation(i, r)
        colour = w.reflected_colour(comps)

        assert colour == Colours.BLACK

    def test_a_reflective_surface_reflects_the_correct_colour(self) -> None:
        w = World(default=True)

        shape = Plane()
        shape.material.reflective = 0.5
        shape.set_transform(Transforms.translation(0, -1, 0))

        w.objects.extend([shape])

        r = Ray(Point(0, 0, -3), Vector(0, -ROOT2 / 2, ROOT2 / 2))
        i = Intersection(ROOT2, shape)

        comps = Computation(i, r)
        colour = w.reflected_colour(comps)

        assert math.isclose(colour.r, 0.19033, abs_tol=EPSILON)
        assert math.isclose(colour.g, 0.23791, abs_tol=EPSILON)
        assert math.isclose(colour.b, 0.14274, abs_tol=EPSILON)

    def test_shade_hit_incorporates_reflected_colour_into_output(self) -> None:
        w = World(default=True)

        shape = Plane()
        shape.material.reflective = 0.5
        shape.set_transform(Transforms.translation(0, -1, 0))

        w.objects.extend([shape])

        r = Ray(Point(0, 0, -3), Vector(0, -ROOT2 / 2, ROOT2 / 2))
        i = Intersection(ROOT2, shape)

        comps = Computation(i, r)
        colour = w.shade_hit(comps)

        assert math.isclose(colour.r, 0.87675, abs_tol=EPSILON)
        assert math.isclose(colour.g, 0.92434, abs_tol=EPSILON)
        assert math.isclose(colour.b, 0.82918, abs_tol=EPSILON)

    def test_we_avoid_infinite_reflective_recursion(self) -> None:
        w = World()
        w.lights = [PointLight(Point(0, 0, 0), Colour(1, 1, 1))]
        lower = Plane()
        lower.material.reflective = 1.0
        lower.set_transform(Transforms.translation(0, -1, 0))

        upper = Plane()
        upper.material.reflective = 1.0
        upper.set_transform(Transforms.translation(0, 1, 0))

        w.objects.extend([lower, upper])

        r = Ray(Point(0, 0, 0), Vector(0, 1, 0))

        try:
            w.colour_at(r)
        except Exception as e:
            pytest.fail(f"colour_at threw exception {e}")

    def test_the_reflected_colour_at_max_recursion_depth(self) -> None:
        w = World(default=True)

        shape = Plane()
        shape.material.reflective = 0.5
        shape.set_transform(Transforms.translation(0, -1, 0))

        w.objects.extend([shape])

        r = Ray(Point(0, 0, -3), Vector(0, -ROOT2 / 2, ROOT2 / 2))
        i = Intersection(ROOT2, shape)

        comps = Computation(i, r)
        colour = w.reflected_colour(comps, 0)

        assert colour == Colours.BLACK
