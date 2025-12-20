import math

import pytest

from ray_tracer.classes.computation import Computation
from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON, ROOT2
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.world import World


class TestIntersections:
    def test_an_intersection_encapsulates_time_and_object(self) -> None:
        s = Sphere()
        i = Intersection(3.5, s)

        assert i.t == 3.5
        assert i.obj == s

    def test_aggregating_intersections(self) -> None:
        s = Sphere()
        i1 = Intersection(1, s)
        i2 = Intersection(2, s)

        xs = [i1, i2]

        assert len(xs) == 2
        assert xs[0].t == 1
        assert xs[1].t == 2

    def test_intersect_sets_the_object_on_the_intersection(self) -> None:
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere()

        xs = s.intersect(r)

        assert len(xs) == 2
        assert xs[0].obj == s
        assert xs[1].obj == s

    def test_the_hit_when_all_intersections_have_positive_t(self) -> None:
        s = Sphere()
        i1 = Intersection(1, s)
        i2 = Intersection(2, s)
        xs = [i1, i2]

        i = Intersection.hit(xs)

        assert i == i1

    def test_the_hit_when_some_intersections_have_negative_t(self) -> None:
        s = Sphere()
        i1 = Intersection(-1, s)
        i2 = Intersection(1, s)
        xs = [i2, i1]

        i = Intersection.hit(xs)

        assert i == i2

    def test_the_hit_when_all_intersections_have_negative_t(self) -> None:
        s = Sphere()
        i1 = Intersection(-2, s)
        i2 = Intersection(-1, s)
        xs = [i2, i1]

        i = Intersection.hit(xs)

        assert i is None

    def test_the_hit_is_always_the_lowest_nonnegative_intersection(self) -> None:
        s = Sphere()
        i1 = Intersection(5, s)
        i2 = Intersection(7, s)
        i3 = Intersection(-3, s)
        i4 = Intersection(2, s)
        xs = [i1, i2, i3, i4]

        i = Intersection.hit(xs)

        assert i == i4

    def test_precomputing_the_state_of_an_intersection(self) -> None:
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        shape = Sphere()
        i = Intersection(4, shape)

        comps = Computation(i, r)

        assert comps.t == i.t
        assert comps.obj == i.obj
        assert comps.point == Point(0, 0, -1)
        assert comps.eyev == Vector(0, 0, -1)
        assert comps.normalv == Vector(0, 0, -1)

    def test_hit_when_an_intersection_is_on_the_outside(self) -> None:
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        shape = Sphere()
        i = Intersection(4, shape)

        comps = Computation(i, r)

        assert comps.inside is False

    def test_hit_when_an_intersection_is_on_the_inside(self) -> None:
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        shape = Sphere()
        i = Intersection(1, shape)

        comps = Computation(i, r)

        assert comps.point == Point(0, 0, 1)
        assert comps.eyev == Vector(0, 0, -1)
        assert comps.inside is True
        assert comps.normalv == Vector(0, 0, -1)

    def test_the_hit_should_offset_the_point(self) -> None:
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        shape = Sphere()
        shape.set_transform(Transforms.translation(0, 0, 1))
        i = Intersection(5, shape)

        comps = Computation(i, r)

        assert comps.over_point.z < -EPSILON / 2
        assert comps.point.z > comps.over_point.z

    def test_precomputation_of_the_reflectv_vector(self) -> None:
        shape = Plane()
        r = Ray(Point(0, 1, -1), Vector(0, -ROOT2 / 2, ROOT2 / 2))
        i = Intersection(ROOT2, shape)

        comps = Computation(i, r)

        assert comps.reflectv == Vector(0, ROOT2 / 2, ROOT2 / 2)

    @pytest.mark.parametrize(
        "index,n1,n2",
        [
            (0, 1.0, 1.5),
            (1, 1.5, 2.0),
            (2, 2.0, 2.5),
            (3, 2.5, 2.5),
            (4, 2.5, 1.5),
            (5, 1.5, 1.0),
        ],
    )
    def test_n1_and_n2_at_various_intersections_for_transparent_materials(
        self, index: int, n1: float, n2: float
    ) -> None:
        a = Sphere.glass()
        a.set_transform(Transforms.scaling(2, 2, 2))
        a.material.refractive_index = 1.5

        b = Sphere.glass()
        b.set_transform(Transforms.translation(0, 0, -0.25))
        b.material.refractive_index = 2.0

        c = Sphere.glass()
        c.set_transform(Transforms.translation(0, 0, 0.25))
        c.material.refractive_index = 2.5

        w = World()
        w.objects = [a, b, c]

        r = Ray(Point(0, 0, -4), Vector(0, 0, 1))
        xs = w.intersect(r)
        comps = Computation(xs[index], r, xs)

        assert comps.n1 == n1
        assert comps.n2 == n2

    def test_the_under_point_is_offset_below_the_surface(self) -> None:
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere.glass()
        s.set_transform(Transforms.translation(0, 0, 1))

        w = World()
        w.objects = [s]

        i = Intersection(5, s)
        xs = w.intersect(r)
        comps = Computation(i, r, xs)

        assert comps.under_point.z > EPSILON / 2
        assert comps.point.z < comps.under_point.z

    def test_the_schlick_approximation_under_total_internal_reflection(self) -> None:
        w = World()
        w.objects = [Sphere.glass()]

        r = Ray(Point(0, 0, ROOT2 / 2), Vector(0, 1, 0))
        xs = w.intersect(r)
        comps = Computation(xs[1], r, xs)
        reflectance = comps.schlick()

        assert reflectance == 1.0

    def test_reflectance_is_small_with_a_perpendicular_view_angle(self) -> None:
        w = World()
        w.objects = [Sphere.glass()]

        r = Ray(Point(0, 0, 0), Vector(0, 1, 0))
        xs = w.intersect(r)
        comps = Computation(xs[1], r, xs)
        reflectance = comps.schlick()

        assert math.isclose(reflectance, 0.04, abs_tol=EPSILON)

    def test_reflectance_with_small_angle_and_n2_greater_than_n2(self) -> None:
        w = World()
        w.objects = [Sphere.glass()]

        r = Ray(Point(0, 0.99, -2), Vector(0, 0, 1))
        xs = w.intersect(r)
        comps = Computation(xs[0], r, xs)
        reflectance = comps.schlick()

        assert math.isclose(reflectance, 0.48881, abs_tol=EPSILON)
