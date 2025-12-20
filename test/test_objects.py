import math

import pytest

from ray_tracer.classes.colour import Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import ROOT2, ROOT3
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.objects.test_shape import TestShape


class TestShapes:
    def test_the_default_transformation(self) -> None:
        s = TestShape()

        assert s.transform == Matrix.Identity()

    def test_assigning_a_transformation(self) -> None:
        s = TestShape()
        s.set_transform(Transforms.translation(2, 3, 4))

        assert s.transform == Transforms.translation(2, 3, 4)

    def test_a_shape_has_a_default_material(self) -> None:
        s = TestShape()

        assert s.material == Material()

    def test_a_shape_can_be_assigned_a_material(self) -> None:
        s = TestShape()
        m = Material(Colours.BLUE, ambient=1)
        s.material = m

        assert s.material == m
        assert s.material.colour == Colours.BLUE
        assert s.material.ambient == 1


class TestSphere:
    def test_intersecting_a_scaled_sphere_with_a_ray(self) -> None:
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere()

        s.set_transform(Transforms.scaling(2, 2, 2))
        xs = s.intersect(r)

        assert len(xs) == 2
        assert xs[0].t == 3
        assert xs[1].t == 7

    @pytest.mark.parametrize(
        "p,expected",
        [
            pytest.param(Point(1, 0, 0), Vector(1, 0, 0), id="on the x axis"),
            pytest.param(Point(0, 1, 0), Vector(0, 1, 0), id="on the y axis"),
            pytest.param(Point(0, 0, 1), Vector(0, 0, 1), id="on the z axis"),
            pytest.param(
                Point(ROOT3 / 3, ROOT3 / 3, ROOT3 / 3),
                Vector(ROOT3 / 3, ROOT3 / 3, ROOT3 / 3),
                id="that is non-axial",
            ),
        ],
    )
    def test_the_normal_on_a_sphere_at_a_point(
        self,
        p: Point,
        expected: Vector,
    ) -> None:
        s = Sphere()

        assert s.normal_at(p) == expected

    def test_the_normal_is_a_normalized_vector(self) -> None:
        s = Sphere()
        n = s.normal_at(Point(ROOT3 / 3, ROOT3 / 3, ROOT3 / 3))

        assert n == n.normalize()

    def test_computing_the_normal_on_a_translated_sphere(self) -> None:
        s = Sphere()
        s.set_transform(Transforms.translation(0, 1, 0))
        n = s.normal_at(Point(0, 1.70711, -0.70711))

        assert n == Vector(0, 0.70711, -0.70711)

    def test_computing_the_normal_on_a_transformed_sphere(self) -> None:
        s = Sphere()
        m = Transforms.scaling(1, 0.5, 1) * Transforms.rotation_z(math.pi / 5)
        s.set_transform(m)

        n = s.normal_at(Point(0, ROOT2 / 2, -ROOT2 / 2))

        assert n == Vector(0, 0.97014, -0.24254)


class TestPlane:
    def test_normal_of_a_plane_is_constant_everywhere(self) -> None:
        p = Plane()

        assert p.normal_at(Point(0, 0, 0)) == Vector(0, 1, 0)
        assert p.normal_at(Point(10, 0, -10)) == Vector(0, 1, 0)
        assert p.normal_at(Point(-5, 0, 150)) == Vector(0, 1, 0)

    @pytest.mark.parametrize(
        "input_ray",
        [
            (Ray(Point(0, 10, 0), Vector(0, 0, 1))),
            (Ray(Point(0, 0, 0), Vector(0, 0, 1))),
        ],
        ids=["a ray parallel to the plane", "a coplanar ray"],
    )
    def test_intersection_with_a_ray_parallel_to_the_plane(
        self, input_ray: Ray
    ) -> None:
        p = Plane()
        r = input_ray

        xs = p.intersect(r)

        assert len(xs) == 0

    @pytest.mark.parametrize(
        "input_ray",
        [
            (Ray(Point(0, 1, 0), Vector(0, -1, 0))),
            (Ray(Point(0, -1, 0), Vector(0, 1, 0))),
        ],
        ids=["above", "below"],
    )
    def test_a_ray_intersecting_a_plane_from(self, input_ray: Ray) -> None:
        p = Plane()
        r = input_ray

        xs = p.intersect(r)

        assert len(xs) == 1
        assert xs[0].t == 1
        assert xs[0].obj == p
