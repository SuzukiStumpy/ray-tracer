import math

import pytest

from ray_tracer.classes.colour import Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON, ROOT2, ROOT3
from ray_tracer.objects.cube import Cube
from ray_tracer.objects.cylinder import Cylinder
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


class TestCube:
    @pytest.mark.parametrize(
        "origin,direction,t1,t2",
        [
            (Point(5, 0.5, 0), Vector(-1, 0, 0), 4.0, 6.0),
            (Point(-5, 0.5, 0), Vector(1, 0, 0), 4.0, 6.0),
            (Point(0.5, 5, 0), Vector(0, -1, 0), 4.0, 6.0),
            (Point(0.5, -5, 0), Vector(0, 1, 0), 4.0, 6.0),
            (Point(0.5, 0, 5), Vector(0, 0, -1), 4.0, 6.0),
            (Point(0.5, 0, -5), Vector(0, 0, 1), 4.0, 6.0),
            (Point(0, 0.5, 0), Vector(0, 0, 1), -1, 1),
        ],
        ids=[
            "+x",
            "-x",
            "+y",
            "-y",
            "+z",
            "-z",
            "inside",
        ],
    )
    def test_a_ray_intersects_a_cube(
        self, origin: Point, direction: Vector, t1: float, t2: float
    ) -> None:
        c = Cube()
        r = Ray(origin, direction)
        xs = c._local_intersect(r)

        assert len(xs) == 2
        assert math.isclose(xs[0].t, t1, abs_tol=EPSILON)
        assert math.isclose(xs[1].t, t2, abs_tol=EPSILON)

    @pytest.mark.parametrize(
        "origin,direction",
        [
            (Point(-2, 0, 0), Vector(0.2673, 0.5345, 0.8018)),
            (Point(0, -2, 0), Vector(0.8018, 0.2673, 0.5345)),
            (Point(0, 0, -2), Vector(0.5345, 0.8018, 0.2673)),
            (Point(2, 0, 2), Vector(0, 0, -1)),
            (Point(0, 2, 2), Vector(0, -1, 0)),
            (Point(2, 2, 0), Vector(-1, 0, 0)),
        ],
    )
    def test_a_ray_misses_a_cube(self, origin: Point, direction: Vector) -> None:
        c = Cube()
        r = Ray(origin, direction)
        xs = c._local_intersect(r)

        assert len(xs) == 0

    @pytest.mark.parametrize(
        "point,normal",
        [
            (Point(1, 0.5, -0.8), Vector(1, 0, 0)),
            (Point(-1, -0.2, 0.9), Vector(-1, 0, 0)),
            (Point(-0.4, 1, -0.1), Vector(0, 1, 0)),
            (Point(0.3, -1, -0.7), Vector(0, -1, 0)),
            (Point(-0.6, 0.3, 1), Vector(0, 0, 1)),
            (Point(0.4, 0.4, -1), Vector(0, 0, -1)),
            (Point(1, 1, 1), Vector(1, 0, 0)),
            (Point(-1, -1, -1), Vector(-1, 0, 0)),
        ],
    )
    def test_the_normal_on_the_surface_of_a_cube(
        self, point: Point, normal: Vector
    ) -> None:
        c = Cube()
        p = point

        assert c.normal_at(p) == normal


class TestCylinder:
    @pytest.mark.parametrize(
        "origin,direction",
        [
            (Point(1, 0, 0), Vector(0, 1, 0)),
            (Point(0, 0, 0), Vector(0, 1, 0)),
            (Point(1, 0, -5), Vector(1, 1, 1)),
        ],
    )
    def test_ray_misses_a_cylinder(self, origin: Point, direction: Vector) -> None:
        c = Cylinder()
        d = direction.normalize()
        r = Ray(origin, d)
        xs = c._local_intersect(r)

        assert len(xs) == 0

    @pytest.mark.parametrize(
        "origin,direction,t0,t1",
        [
            (Point(1, 0, -5), Vector(0, 0, 1), 5, 5),
            (Point(0, 0, -5), Vector(0, 0, 1), 4, 6),
            (Point(0.5, 0, -5), Vector(0.1, 1, 1), 6.80798, 7.08872),
        ],
    )
    def test_ray_hits_a_cylinder(
        self, origin: Point, direction: Vector, t0: float, t1: float
    ) -> None:
        c = Cylinder()
        d = direction.normalize()
        r = Ray(origin, d)
        xs = c._local_intersect(r)

        assert len(xs) == 2
        assert math.isclose(xs[0].t, t0, abs_tol=EPSILON)
        assert math.isclose(xs[1].t, t1, abs_tol=EPSILON)

    @pytest.mark.parametrize(
        "point,normal",
        [
            (Point(1, 0, 0), Vector(1, 0, 0)),
            (Point(0, 5, -1), Vector(0, 0, -1)),
            (Point(0, -2, 1), Vector(0, 0, 1)),
            (Point(-1, 1, 0), Vector(-1, 0, 0)),
        ],
    )
    def test_normal_vector_of_a_cylinder(self, point: Point, normal: Vector) -> None:
        c = Cylinder()

        assert c.normal_at(point) == normal

    def test_default_minimum_and_maximum_for_cylinders(self) -> None:
        c = Cylinder()

        assert c.min == -math.inf
        assert c.max == math.inf

    @pytest.mark.parametrize(
        "point,direction,count",
        [
            (Point(0, 1.5, 0), Vector(0.1, 1, 0), 0),
            (Point(0, 3, -5), Vector(0, 0, 1), 0),
            (Point(0, 0, -5), Vector(0, 0, 1), 0),
            (Point(0, 2, -5), Vector(0, 0, 1), 0),
            (Point(0, 1, -5), Vector(0, 0, 1), 0),
            (Point(0, 1.5, -2), Vector(0, 0, 1), 2),
        ],
    )
    def test_intersecting_a_truncated_cylinder(
        self,
        point: Point,
        direction: Vector,
        count: int,
    ) -> None:
        c = Cylinder(1, 2)
        r = Ray(point, direction.normalize())
        xs = c._local_intersect(r)

        assert len(xs) == count

    def test_default_cylinder_is_uncapped(self) -> None:
        c = Cylinder()

        assert c.closed is False

    @pytest.mark.parametrize(
        "point,direction,count",
        [
            (Point(0, 3, 0), Vector(0, -1, 0), 2),
            (Point(0, 3, -2), Vector(0, -1, 2), 2),
            (Point(0, 4, -2), Vector(0, -1, 1), 2),  # corner case
            (Point(0, 0, -2), Vector(0, 1, 2), 2),
            (Point(0, -1, -2), Vector(0, 1, 1), 2),  # corner case
        ],
    )
    def test_intersecting_a_cylinders_end_caps(
        self, point: Point, direction: Vector, count: int
    ) -> None:
        c = Cylinder(1, 2, True)
        r = Ray(point, direction.normalize())
        xs = c._local_intersect(r)

        assert len(xs) == count

    @pytest.mark.parametrize(
        "point,normal",
        [
            (Point(0, 1, 0), Vector(0, -1, 0)),
            (Point(0.5, 1, 0), Vector(0, -1, 0)),
            (Point(0, 1, 0.5), Vector(0, -1, 0)),
            (Point(0, 2, 0), Vector(0, 1, 0)),
            (Point(0.5, 2, 0), Vector(0, 1, 0)),
            (Point(0, 2, 0.5), Vector(0, 1, 0)),
        ],
    )
    def test_the_normal_vector_for_a_capped_cylinders_end_caps(
        self, point: Point, normal: Vector
    ) -> None:
        c = Cylinder(1, 2, True)
        assert c.normal_at(point) == normal
