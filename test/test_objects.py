import math

import pytest

from ray_tracer.classes.colour import Colours
from ray_tracer.classes.computation import Computation
from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON, ROOT2, ROOT3
from ray_tracer.objects.abstract_object import Bounds
from ray_tracer.objects.cone import Cone
from ray_tracer.objects.cube import Cube
from ray_tracer.objects.cylinder import Cylinder
from ray_tracer.objects.group import Group
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.smooth_triangle import SmoothTriangle
from ray_tracer.objects.sphere import Sphere
from ray_tracer.objects.test_shape import TestShape
from ray_tracer.objects.triangle import Triangle


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

    def test_a_shape_has_a_parent_attribute(self) -> None:
        s = TestShape()

        assert s.parent is None

    def test_a_shape_has_a_bounding_box(self) -> None:
        s = TestShape()

        assert s.bounds == Bounds(Point(-1, -1, -1), Point(1, 1, 1))


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

    def test_a_sphere_has_unit_cube_bounding_box(self) -> None:
        s = Sphere()

        assert s.bounds == Bounds(Point(-1, -1, -1), Point(1, 1, 1))


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

    def test_a_planes_bounding_box_is_infinite_in_x_and_z_and_zero_in_y(self) -> None:
        p = Plane()

        assert p.bounds == Bounds(
            Point(-math.inf, 0, -math.inf), Point(math.inf, 0, math.inf)
        )


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

    def test_a_cube_has_a_bounding_box(self) -> None:
        c = Cube()

        assert c.bounds == Bounds(Point(-1, -1, -1), Point(1, 1, 1))


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

    def test_a_normal_cylinder_has_infinite_bounds(self) -> None:
        c = Cylinder()

        assert c.bounds == Bounds(Point(-1, -math.inf, -1), Point(1, math.inf, 1))

    def test_a_truncated_cylinder_has_capped_bounds(self) -> None:
        c = Cylinder(-5, 13, False)

        assert c.bounds == Bounds(Point(-1, -5, -1), Point(1, 13, 1))


class TestCones:
    @pytest.mark.parametrize(
        "origin,direction,t0,t1",
        [
            (Point(0, 0, -5), Vector(0, 0, 1), 5.0, 5.0),
            (Point(0, 0, -5), Vector(1, 1, 1), 8.66025, 8.66025),
            (Point(1, 1, -5), Vector(-0.5, -1, 1), 4.55006, 49.44994),
        ],
    )
    def test_intersecting_a_cone_with_a_ray(
        self, origin: Point, direction: Vector, t0: float, t1: float
    ) -> None:
        c = Cone()
        r = Ray(origin, direction.normalize())

        xs = c._local_intersect(r)

        assert len(xs) == 2
        assert math.isclose(xs[0].t, t0, abs_tol=EPSILON)
        assert math.isclose(xs[1].t, t1, abs_tol=EPSILON)

    def test_intersecting_a_cone_with_a_ray_parallel_to_one_of_its_halves(self) -> None:
        c = Cone()
        r = Ray(Point(0, 0, -1), Vector(0, 1, 1).normalize())
        xs = c._local_intersect(r)

        assert len(xs) == 1
        assert math.isclose(xs[0].t, 0.35355, abs_tol=EPSILON)

    @pytest.mark.parametrize(
        "origin,direction,count",
        [
            (Point(0, 0, -5), Vector(0, 1, 0), 0),
            (Point(0, 0, -0.25), Vector(0, 1, 1), 1),  # count was 2 as written
            (Point(0, 0, -0.25), Vector(0, 1, 0), 4),
        ],
    )
    def test_intersecting_a_cones_end_caps(
        self, origin: Point, direction: Vector, count: int
    ) -> None:
        c = Cone(-0.5, 0.5, True)
        r = Ray(origin, direction.normalize())

        xs = c._local_intersect(r)

        assert len(xs) == count

    @pytest.mark.parametrize(
        "point,normal",
        [
            (Point(0, 0, 0), Vector(0, 0, 0)),
            (Point(1, 1, 1), Vector(1, -ROOT2, 1)),
            (Point(-1, -1, 0), Vector(-1, 1, 0)),
        ],
    )
    def test_computing_normal_vector_for_a_cone(
        self, point: Point, normal: Vector
    ) -> None:
        c = Cone()

        assert c._normal_func(point) == normal

    def test_conic_bounds_are_infinite_for_default_cone(self) -> None:
        c = Cone()

        assert c.bounds == Bounds(
            Point(-math.inf, -math.inf, -math.inf), Point(math.inf, math.inf, math.inf)
        )

    def test_truncated_cone_bounds_track_largest_and_smallest_radius(self) -> None:
        c = Cone(-5, 10, False)

        assert c.bounds == Bounds(Point(-10, -5, -10), Point(10, 10, 10))


class TestGroup:
    def test_creation_of_a_new_group(self) -> None:
        g = Group()

        assert g.transform == Matrix.Identity()
        assert len(g.children) == 0

    def test_adding_a_child_to_a_group(self) -> None:
        g = Group()
        s = TestShape()

        g.add_child(s)

        assert len(g.children) == 1
        assert s in g.children
        assert s.parent == g

    def test_intersecting_a_ray_with_an_empty_group(self) -> None:
        g = Group()
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        xs = g._local_intersect(r)

        assert len(xs) == 0

    def test_intersecting_a_ray_with_a_nonempty_group(self) -> None:
        g = Group()
        s1 = Sphere()
        s2 = Sphere()
        s2.set_transform(Transforms.translation(0, 0, -3))
        s3 = Sphere()
        s3.set_transform(Transforms.translation(5, 0, 0))

        g.add_child(s1)
        g.add_child(s2)
        g.add_child(s3)

        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        xs = g._local_intersect(r)

        assert len(xs) == 4
        assert xs[0].obj == s2
        assert xs[1].obj == s2
        assert xs[2].obj == s1
        assert xs[3].obj == s1

    def test_intersecting_a_transformed_group(self) -> None:
        g = Group()
        g.set_transform(Transforms.scaling(2, 2, 2))

        s = Sphere()
        s.set_transform(Transforms.translation(5, 0, 0))

        g.add_child(s)

        r = Ray(Point(10, 0, -10), Vector(0, 0, 1))
        xs = g.intersect(r)

        assert len(xs) == 2

    def test_convert_a_point_from_world_to_object_space(self) -> None:
        g1 = Group()
        g1.set_transform(Transforms.rotation_y(math.pi / 2))

        g2 = Group()
        g2.set_transform(Transforms.scaling(2, 2, 2))

        g1.add_child(g2)

        s = Sphere()
        s.set_transform(Transforms.translation(5, 0, 0))

        g2.add_child(s)

        p = s.world_to_object(Point(-2, 0, -10))

        assert p == Point(0, 0, -1)

    def test_converting_a_normal_from_object_to_world_space(self) -> None:
        g1 = Group()
        g1.set_transform(Transforms.rotation_y(math.pi / 2))

        g2 = Group()
        g2.set_transform(Transforms.scaling(1, 2, 3))
        g1.add_child(g2)

        s = Sphere()
        s.set_transform(Transforms.translation(5, 0, 0))
        g2.add_child(s)

        n = s.normal_to_world(Vector(ROOT3 / 3, ROOT3 / 3, ROOT3 / 3))

        assert math.isclose(n.x, 0.28571, abs_tol=EPSILON)
        assert math.isclose(n.y, 0.42857, abs_tol=EPSILON)
        assert math.isclose(n.z, -0.85714, abs_tol=EPSILON)

    def test_finding_the_normal_on_a_child_object(self) -> None:
        g1 = Group()
        g1.set_transform(Transforms.rotation_y(math.pi / 2))

        g2 = Group()
        g2.set_transform(Transforms.scaling(1, 2, 3))

        g1.add_child(g2)

        s = Sphere()
        s.set_transform(Transforms.translation(5, 0, 0))

        g2.add_child(s)

        n = s.normal_at(Point(1.7321, 1.1547, -5.5774))

        assert math.isclose(n.x, 0.28571, abs_tol=EPSILON)
        assert math.isclose(n.y, 0.42854, abs_tol=EPSILON)
        assert math.isclose(n.z, -0.85716, abs_tol=EPSILON)

    def test_an_empty_group_has_a_point_bounding_box(self) -> None:
        g = Group()

        assert g.bounds == Bounds(Point(0, 0, 0), Point(0, 0, 0))

    def test_a_group_bounding_box_uses_the_transformed_object_bounds(self) -> None:
        g = Group()

        s = Sphere()
        s.set_transform(Transforms.scaling(2, 2, 2))

        g.add_child(s)

        assert g.bounds == Bounds(Point(-2, -2, -2), Point(2, 2, 2))

    def test_group_bounding_box_contains_the_bounds_of_all_its_transformed_objects(
        self,
    ) -> None:
        g = Group()

        s1 = Sphere()
        s1.set_transform(Transforms.translation(2, 2, 2))

        c1 = Cube()
        c1.set_transform(Transforms.rotation_x(math.pi / 4))

        g.add_child(s1)
        g.add_child(c1)

        assert g.bounds == Bounds(Point(-1, -1.41421, -1.41421), Point(3, 3, 3))

    def test_ray_hits_a_group_bounding_box(self) -> None:
        g = Group()
        s1 = Sphere()
        s1.set_transform(Transforms.translation(0, 1.5, 0))

        # r is defined so that it misses the sphere, but should hit the bounding box
        r = Ray(Point(0, 0, 0), Vector(1, 1, 1))

        g.add_child(s1)

        assert len(s1.intersect(r)) == 0
        assert g._bb_hit(r) is True


class TestTriangle:
    @staticmethod
    def create_standard_triangle() -> Triangle:
        return Triangle(Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0))

    def test_construction_of_a_triangle(self) -> None:
        p1 = Point(0, 1, 0)
        p2 = Point(-1, 0, 0)
        p3 = Point(1, 0, 0)

        t = Triangle(p1, p2, p3)

        assert t.verts[0] == p1
        assert t.verts[1] == p2
        assert t.verts[2] == p3
        assert t.edges[0] == Vector(-1, -1, 0)
        assert t.edges[1] == Vector(1, -1, 0)
        assert t.normal == Vector(0, 0, -1)

    def test_finding_the_normal_of_a_triangle(self) -> None:
        t = TestTriangle.create_standard_triangle()

        n1 = t._normal_func(Point(0, 0.5, 0))
        n2 = t._normal_func(Point(-0.5, 0.75, 0))
        n3 = t._normal_func(Point(0.5, 0.25, 0))

        assert n1 == t.normal
        assert n2 == t.normal
        assert n3 == t.normal

    def test_intersecting_a_ray_parallel_to_a_triangle(self) -> None:
        t = TestTriangle.create_standard_triangle()
        r = Ray(Point(0, -1, -2), Vector(0, 1, 0))
        xs = t._local_intersect(r)

        assert len(xs) == 0

    @pytest.mark.parametrize(
        "origin",
        [
            (Point(1, 1, -2)),
            (Point(-1, 1, -2)),
            (Point(0, -1, -2)),
        ],
    )
    def test_a_ray_misses_a_triangle_at_each_edge(self, origin: Point) -> None:
        t = TestTriangle.create_standard_triangle()
        r = Ray(origin, Vector(0, 0, 1))
        xs = t._local_intersect(r)

        assert len(xs) == 0

    def test_a_ray_strikes_a_triangle(self) -> None:
        t = TestTriangle.create_standard_triangle()
        r = Ray(Point(0, 0.5, -2), Vector(0, 0, 1))
        xs = t._local_intersect(r)

        assert len(xs) == 1
        assert xs[0].t == 2


class TestSmoothTriangle:
    def create_standard_triangle(self) -> None:
        return SmoothTriangle(
            Point(0, 1, 0),
            Point(-1, 0, 0),
            Point(1, 0, 0),
            Vector(0, 1, 0),
            Vector(-1, 0, 0),
            Vector(1, 0, 0),
        )

    def test_constructing_a_smmoth_triangle(self) -> None:
        tri = self.create_standard_triangle()

        assert tri.verts[0] == Point(0, 1, 0)
        assert tri.verts[1] == Point(-1, 0, 0)
        assert tri.verts[2] == Point(1, 0, 0)
        assert tri.normals[0] == Vector(0, 1, 0)
        assert tri.normals[1] == Vector(-1, 0, 0)
        assert tri.normals[2] == Vector(1, 0, 0)

    def test_intersection_with_smooth_triangle_stores_u_and_v(self) -> None:
        tri = self.create_standard_triangle()
        r = Ray(Point(-0.2, 0.3, -2), Vector(0, 0, 1))
        xs = tri._local_intersect(r)

        assert math.isclose(xs[0].u, 0.45, abs_tol=EPSILON)
        assert math.isclose(xs[0].v, 0.25, abs_tol=EPSILON)

    def test_a_smooth_triangle_uses_u_and_v_to_interpolate_normals(self) -> None:
        t = self.create_standard_triangle()
        i = Intersection(1, t, 0.45, 0.25)
        n = t.normal_at(Point(0, 0, 0), i)

        assert n == Vector(-0.5547, 0.83205, 0)

    def test_preparing_the_normal_on_a_smooth_triangle(self) -> None:
        t = self.create_standard_triangle()
        i = Intersection(1, t, 0.45, 0.25)
        r = Ray(Point(-0.2, 0.3, -2), Vector(0, 0, 1))
        xs = [i]
        comps = Computation(i, r, xs)

        assert comps.normalv == Vector(-0.5547, 0.83205, 0)
