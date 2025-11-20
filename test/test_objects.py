import math

import pytest

from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.sphere import Sphere

root2 = math.sqrt(2)
root3 = math.sqrt(3)


class TestSphere:
    def test_a_spheres_default_transformation(self) -> None:
        s = Sphere()

        assert s.transform == Matrix.Identity()

    def test_changing_a_spheres_transformation(self) -> None:
        s = Sphere()
        t = Transforms.translation(2, 3, 4)

        s.set_transform(t)

        assert s.transform == t

    def test_intersecting_a_scaled_sphere_with_a_ray(self) -> None:
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere()

        s.set_transform(Transforms.scaling(2, 2, 2))
        xs = r.intersect(s)

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
                Point(root3 / 3, root3 / 3, root3 / 3),
                Vector(root3 / 3, root3 / 3, root3 / 3),
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
        n = s.normal_at(Point(root3 / 3, root3 / 3, root3 / 3))

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

        n = s.normal_at(Point(0, root2 / 2, -root2 / 2))

        assert n == Vector(0, 0.97014, -0.24254)

    def test_a_sphere_has_a_default_material(self) -> None:
        s = Sphere()

        assert s.material == Material()

    def test_a_sphere_may_be_assigned_a_material(self) -> None:
        s = Sphere()
        m = Material()
        m.ambient = 1
        s.material = m

        assert s.material == m
