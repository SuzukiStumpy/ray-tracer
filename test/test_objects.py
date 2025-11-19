from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.sphere import Sphere


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
