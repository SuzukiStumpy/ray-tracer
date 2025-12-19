import math

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON, ROOT2
from ray_tracer.world import World


class TestCamera:
    def test_constructing_a_camera(self) -> None:
        hsize = 160
        vsize = 120
        field_of_view = math.pi / 2

        c = Camera(hsize, vsize, field_of_view)

        assert c.hsize == 160
        assert c.vsize == 120
        assert c.field_of_view == math.pi / 2
        assert c.transform == Matrix.Identity()

    def test_the_pixel_size_for_a_horizontal_camera(self) -> None:
        c = Camera(200, 125, math.pi / 2)

        assert math.isclose(c.pixel_size, 0.01, rel_tol=EPSILON)

    def test_the_pixel_size_for_a_vertical_camera(self) -> None:
        c = Camera(125, 200, math.pi / 2)

        assert math.isclose(c.pixel_size, 0.01, rel_tol=EPSILON)

    def test_firing_a_ray_through_the_centre_of_the_canvas(self) -> None:
        c = Camera(201, 101, math.pi / 2)
        r = c.ray_for_pixel(100, 50)

        assert r.origin == Point(0, 0, 0)
        assert r.direction == Vector(0, 0, -1)

    def test_firing_a_ray_through_a_corner_of_the_canvas(self) -> None:
        c = Camera(201, 101, math.pi / 2)
        r = c.ray_for_pixel(0, 0)

        assert r.origin == Point(0, 0, 0)
        assert r.direction == Vector(0.66519, 0.33259, -0.66851)

    def test_firing_a_ray_when_the_camera_is_transformed(self) -> None:
        c = Camera(201, 101, math.pi / 2)
        c.transform = Transforms.rotation_y(math.pi / 4) * Transforms.translation(
            0, -2, 5
        )
        r = c.ray_for_pixel(100, 50)

        assert r.origin == Point(0, 2, -5)
        assert r.direction == Vector(ROOT2 / 2, 0, -ROOT2 / 2)

    def test_rendering_the_world_with_a_camera(self) -> None:
        w = World(True)
        c = Camera(11, 11, math.pi / 2)
        from_ = Point(0, 0, -5)
        to = Point(0, 0, 0)
        up = Vector(0, 1, 0)
        c.transform = Transforms.view(from_, to, up)

        image = c.render(w)

        assert image.get_pixel(5, 5) == Colour(0.38066, 0.47583, 0.2855)
