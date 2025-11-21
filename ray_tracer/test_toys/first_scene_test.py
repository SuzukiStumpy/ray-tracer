import math
from typing import cast

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.sphere import Sphere
from ray_tracer.world import World


def run() -> None:
    floor = Sphere()
    floor.transform = Transforms.scaling(10, 0.01, 10)
    floor.material = Material(Colour(1, 0.9, 0.9), specular=0)

    left_wall = Sphere()
    left_wall.set_transform(
        cast(
            Matrix,
            (
                Transforms.translation(0, 0, 5)  # type: ignore
                * Transforms.rotation_y(math.pi / 4)
                * Transforms.rotation_x(math.pi / 2)
                * Transforms.scaling(10, 0.01, 10)
            ),
        )
    )
    left_wall.material = floor.material

    right_wall = Sphere()
    right_wall.set_transform(
        cast(
            Matrix,
            (
                Transforms.translation(0, 0, 5)  # type: ignore
                * Transforms.rotation_y(math.pi / 4)
                * Transforms.rotation_x(math.pi / 2)
                * Transforms.scaling(10, 0.01, 10)
            ),
        )
    )

    middle = Sphere()
    middle.set_transform(Transforms.translation(-0.5, 1, 0.5))
    middle.material = Material(Colour(0.1, 1, 0.5), diffuse=0.7, specular=0.3)

    right = Sphere()
    right.set_transform(
        cast(
            Matrix,
            Transforms.translation(1.5, 0.5, -0.5) * Transforms.scaling(0.5, 0.5, 0.5),
        )
    )
    right.material = Material(Colour(0.5, 1, 0.1), diffuse=0.7, specular=0.3)

    left = Sphere()
    left.set_transform(
        cast(
            Matrix,
            Transforms.translation(-1.5, 0.33, -0.75)
            * Transforms.scaling(0.33, 0.33, 0.33),
        )
    )
    left.material = Material(Colour(1, 0.8, 0.1), diffuse=0.7, specular=0.3)

    world = World()
    world.lights.append(PointLight(Point(-10, 10, -10), Colour(1, 1, 1)))
    world.objects.extend([floor, left_wall, right_wall, middle, right, left])

    camera = Camera(300, 150, math.pi / 3)
    camera.transform = Transforms.view(
        Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0)
    )

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
