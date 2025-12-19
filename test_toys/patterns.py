import math
from typing import cast

import ray_tracer.patterns as Patterns
from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import ROOT2
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.world import World


def run() -> None:
    floor = Plane()
    floor.material = Material(
        Patterns.Rings(Colour(1, 0.2, 0.1), Colour(0.4, 0.3, 0.3)),
        diffuse=0.7,
        specular=0.3,
    )
    floor.material.colour.set_transform(Transforms.scaling(0.1, 0.1, 0.1))  # type: ignore

    middle = Sphere()
    middle.set_transform(Transforms.translation(-0.5, 1, 0.5))
    middle.material = Material(
        Patterns.Stripes(Colour(0.1, 1, 0.5), Colour(0.2, 0.4, 0.7)),
        diffuse=0.7,
        specular=0.3,
    )
    middle.material.colour.set_transform(  # type:ignore
        Transforms.rotation_x(ROOT2 / 3)  # type: ignore
        * Transforms.rotation_z(1.2)
        * Transforms.scaling(0.3, 2, 0.75)
    )

    right = Sphere()
    right.set_transform(
        cast(
            Matrix,
            Transforms.translation(1.5, 0.5, -0.5) * Transforms.scaling(0.5, 0.5, 0.5),
        )
    )
    right.material = Material(
        Patterns.Checkerboard(Colours.BLUE, Colours.WHITE), diffuse=0.7, specular=0.3
    )
    right.material.colour.set_transform(Transforms.scaling(0.2, 0.2, 0.2))  # type: ignore

    left = Sphere()
    left.set_transform(
        cast(
            Matrix,
            Transforms.translation(-1.5, 0.33, -0.75)
            * Transforms.scaling(0.33, 0.33, 0.33),
        )
    )
    left.material = Material(
        Patterns.Gradient(Colour(1, 0.9, 0.9), Colour(1, 0.9, 0.2)), specular=0
    )

    world = World()
    world.lights.append(PointLight(Point(-10, 10, -10), Colour(1, 1, 1)))
    world.objects.extend([floor, middle, right, left])

    camera = Camera(500, 250, math.pi / 3)
    camera.transform = Transforms.view(
        Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0)
    )

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
