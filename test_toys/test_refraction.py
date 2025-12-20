import math
from typing import cast

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.patterns import Checkerboard
from ray_tracer.patterns.noise import Noise
from ray_tracer.utils import profileit
from ray_tracer.world import World

"""Reflective and refractive materials on objects"""


# Comment out the decorator to speed up the execution of the program a little
# [still takes ages, reduce the recursion limit in the World instantiation for
# a more significant speed up [although don't set to zero else all reflections
# will disappear]
@profileit()
def run() -> None:
    back_wall = Plane()
    back_wall.material = Material(Checkerboard(Colours.WHITE, Colour(0.65, 0.65, 0.65)))
    back_wall.set_transform(
        cast(
            Matrix,
            Transforms.translation(0, 0, 10)  # type: ignore
            * Transforms.rotation_x(math.pi / 2)
            * Transforms.scaling(0.25, 0.25, 0.25),
        )
    )

    floor = Plane()
    floor.material = Material(
        Checkerboard(Colour(0.7, 0.3, 0.2), Colour(0.4, 0.8, 0.2)), reflective=0.1
    )
    floor.set_transform(Transforms.translation(0, -3, 0))

    sphere = Sphere()
    sphere.material = Material(
        Colour(0, 0.02, 0.0),
        diffuse=0.8,
        specular=0.75,
        shininess=300.0,
        reflective=0.9,
        refractive_index=1.5,
        transparency=0.99,
    )

    noise_pattern = Noise(Colours.WHITE, Colours.BLACK)
    noise_pattern.set_transform(
        Transforms.rotation_z(math.pi / 8) * Transforms.scaling(0.2, 1, 0.2)  # type: ignore
    )

    sphere2 = Sphere()
    sphere2.material = Material(
        noise_pattern, diffuse=0.1, specular=0.9, shininess=500, reflective=0.99
    )
    sphere2.set_transform(
        Transforms.translation(-1, -1, -1) * Transforms.scaling(0.5, 0.5, 0.5)  # type: ignore
    )

    world = World(max_recursion=5)
    world.lights = [PointLight(Point(10, 10, 0), Colours.WHITE)]
    world.objects = [floor, back_wall, sphere, sphere2]

    camera = Camera(500, 500, math.pi / 4)
    camera.transform = Transforms.view(
        Point(2, 0.5, -4), Point(0, 0, 0), Vector(0, 1, 0)
    )

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
