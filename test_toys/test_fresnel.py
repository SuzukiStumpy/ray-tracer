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

"""Test of the fresnel effect"""


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
            Transforms.translation(0, 0, 20)  # type: ignore
            * Transforms.rotation_x(math.pi / 2)
            * Transforms.scaling(0.25, 0.25, 0.25),
        )
    )

    bottom = Plane()
    bottom.material = Material(
        Checkerboard(Colour(0.7, 0.3, 0.2), Colour(0.4, 0.8, 0.2))
    )
    bottom.set_transform(Transforms.translation(0, -3, 0))

    water = Plane()
    water.material = Material(
        Colours.BLACK,
        transparency=1.0,
        reflective=0.2,
        refractive_index=1.01,
        cast_shadows=False,
    )
    water.set_transform(Transforms.translation(0, -1, 0))

    world = World(max_recursion=5)
    world.lights = [PointLight(Point(10, 10, 0), Colours.WHITE)]
    world.objects = [back_wall, bottom, water]

    camera = Camera(500, 200, math.pi / 2)
    camera.transform = Transforms.view(Point(0, 0, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
