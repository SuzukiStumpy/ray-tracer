import math
from typing import cast

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.cone import Cone
from ray_tracer.utils import profileit
from ray_tracer.world import World

"""Test of the Cone object"""


# Comment out the decorator to speed up the execution of the program a little
# [still takes ages, reduce the recursion limit in the World instantiation for
# a more significant speed up [although don't set to zero else all reflections
# will disappear]
@profileit()
def run() -> None:
    c = Cone(-1, 0, False)
    c.set_transform(
        cast(
            Matrix,
            Transforms.translation(0, 1, 0) * Transforms.rotation_x(math.pi / 4),
        )
    )

    world = World(max_recursion=5)
    world.lights = [PointLight(Point(5, 5, -2), Colours.WHITE)]
    world.objects = [c]

    camera = Camera(500, 500, math.pi / 2)
    camera.transform = Transforms.view(Point(0, 2, -2), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
