import math
from typing import cast

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.cylinder import Cylinder
from ray_tracer.objects.group import Group
from ray_tracer.objects.sphere import Sphere
from ray_tracer.utils import profileit
from ray_tracer.world import World

"""Test of the Group object"""


def hexagon_corner() -> Sphere:
    corner = Sphere()
    corner.set_transform(
        Transforms.translation(0, 0, -1) * Transforms.scaling(0.25, 0.25, 0.25)  # type: ignore
    )
    return corner


def hexagon_edge() -> Cylinder:
    edge = Cylinder(0, 1, False)
    edge.set_transform(
        cast(
            Matrix,
            Transforms.translation(0, 0, -1)  # type: ignore
            * Transforms.rotation_y(-math.pi / 6)
            * Transforms.rotation_z(-math.pi / 2)
            * Transforms.scaling(0.25, 1, 0.25),
        )
    )
    return edge


def hexagon_side() -> Group:
    side = Group()
    side.add_child(hexagon_corner())
    side.add_child(hexagon_edge())

    return side


def hexagon() -> Group:
    hex_ = Group()

    for n in range(0, 6):
        side = hexagon_side()
        side.set_transform(Transforms.rotation_y(n * math.pi / 3))
        hex_.add_child(side)

    return hex_


# Comment out the decorator to speed up the execution of the program a little
# [still takes ages, reduce the recursion limit in the World instantiation for
# a more significant speed up [although don't set to zero else all reflections
# will disappear]
@profileit()
def run() -> None:
    world = World(max_recursion=1)
    world.lights = [PointLight(Point(5, 5, -2), Colours.WHITE)]
    world.objects = [hexagon()]

    camera = Camera(500, 500, math.pi / 2)
    camera.transform = Transforms.view(Point(2, 2, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
