import math
from typing import cast

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.patterns import Checkerboard
from ray_tracer.utils import profileit
from ray_tracer.world import World

"""Tests out the new reflective materials.  Renders a single sphere surrounded
by black/white check walls.  Note that increasing the recursion depth will slow
things down significantly"""


# Comment out the decorator to speed up the execution of the program a little
# [still takes ages, reduce the recursion limit in the World instantiation for
# a more significant speed up [although don't set to zero else all reflections
# will disappear]
@profileit()
def run() -> None:
    floor = Plane()
    floor.material = Material(
        Checkerboard(Colours.BLACK, Colours.WHITE), reflective=0.1
    )
    floor.set_transform(Transforms.translation(0, -2, 0))

    side_wall = Plane()
    side_wall.material = Material(
        Checkerboard(Colours.BLACK, Colours.WHITE), reflective=0.1
    )
    side_wall.set_transform(
        cast(
            Matrix,
            Transforms.translation(-2, 0, 0) * Transforms.rotation_z(math.pi / 2),
        )
    )

    rear_wall = Plane()
    rear_wall.material = Material(
        Checkerboard(Colours.BLACK, Colours.WHITE), reflective=0.1
    )
    rear_wall.set_transform(
        cast(
            Matrix,
            Transforms.translation(0, 0, 2) * Transforms.rotation_x(math.pi / 2),
        )
    )

    sphere = Sphere()
    sphere.material = Material(
        Colours.RED, diffuse=0.8, specular=0.75, shininess=300.0, reflective=0.1
    )

    world = World(max_recursion=1)
    world.lights.append(PointLight(Point(10, 10, 0), Colours.WHITE))
    world.objects.extend([floor, side_wall, rear_wall, sphere])

    camera = Camera(500, 500, math.pi / 4)
    camera.transform = Transforms.view(Point(2, 2, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
