import math

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.cube import Cube
from ray_tracer.patterns import Checkerboard
from ray_tracer.patterns.blend import Blend
from ray_tracer.patterns.stripes import Stripes
from ray_tracer.utils import profileit
from ray_tracer.world import World

"""Test of the Cube object"""


# Comment out the decorator to speed up the execution of the program a little
# [still takes ages, reduce the recursion limit in the World instantiation for
# a more significant speed up [although don't set to zero else all reflections
# will disappear]
@profileit()
def run() -> None:
    walls = Cube()

    stripe1 = Stripes(Colour(0.4, 0.02, 0.02), Colour(0.85, 0.85, 0.85))
    stripe1.set_transform(Transforms.scaling(0.2, 0.2, 0.2))

    stripe2 = Stripes(Colour(0.4, 0.02, 0.02), Colour(0.85, 0.85, 0.85))
    stripe2.set_transform(
        Transforms.rotation_y(math.pi / 2) * Transforms.scaling(0.2, 0.2, 0.2)  # type: ignore
    )

    blend = Blend(stripe1, stripe2)
    blend.set_transform(Transforms.scaling(0.2, 0.2, 0.2))

    walls.material = Material(
        blend,
        ambient=0.8,
        diffuse=0.8,
        specular=0.01,
        shininess=0.1,
    )
    walls.set_transform(Transforms.scaling(10.0, 10.2, 10.0))

    floor_ceiling = Cube()
    floor_ceiling.material = Material(
        Checkerboard(Colours.BLACK, Colours.WHITE),
        specular=0.9,
        shininess=200,
        reflective=0.01,
    )
    floor_ceiling.material.colour.set_transform(Transforms.scaling(0.2, 0.2, 0.2))  # type: ignore
    floor_ceiling.set_transform(Transforms.scaling(10.2, 10.0, 10.2))

    world = World(max_recursion=5)
    world.lights = [PointLight(Point(5, 5, -2), Colours.WHITE)]
    world.objects = [walls, floor_ceiling]

    camera = Camera(500, 500, math.pi / 2)
    camera.transform = Transforms.view(Point(2, 2, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
