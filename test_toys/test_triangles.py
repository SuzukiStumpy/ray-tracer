import math

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.cube import Cube
from ray_tracer.objects.triangle import Triangle
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
    t1 = Triangle(Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0))
    t2 = Triangle(Point(0, 1, 0), Point(1, 0, 0), Point(0.5, 1, 0.5))

    world = World(max_recursion=5)
    world.lights = [PointLight(Point(5, 5, -2), Colours.WHITE)]
    world.objects = [t1, t2]

    camera = Camera(500, 500, math.pi / 2)
    camera.transform = Transforms.view(Point(2, 2, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
