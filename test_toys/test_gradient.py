import math

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.sphere import Sphere
from ray_tracer.patterns import Gradient
from ray_tracer.world import World


def run() -> None:
    sphere = Sphere()
    sphere.material = Material(Gradient(Colours.WHITE, Colours.RED), specular=0.2)
    sphere.set_transform(Transforms.rotation_y(math.pi / 2))

    world = World()
    world.lights.append(PointLight(Point(-10, 10, -10), Colour(1, 1, 1)))
    world.objects.extend([sphere])

    camera = Camera(500, 500, math.pi / 4)
    camera.transform = Transforms.view(Point(-4, 0, 0), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
