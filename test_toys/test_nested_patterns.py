import math

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.plane import Plane
from ray_tracer.patterns import Checkerboard, Gradient, Stripes
from ray_tracer.world import World


def run() -> None:
    floor = Plane()

    pattern_a2 = Stripes(Colour(0.2, 0.75, 0.4), Colour(0.8, 0.6, 0.3))
    pattern_a2.set_transform(
        Transforms.rotation_y(math.pi / 2) * Transforms.scaling(0.2, 0.2, 0.2)  # type: ignore
    )

    pattern_a = Stripes(Colours.BLUE, pattern_a2)
    pattern_b = Gradient(Colours.RED, Colours.WHITE)

    pattern_a.set_transform(
        Transforms.rotation_y(math.pi / 4) * Transforms.scaling(0.2, 0.2, 0.2)  # type: ignore
    )
    pattern_b.set_transform(
        Transforms.rotation_y(math.pi / 2) * Transforms.scaling(0.5, 0.5, 0.5)  # type: ignore
    )

    floor.material = Material(Checkerboard(pattern_a, pattern_b))

    world = World()
    world.lights.append(PointLight(Point(-10, 10, 0), Colours.WHITE))
    world.objects.extend([floor])

    camera = Camera(500, 500, math.pi / 4)
    camera.transform = Transforms.view(Point(0, 2, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
