import math

from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.patterns import Noise, Stripes
from ray_tracer.world import World


def run() -> None:
    floor = Plane()

    pattern_a = Stripes(Colours.WHITE, Colours.GREEN)
    pattern_b = Stripes(Colours.BLUE, Colours.WHITE)

    pattern_a.set_transform(
        Transforms.rotation_y(math.pi / 4) * Transforms.scaling(0.5, 0.5, 0.5)  # type: ignore
    )
    pattern_b.set_transform(
        Transforms.rotation_y(2.35619) * Transforms.scaling(0.5, 0.5, 0.5)  # type: ignore
    )

    floor.material = Material(Noise(pattern_a, pattern_b))

    floor.set_transform(Transforms.translation(0, -1, 0))

    sphere_noise = Noise(Colours.RED, Colours.WHITE, 1024)
    sphere_noise.set_transform(Transforms.scaling(0.2, 1.0, 1.0))

    sphere = Sphere()
    sphere.material = Material(sphere_noise, specular=0.75)

    world = World()
    world.lights.append(PointLight(Point(-10, 10, 0), Colours.WHITE))
    world.objects.extend([floor, sphere])

    camera = Camera(500, 500, math.pi / 4)
    camera.transform = Transforms.view(Point(0, 2, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(world)

    canvas.to_image().show()


if __name__ == "__main__":
    run()
