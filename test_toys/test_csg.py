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
from ray_tracer.objects.csg import CSG, CSGOperation
from ray_tracer.objects.cube import Cube
from ray_tracer.objects.sphere import Sphere
from ray_tracer.patterns.checkerboard import Checkerboard
from ray_tracer.patterns.noise import Noise
from ray_tracer.world import World


def run() -> None:
    room_pattern = Checkerboard(Colours.WHITE, Colour(0.7, 0.7, 0.7))
    room_pattern.set_transform(Transforms.scaling(0.2, 0.2, 0.2))

    room = Cube()
    room.material = Material(room_pattern)
    room.set_transform(
        cast(Matrix, Transforms.translation(0, 10, 0) * Transforms.scaling(10, 10, 10))
    )

    cube_pattern = Noise(Colour(0.75, 0.02, 0.15), Colour(1, 0.2, 0.2))
    cube_pattern.set_transform(Transforms.scaling(0.1, 3, 0.05))
    c1 = Cube()
    c1.material = Material(
        cube_pattern, ambient=0.6, diffuse=0.96, specular=0.95, shininess=500
    )

    s1 = Sphere.glass()
    s1.set_transform(Transforms.translation(0.25, 0.5, 0.25))

    csg = CSG(CSGOperation.union, c1, s1)
    csg.set_transform(
        cast(
            Matrix,
            Transforms.translation(0, 0.5, 0) * Transforms.rotation_y(math.pi / 8),
        )
    )

    w = World(max_recursion=5)
    w.lights = [PointLight(Point(7, 7, -7), Colours.WHITE)]
    w.objects = [room, csg]

    camera = Camera(500, 500, math.pi / 4)
    camera.transform = Transforms.view(Point(5, 3, -8), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = camera.render(w)
    canvas.to_image().show()


if __name__ == "__main__":
    run()
