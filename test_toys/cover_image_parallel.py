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
from ray_tracer.objects.cube import Cube
from ray_tracer.objects.plane import Plane
from ray_tracer.objects.sphere import Sphere
from ray_tracer.world import World

materials = {
    "white": Material(
        Colours.WHITE, diffuse=0.7, ambient=0.1, specular=0.0, reflective=0.1
    ),
    "blue": Material(
        Colour(0.537, 0.831, 0.914),
        diffuse=0.7,
        ambient=0.1,
        specular=0.0,
        reflective=0.1,
    ),
    "red": Material(
        Colour(0.941, 0.322, 0.388),
        diffuse=0.7,
        ambient=0.1,
        specular=0.0,
        reflective=0.1,
    ),
    "purple": Material(
        Colour(0.373, 0.404, 0.550),
        diffuse=0.7,
        ambient=0.1,
        specular=0.0,
        reflective=0.1,
    ),
}

transforms = {
    "standard": cast(
        Matrix, Transforms.translation(1, -1, 1) * Transforms.scaling(0.5, 0.5, 0.5)
    ),
}

transforms["large"] = cast(
    Matrix, transforms["standard"] * Transforms.scaling(3.5, 3.5, 3.5)
)
transforms["medium"] = cast(
    Matrix, transforms["standard"] * Transforms.scaling(3, 3, 3)
)
transforms["small"] = cast(Matrix, transforms["standard"] * Transforms.scaling(2, 2, 2))


def add_cube(mat: Material, obj_type: Matrix, position: Matrix) -> Cube:
    c = Cube()
    c.material = mat
    c.set_transform(cast(Matrix, position * obj_type))
    return c


def run() -> None:
    camera = Camera(500, 500, 0.785)
    camera.transform = Transforms.view(
        Point(-6, 6, -10), Point(6, 0, 6), Vector(-0.45, 1, 0)
    )

    world = World(max_recursion=5)
    world.lights = [
        PointLight(Point(50, 100, -50), Colour(1, 1, 1)),
        PointLight(Point(-400, 50, -10), Colour(0.2, 0.2, 0.2)),
    ]

    backdrop = Plane()
    backdrop.material = Material(Colours.WHITE, ambient=1, diffuse=0, specular=0)
    backdrop.set_transform(
        cast(
            Matrix,
            Transforms.translation(0, 0, 500) * Transforms.rotation_x(math.pi / 2),
        )
    )

    s1 = Sphere()
    s1.material = Material(
        Colour(0.373, 0.404, 0.550),
        diffuse=0.2,
        ambient=0.0,
        specular=1.0,
        shininess=200,
        reflective=0.7,
        transparency=0.7,
        refractive_index=1.5,
    )
    s1.set_transform(transforms["large"])

    world.objects = [backdrop, s1]

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["medium"],
            Transforms.translation(4, 0, 0),
        )
    )
    world.objects.append(
        add_cube(
            materials["blue"],
            transforms["large"],
            Transforms.translation(8.5, 1.5, -0.5),
        )
    )

    world.objects.append(
        add_cube(
            materials["red"],
            transforms["large"],
            Transforms.translation(0, 0, 4),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["small"],
            Transforms.translation(4, 0, 4),
        )
    )

    world.objects.append(
        add_cube(
            materials["purple"],
            transforms["medium"],
            Transforms.translation(7.5, 0.5, 4),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["medium"],
            Transforms.translation(-0.25, 0.25, 8),
        )
    )

    world.objects.append(
        add_cube(
            materials["blue"],
            transforms["large"],
            Transforms.translation(4, 1, 7.5),
        )
    )

    world.objects.append(
        add_cube(
            materials["red"],
            transforms["medium"],
            Transforms.translation(10, 2, 7.5),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["small"],
            Transforms.translation(8, 2, 12),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["small"],
            Transforms.translation(20, 1, 9),
        )
    )

    world.objects.append(
        add_cube(
            materials["blue"],
            transforms["large"],
            Transforms.translation(-0.5, -5, 0.25),
        )
    )

    world.objects.append(
        add_cube(
            materials["red"],
            transforms["large"],
            Transforms.translation(4, -4, 0),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["large"],
            Transforms.translation(8.5, -4, 0),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["large"],
            Transforms.translation(0, -4, 4),
        )
    )

    world.objects.append(
        add_cube(
            materials["purple"],
            transforms["large"],
            Transforms.translation(-0.5, -4.5, 8),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["large"],
            Transforms.translation(0, -8, 4),
        )
    )

    world.objects.append(
        add_cube(
            materials["white"],
            transforms["large"],
            Transforms.translation(-0.5, -8.5, 8),
        )
    )

    canvas = camera.render(world, parallel_render=True)
    canvas.to_image().show()


if __name__ == "__main__":
    run()
