import math
from pathlib import Path

import ray_tracer.objects.loader as Loader
from ray_tracer.camera import Camera
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.group import Group
from ray_tracer.utils import profileit
from ray_tracer.world import World

files = "cow-nonormals,pumpkin_tall_10k,teapot,teddy"


@profileit()
def do_render(data: Group) -> None:
    w = World(max_recursion=1)

    w.lights = [PointLight(Point(10, 10, 10), Colours.WHITE)]
    w.objects = [data]

    c = Camera(300, 300, math.pi / 4)
    c.transform = Transforms.view(Point(0, 2, -4), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = c.render(w)

    canvas.to_image().show()


def run() -> None:
    for file in files.split(","):
        filepath = Path(f"test_toys/obj_files/{file}.obj")

        print(f"Loading file {filepath}...")

        loader = Loader.parse_obj_file(filepath)

        print("File loaded, proceeding to render...")
        do_render(loader.obj_to_group())


if __name__ == "__main__":
    run()
