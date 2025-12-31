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
from ray_tracer.utils import calculate_camera_distance, profileit
from ray_tracer.world import World

# Pumpkin doesn't play ball for now ... just renders as a grey blob - might be
# too large so light is inside?
# files = "cow-nonormals,pumpkin_tall_10k,teapot,teddy"
files = "cow-nonormals,teapot,teddy"


# Uncomment for full call profile data if needed
# @profileit()
def do_render(data: Group) -> None:
    w = World(max_recursion=1)

    w.lights = [PointLight(Point(10, 10, -10), Colours.WHITE)]
    w.objects = [data]

    c = Camera(300, 300, math.pi / 4)
    distance = calculate_camera_distance(data.bounds, math.pi / 4)
    center_x = (data.bounds.high.x + data.bounds.low.x) / 2
    center_y = (data.bounds.high.y + data.bounds.low.y) / 2
    center_z = (data.bounds.high.z + data.bounds.low.z) / 2
    c.transform = Transforms.view(
        Point(center_x, center_y, center_z - distance),
        Point(center_x, center_y, center_z),
        Vector(0, 1, 0),
    )

    canvas = c.render(w)

    canvas.to_image().show()


def run() -> None:
    for file in files.split(","):
        filepath = Path(f"test_toys/obj_files/{file}.obj")

        print(f"Loading file {filepath}...")

        loader = Loader.parse_obj_file(filepath)

        print("File loaded, optimizing data...")
        data = loader.obj_to_group()

        print("Optimization complete ... preparing to render")
        do_render(data)


if __name__ == "__main__":
    run()
