"""Basic test for the matrix transformations"""

import math
from typing import cast

from ray_tracer.classes.canvas import Canvas
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms


def run() -> None:
    # Define a square canvas
    c = Canvas(500, 500)

    origin = (int(c.width / 2), int(c.height / 2))
    face_radius = 500 * (3 / 8)
    unit_angle = math.pi / 6

    for i in range(12):
        # Place the point at 12 o'clock
        p = Point(0, 0, 1)

        # Rotate the point by the requisite amount
        r = Transforms.rotation_y((i + 1) * unit_angle)

        p2: Point = cast(Point, r * p)

        c.set_pixel(
            origin[0] + round(p2.x * face_radius),
            origin[1] + round(p2.z * face_radius),
            Colours.GREEN,
        )

    c.to_image().show()


if __name__ == "__main__":
    run()
