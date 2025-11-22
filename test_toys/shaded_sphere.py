"""Test the raycasting by basic silhouette plot of a sphere"""

from typing import cast

from ray_tracer.classes.canvas import Canvas
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.sphere import Sphere


def run() -> None:
    canvas_size = 500
    background_z = 10
    background_size = 7
    pixel_size = background_size / canvas_size
    half_width = background_size / 2

    c = Canvas(canvas_size, canvas_size)

    ray_origin = Point(0, 0, -5)

    s = Sphere()
    s.material.colour = Colour(1, 0.2, 1)

    light = PointLight(Point(-10, 10, -10), Colours.WHITE)

    for row in range(canvas_size):
        world_y = half_width - pixel_size * row

        for col in range(canvas_size):
            world_x = -half_width + pixel_size * col

            # Target position on the background that the ray is firing at
            position = Point(world_x, world_y, background_z)

            r = Ray(ray_origin, cast(Vector, (position - ray_origin)).normalize())

            xs = s.intersect(r)
            hit = Intersection.hit(xs)

            if hit:
                point = r.position(hit.t)
                normal = hit.obj.normal_at(point)
                eye = -r.direction

                colour = hit.obj.material.lighting(Sphere(), light, point, eye, normal)
                c.set_pixel(row, col, colour)

    c.to_image().show()


if __name__ == "__main__":
    run()
