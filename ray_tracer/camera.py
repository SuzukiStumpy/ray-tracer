import math
from typing import cast

from ray_tracer.classes.canvas import Canvas
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.world import World


class Camera:
    def __init__(self, hsize: int, vsize: int, field_of_view: float) -> None:
        self.hsize = hsize
        self.vsize = vsize
        self.field_of_view = field_of_view
        self.__dict__["transform"] = Matrix.Identity()
        self.__dict__["inverse_transform"] = self.__dict__["transform"].inverse()

        half_view = math.tan(self.field_of_view / 2)
        aspect = self.hsize / self.vsize

        if aspect >= 1:
            self.half_width = half_view
            self.half_height = half_view / aspect
        else:
            self.half_width = half_view * aspect
            self.half_height = half_view

        self.pixel_size = (self.half_width * 2) / self.hsize

    def ray_for_pixel(self, px: int, py: int) -> Ray:
        # precompute the inverse of the transformation matrix
        inv = self.inverse_transform

        # the offset from the edge of the canvas to the pixel's centre
        xoffset = (px + 0.5) * self.pixel_size
        yoffset = (py + 0.5) * self.pixel_size

        # the untransformed coordinates of the pixel in world space
        world_x = self.half_width - xoffset
        world_y = self.half_height - yoffset

        # using the camera matrix, transform the canvas point and the origin then
        # compute the ray's direction
        pixel = cast(Point, inv * Point(world_x, world_y, -1))
        origin = cast(Point, inv * Point(0, 0, 0))
        direction = cast(Vector, (pixel - origin)).normalize()

        return Ray(origin, direction)

    def render(self, world: World) -> Canvas:
        image = Canvas(self.hsize, self.vsize)

        for y in range(self.vsize):
            print(f"Rendering row {y + 1} of {self.vsize}")
            for x in range(self.hsize):
                ray = self.ray_for_pixel(x, y)
                colour = world.colour_at(ray)
                # Clamp prevents the image from corrupting when colours go past white
                image.set_pixel(x, y, colour.clamp())

        return image

    @property
    def transform(self) -> Matrix:
        return self.__dict__["transform"]

    @transform.setter
    def transform(self, m: Matrix) -> None:
        self.__dict__["transform"] = m
        self.__dict__["inverse_transform"] = m.inverse()

    @property
    def inverse_transform(self) -> Matrix:
        v = self.__dict__.get("inverse_transform")
        if v is None:
            v = self.__dict__["transform"].inverse()
            self.__dict__["inverse_transform"] = v
        return v

    @inverse_transform.setter
    def inverse_transform(self, m: Matrix) -> None:
        self.__dict__["inverse_transform"] = m
