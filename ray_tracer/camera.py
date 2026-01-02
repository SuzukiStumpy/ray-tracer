import math
import time
from multiprocessing import Pool
from typing import cast

from ray_tracer.classes.canvas import Canvas
from ray_tracer.classes.colour import Colour
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.world import World

BLOCK_SIZE = 64  # Chunk size for each process to render with when parallel processing


# This is the main worker process when parallel rendering
def render_block(args: tuple) -> tuple[int, int, list[tuple[int, int, Colour]]]:
    """Render a single block of the image when multiprocessing

    Args:
        args: (camera, world, x_start, y_start, block_size)

    Returns:
        (x_start, y_start, pixel_colours)
    """
    camera: Camera = args[0]
    world: World = args[1]
    x_start: int = args[2]
    y_start: int = args[3]
    block_size: int = args[4]

    pixels: list[tuple[int, int, Colour]] = []

    for y in range(y_start, min(y_start + block_size, camera.vsize)):
        for x in range(x_start, min(x_start + block_size, camera.hsize)):
            ray = camera.ray_for_pixel(x, y)
            colour: Colour = world.colour_at(ray)
            pixels.append((x, y, colour.clamp()))

    return (x_start, y_start, pixels)


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

    def render(self, world: World, parallel_render: bool = False) -> Canvas:
        image_start = time.perf_counter()

        if parallel_render:
            image = self.render_parallel(world, BLOCK_SIZE)
        else:
            image = self.render_single(world)

        image_end = time.perf_counter()
        elapsed_seconds = image_end - image_start
        elapsed_minutes = int(elapsed_seconds / 60)
        remaining_seconds = elapsed_seconds - (elapsed_minutes * 60)
        print(f"\nImage rendered in {elapsed_minutes}m {remaining_seconds:.2f}s")

        return image

    def render_parallel(self, world: World, block_size: int = 64) -> Canvas:
        """Main rendering function used when running multi-threaded"""
        image = Canvas(self.hsize, self.vsize)

        # Generate block tasks to submit
        tasks = []

        print("Preparing task definitions for parallel rendering")
        for y in range(0, self.vsize, block_size):
            for x in range(0, self.hsize, block_size):
                tasks.append((self, world, x, y, block_size))

        # Render all blocks in parallel
        print("Beginning render...")
        with Pool() as pool:
            results = pool.map(render_block, tasks)

        print("Rendering complete.  Assembling final image...")
        # Assemble the results back into the final image
        for x_start, y_start, pixels in results:
            for x, y, colour in pixels:
                image.set_pixel(x, y, colour)

        return image

    def render_single(self, world: World) -> Canvas:
        """Main rendering function used when running single-threaded"""
        image = Canvas(self.hsize, self.vsize)

        for y in range(self.vsize):
            print(
                " " * (79), end="\r", flush=True
            )  # Print a blank row to erase any previous output...
            print(f"Rendering row {y + 1} of {self.vsize}", end="", flush=True)
            start = time.perf_counter()

            for x in range(self.hsize):
                ray = self.ray_for_pixel(x, y)
                colour = world.colour_at(ray)
                # Clamp prevents the image from corrupting when colours go past white
                image.set_pixel(x, y, colour.clamp())

            end = time.perf_counter()
            print(
                f" : row {y + 1} rendered in {(end - start):.2f}s", end="", flush=True
            )

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
