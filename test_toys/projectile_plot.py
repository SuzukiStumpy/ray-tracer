"""
Test program to exercise the point and vector objects.  Given a projectile (p - a point)
with velocity (v - a vector) and wind (w - also a vector), track the projectile's
position after each arbitrary unit of time (a tick) until it hits the ground.
"""

from typing import cast

from ray_tracer.classes.canvas import Canvas
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector


class Projectile:
    def __init__(self, position: Point, velocity: Vector) -> None:
        self.position = position
        self.velocity = velocity


class Environment:
    def __init__(self, gravity: Vector, wind: Vector) -> None:
        self.gravity = gravity
        self.wind = wind


def tick(projectile: Projectile, environment: Environment) -> Projectile:
    position = cast(Point, projectile.position + projectile.velocity)
    velocity = cast(
        Vector, projectile.velocity + environment.gravity + environment.wind
    )

    return Projectile(position, velocity)


def main() -> None:
    # Define our projectile
    p = Projectile(Point(0, 1, 0), (Vector(1, 1.8, 0).normalize()) * 11.25)

    # Define the world environment
    # (Gravity == down 0.1 unit per tick, Wind is -0.01 unit per tick)
    e = Environment(Vector(0, -0.1, 0), Vector(-0.01, 0, 0))

    # Define our canvas for plotting
    c = Canvas(900, 550)

    # Counter for number of ticks
    count = 0

    print("Starting simulation:")

    while p.position.y > 0:
        count += 1
        print(f"Tick: {count:3}, Position: {p.position}")

        c.set_pixel(round(p.position.x), c.height - round(p.position.y), Colours.RED)

        p = tick(p, e)

    print(f"Simulation ended at tick {count:3}.")

    c.to_image().show()


if __name__ == "__main__":
    main()
