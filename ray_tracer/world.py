from itertools import chain

from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.computation import Computation
from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.material import Material
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.point_light import PointLight
from ray_tracer.objects.sphere import Sphere


class World:
    """Defines the default scene for populating"""

    def __init__(self, default: bool = False) -> None:
        self.lights = []
        self.objects = []

        if default is True:
            self.lights = [PointLight(Point(-10, 10, -10), Colours.WHITE)]

            self.objects = [Sphere(), Sphere()]
            self.objects[0].material = Material(
                Colour(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2
            )
            self.objects[1].transform = Transforms.scaling(0.5, 0.5, 0.5)

    def intersect(self, ray: Ray) -> list[Intersection]:
        return sorted(
            chain.from_iterable(ray.intersect(o) for o in self.objects),
            key=lambda hit: hit.t,
        )

    def shade_hit(self, comps: Computation) -> Colour:
        shadowed = self.is_shadowed(comps.over_point)

        return comps.obj.material.lighting(
            self.lights[0], comps.point, comps.eyev, comps.normalv, shadowed
        )

    def colour_at(self, r: Ray) -> Colour:
        xs = self.intersect(r)

        try:
            hit = next(x for x in xs if x.t >= 0)
            return self.shade_hit(Computation(hit, r))
        except StopIteration:
            return Colours.BLACK

    def is_shadowed(self, p: Point) -> bool:
        v: Vector = self.lights[0].position - p
        distance = abs(v)
        direction = v.normalize()

        r = Ray(p, direction)
        xs = self.intersect(r)

        h = Intersection.hit(xs)

        # Have to cast to boolean here since h.t < distance returns a numpy
        # boolean.  so np.True_ rather than True.  Much time was wasted tracking
        # down this little nugget.  Bastard.
        return bool(h is not None and h.t < distance)
