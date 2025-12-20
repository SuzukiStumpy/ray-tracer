import math
from itertools import chain
from typing import cast

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

    def __init__(self, default: bool = False, max_recursion: int = 1) -> None:
        self.lights = []
        self.objects = []
        self.max_recursion = max_recursion

        if default is True:
            self.lights = [PointLight(Point(-10, 10, -10), Colours.WHITE)]

            self.objects = [Sphere(), Sphere()]
            self.objects[0].material = Material(
                Colour(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2
            )
            self.objects[1].transform = Transforms.scaling(0.5, 0.5, 0.5)

    def intersect(self, ray: Ray) -> list[Intersection]:
        return sorted(
            chain.from_iterable(o.intersect(ray) for o in self.objects),
            key=lambda hit: hit.t,
        )

    def shade_hit(self, comps: Computation, remaining: int | None = None) -> Colour:
        if remaining is None:
            remaining = self.max_recursion

        shadowed = self.is_shadowed(comps.over_point)

        surface = comps.obj.material.lighting(
            comps.obj, self.lights[0], comps.point, comps.eyev, comps.normalv, shadowed
        )
        reflected = self.reflected_colour(comps, remaining)
        refracted = self.refracted_colour(comps, remaining)

        return surface + reflected + refracted

    def colour_at(self, r: Ray, remaining: int | None = None) -> Colour:
        if remaining is None:
            remaining = self.max_recursion

        xs = self.intersect(r)

        try:
            hit = next(x for x in xs if x.t >= 0)
            return self.shade_hit(Computation(hit, r), remaining)
        except StopIteration:
            return Colours.BLACK

    def reflected_colour(
        self, comps: Computation, remaining: int | None = None
    ) -> Colour:
        if remaining is None:
            remaining = self.max_recursion

        if comps.obj.material.reflective == 0.0 or remaining <= 0:
            return Colours.BLACK

        reflect_ray = Ray(comps.over_point, comps.reflectv)
        colour = self.colour_at(reflect_ray, remaining - 1)

        return colour * comps.obj.material.reflective

    def refracted_colour(
        self, comps: Computation, remaining: int | None = None
    ) -> Colour:
        if remaining is None:
            remaining = self.max_recursion

        if comps.obj.material.transparency == 0 or remaining == 0:
            return Colours.BLACK

        # Snell's law for computation of total internal reflection.  If ray is
        # internally reflected, then return Black
        n_ratio = comps.n1 / comps.n2
        cos_i = comps.eyev.dot(comps.normalv)
        sin2_t = n_ratio**2 * (1 - cos_i**2)

        if sin2_t > 1.0:
            return Colours.BLACK

        # Now, compute the actual refracted colour...
        # Get cos_t via trigonometric identity
        cos_t = math.sqrt(1.0 - sin2_t)

        # Compute the direction of the refracted ray
        direction = cast(
            Vector, comps.normalv * (n_ratio * cos_i - cos_t) - comps.eyev * n_ratio
        )

        # Spawn the refracted ray
        refract_ray = Ray(comps.under_point, direction)

        # Find the colour of the refracted ray, making sure to multiply by the
        # transparency value to account for any opacity and return it
        return self.colour_at(refract_ray, remaining - 1) * (
            comps.obj.material.transparency
        )

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
