import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.lights.light import Light

if TYPE_CHECKING:
    from ray_tracer.objects.abstract_object import AbstractObject

from ray_tracer.patterns.abstract_pattern import AbstractPattern


@dataclass
class Material:
    colour: Colour | AbstractPattern = field(default_factory=lambda: Colours.WHITE)
    ambient: float = 0.1
    diffuse: float = 0.9
    specular: float = 0.9
    shininess: float = 200.0
    reflective: float = 0.0
    transparency: float = 0.0
    refractive_index: float = 1.0

    def lighting(
        self,
        obj: "AbstractObject",
        light: Light,
        point: Point,
        eye_vector: Vector,
        normal_vector: Vector,
        in_shadow: bool = False,
    ) -> Colour:
        # Combine the surface colour with the light's colour/intensity
        if isinstance(self.colour, AbstractPattern):
            effective_colour: Colour = self.colour.colour_at_object(obj, point)
        else:
            effective_colour: Colour = self.colour

        effective_colour *= light.intensity

        # find the direction to the light source
        lightv: Vector = cast(Vector, light.position - point).normalize()

        # compute the ambient contribution
        ambient: Colour = effective_colour * self.ambient

        # light dot normal represents the cosine of the angle between the
        # light vector and the normal vector.  A negative number here means
        # the light is on the other side of the surface.
        light_dot_normal: float = lightv.dot(normal_vector)

        if light_dot_normal < 0 or in_shadow is True:
            diffuse: Colour = Colours.BLACK
            specular: Colour = Colours.BLACK
        else:
            # compute the diffuse contribution
            diffuse = effective_colour * self.diffuse * light_dot_normal

            # reflect dot eye represents the cosine of the angle between the
            # reflection vector and the eye vector.  A negative number means the
            # light reflects away from the eye
            reflectv: Vector = -lightv.reflect(normal_vector)
            reflect_dot_eye: float = reflectv.dot(eye_vector)

            if reflect_dot_eye <= 0:
                specular = Colours.BLACK
            else:
                # compute the specular contribution
                factor = math.pow(reflect_dot_eye, self.shininess)
                specular = light.intensity * self.specular * factor

        # combine the three contributions to get the final shading
        return (ambient + diffuse + specular).clamp()
