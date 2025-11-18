from dataclasses import dataclass
from operator import attrgetter
from typing import Optional

from ray_tracer.objects.abstract_object import AbstractObject


@dataclass
class Intersection:
    t: float
    obj: AbstractObject

    @staticmethod
    def hit(intersections: list[Intersection]) -> Optional[Intersection]:
        # sort the list first by ascending time then return the first element
        # found with a non-negative 't'.  If we don't have _any_ hits, return
        # None
        intersections.sort(key=attrgetter("t"))
        try:
            return next((i for i in intersections if i.t >= 0))
        except StopIteration:
            return None
