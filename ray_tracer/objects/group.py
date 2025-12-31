import math
from typing import cast, override

from ray_tracer.classes.intersection import Intersection
from ray_tracer.classes.point import Point
from ray_tracer.classes.ray import Ray
from ray_tracer.classes.vector import Vector
from ray_tracer.constants import EPSILON
from ray_tracer.objects.abstract_object import AbstractObject, Bounds

# The most number of objects we can have directly in a group.
MAX_CHILDREN = 100

# Maximum recursion depth we'll drop to for the purposes of optimization
MAX_DEPTH = 20


class Group(AbstractObject):
    @override
    def __init__(self) -> None:
        super().__init__()

        self.children: list[AbstractObject] = []
        self.bounds = Bounds(Point(0, 0, 0), Point(0, 0, 0))

    @override
    def _normal_func(self, op: Point, i: Intersection | None = None) -> Vector:
        raise NotImplementedError

    @override
    def _local_intersect(self, ray: Ray) -> list[Intersection]:
        if self._bb_hit(ray):
            xs = []

            for c in self.children:
                xs.extend(c.intersect(ray))

            return sorted(xs, key=lambda i: i.t)
        else:
            return []

    def _bb_hit(self, ray: Ray) -> bool:
        """Private function to test if an incoming ray hits the group bounding box"""
        (xtmin, xtmax) = self._check_axis(
            ray.origin.x, ray.direction.x, (self.bounds.low.x, self.bounds.high.x)
        )
        (ytmin, ytmax) = self._check_axis(
            ray.origin.y, ray.direction.y, (self.bounds.low.y, self.bounds.high.y)
        )
        (ztmin, ztmax) = self._check_axis(
            ray.origin.z, ray.direction.z, (self.bounds.low.z, self.bounds.high.z)
        )

        tmin = max(xtmin, ytmin, ztmin)
        tmax = min(xtmax, ytmax, ztmax)

        return False if tmin > tmax else True

    def _check_axis(
        self, origin: float, direction: float, limits: tuple[float, float]
    ) -> tuple[float, float]:
        """Helper function to get planar intersects for a specific axis"""
        tmin_numerator = limits[0] - origin
        tmax_numerator = limits[1] - origin

        if abs(direction) >= EPSILON:
            tmin = tmin_numerator / direction
            tmax = tmax_numerator / direction
        else:
            tmin = tmin_numerator * math.inf
            tmax = tmax_numerator * math.inf

        return (tmax, tmin) if tmin > tmax else (tmin, tmax)

    def add_child(self, child: AbstractObject) -> None:
        if child == self:
            raise ValueError("A group cannot contain itself")

        child.set_parent(self)  # type: ignore[arg-type]
        self.children.append(child)

        # Now, figure out the transformed bounding box for the new child
        cb = child.bounds

        # 1. get the bounds and convert to the eight cube corners
        corners = [
            Point(cb.low.x, cb.high.y, cb.high.z),
            Point(cb.high.x, cb.high.y, cb.high.z),
            Point(cb.high.x, cb.high.y, cb.low.z),
            Point(cb.low.x, cb.high.y, cb.low.z),
            Point(cb.low.x, cb.low.y, cb.high.z),
            Point(cb.high.x, cb.low.y, cb.high.z),
            Point(cb.high.x, cb.low.y, cb.low.z),
            Point(cb.low.x, cb.low.y, cb.low.z),
        ]
        # 2. transform each of the corners by the object's transformation matrix
        corners = [cast(Point, child.transform * c) for c in corners]

        # 3. generate the new AABB coords for the transformed shape
        min_x, min_y, min_z = map(min, zip(*[(c.x, c.y, c.z) for c in corners]))
        max_x, max_y, max_z = map(max, zip(*[(c.x, c.y, c.z) for c in corners]))

        # 4. integrate this into the current group bounds
        if len(self.children) == 1:
            # This is the first child, so just set the bb for this object
            self.bounds = Bounds(Point(min_x, min_y, min_z), Point(max_x, max_y, max_z))
        else:
            self.bounds = Bounds(
                Point(
                    min(min_x, self.bounds.low.x),
                    min(min_y, self.bounds.low.y),
                    min(min_z, self.bounds.low.z),
                ),
                Point(
                    max(max_x, self.bounds.high.x),
                    max(max_y, self.bounds.high.y),
                    max(max_z, self.bounds.high.z),
                ),
            )

    def optimize(self, _depth: int = MAX_DEPTH) -> Group:
        """Optimizes the current group to ensure that we can render objects in a
        reasonable length of time.  We do this by limiting the number of objects
        within a group to MAX_CHILDREN and if this is overstepped, then we separate
        the group into two sub-groups along the longest axis"""

        # Bail if we've gone too far down the rabbit hole...
        if _depth <= 0:
            return self

        # If we don't have too many children, just optimize any child groups
        if len(self.children) <= MAX_CHILDREN:
            for i, child in enumerate(self.children):
                if isinstance(child, Group):
                    self.children[i] = child.optimize(_depth - 1)
            return self

        # First, get the largest dimension
        xlen = self.bounds.high.x - self.bounds.low.x
        ylen = self.bounds.high.y - self.bounds.low.y
        zlen = self.bounds.high.z - self.bounds.low.z

        max_len = max(xlen, ylen, zlen)
        split_axis = "x" if xlen == max_len else "y" if ylen == max_len else "z"

        # Initialise test bounds so that the linter is satisfied...
        test_bounds = Bounds(Point(0, 0, 0), Point(0, 0, 0))

        match split_axis:
            case "x":
                mid = (self.bounds.low.x + self.bounds.high.x) / 2
                test_bounds = Bounds(
                    self.bounds.low, Point(mid, self.bounds.high.y, self.bounds.high.z)
                )

            case "y":
                mid = (self.bounds.low.y + self.bounds.high.y) / 2
                test_bounds = Bounds(
                    self.bounds.low, Point(self.bounds.high.x, mid, self.bounds.high.z)
                )

            case "z":
                mid = (self.bounds.low.z + self.bounds.high.z) / 2
                test_bounds = Bounds(
                    self.bounds.low, Point(self.bounds.high.x, self.bounds.high.y, mid)
                )

        low_group = Group()
        high_group = Group()

        # Split all children into correct sub-groups
        for child in self.children:
            if child.is_in(test_bounds, full_containment=False):
                low_group.add_child(child)
            else:
                high_group.add_child(child)

        low_group = low_group.optimize(_depth - 1)
        high_group = high_group.optimize(_depth - 1)

        g = Group()

        if len(low_group.children) > 0:
            g.add_child(low_group)

        if len(high_group.children) > 0:
            g.add_child(high_group)

        return g
