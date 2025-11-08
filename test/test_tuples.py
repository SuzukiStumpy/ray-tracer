import math
from typing import cast

import pytest

import ray_tracer.constants as const
from ray_tracer.base_tuple import Tuple
from ray_tracer.point import Point
from ray_tracer.vector import Vector


class TestTuples:
    @pytest.mark.parametrize(
        "x,y,z,w,klass",
        [
            (4.3, -4.2, 3.1, 1.0, Point),
            (4.3, -4.2, 3.1, 0.0, Vector),
        ],
        ids=[
            "Tuple with w=1.0 is a Point",
            "Tuple with w=0.0 is a Vector",
        ],
    )
    def test_tuple_construction(
        self, x: float, y: float, z: float, w: float, klass: type
    ) -> None:
        a: Point = cast(Point, Tuple(x, y, z, w))

        assert math.isclose(a.x, x, abs_tol=const.EPSILON)
        assert math.isclose(a.y, y, abs_tol=const.EPSILON)
        assert math.isclose(a.z, z, abs_tol=const.EPSILON)
        assert math.isclose(a.w, w, abs_tol=const.EPSILON)
        assert type(a) is klass

    def test_points_are_tuples_with_w_1(self) -> None:
        a: Point = Point(4, -4, 3)
        b: object = Tuple(4, -4, 3, 1)

        assert a == b

    def test_vectors_are_tuples_with_w_0(self) -> None:
        a: Vector = Vector(4, -4, 3)
        b = Tuple(4, -4, 3, 0)

        assert a == b
