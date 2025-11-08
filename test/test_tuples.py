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

        assert a == Tuple(4, -4, 3, 1)

    def test_vectors_are_tuples_with_w_0(self) -> None:
        a: Vector = Vector(4, -4, 3)

        assert a == Tuple(4, -4, 3, 0)

    def test_adding_two_tuples(self) -> None:
        a1 = Tuple(3, -2, 5, 1)
        a2 = Tuple(-2, 3, 1, 0)

        assert a1 + a2 == Tuple(1, 1, 6, 1)

    def test_adding_point_to_vector_gives_new_point(self) -> None:
        a1 = Point(1, 2, 3)
        a2 = Vector(1, 2, 3)

        assert a1 + a2 == Point(2, 4, 6)

    def test_adding_vector_to_vector_gives_new_vector(self) -> None:
        a1 = Vector(1, 2, 3)
        a2 = Vector(1, 2, 3)

        assert a1 + a2 == Vector(2, 4, 6)

    def test_cannot_add_two_points(self) -> None:
        a1 = Point(3, -2, 5)
        a2 = Point(3, -2, 1)

        with pytest.raises(TypeError, match="adding two points is unsupported"):
            a1 + a2

    def test_subtracting_two_points_gives_vector(self) -> None:
        a1 = Point(3, 2, 1)
        a2 = Point(5, 6, 7)

        assert a1 - a2 == Vector(-2, -4, -6)

    def test_subtracting_vector_from_point_gives_point(self) -> None:
        a1 = Point(3, 2, 1)
        a2 = Vector(5, 6, 7)

        assert a1 - a2 == Point(-2, -4, -6)

    def test_subtracting_two_vectors_gives_vector(self) -> None:
        a1 = Vector(3, 2, 1)
        a2 = Vector(5, 6, 7)

        assert a1 - a2 == Vector(-2, -4, -6)

    def test_cannot_subtract_point_from_vector(self) -> None:
        a1 = Vector(3, 2, 1)
        a2 = Point(5, 6, 7)

        with pytest.raises(
            TypeError,
            match="unsupported operation 'Vector' - 'Point'",
        ):
            a1 - a2

    def test_subtracting_vector_from_zero_vector_yields_inverse(self) -> None:
        zero = Vector(0, 0, 0)
        v = Vector(1, -2, 3)

        assert zero - v == Vector(-1, 2, -3)

    def test_a_tuple_can_be_negated(self) -> None:
        a = Tuple(1, -2, 3, -4)

        assert -a == Tuple(-1, 2, -3, 4)

    def test_a_vector_can_be_negated(self) -> None:
        a = Vector(1, -2, 3)

        assert -a == Vector(-1, 2, -3)

    # This one probably doesn't make too much sense
    # (point reflects around X, Y and Z axes)
    def test_a_point_can_be_negated(self) -> None:
        a = Point(1, -2, 3)

        assert -a == Point(-1, 2, -3)

    def test_scalar_multiplication_of_a_tuple(self) -> None:
        a = Tuple(1, -2, 3, -4)

        assert a * 3.5 == Tuple(3.5, -7, 10.5, -14)

        # Also test that it works the other way around (using __rmul__)
        assert 3.5 * a == Tuple(3.5, -7, 10.5, -14)

    def test_scalar_multiplication_by_a_fractional_amount(self) -> None:
        a = Tuple(1, -2, 3, -4)

        assert a * 0.5 == Tuple(0.5, -1, 1.5, -2)
        assert 0.5 * a == Tuple(0.5, -1, 1.5, -2)

    def test_scalar_division(self) -> None:
        a = Tuple(1, -2, 3, -4)

        assert a / 2 == Tuple(0.5, -1, 1.5, -2)
