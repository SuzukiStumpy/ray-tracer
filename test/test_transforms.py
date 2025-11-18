import math

import pytest

from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.classes.vector import Vector


class TestTransformations:
    class TestTranslation:
        def test_multiplying_by_a_translation_matrix(self) -> None:
            t = Transforms.translation(5, -3, 2)
            p = Point(-3, 4, 5)

            assert t * p == Point(2, 1, 7)

        def test_multiplying_by_the_inverse_of_a_translation_matrix(self) -> None:
            t = Transforms.translation(5, -3, 2)
            inv = t.inverse()
            p = Point(-3, 4, 5)

            assert inv * p == Point(-8, 7, 3)

        def test_translation_does_not_affect_vectors(self) -> None:
            t = Transforms.translation(5, -3, 2)
            v = Vector(-3, 4, 5)

            assert t * v == v

    class TestScaling:
        def test_scaling_matrix_applied_to_a_point(self) -> None:
            t = Transforms.scaling(2, 3, 4)
            p = Point(-4, 6, 8)

            assert t * p == Point(-8, 18, 32)

        def test_scaling_matrix_applied_to_a_vector(self) -> None:
            t = Transforms.scaling(2, 3, 4)
            v = Vector(-4, 6, 8)

            assert t * v == Vector(-8, 18, 32)

        def test_multiplying_by_the_inverse_of_a_scaling_matrix(self) -> None:
            t = Transforms.scaling(2, 3, 4)
            inv = t.inverse()
            v = Vector(-4, 6, 8)

            assert inv * v == Vector(-2, 2, 2)

        def test_reflection_is_scaling_by_a_negative_value(self) -> None:
            t = Transforms.scaling(-1, 1, 1)
            p = Point(2, 3, 4)

            assert t * p == Point(-2, 3, 4)

    class TestRotation:
        def test_rotating_a_point_around_the_x_axis(self) -> None:
            p = Point(0, 1, 0)
            half_quarter = Transforms.rotation_x(math.pi / 4)
            full_quarter = Transforms.rotation_x(math.pi / 2)

            assert half_quarter * p == Point(0, math.sqrt(2) / 2, math.sqrt(2) / 2)
            assert full_quarter * p == Point(0, 0, 1)

        def test_inverse_rotation_rotates_opposite_direction(self) -> None:
            p = Point(0, 1, 0)
            half_quarter = Transforms.rotation_x(math.pi / 4)
            inv = half_quarter.inverse()

            assert inv * p == Point(0, math.sqrt(2) / 2, -math.sqrt(2) / 2)

        def test_rotating_a_point_around_the_y_axis(self) -> None:
            p = Point(0, 0, 1)
            half_quarter = Transforms.rotation_y(math.pi / 4)
            full_quarter = Transforms.rotation_y(math.pi / 2)

            assert half_quarter * p == Point(math.sqrt(2) / 2, 0, math.sqrt(2) / 2)
            assert full_quarter * p == Point(1, 0, 0)

        def test_rotating_a_point_around_the_z_axis(self) -> None:
            p = Point(0, 1, 0)
            half_quarter = Transforms.rotation_z(math.pi / 4)
            full_quarter = Transforms.rotation_z(math.pi / 2)

            assert half_quarter * p == Point(-math.sqrt(2) / 2, math.sqrt(2) / 2, 0)
            assert full_quarter * p == Point(-1, 0, 0)

    class TestShearing:
        @pytest.mark.parametrize(
            "input,expected",
            [
                (Transforms.shearing(1, 0, 0, 0, 0, 0), Point(5, 3, 4)),
                (Transforms.shearing(0, 1, 0, 0, 0, 0), Point(6, 3, 4)),
                (Transforms.shearing(0, 0, 1, 0, 0, 0), Point(2, 5, 4)),
                (Transforms.shearing(0, 0, 0, 1, 0, 0), Point(2, 7, 4)),
                (Transforms.shearing(0, 0, 0, 0, 1, 0), Point(2, 3, 6)),
                (Transforms.shearing(0, 0, 0, 0, 0, 1), Point(2, 3, 7)),
            ],
            ids=[
                "X in proportion to Y",
                "X in proportion to Z",
                "Y in proportion to X",
                "Y in proportion to Z",
                "Z in proportion to X",
                "Z in proportion to Y",
            ],
        )
        def test_shearing_movements(self, input: Matrix, expected: Point) -> None:
            t = input
            p = Point(2, 3, 4)

            assert t * p == expected
