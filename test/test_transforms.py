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
