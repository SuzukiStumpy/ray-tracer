import math

import pytest

from ray_tracer.classes.base_tuple import Tuple
from ray_tracer.classes.matrix import Matrix
from ray_tracer.constants import EPSILON


class TestMatrix:
    def test_4x4_matrix_can_be_created_and_referenced(self) -> None:
        m = Matrix(
            [
                [1, 2, 3, 4],
                [5.5, 6.5, 7.5, 8.5],
                [9, 10, 11, 12],
                [13.5, 14.5, 15.5, 16.5],
            ],
        )

        assert m[0, 0] == 1
        assert m[0, 3] == 4
        assert m[1, 0] == 5.5
        assert m[1, 2] == 7.5
        assert m[2, 2] == 11
        assert m[3, 0] == 13.5
        assert m[3, 2] == 15.5

    def test_2x2_matrix_can_be_created_and_referenced(self) -> None:
        m = Matrix(
            [
                [-3, 5],
                [1, -2],
            ]
        )

        assert m[0, 0] == -3
        assert m[0, 1] == 5
        assert m[1, 0] == 1
        assert m[1, 1] == -2

    def test_3x3_matrix_can_be_created_and_referenced(self) -> None:
        m = Matrix(
            [
                [-3, 5, 0],
                [1, -2, -7],
                [0, 1, 1],
            ]
        )

        assert m[0, 0] == -3
        assert m[1, 1] == -2
        assert m[2, 2] == 1

    @pytest.mark.parametrize(
        "A,B,expected",
        [
            (
                Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]]),
                Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]]),
                True,
            ),
            (
                Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]]),
                Matrix([[2, 3, 4, 5], [6, 7, 8, 9], [8, 7, 6, 5], [4, 3, 2, 1]]),
                False,
            ),
        ],
        ids=["Equality with same matrices", "Inequality with different matrices"],
    )
    def test_matrix_equality_with_identical_matrices(
        self,
        A: Matrix,
        B: Matrix,
        expected: bool,
    ) -> None:
        assert (A == B) is expected

    def test_matrix_multiplication(self) -> None:
        a = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
        b = Matrix([[-2, 1, 2, 3], [3, 2, 1, -1], [4, 3, 6, 5], [1, 2, 7, 8]])

        assert a * b == Matrix(
            [[20, 22, 50, 48], [44, 54, 114, 108], [40, 58, 110, 102], [16, 26, 46, 42]]
        )

    def test_matrix_multiplied_by_a_tuple(self) -> None:
        a = Matrix([[1, 2, 3, 4], [2, 4, 4, 2], [8, 6, 4, 1], [0, 0, 0, 1]])
        b = Tuple(1, 2, 3, 1)

        assert a * b == Tuple(18, 24, 33, 1)

    def test_matrix_multiplied_by_identity_matrix_is_itself(self) -> None:
        a = Matrix([[0, 1, 2, 3], [1, 2, 4, 8], [2, 4, 8, 16], [4, 8, 16, 32]])

        assert a * Matrix.Identity() == a

    def test_matrix_transposition(self) -> None:
        a = Matrix([[0, 9, 3, 0], [9, 8, 0, 8], [1, 8, 5, 3], [0, 0, 5, 8]])

        assert a.transpose() == Matrix(
            [[0, 9, 1, 0], [9, 8, 8, 0], [3, 0, 5, 5], [0, 8, 3, 8]]
        )

    def test_transposing_the_identity_matrix_yields_itself(self) -> None:
        a = Matrix.Identity()

        assert a.transpose() == a

    def test_calculating_the_determinant_of_a_2x2_matrix(self) -> None:
        a = Matrix([[1, 5], [-3, 2]])

        assert a.det() == 17

    def test_a_submatrix_of_a_3x3_matrix_is_a_2x2_matrix(self) -> None:
        a = Matrix([[1, 5, 0], [-3, 2, 7], [0, 6, -3]])

        assert a.sub(0, 2) == Matrix([[-3, 2], [0, 6]])

    def test_a_submatrix_of_a_4x4_matrix_is_a_3x3_matrix(self) -> None:
        a = Matrix([[-6, 1, 1, 6], [-8, 5, 8, 6], [-1, 0, 8, 2], [-7, 1, -1, 1]])

        assert a.sub(2, 1) == Matrix([[-6, 1, 6], [-8, 8, 6], [-7, -1, 1]])

    def test_calculating_the_minor_of_a_3x3_matrix(self) -> None:
        a = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])
        b = a.sub(1, 0)

        assert math.isclose(b.det(), 25, rel_tol=EPSILON)
        assert math.isclose(a.minor(1, 0), 25, rel_tol=EPSILON)

    def test_calculating_the_cofactor_of_a_3x3_matrix(self) -> None:
        a = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])

        assert math.isclose(a.minor(0, 0), -12, rel_tol=EPSILON)
        assert math.isclose(a.cofactor(0, 0), -12, rel_tol=EPSILON)
        assert math.isclose(a.minor(1, 0), 25, rel_tol=EPSILON)
        assert math.isclose(a.cofactor(1, 0), -25, rel_tol=EPSILON)

    def test_calculating_the_determinant_of_a_3x3_matrix(self) -> None:
        a = Matrix([[1, 2, 6], [-5, 8, -4], [2, 6, 4]])

        assert math.isclose(a.cofactor(0, 0), 56, rel_tol=EPSILON)
        assert math.isclose(a.cofactor(0, 1), 12, rel_tol=EPSILON)
        assert math.isclose(a.cofactor(0, 2), -46, rel_tol=EPSILON)
        assert math.isclose(a.det(), -196, rel_tol=EPSILON)

    def test_calculating_the_determinant_of_a_4x4_matrix(self) -> None:
        a = Matrix([[-2, -8, 3, 5], [-3, 1, 7, 3], [1, 2, -9, 6], [-6, 7, 7, -9]])

        assert math.isclose(a.cofactor(0, 0), 690, rel_tol=EPSILON)
        assert math.isclose(a.cofactor(0, 1), 447, rel_tol=EPSILON)
        assert math.isclose(a.cofactor(0, 2), 210, rel_tol=EPSILON)
        assert math.isclose(a.cofactor(0, 3), 51, rel_tol=EPSILON)
        assert math.isclose(a.det(), -4071, rel_tol=EPSILON)
