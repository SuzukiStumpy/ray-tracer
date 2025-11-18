import math

import numpy as np
from numpy.linalg import LinAlgError

from ray_tracer.classes.base_tuple import Tuple
from ray_tracer.constants import EPSILON


class Matrix:
    """Defines a basic matrix of a specific shape"""

    def __init__(self, data: list[list[float]]) -> None:
        # validate that the matrix data is square before doing anything else
        it = iter(data)
        length = len(next(it))
        if not all(len(lst) == length for lst in it):
            raise ValueError("Matrix shape is not square")

        self.data = np.array(data)
        self.size = length

    def __getitem__(self, index: tuple[int, int]) -> float:
        if 0 > index[0] >= self.size or 0 > index[1] >= self.size:
            raise IndexError(f"Index [{index[0]}, {index[1]}] is out of bounds")

        return self.data[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False

        if self.size != other.size:
            return False

        return np.allclose(self.data, other.data, atol=EPSILON)

    def __mul__(self, other: object) -> Matrix | Tuple:
        if isinstance(other, Tuple):
            return Tuple(
                *np.matmul(self.data, np.array([other.x, other.y, other.z, other.w]))
            )

        if isinstance(other, Matrix):
            return Matrix(np.matmul(self.data, other.data))

        raise NotImplementedError

    def transpose(self) -> Matrix:
        return Matrix(self.data.transpose().tolist())

    def det(self) -> float:
        """Returns the determinant of a matrix"""
        return np.linalg.det(self.data)

    def sub(self, row: int, col: int) -> Matrix:
        """Returns the submatrix of M with row 'row' and column 'col' removed"""
        if 0 > row >= self.size or 0 > col >= self.size:
            raise IndexError(f"Index [{row}, {col}] is out of bounds")

        # Delete the corresponding row and column from the existing data
        arr = np.delete(self.data, row, 0)
        arr = np.delete(arr, col, 1)

        return Matrix(arr.tolist())

    def minor(self, row: int, col: int) -> float:
        return self.sub(row, col).det()

    def cofactor(self, row: int, col: int) -> float:
        m = self.minor(row, col)
        return m if (row + col) % 2 == 0 else -m

    def is_invertible(self) -> bool:
        return not math.isclose(self.det(), 0, rel_tol=EPSILON)

    def inverse(self) -> Matrix:
        if not self.is_invertible():
            raise LinAlgError("Matrix is not invertible")

        return Matrix(np.linalg.inv(self.data))

    @classmethod
    def Identity(cls, size: int = 4) -> Matrix:
        """Return the identity matrix of a given size (default 4x4)"""
        return Matrix(np.identity(size).tolist())
