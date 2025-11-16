import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from ..constants import EPSILON


@dataclass
class AbstractTuple(ABC):
    """An abstract class representing the basic operations permissible
    by all Tuple based objects"""

    x: float
    y: float
    z: float
    w: float

    def __repr__(self) -> str:
        """Return the string representation of the object.  Override if necessary in
        derived classes"""
        s: str = (
            f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z}, w={self.w})"
        )
        return s

    def __eq__(self, other: object) -> bool:
        """Implementation of equality for two tuple-based types"""
        if not isinstance(other, type(self)):
            return False

        return (
            math.isclose(self.x, other.x, abs_tol=EPSILON)
            and math.isclose(self.y, other.y, abs_tol=EPSILON)
            and math.isclose(self.z, other.z, abs_tol=EPSILON)
            and math.isclose(self.w, other.w, abs_tol=EPSILON)
        )

    @abstractmethod
    def __add__(self, other: AbstractTuple) -> AbstractTuple:
        """Define addition between two types derived from AbstractTuple"""

    @abstractmethod
    def __sub__(self, other: AbstractTuple) -> AbstractTuple:
        """Define subtraction of two types derived from AbstractTuple"""

    @abstractmethod
    def __neg__(self) -> Self:
        """Define unary negation for a type derived from AbstractTuple"""

    @abstractmethod
    def __mul__(self, other: float) -> Self:
        """Define multiplication by scalar in the form:  s * Self"""

    def __rmul__(self, other: float) -> Self:
        """Define multiplication by scalar in form:  Self * s"""
        return self * other

    @abstractmethod
    def __truediv__(self, other: float) -> Self:
        """Define division by a scalar value for a type derived from AbstractTuple"""
