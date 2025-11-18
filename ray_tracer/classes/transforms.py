import math

from ray_tracer.classes.matrix import Matrix


class Transforms:
    """A class of static factories that generate standard transformation Matrices"""

    @staticmethod
    def translation(x: float, y: float, z: float) -> Matrix:
        m = Matrix.Identity()
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return m

    @staticmethod
    def scaling(x: float, y: float, z: float) -> Matrix:
        m = Matrix.Identity()
        m[0, 0] = x
        m[1, 1] = y
        m[2, 2] = z
        return m

    @staticmethod
    def rotation_x(angle: float) -> Matrix:
        m = Matrix.Identity()
        m[1, 1] = math.cos(angle)
        m[1, 2] = -math.sin(angle)
        m[2, 1] = math.sin(angle)
        m[2, 2] = math.cos(angle)
        return m

    @staticmethod
    def rotation_y(angle: float) -> Matrix:
        m = Matrix.Identity()
        m[0, 0] = math.cos(angle)
        m[0, 2] = math.sin(angle)
        m[2, 0] = -math.sin(angle)
        m[2, 2] = math.cos(angle)
        return m

    @staticmethod
    def rotation_z(angle: float) -> Matrix:
        m = Matrix.Identity()
        m[0, 0] = math.cos(angle)
        m[0, 1] = -math.sin(angle)
        m[1, 0] = math.sin(angle)
        m[1, 1] = math.cos(angle)
        return m

    @staticmethod
    def shearing(
        xy: float, xz: float, yx: float, yz: float, zx: float, zy: float
    ) -> Matrix:
        m = Matrix.Identity()
        m[0, 1] = xy
        m[0, 2] = xz
        m[1, 0] = yx
        m[1, 2] = yz
        m[2, 0] = zx
        m[2, 1] = zy
        return m
