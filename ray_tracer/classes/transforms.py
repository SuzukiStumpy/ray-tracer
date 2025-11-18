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
