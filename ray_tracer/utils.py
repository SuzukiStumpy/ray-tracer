"""Various utility functions used elsewhere"""


def clamp(value: float, lower: float, upper: float) -> float:
    """Restricts a floating point value to between two bounds."""
    return max(min(value, upper), lower)
