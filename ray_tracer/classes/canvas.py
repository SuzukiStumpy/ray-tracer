from .colour import Colour, Colours


class Canvas:
    """Defines a 2D canvas of pixels"""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = [[Colours.BLACK.value] * width] * height

    def pixel(self, x: int, y: int) -> Colour:
        return self.pixels[y][x]
