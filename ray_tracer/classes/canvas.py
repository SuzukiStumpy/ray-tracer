import numpy as np
from PIL import Image

from .colour import Colour


class Canvas:
    """Defines a 2D canvas of pixels"""

    def __init__(self, width: int, height: int) -> None:
        self.pixels = np.zeros((height, width, 3), dtype=np.float64)
        self.width = width
        self.height = height

    def get_pixel(self, x: int, y: int) -> Colour:
        (r, g, b) = self.pixels[y, x]
        return Colour(r, g, b)

    def set_pixel(self, x: int, y: int, value: Colour) -> None:
        self.pixels[y, x] = [value.r, value.g, value.b]

    def to_image(self) -> Image.Image:
        # Convert the pixels array to integers in the range 0-255
        img = (self.pixels * 255).round(0).astype(np.uint8)
        return Image.fromarray(img, "RGB")
