from itertools import product

from ray_tracer.classes.canvas import Canvas
from ray_tracer.classes.colour import Colours


class TestCanvas:
    def test_canvas_has_a_width_and_a_height(self) -> None:
        c = Canvas(10, 20)

        assert c.width == 10
        assert c.height == 20

    def test_canvas_is_initially_all_black(self) -> None:
        c = Canvas(10, 20)

        for x, y in product(range(c.width), range(c.height)):
            c.pixel(x, y) == Colours.BLACK
