import math
from typing import cast

import pytest

import ray_tracer.constants as const
from ray_tracer.classes.colour import Colour


class TestColours:
    def test_colours_are_red_green_blue_tuples(self) -> None:
        c = Colour(-0.5, 0.4, 1.7)

        assert c.red == -0.5
        assert c.green == 0.4
        assert c.blue == 1.7

    def test_rgba_are_synonyms_for_red_green_blue_alpha(self) -> None:
        c = Colour(-0.5, 0.1, 1.7)

        assert c.r == c.red
        assert c.g == c.green
        assert c.b == c.blue
        assert c.a == c.alpha
