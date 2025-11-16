import math
from typing import Collection, cast

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

    def test_colours_can_be_added_together(self) -> None:
        c1 = Colour(0.9, 0.6, 0.75)
        c2 = Colour(0.7, 0.1, 0.25)

        assert c1 + c2 == Colour(1.6, 0.7, 1.0)

    def test_colours_can_be_subtracted(self) -> None:
        c1 = Colour(0.9, 0.6, 0.75)
        c2 = Colour(0.7, 0.1, 0.25)

        assert c1 - c2 == Colour(0.2, 0.5, 0.5)

    def test_colours_can_be_multiplied_by_a_scalar(self) -> None:
        c = Colour(0.2, 0.3, 0.4)

        assert 2 * c == Colour(0.4, 0.6, 0.8)
        # ... and also testing commutativity
        assert c * 2 == Colour(0.4, 0.6, 0.8)

    def test_two_colours_can_be_multiplied_together(self) -> None:
        c1 = Colour(1, 0.2, 0.4)
        c2 = Colour(0.9, 1, 0.1)

        assert c1 * c2 == Colour(0.9, 0.2, 0.04)
