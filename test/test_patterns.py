import ray_tracer.patterns as Patterns
from ray_tracer.classes.colour import Colours
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.objects.sphere import Sphere


class TestPatterns:
    class TestStripes:
        def test_creating_a_stripe_pattern(self) -> None:
            pattern = Patterns.Stripes(Colours.WHITE, Colours.BLACK)

            assert pattern.a == Colours.WHITE
            assert pattern.b == Colours.BLACK

        def test_a_stripe_pattern_is_constant_in_y(self) -> None:
            pattern = Patterns.Stripes(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 1, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 2, 0)) == Colours.WHITE

        def test_a_stripe_pattern_is_constant_in_z(self) -> None:
            pattern = Patterns.Stripes(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 0, 1)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 0, 2)) == Colours.WHITE

        def test_a_stripe_pattern_alternates_in_x(self) -> None:
            pattern = Patterns.Stripes(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0.9, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(1, 0, 0)) == Colours.BLACK
            assert pattern.colour_at(Point(-0.1, 0, 0)) == Colours.BLACK
            assert pattern.colour_at(Point(-1, 0, 0)) == Colours.BLACK
            assert pattern.colour_at(Point(-1.1, 0, 0)) == Colours.WHITE

        def test_stripes_with_an_object_transformation(self) -> None:
            o = Sphere()
            o.set_transform(Transforms.scaling(2, 2, 2))
            p = Patterns.Stripes(Colours.WHITE, Colours.BLACK)

            assert p.colour_at_object(o, Point(1.5, 0, 0)) == Colours.WHITE

        def test_stripes_with_a_pattern_transformation(self) -> None:
            o = Sphere()
            p = Patterns.Stripes(Colours.WHITE, Colours.BLACK)
            p.set_transform(Transforms.scaling(2, 2, 2))

            assert p.colour_at_object(o, Point(1.5, 0, 0)) == Colours.WHITE

        def test_stripes_with_both_an_object_and_pattern_transform(self) -> None:
            o = Sphere()
            o.set_transform(Transforms.scaling(2, 2, 2))
            p = Patterns.Stripes(Colours.WHITE, Colours.BLACK)
            p.set_transform(Transforms.translation(0.5, 0, 0))

            assert p.colour_at_object(o, Point(2.5, 0, 0)) == Colours.WHITE
