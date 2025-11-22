import ray_tracer.patterns as Patterns
from ray_tracer.classes.colour import Colour, Colours
from ray_tracer.classes.matrix import Matrix
from ray_tracer.classes.point import Point
from ray_tracer.classes.transforms import Transforms
from ray_tracer.objects.sphere import Sphere


class TestPatterns:
    class TestPatterns:
        def test_a_pattern_has_a_transformation(self) -> None:
            pattern = Patterns.TestPattern()

            assert pattern.transform == Matrix.Identity()

        def test_a_pattern_can_have_a_transformation_assigned(self) -> None:
            pattern = Patterns.TestPattern()
            pattern.set_transform(Transforms.translation(1, 2, 3))

            assert pattern.transform == Transforms.translation(1, 2, 3)

        def test_pattern_with_an_object_transformation(self) -> None:
            o = Sphere()
            o.set_transform(Transforms.scaling(2, 2, 2))
            p = Patterns.TestPattern()

            assert p.colour_at_object(o, Point(2, 3, 4)) == Colour(1, 1.5, 2)

        def test_pattern_with_a_pattern_transformation(self) -> None:
            o = Sphere()
            p = Patterns.TestPattern()
            p.set_transform(Transforms.scaling(2, 2, 2))

            assert p.colour_at_object(o, Point(2, 3, 4)) == Colour(1, 1.5, 2)

        def test_stripes_with_both_an_object_and_pattern_transform(self) -> None:
            o = Sphere()
            o.set_transform(Transforms.scaling(2, 2, 2))
            p = Patterns.Stripes(Colours.WHITE, Colours.BLACK)
            p.set_transform(Transforms.translation(0.5, 0, 0))

            assert p.colour_at_object(o, Point(2.5, 0, 0)) == Colours.WHITE

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

    class TestGradient:
        def test_a_gradient_linearly_interpolates_between_colours(self) -> None:
            pattern = Patterns.Gradient(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0.25, 0, 0)) == Colour(0.375, 0.375, 0.375)
            assert pattern.colour_at(Point(0.5, 0, 0)) == Colour(0.25, 0.25, 0.25)
            assert pattern.colour_at(Point(0.75, 0, 0)) == Colour(0.125, 0.125, 0.125)

    class TestRings:
        def test_a_ring_should_extend_in_both_x_and_z(self) -> None:
            pattern = Patterns.Rings(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(1, 0, 0)) == Colours.BLACK
            assert pattern.colour_at(Point(0, 0, 1)) == Colours.BLACK
            assert pattern.colour_at(Point(0.708, 0, 0.708)) == Colours.BLACK

    class TestCheckerboard:
        def test_checkers_repeat_in_x(self) -> None:
            pattern = Patterns.Checkerboard(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0.99, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(1.01, 0, 0)) == Colours.BLACK

        def test_checkers_repeat_in_y(self) -> None:
            pattern = Patterns.Checkerboard(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 0.99, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 1.01, 0)) == Colours.BLACK

        def test_checkers_repeat_in_z(self) -> None:
            pattern = Patterns.Checkerboard(Colours.WHITE, Colours.BLACK)

            assert pattern.colour_at(Point(0, 0, 0)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 0, 0.99)) == Colours.WHITE
            assert pattern.colour_at(Point(0, 0, 1.01)) == Colours.BLACK
