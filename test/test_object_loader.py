from pathlib import Path

import pytest

import ray_tracer.objects.loader as Loader
from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.group import Group


class TestObjectLoader:
    def test_loader_handles_invalid_file_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            Loader.parse_obj_file(Path("this/path/does/not/exist.obj"))

    def test_loader_ignores_unrecognised_lines(self) -> None:
        loader = Loader.parse_obj_file(Path("test/inputs/dummy.obj"))

        assert loader.ignored == 5

    def test_loader_parses_vertex_records(self) -> None:
        loader = Loader.parse_obj_file(Path("test/inputs/vertex_parse_test.obj"))

        assert loader.verts[0] == Point(-1, 1, 0)
        assert loader.verts[1] == Point(-1, 0.5, 0)
        assert loader.verts[2] == Point(1, 0, 0)
        assert loader.verts[3] == Point(1, 1, 0)

    def test_loader_parses_face_records(self) -> None:
        loader = Loader.parse_obj_file(Path("test/inputs/face_parse_test.obj"))

        g = loader.default_group
        t1 = g.children[0]
        t2 = g.children[1]

        assert t1.verts[0] == loader.verts[0]
        assert t1.verts[1] == loader.verts[1]
        assert t1.verts[2] == loader.verts[2]
        assert t2.verts[0] == loader.verts[0]
        assert t2.verts[1] == loader.verts[2]
        assert t2.verts[2] == loader.verts[3]

    def test_loader_triangulates_polygons(self) -> None:
        loader = Loader.parse_obj_file(Path("test/inputs/poly_parse_test.obj"))

        g = loader.default_group
        t1 = g.children[0]
        t2 = g.children[1]
        t3 = g.children[2]

        assert t1.verts[0] == loader.verts[0]
        assert t1.verts[1] == loader.verts[1]
        assert t1.verts[2] == loader.verts[2]
        assert t2.verts[0] == loader.verts[0]
        assert t2.verts[1] == loader.verts[2]
        assert t2.verts[2] == loader.verts[3]
        assert t3.verts[0] == loader.verts[0]
        assert t3.verts[1] == loader.verts[3]
        assert t3.verts[2] == loader.verts[4]

    def test_loader_inserts_groups(self) -> None:
        loader = Loader.parse_obj_file(Path("test/inputs/named_groups_test.obj"))

        g1 = loader.groups["FirstGroup"]
        g2 = loader.groups["SecondGroup"]
        t1 = g1.children[0]
        t2 = g2.children[0]

        assert t1.verts[0] == loader.verts[0]
        assert t1.verts[1] == loader.verts[1]
        assert t1.verts[2] == loader.verts[2]
        assert t2.verts[0] == loader.verts[0]
        assert t2.verts[1] == loader.verts[2]
        assert t2.verts[2] == loader.verts[3]

    def test_loader_outputs_all_loaded_groups_to_a_single_hierarchy(self) -> None:
        loader = Loader.parse_obj_file(Path("test/inputs/named_groups_test.obj"))

        g: Group = loader.obj_to_group()

        assert g.children[0] == loader.groups["FirstGroup"]
        assert g.children[1] == loader.groups["SecondGroup"]

    def test_loader_suuprts_vertex_normal_records(self) -> None:
        loader = Loader.parse_obj_file(Path("test/inputs/vertex_normal_test.obj"))

        assert loader.normals[0] == Vector(0, 0, 1)
        assert loader.normals[1] == Vector(0.707, 0, -0.707)
        assert loader.normals[2] == Vector(1, 2, 3)

    def test_faces_with_normals(self) -> None:
        loader = Loader.parse_obj_file(
            Path("test/inputs/faces_with_vertex_normals_test.obj")
        )

        g: Group = loader.obj_to_group()
        t1 = g.children[0]
        t2 = g.children[1]

        assert t1.verts[0] == loader.verts[0]
        assert t1.verts[1] == loader.verts[1]
        assert t1.verts[2] == loader.verts[2]
        assert t1.normals[0] == loader.normals[2]
        assert t1.normals[1] == loader.normals[0]
        assert t2.normals[2] == loader.normals[1]
        assert t1 == t2
