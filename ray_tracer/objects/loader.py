"""Object parser for Alias/Wavefront obj format files"""

from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard

from ray_tracer.classes.point import Point
from ray_tracer.classes.vector import Vector
from ray_tracer.objects.group import Group
from ray_tracer.objects.smooth_triangle import SmoothTriangle
from ray_tracer.objects.triangle import Triangle


# If we read a malformed vertex definition, then we raise this error
class InvalidVertexCommandError(RuntimeError):
    pass


# If we read a malformed face definition, then we raise this error
class InvalidFaceCommandError(RuntimeError):
    pass


# If we read a malformed vertex normal definition, then we raise this error
class InvalidNormalCommandError(RuntimeError):
    pass


@dataclass
class Loader:
    ignored: int = 0
    verts: list[Point] = field(default_factory=list)
    normals: list[Vector] = field(default_factory=list)
    default_group: Group = field(default_factory=lambda: Group())
    groups: dict[str, Group] = field(default_factory=dict)

    def obj_to_group(self) -> Group:
        """Return a single group object which contains all the face data from
        the loaded object file"""
        # Make a copy of the default group so we don't mutate it (note, we don't need
        # to use deep copy since we'll only be mutating the group itself)
        g = copy(self.default_group)

        for key in self.groups.keys():
            g.add_child(self.groups[key])

        # For now, this isn't functioning...
        g = g.optimize()

        return g


def parse_obj_file(filepath: Path) -> Loader:
    """Loads and parses the file.  If the path is invalid then raises
    a FileNotFoundError exception"""

    loader = Loader()
    latest_group = ""

    with filepath.open("r", encoding="utf-8") as f:
        objdata = [line.strip() for line in f.readlines()]

    for command, *params in (line.split(" ") for line in objdata):
        match command.lower():
            case "v":
                if len(params) != 3:
                    raise InvalidVertexCommandError

                loader.verts.append(
                    Point(float(params[0]), float(params[1]), float(params[2]))
                )

            case "f":
                # Face data takes three possible forms:
                #   integer:  Vertex index
                #   integer/integer/integer:
                #               Vertex index / Texture vertex / Vertex normal index
                #   integer//integer:  Same as above but with no texture vertex

                # Face data parameters are 1 based indexes, so we need to subtract 1
                # from each when we perform the lookup into the vertex array
                if len(params) < 3:
                    raise InvalidFaceCommandError

                parms: list[tuple[int, int | None, int | None]] = []

                for param in params:
                    if "/" in param:
                        v, t, n = param.split("/")
                        parms.append(
                            (
                                int(v),
                                int(t) if t != "" else None,
                                int(n) if n != "" else None,
                            )
                        )
                    else:
                        parms.append((int(param), None, None))

                verts = [x[0] for x in parms]
                # tex = [x[1] for x in parms]    # We don't currently use texture parms
                norms = [x[2] for x in parms]

                if min(verts) < 1 or max(verts) > len(loader.verts):
                    raise IndexError("Vertex index out of range")

                if latest_group != "" and latest_group not in loader.groups:
                    loader.groups[latest_group] = Group()

                if is_all_ints(norms):
                    if min(norms) < 1 or max(norms) > len(loader.normals):
                        raise IndexError("Vertex normal index out of range")

                for t in fan_triangulation(verts, loader.verts, norms, loader.normals):
                    loader.default_group.add_child(
                        t
                    ) if latest_group == "" else loader.groups[latest_group].add_child(
                        t
                    )

            case "g":
                latest_group = params[0]

            case "vn":
                if len(params) != 3:
                    raise InvalidNormalCommandError

                loader.normals.append(
                    Vector(float(params[0]), float(params[1]), float(params[2]))
                )

            case _:
                loader.ignored += 1

    return loader


def is_all_ints(lst: list[int | None]) -> TypeGuard[list[int]]:
    return None not in lst


def fan_triangulation(
    indexes: list[int],
    verts: list[Point],
    normal_idx: list[int | None],
    normals: list[Vector],
) -> list[Triangle | SmoothTriangle]:
    tris = []

    # If we have vertex normals in play
    if is_all_ints(normal_idx):
        for idx in range(1, len(indexes) - 1):
            tris.append(
                SmoothTriangle(
                    verts[indexes[0] - 1],
                    verts[indexes[idx] - 1],
                    verts[indexes[idx + 1] - 1],
                    normals[normal_idx[0] - 1],
                    normals[normal_idx[idx] - 1],
                    normals[normal_idx[idx + 1] - 1],
                )
            )
    else:
        for idx in range(1, len(indexes) - 1):
            tris.append(
                Triangle(
                    verts[indexes[0] - 1],
                    verts[indexes[idx] - 1],
                    verts[indexes[idx + 1] - 1],
                )
            )

    return tris
