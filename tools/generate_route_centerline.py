#!/usr/bin/env python3
"""Generate and validate a dense route centerline from offline snapshots."""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import math
from pathlib import Path
import statistics
import struct
import sys
from typing import Iterable, NamedTuple


TRACK_HEADER_SIZE = 0x150
ROUTE_HEADER_SIZE = 0xE0
TRACK_SECTION_COUNT = 12
ROUTE_SECTION_COUNT = 5
# Version 3 carries the respawn isometry of the start block and of every
# checkpoint block (src/route.h). Version 1 is the tracer dump before the
# centerline is added; version 2 files predate spawn locations and must be
# redumped (tools/redump_routes.py).
ROUTE_VERSION = 3
START_STRIDE = 0x114
TRIGGER_STRIDE = 0x90
RUN_RECORD_SIZE = 1668
RUN_POSITION_OFFSET = 56
RUN_WHEEL_BASE = 356
RUN_WHEEL_SIZE = 328
RUN_WHEEL_CONTACT_OFFSET = 272
GRID_STEP = 1.0
OUTPUT_SPACING = 2.0
HEIGHT_CLUSTER = 0.75
MIN_UP_NORMAL = 0.25
MAX_SEAM_GAP = 4.25
MAX_JUMP_GAP = 20.0
JUMP_ANCHOR_RADIUS = 32.0
CAR_SURFACE_OFFSET = 0.21


class Point(NamedTuple):
    x: float
    y: float
    z: float


class Triangle(NamedTuple):
    a: Point
    b: Point
    c: Point
    material: int


class GridNode(NamedTuple):
    ix: int
    iz: int
    y: float
    material: int


class CenterPoint(NamedTuple):
    position: Point
    arc_length: float
    half_width: float
    leg_index: int


def fail(message: str) -> None:
    raise RuntimeError(message)


def align8(value: int) -> int:
    return (value + 7) & ~7


def distance(a: Point, b: Point) -> float:
    return math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def transform_point(matrix: tuple[float, ...], point: Point) -> Point:
    return Point(
        matrix[0] * point.x + matrix[1] * point.y
        + matrix[2] * point.z + matrix[9],
        matrix[3] * point.x + matrix[4] * point.y
        + matrix[5] * point.z + matrix[10],
        matrix[6] * point.x + matrix[7] * point.y
        + matrix[8] * point.z + matrix[11],
    )


def route_sections(data: bytes) -> list[tuple[int, int, int]]:
    if len(data) < ROUTE_HEADER_SIZE:
        fail("route snapshot is shorter than its header")
    magic, version, endian, header_size, count, file_size = struct.unpack_from(
        "<8sIIIIQ", data, 0)
    if (
        magic != b"TMNFROU1"
        or version not in (1, ROUTE_VERSION)
        or endian != 0x12345678
        or header_size != ROUTE_HEADER_SIZE
        or count != ROUTE_SECTION_COUNT
        or file_size != len(data)
    ):
        fail("invalid route snapshot header")
    sections = [
        struct.unpack_from("<QII", data, 96 + index * 16)
        for index in range(ROUTE_SECTION_COUNT)
    ]
    expected = (0x40, START_STRIDE, TRIGGER_STRIDE, TRIGGER_STRIDE)
    for index, stride in enumerate(expected):
        offset, section_count, actual_stride = sections[index]
        if (
            offset % 8 != 0
            or (section_count == 0 and index != 2)
            or actual_stride != stride
            or offset + section_count * actual_stride > len(data)
        ):
            fail("invalid fixed route section")
    expected_reference_stride = 0x10 if version == 1 else 0x18
    offset, section_count, stride = sections[4]
    if (
        offset % 8 != 0
        or section_count < 2
        or stride != expected_reference_stride
        or offset + section_count * stride > len(data)
    ):
        fail("invalid route reference section")
    if hashlib.sha256(data[ROUTE_HEADER_SIZE:]).digest() != data[176:208]:
        fail("route payload digest does not match")
    return sections


def route_anchor_points(
    data: bytes, sections: list[tuple[int, int, int]]
) -> list[Point]:
    start_offset = sections[1][0]
    start_matrix = struct.unpack_from("<12f", data, start_offset)
    anchors = [Point(*start_matrix[9:12])]
    for section_index in (2, 3):
        offset, count, stride = sections[section_index]
        for index in range(count):
            record = offset + index * stride
            box = struct.unpack_from("<6f", data, record + 16)
            matrix = struct.unpack_from("<12f", data, record + 40)
            anchors.append(transform_point(matrix, Point(*box[:3])))
    return anchors


def reorder_checkpoints_from_ghost(
    data: bytes,
    sections: list[tuple[int, int, int]],
    ghost: list[Point],
) -> bytes:
    offset, count, stride = sections[2]
    if count < 2:
        return data
    anchors = route_anchor_points(data, sections)[1:1 + count]
    passages = []
    for checkpoint, anchor in enumerate(anchors):
        index = min(
            range(len(ghost)),
            key=lambda candidate: distance(ghost[candidate], anchor),
        )
        separation = distance(ghost[index], anchor)
        if separation > 32.0:
            fail(
                f"route ghost stays {separation:.3f} m from "
                f"checkpoint record {checkpoint}")
        passages.append(index)
    if len(set(passages)) != count:
        fail("multiple checkpoint records map to one ghost sample")
    order = sorted(range(count), key=passages.__getitem__)
    if order == list(range(count)) and all(
        struct.unpack_from("<I", data, offset + index * stride)[0] == index
        for index in range(count)
    ):
        return data
    records = [
        bytearray(data[offset + index * stride:offset + (index + 1) * stride])
        for index in order
    ]
    output = bytearray(data)
    for index, record in enumerate(records):
        struct.pack_into("<I", record, 0, index)
        output[offset + index * stride:offset + (index + 1) * stride] = record
    output[176:208] = hashlib.sha256(output[ROUTE_HEADER_SIZE:]).digest()
    return bytes(output)


def read_drive(path: Path) -> list[Point]:
    data = path.read_bytes()
    if len(data) == 0 or len(data) % RUN_RECORD_SIZE != 0:
        fail(f"{path} is not a whole number of {RUN_RECORD_SIZE}-byte records")
    return [
        Point(*struct.unpack_from(
            "<3f", data, offset + RUN_POSITION_OFFSET))
        for offset in range(0, len(data), RUN_RECORD_SIZE)
    ]


class Ghost(NamedTuple):
    positions: list[Point]
    checkpoint_times_ms: list[int]
    sample_period_ms: int


REPLAY_CHALLENGE_CHUNK = 0x03093002
REPLAY_U01_CHUNK = 0x03093007
REPLAY_EVENTS_CHUNK = 0x0309300E
REPLAY_EMPTY_CHUNK = 0x03093011
REPLAY_GHOSTS_CHUNK = 0x03093014
OLD_REPLAY_CHALLENGE_CHUNK = 0x2403F002
OLD_REPLAY_GHOSTS_CHUNK = 0x2403F004
GBX_SKIP_MARKER = 0x534B4950


def replay_ghosts(gbx, ghost_class_ids: list[int]) -> list:
    """Ghost nodes of a CGameCtnReplayRecord.

    pygbx stops at the first chunk it does not know, and the United campaign
    replays come in two shapes it does not finish:

    * TMUF records (0x03093000) carry 0x0309300E (a null
      CCtnMediaBlockEventTrackMania node ref) and the empty 0x03093011 before
      the ghost list 0x03093014. Walk the body up to the list and let pygbx
      resume there; all ghosts parse.
    * TMU records (0x2403F000) keep the ghosts in 0x2403F004, which pygbx
      treats as 16 opaque bytes. Read its header and hand the first ghost to
      pygbx; it parses everything the route needs (samples, race time,
      respawns, checkpoint times) and stops at 0x2401B00D, after which the
      remaining ghosts are unreachable. The first ghost is the fastest one in
      every campaign replay looked at."""
    ghosts = gbx.get_classes_by_ids(ghost_class_ids)
    if ghosts:
        return ghosts
    from pygbx.bytereader import ByteReader

    reader = ByteReader(gbx.data)
    while True:
        chunk_id = reader.read_uint32()
        skippable = reader.read_uint32() == GBX_SKIP_MARKER
        if skippable:
            skip_size = reader.read_uint32()
        else:
            reader.pos -= 4
        if chunk_id in (REPLAY_CHALLENGE_CHUNK, OLD_REPLAY_CHALLENGE_CHUNK):
            reader.skip(reader.read_uint32())
        elif chunk_id == REPLAY_U01_CHUNK:
            reader.skip(4)
        elif chunk_id == REPLAY_EVENTS_CHUNK:
            events = reader.read_int32()
            if events != -1:
                fail(f"replay record carries an events node ({events}); not a TMUF campaign replay")
        elif chunk_id == REPLAY_EMPTY_CHUNK:
            pass
        elif chunk_id == REPLAY_GHOSTS_CHUNK:
            reader.pos -= 4
            gbx._read_node(gbx.class_id, -1, reader, add=False)
            break
        elif chunk_id == OLD_REPLAY_GHOSTS_CHUNK:
            reader.skip(4)                       # chunk version (6)
            reader.skip(4)                       # node array marker (10)
            count = reader.read_uint32()
            if count == 0:
                fail("old replay record has no ghosts")
            index = reader.read_int32()
            if index < 0 or index in gbx.classes:
                fail(f"old replay record ghost 0 is node ref {index}")
            gbx._read_node(reader.read_uint32(), index, reader)
            break
        elif skippable:
            reader.skip(skip_size)
        else:
            fail(f"unknown replay record chunk 0x{chunk_id:08x} at body offset {reader.pos}")
    return gbx.get_classes_by_ids(ghost_class_ids)


def read_ghost(path: Path, optional: bool) -> Ghost | None:
    """Read the fastest no-respawn ghost. With `optional`, a replay without a
    usable ghost (unparsable, or every ghost respawned) yields None and the
    empirical road grid is used instead."""
    try:
        from pygbx import Gbx, GbxType
    except ImportError as error:
        fail(f"pygbx is required to read {path}: {error}")
    ghosts = [
        ghost for ghost in replay_ghosts(
            Gbx(str(path)), [GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
        if ghost.sample_period > 0 and ghost.cp_times
    ]
    if not ghosts:
        if optional:
            print(f"{path.name} has no ghost; using the empirical road grid")
            return None
        fail(f"{path} contains no ghost")
    # Prefer a clean drive; a respawning ghost is still usable when the lap
    # that is kept has no teleport (checked by first_lap).
    ghost = min(ghosts, key=lambda candidate: (candidate.num_respawns, candidate.cp_times[-1]))
    if ghost.num_respawns:
        print(f"{path.name}: fastest ghost respawned {ghost.num_respawns} time(s)")
    count = round(ghost.race_time / ghost.sample_period)
    if count < 2 or count > len(ghost.records):
        fail("route ghost has invalid race-time framing")
    return Ghost(
        [
            Point(record.position.x, record.position.y, record.position.z)
            for record in ghost.records[:count]
        ],
        [int(time) for time in ghost.cp_times],
        int(ghost.sample_period),
    )


def first_lap(ghost: Ghost, lap_count: int, checkpoint_count: int) -> list[Point]:
    """Cut a multi-lap ghost after its first finish-line crossing and reject
    a kept segment that contains a respawn teleport."""
    positions = ghost.positions
    if lap_count > 1:
        expected = (checkpoint_count + 1) * lap_count
        if len(ghost.checkpoint_times_ms) != expected:
            fail(
                f"route ghost records {len(ghost.checkpoint_times_ms)} checkpoint "
                f"times, expected {expected} for {lap_count} laps of "
                f"{checkpoint_count} checkpoints")
        lap_end = round(ghost.checkpoint_times_ms[checkpoint_count] / ghost.sample_period_ms)
        if lap_end < 2 or lap_end >= len(positions):
            fail("route ghost first-lap end lies outside its samples")
        print(
            f"multi-lap ghost: using lap 1 of {lap_count}, "
            f"{lap_end + 1} of {len(positions)} samples")
        positions = positions[:lap_end + 1]
    return unfold_respawns(positions)


def unfold_respawns(positions: list[Point]) -> list[Point]:
    """Replace each respawn teleport by driving the excursion back.

    E04-Obstacle checkpoint 10 sits off the main line; every ghost grabs it
    and respawns to the previous checkpoint. The kept path drives out to the
    checkpoint and returns along the same samples, so the centerline stays
    continuous and passes every checkpoint. 100 ms samples: 30 m per sample
    is 1,080 km/h, far above any drive, so larger steps are teleports.
    """
    result = list(positions)
    index = 1
    while index < len(result):
        if distance(result[index - 1], result[index]) <= 30.0:
            index += 1
            continue
        destination = result[index]
        origin = min(
            range(index), key=lambda k: distance(result[k], destination))
        if distance(result[origin], destination) > 8.0:
            fail(
                f"route ghost teleports {distance(result[index - 1], destination):.1f} m "
                "to a point it never drove through")
        excursion = result[origin + 1:index]
        print(
            f"respawn after sample {index - 1}: unfolding a {len(excursion)}-sample "
            f"excursion back to sample {origin}")
        result[index:index] = list(reversed(excursion[:-1]))
        index += len(excursion)
    return result


def contact_materials(path: Path) -> collections.Counter[int]:
    data = path.read_bytes()
    if len(data) == 0 or len(data) % RUN_RECORD_SIZE != 0:
        fail(f"{path} has invalid drive framing")
    result: collections.Counter[int] = collections.Counter()
    for record in range(len(data) // RUN_RECORD_SIZE):
        for wheel in range(4):
            offset = (
                record * RUN_RECORD_SIZE + RUN_WHEEL_BASE
                + wheel * RUN_WHEEL_SIZE + RUN_WHEEL_CONTACT_OFFSET
            )
            contact, encoded_material, _ = struct.unpack_from("<3I", data, offset)
            if contact not in (0, 1):
                fail(f"{path} has an invalid wheel-contact flag")
            if contact == 0:
                continue
            material = encoded_material & 0xFFFF
            if material >= 31:
                fail(f"{path} has a contact material outside the track table")
            result[material] += 1
    if not result:
        fail(f"{path} contains no wheel contacts")
    return result


def parse_track(path: Path) -> list[Triangle]:
    data = path.read_bytes()
    if len(data) < TRACK_HEADER_SIZE:
        fail("track snapshot is shorter than its header")
    magic, version, endian, header_size, count, file_size = struct.unpack_from(
        "<8sIIIIQ", data, 0)
    if (
        magic != b"TMNFTRK1"
        or version != 3
        or endian != 0x12345678
        or header_size != TRACK_HEADER_SIZE
        or count != TRACK_SECTION_COUNT
        or file_size != len(data)
    ):
        fail("invalid track snapshot header")
    if hashlib.sha256(data[TRACK_HEADER_SIZE:]).digest() != data[288:320]:
        fail("track payload digest does not match")
    sections = [
        struct.unpack_from("<QII", data, 96 + index * 16)
        for index in range(TRACK_SECTION_COUNT)
    ]
    expected_strides = (
        0x60, 0x18, 0x38, 0x0C, 0x20, 0x20, 1, 8, 0x14, 0x30, 0x28, 1)
    for section, expected_stride in zip(
        sections, expected_strides, strict=True
    ):
        offset, section_count, stride = section
        if (
            offset % 8 != 0
            or section_count == 0
            or stride != expected_stride
            or offset + section_count * stride > len(data)
        ):
            fail("invalid track section")

    mesh_offset, mesh_count, mesh_stride = sections[2]
    meshes: dict[int, tuple[list[Point], list[tuple[int, int, int, int]]]] = {}
    for index in range(mesh_count):
        offset = mesh_offset + index * mesh_stride
        fields = struct.unpack_from("<IHBBIIQIIQIIQ", data, offset)
        vertex_count, vertices_offset = fields[4], fields[6]
        face_count, faces_offset = fields[7], fields[9]
        vertices = [
            Point(*struct.unpack_from(
                "<3f", data, vertices_offset + vertex * 12))
            for vertex in range(vertex_count)
        ]
        faces = []
        for face in range(face_count):
            face_offset = faces_offset + face * 32
            indices = struct.unpack_from("<3I", data, face_offset + 16)
            if any(vertex >= vertex_count for vertex in indices):
                fail("mesh face references a vertex outside its mesh")
            material_index = struct.unpack_from("<H", data, face_offset + 28)[0]
            faces.append((*indices, material_index))
        meshes[offset] = (vertices, faces)

    surface_offset, surface_count, surface_stride = sections[1]
    surfaces: dict[
        int, tuple[list[Point], list[tuple[int, int, int, int]], bytes]
    ] = {}
    for index in range(surface_count):
        offset = surface_offset + index * surface_stride
        mesh_relative, material_relative, material_count, reserved = (
            struct.unpack_from("<QQII", data, offset)
        )
        if reserved != 0 or mesh_relative not in meshes or material_count == 0:
            fail("invalid track surface")
        material_ids = data[material_relative:material_relative + material_count]
        if len(material_ids) != material_count:
            fail("surface material remap is outside the track snapshot")
        vertices, faces = meshes[mesh_relative]
        if any(face[3] >= material_count for face in faces):
            fail("surface face material is outside its remap")
        surfaces[offset] = (vertices, faces, material_ids)

    result: list[Triangle] = []
    entry_offset, entry_count, entry_stride = sections[0]
    for index in range(entry_count):
        offset = entry_offset + index * entry_stride
        fields = struct.unpack_from("<I6f12fIQII", data, offset)
        tree_flags = fields[19]
        if tree_flags & 0x80 == 0:
            continue
        matrix = fields[7:19]
        surface_relative = fields[20]
        if surface_relative not in surfaces:
            fail("static entry references an invalid surface")
        vertices, faces, material_ids = surfaces[surface_relative]
        world_vertices = [
            transform_point(matrix, vertex) for vertex in vertices
        ]
        for a_index, b_index, c_index, material_index in faces:
            result.append(Triangle(
                world_vertices[a_index],
                world_vertices[b_index],
                world_vertices[c_index],
                material_ids[material_index],
            ))
    if not result:
        fail("track snapshot contains no leaf triangles")
    return result


def triangle_height(triangle: Triangle, x: float, z: float) -> float | None:
    a, b, c = triangle.a, triangle.b, triangle.c
    denominator = (
        (b.z - c.z) * (a.x - c.x)
        + (c.x - b.x) * (a.z - c.z)
    )
    if abs(denominator) < 1.0e-9:
        return None
    u = (
        (b.z - c.z) * (x - c.x)
        + (c.x - b.x) * (z - c.z)
    ) / denominator
    v = (
        (c.z - a.z) * (x - c.x)
        + (a.x - c.x) * (z - c.z)
    ) / denominator
    w = 1.0 - u - v
    if min(u, v, w) < -1.0e-6:
        return None
    return u * a.y + v * b.y + w * c.y


def upward_fraction(triangle: Triangle) -> float:
    ab = Point(
        triangle.b.x - triangle.a.x,
        triangle.b.y - triangle.a.y,
        triangle.b.z - triangle.a.z,
    )
    ac = Point(
        triangle.c.x - triangle.a.x,
        triangle.c.y - triangle.a.y,
        triangle.c.z - triangle.a.z,
    )
    normal = Point(
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x,
    )
    length = math.sqrt(normal.x ** 2 + normal.y ** 2 + normal.z ** 2)
    return 0.0 if length == 0.0 else abs(normal.y) / length


def drive_ray_materials(
    triangles: list[Triangle], positions: list[Point]
) -> collections.Counter[int]:
    """Material of the nearest face under each position. Positions with no
    face below are skipped: cars float on open sea in Bay, Coast and Island
    (BayD2's wall drive spends 110 ms over water without a face), and ghosts
    fly off the map."""
    bins: dict[tuple[int, int], list[Triangle]] = collections.defaultdict(list)
    bin_size = 32.0
    min_drive_x = min(point.x for point in positions)
    max_drive_x = max(point.x for point in positions)
    min_drive_z = min(point.z for point in positions)
    max_drive_z = max(point.z for point in positions)
    for triangle in triangles:
        min_x = max(
            min_drive_x, min(triangle.a.x, triangle.b.x, triangle.c.x))
        max_x = min(
            max_drive_x, max(triangle.a.x, triangle.b.x, triangle.c.x))
        min_z = max(
            min_drive_z, min(triangle.a.z, triangle.b.z, triangle.c.z))
        max_z = min(
            max_drive_z, max(triangle.a.z, triangle.b.z, triangle.c.z))
        if min_x > max_x or min_z > max_z:
            continue
        for ix in range(math.floor(min_x / bin_size), math.floor(max_x / bin_size) + 1):
            for iz in range(
                math.floor(min_z / bin_size), math.floor(max_z / bin_size) + 1
            ):
                bins[(ix, iz)].append(triangle)

    result: collections.Counter[int] = collections.Counter()
    for position in positions:
        best_drop = math.inf
        best_material = None
        key = (
            math.floor(position.x / bin_size),
            math.floor(position.z / bin_size),
        )
        for triangle in bins.get(key, ()):
            height = triangle_height(triangle, position.x, position.z)
            if height is None:
                continue
            drop = position.y - height
            if -0.25 <= drop < best_drop:
                best_drop = drop
                best_material = triangle.material
        if best_material is None:
            continue
        result[best_material] += 1
    return result


class RoadGrid:
    def __init__(
        self,
        triangles: list[Triangle],
        material_ids: set[int],
        anchors: list[Point],
        extent: list[Point],
    ) -> None:
        if len(anchors) < 2:
            fail("route generator requires a start and finish")
        self.jump_centers: list[Point] = []
        # The raster covers the anchors and, when there is one, the ghost lap
        # in x/z (United tracks loop far outside the anchors' box; DesertA5
        # runs 70 m past it and the road came back disconnected). The height
        # window stays on the anchors: the ghost's airborne samples would pull
        # in roofs above Stadium roads.
        margin = 256.0
        bounds = [*anchors, *extent]
        min_x = min(point.x for point in bounds) - margin
        max_x = max(point.x for point in bounds) + margin
        min_z = min(point.z for point in bounds) - margin
        max_z = max(point.z for point in bounds) + margin
        min_y = min(point.y for point in anchors) - 128.0
        max_y = max(point.y for point in anchors) + 64.0
        samples: dict[
            tuple[int, int], list[tuple[float, int]]
        ] = collections.defaultdict(list)
        selected_triangles = 0
        for triangle in triangles:
            if (
                triangle.material not in material_ids
                or upward_fraction(triangle) < MIN_UP_NORMAL
            ):
                continue
            tri_min_x = max(min_x, min(
                triangle.a.x, triangle.b.x, triangle.c.x))
            tri_max_x = min(max_x, max(
                triangle.a.x, triangle.b.x, triangle.c.x))
            tri_min_z = max(min_z, min(
                triangle.a.z, triangle.b.z, triangle.c.z))
            tri_max_z = min(max_z, max(
                triangle.a.z, triangle.b.z, triangle.c.z))
            if tri_min_x > tri_max_x or tri_min_z > tri_max_z:
                continue
            ix_min = math.ceil(tri_min_x / GRID_STEP - 0.5)
            ix_max = math.floor(tri_max_x / GRID_STEP - 0.5)
            iz_min = math.ceil(tri_min_z / GRID_STEP - 0.5)
            iz_max = math.floor(tri_max_z / GRID_STEP - 0.5)
            for ix in range(ix_min, ix_max + 1):
                x = (ix + 0.5) * GRID_STEP
                for iz in range(iz_min, iz_max + 1):
                    z = (iz + 0.5) * GRID_STEP
                    y = triangle_height(triangle, x, z)
                    if y is not None and min_y <= y <= max_y:
                        samples[(ix, iz)].append((y, triangle.material))
            selected_triangles += 1
        if selected_triangles == 0 or not samples:
            fail("empirical materials produced an empty road raster")

        self.nodes: list[GridNode] = []
        self.by_cell: dict[tuple[int, int], list[int]] = {}
        for cell in sorted(samples):
            values = sorted(samples[cell])
            clusters: list[list[tuple[float, int]]] = []
            for value in values:
                if not clusters or value[0] - clusters[-1][-1][0] > HEIGHT_CLUSTER:
                    clusters.append([value])
                else:
                    clusters[-1].append(value)
            indices = []
            for cluster in clusters:
                material = collections.Counter(
                    value[1] for value in cluster).most_common(1)[0][0]
                y = sum(value[0] for value in cluster) / len(cluster)
                indices.append(len(self.nodes))
                self.nodes.append(GridNode(cell[0], cell[1], y, material))
            self.by_cell[cell] = indices
        self.clearance = self._layer_clearance()

    def point(self, index: int) -> Point:
        node = self.nodes[index]
        return Point(
            (node.ix + 0.5) * GRID_STEP,
            node.y,
            (node.iz + 0.5) * GRID_STEP,
        )

    def nearest_layer(
        self, ix: int, iz: int, y: float, max_vertical: float
    ) -> int | None:
        result = None
        best = max_vertical
        for index in self.by_cell.get((ix, iz), ()):
            delta = abs(self.nodes[index].y - y)
            if delta <= best:
                best = delta
                result = index
        return result

    def _layer_clearance(self) -> list[float]:
        result = [math.inf] * len(self.nodes)
        queue: list[tuple[float, int]] = []
        for index, node in enumerate(self.nodes):
            if any(
                self.nearest_layer(
                    node.ix + dx, node.iz + dz, node.y, 1.5
                ) is None
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                result[index] = 0.5 * GRID_STEP
                heapq.heappush(queue, (result[index], index))
        if not queue:
            fail("road raster has no boundary")
        while queue:
            current, index = heapq.heappop(queue)
            if current != result[index]:
                continue
            node = self.nodes[index]
            for dx, dz in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ):
                for other in self.by_cell.get(
                    (node.ix + dx, node.iz + dz), ()
                ):
                    if abs(self.nodes[other].y - node.y) > 1.5:
                        continue
                    candidate = current + GRID_STEP * math.hypot(dx, dz)
                    if candidate < result[other]:
                        result[other] = candidate
                        heapq.heappush(queue, (candidate, other))
        if any(not math.isfinite(value) for value in result):
            fail("a road-raster layer has no measurable boundary")
        return result

    def snap(self, target: Point) -> int:
        center_ix = math.floor(target.x / GRID_STEP)
        center_iz = math.floor(target.z / GRID_STEP)
        result = None
        best = math.inf
        for radius in range(0, 33):
            for ix in range(center_ix - radius, center_ix + radius + 1):
                for iz in range(center_iz - radius, center_iz + radius + 1):
                    if radius and (
                        ix not in (center_ix - radius, center_ix + radius)
                        and iz not in (center_iz - radius, center_iz + radius)
                    ):
                        continue
                    for index in self.by_cell.get((ix, iz), ()):
                        point = self.point(index)
                        delta = (
                            (point.x - target.x) ** 2
                            + (point.y - target.y) ** 2
                            + (point.z - target.z) ** 2
                        )
                        if delta < best:
                            best = delta
                            result = index
            if result is not None and radius * GRID_STEP > math.sqrt(best):
                break
        if result is None or math.sqrt(best) > 32.0:
            fail("route anchor is more than 32 m from an empirical road surface")
        return result

    def neighbours(self, index: int) -> Iterable[tuple[int, float]]:
        node = self.nodes[index]
        origin = self.point(index)
        near_jump = any(
            math.hypot(origin.x - center.x, origin.z - center.z)
            <= JUMP_ANCHOR_RADIUS
            for center in self.jump_centers
        )
        maximum_gap = MAX_JUMP_GAP if near_jump else MAX_SEAM_GAP
        cell_radius = math.ceil(maximum_gap / GRID_STEP)
        for dx in range(-cell_radius, cell_radius + 1):
            for dz in range(-cell_radius, cell_radius + 1):
                if dx == 0 and dz == 0:
                    continue
                horizontal = GRID_STEP * math.hypot(dx, dz)
                if horizontal > maximum_gap:
                    continue
                for other in self.by_cell.get((node.ix + dx, node.iz + dz), ()):
                    other_point = self.point(other)
                    if horizontal > MAX_SEAM_GAP and not any(
                        math.hypot(
                            other_point.x - center.x,
                            other_point.z - center.z,
                        ) <= JUMP_ANCHOR_RADIUS
                        for center in self.jump_centers
                    ):
                        continue
                    vertical = abs(self.nodes[other].y - node.y)
                    if vertical > max(2.0, 1.25 * horizontal):
                        continue
                    step_distance = math.hypot(horizontal, vertical)
                    width = min(self.clearance[index], self.clearance[other])
                    center_penalty = 1.0 + 20.0 / ((width + 1.0) ** 2)
                    if horizontal > MAX_SEAM_GAP:
                        center_penalty += 10.0
                    yield other, step_distance * center_penalty

    def path(self, start: int, finish: int) -> list[int]:
        target = self.point(finish)
        for _ in range(8):
            queue = [(0.0, start)]
            cost = {start: 0.0}
            parent: dict[int, int] = {}
            visited: set[int] = set()
            while queue:
                _, current = heapq.heappop(queue)
                if current in visited:
                    continue
                visited.add(current)
                if current == finish:
                    result = [current]
                    while current != start:
                        current = parent[current]
                        result.append(current)
                    result.reverse()
                    return result
                for other, edge_cost in self.neighbours(current):
                    candidate = cost[current] + edge_cost
                    if candidate >= cost.get(other, math.inf):
                        continue
                    cost[other] = candidate
                    parent[other] = current
                    point = self.point(other)
                    heuristic = distance(point, target)
                    heapq.heappush(queue, (candidate + heuristic, other))
            closest = min(
                visited, key=lambda index: distance(self.point(index), target)
            )
            closest_point = self.point(closest)
            start_point = self.point(start)
            if not any(
                distance(start_point, center) <= 1.0
                for center in self.jump_centers
            ):
                self.jump_centers.append(start_point)
                continue
            if any(
                distance(closest_point, center) <= 1.0
                for center in self.jump_centers
            ):
                fail(
                    "empirical road surfaces are disconnected between ordered "
                    f"route anchors; closest reachable point is "
                    f"({closest_point.x:.3f}, {closest_point.y:.3f}, "
                    f"{closest_point.z:.3f}), "
                    f"{distance(closest_point, target):.3f} m from the target"
                )
            self.jump_centers.append(closest_point)
        fail("route requires more than eight disconnected-surface jumps")

    def line_supported(self, a: Point, b: Point) -> bool:
        length = distance(a, b)
        count = max(1, math.ceil(length / 0.5))
        for sample in range(count + 1):
            t = sample / count
            x = a.x + t * (b.x - a.x)
            y = a.y + t * (b.y - a.y)
            z = a.z + t * (b.z - a.z)
            ix = math.floor(x / GRID_STEP)
            iz = math.floor(z / GRID_STEP)
            layer = self.nearest_layer(ix, iz, y, 2.0)
            if layer is None or self.clearance[layer] < 2.0:
                return False
        return True

    def simplify(self, path: list[int]) -> list[Point]:
        points = [self.point(index) for index in path]
        result = [points[0]]
        current = 0
        while current + 1 < len(points):
            furthest = current + 1
            maximum = min(len(points) - 1, current + 64)
            for candidate in range(current + 2, maximum + 1):
                if self.line_supported(points[current], points[candidate]):
                    furthest = candidate
            result.append(points[furthest])
            current = furthest
        return result

    def half_width(self, point: Point, tangent: Point) -> float:
        horizontal = math.hypot(tangent.x, tangent.z)
        if horizontal <= 1.0e-6:
            fail("centerline has a vertical tangent")
        perpendicular = Point(-tangent.z / horizontal, 0.0, tangent.x / horizontal)

        def side(sign: float) -> float:
            previous = 0.0
            for step in range(1, 1025):
                candidate = step * 0.25
                x = point.x + sign * candidate * perpendicular.x
                z = point.z + sign * candidate * perpendicular.z
                ix = math.floor(x / GRID_STEP)
                iz = math.floor(z / GRID_STEP)
                if self.nearest_layer(
                    ix, iz, point.y - CAR_SURFACE_OFFSET,
                    2.5,
                ) is None:
                    return previous
                previous = candidate
            return previous

        width = min(side(-1.0), side(1.0))
        return width


def highest_face_under(
    triangles: list[Triangle], anchor: Point, material_ids: set[int] | None
) -> tuple[float, int]:
    """Height and material of the highest face below a route anchor.

    Anchors sit on block corners, and United terrain meshes leave sub-millimetre
    slivers between neighbouring blocks there (RallyB2's fourth checkpoint is
    over a 0.2 mm gap at z = 384), so the anchor and four points 0.1 m around
    it are all sampled."""
    best_y = -math.inf
    best_material = -1
    samples = [(anchor.x, anchor.z), (anchor.x + 0.1, anchor.z), (anchor.x - 0.1, anchor.z),
               (anchor.x, anchor.z + 0.1), (anchor.x, anchor.z - 0.1)]
    for triangle in triangles:
        if material_ids is not None and triangle.material not in material_ids:
            continue
        for x, z in samples:
            y = triangle_height(triangle, x, z)
            if y is not None and y <= anchor.y + 0.25 and y > best_y:
                best_y = y
                best_material = triangle.material
    return best_y, best_material


def anchor_material(triangles: list[Triangle], anchor: Point) -> int:
    """Material of the highest face under a route anchor. Start, checkpoint
    and finish blocks stand on their own drivable surface, so this is road
    even when the blind drives never touched it (RallyA1's checkpoint is on
    WetDirtRoad; the drives only see pavement, grass and wood)."""
    _, material = highest_face_under(triangles, anchor, None)
    if material < 0:
        fail("route anchor has no surface below it")
    return material


def ground_anchor(
    triangles: list[Triangle], material_ids: set[int], anchor: Point
) -> Point:
    best_y, _ = highest_face_under(triangles, anchor, material_ids)
    if not math.isfinite(best_y):
        fail("ordered route anchor has no empirical road surface below it")
    return Point(anchor.x, best_y, anchor.z)


def resample(
    legs: list[list[Point]],
    anchors: list[Point],
    grid: RoadGrid,
    start_position: Point,
    anchor_endpoints: bool,
) -> list[CenterPoint]:
    dense: list[tuple[Point, int]] = []
    for leg_index, leg in enumerate(legs):
        leg = list(leg)
        if len(leg) < 2:
            fail("centerline leg contains fewer than two points")
        if leg_index == 0:
            leg[0] = anchors[0]
        if anchor_endpoints:
            leg[-1] = anchors[leg_index + 1]
        elif leg_index == len(legs) - 1:
            # A ghost stops recording before the car crosses the finish
            # volume. Keep its driven line, then extend the last leg to the
            # grounded finish anchor so lookahead does not end off the road
            # before the race ends (notably Rally A1). Intermediate ghost
            # checkpoint crossings need not pass through the gate centre.
            if distance(leg[-1], anchors[-1]) > 0.25:
                leg.append(anchors[-1])
            else:
                leg[-1] = anchors[-1]
        cumulative = [0.0]
        for a, b in zip(leg, leg[1:]):
            cumulative.append(cumulative[-1] + distance(a, b))
        target = 0.0
        while target < cumulative[-1]:
            segment = 0
            while (cumulative[segment + 1] < target
                   or (cumulative[segment + 1] == cumulative[segment]
                       and segment + 2 < len(cumulative))):
                segment += 1
            span = cumulative[segment + 1] - cumulative[segment]
            t = (target - cumulative[segment]) / span
            a, b = leg[segment], leg[segment + 1]
            point = Point(
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            )
            if not dense or distance(dense[-1][0], point) > 0.25:
                dense.append((point, leg_index))
            target += OUTPUT_SPACING
        if not dense or distance(dense[-1][0], leg[-1]) > 0.25:
            dense.append((leg[-1], leg_index))
        elif leg_index == len(legs) - 1:
            # Keep the actual finish even when it lies just after the last
            # regular sample. Replacing that sample preserves the spacing
            # tolerance and avoids silently truncating the final centimetres.
            dense[-1] = (leg[-1], leg_index)

    positioned: list[tuple[Point, int, Point]] = []
    arc = 0.0
    for index, (surface_point, leg_index) in enumerate(dense):
        position = start_position if index == 0 else Point(
            surface_point.x, surface_point.y + CAR_SURFACE_OFFSET,
            surface_point.z)
        if index:
            arc += distance(positioned[-1][0], position)
        before = dense[max(0, index - 1)][0]
        after = dense[min(len(dense) - 1, index + 1)][0]
        tangent = Point(
            after.x - before.x,
            after.y - before.y,
            after.z - before.z,
        )
        positioned.append((position, leg_index, tangent))

    widths: list[float | None] = []
    for position, _, tangent in positioned:
        ix = math.floor(position.x / GRID_STEP)
        iz = math.floor(position.z / GRID_STEP)
        if grid.nearest_layer(
            ix, iz, position.y - CAR_SURFACE_OFFSET, 2.0
        ) is None:
            widths.append(None)
        else:
            width = grid.half_width(position, tangent)
            widths.append(width if width >= 1.0 else None)
    index = 0
    while index < len(widths):
        if widths[index] is not None:
            index += 1
            continue
        start = index
        while index < len(widths) and widths[index] is None:
            index += 1
        if start == 0 and index == len(widths):
            fail("no centerline point lies on a road surface")
        if start == 0 or index == len(widths):
            # A ghost may leave the start pad or cross the finish airborne;
            # the run takes the width of its one road-side neighbour.
            jump_width = widths[index] if start == 0 else widths[start - 1]
            print(
                f"{'leading' if start == 0 else 'trailing'} airborne run of "
                f"{index - start} points takes half-width {jump_width:.3f} m")
        else:
            jump_width = min(widths[start - 1], widths[index])
        if jump_width is None:
            fail("jump width endpoints are missing")
        for fill in range(start, index):
            widths[fill] = jump_width

    result = []
    arc = 0.0
    for index, ((position, leg_index, _), width) in enumerate(
        zip(positioned, widths, strict=True)
    ):
        if index:
            # Accumulate exactly as src/route.c verifies: float32 positions,
            # float32 segment length, float32 running sum. Double accumulation
            # rounded once drifts past the loader's 1e-3 tolerance on routes
            # longer than 8 km.
            arc = f32(arc + f32_distance(result[-1].position, position))
        if width is None:
            fail("centerline width interpolation was incomplete")
        result.append(CenterPoint(position, arc, width, leg_index))
    return result


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_distance(a: Point, b: Point) -> float:
    dx = f32(f32(b.x) - f32(a.x))
    dy = f32(f32(b.y) - f32(a.y))
    dz = f32(f32(b.z) - f32(a.z))
    return f32(math.sqrt(f32(f32(f32(dx * dx) + f32(dy * dy)) + f32(dz * dz))))


def ghost_legs(
    positions: list[Point], anchors: list[Point]
) -> list[list[Point]]:
    if distance(positions[0], anchors[0]) > 1.0:
        fail("route ghost does not begin at the captured start")
    surface_positions = [
        Point(point.x, point.y - CAR_SURFACE_OFFSET, point.z)
        for point in positions
    ]
    split_indices = [0]
    minimum = 1
    for anchor in anchors[1:-1]:
        index = min(
            range(minimum, len(positions)),
            key=lambda candidate: distance(positions[candidate], anchor),
        )
        if distance(positions[index], anchor) > 32.0:
            fail("route ghost does not pass an ordered checkpoint")
        split_indices.append(index)
        minimum = index + 1
    split_indices.append(len(positions) - 1)
    return [
        surface_positions[start:finish + 1]
        for start, finish in zip(split_indices, split_indices[1:])
    ]


def project(
    points: list[CenterPoint], position: Point, first_segment: int = 0
) -> tuple[float, float, int]:
    best_distance_sq = math.inf
    best_arc = 0.0
    best_segment = first_segment
    for index, (a, b) in enumerate(zip(points, points[1:])):
        if index < first_segment:
            continue
        delta = Point(
            b.position.x - a.position.x,
            b.position.y - a.position.y,
            b.position.z - a.position.z,
        )
        relative = Point(
            position.x - a.position.x,
            position.y - a.position.y,
            position.z - a.position.z,
        )
        length_sq = delta.x ** 2 + delta.y ** 2 + delta.z ** 2
        t = (
            relative.x * delta.x
            + relative.y * delta.y
            + relative.z * delta.z
        ) / length_sq
        t = min(1.0, max(0.0, t))
        offset = Point(
            relative.x - t * delta.x,
            relative.y - t * delta.y,
            relative.z - t * delta.z,
        )
        distance_sq = offset.x ** 2 + offset.y ** 2 + offset.z ** 2
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_arc = a.arc_length + t * (b.arc_length - a.arc_length)
            best_segment = index
    return best_arc, math.sqrt(best_distance_sq), best_segment


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - index)
        + ordered[upper] * (index - lower)
    )


def validate_centerline(
    points: list[CenterPoint],
    anchors: list[Point],
    run1: list[Point],
) -> dict[str, float | int | list[float]]:
    if len(points) < 2:
        fail("generated fewer than two centerline points")
    if points[0].arc_length != 0.0:
        fail("centerline does not start at arc length zero")
    for previous, current in zip(points, points[1:]):
        if current.arc_length <= previous.arc_length:
            fail("centerline arc length is not strictly increasing")
        spacing = distance(previous.position, current.position)
        # resample() drops a sample that lands within 0.25 m of the previous
        # point (leg ends, leg starts), so consecutive kept points can be up
        # to OUTPUT_SPACING + 0.25 m apart.
        if spacing > OUTPUT_SPACING + 0.25:
            fail(
                f"centerline spacing {spacing:.3f} m exceeds its "
                f"configured bound after ({previous.position.x:.3f}, "
                f"{previous.position.y:.3f}, {previous.position.z:.3f})")
        if current.half_width <= 0.0:
            fail("centerline has a non-positive half-width")

    # Anchors are ordered along the route, so each one projects onto the
    # centerline after the previous anchor. On lap tracks the finish trigger
    # sits on the start block and would otherwise project back to arc zero.
    checkpoint_arcs = []
    segment = 0
    for anchor in anchors:
        arc, _, segment = project(points, anchor, segment)
        checkpoint_arcs.append(arc)
    if any(
        current <= previous
        for previous, current in zip(checkpoint_arcs, checkpoint_arcs[1:])
    ):
        fail("ordered route anchors do not project in increasing order")

    projections = [project(points, position) for position in run1]
    offsets = [projection[1] for projection in projections]
    arcs = [projection[0] for projection in projections]
    decreases = [
        previous - current
        for previous, current in zip(arcs, arcs[1:])
        if current + 1.0e-5 < previous
    ]
    return {
        "run1_median": statistics.median(offsets),
        "run1_p95": percentile(offsets, 0.95),
        "run1_max": max(offsets),
        "run1_decreasing_steps": len(decreases),
        "run1_max_decrease": max(decreases, default=0.0),
        "anchor_arcs": checkpoint_arcs,
    }


def unsupported_runs(
    points: list[CenterPoint], grid: RoadGrid
) -> tuple[int, int, float]:
    supported = []
    for point in points:
        ix = math.floor(point.position.x / GRID_STEP)
        iz = math.floor(point.position.z / GRID_STEP)
        supported.append(
            grid.nearest_layer(
                ix,
                iz,
                point.position.y - CAR_SURFACE_OFFSET,
                2.5,
            ) is not None
        )
    run_count = 0
    point_count = 0
    maximum_length = 0.0
    index = 0
    while index < len(points):
        if supported[index]:
            index += 1
            continue
        run_count += 1
        start = index
        while index < len(points) and not supported[index]:
            point_count += 1
            index += 1
        before = max(0, start - 1)
        after = min(len(points) - 1, index)
        maximum_length = max(
            maximum_length,
            points[after].arc_length - points[before].arc_length,
        )
    return run_count, point_count, maximum_length


def build_route(
    source: bytes,
    old_sections: list[tuple[int, int, int]],
    points: list[CenterPoint],
) -> bytes:
    section_payloads = []
    for index in range(4):
        offset, count, stride = old_sections[index]
        section_payloads.append(bytearray(source[offset:offset + count * stride]))
    section_payloads.append(bytearray().join(
        struct.pack(
            "<5fI",
            point.position.x,
            point.position.y,
            point.position.z,
            point.arc_length,
            point.half_width,
            point.leg_index,
        )
        for point in points
    ))
    metadata = section_payloads[0]
    checkpoint_count = struct.unpack_from("<I", metadata, 4)[0]
    struct.pack_into("<I", metadata, 12, checkpoint_count + 2)
    struct.pack_into("<I", metadata, 28, len(points))
    return layout_route(
        source, section_payloads,
        (0x40, START_STRIDE, TRIGGER_STRIDE, TRIGGER_STRIDE, 0x18), ROUTE_VERSION)


def select_finish_from_ghost(
    data: bytes, sections: list[tuple[int, int, int]], ghost: list[Point]
) -> bytes:
    """Put the ghost's finish first for the reference line, retaining every
    legal alternative finish. Returns re-laid-out route bytes."""
    offset, count, stride = sections[3]
    if count == 1:
        return data
    anchors = route_anchor_points(data, sections)
    finish_anchors = anchors[1 + sections[2][1]:]
    end = ghost[-1]
    distances = [distance(anchor, end) for anchor in finish_anchors]
    chosen = min(range(count), key=distances.__getitem__)
    # The trigger anchor is the box centre, well above the road on the tall
    # Desert gates (19 m on DesertD4), so the closeness test is horizontal:
    # last sample up to 100 ms before the line plus half a block.
    horizontal = math.hypot(
        finish_anchors[chosen].x - end.x, finish_anchors[chosen].z - end.z)
    if horizontal > 32.0:
        fail(
            f"route ghost ends {horizontal:.3f} m (horizontal) from the nearest "
            f"of {count} finish triggers")
    print(
        f"multi-finish route: reference finish {chosen} of {count}, retaining all, "
        f"{distances[chosen]:.3f} m from the ghost end")
    payloads = [
        bytearray(data[start:start + n * s])
        for start, n, s in sections
    ]
    order = [chosen, *(i for i in range(count) if i != chosen)]
    records = []
    for index, original in enumerate(order):
        record = bytearray(data[offset + original * stride:offset + (original + 1) * stride])
        struct.pack_into("<I", record, 0, index)
        records.append(record)
    payloads[3] = bytearray().join(records)
    struct.pack_into("<I", payloads[0], 8, count)
    version = struct.unpack_from("<I", data, 8)[0]
    strides = (0x40, START_STRIDE, TRIGGER_STRIDE, TRIGGER_STRIDE,
               0x10 if version == 1 else 0x18)
    return layout_route(data, payloads, strides, version)


def layout_route(
    source: bytes,
    section_payloads: list[bytearray],
    strides: tuple[int, ...],
    version: int,
) -> bytes:
    counts = []
    for payload, stride in zip(section_payloads, strides, strict=True):
        if len(payload) % stride != 0:
            fail("route section payload is not a whole number of records")
        counts.append(len(payload) // stride)
    offsets = []
    cursor = ROUTE_HEADER_SIZE
    for payload in section_payloads:
        cursor = align8(cursor)
        offsets.append(cursor)
        cursor += len(payload)
    struct.pack_into("<4Q", section_payloads[0], 32, *offsets[1:])

    image = bytearray(cursor)
    for offset, payload in zip(offsets, section_payloads, strict=True):
        image[offset:offset + len(payload)] = payload
    image[:ROUTE_HEADER_SIZE] = source[:ROUTE_HEADER_SIZE]
    struct.pack_into(
        "<8sIIIIQ",
        image,
        0,
        b"TMNFROU1",
        version,
        0x12345678,
        ROUTE_HEADER_SIZE,
        ROUTE_SECTION_COUNT,
        len(image),
    )
    for index, (offset, count, stride) in enumerate(
        zip(offsets, counts, strides, strict=True)
    ):
        struct.pack_into("<QII", image, 96 + index * 16, offset, count, stride)
    image[176:208] = hashlib.sha256(image[ROUTE_HEADER_SIZE:]).digest()
    return bytes(image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track", type=Path, default=Path("oracle/tracks/A01-Race.tmnftrack"))
    parser.add_argument(
        "--route", type=Path, default=Path("oracle/routes/A01-Race.tmnfroute"))
    parser.add_argument(
        "--run1", type=Path, default=Path("oracle/results/run1.bin"))
    parser.add_argument(
        "--long-drive",
        type=Path,
        default=Path("oracle/results/a01_long_drive.bin"),
    )
    parser.add_argument("--validation-drive", type=Path)
    parser.add_argument("--ghost-replay", type=Path)
    parser.add_argument(
        "--ghost-optional", action="store_true",
        help="fall back to the empirical road grid when the replay has no "
             "usable no-respawn ghost")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = args.route.read_bytes()
    old_sections = route_sections(source)
    ghost = None
    if args.ghost_replay is not None:
        parsed = read_ghost(args.ghost_replay, args.ghost_optional)
        if parsed is not None:
            lap_count = struct.unpack_from("<I", source, old_sections[0][0])[0]
            ghost = first_lap(parsed, lap_count, old_sections[2][1])
    if old_sections[3][1] != 1 and ghost is None:
        fail(
            f"route has {old_sections[3][1]} finish triggers and no ghost to "
            "choose the one the race ends at")
    if ghost is not None:
        source = select_finish_from_ghost(source, old_sections, ghost)
        old_sections = route_sections(source)
        source = reorder_checkpoints_from_ghost(source, old_sections, ghost)
        old_sections = route_sections(source)
    route_anchors = route_anchor_points(source, old_sections)[:old_sections[2][1] + 2]
    run1 = read_drive(args.run1)
    long_drive = read_drive(args.long_drive)
    validation_drive = (
        read_drive(args.validation_drive)
        if args.validation_drive is not None
        else run1
    )
    run1_contacts = contact_materials(args.run1)
    long_contacts = contact_materials(args.long_drive)
    material_ids = set(run1_contacts) | set(long_contacts)

    triangles = parse_track(args.track)
    ray_materials = drive_ray_materials(triangles, run1 + long_drive)
    # Wheels also grind along wall sides and borders (the wall-heavy schedule
    # does so for hundreds of ticks). Those materials never lie under the car
    # and are reported, not rejected. Every contacted material still counts as
    # road for the grid; its non-upward faces are filtered there.
    side_only = {
        material for material in material_ids if material not in ray_materials
    }
    if side_only:
        print(
            "side-contact materials never under the car: "
            + ", ".join(
                f"{material}={run1_contacts[material] + long_contacts[material]}"
                for material in sorted(side_only)
            )
        )
    if ghost is not None:
        # The two blind drives cover the first few hundred metres; the ghost
        # covers the whole lap and may run on surfaces they never touched.
        ghost_materials = drive_ray_materials(triangles, ghost)
        added = set(ghost_materials) - material_ids
        material_ids |= set(ghost_materials)
        print(
            "ghost road materials: "
            + ", ".join(f"{m}={ghost_materials[m]}" for m in sorted(ghost_materials))
            + (f" (added {sorted(added)})" if added else "")
        )
    anchor_materials = {
        anchor_material(triangles, anchor) for anchor in route_anchors
    }
    added = anchor_materials - material_ids
    material_ids |= anchor_materials
    if added:
        print(f"anchor road materials added: {sorted(added)}")
    grounded_anchors = [
        ground_anchor(triangles, material_ids, anchor)
        for anchor in route_anchors
    ]
    grid = RoadGrid(triangles, material_ids, grounded_anchors, ghost or [])
    if ghost is None:
        # The wall drive's end is only a route point without a ghost; with one
        # it may sit anywhere (CoastA5's drives both end in the sea).
        recorded_prefix_end = ground_anchor(
            triangles, material_ids, long_drive[-1])
        snapped = [grid.snap(anchor) for anchor in grounded_anchors]
        prefix_end = grid.snap(recorded_prefix_end)
        first_tail = grid.simplify(grid.path(prefix_end, snapped[1]))
        first_tail[0] = recorded_prefix_end
        legs = [[grounded_anchors[0], recorded_prefix_end, *first_tail[1:]]]
        for leg_index, (start, finish) in enumerate(
            zip(snapped[1:], snapped[2:]), start=1
        ):
            try:
                legs.append(grid.simplify(grid.path(start, finish)))
            except RuntimeError as error:
                fail(f"route leg {leg_index}: {error}")
    else:
        legs = ghost_legs(ghost, route_anchors)
    points = resample(
        legs, grounded_anchors, grid, route_anchors[0], ghost is None)
    validation = validate_centerline(
        points, route_anchors, validation_drive)
    jump_runs, airborne_points, maximum_jump = unsupported_runs(points, grid)
    output = build_route(source, old_sections, points)

    if args.check:
        if output != source:
            fail("committed route snapshot is not reproducible")
        action = "verified"
    else:
        args.route.write_bytes(output)
        action = "wrote"

    print(
        "wheel contact materials: "
        + ", ".join(
            f"{material}={run1_contacts[material] + long_contacts[material]}"
            for material in sorted(material_ids)
        )
    )
    print(
        "vertical-ray materials: "
        + ", ".join(
            f"{material}={ray_materials[material]}"
            for material in sorted(ray_materials)
        )
    )
    print(
        f"centerline: points={len(points)} spacing={OUTPUT_SPACING:.1f}m "
        f"length={points[-1].arc_length:.3f}m "
        f"half_width_min={min(point.half_width for point in points):.3f}m "
        f"half_width_max={max(point.half_width for point in points):.3f}m"
    )
    print(
        f"surface support: jump_runs={jump_runs} "
        f"airborne_points={airborne_points} "
        f"maximum_jump={maximum_jump:.3f}m"
    )
    print(
        "validation-drive projection: "
        f"median={validation['run1_median']:.3f}m "
        f"p95={validation['run1_p95']:.3f}m "
        f"max={validation['run1_max']:.3f}m"
    )
    print(
        "validation-drive monotonicity: "
        f"decreasing_steps={validation['run1_decreasing_steps']} "
        f"max_decrease={validation['run1_max_decrease']:.3f}m"
    )
    anchor_arcs = validation["anchor_arcs"]
    if not isinstance(anchor_arcs, list):
        fail("route anchor validation result has the wrong type")
    print(
        "ordered anchors: "
        + ", ".join(
            ("start" if index == 0 else
             "finish" if index == len(anchor_arcs) - 1 else
             f"checkpoint_{index - 1}") + f"={arc:.3f}m"
            for index, arc in enumerate(anchor_arcs)
        )
    )
    print(
        f"{action} {args.route}: bytes={len(output)} "
        f"sha256={hashlib.sha256(output).hexdigest()} "
        f"payload_sha256={output[176:208].hex()}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"generate_route_centerline: {error}", file=sys.stderr)
        sys.exit(1)
