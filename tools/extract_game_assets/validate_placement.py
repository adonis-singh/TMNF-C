#!/usr/bin/env python3
"""Measure TMNF viewer block placement against the collision snapshot."""

import argparse
import json
import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GRID_XZ = 32.0
GRID_Y = 8.0
ROAD_NORMAL_Y = 0.25
CELL_EPSILON = 0.05


def fail(message):
    raise SystemExit(f"validate_placement: {message}")


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")


def percentile(values, fraction):
    if not values:
        return None
    return float(np.percentile(np.asarray(values), fraction * 100))


def rounded(value, digits=4):
    return None if value is None else round(float(value), digits)


def compact_ranges(values):
    if not values:
        return "-"
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def triangle_normals(triangles):
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = lengths > 1e-10
    normals[valid] = cross[valid] / lengths[valid, None]
    return normals, lengths * 0.5


@dataclass
class Geometry:
    triangles: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def from_triangles(cls, triangles):
        if triangles.size == 0:
            fail("geometry contains no triangles")
        normals, areas = triangle_normals(triangles)
        points = triangles.reshape(-1, 3)
        return cls(
            triangles=triangles,
            normals=normals,
            areas=areas,
            minimum=points.min(axis=0),
            maximum=points.max(axis=0),
        )


@dataclass
class CollisionSurface:
    entry_index: int
    geometry: Geometry
    minimum: np.ndarray
    maximum: np.ndarray


class PointRayIndex:
    def __init__(self, geometry, cell_size=GRID_XZ, maximum_cells=64):
        self.geometry = geometry
        self.cell_size = cell_size
        self.cells = defaultdict(list)
        self.large = []
        minimum = geometry.triangles.min(axis=1)
        maximum = geometry.triangles.max(axis=1)
        for index in range(len(geometry.triangles)):
            x0 = math.floor(minimum[index, 0] / cell_size)
            x1 = math.floor(maximum[index, 0] / cell_size)
            z0 = math.floor(minimum[index, 2] / cell_size)
            z1 = math.floor(maximum[index, 2] / cell_size)
            if (x1 - x0 + 1) * (z1 - z0 + 1) > maximum_cells:
                self.large.append(index)
                continue
            for x in range(x0, x1 + 1):
                for z in range(z0, z1 + 1):
                    self.cells[(x, z)].append(index)
        self.large = np.asarray(self.large, dtype=np.int64)
        self.cells = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in self.cells.items()
        }

    def candidates(self, x, z):
        cell = (
            math.floor(x / self.cell_size),
            math.floor(z / self.cell_size),
        )
        local = self.cells.get(cell)
        if local is None:
            return self.large
        if self.large.size == 0:
            return local
        return np.concatenate((local, self.large))

    def ray_down(self, x, z, origin_y):
        indices = self.candidates(x, z)
        if indices.size == 0:
            return None
        triangles = self.geometry.triangles[indices]
        x0 = triangles[:, 0, 0]
        z0 = triangles[:, 0, 2]
        x1 = triangles[:, 1, 0]
        z1 = triangles[:, 1, 2]
        x2 = triangles[:, 2, 0]
        z2 = triangles[:, 2, 2]
        denominator = (z1 - z2) * (x0 - x2) + (x2 - x1) * (z0 - z2)
        valid = np.abs(denominator) > 1e-9
        a = np.zeros_like(denominator)
        b = np.zeros_like(denominator)
        a[valid] = (
            (z1[valid] - z2[valid]) * (x - x2[valid])
            + (x2[valid] - x1[valid]) * (z - z2[valid])
        ) / denominator[valid]
        b[valid] = (
            (z2[valid] - z0[valid]) * (x - x2[valid])
            + (x0[valid] - x2[valid]) * (z - z2[valid])
        ) / denominator[valid]
        c = 1.0 - a - b
        valid &= (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        if not np.any(valid):
            return None
        heights = (
            a * triangles[:, 0, 1]
            + b * triangles[:, 1, 1]
            + c * triangles[:, 2, 1]
        )
        valid &= heights <= origin_y + 1e-5
        if not np.any(valid):
            return None
        return float(np.max(heights[valid]))


def accessor(document, binary, accessor_index):
    entry = document["accessors"][accessor_index]
    view = document["bufferViews"][entry["bufferView"]]
    if view.get("byteStride"):
        fail("strided glTF accessors are not supported")
    types = {
        5123: np.dtype("<u2"),
        5125: np.dtype("<u4"),
        5126: np.dtype("<f4"),
    }
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}
    try:
        dtype = types[entry["componentType"]]
        width = widths[entry["type"]]
    except KeyError:
        fail(f"unsupported glTF accessor {entry}")
    offset = view.get("byteOffset", 0) + entry.get("byteOffset", 0)
    values = np.frombuffer(
        binary,
        dtype=dtype,
        count=entry["count"] * width,
        offset=offset,
    )
    return values.reshape(entry["count"], width)


def node_matrix(entry):
    if "matrix" in entry:
        return np.asarray(entry["matrix"], dtype=np.float64).reshape(
            (4, 4), order="F")
    matrix = np.identity(4, dtype=np.float64)
    if "translation" in entry:
        matrix[:3, 3] = entry["translation"]
    if "scale" in entry:
        matrix[:3, :3] *= np.asarray(entry["scale"], dtype=np.float64)
    if "rotation" in entry:
        x, y, z, w = entry["rotation"]
        matrix[:3, :3] = np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
             2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
             2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w),
             1 - 2 * (x * x + y * y)],
        ])
    return matrix


def load_gltf(path):
    document = load_json(path)
    if document.get("asset", {}).get("version") != "2.0":
        fail(f"{path} is not glTF 2.0")
    if len(document.get("buffers", [])) != 1:
        fail(f"{path} must contain exactly one external buffer")
    binary_path = path.parent / document["buffers"][0]["uri"]
    try:
        binary = binary_path.read_bytes()
    except OSError as error:
        fail(f"cannot read {binary_path}: {error}")
    if len(binary) != document["buffers"][0]["byteLength"]:
        fail(f"{binary_path} has the wrong byte length")

    triangles = []
    active = set()

    def visit(index, parent):
        if index in active:
            fail(f"{path} has a node cycle")
        active.add(index)
        entry = document["nodes"][index]
        transform = parent @ node_matrix(entry)
        if "mesh" in entry:
            mesh = document["meshes"][entry["mesh"]]
            for primitive in mesh["primitives"]:
                if primitive.get("mode", 4) != 4:
                    fail(f"{path} contains a non-triangle primitive")
                positions = accessor(
                    document, binary, primitive["attributes"]["POSITION"])
                indices = accessor(
                    document, binary, primitive["indices"]).reshape(-1)
                if len(indices) % 3:
                    fail(f"{path} has a partial triangle")
                points = np.column_stack((
                    positions.astype(np.float64),
                    np.ones(len(positions)),
                ))
                world = (points @ transform.T)[:, :3]
                triangles.append(world[indices.reshape(-1, 3)])
        for child in entry.get("children", []):
            visit(child, transform)
        active.remove(index)

    scene_index = document.get("scene", 0)
    for node in document["scenes"][scene_index]["nodes"]:
        visit(node, np.identity(4, dtype=np.float64))
    if not triangles:
        fail(f"{path} contains no triangles")
    return Geometry.from_triangles(np.concatenate(triangles))


def placement_geometry(template, placement):
    angle = float(placement["rotationY"])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray([
        [cosine, 0, sine],
        [0, 1, 0],
        [-sine, 0, cosine],
    ])
    translation = np.asarray(placement["position"], dtype=np.float64)
    triangles = template.triangles @ rotation.T + translation
    return Geometry.from_triangles(triangles)


def read_collision_snapshot(path):
    try:
        data = path.read_bytes()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    if len(data) < 0x130:
        fail(f"{path} is shorter than a TMNFTRK1 header")
    magic, version, endian, header_size, section_count, file_size = (
        struct.unpack_from("<8sIIIIQ", data, 0))
    if magic != b"TMNFTRK1" or version != 3 or endian != 0x12345678:
        fail(f"{path} is not a supported TMNFTRK1 snapshot")
    if header_size != 0x150 or section_count != 12 or file_size != len(data):
        fail(f"{path} has an invalid header")
    sections = [
        struct.unpack_from("<QII", data, 96 + index * 16)
        for index in range(section_count)
    ]
    entry_offset, entry_count, entry_stride = sections[0]
    if entry_stride != 0x60:
        fail(f"{path} has an unexpected static-entry stride")

    triangles = []
    surfaces = []
    surface_count = 0
    for entry_index in range(entry_count):
        offset = entry_offset + entry_index * entry_stride
        values = struct.unpack_from("<I6f12fIQII", data, offset)
        transform = np.asarray(values[7:16], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(values[16:19], dtype=np.float64)
        tree_flags = values[19]
        surface_offset = values[20]
        if not tree_flags & 0x80:
            continue
        mesh_offset = struct.unpack_from("<Q", data, surface_offset)[0]
        surface_type = data[mesh_offset + 6]
        if surface_type != 7:
            fail(
                f"active collision entry {entry_index} is type "
                f"{surface_type}, expected mesh")
        vertex_count = struct.unpack_from("<I", data, mesh_offset + 8)[0]
        vertices_offset = struct.unpack_from("<Q", data, mesh_offset + 16)[0]
        face_count = struct.unpack_from("<I", data, mesh_offset + 24)[0]
        faces_offset = struct.unpack_from("<Q", data, mesh_offset + 32)[0]
        vertices = np.frombuffer(
            data,
            dtype=np.dtype("<f4"),
            count=vertex_count * 3,
            offset=vertices_offset,
        ).reshape(-1, 3).astype(np.float64)
        world = vertices @ transform.T + translation
        faces = np.empty((face_count, 3), dtype=np.uint32)
        for face_index in range(face_count):
            faces[face_index] = struct.unpack_from(
                "<III", data, faces_offset + face_index * 0x20 + 0x10)
        if np.any(faces >= vertex_count):
            fail(f"collision entry {entry_index} has an invalid face index")
        surface_triangles = world[faces]
        geometry = Geometry.from_triangles(surface_triangles)
        box_center = np.asarray(values[1:4], dtype=np.float64)
        box_half = np.asarray(values[4:7], dtype=np.float64)
        surfaces.append(CollisionSurface(
            entry_index=entry_index,
            geometry=geometry,
            minimum=box_center - box_half,
            maximum=box_center + box_half,
        ))
        triangles.append(surface_triangles)
        surface_count += 1
    if not triangles:
        fail(f"{path} contains no active collision triangles")
    return Geometry.from_triangles(np.concatenate(triangles)), surfaces


def nominal_footprint(template):
    maximum = template.maximum

    def extent(value):
        return GRID_XZ * max(
            1, math.ceil((float(value) - 1.0) / GRID_XZ))

    return extent(maximum[0]), extent(maximum[2])


def expected_footprint(placement, template):
    width, depth = nominal_footprint(template)
    quarter_turn = int(round(
        -float(placement["rotationY"]) / (math.pi / 2))) % 4
    if quarter_turn % 2:
        width, depth = depth, width
    if "coord" in placement:
        x = placement["coord"][0] * GRID_XZ
        z = placement["coord"][2] * GRID_XZ
    else:
        x, _, z = placement["position"]
    return np.asarray([x, z]), np.asarray([x + width, z + depth])


def aabb_centroid(bounds):
    return (bounds[0] + bounds[1]) * 0.5


def aabb_overlap_ratio(first, second):
    first_size = first[1] - first[0]
    second_size = second[1] - second[0]
    dimensions = [0, 1, 2]
    if first_size[1] < 1e-5 or second_size[1] < 1e-5:
        dimensions = [0, 2]
    overlap = np.maximum(
        np.minimum(first[1][dimensions], second[1][dimensions])
        - np.maximum(first[0][dimensions], second[0][dimensions]),
        0,
    )
    first_measure = np.prod(np.maximum(first_size[dimensions], 1e-6))
    second_measure = np.prod(np.maximum(second_size[dimensions], 1e-6))
    denominator = min(float(first_measure), float(second_measure))
    return float(np.prod(overlap) / denominator)


def placement_grid_offset(placement, template):
    width, depth = nominal_footprint(template)
    angle = float(placement["rotationY"])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    corners = np.asarray([
        [0, 0],
        [width, 0],
        [0, depth],
        [width, depth],
    ])
    rotated = np.column_stack((
        cosine * corners[:, 0] + sine * corners[:, 1],
        -sine * corners[:, 0] + cosine * corners[:, 1],
    ))
    minimum = rotated.min(axis=0)
    return np.asarray([-minimum[0], 0, -minimum[1]])


def find_collision_surface(placement, template, visual, surfaces):
    footprint = expected_footprint(placement, template)
    expected_offset = placement_grid_offset(placement, template)
    applied_offset = np.asarray(
        placement.get("gridOffset", [0, 0, 0]), dtype=np.float64)
    correction = expected_offset - applied_offset
    expected = (visual.minimum + correction, visual.maximum + correction)
    expected_center = aabb_centroid(expected)
    expected_size = expected[1] - expected[0]
    origin_y = float(placement["position"][1])
    vertical_minimum = origin_y - 2.0
    vertical_maximum = origin_y + max(float(template.maximum[1]), GRID_Y) + 2.0
    candidates = []
    for surface in surfaces:
        horizontal_overlap = (
            surface.maximum[0] >= footprint[0][0] - CELL_EPSILON
            and surface.minimum[0] <= footprint[1][0] + CELL_EPSILON
            and surface.maximum[2] >= footprint[0][1] - CELL_EPSILON
            and surface.minimum[2] <= footprint[1][1] + CELL_EPSILON
        )
        vertical_overlap = (
            surface.maximum[1] >= vertical_minimum
            and surface.minimum[1] <= vertical_maximum
        )
        if not horizontal_overlap or not vertical_overlap:
            continue
        center = (surface.minimum + surface.maximum) * 0.5
        size = surface.maximum - surface.minimum
        score = float(
            np.linalg.norm(center - expected_center)
            + np.linalg.norm(size - expected_size)
        )
        candidates.append((score, surface.entry_index, surface))
    if not candidates:
        return None, None, expected_offset, applied_offset
    score, _, surface = min(candidates, key=lambda item: (item[0], item[1]))
    return surface, score, expected_offset, applied_offset


def mean_normal(normals, areas):
    if len(normals) == 0:
        return None
    upward = normals.copy()
    upward[upward[:, 1] < 0] *= -1
    result = np.sum(upward * areas[:, None], axis=0)
    length = np.linalg.norm(result)
    if length < 1e-8:
        return None
    return result / length


def vector_angle(first, second):
    if first is None or second is None:
        return None
    cosine = float(np.clip(np.dot(first, second), -1, 1))
    return math.degrees(math.acos(cosine))


def principal_axis(triangles, areas):
    if len(triangles) < 3:
        return None
    points = triangles.mean(axis=1)[:, [0, 2]]
    weights = np.maximum(areas, 1e-8)
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (
        (centered * weights[:, None]).T @ centered / np.sum(weights))
    values, vectors = np.linalg.eigh(covariance)
    if values[1] < 1e-8 or values[1] / max(values[0], 1e-8) < 1.2:
        return None
    return vectors[:, 1]


def axis_angle(first, second):
    if first is None or second is None:
        return None
    cosine = float(np.clip(abs(np.dot(first, second)), -1, 1))
    return math.degrees(math.acos(cosine))


def validate_blocks(manifest_path, placement_path, surfaces):
    manifest = load_json(manifest_path)
    placement_scene = load_json(placement_path)
    if manifest.get("format") != "tmnf-game-visual-manifest":
        fail(f"{manifest_path} is not a game visual manifest")
    if placement_scene.get("format") != "tmnf-game-visual-scene":
        fail(f"{placement_path} is not a game visual placement scene")
    assets = manifest.get("assets")
    placements = placement_scene.get("placements")
    if not isinstance(assets, dict) or not isinstance(placements, list):
        fail("manifest assets or scene placements are malformed")

    template_cache = {}
    placed = []
    for index, placement in enumerate(placements):
        key = placement["asset"]
        asset = assets.get(key)
        if not asset or not asset.get("visual"):
            fail(f"placement {index} references missing asset {key}")
        model_path = (manifest_path.parent / asset["visual"]).resolve()
        if model_path not in template_cache:
            template_cache[model_path] = load_gltf(model_path)
        template = template_cache[model_path]
        placed.append((
            index,
            placement,
            template,
            placement_geometry(template, placement),
        ))

    reports = []
    for index, placement, template, visual in placed:
        footprint = expected_footprint(placement, template)
        (
            surface,
            match_score,
            expected_offset,
            applied_offset,
        ) = find_collision_surface(
            placement, template, visual, surfaces,
        )
        visual_bounds = (visual.minimum, visual.maximum)
        collision_bounds = (
            None if surface is None
            else (surface.minimum, surface.maximum))
        reasons = []
        overlap = None
        centroid_offset = None
        normal_angle = None
        principal_angle = None
        if collision_bounds is None:
            reasons.append("no collision surface in grid cells")
        else:
            overlap = aabb_overlap_ratio(visual_bounds, collision_bounds)
            centroid_offset = float(np.linalg.norm(
                aabb_centroid(visual_bounds)
                - aabb_centroid(collision_bounds)))
            visual_road = np.abs(visual.normals[:, 1]) >= ROAD_NORMAL_Y
            collision_road = (
                np.abs(surface.geometry.normals[:, 1]) >= ROAD_NORMAL_Y)
            normal_angle = vector_angle(
                mean_normal(
                    visual.normals[visual_road],
                    visual.areas[visual_road]),
                mean_normal(
                    surface.geometry.normals[collision_road],
                    surface.geometry.areas[collision_road]),
            )
            principal_angle = axis_angle(
                principal_axis(
                    visual.triangles[visual_road],
                    visual.areas[visual_road]),
                principal_axis(
                    surface.geometry.triangles[collision_road],
                    surface.geometry.areas[collision_road]),
            )
            if overlap < 0.5:
                reasons.append("overlap < 0.5")
            if centroid_offset > 1.0:
                reasons.append("centroid offset > 1 m")
        reports.append({
            "placementIndex": index,
            "blockIndex": placement.get("blockIndex"),
            "asset": placement["asset"],
            "model": placement["model"],
            "position": placement["position"],
            "rotationY": placement["rotationY"],
            "nominalFootprintMeters": list(nominal_footprint(template)),
            "expectedGridBoundsXZ": [
                footprint[0].tolist(), footprint[1].tolist()],
            "expectedGridOffset": expected_offset.tolist(),
            "appliedGridOffset": applied_offset.tolist(),
            "visualMeshAabb": [
                visual.minimum.tolist(), visual.maximum.tolist()],
            "collisionEntryIndex": (
                None if surface is None else surface.entry_index),
            "collisionMatchScore": rounded(match_score),
            "collisionSurfaceAabb": None if collision_bounds is None else [
                collision_bounds[0].tolist(), collision_bounds[1].tolist()],
            "overlapRatio": rounded(overlap),
            "centroidOffsetMeters": rounded(centroid_offset),
            "normalAngleDegrees": rounded(normal_angle),
            "principalAxisAngleDegrees": rounded(principal_angle),
            "rotationConsistent": (
                (normal_angle is None or normal_angle <= 15)
                and (principal_angle is None or principal_angle <= 15)
            ),
            "flagged": bool(reasons),
            "reasons": reasons,
        })
    return reports, placed


class VisualRayCaster:
    def __init__(self, placed):
        self.indices = []
        self.cells = defaultdict(list)
        for placed_index, (_, _, _, geometry) in enumerate(placed):
            index = PointRayIndex(geometry)
            self.indices.append(index)
            x0 = math.floor(geometry.minimum[0] / GRID_XZ)
            x1 = math.floor(geometry.maximum[0] / GRID_XZ)
            z0 = math.floor(geometry.minimum[2] / GRID_XZ)
            z1 = math.floor(geometry.maximum[2] / GRID_XZ)
            for x in range(x0, x1 + 1):
                for z in range(z0, z1 + 1):
                    self.cells[(x, z)].append(placed_index)

    def ray_down(self, x, z, origin_y):
        cell = (math.floor(x / GRID_XZ), math.floor(z / GRID_XZ))
        height = None
        for index in self.cells.get(cell, []):
            candidate = self.indices[index].ray_down(x, z, origin_y)
            if candidate is not None and (height is None or candidate > height):
                height = candidate
        return height


def lap_statistics(lap_path, collision, placed):
    scene = load_json(lap_path)
    lap = scene.get("lap")
    if (scene.get("format") != "tmnf-c-viewer-scene"
            or not isinstance(lap, dict)
            or not isinstance(lap.get("ticks"), list)):
        fail(f"{lap_path} is not a viewer replay scene")
    fields = lap.get("fields")
    required = {"x", "y", "z", "contactMask"}
    if not isinstance(fields, list) or not required.issubset(fields):
        fail(f"{lap_path} lacks required lap fields")
    field = {name: fields.index(name) for name in required}
    collision_rays = PointRayIndex(collision)
    visual_rays = VisualRayCaster(placed)
    results = []

    for tick, record in enumerate(lap["ticks"]):
        x = float(record[field["x"]])
        car_y = float(record[field["y"]])
        z = float(record[field["z"]])
        grounded = int(record[field["contactMask"]]) != 0
        origin_y = car_y + 8.0
        collision_height = collision_rays.ray_down(x, z, origin_y)
        visual_height = visual_rays.ray_down(x, z, origin_y)
        delta = (
            None
            if collision_height is None or visual_height is None
            else visual_height - collision_height
        )
        results.append({
            "tick": tick,
            "grounded": grounded,
            "carPosition": [x, car_y, z],
            "physicsContactHeight": rounded(collision_height, 6),
            "visualHitHeight": rounded(visual_height, 6),
            "deltaMeters": rounded(delta, 6),
        })

    def summarize(selected):
        matched = [
            result for result in selected
            if result["deltaMeters"] is not None
        ]
        absolute = [abs(result["deltaMeters"]) for result in matched]
        return {
            "ticks": len(selected),
            "matchedTicks": len(matched),
            "medianAbsDeltaMeters": rounded(percentile(absolute, 0.5)),
            "p95AbsDeltaMeters": rounded(percentile(absolute, 0.95)),
            "maxAbsDeltaMeters": rounded(max(absolute) if absolute else None),
        }

    grounded = [result for result in results if result["grounded"]]
    airborne = [result for result in results if not result["grounded"]]
    no_collision = [
        result["tick"] for result in results
        if result["physicsContactHeight"] is None
    ]
    no_visual_grounded = [
        result["tick"] for result in grounded
        if result["visualHitHeight"] is None
    ]
    no_visual_airborne = [
        result["tick"] for result in airborne
        if result["visualHitHeight"] is None
    ]
    violations = [
        result["tick"] for result in grounded
        if result["deltaMeters"] is not None
        and abs(result["deltaMeters"]) > 0.3
    ]
    return {
        "tickCount": len(results),
        "all": summarize(results),
        "grounded": summarize(grounded),
        "airborne": summarize(airborne),
        "noCollisionHitTicks": no_collision,
        "noVisualHitGroundedTicks": no_visual_grounded,
        "noVisualHitAirborneTicks": no_visual_airborne,
        "groundedDeltaOver0_3mTicks": violations,
        "ticks": results,
    }


def print_report(blocks, lap, collision, surface_count):
    flagged = [block for block in blocks if block["flagged"]]
    print("BLOCK-VS-COLLISION")
    print(
        f"placements={len(blocks)} flagged={len(flagged)} "
        f"collision_surfaces={surface_count} "
        f"collision_triangles={len(collision.triangles)}")
    print("index model rotation overlap centroid_m normal_deg axis_deg reasons")
    for block in flagged:
        rotation = int(round(
            -block["rotationY"] / (math.pi / 2))) % 4

        def value(name):
            item = block[name]
            return "-" if item is None else f"{item:.3f}"

        print(
            f"{block['placementIndex']:4d} {block['model']} "
            f"{rotation} {value('overlapRatio')} "
            f"{value('centroidOffsetMeters')} "
            f"{value('normalAngleDegrees')} "
            f"{value('principalAxisAngleDegrees')} "
            f"{'; '.join(block['reasons'])}")

    print("\nLAP-RAYCAST")
    print("set ticks matched median_abs_m p95_abs_m max_abs_m")
    for name in ("all", "grounded", "airborne"):
        row = lap[name]

        def stat(key):
            item = row[key]
            return "-" if item is None else f"{item:.4f}"

        print(
            f"{name:8s} {row['ticks']:5d} {row['matchedTicks']:7d} "
            f"{stat('medianAbsDeltaMeters'):>12s} "
            f"{stat('p95AbsDeltaMeters'):>9s} "
            f"{stat('maxAbsDeltaMeters'):>9s}")
    print(
        "no collision hit ticks: "
        f"{compact_ranges(lap['noCollisionHitTicks'])}")
    print(
        "no visual hit grounded ticks: "
        f"{compact_ranges(lap['noVisualHitGroundedTicks'])}")
    print(
        "no visual hit airborne ticks: "
        f"{compact_ranges(lap['noVisualHitAirborneTicks'])}")
    print(
        "|delta| > 0.3 m grounded ticks: "
        f"{compact_ranges(lap['groundedDeltaOver0_3mTicks'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "viewer/assets/game/manifest.json")
    parser.add_argument(
        "--placement", type=Path,
        default=ROOT / "viewer/assets/game/scenes/a01.json")
    parser.add_argument(
        "--collision", type=Path,
        default=ROOT / "oracle/tracks/A01-Race.tmnftrack")
    parser.add_argument(
        "--lap", type=Path,
        default=ROOT / "viewer/scenes/policy_lap.json")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    for path in (args.manifest, args.placement, args.collision, args.lap):
        if not path.is_file():
            fail(f"missing input {path}")

    collision, surfaces = read_collision_snapshot(args.collision)
    blocks, placed = validate_blocks(
        args.manifest.resolve(),
        args.placement.resolve(),
        surfaces,
    )
    lap = lap_statistics(args.lap.resolve(), collision, placed)
    report = {
        "format": "tmnf-visual-placement-validation",
        "version": 1,
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "placement": str(args.placement.resolve()),
            "collision": str(args.collision.resolve()),
            "lap": str(args.lap.resolve()),
        },
        "collision": {
            "activeSurfaces": len(surfaces),
            "triangles": len(collision.triangles),
        },
        "blockSummary": {
            "placements": len(blocks),
            "flagged": sum(block["flagged"] for block in blocks),
        },
        "blocks": blocks,
        "lap": lap,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
    print_report(blocks, lap, collision, len(surfaces))


if __name__ == "__main__":
    main()
