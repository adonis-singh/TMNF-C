"""Headless "game view" capture: render a viewer scene lap through the Three.js viewer
(real TrackMania block and car models) in headless Chromium into an h264 mp4.

    python -m tmnf_fly.fly_view <scene.json> <out.mp4> [--fps 50] [--size 1920 1080] [--supersample 2]
        [--camera chase|onboard|orbit] [--start-tick N --end-tick M]
        [--fly-filter] [--fly-filter-blend 0.5] [--still-time SECONDS]
        [--gl gl|swiftshader] [--taskset 22-29]

The viewer page is loaded once (own static server over viewer/, the scene mapped to
scenes/fly_view_scene.json); fly_view.mjs next to this file drives Chromium over the DevTools
protocol and, per frame, calls the viewer's window.__TMNF_APPLY_RENDER_PARAMETERS
with cam=px,py,pz,tx,ty,tz&tick=N&fov=F (its free "orbit" camera placed at a world
position), waits for the render, and reads the WebGL drawing buffer back. The chase
camera collides with the scene's collision mesh the way the game's does (see
chase_collide). Anti-aliasing is
--supersample x supersampling (rendered larger, box-downscaled in the page); MSAA is off
because on ANGLE/NVIDIA it dropped a draw in about one frame per 400. Camera paths are
computed here from lap.ticks. A still PNG (<out stem>.png) is written next to the
mp4. Scenes whose track has no game visuals in the manifest (Island) are rendered in
the viewer's collision-geometry mode. For Stadium scenes the served placement list is
corrected against the game's terrain composition (see patch_placements).
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import math
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import trimesh
from PIL import Image

from tmnf_fly import ROOT

VIEWER_DIR = ROOT / "viewer"
SEQUENCER = Path(__file__).with_name("fly_view.mjs")
SCENE_URL_PATH = "/scenes/fly_view_scene.json"
FRAME_SINK_PATH = "/fly_view/frame"
MANIFEST_URL_PATH = "/assets/game/manifest.json"
EMPTY_PLACEMENTS_URL_PATH = "/assets/game/scenes/fly_view_empty.json"
GRID_METERS = 32.0
ROW_METERS = 8.0
# Stadium terrain (see patch_placements): a terrain cell at row r has its surface at
# world y 8r+9. The flat filler block is one 32x32 m quad at that height; a DirtHill block
# is the slope from the level two rows below up to the cell's level (its model spans
# 16 m), but the exported placements put it 16 m too high, so it floats above its cell
# while the cell itself stays open to the sky.
TERRAIN_FILLER = "StadiumDirt|ground|0"
HILL_ASSET_PREFIX = "StadiumDirtHill|"
# Stadium terrain tiles; the game draws none of them in a cell a ground block occupies
TERRAIN_MODELS = frozenset({"StadiumDirt", "StadiumDirtBorder", "StadiumDirtHill", "StadiumWater", "StadiumPool"})
HILL_Y_OFFSET = -2 * ROW_METERS
# A ground block occupies a footprint cell (and replaces the terrain filler under it) when
# its model covers the cell seen from above; unoccupied cells of a multi-cell block's
# bounding footprint (the inner corner of a 3x3 curve, a castle gate's passage) keep
# their filler. Coverage is rasterised at 1 m; occupied cells measure 1.00, open ones
# 0.00 or 0.56.
CELL_COVERAGE_MIN = 0.9
COVERAGE_SAMPLES_PER_CELL = 32
# Default terrain (see grass_gltf / patch_placements): the Stadium map grid is 32x32
# cells; the game draws plain grass at the ground level (y 9) on every cell the mapper
# left empty. The export has no block for it, so one is synthesised: a 32x32 quad with
# the texture patch the game's flat grass caps sample (StadiumGrassOcc quad of the
# inflatable blocks, ../textures/stadiumgrass1.png).
STADIUM_CELLS = 32
TERRAIN_LEVEL = 9.0
GRASS_ASSET = "StadiumGrass|ground|0"
GRASS_MODEL_URL = "models/fly_view_grass.gltf"
GRASS_TEXTURE_URL = "../textures/stadiumgrass1.png"
GRASS_UV = (0.565, 0.190, 0.622, 0.246)
# A cell without blocks whose collision geometry near ground level is not the flat ground
# (a road trench under a checkpoint the export lacks a model for) is not plain terrain.
TERRAIN_BAND = 4.0
SURFACE_TOLERANCE = 0.3
SURFACE_HEADROOM = 2.0
SURFACE_PROBE_HEIGHT = 50.0
SURFACE_MATCH_MIN = 0.9

CAMERAS = {
    # back, up, target ahead, target up, fov, heading smoothing rate (1/s)
    "chase": dict(back=8.4, up=3.15, ahead=6.5, target_up=0.65, fov=60.0, rate=4.5),
    # Chase collision (the game's behaviour): the camera is pulled in along the ray from
    # the car centre to the first collision hit minus CAMERA_CLEARANCE, kept at least
    # CAMERA_FLOOR above the surface below it, pulled further in while it is enclosed in a
    # block volume, and the pull-in distance is smoothed at CAMERA_RELEASE_RATE when
    # releasing outward (pulling in is immediate).
    "onboard": dict(cam_local=(0.0, 1.32, -0.3), target_local=(0.0, 0.9, 30.0), fov=80.0),
    "orbit": dict(radius=11.0, height=4.0, target_up=0.7, fov=54.0, rate=0.25),
}
CAMERA_CLEARANCE = 0.4
CAMERA_FLOOR = 1.0
CAMERA_RELEASE_RATE = 3.0
CAMERA_PULL_STEP = 0.5
CAMERA_ENCLOSURE_REACH = 40.0


# --- scene -------------------------------------------------------------------


def load_scene(path: Path) -> dict:
    scene = json.loads(path.read_text(encoding="utf-8"))
    if scene.get("format") != "tmnf-c-viewer-scene" or scene.get("version") != 3:
        raise SystemExit(f"{path} is not a version 3 tmnf-c-viewer-scene")
    return scene


def lap_arrays(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """Positions (N,3) and Three.js quaternions (N,4: x,y,z,w). The lap record stores the
    quaternion as w,x,y,z under the field names qx,qy,qz,qw; the viewer's gameQuaternion
    reads (qy,qz,qw,qx) as (x,y,z,w)."""
    lap = scene["lap"]
    fields = lap["fields"]
    ticks = np.asarray(lap["ticks"], dtype=np.float64)
    ix = [fields.index(name) for name in ("x", "y", "z")]
    iq = [fields.index(name) for name in ("qy", "qz", "qw", "qx")]
    quats = ticks[:, iq]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return ticks[:, ix], quats


class TrackCollision:
    """Ray queries against the scene's collision mesh (embree through trimesh)."""

    def __init__(self, scene: dict):
        if not trimesh.ray.has_embree:
            raise SystemExit("trimesh has no embree backend (embreex)")
        track = scene["track"]
        positions = np.asarray(track["positions"], dtype=np.float64).reshape(-1, 3)
        indices = np.asarray(track["indices"], dtype=np.int64).reshape(-1, 3)
        self.mesh = trimesh.Trimesh(vertices=positions, faces=indices, process=False)
        self.material = np.zeros(len(indices), dtype=int)
        for start, count, material in track["groups"]:
            self.material[start // 3:(start + count) // 3] = material

    def first_hit(self, origin: np.ndarray, direction: np.ndarray, reach: float) -> tuple[float, int] | None:
        """(distance, face index) of the nearest face within reach along the unit direction."""
        location, _, faces = self.mesh.ray.intersects_location(origin[None, :], direction[None, :],
                                                               multiple_hits=False)
        if len(faces) == 0:
            return None
        distance = float(np.linalg.norm(location[0] - origin))
        return (distance, int(faces[0])) if distance <= reach else None

    def enclosed(self, point: np.ndarray) -> bool:
        """Inside a block volume: the ray up meets the back of an upward-facing face and
        the ray down meets a face."""
        up = self.first_hit(point, UP, CAMERA_ENCLOSURE_REACH)
        return (up is not None and self.mesh.face_normals[up[1]][1] > 0
                and self.first_hit(point, -UP, CAMERA_ENCLOSURE_REACH) is not None)


UP = np.array([0.0, 1.0, 0.0])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def chase_collide(collision: TrackCollision, origin: np.ndarray, cam: np.ndarray, released: float,
                  dt: float) -> tuple[np.ndarray, float]:
    """Camera position after collision against the track, and the smoothed pull-in
    fraction to carry to the next frame."""
    to_cam = cam - origin
    length = float(np.linalg.norm(to_cam))
    direction = to_cam / length
    hit = collision.first_hit(origin, direction, length)
    fraction = max(hit[0] - CAMERA_CLEARANCE, 0.0) / length if hit else 1.0
    released = released + (fraction - released) * (1.0 - math.exp(-dt * CAMERA_RELEASE_RATE))
    fraction = min(fraction, released)
    while True:
        cam = origin + direction * (length * fraction)
        below = collision.first_hit(cam, -UP, CAMERA_ENCLOSURE_REACH)
        if below is not None:
            cam[1] = max(cam[1], cam[1] - below[0] + CAMERA_FLOOR)
        if not collision.enclosed(cam) or length * fraction <= CAMERA_PULL_STEP:
            return cam, min(released, fraction)
        fraction -= CAMERA_PULL_STEP / length


def camera_path(mode: str, positions: np.ndarray, quats: np.ndarray, ticks: list[int],
                dt: float, collision: TrackCollision) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Per frame (camera position, look-at target, fov) in world space."""
    spec = CAMERAS[mode]
    out = []
    forward_local = np.array([0.0, 0.0, 1.0])
    heading = None
    released = 1.0
    for index, tick in enumerate(ticks):
        p = positions[tick]
        q = quats[tick]
        if mode == "chase":
            f = rotate(q, forward_local)
            f[1] *= 0.35
            f /= np.linalg.norm(f)
            if heading is None:
                heading = f
            else:
                blend = 1.0 - math.exp(-dt * spec["rate"])
                heading = heading + (f - heading) * blend
                heading /= np.linalg.norm(heading)
            cam = p - heading * spec["back"] + np.array([0.0, spec["up"], 0.0])
            target = p + heading * spec["ahead"] + np.array([0.0, spec["target_up"], 0.0])
            cam, released = chase_collide(collision, p + np.array([0.0, spec["target_up"], 0.0]), cam, released, dt)
        elif mode == "onboard":
            cam = p + rotate(q, np.asarray(spec["cam_local"]))
            target = p + rotate(q, np.asarray(spec["target_local"]))
        else:
            angle = spec["rate"] * index * dt
            cam = p + np.array([math.sin(angle) * spec["radius"], spec["height"], math.cos(angle) * spec["radius"]])
            target = p + np.array([0.0, spec["target_up"], 0.0])
        out.append((cam, target, spec["fov"]))
    return out


def frame_params(tick: int, cam: np.ndarray, target: np.ndarray, fov: float) -> str:
    values = ",".join(f"{v:.4f}" for v in (*cam, *target))
    return f"cam={values}&tick={tick}&fov={fov:g}"


# --- static server -----------------------------------------------------------


class FlyViewHandler(SimpleHTTPRequestHandler):
    """viewer/ as the document root (symlinked assets followed) plus the scene under
    scenes/fly_view_scene.json and, for tracks without game visuals, a manifest that
    names an empty placement list for the track. POST /fly_view/frame/<index> is the
    frame sink: the page posts each frame's raw RGBA pixels here."""

    protocol_version = "HTTP/1.1"
    overrides: dict[str, Path] = {}
    frames: queue.Queue

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean in self.overrides:
            return str(self.overrides[clean])
        return super().translate_path(clean)

    def do_POST(self) -> None:
        prefix = FRAME_SINK_PATH + "/"
        if not self.path.startswith(prefix):
            self.send_error(404)
            return
        index = int(self.path[len(prefix):])
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        if len(body) != length:
            self.send_error(400, f"frame {index}: {len(body)} of {length} bytes")
            return
        self.frames.put((index, body))
        self.send_response(204)
        self.end_headers()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


def gltf_triangles(gltf_path: Path) -> np.ndarray:
    """All triangles (N, 3, 3) of a block glTF in placement-local coordinates (node
    matrices applied)."""
    model = json.loads(gltf_path.read_text(encoding="utf-8"))
    buffers = [gltf_path.with_name(b["uri"]).read_bytes() for b in model["buffers"]]
    dtypes = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(index: int) -> np.ndarray:
        acc = model["accessors"][index]
        view = model["bufferViews"][acc["bufferView"]]
        if "byteStride" in view:
            raise SystemExit(f"{gltf_path}: strided buffer views are not handled")
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        width = widths[acc["type"]]
        data = np.frombuffer(buffers[view["buffer"]], dtype=dtypes[acc["componentType"]],
                             count=acc["count"] * width, offset=offset)
        return data.reshape(acc["count"], width)

    triangles = []

    def walk(node_index: int, parent: np.ndarray) -> None:
        node = model["nodes"][node_index]
        matrix = parent @ np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T if "matrix" in node else parent
        if "mesh" in node:
            for primitive in model["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise SystemExit(f"{gltf_path}: non-triangle primitive")
                positions = accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
                indices = accessor(primitive["indices"]).ravel()
                homogeneous = np.c_[positions, np.ones(len(positions))] @ matrix.T
                triangles.append(homogeneous[indices, :3].reshape(-1, 3, 3))
        for child in node.get("children", []):
            walk(child, matrix)

    for root_index in model["scenes"][model.get("scene", 0)]["nodes"]:
        walk(root_index, np.eye(4))
    return np.concatenate(triangles)


@functools.lru_cache(maxsize=None)
def occupied_local_cells(gltf_path: Path, width: float, depth: float) -> dict[tuple[int, int], float]:
    """Footprint cells (i, j) of a block model covered from above by its geometry, each
    with the model's highest local y over that cell."""
    triangles = gltf_triangles(gltf_path)
    step = GRID_METERS / COVERAGE_SAMPLES_PER_CELL
    xs = np.arange(step / 2, width, step)
    zs = np.arange(step / 2, depth, step)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    px, pz = grid_x.ravel(), grid_z.ravel()
    covered = np.zeros(px.shape, dtype=bool)
    for a, b, c in triangles[:, :, [0, 2]]:
        d1 = (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (pz - b[1])
        d2 = (px - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (pz - c[1])
        d3 = (px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (pz - a[1])
        covered |= ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
    covered = covered.reshape(len(xs), len(zs))
    n = COVERAGE_SAMPLES_PER_CELL
    lo = triangles[:, :, [0, 2]].min(axis=1)
    hi = triangles[:, :, [0, 2]].max(axis=1)
    top = triangles[:, :, 1].max(axis=1)
    cells = {}
    for i in range(int(width // GRID_METERS)):
        for j in range(int(depth // GRID_METERS)):
            if covered[i * n:(i + 1) * n, j * n:(j + 1) * n].mean() < CELL_COVERAGE_MIN:
                continue
            x0, z0 = i * GRID_METERS, j * GRID_METERS
            inside = (hi[:, 0] > x0) & (lo[:, 0] < x0 + GRID_METERS) & (hi[:, 1] > z0) & (lo[:, 1] < z0 + GRID_METERS)
            cells[(i, j)] = float(top[inside].max())
    return cells


def world_cells(placement: dict, local_cells: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Grid cells (x, z) the given footprint cells land on, following the viewer's
    placement matrix (rotation about y by rotationY, then translation to position),
    each with the block's top in world y."""
    theta = placement["rotationY"]
    px, py, pz = placement["position"]
    cells = {}
    for (i, j), top in local_cells.items():
        x = GRID_METERS * (i + 0.5)
        z = GRID_METERS * (j + 0.5)
        wx = px + x * math.cos(theta) + z * math.sin(theta)
        wz = pz - x * math.sin(theta) + z * math.cos(theta)
        cells[(math.floor(wx / GRID_METERS), math.floor(wz / GRID_METERS))] = py + top
    return cells


def grass_gltf() -> dict:
    """The default grass terrain block: one 32x32 m quad at the terrain level, buffer
    embedded, texture shared with the game's grass caps."""
    u0, v0, u1, v1 = GRASS_UV
    positions = np.array([[0, TERRAIN_LEVEL, 0], [0, TERRAIN_LEVEL, GRID_METERS],
                          [GRID_METERS, TERRAIN_LEVEL, GRID_METERS], [GRID_METERS, TERRAIN_LEVEL, 0]], dtype=np.float32)
    normals = np.tile(np.array([0, 1, 0], dtype=np.float32), (4, 1))
    uvs = np.array([[u0, v0], [u0, v1], [u1, v1], [u1, v0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    blobs = [positions.tobytes(), normals.tobytes(), uvs.tobytes(), indices.tobytes()]
    views, offset = [], 0
    for blob, target in zip(blobs, (34962, 34962, 34962, 34963)):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": target})
        offset += len(blob)
    buffer = b"".join(blobs)
    return {
        "asset": {"version": "2.0", "generator": "tmnf_fly.fly_view"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "StadiumGrass", "mesh": 0}],
        "meshes": [{"name": "StadiumGrass", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0, "mode": 4}]}],
        "materials": [{"name": "StadiumGrassOcc", "doubleSided": True, "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1, "baseColorTexture": {"index": 0}}}],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"uri": GRASS_TEXTURE_URL}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": 6, "type": "SCALAR", "min": [0], "max": [3]},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(buffer),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(buffer).decode("ascii")}],
    }


def uneven_cells(scene: dict) -> set[tuple[int, int]]:
    """Grid cells where the collision mesh has geometry near ground level other than the
    flat ground itself (sunken road, banks)."""
    vertices = np.asarray(scene["track"]["positions"], dtype=np.float64).reshape(-1, 3)
    y = vertices[:, 1]
    near = (np.abs(y - TERRAIN_LEVEL) <= TERRAIN_BAND) & (np.abs(y - TERRAIN_LEVEL) > 0.05)
    cells = np.floor(vertices[near][:, [0, 2]] / GRID_METERS).astype(int)
    inside = (cells >= 0).all(axis=1) & (cells < STADIUM_CELLS).all(axis=1)
    return {(int(x), int(z)) for x, z in cells[inside]}


def grass_placement(cell: tuple[int, int]) -> dict:
    x, z = cell
    return {"blockIndex": -1, "asset": GRASS_ASSET, "model": "StadiumGrass", "coord": [x, 0, z], "direction": 0,
            "ground": True, "variant": 0, "subVariant": 0, "position": [x * GRID_METERS, 0.0, z * GRID_METERS],
            "rotationY": 0.0, "gridOffset": [0, 0, 0], "nominalFootprint": [GRID_METERS, GRID_METERS]}


def surface_match(collision: TrackCollision, gltf_path: Path, position: np.ndarray, rotation_y: float,
                  ground_level: float) -> float:
    """Fraction of a block model's upward faces near ground level (below ground_level +
    SURFACE_HEADROOM) whose height agrees with the collision mesh within SURFACE_TOLERANCE
    when placed at position/rotation_y. Faces under collision geometry standing above
    ground level are not counted: that is the superstructure of the block stood in for."""
    triangles = gltf_triangles(gltf_path)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centres = triangles.mean(axis=1)[normals[:, 1] > 0.7 * np.linalg.norm(normals, axis=1)]
    cos, sin = math.cos(rotation_y), math.sin(rotation_y)
    world = np.stack([position[0] + centres[:, 0] * cos + centres[:, 2] * sin, position[1] + centres[:, 1],
                      position[2] - centres[:, 0] * sin + centres[:, 2] * cos], axis=1)
    world = world[world[:, 1] < ground_level + SURFACE_HEADROOM]
    origins = world + np.array([0.0, SURFACE_PROBE_HEIGHT, 0.0])
    location, ray, _ = collision.mesh.ray.intersects_location(origins, np.tile(-UP, (len(origins), 1)), multiple_hits=False)
    heights = np.full(len(world), np.nan)
    heights[ray] = location[:, 1]
    counted = ~(heights > ground_level + SURFACE_TOLERANCE)
    return float((np.abs(world[counted, 1] - heights[counted]) < SURFACE_TOLERANCE).mean())


def footprint_position(cell: tuple[int, int], row: int, rotation_y: float, footprint: tuple[float, float]) -> np.ndarray:
    """Placement position that keeps a footprint rotated by rotation_y inside its cell."""
    cos, sin = math.cos(rotation_y), math.sin(rotation_y)
    corners = np.array([[0.0, 0.0], [footprint[0], 0.0], [0.0, footprint[1]], list(footprint)])
    rotated_x = corners[:, 0] * cos + corners[:, 1] * sin
    rotated_z = -corners[:, 0] * sin + corners[:, 1] * cos
    return np.array([GRID_METERS * cell[0] - rotated_x.min(), ROW_METERS * row, GRID_METERS * cell[1] - rotated_z.min()])


def checkpoint_substitutes(placements: list[dict], manifest: dict, game_dir: Path, scene: dict,
                           collision: "TrackCollision") -> list[tuple[str, dict]]:
    """Checkpoint blocks the export dropped (their asset has no visual) rebuilt from the
    scene's route triggers, each as (missing asset, stand-in placement).

    The trigger carries the game's block transform (position, quarter turn) and the
    gate box, whose thin axis is the road direction. The block family is the longest
    model-name prefix shared with the ground blocks on either side of the gate that
    names a <family>Checkpoint asset; the stand-in is the family variant and quarter
    turn whose ground surface best matches the collision mesh in the cell (the
    checkpoint without its gantry), the trigger's own turn winning ties."""
    indices = {placement["blockIndex"] for placement in placements}
    by_cell: dict[tuple[int, int, int], list[dict]] = {}
    terrain_row: dict[tuple[int, int], int] = {}
    for placement in placements:
        by_cell.setdefault(tuple(placement["coord"]), []).append(placement)
        if placement["model"] in TERRAIN_MODELS:
            terrain_row[(placement["coord"][0], placement["coord"][2])] = placement["coord"][1]
    substitutes = []
    for trigger in scene["route"]["checkpoints"]:
        if trigger["blockIndex"] in indices:
            continue
        rotation = np.asarray(trigger["transform"]["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(trigger["transform"]["translation"], dtype=np.float64)
        centre = rotation @ np.asarray(trigger["box"]["center"]) + translation
        x, z = int(centre[0] // GRID_METERS), int(centre[2] // GRID_METERS)
        row = int(round(translation[1] / ROW_METERS))
        half = trigger["box"]["halfExtent"]
        axis = rotation @ (np.array([0.0, 0.0, 1.0]) if half[2] < half[0] else np.array([1.0, 0.0, 0.0]))
        dx, dz = int(round(axis[0])), int(round(axis[2]))
        ground = row == terrain_row.get((x, z), 0) + 1
        side = "ground" if ground else "air"
        neighbours = by_cell.get((x - dx, row, z - dz), []) + by_cell.get((x + dx, row, z + dz), [])
        families = {
            neighbour["model"][:length]
            for neighbour in neighbours if neighbour["ground"] == ground
            for length in range(len(neighbour["model"]), 0, -1)
            if f"{neighbour['model'][:length]}Checkpoint|{side}|0" in manifest["assets"]
        }
        if not families:
            raise SystemExit(f"route checkpoint block {trigger['blockIndex']} at cell ({x}, {row}, {z}) has no "
                             f"exported placement and no road family among its neighbours {neighbours}")
        family = max(families, key=len)
        missing = f"{family}Checkpoint|{side}|0"
        trigger_turn = int(round(-math.atan2(rotation[0, 2], rotation[0, 0]) / (math.pi / 2))) % 4
        ground_level = ROW_METERS * (row - 1) + TERRAIN_LEVEL
        footprint = (GRID_METERS, GRID_METERS)
        candidates = []
        for key, asset in manifest["assets"].items():
            if not key.startswith(f"{family}|{side}|") or not asset["visual"]:
                continue
            for turn in range(4):
                rotation_y = -turn * math.pi / 2
                position = footprint_position((x, z), row, rotation_y, footprint)
                score = surface_match(collision, game_dir / asset["visual"], position, rotation_y, ground_level)
                candidates.append((score, turn == trigger_turn, -int(key.rsplit("|", 1)[1]), key, turn, position))
        if not candidates:
            raise SystemExit(f"{missing} has no {family}|{side} block to stand in")
        score, _, variant, key, turn, position = max(candidates)
        if score < SURFACE_MATCH_MIN:
            raise SystemExit(f"no {family}|{side} variant matches the collision at cell ({x}, {row}, {z}) "
                             f"for {missing}: best {key} turn {turn} at {score:.2f}")
        substitutes.append((missing, {
            "blockIndex": trigger["blockIndex"], "asset": key, "model": family, "coord": [x, row, z],
            "direction": turn, "ground": ground, "variant": -variant, "subVariant": 0,
            "position": position.tolist(), "rotationY": -turn * math.pi / 2, "gridOffset": [0, 0, 0],
            "nominalFootprint": list(footprint), "surfaceMatch": round(score, 3)}))
    return substitutes


def patch_placements(placements: dict, manifest: dict, game_dir: Path, scene: dict) -> tuple[dict, str]:
    """Correct the exported placements against the game's terrain composition.

    Stadium terrain is a heightfield of 32x32 m cells; a cell at row r has its surface
    at y 8r+9 and holds exactly one terrain block at row r: the flat filler quad, a
    dirt/grass border edge, or a DirtHill slope from the level two rows below up to the
    cell's level. Ground blocks (roads, inflatables, tents) sit at row r+1 on top of
    that surface and carry their own ground cap, banks and slopes, so the game draws no
    terrain in the cells they occupy. The corrections, all checked against the scene's
    collision mesh:

    * DirtHill placements are lowered by two rows: exported 16 m too high, they float
      above their cell as slabs while the slope itself is missing (the cell shows the
      sky sphere below the horizon).
    * Terrain blocks (TERRAIN_MODELS) are dropped in cells where a ground block's
      volume spans their height range (a road at the row above, a two-row slope block
      rising through the plateau level, a road-to-grass transition over a border edge);
      kept, they cap the sunken dirt road (the chase camera sees no car), roof the
      climbing road, or z-fight the block's own cap.
    * Checkpoint blocks the export dropped for lack of a model are rebuilt from the
      route triggers with the gantry-less block of their road family whose surface
      matches the collision standing in (see checkpoint_substitutes).
    * Every grid cell left without a terrain block or a ground block over it gets the
      default grass block, except cells whose collision geometry shows the terrain is
      not the flat ground there (a block the export has no model for)."""
    hills = 0
    # cell -> [(block base y, block top y)] of the ground blocks occupying it
    covering: dict[tuple[int, int], list[tuple[float, float]]] = {}
    patched = []

    def cover(placement: dict) -> None:
        gltf_path = game_dir / manifest["assets"][placement["asset"]]["visual"]
        local = occupied_local_cells(gltf_path, *placement["nominalFootprint"])
        for cell, top in world_cells(placement, local).items():
            covering.setdefault(cell, []).append((placement["position"][1], top))

    def is_terrain(placement: dict) -> bool:
        return placement["model"] in TERRAIN_MODELS

    for placement in placements["placements"]:
        placement = dict(placement)
        if max(placement["coord"][0], placement["coord"][2]) >= STADIUM_CELLS:
            raise SystemExit(f"placement outside the {STADIUM_CELLS}x{STADIUM_CELLS} Stadium grid: {placement}")
        if placement["asset"].startswith(HILL_ASSET_PREFIX):
            position = list(placement["position"])
            position[1] += HILL_Y_OFFSET
            placement["position"] = position
            hills += 1
        if placement["ground"] and not is_terrain(placement):
            cover(placement)
        patched.append(placement)
    substituted = []
    for missing, placement in checkpoint_substitutes(patched, manifest, game_dir, scene, TrackCollision(scene)):
        if placement["ground"]:
            cover(placement)
        patched.append(placement)
        substituted.append(f"{missing} -> {placement['asset']} turn {placement['direction']} at "
                           f"{tuple(placement['coord'])} (surface match {placement['surfaceMatch']})")
    kept = []
    dropped: list[str] = []
    terrain = set(covering)
    for placement in patched:
        if is_terrain(placement):
            cell = (placement["coord"][0], placement["coord"][2])
            heights = gltf_triangles(game_dir / manifest["assets"][placement["asset"]]["visual"])[:, :, 1]
            bottom = placement["position"][1] + float(heights.min())
            surface = placement["position"][1] + float(heights.max())
            if any(base <= surface and top >= bottom for base, top in covering.get(cell, [])):
                dropped.append(f"{placement['asset']}@{tuple(placement['coord'])}")
                continue
            terrain.add(cell)
        kept.append(placement)
    uneven = uneven_cells(scene)
    empty = {(x, z) for x in range(STADIUM_CELLS) for z in range(STADIUM_CELLS)} - terrain
    skipped = sorted(empty & uneven)
    grass = sorted(empty - uneven)
    kept.extend(grass_placement(cell) for cell in grass)
    fillers_dropped = sum(name.startswith(TERRAIN_FILLER) for name in dropped)
    others_dropped = [name for name in dropped if not name.startswith(TERRAIN_FILLER)]
    return {**placements, "placements": kept}, (
        f"hills lowered {hills}, terrain blocks dropped in ground-block cells {len(dropped)} "
        f"({fillers_dropped} fillers, others {others_dropped}), "
        f"checkpoints substituted {substituted}, grass added {len(grass)}, empty cells left uneven {skipped}")


def start_server(scene_path: Path, scene: dict, scratch: Path,
                 frames: queue.Queue) -> tuple[ThreadingHTTPServer, bool]:
    """Returns (server, has_game_visuals)."""
    scene_id = scene["gameVisuals"]["scene"]
    game_dir = VIEWER_DIR / "assets" / "game"
    manifest = json.loads((game_dir / "manifest.json").read_text(encoding="utf-8"))
    overrides = {SCENE_URL_PATH: scene_path.resolve()}
    has_visuals = scene_id in manifest["scenes"]
    if has_visuals:
        placements_url = manifest["scenes"][scene_id]["url"]
        placements = json.loads((game_dir / placements_url).read_text(encoding="utf-8"))
        patched, summary = patch_placements(placements, manifest, game_dir, scene)
        patched_path = scratch / "placements.json"
        patched_path.write_text(json.dumps(patched), encoding="utf-8")
        overrides[f"/assets/game/{placements_url}"] = patched_path
        print(f"placements patched: {summary}", flush=True)
        grass_path = scratch / "fly_view_grass.gltf"
        grass_path.write_text(json.dumps(grass_gltf()), encoding="utf-8")
        overrides[f"/assets/game/{GRASS_MODEL_URL}"] = grass_path
        manifest["assets"][GRASS_ASSET] = {"visual": GRASS_MODEL_URL}
    else:
        manifest["scenes"][scene_id] = {"url": "scenes/fly_view_empty.json", "placements": 0}
        empty = scratch / "fly_view_empty.json"
        empty.write_text(json.dumps({"format": "tmnf-game-visual-scene", "version": 2, "placements": []}),
                         encoding="utf-8")
        overrides[EMPTY_PLACEMENTS_URL_PATH] = empty
    manifest_path = scratch / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    overrides[MANIFEST_URL_PATH] = manifest_path
    handler = type("Handler", (FlyViewHandler,), {"overrides": overrides, "frames": frames})
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(handler, directory=str(VIEWER_DIR)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, has_visuals


# --- compound-eye filter -----------------------------------------------------


class FlyFilter:
    """Hexagonal compound-eye mosaic: every hex cell (flat-to-flat width cell_frac of the
    frame width) takes the mean colour of its pixels; a faint hex grid and a vignette
    on top. blend=1 is the pure mosaic, blend=0.5 half mosaic half original."""

    def __init__(self, width: int, height: int, cell_frac: float = 0.012, blend: float = 1.0,
                 grid_strength: float = 0.22, vignette_strength: float = 0.38):
        self.blend = blend
        size = cell_frac * width / math.sqrt(3.0)  # circumradius of a pointy-top hex
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
        xs += 0.5
        ys += 0.5
        q = (xs * (math.sqrt(3.0) / 3.0) - ys / 3.0) / size
        r = (ys * 2.0 / 3.0) / size
        s = -q - r
        rq, rr, rs = np.rint(q), np.rint(r), np.rint(s)
        dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
        fix_q = (dq > dr) & (dq > ds)
        fix_r = ~fix_q & (dr > ds)
        rq = np.where(fix_q, -rr - rs, rq)
        rr = np.where(fix_r, -rq - rs, rr)
        rq = rq.astype(np.int64) - int(rq.min())
        rr = rr.astype(np.int64) - int(rr.min())
        labels = rr * (int(rq.max()) + 1) + rq
        _, self.labels = np.unique(labels, return_inverse=True)
        self.labels = self.labels.reshape(height, width)
        self.cells = int(self.labels.max()) + 1
        self.counts = np.bincount(self.labels.ravel(), minlength=self.cells).astype(np.float32)
        edge = np.zeros((height, width), dtype=bool)
        edge[:, 1:] |= self.labels[:, 1:] != self.labels[:, :-1]
        edge[1:, :] |= self.labels[1:, :] != self.labels[:-1, :]
        gain = np.ones((height, width), dtype=np.float32)
        gain[edge] = 1.0 - grid_strength
        cx, cy = width / 2.0, height / 2.0
        d = np.hypot((xs - cx) / cx, (ys - cy) / cy) / math.sqrt(2.0)
        gain *= (1.0 - vignette_strength * d ** 2.2).astype(np.float32)
        self.gain = gain[..., None]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        flat = rgb.reshape(-1, 3).astype(np.float32)
        labels = self.labels.ravel()
        means = np.stack([np.bincount(labels, weights=flat[:, c], minlength=self.cells) for c in range(3)], axis=1)
        means = (means / self.counts[:, None]).astype(np.float32)
        mosaic = means[self.labels]
        if self.blend < 1.0:
            mosaic = mosaic * self.blend + rgb.astype(np.float32) * (1.0 - self.blend)
        return np.clip(mosaic * self.gain, 0.0, 255.0).astype(np.uint8)


# --- capture -----------------------------------------------------------------


def next_frame(frames: queue.Queue, node: subprocess.Popen, index: int) -> bytes:
    while True:
        try:
            got_index, body = frames.get(timeout=1.0)
        except queue.Empty:
            if node.poll() is not None:
                raise SystemExit(f"fly_view.mjs exited with {node.returncode} before frame {index}")
            continue
        if got_index != index:
            raise SystemExit(f"frame sink received frame {got_index}, expected {index}")
        return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    parser.add_argument("--supersample", type=int, default=2,
                        help="render at this multiple of --size and box-downscale (anti-aliasing; no MSAA)")
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="chase")
    parser.add_argument("--start-tick", type=int, default=0)
    parser.add_argument("--end-tick", type=int, default=None, help="exclusive; default tickCount")
    parser.add_argument("--fly-filter", action="store_true")
    parser.add_argument("--fly-filter-blend", type=float, default=1.0)
    parser.add_argument("--still-time", type=float, default=None,
                        help="seconds into the clip for <out stem>.png; default midpoint")
    parser.add_argument("--gl", choices=("gl", "swiftshader"), default="gl",
                        help="gl: ANGLE on the GPU's OpenGL; swiftshader: CPU")
    parser.add_argument("--taskset", default=None, help="CPU list for taskset, e.g. 22-29")
    parser.add_argument("--chrome", default="/usr/bin/chromium")
    args = parser.parse_args()

    width, height = args.size
    if width % 2 or height % 2:
        raise SystemExit("--size must be even for yuv420p")
    scene = load_scene(args.scene)
    lap = scene["lap"]
    stride = 1000 / (args.fps * lap["tickMs"])
    if stride != int(stride) or stride < 1:
        raise SystemExit(f"--fps {args.fps} is not an integer tick stride at {lap['tickMs']} ms/tick")
    stride = int(stride)
    end_tick = lap["tickCount"] if args.end_tick is None else args.end_tick
    if not 0 <= args.start_tick < end_tick <= lap["tickCount"]:
        raise SystemExit(f"tick range {args.start_tick}..{end_tick} outside 0..{lap['tickCount']}")
    ticks = list(range(args.start_tick, end_tick, stride))
    positions, quats = lap_arrays(scene)
    dt = 1.0 / args.fps
    path = camera_path(args.camera, positions, quats, ticks, dt, TrackCollision(scene))
    frames = [frame_params(tick, cam, target, fov) for tick, (cam, target, fov) in zip(ticks, path)]
    still_time = (len(frames) / 2) / args.fps if args.still_time is None else args.still_time
    still_index = int(round(still_time * args.fps))
    if not 0 <= still_index < len(frames):
        raise SystemExit(f"--still-time {still_time} outside the clip's {len(frames) / args.fps:.2f} s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    still_path = args.out.with_suffix(".png")
    fly = FlyFilter(width, height, blend=args.fly_filter_blend) if args.fly_filter else None

    scratch = Path(tempfile.mkdtemp(prefix="fly_view_"))
    frames_queue: queue.Queue = queue.Queue(maxsize=32)
    server, has_visuals = start_server(args.scene, scene, scratch, frames_queue)
    port = server.server_address[1]
    visual = "game" if has_visuals else "collision"
    print(f"scene {args.scene} track {scene['gameVisuals']['scene']}: game visuals {'present' if has_visuals else 'absent, collision-geometry mode'}; "
          f"{len(frames)} frames at {args.fps} fps ({len(frames) / args.fps:.2f} s), camera {args.camera}, {width}x{height}",
          flush=True)
    frames_path = scratch / "frames.json"
    frames_path.write_text(json.dumps(frames), encoding="utf-8")
    base = f"http://127.0.0.1:{port}"
    url = f"{base}/?scene={SCENE_URL_PATH[1:]}&hud=0&overlays=0"
    command = ["node", str(SEQUENCER), "--chrome", args.chrome, "--url", url,
               "--sink", base + FRAME_SINK_PATH, "--width", str(width), "--height", str(height),
               "--supersample", str(args.supersample),
               "--gl", args.gl, "--frames", str(frames_path), "--visual", visual]
    if args.taskset:
        command = ["taskset", "-c", args.taskset, *command]
    node = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=sys.stderr, cwd=str(ROOT), text=True)
    info_line = node.stdout.readline()
    if not info_line:
        raise SystemExit(f"fly_view.mjs exited with {node.wait()} before the page was ready")
    info = json.loads(info_line)
    print(f"WebGL renderer: {info['renderer']} (page ready in {info['loadMs']} ms)", flush=True)

    writer = imageio_ffmpeg.write_frames(
        str(args.out), size=(width, height), fps=args.fps, codec="libx264", quality=None,
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-crf", "18", "-preset", "medium"], ffmpeg_log_level="error",
    )
    writer.send(None)
    started = time.monotonic()
    filter_seconds = 0.0
    for index in range(len(frames)):
        raw = next_frame(frames_queue, node, index)
        if len(raw) != width * height * 4:
            raise SystemExit(f"frame {index} is {len(raw)} bytes, wanted {width * height * 4}")
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        if fly is not None:
            t0 = time.monotonic()
            rgb = fly.apply(rgb)
            filter_seconds += time.monotonic() - t0
        if index == still_index:
            Image.fromarray(rgb).save(still_path)
        writer.send(np.ascontiguousarray(rgb).tobytes())
        if (index + 1) % 100 == 0 or index + 1 == len(frames):
            elapsed = time.monotonic() - started
            print(f"  {index + 1}/{len(frames)} frames, {(index + 1) / elapsed:.2f} fps end-to-end", flush=True)
    writer.close()
    if node.wait() != 0:
        raise SystemExit(f"fly_view.mjs exited with {node.returncode}")
    server.shutdown()
    shutil.rmtree(scratch, ignore_errors=True)
    elapsed = time.monotonic() - started
    print(f"wrote {args.out} ({len(frames)} frames, {len(frames) / args.fps:.2f} s, {width}x{height}, "
          f"{len(frames) / elapsed:.2f} fps end-to-end"
          + (f", fly filter {1000 * filter_seconds / len(frames):.1f} ms/frame" if fly else "")
          + f") and {still_path}", flush=True)


if __name__ == "__main__":
    main()
