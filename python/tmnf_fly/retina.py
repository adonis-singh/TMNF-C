"""Compound-eye renderer: Gaussian acceptance per ommatidium sampled with 19
fixed rays, Embree ray casting against the track collision mesh, Lambert
shading."""

import sys
from pathlib import Path

import numpy as np
import trimesh
from embreex import rtcore_scene
from embreex.mesh_construction import TriangleMesh

from tmnf_fly import ROOT
from tmnf_fly.eye import Eye

sys.path.insert(0, str(ROOT / "tools"))
from generate_route_centerline import Triangle, parse_track  # noqa: E402

# TMNF world is y-up. Car local frame (from the lap quaternions, checked against
# the motion direction and the steer sign): x left, y up, z forward; the head
# frame (x forward, y left, z up) is built from it in ``tmnf_fly.pose``.
# Acceptance function: full width at half maximum 4.5 deg (Drosophila R1-6).
ACCEPTANCE_FWHM_DEG = 4.5
ACCEPTANCE_SIGMA_DEG = ACCEPTANCE_FWHM_DEG / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 1.911 deg
SAMPLE_RING_DEG = 0.75 * ACCEPTANCE_SIGMA_DEG  # 2-ring hex pattern reaches 1.5 sigma
AMBIENT = 0.3
DIFFUSE = 0.7
SKY_LUMINANCE = 1.0
GROUND_MISS_LUMINANCE = 0.05
# Ids from src/surface_material.h: 16 Asphalt, 0 Concrete, 4 Metal,
# 22 ResonantMetal, 2 Grass, 9 Rubber, 13 Water.
MATERIAL_ALBEDO = {16: 0.25, 0: 0.6, 4: 0.5, 22: 0.5, 2: 0.35, 9: 0.3, 13: 0.2}
DEFAULT_ALBEDO = 0.4
MATERIAL_NOT_COLLIDABLE = 28
# The snapshot contains the game's out-of-bounds enclosure: a 15 km radius
# dome of 960 concrete triangles around (512, 0, 512). It is not scenery.
WORLD_BOUNDARY_EXTENT = 5000.0


def sample_pattern() -> tuple[np.ndarray, np.ndarray]:
    """19 (angular_radius_rad, azimuth_rad) offsets and Gaussian weights:
    centre, 6 at r, 6 at sqrt(3) r, 6 at 2 r (hex lattice, 2 rings)."""
    r = np.radians(SAMPLE_RING_DEG)
    radii = [0.0] + [r] * 6 + [np.sqrt(3) * r] * 6 + [2 * r] * 6
    phis = [0.0] + list(np.arange(6) * np.pi / 3) + list(np.arange(6) * np.pi / 3 + np.pi / 6) + list(np.arange(6) * np.pi / 3)
    radii = np.array(radii)
    phis = np.array(phis)
    sigma = np.radians(ACCEPTANCE_SIGMA_DEG)
    weights = np.exp(-0.5 * (radii / sigma) ** 2)
    return np.stack([radii, phis], axis=1), weights / weights.sum()


def orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Branchless tangent basis (Duff et al. 2017) for unit vectors n (..., 3)."""
    s = np.copysign(1.0, n[..., 2])
    a = -1.0 / (s + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    u = np.stack([1.0 + s * n[..., 0] ** 2 * a, s * b, -s * n[..., 0]], axis=-1)
    v = np.stack([b, s + n[..., 1] ** 2 * a, -n[..., 1]], axis=-1)
    return u, v


def sample_directions(dirs: np.ndarray) -> np.ndarray:
    """(N, 3) unit centre directions -> (N, 19, 3) sample directions."""
    offsets, _ = sample_pattern()
    rho = offsets[:, 0][None, :, None]
    phi = offsets[:, 1][None, :, None]
    u, v = orthonormal_basis(dirs)
    tangent = u[:, None] * np.cos(phi) + v[:, None] * np.sin(phi)
    return dirs[:, None] * np.cos(rho) + tangent * np.sin(rho)


class Retina:
    def __init__(self, triangles: list[Triangle], eye: Eye, sun_direction: np.ndarray):
        tri = np.array([[t.a, t.b, t.c] for t in triangles], dtype=np.float64)
        material = np.array([t.material for t in triangles], dtype=np.int32)
        keep = (material != MATERIAL_NOT_COLLIDABLE) & (np.abs(tri).max(axis=(1, 2)) < WORLD_BOUNDARY_EXTENT)
        tri, material = tri[keep], material[keep]
        self.vertices = tri.reshape(-1, 3)
        self.faces = np.arange(len(self.vertices), dtype=np.int32).reshape(-1, 3)
        self.mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
        self.face_normals = np.asarray(self.mesh.face_normals, dtype=np.float64)
        self.face_material = material
        self.face_albedo = np.array([MATERIAL_ALBEDO.get(int(m), DEFAULT_ALBEDO) for m in material])
        self.scene = rtcore_scene.EmbreeScene()
        TriangleMesh(self.scene, self.vertices.astype(np.float32), self.faces)
        self.sun = sun_direction / np.linalg.norm(sun_direction)
        # (2, 852, 19, 3) head-frame sample directions, left eye first.
        self.samples_head = np.stack([sample_directions(eye.dirs_left), sample_directions(eye.dirs_right)])
        self.weights = sample_pattern()[1]
        self.n_rays = self.samples_head.size // 3

    @classmethod
    def from_track(cls, track: Path, eye: Eye, sun_direction: np.ndarray) -> "Retina":
        return cls(parse_track(track), eye, sun_direction)

    def render(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """position (3,) eye position in world; rotation (3, 3) head-to-world.
        Returns (2, 852) luminance, index 0 = left eye, 1 = right eye."""
        dirs = (self.samples_head.reshape(-1, 3) @ rotation.T).astype(np.float32)
        origins = np.broadcast_to(position.astype(np.float32), dirs.shape)
        face = np.asarray(self.scene.run(np.ascontiguousarray(origins), dirs))
        hit = face >= 0
        lum = np.where(dirs[:, 1] > 0.0, SKY_LUMINANCE, GROUND_MISS_LUMINANCE)
        n = self.face_normals[face[hit]]
        d = dirs[hit].astype(np.float64)
        # Track winding is not consistent; make the normal face the viewer.
        n = n * -np.sign((n * d).sum(1, keepdims=True))
        lum[hit] = self.face_albedo[face[hit]] * (AMBIENT + DIFFUSE * np.maximum(0.0, n @ self.sun))
        return (lum.reshape(2, -1, len(self.weights)) * self.weights).sum(2)
