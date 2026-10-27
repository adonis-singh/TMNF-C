"""Build exact waypoint trigger geometry for an offline validation track.

The ordinary offline track contains the map's static collision surfaces.  A
second, deliberately small TMNFTRK1 v3 file can carry the linked waypoint
mobils as their real CPlugSurface meshes.  Keeping this data in a sidecar
means the native loader can opt into exact checkpoint/finish geometry without
changing the established static-track format.

``build`` accepts an already initialised :class:`scene.Challenge`; callers
which have an asset cache should therefore initialise the challenge once and
reuse it for both builders.
"""
from __future__ import annotations

from pathlib import Path

import fp
import bvh
import trkfile
from assets import BuildError, Tree
from build_route import WAYPOINT, route_finishes, route_waypoints, trigger_details
from build_track import COLLISION_PAIRS, MATERIAL_DATA, MESH_VTABLE
from scene import Challenge


WATER_GRID = 32
WATER_CELL = 32.0
WATER_LEVEL = 8.0
WATER_FLOOR = -992.0


def _surface_leaves(tree: Tree, parent: tuple):
    """Yield ``(surface, leaf, world_iso)`` in game tree traversal order.

    This follows ``bvh.add_static_surfaces`` exactly: a non-collidable tree
    prunes its subtree, and bit 2 means that the tree's local transform is
    composed before its parent's world transform.
    """
    if not (tree.flags & 0x80):
        return
    world = parent
    if tree.flags & 4:
        if tree.loc is None:
            raise BuildError(f"trigger tree {tree.name!r} has location flag without location")
        world = fp.iso_set_mult(tree.loc, parent)
    for child in tree.children:
        yield from _surface_leaves(child, world)
    surface = tree.surface
    if surface is not None:
        if surface.surf_type != "Mesh":
            raise BuildError(f"trigger surface {tree.name!r} is a {surface.surf_type}, not a mesh")
        if surface.box[3] < 0.0:
            return
        yield surface, tree, world


def _water() -> trkfile.Water:
    # Trigger geometry never contributes water cells, but the TMNFTRK format
    # requires a valid CHmsZone owner.  This is the same empty Stadium map
    # used by build_track.water_map before it scans map surfaces.
    return trkfile.Water(WATER_CELL, WATER_CELL, 0.0, 0.0,
                         WATER_GRID, WATER_GRID, 0, WATER_LEVEL, WATER_FLOOR,
                         bytes(WATER_GRID * WATER_GRID))


def _build_from_triggers(challenge_sha256: bytes, triggers) -> trkfile.Track:
    """Serialize ``[(tree, parent_iso, corpus_ref), ...]``.

    This small internal seam is intentionally independent of Challenge asset
    lookup, making the transform/round-trip behavior cheap to unit test.
    ``parent_iso`` is the block-plus-link placement. Input follows scene
    registration order; one-based corpus refs retain natural route order.
    """
    if not isinstance(challenge_sha256, bytes) or len(challenge_sha256) != 32:
        raise ValueError("challenge_sha256 must be exactly 32 bytes")
    track = trkfile.Track(track_sha256=challenge_sha256,
                          material_data=MATERIAL_DATA,
                          collision_pairs=list(COLLISION_PAIRS),
                          water=_water())
    mesh_index: dict[int, int] = {}
    tree_ids: dict[int, int] = {}
    cells = []
    track.corpus_isos = [None] * len(triggers)
    for tree, parent_iso, corpus_ref in triggers:
        if not 1 <= corpus_ref <= len(triggers) or track.corpus_isos[corpus_ref - 1] is not None:
            raise BuildError('trigger corpus references must be a permutation of 1..N')
        leaves = list(_surface_leaves(tree, parent_iso))
        if not leaves:
            raise BuildError(f"trigger {corpus_ref} has no collidable mesh surfaces")
        for surface, leaf, world_iso in leaves:
            cells.append(bvh.Cell(fp.box_set_mult(surface.box, world_iso),
                                  world_iso, surface, leaf, corpus_ref - 1))
        track.corpus_isos[corpus_ref - 1] = fp.iso_to_bytes(parent_iso)
    if not cells:
        raise BuildError("no collidable waypoint trigger surfaces")
    # Waypoints are static collision group 1 in the game. Its bintree can
    # reorder even coincident trigger surfaces. That order affects ties in
    # the response sort, including ordinary road/booster contacts.
    for node in bvh.build(cells):
        cell = node.cell
        if cell is None:
            track.entries.append(trkfile.Entry(node.skip, fp.box_to_bytes(node.box),
                                               bytes(48), 0, 0, 0, 0))
        else:
            surface, leaf, world_iso = cell.surface, cell.tree, cell.iso
            key = surface.mesh_key
            if key not in mesh_index:
                mesh_index[key] = len(track.meshes)
                track.meshes.append(trkfile.Mesh(
                    MESH_VTABLE, 0, 7, 0, surface.vertices,
                    surface.faces, surface.nodes))
            surface_index = len(track.surfaces)
            track.surfaces.append(trkfile.Surface(mesh_index[key], surface.materials))
            leaf_key = id(leaf)
            if leaf_key not in tree_ids:
                tree_ids[leaf_key] = len(tree_ids) + 1
            track.entries.append(trkfile.Entry(
                node.skip, fp.box_to_bytes(node.box),
                fp.iso_to_bytes(world_iso), leaf.flags, surface_index,
                tree_ids[leaf_key], cell.corpus + 1))
    return track


def build(challenge: Challenge, challenge_sha256: bytes) -> trkfile.Track:
    """Build exact trigger meshes from an initialized Stadium challenge.

    Checkpoints use corpus refs ``1..N`` and finishes use
    ``N+1..N+M``. Corpus references match ``build_route`` even when spatial
    tree construction changes their traversal order.
    """
    if challenge.doc['collection'] != 'Stadium':
        raise BuildError('exact trigger geometry currently supports Stadium only')
    checkpoints = route_waypoints(challenge, 'Checkpoint')
    finishes = route_finishes(challenge)
    if not finishes:
        raise BuildError('no active finish trigger')
    triggers = []
    for index, block in enumerate(checkpoints + finishes, start=1):
        tree, location, _spawn = trigger_details(challenge, block)
        triggers.append((block.index, tree, location, index))
    # Corpus IDs retain route order; the BVH's input is scene registration
    # order, where checkpoints and finishes are interleaved by block.
    triggers.sort(key=lambda row: row[0])
    return _build_from_triggers(challenge_sha256,
                                [(tree, location, index) for _, tree, location, index in triggers])


def sidecar_path(route: str | Path) -> Path:
    """Return the stable sidecar name associated with an offline route."""
    return Path(route).with_suffix('.tmnftriggers')


__all__ = ['build', 'sidecar_path']
