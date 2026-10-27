"""Build validation-only Stadium race data directly from installed assets.

Block placement and GetSpawnLoc follow the same float32 operations as the
collision builder. Linked trigger mobils supply the exact CPlugTree AABB;
their corpus location excludes the root tree's own location. Vehicle tuning
is supplied by one captured graph per environment, never by a per-map game.
"""
from __future__ import annotations

import base64
import ctypes as ct
from pathlib import Path
import struct
import sys

from assets import BuildError, Tree
import fp
from scene import Challenge, get_mobil

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from build_validation_route import build_validation_route  # noqa: E402
from generate_route_centerline import layout_route  # noqa: E402

WAYPOINT = {'Start': 0, 'Finish': 1, 'Checkpoint': 2, 'None': 3, 'StartFinish': 4}


def waypoint_mobil(block):
    """Resolve a waypoint mobil exactly as ``Challenge.create_block_mobil``.

    A TMX block's serialized ground bit does not guarantee that its mobil is
    authored in that side of the block-info table.  The game falls back to
    the opposite ground/air table when the preferred entry is absent; route
    extraction must make the same choice or valid checkpoints become
    unusable offline.
    """
    mobil = get_mobil(block.info, block.ground, block.variant, block.sub_variant)
    if mobil is None:
        mobil = get_mobil(block.info, not block.ground, block.variant, block.sub_variant)
    return mobil


def route_waypoints(challenge: Challenge, waypoint_type: str) -> list:
    """Return authored race waypoints that create a game scene mobil.

    A few maps contain decorative/alternate ``Finish`` blocks whose variant
    is outside the block-info mobil table.  ``CreateBlockMobil`` leaves those
    blocks without a scene object, and replay ghost checkpoint/finish times
    confirm that they are not race triggers.  They must therefore not be
    emitted into the native route.  Checkpoints remain strict: omitting an
    authored checkpoint would silently change race semantics, so
    ``trigger_details`` still raises for one with no mobil.
    """
    blocks = [b for b in challenge.blocks
              if b.info['wayPointType'] == waypoint_type]
    if waypoint_type == 'Finish':
        return [b for b in blocks if waypoint_mobil(b) is not None]
    return blocks


def route_finishes(challenge: Challenge) -> list:
    """Return active Finish and StartFinish blocks in authored order."""
    return [b for b in challenge.blocks
            if b.info['wayPointType'] in ('Finish', 'StartFinish')
            and waypoint_mobil(b) is not None]


def tree_box(tree: Tree) -> tuple:
    result = tree.surface.box if tree.surface else (0., 0., 0., -1., -1., -1.)
    for child in tree.children:
        result = fp.box_union(result, tree_box(child))
    if tree.flags & 4:
        if tree.loc is None:
            raise BuildError('trigger tree has location flag without location')
        result = fp.box_set_mult(result, tree.loc)
    return result


def spawn_location(challenge: Challenge, block) -> tuple:
    encoded = block.info['spawnLocGround' if block.ground else 'spawnLocAir']
    if encoded is None:
        raise BuildError(f'{block}: missing spawn location')
    local = fp.iso_from_bytes(base64.b64decode(encoded))
    return fp.iso_set_mult(local, challenge.block_mobil_loc(block))


def initial_state(spawn: tuple, vehicle: bytes, library: Path) -> bytes:
    """CHmsDyna::SetLocation, using the native port's exact matrix operations."""
    if (len(vehicle) < 320 or vehicle[:8] != b'TMNFM6G1'
            or struct.unpack_from('<I', vehicle, 8)[0] != 4
            or struct.unpack_from('<I', vehicle, 12)[0] != len(vehicle)):
        raise BuildError('vehicle template must be a complete version-4 input graph')
    params = struct.unpack_from('<I', vehicle, 8 + (29 - 1) * 4)[0]
    if params < 320 or params + 0x5c > len(vehicle):
        raise BuildError('vehicle template dynamics parameters are out of bounds')
    body = (ct.c_float * 9).from_buffer_copy(vehicle[params + 4:params + 40])
    rotation = (ct.c_float * 9)(*spawn[:9])
    quat, inertia = (ct.c_float * 4)(), (ct.c_float * 9)()
    lib = ct.CDLL(str(library.resolve()))
    for name, count in [('GmQuat_SetFromMat3', 2), ('GmMat3_SetMult', 3),
                        ('GmMat3_MultTranspose', 2)]:
        function = getattr(lib, name)
        function.argtypes, function.restype = [ct.c_void_p] * count, None
    lib.GmQuat_SetFromMat3(quat, rotation)
    lib.GmMat3_SetMult(inertia, rotation, body)
    lib.GmMat3_MultTranspose(inertia, rotation)
    return struct.pack('<43f', *quat, *spawn, *([0.] * 15), *inertia, 0., 0., 0.)


def trigger_details(challenge: Challenge, block):
    """Return the linked waypoint trigger and its world placement.

    The route serializer and the exact-trigger collision sidecar must resolve
    the link in precisely the same way.  Keeping this small lookup in one
    place avoids the two builders drifting on mobil selection or relative
    transform order.
    """
    mobil = waypoint_mobil(block)
    if mobil is None:
        raise BuildError(f'{block}: waypoint has no mobil in ground or air block-info tables')
    # Shipped Stadium waypoints link one trigger directly from the main
    # mobil. Reject new layouts explicitly until their attachment is ported.
    expected_name = 'TriggerCheckpoint' if block.info['wayPointType'] == 'Checkpoint' else 'TriggerFinish'
    links = [link for link in mobil['mobil']['objectLinks']
             if link.get('name') and link['name'].startswith(expected_name)]
    if len(links) != 1:
        raise BuildError(f'{block}: expected one {expected_name} link, found {len(links)}')
    link = links[0]
    if link['mobil'] is None or link['mobilTreeId']:
        raise BuildError(f'{block}: unsupported trigger attachment')
    tree = challenge.assets.mobil_tree(link['mobil'], repr(block), link['mobilFile'])
    box = tree_box(tree)
    if box[3] < 0:
        raise BuildError(f'{block}: empty trigger box')
    placement = challenge.block_mobil_loc(block)
    relative = fp.iso_from_bytes(base64.b64decode(link['relativeLocation']))
    # The game copies the parent's location for an identity attachment;
    # multiplying would erase the signed zeros from quarter rotations.
    location = placement if relative == fp.IDENTITY else fp.iso_set_mult(relative, placement)
    return tree, location, spawn_location(challenge, block)


def trigger_record(challenge: Challenge, block, index: int) -> bytes:
    tree, location, spawn = trigger_details(challenge, block)
    box = tree_box(tree)
    return (struct.pack('<4I', index, block.index, WAYPOINT[block.info['wayPointType']], tree.flags)
            + fp.box_to_bytes(box) + fp.iso_to_bytes(location)
            + fp.iso_to_bytes(spawn)
            + struct.pack('<2I', bool(block.info['noRespawn']), 0))


def build(challenge: Challenge, challenge_sha256: bytes, vehicle: bytes,
          library: Path) -> bytes:
    """Return a route for recorded playback; no dense guidance is generated."""
    if challenge.doc['collection'] != 'Stadium':
        raise BuildError('offline race-data generation currently supports Stadium only')
    starts = [b for b in challenge.blocks if b.info['wayPointType'] in ('Start', 'StartFinish')]
    checkpoints = route_waypoints(challenge, 'Checkpoint')
    finishes = route_finishes(challenge)
    # One finish flag plus checkpoints must fit the collection frame's packed
    # 64 uint32 words and the race layer's pointer-free checkpoint set.
    if not starts or not finishes or len(checkpoints) >= 2048:
        raise BuildError(f'unsupported start/checkpoint/finish counts: '
                         f'starts={len(starts)}, checkpoints={len(checkpoints)}, finishes={len(finishes)}')
    lap_race = challenge.doc['isLapRace']
    laps = challenge.doc['nbLaps'] if lap_race else 1
    if not isinstance(laps, int) or laps < 1:
        raise BuildError(f'invalid lap count {laps!r}')
    if laps > 1 and len(finishes) + len(checkpoints) > 2048:
        raise BuildError('multilap route has too many independent trigger contacts')
    count = len(checkpoints)
    total = (count + 1) * laps
    # Race initialization calls CGameCtnChallenge::GetStartBlock(0)
    # (0x005AC4E0), which returns the first Start/StartFinish in active block
    # order. Extra authored starts do not change that selection.
    spawn = spawn_location(challenge, starts[0])
    start = (fp.iso_to_bytes(spawn) + initial_state(spawn, vehicle, library)
             + struct.pack('<2I', starts[0].index, WAYPOINT[starts[0].info['wayPointType']])
             + fp.iso_to_bytes(spawn))
    metadata = bytearray(struct.pack('<8I4Q', laps, count, len(finishes), count + 2,
                                    total, total, int(laps > 1), count + 2, 0, 0, 0, 0))
    records = [metadata, bytearray(start),
               bytearray().join(trigger_record(challenge, block, i) for i, block in enumerate(checkpoints)),
               bytearray().join(trigger_record(challenge, block, i) for i, block in enumerate(finishes)),
               bytearray(16 * (count + 2))]
    header = bytearray(224)
    # Keep executable identity of the source graph for provenance; all
    # per-map challenge and payload identities are recomputed here.
    header[32:64] = vehicle[256:288]
    header[64:96] = challenge_sha256
    raw = layout_route(header, records, (64, 276, 144, 144, 16), 1)
    return build_validation_route(raw)
