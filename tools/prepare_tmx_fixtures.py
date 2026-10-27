#!/usr/bin/env python3
"""Capture TMNF-C fixtures for embedded arbitrary-TMX replay maps.

Each fixture is stored under ``local/collection-fixtures/<UID>/<geometry>``
and appended to a geometry-aware catalogue.  The work list is deterministic,
idempotent, and can be split across isolated Wine lanes with ``--workers``.
The default manifest is the small sampled validation manifest; this tool does
not discover or process the full replay corpus unless replay paths are passed
explicitly.  ``--reference-mode assets`` captures only the track, route, and
vehicle fixtures needed by native validation; ``full`` additionally captures
the long mixed and wall-contact reference schedules.  ``--route-mode dense``
keeps the full centerline reference, while ``validation`` upgrades the route
to the sparse start/checkpoint/finish reference used by validation.  The
optional ``--track-backend offline`` builds collision snapshots from the local
game packs and is currently limited to assets-only captures; route and vehicle
capture still use the game.  The experimental ``--fixture-backend offline`` path
builds the Stadium track, graph vehicle, and validation route without Wine.
``--exact-triggers`` additionally emits linked waypoint meshes as a
``.tmnftriggers`` sidecar for that fully offline mode.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time

from pygbx import Gbx, GbxType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = ROOT / "local/collection-fixtures"
DEFAULT_REGISTRY = ROOT / "local/collection_track_variants.json"
DEFAULT_MANIFEST = ROOT / "local/fixture-manifest.json"
# IDE-friendly controls.  CLI flags override these values for batch jobs.
SETTINGS = {
    "manifest": DEFAULT_MANIFEST,
    "limit": None,
    "seed": 42,
    "workers": 1,
    "plan_workers": None,
    "reuse_captures": False,
    "socket_timeout": 180.0,
    "reference_mode": "full",
    "route_mode": "dense",
    "track_backend": "game",
    "packs": None,  # Use build_track's installed-game default, or --packs.
    "asset_cache": None,  # Select the cache from the installed package profile.
    "fixture_backend": "game",
    "vehicle_template": ROOT / "oracle/vehicles/A01-Stadium.tmnfvehicle",
    "physics_library": ROOT / "build/local/libtmnf_physics.so",
    "exact_triggers": False,
    "lane_root": None,
    "template_prefix": None,
}

RUN_RECORD_SIZE = 1668
FIXED_DECORATION_MARKER = b'TMNF-FIXED-DECO\0'


def _has_fixed_decoration_collisions(path: Path) -> bool:
    with path.open('rb') as stream:
        stream.seek(0x140)
        return stream.read(16) == FIXED_DECORATION_MARKER
RUN_POSITION_OFFSET = 56
FULL_GAME_STARTS = 6
ASSETS_GAME_STARTS = 4
FULL_REFERENCE_TICKS = 1200 + 1400


def _capture_estimate(reference_mode: str, track_backend: str = 'game',
                      fixture_backend: str = 'game') -> dict[str, int | str]:
    """Return nominal per-map capture work, excluding retries.

    ``capture_all`` has four short asset launches (track, route, vehicle
    probe, vehicle dump) and two long reference launches.  Keeping this
    estimate beside the mode switch makes the savings visible in reports;
    retries can only increase the observed launch count.
    """
    assets = reference_mode == 'assets'
    skipped_track_starts = 1 if track_backend == 'offline' else 0
    if fixture_backend == 'offline':
        game_starts = 0
    else:
        game_starts = (ASSETS_GAME_STARTS if assets else FULL_GAME_STARTS) - skipped_track_starts
    return {
        'nominal_game_starts_per_map': game_starts,
        'nominal_track_backend': track_backend,
        'nominal_fixture_backend': fixture_backend,
        'saved_game_starts_vs_game_backend': ((ASSETS_GAME_STARTS if assets else FULL_GAME_STARTS)
                                              if fixture_backend == 'offline'
                                              else skipped_track_starts),
        'reference_schedule_ticks': 0 if assets else FULL_REFERENCE_TICKS,
        'saved_game_starts_vs_full': FULL_GAME_STARTS - game_starts,
        'saved_reference_ticks_vs_full': FULL_REFERENCE_TICKS if assets else 0,
    }


@dataclass(frozen=True)
class EmbeddedTrack:
    name: str
    environment_name: str
    output_root: Path

    @property
    def environment(self) -> str:
        return self.environment_name.lower()

    @property
    def campaign(self) -> str:
        return "nations"

    @property
    def code(self) -> str:
        return self.name

    @property
    def id(self) -> str:
        return self.name.lower()

    @property
    def snapshot(self) -> Path:
        return self.output_root / f"{self.name}.tmnftrack"

    @property
    def route(self) -> Path:
        return self.output_root / f"{self.name}.tmnfroute"

    @property
    def vehicle(self) -> Path:
        return self.output_root / f"{self.name}.tmnfvehicle"

    @property
    def trigger_geometry(self) -> Path:
        return self.output_root / f"{self.name}.tmnftriggers"

    def result(self, name: str, suffix: str = "") -> Path:
        return self.output_root / f"{self.name}_{name}{suffix}.bin"


def geometry_sha256(challenge) -> str:
    rows = [(block.name, block.position.x, block.position.y, block.position.z,
             block.rotation, block.flags) for block in challenge.blocks]
    payload = json.dumps([str(challenge.environment), rows], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _route_mode(path: Path) -> str | None:
    """Classify a v3 route by its reference section, if it is loadable."""
    try:
        data = path.read_bytes()
        if data[:8] != b'TMNFROU1' or struct.unpack_from('<I', data, 8)[0] != 3:
            return None
        metadata_offset, _, _ = struct.unpack_from('<QII', data, 96)
        checkpoint_count = struct.unpack_from('<I', data, metadata_offset + 4)[0]
        _, reference_count, _ = struct.unpack_from('<QII', data, 96 + 4 * 16)
        return 'validation' if reference_count <= checkpoint_count + 2 else 'dense'
    except (OSError, struct.error):
        return None


_OFFLINE_BUILDER_IDENTITY = None


def _offline_builder_fingerprint() -> str:
    """Invalidate reusable fixtures when their builder or asset profile changes."""
    global _OFFLINE_BUILDER_IDENTITY
    sources = [Path(__file__), ROOT / 'tools/build_validation_route.py']
    sources.extend(ROOT / 'tools/build_track' / name for name in (
        'assets.py', 'bvh.py', 'fp.py', 'trkfile.py', 'build_track.py',
        'scene.py', 'build_route.py', 'build_trigger_geometry.py', 'GbxDump/Program.cs',
        'GbxDump/GbxDump.csproj', 'build_gbx_compat.py', 'gbx_legacy_clip.patch'))
    context_key = _OFFLINE_ASSET_CONTEXT[0] if _OFFLINE_ASSET_CONTEXT else None
    if context_key is not None:
        for name in ('asset-pack-profile.json', 'capture-game-assets.json'):
            manifest = Path(context_key[1]) / name
            if manifest.exists():
                sources.append(manifest)
                break
    signature = (tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                       for path in sources), context_key)
    if _OFFLINE_BUILDER_IDENTITY is None or _OFFLINE_BUILDER_IDENTITY[0] != signature:
        hashes = [(path.name, _sha256(path)) for path in sources]
        digest = hashlib.sha256(json.dumps((hashes, context_key), separators=(',', ':')).encode()).hexdigest()
        _OFFLINE_BUILDER_IDENTITY = (signature, digest)
    return _OFFLINE_BUILDER_IDENTITY[1]


def _variant_row(challenge_path: Path, current: EmbeddedTrack,
                 source_replay: Path | None = None,
                 reference_mode: str = 'full', route_mode: str = 'dense',
                 fixture_backend: str = 'game', track_backend: str = 'game',
                 vehicle_mode: str = 'snapshot', vehicle_template: Path | None = None,
                 exact_triggers: bool = False) -> dict:
    """Validate a fixture and return its registry row without publishing it."""
    maps = Gbx(challenge_path.read_bytes()).get_classes_by_ids(
        [GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    if len(maps) != 1:
        raise ValueError('captured challenge must contain exactly one map')
    challenge_sha256 = _sha256(challenge_path)
    paths = {'track': current.snapshot, 'vehicle': current.vehicle, 'route': current.route}
    if exact_triggers:
        paths['triggers'] = current.trigger_geometry
    magic_offsets = {'track': (b'TMNFTRK1', 64), 'vehicle': (b'TMNFM6G1', 288),
                     'route': (b'TMNFROU1', 64)}
    magic_offsets['triggers'] = (b'TMNFTRK1', 64)
    hashes = {}
    for key, path in paths.items():
        data = path.read_bytes()
        magic, offset = magic_offsets[key]
        if data[:8] != magic or data[offset:offset + 32].hex() != challenge_sha256:
            raise ValueError(f'{key} fixture is not bound to the captured challenge')
        hashes[key] = _sha256(path)
    checkpoint_order = None
    capture_log = Path(str(paths['route']) + '.log')
    if capture_log.exists():
        original = {int(block): int(index) for index, block in re.findall(
            r'^checkpoint index=(\d+) block=(\d+)', capture_log.read_text(), re.MULTILINE)}
        route = paths['route'].read_bytes()
        offset, count, stride = struct.unpack_from('<QII', route, 96 + 2 * 16)
        blocks = [struct.unpack_from('<I', route, offset + i * stride + 4)[0]
                  for i in range(count)]
        if len(original) != count or any(block not in original for block in blocks):
            raise ValueError('route checkpoint blocks do not match capture log')
        checkpoint_order = [original[block] for block in blocks] + [count]
    def relative(path):
        path = path.resolve()
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    row = {'map_id': maps[0].map_uid, 'challenge_sha256': challenge_sha256,
           **{key: relative(path) for key, path in paths.items()},
           'file_hashes': hashes, 'geometry_sha256': geometry_sha256(maps[0]),
           'checkpoint_order': checkpoint_order,
           'reference_mode': reference_mode,
           'route_mode': _route_mode(paths['route']) or route_mode,
           'fixture_backend': fixture_backend,
           'track_backend': track_backend,
           'vehicle_mode': vehicle_mode,
           'trigger_geometry': bool(exact_triggers),
           'fixed_decoration_collisions': _has_fixed_decoration_collisions(current.snapshot),
           'vehicle_environment': current.environment_name}
    if fixture_backend == 'offline':
        row['builder_fingerprint'] = _offline_builder_fingerprint()
    if vehicle_template is not None:
        row.update({'vehicle_template': str(Path(vehicle_template).resolve()),
                    'vehicle_template_sha256': _sha256(Path(vehicle_template))})
    if source_replay is not None:
        row.update({'source_replay': str(source_replay.resolve()),
                    'source_replay_sha256': _sha256(source_replay)})
    return row


def _merge_variant_row(data: dict, row: dict) -> None:
    """Merge one validated row into an in-memory registry document."""
    if data.get('schema') != 'tmnf-collection-track-variants-v1':
        raise ValueError('unsupported fixture variant registry schema')
    rows = data['tracks'].setdefault(row['map_id'], [])
    identity = (row['challenge_sha256'], row['geometry_sha256'],
                tuple(row[key] for key in ['track', 'vehicle', 'route']),
                row.get('triggers'))
    matching = next((index for index, existing in enumerate(rows)
                     if (existing.get('challenge_sha256'), existing.get('geometry_sha256'),
                         tuple(existing.get(key) for key in ['track', 'vehicle', 'route']),
                         existing.get('triggers')) == identity), None)
    if matching is None:
        rows.append(row)
        rows.sort(key=lambda value: (value.get('geometry_sha256') or '', value['challenge_sha256']))
    elif rows[matching] != row:
        # A resumed run may have upgraded a v1 route to its dense v3 form;
        # refresh hashes for the same challenge/geometry identity.
        rows[matching] = row


def _publish_variant_row(row: dict, registry: Path) -> None:
    """Publish one row under the legacy locked JSON registry semantics."""
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.with_suffix(registry.suffix + '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(registry.read_text()) if registry.exists() else {
            'schema': 'tmnf-collection-track-variants-v1', 'tracks': {}}
        _merge_variant_row(data, row)
        temporary = registry.with_name(f'.{registry.name}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(registry)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _write_variant_row_provenance(row: dict, path: Path) -> None:
    """Persist one worker row without a shared lock.

    A batched run assigns a unique path to each UID/geometry, so a lock file
    would only add filesystem churn.  The temporary replace makes the row
    self-contained and readable after a worker or parent crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'schema': 'tmnf-collection-track-variants-v1',
            'tracks': {row['map_id']: [row]}}
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def _register_variant(challenge_path: Path, current: EmbeddedTrack, registry: Path,
                      source_replay: Path | None = None,
                      reference_mode: str = 'full', route_mode: str = 'dense',
                      fixture_backend: str = 'game', track_backend: str = 'game',
                      vehicle_mode: str = 'snapshot', vehicle_template: Path | None = None,
                      exact_triggers: bool = False) -> dict:
    """Publish a fixture with stdlib+pygbx only (capture Python has no torch)."""
    row = _variant_row(challenge_path, current, source_replay, reference_mode,
                       route_mode, fixture_backend, track_backend, vehicle_mode,
                       vehicle_template, exact_triggers)
    _publish_variant_row(row, registry)
    return row


def _merge_registry_rows(registry: Path, rows_dir: Path) -> int:
    """Merge crash-safe per-map rows into the shared registry.

    Offline workers write each row to its own small registry document before
    returning success.  This keeps the expensive shared JSON load/dump out of
    worker contention while retaining a durable artifact if the parent dies.
    Rows are removed only after the shared registry has been atomically
    replaced; an interrupted merge therefore safely replays them.
    """
    if not rows_dir.is_dir():
        return 0
    paths = sorted(rows_dir.glob('*.json'))
    if not paths:
        return 0
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.with_suffix(registry.suffix + '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(registry.read_text()) if registry.exists() else {
            'schema': 'tmnf-collection-track-variants-v1', 'tracks': {}}
        for path in paths:
            shard = json.loads(path.read_text())
            if shard.get('schema') != 'tmnf-collection-track-variants-v1':
                raise ValueError(f'unsupported fixture row registry: {path}')
            for rows in shard.get('tracks', {}).values():
                for row in rows:
                    _merge_variant_row(data, row)
        temporary = registry.with_name(f'.{registry.name}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(registry)
        for path in paths:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + '.lock').unlink(missing_ok=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return len(paths)


def _registered_complete(challenge_path: Path, current: EmbeddedTrack,
                         uid: str, geometry: str, registry: Path,
                         reference_mode: str = 'full', route_mode: str = 'dense',
                         fixture_backend: str = 'game', track_backend: str = 'game',
                         vehicle_mode: str = 'snapshot', exact_triggers: bool = False) -> bool:
    """Check a published record and every file hash before skipping capture."""
    if not registry.exists():
        return False
    try:
        data = json.loads(registry.read_text())
        rows = data.get('tracks', {}).get(uid, [])
        challenge_hash = _sha256(challenge_path)
        for row in rows:
            if row.get('geometry_sha256') != geometry or row.get('challenge_sha256') != challenge_hash:
                continue
            if (current.environment_name == 'Stadium' and
                    (not row.get('fixed_decoration_collisions') or
                     not _has_fixed_decoration_collisions(current.snapshot))):
                continue
            row_mode = row.get('reference_mode', 'full')
            if row.get('fixture_backend', 'game') != fixture_backend:
                continue
            if fixture_backend == 'offline' and row.get('builder_fingerprint') != _offline_builder_fingerprint():
                continue
            if row.get('track_backend', 'game') != track_backend:
                continue
            if row.get('vehicle_mode', 'snapshot') != vehicle_mode:
                continue
            if exact_triggers and not row.get('trigger_geometry', bool(row.get('triggers'))):
                continue
            # Full fixtures contain the complete static asset set and the
            # optional reference schedules, so they are a valid superset for
            # an assets-only request.  The reverse is intentionally false:
            # assets-only output must never be presented as full validation.
            if reference_mode == 'full' and row_mode != 'full':
                continue
            if reference_mode == 'assets' and row_mode not in ('assets', 'full'):
                continue
            row_route_mode = row.get('route_mode', 'dense')
            if route_mode == 'dense' and row_route_mode != 'dense':
                continue
            if route_mode == 'validation' and row_route_mode not in ('validation', 'dense'):
                continue
            actual_route_mode = _route_mode(current.route)
            if actual_route_mode is not None:
                if route_mode == 'dense' and actual_route_mode != 'dense':
                    continue
                if route_mode == 'validation' and actual_route_mode not in ('validation', 'dense'):
                    continue
            files = [('track', current.snapshot), ('vehicle', current.vehicle),
                     ('route', current.route)]
            if exact_triggers:
                files.append(('triggers', current.trigger_geometry))
            for key, path in files:
                if not path.is_file() or _sha256(path) != row.get('file_hashes', {}).get(key):
                    return False
            return struct.unpack_from('<I', current.route.read_bytes(), 8)[0] == 3
    except (OSError, ValueError, KeyError, struct.error):
        return False
    return False


_OFFLINE_ASSET_CONTEXT = None


def _offline_cache_context(packs: Path | None, asset_cache: Path | None):
    build_root = ROOT / 'tools' / 'build_track'
    if str(build_root) not in sys.path:
        sys.path.insert(0, str(build_root))
    import assets as build_assets
    packs = Path(packs).expanduser().resolve() if packs else Path(build_assets.DEFAULT_PACKS)
    cache = (Path(asset_cache).expanduser().resolve() if asset_cache
             else Path(build_assets.cache_dir_for(str(packs))))
    key = (str(packs), str(cache))
    # A worker keeps one Assets memoizer for its whole lane.  The old code
    # called ensure_cache (and took its interprocess lock) for every map,
    # even though the cache marker was already complete.
    if (_OFFLINE_ASSET_CONTEXT is not None
            and _OFFLINE_ASSET_CONTEXT[0] == key):
        return build_assets, packs, cache
    if not build_assets.cache_complete(str(cache)):
        cache.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache.with_name(cache.name + '.prepare.lock')
        with lock_path.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            build_assets.ensure_cache(str(cache), str(packs))
            fcntl.flock(lock, fcntl.LOCK_UN)
    build_assets.verify_cache_packages(str(cache), str(packs))
    return build_assets, packs, cache


def _offline_track_assets(packs: Path | None, asset_cache: Path | None):
    """Return a process-local build_track Assets object after locked setup."""
    global _OFFLINE_ASSET_CONTEXT
    build_assets, packs, cache = _offline_cache_context(packs, asset_cache)
    key = (str(packs), str(cache))
    if _OFFLINE_ASSET_CONTEXT is None or _OFFLINE_ASSET_CONTEXT[0] != key:
        _OFFLINE_ASSET_CONTEXT = (key, build_assets.Assets(str(packs), str(cache)))
    return _OFFLINE_ASSET_CONTEXT[1]


def _build_offline_track(challenge_path: Path, output: Path,
                         environment: str, packs: Path | None,
                         asset_cache: Path | None) -> None:
    """Build the collision snapshot without launching the game.

    ``allow_ground_mismatch`` is intentional: the game selects TMX unit/mobil
    terrain from runtime flags, while the serialized challenge can carry a
    distinct unit-ground value.  The builder preserves the game-compatible
    track header and rejects all other build errors.
    """
    build_root = ROOT / 'tools' / 'build_track'
    if str(build_root) not in sys.path:
        sys.path.insert(0, str(build_root))
    import build_track
    import trkfile
    assets = _offline_track_assets(packs, asset_cache)
    track = build_track.build(assets, str(challenge_path), env=environment,
                              quality='low', allow_ground_mismatch=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    trkfile.save(track, str(output))


def _build_offline_fixture(challenge_path: Path, current: EmbeddedTrack,
                           challenge_hash: bytes, packs: Path | None,
                           asset_cache: Path | None, vehicle_template: Path,
                           physics_library: Path, exact_triggers: bool = False) -> dict:
    """Build Stadium track, graph vehicle, and validation route without Wine."""
    build_root = ROOT / 'tools' / 'build_track'
    if str(build_root) not in sys.path:
        sys.path.insert(0, str(build_root))
    import build_route
    import build_track
    import trkfile
    from scene import Challenge

    if current.environment_name != 'Stadium':
        raise ValueError('fully offline fixture backend currently supports Stadium only')
    template = Path(vehicle_template).expanduser().resolve()
    library = Path(physics_library).expanduser().resolve()
    if not template.is_file():
        raise FileNotFoundError(f'vehicle graph template not found: {template}')
    if not library.is_file():
        raise FileNotFoundError(f'physics library not found: {library}')
    template_bytes = template.read_bytes()
    if (template_bytes[:8] != b'TMNFM6G1' or len(template_bytes) < 320
            or struct.unpack_from('<I', template_bytes, 8)[0] != 4
            or struct.unpack_from('<I', template_bytes, 12)[0] != len(template_bytes)):
        raise ValueError(f'invalid vehicle graph template: {template}')
    vehicle_bytes = bytearray(template_bytes)
    vehicle_bytes[288:320] = challenge_hash

    assets = _offline_track_assets(packs, asset_cache)
    document = build_track.challenge_json(assets, str(challenge_path))
    challenge = Challenge(assets, document, allow_ground_mismatch=True,
                          allow_missing_blocks=True)
    challenge.init()
    track = build_track.build_from_challenge(
        assets, challenge, str(challenge_path), env='Stadium')
    build_track.append_fixed_decoration_collisions(track, assets, challenge.decoration)
    trkfile.save(track, str(current.snapshot))
    route_bytes = build_route.build(challenge, challenge_hash, bytes(vehicle_bytes), library)
    current.vehicle.write_bytes(vehicle_bytes)
    current.route.write_bytes(route_bytes)
    if exact_triggers:
        import build_trigger_geometry
        trigger_track = build_trigger_geometry.build(challenge, challenge_hash)
        trkfile.save(trigger_track, str(current.trigger_geometry))
    checkpoints = [block for block in challenge.blocks
                   if block.info['wayPointType'] == 'Checkpoint']
    Path(str(current.route) + '.log').write_text(
        ''.join(f'checkpoint index={index} block={block.index}\n'
                for index, block in enumerate(checkpoints)))
    return {'vehicle_mode': 'graph', 'vehicle_template': template,
            'track_backend': 'offline', 'trigger_geometry': bool(exact_triggers),
            'ignored_missing_blocks': challenge.missing_blocks}


def _complete_fixture_collision_scene(challenge_path: Path, current: EmbeddedTrack,
                                      packs: Path | None, asset_cache: Path | None) -> None:
    """Supplement game-captured static geometry with fixed Stadium flag meshes."""
    if current.environment_name != 'Stadium' or _has_fixed_decoration_collisions(current.snapshot):
        return
    build_root = ROOT / 'tools/build_track'
    if str(build_root) not in sys.path:
        sys.path.insert(0, str(build_root))
    import build_track
    import trkfile
    if packs is None:
        import onboard_track
        packs = onboard_track.layout().game_dir / 'Packs'
    assets = _offline_track_assets(packs, asset_cache)
    document = build_track.challenge_json(assets, str(challenge_path))
    decoration = assets.decoration(document['collection'], document['decoration']['id'])
    track = trkfile.load(str(current.snapshot))
    # Retain the original static snapshot for direct game/exporter comparison.
    shutil.copy2(current.snapshot, current.snapshot.with_suffix('.static.tmnftrack'))
    build_track.append_fixed_decoration_collisions(track, assets, decoration)
    temporary = current.snapshot.with_name(current.snapshot.name + '.scene.tmp')
    trkfile.save(track, str(temporary))
    os.replace(temporary, current.snapshot)


def _capture_track(onboard_track, current: EmbeddedTrack, challenge_path: Path,
                   challenge_hash: bytes, staging: Path, track_backend: str,
                   packs: Path | None, asset_cache: Path | None) -> None:
    if track_backend == 'offline':
        _build_offline_track(challenge_path, current.snapshot, current.environment_name,
                             packs, asset_cache)
        onboard_track.verify_bound_snapshot(current.snapshot, b'TMNFTRK1', challenge_hash, 64)
        return
    if track_backend != 'game':
        raise ValueError(f'unsupported track backend: {track_backend}')
    track_name = current.name
    onboard_track.install_tracer('track')
    onboard_track.run_game(track_name, [
        onboard_track.PYTHON, onboard_track.ROOT / 'oracle/determinism_test.py',
        '--port', str(onboard_track.PORT), '--map', track_name, '--ticks', '10',
        '--output-dir', staging / 'track-probe'], staging / 'track.log')
    onboard_track.verify_bound_snapshot(current.snapshot, b'TMNFTRK1', challenge_hash, 64)


def _capture_assets(onboard_track, current: EmbeddedTrack, challenge_path: Path,
                    challenge_hash: bytes, staging: Path,
                    track_backend: str = 'game', packs: Path | None = None,
                    asset_cache: Path | None = None) -> None:
    """Capture track, route and vehicle without publishing replay references."""
    official_maps = onboard_track.layout().official_maps
    official_maps.mkdir(parents=True, exist_ok=True)
    onboard_track.shutil.copyfile(challenge_path, official_maps / challenge_path.name)
    _capture_track(onboard_track, current, challenge_path, challenge_hash, staging,
                   track_backend, packs, asset_cache)
    track_name = current.name
    onboard_track.install_tracer('route')
    onboard_track.run_game(track_name, [
        onboard_track.PYTHON, onboard_track.ROOT / 'oracle/determinism_test.py',
        '--port', str(onboard_track.PORT), '--map', track_name, '--ticks', '10',
        '--output-dir', staging / 'route-probe'], staging / 'route.log')
    onboard_track.verify_bound_snapshot(current.route, b'TMNFROU1', challenge_hash, 64)
    traces = staging / 'vehicle-traces'
    traces.mkdir()
    onboard_track.install_tracer('trace')
    onboard_track.run_game(track_name, [
        onboard_track.PYTHON, onboard_track.ROOT / 'oracle/determinism_test.py',
        '--port', str(onboard_track.PORT), '--map', track_name, '--ticks', '10',
        '--output-dir', staging / 'vehicle-probe'], staging / 'vehicle-phase.log', {
            'TMNF_TRACE_HOST_DIR': str(traces),
            'TMNF_TRACE_TARGETS': ','.join(f'0x{address:08X}'
                                           for address in onboard_track.COMPUTE_FORCES_VAS),
        })
    model_traces = [
        path for path in sorted(traces.glob('*_CSceneVehicleCar_ComputeForcesModel?.bin'))
        if struct.unpack_from('<I', path.read_bytes(), 12)[0] != 0
    ]
    if len(model_traces) != 1:
        raise RuntimeError(f'expected one recorded ComputeForces model trace, found {len(model_traces)}')
    phase = staging / f'{current.vehicle.stem}-phase.tmnfvehicle'
    onboard_track.run([onboard_track.PYTHON, onboard_track.ROOT /
                       'oracle/tracer/extract_model6_vehicle.py', model_traces[0], phase])
    onboard_track.disable_tracer()
    arguments = [onboard_track.PYTHON, onboard_track.ROOT / 'oracle/dump_vehicle_snapshot.py',
                 '--port', str(onboard_track.PORT), '--map', track_name,
                 '--phase-input', phase, '--output', current.vehicle,
                 '--report', staging / 'vehicle.md']
    base = onboard_track.environment_base_vehicle(current)
    if base is not None:
        arguments.extend(('--input', base))
    onboard_track.run_game(track_name, arguments, staging / 'vehicle.log')
    onboard_track.verify_bound_snapshot(current.vehicle, b'TMNFM6G1', challenge_hash, 288)


def _capture_outputs(reference_mode: str, current: EmbeddedTrack,
                     exact_triggers: bool = False) -> list[Path]:
    """Return the files produced by a game-side capture.

    These paths are deliberately kept separate from the final fixture paths:
    Wine and the tracer use fixed-size Windows path buffers, so a perfectly
    valid Linux fixture destination can still be too long for the game.
    """
    paths = [current.snapshot, current.route, current.vehicle]
    if reference_mode == 'full':
        paths.extend([current.result('mixed', '_inputs'), current.result('mixed'),
                      current.result('wall_contact', '_inputs'),
                      current.result('wall_contact')])
    if exact_triggers:
        paths.append(current.trigger_geometry)
    return paths


def _publish_capture_outputs(capture_current: EmbeddedTrack, current: EmbeddedTrack,
                             reference_mode: str, capture_staging: Path,
                             diagnostics: Path) -> None:
    """Atomically publish short-workspace outputs and retain capture logs."""
    outputs = _capture_outputs(reference_mode, capture_current)
    destinations = _capture_outputs(reference_mode, current)
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'capture did not produce expected outputs: ' + ', '.join(map(str, missing)))
    for source, destination in zip(outputs, destinations):
        temporary = destination.with_name(
            f'.{destination.name}.{os.getpid()}.tmp')
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    # The tracer emits the checkpoint mapping beside TMNF_ROUTE_PATH.  The
    # client stdout log has the same basename under ``capture_staging`` but
    # only contains determinism-test output, so do not publish that file as
    # the route mapping.  Keep the sidecar beside the published route because
    # registration reads it there, while preserving both logs in diagnostics.
    route_log = Path(str(capture_current.route) + '.log')
    if route_log.is_file() and 'checkpoint index=' in route_log.read_text(errors='replace'):
        temporary = Path(str(current.route) + f'.{os.getpid()}.log.tmp')
        shutil.copyfile(route_log, temporary)
        os.replace(temporary, Path(str(current.route) + '.log'))
    if capture_staging.exists():
        diagnostics.mkdir(parents=True, exist_ok=True)
        shutil.copytree(capture_staging, diagnostics, dirs_exist_ok=True)
    if route_log.is_file():
        shutil.copyfile(route_log, diagnostics / 'captured-route.log')


def _capture_with_short_workspace(onboard_track, current: EmbeddedTrack,
                                  challenge_path: Path, challenge_hash: bytes,
                                  reference_mode: str, final_staging: Path,
                                  track_backend: str = 'game', packs: Path | None = None,
                                  asset_cache: Path | None = None) -> None:
    """Run game capture under /tmp and publish into the requested fixture root.

    ``TMNF_TRACK_DIR``, ``TMNF_ROUTE_DIR``, tracer output, and all game-side
    command output paths point inside this workspace.  Keeping it in /tmp is
    intentional: Windows ``MAX_PATH`` limits apply to the path string sent
    to the game even when the eventual Linux fixture path is otherwise valid.
    """
    with tempfile.TemporaryDirectory(prefix='tmnf-fixture-', dir='/tmp') as folder:
        workspace = Path(folder)
        capture_staging = workspace / 's'
        capture_staging.mkdir()
        capture_current = EmbeddedTrack(current.name, current.environment_name, workspace)
        old_environment = {
            key: os.environ.get(key)
            for key in ('TMNF_TRACK_DIR', 'TMNF_ROUTE_DIR', 'TMNF_TRACE_HOST_DIR')
        }
        os.environ.update({
            'TMNF_TRACK_DIR': str(workspace),
            'TMNF_ROUTE_DIR': str(workspace),
            'TMNF_TRACE_HOST_DIR': str(workspace / 'traces'),
        })
        try:
            if reference_mode == 'full':
                onboard_track.capture_all(
                    capture_current, challenge_path, challenge_hash, capture_staging,
                    capture_current.snapshot, capture_current.route, capture_current.vehicle,
                    capture_current.result('mixed', '_inputs'), capture_current.result('mixed'),
                    capture_current.result('wall_contact', '_inputs'),
                    capture_current.result('wall_contact'))
            else:
                _capture_assets(onboard_track, capture_current, challenge_path,
                                challenge_hash, capture_staging, track_backend,
                                packs, asset_cache)
        finally:
            # Preserve logs even when the launcher fails before producing any
            # snapshot.  The temporary workspace itself is disposable.
            if capture_staging.exists():
                diagnostics = final_staging / 'capture'
                diagnostics.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(capture_staging, diagnostics, dirs_exist_ok=True)
                route_log = Path(str(capture_current.route) + '.log')
                if route_log.is_file():
                    shutil.copyfile(route_log, diagnostics / 'captured-route.log')
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        _publish_capture_outputs(capture_current, current, reference_mode,
                                 capture_staging, final_staging / 'capture')


def _write_ghost_position_drive(replay: Path, output: Path, track_path: Path) -> None:
    """Write a temporary position-only drive for dense-route generation."""
    if str(ROOT / 'tools') not in sys.path:
        sys.path.insert(0, str(ROOT / 'tools'))
    import generate_route_centerline
    ghost = generate_route_centerline.read_ghost(replay, optional=False)
    if ghost is None or len(ghost.positions) < 2:
        raise ValueError(f'{replay}: no usable ghost positions for assets-only route generation')
    triangles = generate_route_centerline.parse_track(track_path)
    materials = generate_route_centerline.drive_ray_materials(triangles, ghost.positions)
    if not materials:
        raise ValueError(f'{replay}: ghost does not contact a track surface')
    material = min(materials)
    data = bytearray(RUN_RECORD_SIZE * len(ghost.positions))
    for index, point in enumerate(ghost.positions):
        base = index * RUN_RECORD_SIZE
        struct.pack_into('<3f', data, base + RUN_POSITION_OFFSET,
                         float(point.x), float(point.y), float(point.z))
        for wheel in range(4):
            offset = base + 356 + wheel * 328 + 272
            struct.pack_into('<3I', data, offset, 1, material, 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def embedded_map(replay: Path):
    record = Gbx(replay.read_bytes()).get_classes_by_ids(
        [GbxType.REPLAY_RECORD, GbxType.REPLAY_RECORD_OLD])
    if len(record) != 1:
        raise ValueError(f"{replay}: expected one replay record")
    maps = record[0].track.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    if len(maps) != 1:
        raise ValueError(f"{replay}: expected one embedded challenge")
    challenge_bytes = record[0].track.root_parser.data
    # Parsing the extracted bytes is an important guard against accidentally
    # writing a nested parser buffer or an incomplete chunk.
    extracted = Gbx(challenge_bytes).get_classes_by_ids(
        [GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    if len(extracted) != 1 or extracted[0].map_uid != maps[0].map_uid:
        raise ValueError(f"{replay}: extracted challenge UID does not match replay")
    return maps[0], challenge_bytes


def _replay_identity(replay: str):
    """Read only the metadata needed to deduplicate capture work.

    This is deliberately a top-level worker function: a large corpus can be
    planned in parallel before any Wine lane is occupied.  The full challenge
    bytes are read again only for maps that actually need a fixture capture.
    """
    path = Path(replay).resolve()
    try:
        before = path.stat()
    except OSError as error:
        return {'replay': str(path), 'status': 'failed', 'error': str(error)}
    file_stat = {'size': before.st_size, 'mtime_ns': before.st_mtime_ns,
                 'inode': before.st_ino}
    try:
        challenge, _ = embedded_map(path)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != tuple(file_stat.values()):
            return {'replay': str(path), 'status': 'changed_during_parse',
                    'error': 'replay changed while it was being parsed',
                    '_file_stat': file_stat}
        return {'replay': str(path), 'status': 'ok', 'map_uid': challenge.map_uid,
                'geometry_sha256': geometry_sha256(challenge), '_file_stat': file_stat}
    except Exception as error:
        # Cache stable parse failures for resumability, but never cache an
        # error against a file that was edited while the parser was reading it.
        try:
            after = path.stat()
        except OSError:
            after = None
        if after is not None and (after.st_size, after.st_mtime_ns, after.st_ino) != tuple(file_stat.values()):
            return {'replay': str(path), 'status': 'changed_during_parse',
                    'error': 'replay changed while it was being parsed',
                    '_file_stat': file_stat}
        return {'replay': str(path), 'status': 'failed', 'error': str(error),
                '_file_stat': file_stat}


def _identity_index_connection(path: Path):
    """Open the optional replay identity cache used across fixture runs.

    The cache is intentionally small: it stores file identity plus UID and
    geometry, never embedded challenge bytes.  A size/mtime/inode change
    invalidates a row, so a replay edit cannot silently reuse an old map key.
    Writes happen in the parent after planning workers finish.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=NORMAL')
    connection.execute('''CREATE TABLE IF NOT EXISTS replay_identity (
        replay TEXT PRIMARY KEY,
        size INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        inode INTEGER NOT NULL,
        map_uid TEXT,
        geometry_sha256 TEXT,
        status TEXT NOT NULL,
        error TEXT,
        updated_ns INTEGER NOT NULL
    )''')
    connection.commit()
    return connection


def _indexed_identity(connection, replay: str):
    path = Path(replay).resolve()
    try:
        stat = path.stat()
    except OSError:
        return None
    row = connection.execute(
        'SELECT size, mtime_ns, inode, map_uid, geometry_sha256, status, error '
        'FROM replay_identity WHERE replay = ?', (str(path),)).fetchone()
    if row is None or row[:3] != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
        return None
    if row[5] != 'ok':
        return {'replay': str(path), 'status': row[5],
                'error': row[6] or 'cached planning failure'}
    return {'replay': str(path), 'status': 'ok', 'map_uid': row[3],
            'geometry_sha256': row[4]}


def _update_identity_index(connection, identities):
    import time
    rows = []
    for identity in identities:
        path = Path(identity['replay']).resolve()
        file_stat = identity.get('_file_stat')
        if file_stat is None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
                file_stat['size'], file_stat['mtime_ns'], file_stat['inode']):
            # Never bind a parsed identity to a file that changed after the
            # worker read it. The next planning pass will retry it.
            continue
        status = identity.get('status', 'ok')
        if status == 'changed_during_parse':
            continue
        rows.append((str(path), file_stat['size'], file_stat['mtime_ns'], file_stat['inode'],
                     identity.get('map_uid'), identity.get('geometry_sha256'), status,
                     identity.get('error'), time.time_ns()))
    if rows:
        connection.executemany('''INSERT INTO replay_identity
            (replay, size, mtime_ns, inode, map_uid, geometry_sha256, status, error, updated_ns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(replay) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns,
              inode=excluded.inode, map_uid=excluded.map_uid, geometry_sha256=excluded.geometry_sha256,
              status=excluded.status, error=excluded.error, updated_ns=excluded.updated_ns''', rows)
        connection.commit()


def _unique_replays(replays, workers=1, identity_index=None):
    """Return one deterministic source replay per UID/geometry identity.

    Parsing is the expensive part of fixture planning.  It is independent for
    every replay, so use a bounded process pool when multiple capture lanes are
    requested.  Results retain input order, making provenance and retries
    deterministic even though workers finish out of order.
    """
    paths = [str(Path(path).resolve()) for path in replays]
    connection = _identity_index_connection(identity_index) if identity_index else None
    cached = {}
    misses = []
    for path in paths:
        identity = _indexed_identity(connection, path) if connection else None
        if identity is None:
            misses.append(path)
        else:
            cached[path] = identity
    identities_by_path = dict(cached)
    parsed_count = 0
    chunk_size = max(32, workers * 8)
    if workers > 1 and len(misses) > 1:
        pool = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=multiprocessing.get_context('spawn'))
    else:
        pool = None
    try:
        for start in range(0, len(misses), chunk_size):
            chunk = misses[start:start + chunk_size]
            if pool is None:
                rows = [_replay_identity(path) for path in chunk]
            else:
                rows = list(pool.map(_replay_identity, chunk, chunksize=8))
            parsed_count += len(rows)
            identities_by_path.update({row['replay']: row for row in rows})
            if connection:
                _update_identity_index(connection, rows)
    finally:
        if pool is not None:
            pool.shutdown()
        if connection:
            connection.close()
    identities = [identities_by_path[path] for path in paths]
    unique = []
    unique_metadata = []
    failures = []
    seen = set()
    for identity in identities:
        if identity.get('status') != 'ok' or not identity.get('map_uid'):
            failures.append(identity)
            continue
        key = (identity['map_uid'], identity['geometry_sha256'])
        if key not in seen:
            seen.add(key)
            unique.append(Path(identity['replay']))
            unique_metadata.append(identity)
    return unique, failures, unique_metadata, {
        'input_count': len(paths), 'cached_count': len(cached),
        'parsed_count': parsed_count, 'unique_count': len(unique),
    }


def _prepare_one(job):
    started = time.monotonic()
    phase_seconds: dict[str, float] = {}
    replay, fixture_root, registry, lane, force, reuse, socket_timeout, *mode_args = job
    reference_mode = mode_args[0] if mode_args else 'full'
    route_mode = mode_args[1] if len(mode_args) > 1 else 'dense'
    track_backend = mode_args[2] if len(mode_args) > 2 else 'game'
    packs = Path(mode_args[3]) if len(mode_args) > 3 and mode_args[3] else None
    asset_cache = Path(mode_args[4]) if len(mode_args) > 4 and mode_args[4] else None
    fixture_backend = mode_args[5] if len(mode_args) > 5 else 'game'
    vehicle_template = (Path(mode_args[6]) if len(mode_args) > 6 and mode_args[6]
                        else ROOT / 'oracle/vehicles/A01-Stadium.tmnfvehicle')
    physics_library = (Path(mode_args[7]) if len(mode_args) > 7 and mode_args[7]
                       else ROOT / 'build/local/libtmnf_physics.so')
    exact_triggers = bool(mode_args[8]) if len(mode_args) > 8 else False
    registry_batch_size = int(mode_args[9]) if len(mode_args) > 9 else 1
    registry_rows_dir = (Path(mode_args[10]) if len(mode_args) > 10 and mode_args[10]
                         else None)
    if reference_mode not in ('full', 'assets'):
        raise ValueError(f'unsupported reference mode: {reference_mode}')
    if route_mode not in ('dense', 'validation'):
        raise ValueError(f'unsupported route mode: {route_mode}')
    if track_backend not in ('game', 'offline'):
        raise ValueError(f'unsupported track backend: {track_backend}')
    if track_backend == 'offline' and reference_mode != 'assets':
        raise ValueError('offline track backend currently requires --reference-mode assets')
    if fixture_backend not in ('game', 'offline'):
        raise ValueError(f'unsupported fixture backend: {fixture_backend}')
    if fixture_backend == 'offline':
        if reference_mode != 'assets' or route_mode != 'validation' or track_backend != 'offline':
            raise ValueError('fully offline fixture backend requires assets, validation route, and offline track')
        vehicle_mode = 'graph'
    else:
        vehicle_mode = 'snapshot'
    if exact_triggers and (fixture_backend != 'offline' or reference_mode != 'assets'
                           or route_mode != 'validation' or track_backend != 'offline'):
        raise ValueError('exact trigger geometry requires fully offline assets/validation mode')
    replay = Path(replay).resolve()
    fixture_root = Path(fixture_root).resolve()
    registry = Path(registry).resolve()
    phase_started = time.monotonic()
    challenge, challenge_bytes = embedded_map(replay)
    uid = challenge.map_uid
    geometry = geometry_sha256(challenge)
    stem = f"TMX-{uid[:16]}-{geometry[:12]}"
    output = fixture_root / uid / geometry
    output.mkdir(parents=True, exist_ok=True)
    challenge_path = output / f"{stem}.Challenge.Gbx"
    # Geometry is the simulator identity.  A UID/geometry match with different
    # non-physical metadata reuses the first challenge bytes, keeping every
    # snapshot's challenge hash self-consistent and avoiding duplicate variants.
    if not challenge_path.exists():
        challenge_path.write_bytes(challenge_bytes)
    phase_seconds['extract'] = time.monotonic() - phase_started
    current = EmbeddedTrack(stem, str(challenge.environment), output)
    if fixture_backend == 'offline':
        # Validate the selected package/cache pair before considering an
        # existing fixture, and bind its profile into the builder fingerprint.
        _offline_track_assets(packs, asset_cache)
    paths = [current.snapshot, current.route, current.vehicle]
    if exact_triggers:
        paths.append(current.trigger_geometry)
    if reference_mode == 'full':
        paths.extend([current.result("mixed", "_inputs"), current.result("mixed"),
                      current.result("wall_contact", "_inputs"), current.result("wall_contact")])
    if force or not reuse:
        existing = [path for path in paths if path.exists()]
        if existing and len(existing) < len(paths) and not force:
            raise FileExistsError("partial fixture exists; use --reuse-captures or --force: "
                                  + ", ".join(map(str, existing)))
        if len(existing) == len(paths) and not force and _registered_complete(
                challenge_path, current, uid, geometry, registry, reference_mode, route_mode,
                fixture_backend, track_backend, vehicle_mode, exact_triggers):
            effective_route_mode = _route_mode(current.route) or route_mode
            result = {"replay": str(replay), "map_uid": uid, "geometry_sha256": geometry,
                      "challenge_sha256": _sha256(challenge_path), "fixture_root": str(output),
                      "status": "reused", "reference_mode": reference_mode,
                      "route_mode": effective_route_mode, "track_backend": track_backend,
                      "fixture_backend": fixture_backend, "vehicle_mode": vehicle_mode,
                      "trigger_geometry": exact_triggers,
                      "lane": lane}
            result['elapsed_seconds'] = round(time.monotonic() - started, 3)
            result['phase_seconds'] = {key: round(phase_seconds.get(key, 0.0), 3)
                                       for key in ('extract', 'capture', 'route', 'register')}
            return result
        if force:
            for path in paths:
                path.unlink(missing_ok=True)
            Path(str(current.route) + ".log").unlink(missing_ok=True)

    if fixture_backend == 'offline':
        phase_started = time.monotonic()
        staging = output / 'staging'
        staging.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix='offline-', dir=staging))
        try:
            # Build in an attempt directory, then publish all three outputs
            # only after track/vehicle/route and the checkpoint sidecar exist.
            build_current = EmbeddedTrack(current.name, current.environment_name, attempt)
            build_info = _build_offline_fixture(
                challenge_path, build_current, bytes.fromhex(_sha256(challenge_path)),
                packs, asset_cache, vehicle_template, physics_library, exact_triggers)
            for source, destination in zip(
                    _capture_outputs('assets', build_current, exact_triggers),
                    _capture_outputs('assets', current, exact_triggers)):
                temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            shutil.copyfile(Path(str(build_current.route) + '.log'),
                            Path(str(current.route) + '.log'))
            phase_seconds['capture'] = time.monotonic() - phase_started
        finally:
            if attempt.exists():
                shutil.rmtree(attempt, ignore_errors=True)
        phase_started = time.monotonic()
        publish_registry = registry_batch_size <= 1
        row_registry = registry
        if not publish_registry:
            if registry_rows_dir is None:
                raise ValueError('batched offline registration requires a row directory')
            registry_rows_dir.mkdir(parents=True, exist_ok=True)
            row_registry = registry_rows_dir / f'{uid}-{geometry}.json'
        if publish_registry:
            fixture = _register_variant(
                challenge_path, current, row_registry, replay, reference_mode, route_mode,
                fixture_backend, track_backend, vehicle_mode, vehicle_template, exact_triggers)
        else:
            fixture = _variant_row(
                challenge_path, current, replay, reference_mode, route_mode,
                fixture_backend, track_backend, vehicle_mode, vehicle_template, exact_triggers)
            _write_variant_row_provenance(fixture, row_registry)
        phase_seconds['register'] = time.monotonic() - phase_started
        result = {"replay": str(replay), "map_uid": uid, "geometry_sha256": geometry,
                  "challenge_sha256": fixture['challenge_sha256'], "fixture_root": str(output),
                  "status": "registered", "reference_mode": reference_mode,
                  "route_mode": fixture.get('route_mode', route_mode),
                  "track_backend": track_backend, "fixture_backend": fixture_backend,
                  "vehicle_mode": vehicle_mode, "trigger_geometry": exact_triggers,
                  "lane": lane}
        if build_info.get('ignored_missing_blocks'):
            result['ignored_missing_blocks'] = build_info['ignored_missing_blocks']
        if not publish_registry:
            result['registry_row_path'] = str(row_registry)
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        result['phase_seconds'] = {key: round(phase_seconds.get(key, 0.0), 3)
                                   for key in ('extract', 'capture', 'route', 'register')}
        return result

    # Import after argument parsing so this helper remains cheap for planning
    # and so onboard_track's lane globals are private to this worker process.
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import onboard_track
    onboard_track.set_lane(Path(lane["wineprefix"]), int(lane["port"]),
                           str(lane["display"]), str(lane["cpu_set"]))
    # launch_game.sh reads these paths for the tracer's game-side outputs.
    old_environment = {key: os.environ.get(key) for key in
                       ["TMNF_TRACK_DIR", "TMNF_ROUTE_DIR", "TMNF_TRACE_HOST_DIR",
                        "TMNF_TMI_SOCKET_TIMEOUT"]}
    os.environ.update({"TMNF_TRACK_DIR": str(output), "TMNF_ROUTE_DIR": str(output),
                       "TMNF_TRACE_HOST_DIR": str(output / "traces"),
                       "TMNF_TMI_SOCKET_TIMEOUT": str(socket_timeout)})
    try:
        staging = None
        missing = [path for path in paths if not path.is_file()]
        if missing:
            # Each attempt keeps its own logs and trace directories. A retry
            # must not reuse vehicle traces left by an interrupted capture.
            attempts = output / "staging"
            attempts.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts))
            challenge_hash = bytes.fromhex(_sha256(challenge_path))
            phase_started = time.monotonic()
            _capture_with_short_workspace(
                onboard_track, current, challenge_path, challenge_hash,
                reference_mode, staging, track_backend, packs, asset_cache)
            phase_seconds['capture'] = time.monotonic() - phase_started
        # The game tracer emits a version-1 route whose trigger/spawn records
        # are valid but whose reference section is still sparse.  Dense mode
        # generates the centerline from captured drives; validation mode only
        # upgrades the route with a sparse start/checkpoint/finish reference.
        phase_started = time.monotonic()
        route_version = struct.unpack_from('<I', current.route.read_bytes(), 8)[0]
        current_route_mode = _route_mode(current.route)
        needs_route_upgrade = route_version != 3 or (
            route_mode == 'dense' and current_route_mode == 'validation')
        if needs_route_upgrade:
            if staging is None:
                attempts = output / "staging"
                attempts.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts))
            if route_mode == 'validation':
                validation_route = staging / f'{current.route.stem}-validation.tmnfroute'
                route_command = [
                    onboard_track.PYTHON, ROOT / 'tools/build_validation_route.py',
                    '--route', current.route, '--output', validation_route,
                ]
                onboard_track.run(onboard_track.taskset(route_command),
                                  env=onboard_track.clean_environment(), timeout=600)
                validation_route.replace(current.route)
            else:
                if reference_mode == 'assets':
                    ghost_drive = staging / 'ghost_position_drive.bin'
                    _write_ghost_position_drive(replay, ghost_drive, current.snapshot)
                    run1 = long_drive = validation_drive = ghost_drive
                else:
                    run1 = current.result("mixed")
                    long_drive = current.result("wall_contact")
                    validation_drive = current.result("mixed")
                route_command = [
                    onboard_track.PYTHON, ROOT / "tools/generate_route_centerline.py",
                    "--track", current.snapshot, "--route", current.route,
                    "--run1", run1, "--long-drive", long_drive,
                    "--validation-drive", validation_drive,
                    "--ghost-replay", replay, "--ghost-optional",
                ]
                onboard_track.run(onboard_track.taskset(route_command),
                                  env=onboard_track.clean_environment(), timeout=600)
                onboard_track.run(onboard_track.taskset([*route_command, "--check"]),
                                  env=onboard_track.clean_environment(), timeout=600)
            phase_seconds['route'] = time.monotonic() - phase_started
        else:
            phase_seconds.setdefault('route', 0.0)
        _complete_fixture_collision_scene(
            challenge_path, current, packs, asset_cache)
        phase_started = time.monotonic()
        fixture = _register_variant(
            challenge_path, current, registry, replay, reference_mode, route_mode,
            fixture_backend, track_backend, vehicle_mode)
        phase_seconds['register'] = time.monotonic() - phase_started
    finally:
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result = {"replay": str(replay), "map_uid": uid, "geometry_sha256": geometry,
              "challenge_sha256": fixture['challenge_sha256'], "fixture_root": str(output),
              "status": "registered", "reference_mode": reference_mode,
              "route_mode": fixture.get('route_mode', route_mode),
              "track_backend": track_backend, "fixture_backend": fixture_backend,
              "vehicle_mode": vehicle_mode, "trigger_geometry": exact_triggers,
              "lane": lane}
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    result['phase_seconds'] = {key: round(phase_seconds.get(key, 0.0), 3)
                               for key in ('extract', 'capture', 'route', 'register')}
    return result


def _prepare_result(job):
    started = time.monotonic()
    try:
        return _prepare_one(job)
    except Exception as error:
        return {"replay": job[0], "status": "failed", "error": str(error), "lane": job[3],
                "track_backend": job[9] if len(job) > 9 else 'game',
                "fixture_backend": job[12] if len(job) > 12 else 'game',
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "phase_seconds": {}}


def _prepare_lane(jobs):
    """Run one fixed lane's jobs serially; no prefix is shared by processes."""
    return [_prepare_result(job) for job in jobs]


def _iter_lane_results(jobs, workers):
    """Yield capture results while keeping at most one job per lane queued.

    A whole-lane ``executor.map`` hides progress until that lane finishes and
    queues an unbounded amount of work in the process pool. This scheduler
    keeps each lane exclusive, but submits the next map only after that lane's
    previous map has returned. At most ``workers`` capture jobs are resident.
    """
    groups = [[] for _ in range(workers)]
    for index, job in enumerate(jobs):
        groups[index % workers].append(job)
    if workers == 1:
        yield from map(_prepare_result, groups[0])
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterators = [iter(group) for group in groups]
        futures = {}
        for lane, iterator in enumerate(iterators):
            try:
                futures[executor.submit(_prepare_result, next(iterator))] = (lane, iterator)
            except StopIteration:
                pass
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                lane, iterator = futures.pop(future)
                yield future.result()
                try:
                    futures[executor.submit(_prepare_result, next(iterator))] = (lane, iterator)
                except StopIteration:
                    pass


def _write_report_atomic(path: Path, report: dict) -> None:
    """Publish the latest preparation state without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def _load_replays(args) -> list[Path]:
    if args.replay:
        if args.source_dir:
            raise ValueError("use --replay or --source-dir, not both")
        return [Path(path) for path in args.replay]
    if args.source_dir:
        source = Path(args.source_dir)
        if not source.is_dir():
            raise ValueError(f"replay source directory does not exist: {source}")
        paths = sorted(source.rglob('*.Replay.Gbx'))
        if not paths:
            raise ValueError(f"{source}: no *.Replay.Gbx files")
    else:
        if not args.manifest.exists():
            raise ValueError("provide --replay, --source-dir, or an existing --manifest")
        rows = json.loads(args.manifest.read_text()).get("cases", [])
        if not rows:
            raise ValueError(f"{args.manifest}: no matched cases")
        # The manifest is already frozen, but preserve deterministic seeded
        # subsampling when a smaller run is requested.
        paths = [Path(row["replay"]) for row in rows]
    if args.limit is not None and args.limit < len(paths):
        import random
        random.Random(args.seed).shuffle(paths)
        paths = paths[:args.limit]
    return paths


def _default_lane_cpu_set(index: int) -> str:
    """Choose a distinct physical/SMT pair for each default Wine lane."""
    total = os.cpu_count() or 2
    half = max(1, total // 2)
    # Keep the first lane on the pair used by the validated fixture captures;
    # subsequent lanes move to adjacent pairs instead of contending on 14,30.
    first = (min(14, half - 1) + index) % half
    second = first + half if total > half else first
    return f'{first},{second}' if second != first else str(first)


def _prefix_in_use(prefix: Path, proc_root: Path = Path('/proc')) -> bool:
    """Best-effort refusal to clone a prefix being used by Wine.

    Match the process' canonical ``WINEPREFIX`` environment entry and its
    ``comm`` name (or an exact Wine executable argument).  Looking for the
    prefix as a substring of arbitrary command-line text incorrectly matches
    this planner when its ``--template-prefix`` argument contains ``wine``.
    Only the single WINEPREFIX environment entry is read; the rest is never
    exposed or logged.
    """
    resolved = prefix.resolve()
    wine_executables = {
        'wine', 'wine64', 'wine-preloader', 'wine64-preloader',
        'wineserver', 'wineboot', 'wineboot.exe',
    }
    for process in proc_root.glob('[0-9]*'):
        try:
            if process.name == str(os.getpid()):
                continue
            comm = (process / 'comm').read_text(errors='ignore').strip().lower()
            environment = (process / 'environ').read_bytes()
            wineprefix = next(
                (entry[len(b'WINEPREFIX='):].decode(errors='ignore')
                 for entry in environment.split(b'\0')
                 if entry.startswith(b'WINEPREFIX=')),
                None,
            )
            if wineprefix is None or Path(wineprefix).expanduser().resolve() != resolved:
                continue
            command = (process / 'cmdline').read_bytes().split(b'\0')
            executable_args = {
                Path(argument.decode(errors='ignore')).name.lower()
                for argument in command if argument
            }
            is_wine = (
                comm.startswith('wine') or comm.startswith('wineserver') or
                comm.endswith('.exe') or bool(executable_args & wine_executables)
            )
            if is_wine:
                return True
        except (OSError, ValueError):
            continue
    return False


LANE_MARKER = '.tmnf-fixture-lane.json'


def _template_identity(template: Path) -> dict[str, object]:
    """Return a stable identity for a lane-clone source directory."""
    stat = template.stat()
    return {
        'canonical_path': str(template.resolve()),
        'device': stat.st_dev,
        'inode': stat.st_ino,
        'mtime_ns': stat.st_mtime_ns,
    }


def _verified_lane(target: Path, template_identity: dict[str, object], index: int) -> bool:
    """Accept only a lane previously created by this workflow."""
    marker = target / LANE_MARKER
    if not target.is_dir() or not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    return (data.get('schema') == 'tmnf-fixture-lane-v1' and
            data.get('lane_index') == index and
            data.get('template') == template_identity)


def _write_lane_marker(target: Path, template_identity: dict[str, object], index: int) -> None:
    marker = target / LANE_MARKER
    temporary = marker.with_name(f'.{marker.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps({
        'schema': 'tmnf-fixture-lane-v1',
        'lane_index': index,
        'template': template_identity,
    }, sort_keys=True, indent=2) + '\n')
    temporary.replace(marker)


def _provision_lane_prefixes(template_prefix: Path, lane_root: Path, workers: int) -> list[Path]:
    """Clone private lane prefixes with reflinks when the filesystem supports them.

    Existing lane directories are never overwritten.  A lane created by this
    workflow can be reused only when its marker matches the canonical template
    identity; arbitrary pre-existing directories remain an error.  A failed
    newly-created clone is removed so a partial prefix cannot be mistaken for
    a usable lane on retry.
    """
    template = Path(template_prefix).expanduser().resolve()
    root = Path(lane_root).expanduser().resolve()
    if not template.is_dir():
        raise ValueError(f'--template-prefix is not a directory: {template}')
    if root == template or root.is_relative_to(template) or template.is_relative_to(root):
        raise ValueError('--lane-root and --template-prefix must not be nested aliases')
    if _prefix_in_use(template):
        raise RuntimeError(f'template Wine prefix appears to be in use: {template}')
    root.mkdir(parents=True, exist_ok=True)
    template_identity = _template_identity(template)
    targets = [root / f'lane-{index}' for index in range(workers)]
    if len({str(target.resolve()) for target in targets}) != len(targets):
        raise ValueError('--lane-root generated duplicate lane prefixes')
    for index, target in enumerate(targets):
        if os.path.lexists(target):
            if not _verified_lane(target, template_identity, index):
                raise FileExistsError(
                    f'refusing to overwrite unverified existing lane prefix: {target}')
            if _prefix_in_use(target):
                raise RuntimeError(f'lane Wine prefix appears to be in use: {target}')
            continue
        try:
            subprocess.run(['cp', '-a', '--reflink=auto', str(template), str(target)],
                           check=True)
        except FileNotFoundError:
            shutil.copytree(template, target, symlinks=True)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        try:
            _write_lane_marker(target, template_identity, index)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="append", help="embedded replay path (repeatable)")
    parser.add_argument("--source-dir", type=Path,
                        help="recursively discover *.Replay.Gbx files without expanding a shell argument list")
    parser.add_argument("--manifest", type=Path, default=SETTINGS["manifest"])
    parser.add_argument("--limit", type=int, default=SETTINGS["limit"])
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--wineprefix", action="append", default=[])
    parser.add_argument("--port", action="append", type=int, default=[])
    parser.add_argument("--display", action="append", default=[])
    parser.add_argument("--cpu-set", action="append", default=[])
    parser.add_argument("--lane-root", type=Path, default=SETTINGS["lane_root"],
                        help="optional parent for private per-worker Wine prefixes")
    parser.add_argument("--template-prefix", type=Path,
                        default=SETTINGS["template_prefix"],
                        help="template prefix cloned into --lane-root/lane-N")
    parser.add_argument("--workers", type=int, default=SETTINGS["workers"])
    parser.add_argument("--plan-workers", type=int, default=SETTINGS["plan_workers"],
                        help="parallel replay parsers used before capture (defaults to --workers)")
    parser.add_argument("--identity-index", type=Path,
                        help="optional SQLite replay UID/geometry cache for resumable planning")
    parser.add_argument("--reference-mode", choices=['full', 'assets'],
                        default=SETTINGS["reference_mode"],
                        help="full captures validation references; assets keeps only track/route/vehicle")
    parser.add_argument("--route-mode", choices=['dense', 'validation'],
                        default=SETTINGS["route_mode"],
                        help="dense builds the full centerline; validation keeps only route anchors")
    parser.add_argument("--track-backend", choices=['game', 'offline'],
                        default=SETTINGS["track_backend"],
                        help="build the track in-game (default) or with tools/build_track")
    parser.add_argument("--packs", type=Path, default=SETTINGS["packs"],
                        help="game Packs directory for --track-backend offline")
    parser.add_argument("--asset-cache", type=Path, default=SETTINGS["asset_cache"],
                        help="build_track extracted/JSON cache for --track-backend offline")
    parser.add_argument("--fixture-backend", choices=['game', 'offline'],
                        default=SETTINGS["fixture_backend"],
                        help="game (default) or experimental fully offline Stadium assets")
    parser.add_argument("--vehicle-template", type=Path,
                        default=SETTINGS["vehicle_template"],
                        help="v4 Stadium graph template for --fixture-backend offline")
    parser.add_argument("--physics-library", type=Path,
                        default=SETTINGS["physics_library"],
                        help="native physics library for the offline route builder")
    parser.add_argument("--exact-triggers", action="store_true",
                        default=SETTINGS["exact_triggers"],
                        help="offline-only: emit a .tmnftriggers sidecar with linked waypoint meshes")
    parser.add_argument("--registry-batch-size", type=int, default=1,
                        help="offline-only: merge crash-safe per-map registry rows every N results")
    parser.add_argument("--assets-only", action="store_true",
                        help="alias for --reference-mode assets (convenient for IDE/batch overrides)")
    parser.add_argument("--socket-timeout", type=float, default=SETTINGS["socket_timeout"])
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--force", action="store_true")
    recovery.add_argument("--reuse-captures", action="store_true", default=SETTINGS["reuse_captures"])
    parser.add_argument("--plan", action="store_true",
                        help="extract and print deterministic jobs without launching Wine")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "local/fixture-preparation.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.plan_workers is not None and args.plan_workers < 1:
        parser.error("--plan-workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.registry_batch_size < 1:
        parser.error("--registry-batch-size must be positive")
    if (args.lane_root is None) != (args.template_prefix is None):
        parser.error("--lane-root and --template-prefix must be provided together")
    if args.assets_only and args.reference_mode == 'full':
        reference_mode = 'assets'
    else:
        reference_mode = args.reference_mode
    if args.track_backend == 'offline' and reference_mode != 'assets':
        parser.error('--track-backend offline currently requires --assets-only or --reference-mode assets')
    if args.fixture_backend == 'offline':
        if reference_mode != 'assets' or args.route_mode != 'validation' or args.track_backend != 'offline':
            parser.error('--fixture-backend offline requires --assets-only, --route-mode validation, and --track-backend offline')
        if args.lane_root is not None or args.template_prefix is not None:
            parser.error('--fixture-backend offline does not use Wine lanes; omit --lane-root/--template-prefix')
        if not args.vehicle_template.is_file():
            parser.error(f'--vehicle-template is not a file: {args.vehicle_template}')
        if not args.physics_library.is_file():
            parser.error(f'--physics-library is not a file: {args.physics_library}')
    elif args.registry_batch_size != 1:
        parser.error('--registry-batch-size is only supported with --fixture-backend offline')
    if args.exact_triggers and (args.fixture_backend != 'offline'
                                or reference_mode != 'assets'
                                or args.route_mode != 'validation'
                                or args.track_backend != 'offline'):
        parser.error('--exact-triggers requires fully offline assets/validation mode')
    replays = _load_replays(args)
    lane_defaults = {
        "wineprefix": str(ROOT / "oracle/wineprefix-validation"),
        "port": 8488, "display": "98"}
    provisioned_prefixes = []
    if args.fixture_backend != 'offline' and args.lane_root is not None:
        provisioned_prefixes = _provision_lane_prefixes(
            args.template_prefix, args.lane_root, args.workers)
    lanes = []
    for index in range(args.workers):
        if args.fixture_backend == 'offline':
            lanes.append({"wineprefix": "offline", "port": 0,
                          "display": "", "cpu_set": str(index)})
        else:
            lanes.append({
                "wineprefix": str(Path(args.wineprefix[index] if index < len(args.wineprefix)
                                       else (provisioned_prefixes[index] if provisioned_prefixes
                                             else lane_defaults["wineprefix"])).expanduser().resolve()),
                "port": args.port[index] if index < len(args.port) else lane_defaults["port"] + index,
                "display": args.display[index] if index < len(args.display) else str(98 + index),
                "cpu_set": args.cpu_set[index] if index < len(args.cpu_set)
                           else _default_lane_cpu_set(index),
            })
    if args.workers > 1:
        lane_keys = ["cpu_set"] if args.fixture_backend == 'offline' else [
            "wineprefix", "port", "display", "cpu_set"]
        for key in lane_keys:
            values = [lane[key] for lane in lanes]
            if len(set(values)) != len(values):
                parser.error(f"--workers requires distinct lane {key} values")
    # A replay may occur in several source locations; capture once per
    # (UID,geometry), retaining the first deterministic replay as provenance.
    plan_workers = args.plan_workers if args.plan_workers is not None else args.workers
    replays, planning_failures, planned_identities, planning_stats = _unique_replays(
        replays, plan_workers, args.identity_index)
    registry_rows_dir = None
    if args.registry_batch_size > 1 and not args.plan:
        registry_rows_dir = args.registry.with_name(args.registry.name + '.rows')
        # Complete a prior interrupted merge before workers inspect the shared
        # registry, making per-map provenance restartable without rescanning
        # the replay corpus.
        _merge_registry_rows(args.registry, registry_rows_dir)
    if args.track_backend == 'offline' and not args.plan and replays:
        # Build/extract the shared asset cache before capture workers start.
        # Worker processes still create their own read-only Assets memoizers.
        _offline_cache_context(args.packs, args.asset_cache)
    jobs = [(str(path), str(args.fixture_root), str(args.registry), lanes[i % args.workers],
             args.force, args.reuse_captures, args.socket_timeout, reference_mode,
             args.route_mode, args.track_backend,
             str(args.packs) if args.packs else None,
             str(args.asset_cache) if args.asset_cache else None,
             args.fixture_backend,
             str(args.vehicle_template) if args.vehicle_template else None,
             str(args.physics_library) if args.physics_library else None,
             args.exact_triggers, args.registry_batch_size,
             str(registry_rows_dir) if registry_rows_dir is not None else None)
            for i, path in enumerate(replays)]
    if args.plan:
        for job, identity in zip(jobs, planned_identities):
            print(json.dumps({**identity, "lane": job[3], "route_mode": args.route_mode,
                              "track_backend": args.track_backend,
                              "fixture_backend": args.fixture_backend,
                              "exact_triggers": args.exact_triggers},
                             sort_keys=True))
        for failure in planning_failures:
            print(json.dumps(failure, sort_keys=True))
        return 0 if not planning_failures else 1
    report = {"schema": "tmnf-collection-fixture-preparation-v1", "seed": args.seed,
              "workers": args.workers, "plan_workers": plan_workers,
              "identity_index": str(args.identity_index) if args.identity_index else None,
              "reference_mode": reference_mode,
              "route_mode": args.route_mode,
              "track_backend": args.track_backend,
              "fixture_backend": args.fixture_backend,
              "vehicle_mode": "graph" if args.fixture_backend == "offline" else "snapshot",
              "exact_triggers": args.exact_triggers,
              "registry_batch_size": args.registry_batch_size,
              "registry_rows_dir": str(registry_rows_dir) if registry_rows_dir else None,
              "vehicle_template": str(args.vehicle_template) if args.fixture_backend == "offline" else None,
              "physics_library": str(args.physics_library) if args.fixture_backend == "offline" else None,
              "packs": str(args.packs) if args.packs else None,
              "asset_cache": str(args.asset_cache) if args.asset_cache else None,
              "capture_estimate": _capture_estimate(reference_mode, args.track_backend,
                                                     args.fixture_backend),
              "planning": planning_stats, "planned_replay_count": len(replays), "results": [],
              "failures": list(planning_failures)}
    _write_report_atomic(args.report, report)
    pending_rows = 0
    try:
        for result in _iter_lane_results(jobs, args.workers):
            if result["status"] == "failed":
                report["failures"].append(result)
            else:
                report["results"].append(result)
                if registry_rows_dir is not None:
                    pending_rows += 1
                    if pending_rows >= args.registry_batch_size:
                        _merge_registry_rows(args.registry, registry_rows_dir)
                        pending_rows = 0
            print(json.dumps(result), flush=True)
            _write_report_atomic(args.report, report)
    finally:
        if registry_rows_dir is not None:
            _merge_registry_rows(args.registry, registry_rows_dir)
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
