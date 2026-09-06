#!/usr/bin/env python3
"""Generate reviewable route candidates from existing offline fixtures.

No committed route is overwritten. Completed cases are reusable on rerun;
source hashes must still match. Each generator child has a memory/time cap.
PYTHONPATH=python python tools/regenerate_route_corpus.py --output DIRECTORY
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import time

from audit_route_endpoints import audit_route
from generate_route_centerline import route_sections
from refresh_route_trigger_transforms import align_selected_finish, refresh
from onboard_track import replay_for, replay_state, reference_paths
from track_env import game_layout, track_from_manifest
from tmnf_rl.tracks import project_root, track_catalogue


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def changes(before, after):
    old, new = before.read_bytes(), after.read_bytes()
    a, b = route_sections(old), route_sections(new)
    fixed_equal = all(old[o:o+n*s] == new[p:p+m*t]
                      for (o,n,s),(p,m,t) in zip(a[1:4], b[1:4]))
    o,n,s = a[4];p,m,t = b[4]
    prefix = 0
    for i in range(min(n,m)):
        if old[o+i*s:o+(i+1)*s] != new[p+i*t:p+(i+1)*t]:
            break
        prefix += 1
    return dict(start_and_triggers_identical=fixed_equal,
                identical_prefix_points=prefix, original_points=n, candidate_points=m,
                first_changed_arc_m=(struct.unpack_from('<f', old, o+prefix*s+12)[0]
                                     if prefix<n else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tracks', nargs='+')
    parser.add_argument('--trigger-audit', type=Path,
                        help='aligned captured-trigger audit; refresh verified transforms before generation')
    parser.add_argument('--memory-gib', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--generator-python', type=Path,
                        default=project_root()/'third_party/venv/bin/python')
    args = parser.parse_args()
    if args.memory_gib < 1 or args.timeout < 1:
        parser.error('memory and timeout must be positive')
    probe = subprocess.run([str(args.generator_python), '-c',
        'import sys,pygbx,json; print(json.dumps(dict(python=sys.version,package=pygbx.__file__)))'],
        capture_output=True, text=True)
    if probe.returncode:
        parser.error('generator Python must provide pygbx: ' + probe.stderr.strip())
    root = project_root(); output = args.output.resolve()
    trigger_audit = None
    if args.trigger_audit is not None:
        trigger_audit = json.loads(args.trigger_audit.read_text())
        if trigger_audit.get('format') != 'tmnf-captured-trigger-audit-v1':
            parser.error('unsupported trigger audit format')
        trigger_rows = {row['track']: row for row in trigger_audit['rows']}
        if len(trigger_rows) != len(trigger_audit['rows']):
            parser.error('duplicate tracks in trigger audit')
    output.mkdir(parents=True, exist_ok=True)
    source = root/'tools/generate_route_centerline.py'
    frozen = output/'generate_route_centerline.py'
    if frozen.exists():
        if sha(frozen) != sha(source):
            raise RuntimeError('generator changed; use a separate output directory')
    else:
        shutil.copy2(source, frozen)
    replay_modes = {}
    for line in (root/'oracle/tracks/manifest.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            track_id, _, _, modes = line.split('|')
            replay_modes[track_id] = replay_state(modes)
    catalogue = track_catalogue(root)
    ids = args.tracks if args.tracks else sorted(catalogue)
    if len(ids) != len(set(ids)) or any(track not in catalogue for track in ids):
        parser.error('tracks must be unique catalogue IDs')
    runtime = json.loads(probe.stdout)
    package = Path(runtime['package']).parent
    package_digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        package_digest.update(str(path.relative_to(package)).encode()+b'\0'+path.read_bytes()+b'\0')
    report = dict(format='tmnf-route-regeneration-v1', generator_sha256=sha(frozen),
                  pygbx_sha256=package_digest.hexdigest(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  memory_limit_gib=args.memory_gib, timeout_seconds=args.timeout,
                  generator_python=str(args.generator_python.absolute()),
                  generator_runtime=runtime, rows=[])
    if trigger_audit is not None:
        report['trigger_audit_sha256'] = sha(args.trigger_audit)
        report['trigger_refresh_sha256'] = sha(root/'tools/refresh_route_trigger_transforms.py')
    for track_id in ids:
        spec = catalogue[track_id]; folder=output/track_id;folder.mkdir(exist_ok=True)
        track = track_from_manifest(track_id, spec.name)
        prefix = root/'oracle'/('wineprefix_united' if track.campaign=='united' else 'wineprefix')
        row = dict(track=track_id, passed=False, generator_sha256=report['generator_sha256'],
                   pygbx_sha256=report['pygbx_sha256'], memory_limit_gib=args.memory_gib)
        try:
            challenge = track.challenge(game_layout(prefix).game_dir)
            ghost = replay_for(challenge)
            if track_id == 'a01':
                mixed, wall = root/'oracle/results/run1.bin', root/'oracle/results/a01_long_drive.bin'
                ghost = None  # A01's retained route uses the empirical road graph.
            else:
                modes = replay_modes[track_id]
                mixed = reference_paths(track_id, 'mixed', modes['mixed'])[0]
                wall = reference_paths(track_id, 'wall_contact', modes['wall_contact'])[0]
            inputs = [spec.track_path(root), spec.route_path(root), mixed, wall]
            if ghost is not None:
                inputs.append(ghost)
            row['inputs'] = {str(path.relative_to(root)):sha(path) for path in inputs}
            repaired = None
            if trigger_audit is not None:
                audited = trigger_rows[track_id]
                if not audited.get('passed'):
                    raise RuntimeError('trigger capture did not pass aligned validation')
                capture = Path(trigger_audit['captures'])/track_id/'routes'/spec.route_path(root).name
                original_data = spec.route_path(root).read_bytes()
                captured_data = capture.read_bytes()
                if hashlib.sha256(original_data).hexdigest() != audited['original_sha256'] or \
                        hashlib.sha256(captured_data).hexdigest() != audited['captured_sha256']:
                    raise RuntimeError('trigger audit input hash changed')
                aligned, omitted = align_selected_finish(original_data, captured_data)
                repaired, corrected = refresh(original_data, aligned)
                if hashlib.sha256(repaired).hexdigest() != audited['transform_only_candidate_sha256']:
                    raise RuntimeError('trigger repair disagrees with audit candidate')
                row['inputs'].update(trigger_audit=report['trigger_audit_sha256'],
                                     trigger_refresh=report['trigger_refresh_sha256'],
                                     captured_route=audited['captured_sha256'])
                row['trigger_refresh'] = dict(changes=corrected, omitted_finish_blocks=omitted)
            old_report = folder/'result.json'
            if old_report.exists():
                previous = json.loads(old_report.read_text())
                if previous.get('inputs') != row['inputs']:
                    raise RuntimeError('case inputs changed; use a separate output directory')
                if previous.get('passed'):
                    if (not previous.get('idempotence_checked') or
                        previous.get('generator_sha256') != row['generator_sha256'] or
                        previous.get('pygbx_sha256') != row['pygbx_sha256']):
                        raise RuntimeError('validation protocol changed; use a separate output directory')
                    if sha(folder/spec.route_path(root).name) != previous['candidate']['route_sha256']:
                        raise RuntimeError('completed candidate hash changed')
                    report['rows'].append(previous)
                    continue
                raise RuntimeError('previous attempt failed; inspect it before a separate retry')
            candidate = folder/spec.route_path(root).name
            shutil.copy2(spec.route_path(root), folder/'original.tmnfroute')
            shutil.copy2(spec.route_path(root), candidate)
            if repaired is not None:
                shutil.copy2(capture, folder/'captured.tmnfroute')
                candidate.write_bytes(repaired)
                (folder/'transform-corrected.tmnfroute').write_bytes(repaired)
            # Resolving the venv's interpreter symlink loses its site-packages.
            command = [str(args.generator_python.absolute()),str(frozen),'--track',str(spec.track_path(root)),
                       '--route',str(candidate),'--run1',str(mixed),'--long-drive',str(wall),
                       '--validation-drive',str(mixed)]
            if ghost is not None:
                command += ['--ghost-replay',str(ghost),'--ghost-optional']
            def limits():
                memory=args.memory_gib*1024**3
                resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
            started = time.monotonic()
            with (folder/'generator.log').open('w') as log:
                result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                    timeout=args.timeout,preexec_fn=limits,
                    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            row.update(seconds=time.monotonic()-started,command=command,exit_code=result.returncode)
            if result.returncode:
                raise RuntimeError(f'generator exited {result.returncode}; see generator.log')
            with (folder/'idempotence.log').open('w') as log:
                checked = subprocess.run(command+['--check'],stdout=log,stderr=subprocess.STDOUT,
                    timeout=args.timeout,preexec_fn=limits,
                    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            if checked.returncode:
                raise RuntimeError(f'idempotence check exited {checked.returncode}; see idempotence.log')
            row['idempotence_checked'] = True
            row['seconds_including_check'] = time.monotonic()-started
            row['peak_child_rss_kib_so_far'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            row.update(original=audit_route(folder/'original.tmnfroute'),candidate=audit_route(candidate),
                       changes=changes(folder/'original.tmnfroute',candidate),passed=True)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
            row['error'] = str(error)
        if not (folder/'result.json').exists():
            write_json(folder/'result.json',row)
        report['rows'].append(row)
        write_json(output/'report.json',report)
        print(json.dumps({k:row[k] for k in ['track','passed','seconds','error'] if k in row}),flush=True)
    report['passed'] = all(row['passed'] for row in report['rows'])
    write_json(output/'report.json',report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
