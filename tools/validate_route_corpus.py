#!/usr/bin/env python3
"""Validate completed regeneration cases without modifying the route corpus.

Compare original/candidate native exposed-state captures and check exact
oracle fixtures. Pending oracle fixtures are labelled, never called exact.
Rerun against an incremental generation directory to validate new cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from onboard_track import reference_paths, replay_state
from regenerate_route_corpus import sha, write_json
from generate_route_centerline import route_sections
from refresh_route_trigger_transforms import align_selected_finish, refresh
from tmnf_rl.tracks import project_root, track_catalogue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('generation', type=Path)
    parser.add_argument('--harness', type=Path,
                        default=project_root() / 'build/tests/replay_tick')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('timeout must be positive')
    root = project_root()
    folder = args.generation.resolve()
    harness = args.harness.resolve()
    harness_hash = sha(harness)
    catalogue = track_catalogue(root)
    manifest = {}
    for line in (root / 'oracle/tracks/manifest.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            track_id, _, _, modes = line.split('|')
            manifest[track_id] = replay_state(modes)
    generation = json.loads((folder / 'report.json').read_text())
    output = folder / 'native-validation'
    output.mkdir(exist_ok=True)
    rows = []
    for generated in generation['rows']:
        if not generated['passed']:
            continue
        track_id = generated['track']
        spec = catalogue[track_id]
        case = folder / track_id
        candidate = case / spec.route_path(root).name
        original = case / 'original.tmnfroute'
        record = output / f'{track_id}.json'
        row = dict(track=track_id, passed=False, schedules=[])
        try:
            if sha(candidate) != generated['candidate']['route_sha256']:
                raise ValueError('candidate no longer matches generation report')
            if sha(original) != generated['original']['route_sha256']:
                raise ValueError('original no longer matches generation report')
            paths = [harness, candidate, original, spec.track_path(root),
                     spec.vehicle_path(root)]
            if not generated['changes']['start_and_triggers_identical']:
                if 'trigger_refresh' not in generated:
                    raise ValueError('start or trigger records changed without captured evidence')
                captured = case/'captured.tmnfroute'
                if sha(captured) != generated['inputs']['captured_route']:
                    raise ValueError('captured route hash changed')
                aligned, omitted = align_selected_finish(original.read_bytes(), captured.read_bytes())
                expected, corrected = refresh(original.read_bytes(), aligned)
                actual = candidate.read_bytes()
                a, b = route_sections(expected), route_sections(actual)
                if any((n, s) != (m, t) or expected[o:o+n*s] != actual[p:p+m*t]
                       for (o, n, s), (p, m, t) in zip(a[1:4], b[1:4])):
                    raise ValueError('candidate start/triggers disagree with captured correction')
                if expected[a[0][0]:a[0][0]+28] != actual[b[0][0]:b[0][0]+28]:
                    raise ValueError('candidate race metadata changed')
                paths.append(captured)
                row['trigger_correction_verified'] = dict(changes=corrected,
                                                          omitted_finish_blocks=omitted)
            schedules = []
            if track_id == 'a01':
                for name, reference in [('run1', 'run1.bin'),
                                        ('long_drive', 'a01_long_drive.bin')]:
                    ref = root / 'oracle/results' / reference
                    paths.append(ref)
                    schedules.append((name, ref, None, True))
            else:
                for name in ('mixed', 'wall_contact'):
                    exact = manifest[track_id][name]
                    ref, inputs = reference_paths(track_id, name, exact)
                    paths.extend([ref, inputs])
                    schedules.append((name, ref, inputs, exact))
            row['input_sha256'] = {str(p): sha(p) for p in paths}
            if record.exists():
                old = json.loads(record.read_text())
                if old.get('input_sha256') != row['input_sha256']:
                    raise ValueError('validation inputs changed; use a separate generation directory')
                rows.append(old)
                continue

            def run(command, log):
                with (output / log).open('w') as stream:
                    result = subprocess.run([str(p) for p in command], stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=args.timeout)
                if result.returncode:
                    raise ValueError(f'{log}: harness exited {result.returncode}')

            for name, ref, inputs, exact in schedules:
                schedule = dict(name=name, oracle_exact=exact,
                                native_capture_compared=inputs is not None)
                if exact:
                    command = [harness, '--route', candidate,
                               spec.track_path(root), spec.vehicle_path(root), ref]
                    command += ['input_file', inputs] if inputs is not None else [name]
                    run(command + [spec.sha256], f'{track_id}-{name}-oracle.log')
                    schedule['oracle_passed'] = True
                if inputs is not None:
                    with tempfile.TemporaryDirectory(dir=output) as temporary:
                        hashes = []
                        for label, route in [('original', original), ('candidate', candidate)]:
                            capture = Path(temporary) / f'{label}.bin'
                            run([harness, '--route', route, '--native-capture',
                                 spec.track_path(root), spec.vehicle_path(root),
                                 inputs, capture, spec.sha256],
                                f'{track_id}-{name}-{label}.log')
                            hashes.append(sha(capture))
                            schedule[f'{label}_bytes'] = capture.stat().st_size
                        schedule['capture_sha256'] = hashes
                        schedule['native_identical'] = hashes[0] == hashes[1]
                    if not schedule['native_identical']:
                        row['schedules'].append(schedule)
                        raise ValueError(f'{name}: exposed native tick states differ')
                row['schedules'].append(schedule)
            if any(sha(p) != row['input_sha256'][str(p)] for p in paths):
                raise ValueError('validation inputs changed during execution')
            row['passed'] = True
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            row['error'] = str(error)
        if not record.exists():
            write_json(record, row)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('track', 'passed', 'error') if k in row}), flush=True)
    report = dict(format='tmnf-route-native-validation-v1', harness_sha256=harness_hash,
                  scope='Oracle replay and exposed 1668-byte tick records; not every hidden field. Captured trigger corrections are checked structurally, not validated as live-game race crossings by this harness.',
                  generation_rows=len(generation['rows']), rows=rows,
                  passed=bool(rows) and all(row['passed'] for row in rows))
    write_json(output / 'report.json', report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
