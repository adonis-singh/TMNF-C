#!/usr/bin/env python3
"""Compare completed private game captures with canonical route fixtures.

Reports existing single-finish projection separately from transform changes.
Never modifies either fixture or promotes a candidate. Incomplete captures
and mismatched metadata are failures, not coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path

from refresh_route_trigger_transforms import align_selected_finish, refresh
from tmnf_rl.tracks import project_root, track_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new report path; previous evidence must be preserved')
    root = project_root()
    rows = []
    for record in sorted(args.captures.glob('*/result.json')):
        row = dict(track=record.parent.name, passed=False)
        try:
            prior = json.loads(record.read_text())
            spec = track_spec(record.parent.name)
            old = spec.route_path(root).read_bytes()
            fresh_path = record.parent/'routes'/spec.route_path(root).name
            fresh = fresh_path.read_bytes()
            old_hash = hashlib.sha256(old).hexdigest()
            if old_hash != prior['original_sha256']:
                raise ValueError('original fixture changed since capture')
            if fresh[64:96] != bytes.fromhex(spec.sha256):
                raise ValueError('captured challenge differs from manifest')
            aligned, omitted = align_selected_finish(old, fresh)
            candidate, changes = refresh(old, aligned)
            material = [c for c in changes if max(abs(a-b) for a, b in
                        zip(c['old_transform'], c['new_transform'])) > 0.001]
            row.update(passed=True, original_sha256=old_hash,
                       captured_sha256=hashlib.sha256(fresh).hexdigest(),
                       transform_only_candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                       omitted_finish_blocks=omitted, changes=changes,
                       material_geometry_change=bool(material))
        except (ValueError, RuntimeError, KeyError, OSError) as error:
            row['error'] = repr(error)
        rows.append(row)
    report = dict(format='tmnf-captured-trigger-audit-v1',
                  captures=str(args.captures.resolve()),
                  scope='Completed captures only; not a full-campaign coverage claim. Existing selected finish preserved; omitted alternatives are not supported by this projection.',
                  passed=bool(rows) and all(r['passed'] for r in rows), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows), passed=sum(r['passed'] for r in rows),
                         material=sum(r.get('material_geometry_change', False) for r in rows),
                         selected_finish_projections=sum(bool(r.get('omitted_finish_blocks')) for r in rows))))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
