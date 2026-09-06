#!/usr/bin/env python3
"""Audit dense route endpoints against exact finish boxes, without simulation.

PYTHONPATH=python python tools/audit_route_endpoints.py --output report.json
Distances are geometric diagnostics, not a claim about car-trigger contact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from generate_route_centerline import route_sections
from tmnf_rl.tracks import project_root, track_catalogue


def box_distance(point, matrix, center, half_extent):
    """Euclidean distance to a rigidly transformed closed box (zero inside)."""
    delta = [point[i] - matrix[9 + i] for i in range(3)]
    local = [sum(matrix[j * 3 + i] * delta[j] for j in range(3)) for i in range(3)]
    outside = [max(abs(local[i] - center[i]) - half_extent[i], 0.0) for i in range(3)]
    return math.sqrt(sum(value * value for value in outside))


def audit_route(path: Path) -> dict:
    data = path.read_bytes()
    sections = route_sections(data)
    offset, count, stride = sections[4]
    if stride != 24:
        raise ValueError("endpoint audit requires a generated dense route")
    end = struct.unpack_from('<5fI', data, offset + (count - 1) * stride)
    previous = struct.unpack_from('<5fI', data, offset + (count - 2) * stride)
    finish_offset, finish_count, _ = sections[3]
    if finish_count != 1:
        raise ValueError("endpoint audit requires one selected finish")
    box = struct.unpack_from('<6f', data, finish_offset + 16)
    matrix = struct.unpack_from('<12f', data, finish_offset + 40)
    center = [sum(matrix[i * 3 + j] * box[j] for j in range(3)) + matrix[9 + i]
              for i in range(3)]
    horizontal_gap = math.hypot(end[0] - center[0], end[2] - center[2])
    return dict(route_sha256=hashlib.sha256(data).hexdigest(), points=count,
                length_m=end[3], endpoint=list(end[:3]), previous_point=list(previous[:3]),
                finish_center=center, finish_half_extent=list(box[3:]),
                endpoint_to_finish_center_horizontal_m=horizontal_gap,
                endpoint_to_finish_box_m=box_distance(end[:3], matrix, box[:3], box[3:]),
                final_half_width_m=end[4], final_leg=end[5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=project_root())
    args = parser.parse_args()
    rows, errors = [], []
    for track_id, spec in sorted(track_catalogue(args.root).items()):
        path = spec.route_path(args.root)
        try:
            rows.append(dict(track=track_id, route=str(path.relative_to(args.root)),
                             **audit_route(path)))
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(dict(track=track_id, error=str(error)))
    rows.sort(key=lambda row: -row['endpoint_to_finish_box_m'])
    report = dict(format='tmnf-route-endpoint-audit-v1', rows=rows, errors=errors,
                  scope='Geometric endpoint diagnostics; car extent and contact timing are not simulated.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(routes=len(rows), errors=len(errors),
        outside_box_over_5m=sum(row['endpoint_to_finish_box_m'] > 5 for row in rows),
        largest=[{k: row[k] for k in ('track', 'endpoint_to_finish_box_m',
                                     'endpoint_to_finish_center_horizontal_m')} for row in rows[:12]])))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
