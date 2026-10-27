#!/usr/bin/env python3
"""Build a fast replay-validation route, preserving exact race triggers/spawns.

Recorded playback uses checkpoint/finish contacts, not reward corridor geometry.
This replaces only the guidance polyline with chords between race anchors.
The file and catalogue must remain marked validation-only: these chords are
unsuitable for route-progress rewards or off-track/stuck termination.
"""
import argparse
from pathlib import Path
import struct

from generate_route_centerline import (CenterPoint, build_route, f32, f32_distance,
                                       route_anchor_points, route_sections)

# The race trigger records remain authoritative in validation mode.  A map
# can legitimately place two waypoint anchors at the same world position
# (for example, a start/finish pair on a shared block).  The native route
# loader requires both strictly increasing arc values *and* a matching
# positive geometric chord.  Race semantics remain in the copied trigger
# records, so move only the guidance point by one centimetre in this rare
# case; this keeps the serialized reference line loadable without changing
# the challenge's trigger or spawn transforms.
COINCIDENT_POSITION_EPSILON = 1.0e-2

VALIDATION_MARKER = b'TMNF-VALIDATION\0'


def build_validation_route(source):
    sections = route_sections(source)
    checkpoint_count = sections[2][1]
    anchors = route_anchor_points(source, sections)[:checkpoint_count + 2]
    points, arc = [], 0.
    for index, anchor in enumerate(anchors):
        if points:
            position = anchor
            delta = f32_distance(points[-1].position, position)
            if delta <= 0:
                previous = points[-1].position
                position = type(anchor)(
                    f32(previous.x + COINCIDENT_POSITION_EPSILON),
                    previous.y, previous.z)
                delta = f32_distance(previous, position)
            arc = f32(arc + delta)
        points.append(CenterPoint(position if points else anchor, arc, 32.,
                                  min(index, checkpoint_count)))
    result = bytearray(build_route(source, sections, points))
    result[208:224] = VALIDATION_MARKER
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.route
    data = build_validation_route(args.route.read_bytes())
    output.write_bytes(data)
    print(f'validation-only route: {output}, {len(data)} bytes; race triggers/spawns preserved')


if __name__ == '__main__':
    main()
