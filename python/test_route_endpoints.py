"""Endpoint audit geometry is independent of vehicle contact semantics."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_route_endpoints import audit_route, box_distance
sys.path.pop(0)


@pytest.mark.parametrize('point,expected', [((0, 0, 0), 0), ((1, 2, 3), 0),
                                           ((4, 6, 3), 5), ((0, 0, -10), 7)])
def test_box_distance(point, expected):
    assert box_distance(point, (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0), (1, 2, 3)) == pytest.approx(expected)


def test_rotated_translated_box_with_offset_center():
    assert box_distance((107, -41, 17), (0, 0, 1, 0, 1, 0, -1, 0, 0, 100, -50, 23),
                        (2, 3, 4), (1, 2, 3)) == pytest.approx(5)


def test_dense_route_parser(tmp_path):
    import hashlib
    import struct

    data = bytearray(760)
    struct.pack_into('<8sIIIIQ', data, 0, b'TMNFROU1', 3, 0x12345678, 224, 5, len(data))
    sections = [(224, 1, 64), (288, 1, 276), (568, 0, 144), (568, 1, 144), (712, 2, 24)]
    for i, section in enumerate(sections):
        struct.pack_into('<QII', data, 96 + i * 16, *section)
    struct.pack_into('<6f', data, 568 + 16, 0, 0, 0, 1, 2, 3)
    struct.pack_into('<12f', data, 568 + 40, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    struct.pack_into('<5fI', data, 712, 0, 0, -20, 0, 8, 0)
    struct.pack_into('<5fI', data, 736, 0, 0, -10, 10, 8, 0)
    data[176:208] = hashlib.sha256(data[224:]).digest()
    path = tmp_path / 'test.tmnfroute'
    path.write_bytes(data)
    result = audit_route(path)
    assert result['endpoint_to_finish_box_m'] == 7
    assert result['endpoint_to_finish_center_horizontal_m'] == 10
    assert result['length_m'] == 10
    assert result['endpoint'] == [0, 0, -10]
