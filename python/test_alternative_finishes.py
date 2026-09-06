"""Canonical reference selection must never discard a legal finish."""
import hashlib
from pathlib import Path
import struct
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from generate_route_centerline import (route_sections, route_anchor_points,
    select_finish_from_ghost, reorder_checkpoints_from_ghost)
from refresh_route_trigger_transforms import align_selected_finish, refresh
sys.path.pop(0)

@pytest.mark.parametrize('name,count', [('A12-Speed',4), ('DesertB3',2),
    ('DesertB5',3), ('DesertD4',3), ('SnowD2',2)])
def test_reference_selection_retains_every_finish(name, count):
    data = (ROOT / 'artifacts/autoresearch/20260906-trigger-crossings/captures' /
            (name + '.tmnfroute')).read_bytes()
    sections = route_sections(data)
    offset, actual, stride = sections[3]
    assert actual == count
    original = [data[offset+i*stride:offset+(i+1)*stride] for i in range(count)]
    anchors = route_anchor_points(data, sections)
    for chosen in range(count):
        ghost = anchors[:1+sections[2][1]] + [anchors[1+sections[2][1]+chosen]]
        selected = select_finish_from_ghost(data, sections, ghost)
        new_sections = route_sections(selected)
        start, new_count, new_stride = new_sections[3]
        assert new_count == count and new_stride == stride
        records = [selected[start+i*stride:start+(i+1)*stride] for i in range(count)]
        assert records[0][4:] == original[chosen][4:]
        assert sorted(r[4:] for r in records) == sorted(r[4:] for r in original)
        assert [struct.unpack_from('<I',r)[0] for r in records] == list(range(count))
        assert selected == select_finish_from_ghost(selected, new_sections, ghost)
        assert selected[176:208] == hashlib.sha256(selected[224:]).digest()
        reordered = reorder_checkpoints_from_ghost(selected, new_sections, ghost)
        assert reordered == selected
        aligned, omitted = align_selected_finish(selected, data)
        repaired, changes = refresh(selected, aligned)
        assert omitted == [] and changes == [] and repaired == selected
