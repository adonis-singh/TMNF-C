from pathlib import Path
import hashlib
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from generate_route_centerline import route_sections
from refresh_route_trigger_transforms import align_selected_finish, refresh
sys.path.pop(0)

FIXTURES = ROOT / 'artifacts/autoresearch/20260906-trigger-correction'


def fixtures():
    return ((FIXTURES / 'RallyA1-original.tmnfroute').read_bytes(),
            (FIXTURES / 'RallyA1-fresh.tmnfroute').read_bytes())


def test_stale_rally_transforms_use_block_identity_and_keep_reference():
    old, captured = fixtures()
    result, changes = refresh(old, captured)
    assert [(c['section'], c['index'], c['block_index']) for c in changes] == [(2, 1, 249), (3, 0, 278)]
    assert changes[-1]['new_transform'][9:] == [128, 64, 576]
    sections = route_sections(result)
    start, count, stride = sections[4]
    assert result[start:start+count*stride] == old[start:start+count*stride]
    for section in (2, 3):
        start, count, stride = sections[section]
        for i in range(count):
            at = start+i*stride
            assert result[at:at+4] == old[at:at+4]
    assert refresh(result, captured) == (result, [])


@pytest.mark.parametrize('kind,message', [('track', 'challenge hash'),
    ('metadata', 'race metadata'), ('start', 'start state'),
    ('spawn', 'outside its transform'), ('duplicate', 'duplicate'),
    ('nan', 'nonfinite')])
def test_reject_unrelated_or_invalid_capture_changes(kind, message):
    old, capture = fixtures(); capture = bytearray(capture)
    sections = route_sections(capture)
    if kind == 'track':
        capture[64] ^= 1
    elif kind == 'metadata':
        capture[sections[0][0]] ^= 1
    elif kind == 'start':
        capture[sections[1][0]] ^= 1
    elif kind == 'spawn':
        capture[sections[3][0]+88] ^= 1
    elif kind == 'duplicate':
        start, _, stride = sections[2]
        capture[start+stride+4:start+stride+12] = capture[start+4:start+12]
    elif kind == 'nan':
        struct.pack_into('<f', capture, sections[3][0]+40, float('nan'))
    capture[176:208] = hashlib.sha256(capture[224:]).digest()
    with pytest.raises(ValueError, match=message):
        refresh(old, bytes(capture))


@pytest.mark.parametrize('name,count', [('DesertB3', 2), ('DesertB5', 3)])
def test_existing_finish_choice_is_preserved_and_alternatives_reported(name, count):
    old = (FIXTURES / f'{name}-original.tmnfroute').read_bytes()
    captured = (FIXTURES / f'{name}-fresh.tmnfroute').read_bytes()
    with pytest.raises(ValueError, match='metadata'):
        refresh(old, captured)
    aligned, omitted = align_selected_finish(old, captured)
    assert len(omitted) == count - 1
    result, _ = refresh(old, aligned)
    a, _, _ = route_sections(old)[3]
    b, n, _ = route_sections(result)[3]
    assert n == 1
    assert old[a:a+12] == result[b:b+12]
    assert struct.unpack_from('<I', old, a+4)[0] not in omitted


def test_selected_finish_must_exist_in_capture():
    old, captured = fixtures()
    old = bytearray(old)
    a, _, _ = route_sections(old)[3]
    struct.pack_into('<I', old, a+4, 0xFFFFFFFE)
    old[176:208] = hashlib.sha256(old[224:]).digest()
    with pytest.raises(ValueError, match='absent'):
        align_selected_finish(bytes(old), captured)
