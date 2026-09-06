#!/usr/bin/env python3
"""Validate trigger locations and lap progression with live-game crossings.

Teleports the rigid body outside each thin trigger face and gives it a velocity
through that face. These synthetic probes validate trigger positions and event
rules; they are neither policy performance nor full-lap timing evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

from checkpoint_order_probe import DYNA_FIELDS, drive, teleport
from determinism_test import TICK_MS, map_command, wait_for_race_start
from tmi_client import MessageType, TMInterface

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from generate_route_centerline import route_sections


def crossing(record: bytes, direction: int = 1) -> tuple[np.ndarray, np.ndarray]:
    box = np.array(struct.unpack_from('<6f', record, 16), dtype=np.float32)
    transform = np.array(struct.unpack_from('<12f', record, 40), dtype=np.float32)
    axis = int(np.argmin(box[3:]))
    local = box[:3].copy()
    local[axis] -= direction*(box[3+axis]+6.0)
    rotation = transform[:9].reshape(3, 3)
    position = rotation@local+transform[9:]
    velocity = rotation[:, axis]*(40.0*direction)
    return position, velocity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--route', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reverse-blocks', type=int, nargs='*', default=[],
                        help='approach these block IDs from the opposite face when scenery obstructs the default approach')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output path')
    data = args.route.read_bytes()
    sections = route_sections(data)
    laps = struct.unpack_from('<I', data, sections[0][0])[0]
    if laps < 1 or sections[3][1] != 1:
        parser.error('this directed probe requires a selected-finish route with positive lap count')
    def records(section):
        offset, count, stride = sections[section]
        return [data[offset+i*stride:offset+(i+1)*stride] for i in range(count)]
    checkpoints, finish = records(2), records(3)[0]
    per_lap = len(checkpoints)+1
    target = per_lap*laps
    if struct.unpack_from('<I', data, sections[0][0]+16)[0] != target:
        parser.error('route total checkpoint count is inconsistent')
    phases = []
    for lap in range(laps):
        prefix = f'lap_{lap+1}_' if laps > 1 else ''
        if checkpoints:
            phases.append((prefix+'finish_before_checkpoints', finish, None))
        for i, record in enumerate(checkpoints):
            phases.append((prefix+f'checkpoint_{i}', record, (lap*per_lap+i+1, target)))
            phases.append((prefix+f'checkpoint_{i}_repeat', record, None))
        phases.append((prefix+'finish_after_checkpoints', finish, ((lap+1)*per_lap, target)))
    report = dict(map=args.map, lap_count=laps, route_sha256=hashlib.sha256(data).hexdigest(),
                  scope='Synthetic live-game trigger crossings; not policy or lap-timing evidence.',
                  phases=[], passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    interface = TMInterface(args.port, timeout=60)
    try:
        connect = interface.read_message_type()
        if connect != MessageType.SC_ON_CONNECT_SYNC:
            raise RuntimeError(f'unexpected connect message {connect}')
        interface.set_timeout(300_000)
        interface.set_on_step_period(TICK_MS)
        interface.set_speed(1.0)
        interface.execute_command('set countdown_speed 5')
        interface.execute_command('set autorewind false')
        interface.execute_command('set use_valseed false')
        interface.execute_command(map_command(args.map))
        interface.respond(connect)
        blocked, state = wait_for_race_start(interface)
        template = {name: np.asarray(getattr(state.dyna.current_state, name).to_numpy(), dtype=np.float32)
                    for name, _ in DYNA_FIELDS}
        for name, record, expected in phases:
            block_index = struct.unpack_from('<I', record, 4)[0]
            direction = -1 if block_index in args.reverse_blocks else 1
            position, velocity = crossing(record, direction)
            fields = {name: value.copy() for name, value in template.items()}
            fields['position'] = position
            fields['linear_speed'] = velocity
            fields['not_tweaked_linear_speed'] = velocity
            for field in ('add_linear_speed', 'angular_speed', 'force', 'torque'):
                fields[field] = np.zeros(3, dtype=np.float32)
            teleport(interface, fields)
            events, trajectory = [], []
            for _ in range(50):
                blocked, tick_events = drive(interface, blocked, 1)
                events.extend(tick_events)
                current = interface.get_simulation_state().dyna.current_state
                trajectory.append(dict(position=current.position.to_numpy().astype(float).tolist(),
                                       velocity=current.linear_speed.to_numpy().astype(float).tolist()))
            observed = [(e['current'], e['target']) for e in events
                        if e['type'] == 'SC_CHECKPOINT_COUNT_CHANGED_SYNC']
            passed = observed == ([expected] if expected else [])
            if expected == (target, target):
                passed = passed and any(e['race_finished'] for e in events)
            else:
                passed = passed and not any(e['race_finished'] for e in events)
            row = dict(phase=name, block_index=block_index, direction=direction,
                       position=position.tolist(), velocity=velocity.tolist(),
                       expected=list(expected) if expected else None,
                       events=events, trajectory=trajectory, passed=passed)
            report['phases'].append(row)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'trajectory'}), flush=True)
        report['passed'] = all(row['passed'] for row in report['phases'])
        interface.set_input_state(left=False, right=False, accelerate=False, brake=False)
        interface.respond(blocked)
    finally:
        interface.close()
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
