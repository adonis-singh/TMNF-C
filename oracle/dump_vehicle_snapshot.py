#!/usr/bin/env python3
"""Dump the complete live TMNF vehicle graph into snapshot version 4.

Object identities come from `--input`, the environment's base snapshot, so
every snapshot of one environment shares ids for the same live objects. The
first snapshot of an environment has no base: without `--input` the ids of the
phase capture are kept and the remaining objects get fresh ids.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from track_env import game_layout  # noqa: E402

from determinism_test import (
    DEFAULT_MAP,
    TICK_MS,
    f32,
    map_command,
    read_next_run_step,
    scheduled_input,
    wait_for_race_start,
)
from tmi_client import MessageType, TMInterface
from tminterface.structs import SimStateData


ROOT = Path(__file__).resolve().parents[1]
PREFIX = game_layout(
    Path(os.environ.get("TMNF_WINEPREFIX", ROOT / "oracle" / "wineprefix"))
).prefix
READER_SOURCE = ROOT / "oracle" / "vehicle_memory_reader.c"
READER_EXE = ROOT / "build" / PREFIX.name / "vehicle_memory_reader.exe"

V3_MAGIC = b"TMNFM6G1"
V3_VERSION = 4
MATERIAL_SIZE = 40
# The one bump mask every material with a mask references. Its pixels are
# loaded lazily by 0x00845840 on the first bump contact, so at capture time
# the image (CPlugBitmap +0x48) has data +0x28 null; the file identity is
# checked instead. Every collection pak ships the byte-identical
# Vehicles/Media/Texture/Image/TestMaterialHeight.tga (128x128 24-bit, SHA-256
# 5e7e018c624367789c543c663d85b8d0e1827a6b2be17664c288059cbd80cfd8) whose
# first channel is src/vehicle_fake_contact_mask.h.
FAKE_CONTACT_MASK_FILE = "TestMaterialHeight.Texture.gbx"
V3_HEADER_FORMAT = "<8s" + "I" * 42 + "ffffi12s12s12s16sif32s32s"
V3_HEADER_SIZE = struct.calcsize(V3_HEADER_FORMAT)
PHASE_VERSION = 1
PHASE_HEADER_FORMAT = "<8s" + "I" * 29 + "ffffi12s12s12s16sif32s32s"
PHASE_HEADER_SIZE = struct.calcsize(PHASE_HEADER_FORMAT)
# CSceneVehicleMaterialManager entries: 13 in Stadium, per collection elsewhere
# (src/surface_material.h TMNF_MAX_GROUND_MATERIALS).
MAX_GROUND_MATERIALS = 32
COLLISION_TREE_FORMAT = "<IIIIIIIIHBB24s48s24s"
COLLISION_TREE_SIZE = struct.calcsize(COLLISION_TREE_FORMAT)

COLLISION_TREE_KIND_BODY = 0
COLLISION_TREE_KIND_WHEEL = 1
COLLISION_TREE_KIND_ROOT = 2
# The kind field holds the kind in its low byte and the node's direct child
# count in bits 8..15; children follow their parent in pre-order. Stadium cars
# are eight leaves under the root; United cars nest body shapes (a body node
# may carry a surface and children at once) and have sphere wheels.
COLLISION_TREE_KIND_SHIFT = 8
NO_WHEEL_INDEX = 0xFFFFFFFF
GM_SURF_SPHERE = 0
GM_SURF_ELLIPSOID = 1
COLLISION_TREE_MAX_NODES = 16
COLLISION_FLAG_ACTIVE = 0x80
IDENTITY_ISO = struct.pack("<12f", 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)

CAR_SIZE = 0x878
VEHICLE_STRUCT_SIZE = 0x50
TUNING_SIZE = 0x3AC
WHEEL_SIZE = 0x2FC
DYNA_SIZE = 0x344
PARAMS_SIZE = 0x5C
STATE_SIZE = 0xB4
MODEL_ISO_SIZE = 0x30
BODY_REFERENCE_SIZE = 0x0C

VEHICLE_STRUCT_VTABLE = 0x00BA3884
VEHICLE_CAR_VTABLE = 0x00B9EFFC
TUNING_VTABLE = 0x00BA376C
TUNINGS_VTABLE = 0x00BA3924
TUNING_CLASS_ID = 0x0A02E000

OWNER_VEHICLE_STRUCT = 0
OWNER_AUX_TUNING = 1

PRIMARY_CURVE_OFFSETS = (0x44, 0x48)
AUX_CURVE_OFFSETS = (
    0x034,
    0x068,
    0x078,
    0x0A0,
    0x0AC,
    0x0B8,
    0x0BC,
    0x1B4,
    0x1BC,
    0x1E0,
    0x218,
    0x224,
    0x230,
    0x250,
    0x25C,
    0x260,
    0x288,
    0x2A4,
    0x36C,
    0x378,
    0x380,
)
MANDATORY_ACTIVE_CURVES = (
    0x034,
    0x068,
    0x078,
    0x0A0,
    0x0AC,
    0x1BC,
    0x1E0,
    0x224,
    0x230,
    0x250,
    0x25C,
    0x260,
    0x288,
    0x2A4,
    0x36C,
    0x380,
)
# Tuning +0x378 (steering angle from speed) is nullable in the game: Stadium
# sets it, Desert leaves it null and the car steers a fixed 30 degrees.
GEARBOX_OFFSETS = (0x2C4, 0x2D4, 0x2E0, 0x304)
GEARBOX_VALUE_COUNT = 6
TUNING_POINTER_OFFSETS = frozenset(
    AUX_CURVE_OFFSETS
    + tuple(offset + 4 for offset in GEARBOX_OFFSETS)
    + (0x024, 0x1C4, 0x1EC, 0x1F0, 0x210, 0x214, 0x2FC, 0x314)
)
VEHICLE_STRUCT_POINTER_OFFSETS = frozenset(
    (0x08, 0x18, 0x24, 0x30, 0x3C, 0x44, 0x48, 0x4C)
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def section(data: bytes, offset: int, size: int, name: str) -> bytes:
    require(offset <= len(data) and size <= len(data) - offset, f"{name} is out of bounds")
    return data[offset : offset + size]


@dataclass(frozen=True)
class BaseSnapshot:
    data: bytes
    header: tuple[object, ...]

    @classmethod
    def load(cls, path: Path) -> "BaseSnapshot":
        data = path.read_bytes()
        require(len(data) >= V3_HEADER_SIZE, "version-3 snapshot header is truncated")
        header = struct.unpack_from(V3_HEADER_FORMAT, data)
        require(header[0] == V3_MAGIC, "version-3 snapshot magic mismatch")
        require(header[1] == V3_VERSION, "input snapshot is not version 4")
        require(header[2] == len(data), "version-3 total size mismatch")
        require(header[3] == V3_HEADER_SIZE, "version-3 header size mismatch")
        require(header[4] == 0, "version-3 snapshot is not an input graph")
        require(header[17] == 4, "version-3 snapshot does not have four wheels")
        require(header[18] == 31, "version-3 snapshot does not have 31 ground IDs")
        require(1 <= header[19] <= MAX_GROUND_MATERIALS,
                "version-3 snapshot ground material count is out of range")
        require(header[20] != 0, "version-3 snapshot has no tuning objects")
        require(header[21] < header[20], "version-3 active tuning key is out of range")
        require(1 <= header[39] <= COLLISION_TREE_MAX_NODES,
                "version-3 snapshot collision node count is out of range")
        return cls(data, header)

    def integer(self, index: int) -> int:
        value = self.header[index]
        require(isinstance(value, int), f"header field {index} is not an integer")
        return value

    @property
    def root_ids(self) -> tuple[int, ...]:
        collision_ids = []
        for tree in (self.collision_root, *self.collision_children):
            collision_ids.extend(value for value in tree[:4] if value != 0)
        return (
            tuple(self.integer(index) for index in range(6, 17))
            + (self.tuning_id,)
            + tuple(collision_ids)
        )

    @property
    def car_id(self) -> int:
        return self.integer(6)

    @property
    def tuning_id(self) -> int:
        descriptor_offset = self.integer(26) + self.integer(21) * 16
        descriptor = section(self.data, descriptor_offset, 16, "active tuning descriptor")
        key, object_id, _, raw_size = struct.unpack("<IIII", descriptor)
        require(key == self.integer(21), "active tuning descriptor key mismatch")
        require(object_id != 0, "active tuning descriptor has null identity")
        require(raw_size == TUNING_SIZE, "active tuning descriptor size mismatch")
        return object_id

    @property
    def car(self) -> bytes:
        return section(self.data, self.integer(24), CAR_SIZE, "version-3 car")

    @property
    def vehicle_struct(self) -> bytes:
        return section(
            self.data,
            self.integer(25),
            VEHICLE_STRUCT_SIZE,
            "version-3 vehicle struct",
        )

    @property
    def tuning(self) -> bytes:
        descriptor_offset = self.integer(26) + self.integer(21) * 16
        descriptor = section(self.data, descriptor_offset, 16, "active tuning descriptor")
        _, _, raw_offset, raw_size = struct.unpack("<IIII", descriptor)
        require(raw_size == TUNING_SIZE, "active tuning raw size mismatch")
        return section(self.data, raw_offset, TUNING_SIZE, "version-3 active tuning")

    @property
    def wheels(self) -> bytes:
        return section(self.data, self.integer(27), 4 * WHEEL_SIZE, "version-3 wheels")

    @property
    def dyna(self) -> bytes:
        return section(self.data, self.integer(28), DYNA_SIZE, "version-3 dynamics")

    @property
    def params(self) -> bytes:
        return section(self.data, self.integer(29), PARAMS_SIZE, "version-3 dynamics params")

    @property
    def state(self) -> bytes:
        return section(self.data, self.integer(30), STATE_SIZE, "version-3 dynamics state")

    @property
    def ground_ids(self) -> bytes:
        return section(
            self.data,
            self.integer(33),
            self.integer(18) * 4,
            "version-3 ground IDs",
        )

    @property
    def ground_materials(self) -> bytes:
        return section(
            self.data,
            self.integer(34),
            self.integer(19) * MATERIAL_SIZE,
            "version-3 ground materials",
        )

    @property
    def model_iso(self) -> bytes:
        return section(self.data, self.integer(35), MODEL_ISO_SIZE, "version-3 model ISO")

    @property
    def body_reference(self) -> bytes:
        return section(
            self.data,
            self.integer(36),
            BODY_REFERENCE_SIZE,
            "version-3 body reference",
        )

    @property
    def collision_root(self) -> tuple[int | bytes, ...]:
        raw = section(
            self.data,
            self.integer(40),
            COLLISION_TREE_SIZE,
            "version-3 collision root",
        )
        return struct.unpack(COLLISION_TREE_FORMAT, raw)

    @property
    def collision_children(self) -> tuple[tuple[int | bytes, ...], ...]:
        raw = section(
            self.data,
            self.integer(41),
            self.integer(39) * COLLISION_TREE_SIZE,
            "version-3 collision children",
        )
        return tuple(
            struct.unpack_from(COLLISION_TREE_FORMAT, raw, index * COLLISION_TREE_SIZE)
            for index in range(self.integer(39))
        )

    @property
    def tick(self) -> int:
        return self.integer(5)

    @property
    def arguments(self) -> tuple[object, ...]:
        return struct.unpack_from("<ffffi12s12s12s16sif", self.data, 176)

    @property
    def source_exe_sha256(self) -> bytes:
        return self.data[256:288]

    @property
    def source_track_sha256(self) -> bytes:
        return self.data[288:320]


@dataclass(frozen=True)
class PhaseSnapshot:
    data: bytes
    header: tuple[object, ...]

    @classmethod
    def load(cls, path: Path) -> "PhaseSnapshot":
        data = path.read_bytes()
        require(len(data) >= PHASE_HEADER_SIZE, "phase snapshot header is truncated")
        header = struct.unpack_from(PHASE_HEADER_FORMAT, data)
        require(header[0] == V3_MAGIC, "phase snapshot magic mismatch")
        require(header[1] == PHASE_VERSION, "phase snapshot is not version 1")
        require(header[2] == len(data), "phase snapshot total size mismatch")
        require(header[3] == 0, "phase snapshot is not an input graph")
        require(header[14] == 4, "phase snapshot does not have four wheels")
        require(header[15] == 31, "phase snapshot does not have 31 ground IDs")
        require(1 <= header[16] <= MAX_GROUND_MATERIALS,
                "phase snapshot ground material count is out of range")
        return cls(data, header)

    def integer(self, index: int) -> int:
        value = self.header[index]
        require(isinstance(value, int), f"phase header field {index} is not an integer")
        return value

    def raw(self, index: int, size: int, name: str) -> bytes:
        return section(self.data, self.integer(index), size, name)

    @property
    def car(self) -> bytes:
        return self.raw(17, CAR_SIZE, "phase car")

    @property
    def wheels(self) -> bytes:
        return self.raw(19, 4 * WHEEL_SIZE, "phase wheels")

    @property
    def dyna(self) -> bytes:
        return self.raw(20, DYNA_SIZE, "phase dynamics")

    @property
    def params(self) -> bytes:
        return self.raw(21, PARAMS_SIZE, "phase dynamics params")

    @property
    def state(self) -> bytes:
        return self.raw(22, STATE_SIZE, "phase dynamics state")

    @property
    def ground_ids(self) -> bytes:
        return self.raw(24, 31 * 4, "phase ground IDs")

    @property
    def ground_materials(self) -> bytes:
        return self.raw(25, self.integer(16) * 20, "phase ground materials")

    @property
    def model_iso(self) -> bytes:
        return self.raw(26, MODEL_ISO_SIZE, "phase model ISO")

    @property
    def body_reference(self) -> bytes:
        return self.raw(27, BODY_REFERENCE_SIZE, "phase body reference")

    @property
    def tick(self) -> int:
        return self.integer(29)

    @property
    def arguments(self) -> tuple[object, ...]:
        return struct.unpack_from("<ffffi12s12s12s16sif", self.data, 124)

    @property
    def source_exe_sha256(self) -> bytes:
        return self.data[204:236]

    @property
    def source_track_sha256(self) -> bytes:
        return self.data[236:268]


class MemoryReader:
    def __init__(self) -> None:
        READER_EXE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            (
                "i686-w64-mingw32-gcc",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(READER_SOURCE),
                "-o",
                str(READER_EXE),
            ),
            check=True,
        )
        environment = os.environ | {
            "WINEPREFIX": str(PREFIX),
            "WINEDEBUG": "-all",
        }
        self.process = subprocess.Popen(
            ("wine", str(READER_EXE)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        ready = self._readline()
        require(ready.startswith("READY "), f"unexpected memory reader greeting: {ready}")

    def _readline(self) -> str:
        stdout = self.process.stdout
        require(stdout is not None, "memory reader stdout is unavailable")
        line = stdout.readline()
        if not line:
            stderr = self.process.stderr
            details = b"" if stderr is None else stderr.read()
            raise RuntimeError(
                f"memory reader stopped with code {self.process.poll()}: "
                f"{details.decode(errors='replace').strip()}"
            )
        return line.decode("ascii").rstrip("\n")

    def _write(self, command: str) -> None:
        stdin = self.process.stdin
        require(stdin is not None, "memory reader stdin is unavailable")
        stdin.write(command.encode("ascii"))
        stdin.flush()

    def read(self, address: int, size: int) -> bytes:
        require(0 < address <= 0xFFFFFFFF, f"invalid live address 0x{address:x}")
        require(0 < size <= 0xFFFFFFFF, f"invalid live read size {size}")
        self._write(f"READ {address:08x} {size}\n")
        response = self._readline()
        require(response.startswith("DATA "), f"unexpected read response: {response}")
        data = bytes.fromhex(response[5:])
        require(len(data) == size, f"live read returned {len(data)} of {size} bytes")
        return data

    def read_u32(self, address: int) -> int:
        return u32(self.read(address, 4), 0)

    def scan(self, pattern: bytes) -> tuple[int, ...]:
        require(pattern, "cannot scan for an empty pattern")
        self._write(f"SCAN {pattern.hex()}\n")
        response = self._readline().split()
        require(response and response[0] == "MATCHES", "unexpected scan response")
        return tuple(int(value, 16) for value in response[1:])

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self._write("QUIT\n")
        return_code = self.process.wait(timeout=10)
        require(return_code == 0, f"memory reader exited with code {return_code}")


class IdMap:
    def __init__(self, reserved: tuple[int, ...]) -> None:
        self._ids: dict[int, int] = {}
        self._used = set(reserved)
        self._next = max(self._used, default=0) + 1

    def bind(self, address: int, object_id: int) -> None:
        require(address != 0 and object_id != 0, "cannot bind a null graph identity")
        # United cars share one geometry between several collision tree nodes
        # (the four wheel spheres); binding the same pair again is a no-op.
        require(
            self._ids.get(address, object_id) == object_id,
            f"live address 0x{address:08x} is already bound to {self._ids.get(address)}",
        )
        self._ids[address] = object_id
        self._used.add(object_id)

    def get(self, address: int) -> int:
        require(address != 0, "cannot map a null graph identity")
        if address in self._ids:
            return self._ids[address]
        while self._next in self._used:
            self._next += 1
        result = self._next
        self._used.add(result)
        self._ids[address] = result
        self._next += 1
        return result


@dataclass(frozen=True)
class GraphIdentity:
    """Object ids the new snapshot keeps: car, active tuning, the eight fixed
    graph objects (header fields 9..16) and the three counts (17..19)."""

    car_id: int
    tuning_id: int
    fixed: tuple[int, ...]
    reserved: tuple[int, ...]

    @classmethod
    def from_base(cls, base: BaseSnapshot) -> "GraphIdentity":
        return cls(
            base.car_id,
            base.tuning_id,
            tuple(base.integer(index) for index in range(9, 20)),
            base.root_ids,
        )

    @classmethod
    def from_phase(cls, phase: PhaseSnapshot) -> "GraphIdentity":
        # Phase header: 4 car, 5 tuning, 6 wheels, 7 item, 8 corpus, 9 dyna,
        # 10 params, 11 state, 12 model iso, 13 body reference, 14..16 counts.
        return cls(
            phase.integer(4),
            phase.integer(5),
            tuple(phase.integer(index) for index in range(6, 17)),
            tuple(phase.integer(index) for index in range(4, 14)),
        )


@dataclass(frozen=True)
class LiveGraph:
    car_address: int
    car: bytes
    vehicle_struct_address: int
    vehicle_struct: bytes
    tuning_container_address: int
    tuning_key: int
    tuning_addresses: tuple[int, ...]
    tunings: tuple[bytes, ...]


@dataclass(frozen=True)
class CapturedCollisionTree:
    address: int
    flags: int
    surface_address: int
    geometry_address: int
    kind: int
    wheel_index: int
    geometry_material_index: int
    geometry_type: int
    geometry_reserved: int
    box: bytes
    local_iso: bytes
    shape: bytes
    material_ids: tuple[int, ...]
    child_count: int = 0


@dataclass(frozen=True)
class CapturedCollisionGraph:
    root: CapturedCollisionTree
    children: tuple[CapturedCollisionTree, ...]


@dataclass(frozen=True)
class CapturedCurve:
    owner_kind: int
    owner_key: int
    field_offset: int
    object_address: int
    positions_address: int
    values_address: int
    count: int
    interpolation: int
    positions: bytes
    values: bytes


@dataclass(frozen=True)
class CapturedGearbox:
    owner_key: int
    field_offset: int
    buffer_address: int
    data_address: int
    values: bytes


def locate_live_graph(
    reader: MemoryReader,
    reference_state: SimStateData,
) -> LiveGraph:
    """The player's car is the CSceneVehicleCar whose dynamics state equals
    the TMInterface state; its tuning container is car +0x64."""
    containers: dict[int, tuple[int, tuple[int, ...], tuple[bytes, ...]]] = {}
    for container_address in reader.scan(struct.pack("<I", TUNINGS_VTABLE)):
        container = reader.read(container_address, 0x28)
        count = u32(container, 0x14)
        data_address = u32(container, 0x18)
        key = u32(container, 0x24)
        if u32(container, 0x20) != TUNING_CLASS_ID or count == 0 or count > 64 or key >= count:
            continue
        pointer_data = reader.read(data_address, count * 4)
        addresses = struct.unpack(f"<{count}I", pointer_data)
        tunings = tuple(reader.read(address, TUNING_SIZE) for address in addresses)
        if any(u32(tuning, 0) != TUNING_VTABLE for tuning in tunings):
            continue
        containers[container_address] = (key, addresses, tunings)
    require(containers, "found no CSceneVehicleTunings container")

    car_candidates: list[tuple[int, bytes, int, bytes]] = []
    state = reference_state.dyna.current_state
    expected_dyna_prefix = b"".join(
        (
            f32(state.quat),
            f32(state.rotation),
            f32(state.position),
            f32(state.linear_speed),
            f32(state.add_linear_speed),
            f32(state.angular_speed),
            f32(state.force),
            f32(state.torque),
            f32(state.inverse_inertia_tensor),
        )
    )
    expected_not_tweaked = f32(state.not_tweaked_linear_speed)
    require(len(expected_dyna_prefix) == 0xA0, "TMInterface dynamics prefix size changed")
    require(len(expected_not_tweaked) == 0x0C, "TMInterface untweaked speed size changed")
    pointer_addresses = [
        pointer_address
        for container_address in containers
        for pointer_address in reader.scan(struct.pack("<I", container_address))
        if pointer_address >= 0x64
    ]
    for pointer_address in pointer_addresses:
        car_address = pointer_address - 0x64
        car = reader.read(car_address, CAR_SIZE)
        if (
            u32(car, 0) != VEHICLE_CAR_VTABLE
            or u32(car, 0x64) not in containers
            or u32(car, 0x2E8) != 4
        ):
            continue
        item_address = u32(car, 0x28)
        if item_address == 0:
            continue
        item = reader.read(item_address, 0x3C)
        if u32(item, 0x34) == 0:
            continue
        corpus_data_address = u32(item, 0x38)
        corpus_address = reader.read_u32(corpus_data_address)
        dyna_address = reader.read_u32(corpus_address + 0x58)
        live_state_address = reader.read_u32(dyna_address + 0x32C)
        live_state = reader.read(live_state_address, STATE_SIZE)
        if live_state[:0xA0] != expected_dyna_prefix:
            continue
        if live_state[0xA4:0xB0] != expected_not_tweaked:
            continue
        vehicle_struct_address = u32(car, 0x60)
        if vehicle_struct_address == 0:
            continue
        vehicle_struct = reader.read(vehicle_struct_address, VEHICLE_STRUCT_SIZE)
        if u32(vehicle_struct, 0) != VEHICLE_STRUCT_VTABLE:
            continue
        car_candidates.append((car_address, car, vehicle_struct_address, vehicle_struct))
    require(car_candidates, "found no vehicle object matching the TMInterface player state")
    vehicle_struct_addresses = {candidate[2] for candidate in car_candidates}
    require(
        len(vehicle_struct_addresses) == 1,
        "matching vehicle objects reference different CSceneVehicleStruct objects: "
        + ", ".join(f"0x{candidate[0]:08x}" for candidate in car_candidates),
    )
    car_address, car, vehicle_struct_address, vehicle_struct = car_candidates[0]
    container_address = u32(car, 0x64)
    key, tuning_addresses, tunings = containers[container_address]

    return LiveGraph(
        car_address=car_address,
        car=car,
        vehicle_struct_address=vehicle_struct_address,
        vehicle_struct=vehicle_struct,
        tuning_container_address=container_address,
        tuning_key=key,
        tuning_addresses=tuning_addresses,
        tunings=tunings,
    )


def tree_children(reader: MemoryReader, address: int) -> tuple[int, ...]:
    raw = reader.read(address, 0xAC)
    count = u32(raw, 0x28)
    data_address = u32(raw, 0x2C)
    capacity = u32(raw, 0x30)
    require(count <= capacity and capacity <= 4096, "invalid collision-tree child buffer")
    require(count == 0 or data_address != 0, "collision tree has null child storage")
    if count == 0:
        return ()
    pointers = reader.read(data_address, count * 4)
    return struct.unpack(f"<{count}I", pointers)


def capture_collision_leaf(
    reader: MemoryReader,
    address: int,
    kind: int,
    wheel_index: int,
) -> CapturedCollisionTree:
    raw = reader.read(address, 0xAC)
    surface_address = u32(raw, 0x8C)
    require(surface_address != 0, "vehicle collision leaf has no surface")
    surface = reader.read(surface_address, 0x24)
    wrapper_address = u32(surface, 0x14)
    require(wrapper_address != 0, "vehicle collision surface has no geometry wrapper")
    wrapper = reader.read(wrapper_address, 0x3C)
    geometry_address = u32(wrapper, 0x34)
    require(geometry_address != 0, "vehicle collision surface has no geometry")
    geometry_type = reader.read(geometry_address, 8)[6]
    require(
        geometry_type in (GM_SURF_SPHERE, GM_SURF_ELLIPSOID),
        f"vehicle collision geometry type {geometry_type} is not a sphere or ellipsoid",
    )
    # GmSurfSphere is 0x0C bytes (radius at +8), GmSurfEllipsoid 0x14 (radii at +8).
    geometry = reader.read(geometry_address, 0x0C if geometry_type == GM_SURF_SPHERE else 0x14)

    material_count = u32(surface, 0x18)
    material_data_address = u32(surface, 0x1C)
    material_capacity = u32(surface, 0x20)
    require(
        material_count != 0
        and material_count <= material_capacity
        and material_capacity <= 256
        and material_data_address != 0,
        "invalid vehicle collision material buffer",
    )
    material_pointers = reader.read(material_data_address, material_count * 4)
    material_ids = []
    for index in range(material_count):
        material_address = u32(material_pointers, index * 4)
        require(material_address != 0, "vehicle collision material is null")
        material_ids.append(reader.read(material_address + 0x18, 1)[0])

    # One material per shape. Stadium, Desert, Rally and Snow bodies are
    # Concrete (0), the Island body is Metal (4); wheels are Rubber (9).
    require(
        len(material_ids) == 1 and material_ids[0] < 31,
        f"unexpected vehicle collision material IDs {material_ids}",
    )
    return CapturedCollisionTree(
        address=address,
        flags=u32(raw, 0x9C),
        surface_address=surface_address,
        geometry_address=geometry_address,
        kind=kind,
        wheel_index=wheel_index,
        geometry_material_index=struct.unpack_from("<H", geometry, 4)[0],
        geometry_type=geometry_type,
        # GmSurf+7 is struct padding the game never writes; the United heap
        # leaves 0, 64 or 100 there between runs (Stadium always saw 0).
        geometry_reserved=0,
        box=raw[0x34:0x4C],
        local_iso=raw[0x5C:0x8C],
        shape=(geometry[8:] + bytes(24))[:24],
        material_ids=tuple(material_ids),
    )


def collision_role(tree: CapturedCollisionTree) -> str:
    if tree.surface_address == 0:
        shape = "no surface"
    else:
        shape = "sphere" if tree.geometry_type == GM_SURF_SPHERE else "ellipsoid"
    nested = f", {tree.child_count} children" if tree.child_count else ""
    if tree.kind == COLLISION_TREE_KIND_BODY:
        return f"body {shape}{nested}"
    return f"wheel {tree.wheel_index} {shape}{nested}"


def capture_collision_graph(
    reader: MemoryReader,
    graph: LiveGraph,
) -> CapturedCollisionGraph:
    item_address = u32(graph.car, 0x28)
    require(item_address != 0, "vehicle has no CHmsItem")
    item = reader.read(item_address, 0x3C)
    corpus_count = u32(item, 0x34)
    corpus_data_address = u32(item, 0x38)
    require(corpus_count == 1 and corpus_data_address != 0, "vehicle item corpus graph changed")
    corpus_address = reader.read_u32(corpus_data_address)
    scene_address = reader.read_u32(corpus_address + 0x48)
    require(scene_address != 0, "vehicle corpus has no scene object")
    model_address = reader.read_u32(scene_address + 0x14)
    require(model_address != 0, "vehicle scene has no model")
    model_root_address = reader.read_u32(model_address + 0x64)
    require(model_root_address != 0, "vehicle model has no collision tree")

    wheel_count = u32(graph.car, 0x2E8)
    wheel_data_address = u32(graph.car, 0x2EC)
    require(wheel_count == 4 and wheel_data_address != 0, "vehicle wheel graph changed")
    wheel_tree_addresses = tuple(
        reader.read_u32(wheel_data_address + index * WHEEL_SIZE + 0x0C)
        for index in range(wheel_count)
    )
    require(
        len(set(wheel_tree_addresses)) == wheel_count
        and all(wheel_tree_addresses),
        "vehicle wheel collision trees are invalid",
    )

    # The recorded root is the lowest common ancestor of every active leaf
    # (flag 0x80 on the leaf and every ancestor). Nodes above it must be
    # transparent to the game's traversal: active, surface-less, identity
    # local iso, so that starting at the recorded root is exact. Nodes without
    # the active flag are never visited by the collision traversal and are
    # not recorded.
    def node_raw(address: int) -> bytes:
        return reader.read(address, 0xAC)

    def active(raw: bytes) -> bool:
        return (u32(raw, 0x9C) & COLLISION_FLAG_ACTIVE) != 0

    def active_children(address: int) -> tuple[int, ...]:
        return tuple(
            child for child in tree_children(reader, address)
            if active(node_raw(child))
        )

    visited: set[int] = set()

    def collision_leaf_count(address: int, depth: int = 0) -> int:
        require(depth <= 64, "vehicle collision tree is too deep")
        children = active_children(address)
        if not children:
            return 1 if u32(node_raw(address), 0x8C) != 0 else 0
        return sum(collision_leaf_count(child, depth + 1) for child in children)

    root_address = model_root_address
    require(active(node_raw(root_address)), "vehicle model collision tree is inactive")
    total_leaves = collision_leaf_count(root_address)
    require(total_leaves != 0, "vehicle model has no active collision leaves")
    while True:
        raw = node_raw(root_address)
        require(u32(raw, 0x8C) == 0, "vehicle collision ancestor unexpectedly has a surface")
        carrying = [
            child for child in active_children(root_address)
            if collision_leaf_count(child) != 0
        ]
        if len(carrying) != 1 or collision_leaf_count(carrying[0]) != total_leaves:
            break
        require(
            raw[0x5C:0x8C] == IDENTITY_ISO,
            "vehicle collision ancestor above the recorded root has a non-identity iso",
        )
        root_address = carrying[0]

    root_raw = node_raw(root_address)
    root = CapturedCollisionTree(
        address=root_address,
        flags=u32(root_raw, 0x9C),
        surface_address=0,
        geometry_address=0,
        kind=COLLISION_TREE_KIND_ROOT,
        wheel_index=NO_WHEEL_INDEX,
        geometry_material_index=0,
        geometry_type=0,
        geometry_reserved=0,
        box=root_raw[0x34:0x4C],
        local_iso=root_raw[0x5C:0x8C],
        shape=bytes(24),
        material_ids=(),
    )

    children: list[CapturedCollisionTree] = []
    seen_wheels: set[int] = set()

    def record(address: int) -> None:
        require(address not in visited, "vehicle collision tree contains an alias or cycle")
        visited.add(address)
        raw = node_raw(address)
        below = active_children(address)
        require(len(below) < 256, "vehicle collision node has too many children")
        if address in wheel_tree_addresses:
            require(not below, "vehicle wheel collision tree has children")
            wheel_index = wheel_tree_addresses.index(address)
            require(wheel_index not in seen_wheels, "vehicle wheel tree recorded twice")
            seen_wheels.add(wheel_index)
            children.append(capture_collision_leaf(
                reader, address, COLLISION_TREE_KIND_WHEEL, wheel_index))
        elif u32(raw, 0x8C) != 0:
            leaf = capture_collision_leaf(
                reader, address, COLLISION_TREE_KIND_BODY, NO_WHEEL_INDEX)
            children.append(dataclasses.replace(leaf, child_count=len(below)))
        else:
            require(below, "vehicle collision node has neither surface nor children")
            children.append(CapturedCollisionTree(
                address=address,
                flags=u32(raw, 0x9C),
                surface_address=0,
                geometry_address=0,
                kind=COLLISION_TREE_KIND_BODY,
                wheel_index=NO_WHEEL_INDEX,
                geometry_material_index=0,
                geometry_type=0,
                geometry_reserved=0,
                box=raw[0x34:0x4C],
                local_iso=raw[0x5C:0x8C],
                shape=bytes(24),
                material_ids=(),
                child_count=len(below),
            ))
        for child in below:
            record(child)

    for child in active_children(root_address):
        record(child)
    require(len(seen_wheels) == 4, "vehicle collision tree does not hold all four wheels")
    require(
        len(children) <= COLLISION_TREE_MAX_NODES,
        f"vehicle collision tree has {len(children)} nodes, the snapshot holds {COLLISION_TREE_MAX_NODES}",
    )
    return CapturedCollisionGraph(root=root, children=tuple(children))


def read_curve(
    reader: MemoryReader,
    owner_kind: int,
    owner_key: int,
    field_offset: int,
    curve_address: int,
) -> CapturedCurve:
    raw = reader.read(curve_address, 0x2C)
    count = u32(raw, 0x14)
    positions_address = u32(raw, 0x18)
    value_count = u32(raw, 0x20)
    values_address = u32(raw, 0x24)
    interpolation = struct.unpack_from("<i", raw, 0x28)[0]
    require(count != 0, f"curve at 0x{curve_address:08x} is empty")
    require(count == value_count, f"curve at 0x{curve_address:08x} has mismatched buffers")
    require(count <= 4096, f"curve at 0x{curve_address:08x} is too large")
    require(positions_address != 0, f"curve at 0x{curve_address:08x} has no positions")
    require(values_address != 0, f"curve at 0x{curve_address:08x} has no values")
    return CapturedCurve(
        owner_kind=owner_kind,
        owner_key=owner_key,
        field_offset=field_offset,
        object_address=curve_address,
        positions_address=positions_address,
        values_address=values_address,
        count=count,
        interpolation=interpolation,
        positions=reader.read(positions_address, count * 4),
        values=reader.read(values_address, count * 4),
    )


def capture_curves(reader: MemoryReader, graph: LiveGraph) -> tuple[CapturedCurve, ...]:
    curves: list[CapturedCurve] = []
    for offset in PRIMARY_CURVE_OFFSETS:
        address = u32(graph.vehicle_struct, offset)
        require(address != 0, f"vehicle struct curve +0x{offset:03x} is null")
        curves.append(read_curve(reader, OWNER_VEHICLE_STRUCT, 0, offset, address))

    for key, tuning in enumerate(graph.tunings):
        for offset in AUX_CURVE_OFFSETS:
            address = u32(tuning, offset)
            if address != 0:
                curves.append(read_curve(reader, OWNER_AUX_TUNING, key, offset, address))
    active_offsets = {
        curve.field_offset
        for curve in curves
        if curve.owner_kind == OWNER_AUX_TUNING and curve.owner_key == graph.tuning_key
    }
    missing = sorted(set(MANDATORY_ACTIVE_CURVES) - active_offsets)
    require(
        not missing,
        "active tuning is missing required curves: "
        + ", ".join(f"+0x{offset:03x}" for offset in missing),
    )
    return tuple(curves)


def capture_gearboxes(reader: MemoryReader, graph: LiveGraph) -> tuple[CapturedGearbox, ...]:
    gearboxes: list[CapturedGearbox] = []
    for key, (tuning_address, tuning) in enumerate(zip(graph.tuning_addresses, graph.tunings)):
        for offset in GEARBOX_OFFSETS:
            count = u32(tuning, offset)
            data_address = u32(tuning, offset + 4)
            require(
                count == GEARBOX_VALUE_COUNT,
                f"tuning key {key} gearbox +0x{offset:03x} has {count} values",
            )
            require(data_address != 0, f"tuning key {key} gearbox +0x{offset:03x} is null")
            gearboxes.append(
                CapturedGearbox(
                    owner_key=key,
                    field_offset=offset,
                    buffer_address=tuning_address + offset,
                    data_address=data_address,
                    values=reader.read(data_address, count * 4),
                )
            )
    return tuple(gearboxes)


def capture_ground_materials(
    reader: MemoryReader, graph: LiveGraph, phase_materials: bytes
) -> bytes:
    """Ground material records: object id, the four blend values at
    material +0x14 (as the phase dump captured them), then the fake-contact
    descriptor read by 0x007C3C00: mask present (+0x24 non-null), period X
    (+0x28), period Z (+0x2c), impulse scale (+0x30) and limit (+0x34).
    Every mask must be the committed 128x128 image."""
    manager = u32(graph.car, 0x68)
    require(manager != 0, "material manager is null")
    count = reader.read_u32(manager + 0x14)
    require(1 <= count <= MAX_GROUND_MATERIALS, "material manager size is out of range")
    require(len(phase_materials) == count * 20, "phase ground materials disagree with the live manager")
    pointers = struct.unpack(
        f"<{count}I", reader.read(reader.read_u32(manager + 0x18), count * 4)
    )
    records = bytearray()
    for index, address in enumerate(pointers):
        require(address != 0, "ground material is null")
        raw = reader.read(address, 0x38)
        phase_record = phase_materials[index * 20 : index * 20 + 20]
        require(raw[0x14:0x24] == phase_record[4:20], "ground material values changed since the phase dump")
        mask = u32(raw, 0x24)
        if mask != 0:
            fid = reader.read_u32(mask + 0x08)
            require(fid != 0, "bump mask has no file")
            name_address = reader.read_u32(fid + 0x78)
            name = reader.read(name_address, 2 * (len(FAKE_CONTACT_MASK_FILE) + 1))
            require(
                name.decode("utf-16-le") == FAKE_CONTACT_MASK_FILE + "\0",
                f"bump mask of material {index} is not {FAKE_CONTACT_MASK_FILE}",
            )
            image = reader.read_u32(mask + 0x48)
            require(image != 0, "bump mask has no image")
            header = reader.read(image, 0x28)
            width, height, fmt = u32(header, 0x18), u32(header, 0x1C), u32(header, 0x24)
            require(
                width == 128 and height == 128 and (fmt >> 2) & 7 == 3,
                "bump mask image shape changed",
            )
        records += phase_record
        records += struct.pack("<I", 1 if mask != 0 else 0)
        records += raw[0x28:0x38]
    return bytes(records)


class Blob:
    def __init__(self) -> None:
        self.data = bytearray(V3_HEADER_SIZE)

    def append(self, data: bytes) -> int:
        while len(self.data) % 4 != 0:
            self.data.append(0)
        offset = len(self.data)
        self.data.extend(data)
        return offset

    def reserve(self, size: int) -> int:
        return self.append(bytes(size))

    def patch(self, offset: int, data: bytes) -> None:
        require(offset <= len(self.data) and len(data) <= len(self.data) - offset, "patch is out of bounds")
        self.data[offset : offset + len(data)] = data


def build_snapshot(
    identity: GraphIdentity,
    phase: BaseSnapshot | PhaseSnapshot,
    graph: LiveGraph,
    collision: CapturedCollisionGraph,
    ids: IdMap,
    curves: tuple[CapturedCurve, ...],
    gearboxes: tuple[CapturedGearbox, ...],
    ground_materials: bytes,
) -> bytes:
    blob = Blob()
    car_offset = blob.append(phase.car)
    vehicle_struct_offset = blob.append(graph.vehicle_struct)
    tuning_descriptors_offset = blob.reserve(len(graph.tunings) * 16)
    tuning_offsets = tuple(blob.append(tuning) for tuning in graph.tunings)
    wheels_offset = blob.append(phase.wheels)
    dyna_offset = blob.append(phase.dyna)
    params_offset = blob.append(phase.params)
    state_offset = blob.append(phase.state)
    curve_descriptors_offset = blob.reserve(len(curves) * 40)
    gearbox_descriptors_offset = blob.reserve(len(gearboxes) * 24)
    ground_ids_offset = blob.append(phase.ground_ids)
    ground_materials_offset = blob.append(ground_materials)
    model_iso_offset = blob.append(phase.model_iso)
    body_reference_offset = blob.append(phase.body_reference)
    collision_root_offset = blob.reserve(COLLISION_TREE_SIZE)
    collision_children_offset = blob.reserve(
        len(collision.children) * COLLISION_TREE_SIZE
    )
    collision_material_ids = b"".join(
        bytes(child.material_ids) for child in collision.children
    )
    collision_material_ids_offset = blob.append(collision_material_ids)

    def encode_collision_tree(
        tree: CapturedCollisionTree,
        material_index: int,
    ) -> bytes:
        return struct.pack(
            COLLISION_TREE_FORMAT,
            ids.get(tree.address),
            tree.flags,
            0 if tree.surface_address == 0 else ids.get(tree.surface_address),
            0 if tree.geometry_address == 0 else ids.get(tree.geometry_address),
            tree.kind | (tree.child_count << COLLISION_TREE_KIND_SHIFT),
            tree.wheel_index,
            material_index,
            len(tree.material_ids),
            tree.geometry_material_index,
            tree.geometry_type,
            tree.geometry_reserved,
            tree.box,
            tree.local_iso,
            tree.shape,
        )

    blob.patch(collision_root_offset, encode_collision_tree(collision.root, 0))
    collision_child_data = bytearray()
    material_index = 0
    for child in collision.children:
        collision_child_data.extend(encode_collision_tree(child, material_index))
        material_index += len(child.material_ids)
    require(
        material_index == len(collision_material_ids),
        "collision material index accounting failed",
    )
    blob.patch(collision_children_offset, collision_child_data)

    curve_data_offset = len(blob.data)
    data_offsets: dict[int, int] = {}
    curve_descriptor_data = bytearray()
    for curve in curves:
        if curve.positions_address not in data_offsets:
            data_offsets[curve.positions_address] = blob.append(curve.positions)
        if curve.values_address not in data_offsets:
            data_offsets[curve.values_address] = blob.append(curve.values)
        curve_descriptor_data.extend(
            struct.pack(
                "<IIIIIIIiII",
                curve.owner_kind,
                curve.owner_key,
                curve.field_offset,
                ids.get(curve.object_address),
                ids.get(curve.positions_address),
                ids.get(curve.values_address),
                curve.count,
                curve.interpolation,
                data_offsets[curve.positions_address],
                data_offsets[curve.values_address],
            )
        )
    blob.patch(curve_descriptors_offset, curve_descriptor_data)

    gearbox_data_offset = len(blob.data)
    gearbox_offsets: dict[int, int] = {}
    gearbox_descriptor_data = bytearray()
    for gearbox in gearboxes:
        if gearbox.data_address not in gearbox_offsets:
            gearbox_offsets[gearbox.data_address] = blob.append(gearbox.values)
        gearbox_descriptor_data.extend(
            struct.pack(
                "<IIIIII",
                gearbox.owner_key,
                gearbox.field_offset,
                ids.get(gearbox.buffer_address),
                ids.get(gearbox.data_address),
                GEARBOX_VALUE_COUNT,
                gearbox_offsets[gearbox.data_address],
            )
        )
    blob.patch(gearbox_descriptors_offset, gearbox_descriptor_data)

    tuning_descriptor_data = b"".join(
        struct.pack("<IIII", key, ids.get(address), tuning_offsets[key], TUNING_SIZE)
        for key, address in enumerate(graph.tuning_addresses)
    )
    blob.patch(tuning_descriptors_offset, tuning_descriptor_data)

    header_values = (
        V3_MAGIC,
        V3_VERSION,
        len(blob.data),
        V3_HEADER_SIZE,
        0,
        phase.tick,
        identity.car_id,
        ids.get(graph.vehicle_struct_address),
        ids.get(graph.tuning_container_address),
        *identity.fixed,
        len(graph.tunings),
        graph.tuning_key,
        len(curves),
        len(gearboxes),
        car_offset,
        vehicle_struct_offset,
        tuning_descriptors_offset,
        wheels_offset,
        dyna_offset,
        params_offset,
        state_offset,
        curve_descriptors_offset,
        gearbox_descriptors_offset,
        ground_ids_offset,
        ground_materials_offset,
        model_iso_offset,
        body_reference_offset,
        curve_data_offset,
        gearbox_data_offset,
        len(collision.children),
        collision_root_offset,
        collision_children_offset,
        collision_material_ids_offset,
        *phase.arguments,
        phase.source_exe_sha256,
        phase.source_track_sha256,
    )
    header = struct.pack(V3_HEADER_FORMAT, *header_values)
    require(len(header) == V3_HEADER_SIZE, "version-3 header size mismatch")
    blob.patch(0, header)
    return bytes(blob.data)


def dword_differences(snapshot: bytes, live: bytes) -> tuple[tuple[int, int, int], ...]:
    require(len(snapshot) == len(live), "dword diff size mismatch")
    require(len(snapshot) % 4 == 0, "dword diff data is not 4-byte aligned")
    return tuple(
        (offset, u32(snapshot, offset), u32(live, offset))
        for offset in range(0, len(snapshot), 4)
        if snapshot[offset : offset + 4] != live[offset : offset + 4]
    )


def write_report(
    path: Path,
    old: BaseSnapshot | None,
    graph: LiveGraph,
    collision: CapturedCollisionGraph,
    curves: tuple[CapturedCurve, ...],
    gearboxes: tuple[CapturedGearbox, ...],
    tick_107_key: int,
    tick_107_tuning: int,
    tick_107_scale: int,
    snapshot: bytes,
) -> None:
    lines = [
        "# Vehicle snapshot live validation",
        "",
        "The lookup object is `CSceneVehicleTunings`. The key at object offset",
        "`+0x24` is the archived active tuning-array index. The lookup is not keyed",
        "by a surface or terrain material.",
        "",
        f"- tuning objects: {len(graph.tunings)}",
        f"- captured active key: {graph.tuning_key}",
        f"- tick 107 active key: {tick_107_key}",
        f"- tick 107 active tuning address: `0x{tick_107_tuning:08x}`",
        f"- tick 107 active `+0x70`: `0x{tick_107_scale:08x}`",
        f"- captured curves: {len(curves)}",
        f"- captured gearbox arrays: {len(gearboxes)}",
        f"- collision child trees: {len(collision.children)}",
        f"- version-3 snapshot bytes: {len(snapshot)}",
        f"- version-3 snapshot SHA-256: `{hashlib.sha256(snapshot).hexdigest()}`",
        "",
        "## Vehicle collision tree",
        "",
        "The collision root is the lowest live model-tree node below which",
        "every active (flag 0x80) collision shape sits; ancestors above it are",
        "active, surface-less and have an identity iso. Nodes are listed in",
        "pre-order; a node's direct child count is in bits 8..15 of its kind",
        "field. Wheel shapes have material 9, body shapes material 0; each is",
        "a sphere or an ellipsoid, and a body node may carry children.",
        "",
    ]
    labeled_trees = [("root", collision.root)]
    for index, child in enumerate(collision.children):
        labeled_trees.append((f"node {index} {collision_role(child)}", child))
    for label, tree in labeled_trees:
        box_words = struct.unpack("<6I", tree.box)
        box_values = struct.unpack("<6f", tree.box)
        shape_words = struct.unpack("<6I", tree.shape)
        shape_values = struct.unpack("<6f", tree.shape)
        lines.extend(
            (
                f"- {label}: live `0x{tree.address:08x}`, flags "
                f"`0x{tree.flags:08x}`, surface "
                f"`0x{tree.surface_address:08x}`, materials "
                f"`{list(tree.material_ids)}`",
                "  - box bits: "
                + ", ".join(f"`0x{word:08x}`" for word in box_words),
                "  - box floats: "
                + ", ".join(f"`{value:.9g}`" for value in box_values),
                "  - shape bits: "
                + ", ".join(f"`0x{word:08x}`" for word in shape_words[:3]),
                "  - shape floats: "
                + ", ".join(f"`{value:.9g}`" for value in shape_values[:3]),
            )
        )
    lines.extend(
        (
            "",
        "## Tuning object inventory",
        "",
        )
    )
    for key, tuning in enumerate(graph.tunings):
        lines.append(
            f"- key {key}: `+0x70=0x{u32(tuning, 0x70):08x}`, "
            f"raw SHA-256 `{hashlib.sha256(tuning).hexdigest()}`"
        )
    if old is None:
        lines.extend(("", "No base snapshot: first capture of this environment.", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        return
    lines.extend(
        (
            "",
            "## Selected tuning full dword diff",
            "",
            f"Every changed 4-byte field in active key {graph.tuning_key} is listed.",
            "Pointer target data is serialized separately in version 3.",
            "",
        )
    )
    active_differences = dword_differences(old.tuning, graph.tunings[graph.tuning_key])
    for offset, snapshot_value, live_value in active_differences:
        kind = "pointer" if offset in TUNING_POINTER_OFFSETS else "scalar"
        lines.append(
            f"- `+0x{offset:03x}` ({kind}): snapshot `0x{snapshot_value:08x}`, "
            f"live `0x{live_value:08x}`"
        )
    scalar_differences = tuple(
        difference
        for difference in active_differences
        if difference[0] not in TUNING_POINTER_OFFSETS
    )
    lines.extend(
        (
            "",
            f"Scalar differences: {len(scalar_differences)}.",
            "",
            "## Primary vehicle struct",
            "",
            "The direct object at car `+0x60` is a 0x50-byte",
            "`CSceneVehicleStruct`, not another `CSceneVehicleCarTuning`.",
            "Every changed 4-byte field is listed below.",
            "",
        )
    )
    primary_differences = dword_differences(old.vehicle_struct, graph.vehicle_struct)
    if not primary_differences:
        lines.append("No differences.")
    else:
        for offset, snapshot_value, live_value in primary_differences:
            kind = (
                "pointer"
                if offset in VEHICLE_STRUCT_POINTER_OFFSETS
                else "scalar"
            )
            lines.append(
                f"- `+0x{offset:02x}` ({kind}): snapshot `0x{snapshot_value:08x}`, "
                f"live `0x{live_value:08x}`"
            )
    primary_scalar_differences = tuple(
        difference
        for difference in primary_differences
        if difference[0] not in VEHICLE_STRUCT_POINTER_OFFSETS
    )
    lines.append(f"Scalar differences: {len(primary_scalar_differences)}.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def connect_and_load(
    port: int, track_name: str
) -> tuple[TMInterface, MessageType, SimStateData]:
    interface = TMInterface(port, timeout=60)
    connect_message = interface.read_message_type()
    require(
        connect_message == MessageType.SC_ON_CONNECT_SYNC,
        f"expected connect sync, got {connect_message.name}",
    )
    interface.set_timeout(300_000)
    interface.set_on_step_period(TICK_MS)
    interface.set_speed(1.0)
    interface.execute_command("set countdown_speed 5")
    interface.execute_command("set autorewind false")
    interface.execute_command("set use_valseed false")
    interface.execute_command(map_command(track_name))
    interface.respond(connect_message)
    blocked_message, _ = wait_for_race_start(interface)
    interface.set_input_state(**scheduled_input(0))
    interface.respond(blocked_message)
    blocked_message, race_time = read_next_run_step(interface)
    require(race_time == TICK_MS, f"expected race time {TICK_MS}, got {race_time}")
    reference_state = interface.get_simulation_state()
    return interface, blocked_message, reference_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--input", type=Path,
                        help="base snapshot supplying object ids; omit for the "
                             "first snapshot of an environment")
    parser.add_argument("--phase-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--map", default=DEFAULT_MAP)
    args = parser.parse_args()

    old = BaseSnapshot.load(args.input) if args.input is not None else None
    phase = PhaseSnapshot.load(args.phase_input)
    identity = (
        GraphIdentity.from_base(old) if old is not None else GraphIdentity.from_phase(phase)
    )
    interface: TMInterface | None = None
    reader: MemoryReader | None = None
    try:
        interface, blocked_message, reference_state = connect_and_load(
            args.port, args.map
        )
        reader = MemoryReader()
        graph = locate_live_graph(reader, reference_state)
        collision = capture_collision_graph(reader, graph)
        curves = capture_curves(reader, graph)
        gearboxes = capture_gearboxes(reader, graph)

        ids = IdMap(identity.reserved)
        ids.bind(graph.car_address, identity.car_id)
        ids.bind(graph.tuning_addresses[graph.tuning_key], identity.tuning_id)
        if old is not None:
            require(
                old.tuning[0x40:0x60] == graph.tunings[graph.tuning_key][0x40:0x60],
                "live active tuning is not the base snapshot's vehicle",
            )
            ids.bind(graph.vehicle_struct_address, old.integer(7))
            ids.bind(graph.tuning_container_address, old.integer(8))
            old_collision_trees = (
                old.collision_root,
                *old.collision_children,
            )
            new_collision_trees = (
                collision.root,
                *collision.children,
            )
            require(
                len(old_collision_trees) == len(new_collision_trees),
                "collision snapshot tree count changed",
            )
            for tree_index, (old_tree, new_tree) in enumerate(
                zip(old_collision_trees, new_collision_trees)
            ):
                live = {
                    "kind": new_tree.kind | (new_tree.child_count << COLLISION_TREE_KIND_SHIFT),
                    "wheel_index": new_tree.wheel_index,
                    "geometry_material_index": new_tree.geometry_material_index,
                    "geometry_type": new_tree.geometry_type,
                    "local_iso": new_tree.local_iso,
                    "shape": new_tree.shape,
                }
                base = dict(zip(live, (old_tree[i] for i in (4, 5, 8, 9, 12, 13))))
                changed = {
                    name: (base[name], live[name]) for name in live if base[name] != live[name]
                }
                require(
                    not changed,
                    f"live collision tree {tree_index} no longer matches the snapshot: "
                    + ", ".join(
                        f"{name} {old!r} -> {new!r}" for name, (old, new) in changed.items()
                    ),
                )
                ids.bind(new_tree.address, int(old_tree[0]))
                if new_tree.surface_address != 0:
                    ids.bind(new_tree.surface_address, int(old_tree[2]))
                    ids.bind(new_tree.geometry_address, int(old_tree[3]))
        ground_materials = capture_ground_materials(
            reader, graph, phase.ground_materials
        )
        snapshot = build_snapshot(
            identity, phase, graph, collision, ids, curves, gearboxes,
            ground_materials,
        )

        race_time = TICK_MS
        for tick in range(1, 108):
            interface.set_input_state(**scheduled_input(tick))
            interface.respond(blocked_message)
            blocked_message, race_time = read_next_run_step(interface)
            require(race_time == (tick + 1) * TICK_MS, f"unexpected race time {race_time}")
        require(race_time == 1080, f"tick-107 capture stopped at race time {race_time}")

        container = reader.read(graph.tuning_container_address, 0x28)
        tick_107_count = u32(container, 0x14)
        tick_107_data = u32(container, 0x18)
        tick_107_key = u32(container, 0x24)
        require(tick_107_count == len(graph.tunings), "tuning count changed by tick 107")
        require(tick_107_key < tick_107_count, "tick-107 tuning key is out of range")
        tick_107_tuning = reader.read_u32(tick_107_data + tick_107_key * 4)
        tick_107_scale = reader.read_u32(tick_107_tuning + 0x70)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(snapshot)
        write_report(
            args.report,
            old,
            graph,
            collision,
            curves,
            gearboxes,
            tick_107_key,
            tick_107_tuning,
            tick_107_scale,
            snapshot,
        )
        print("lookup key: CSceneVehicleTunings active tuning-array index at +0x24")
        print(f"tuning objects: {len(graph.tunings)}")
        print(f"captured active key: {graph.tuning_key}")
        print(f"tick 107 active key: {tick_107_key}")
        print(f"tick 107 active +0x70: 0x{tick_107_scale:08x}")
        print(f"curves: {len(curves)}")
        print(f"gearbox arrays: {len(gearboxes)}")
        print(f"collision child trees: {len(collision.children)}")
        print(
            "collision root: "
            f"flags=0x{collision.root.flags:08x} "
            f"box={collision.root.box.hex()} surface=null"
        )
        for index, child in enumerate(collision.children):
            print(
                f"collision node {index} {collision_role(child)}: "
                f"flags=0x{child.flags:08x} "
                f"box={child.box.hex()} "
                f"iso={child.local_iso.hex()} "
                f"materials={list(child.material_ids)} "
                f"ellipsoid={child.shape[:12].hex()}"
            )
        print(f"snapshot bytes: {len(snapshot)}")
        print(f"snapshot sha256: {hashlib.sha256(snapshot).hexdigest()}")
        for key, tuning in enumerate(graph.tunings):
            print(
                f"key {key}: +0x70=0x{u32(tuning, 0x70):08x} "
                f"sha256={hashlib.sha256(tuning).hexdigest()}"
            )
        active = graph.tunings[graph.tuning_key]
        print(f"active tuning +0x354 (friction model): {u32(active, 0x354)}")
        if old is not None:
            differences = dword_differences(old.tuning, active)
            print(f"active key {graph.tuning_key} raw dword differences: {len(differences)}")
            for offset, snapshot_value, live_value in differences:
                kind = "pointer" if offset in TUNING_POINTER_OFFSETS else "scalar"
                print(
                    f"  +0x{offset:03x} {kind}: "
                    f"snapshot=0x{snapshot_value:08x} live=0x{live_value:08x}"
                )
    finally:
        if reader is not None:
            reader.close()
        if interface is not None:
            interface.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
