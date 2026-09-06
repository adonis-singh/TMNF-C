"""Minimal TMInterface 2 socket client.

Protocol adapted from Linesight's Python_Link.as and tminterface2.py:
https://github.com/Linesight-RL/linesight
"""

from __future__ import annotations

import socket
import struct
from enum import IntEnum, auto

import numpy as np
from tminterface.structs import CheckpointData, SimStateData


class MessageType(IntEnum):
    SC_RUN_STEP_SYNC = auto()
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = auto()
    SC_LAP_COUNT_CHANGED_SYNC = auto()
    SC_REQUESTED_FRAME_SYNC = auto()
    SC_ON_CONNECT_SYNC = auto()
    C_SET_SPEED = auto()
    C_REWIND_TO_STATE = auto()
    C_REWIND_TO_CURRENT_STATE = auto()
    C_GET_SIMULATION_STATE = auto()
    C_SET_INPUT_STATE = auto()
    C_GIVE_UP = auto()
    C_PREVENT_SIMULATION_FINISH = auto()
    C_SHUTDOWN = auto()
    C_EXECUTE_COMMAND = auto()
    C_SET_TIMEOUT = auto()
    C_RACE_FINISHED = auto()
    C_REQUEST_FRAME = auto()
    C_RESET_CAMERA = auto()
    C_SET_ON_STEP_PERIOD = auto()
    C_UNREQUEST_FRAME = auto()
    C_TOGGLE_INTERFACE = auto()
    C_IS_IN_MENUS = auto()
    C_GET_INPUTS = auto()


class TMInterface:
    def __init__(self, port: int, timeout: float = 10.0):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        self.socket.sendall(struct.pack("<i", MessageType.C_SHUTDOWN))
        self.socket.close()

    def read_message_type(self) -> MessageType:
        return MessageType(self._read_int32())

    def respond(self, message_type: MessageType) -> None:
        self.socket.sendall(struct.pack("<i", message_type))

    def read_run_step(self) -> int:
        return self._read_int32()

    def read_count_change(self) -> tuple[int, int]:
        return self._read_int32(), self._read_int32()

    def set_speed(self, speed: float) -> None:
        self.socket.sendall(struct.pack("<if", MessageType.C_SET_SPEED, np.float32(speed)))

    def rewind_to_state(self, state: SimStateData) -> None:
        self.socket.sendall(struct.pack("<ii", MessageType.C_REWIND_TO_STATE, len(state.data)))
        self.socket.sendall(state.data)

    def rewind_to_current_state(self) -> None:
        self.socket.sendall(struct.pack("<i", MessageType.C_REWIND_TO_CURRENT_STATE))

    def get_simulation_state(self) -> SimStateData:
        self.socket.sendall(struct.pack("<i", MessageType.C_GET_SIMULATION_STATE))
        state_length = self._read_int32()
        state = SimStateData(self._read_exact(state_length))
        state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
        state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)
        return state

    def set_input_state(self, *, left: bool, right: bool, accelerate: bool, brake: bool) -> None:
        self.socket.sendall(
            struct.pack(
                "<iBBBB",
                MessageType.C_SET_INPUT_STATE,
                left,
                right,
                accelerate,
                brake,
            )
        )

    def give_up(self) -> None:
        self.socket.sendall(struct.pack("<i", MessageType.C_GIVE_UP))

    def prevent_simulation_finish(self) -> None:
        self.socket.sendall(struct.pack("<i", MessageType.C_PREVENT_SIMULATION_FINISH))

    def race_finished(self) -> bool:
        self.socket.sendall(struct.pack("<i", MessageType.C_RACE_FINISHED))
        return self._read_int32() != 0

    def execute_command(self, command: str) -> None:
        encoded = command.encode()
        self.socket.sendall(struct.pack("<ii", MessageType.C_EXECUTE_COMMAND, len(encoded)))
        self.socket.sendall(encoded)

    def set_timeout(self, milliseconds: int) -> None:
        self.socket.sendall(struct.pack("<iI", MessageType.C_SET_TIMEOUT, milliseconds))

    def set_on_step_period(self, milliseconds: int) -> None:
        self.socket.sendall(struct.pack("<ii", MessageType.C_SET_ON_STEP_PERIOD, milliseconds))

    def is_in_menus(self) -> bool:
        self.socket.sendall(struct.pack("<i", MessageType.C_IS_IN_MENUS))
        return self._read_int32() != 0

    def _read_exact(self, size: int) -> bytes:
        data = self.socket.recv(size, socket.MSG_WAITALL)
        if len(data) != size:
            raise ConnectionError(f"socket closed after {len(data)} of {size} bytes")
        return data

    def _read_int32(self) -> int:
        return struct.unpack("<i", self._read_exact(4))[0]
