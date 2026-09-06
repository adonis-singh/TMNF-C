"""Device-resident binding for the CUDA TMNF environment (``TmnfCudaVecEnv``).

The RL data path never leaves the GPU: actions are device tensors, and the
observations, final observations, transitions and step results are torch
tensors wrapped zero-copy around the env's own device buffers (valid for the
env's lifetime, overwritten in place by the next step). Every launch and
copy of the env is ordered on the caller's current torch stream, so no
explicit synchronisation is needed between the policy's kernels and the
env's; each env call blocks the host until its stream has drained.

What does reach the host after every step is the bookkeeping the trainer
does per episode: the 11 race fields of the observation and final
observation (columns 35:46 of the flat row) and the 40-byte result tail
(reward, discount, completed return and ticks, race time, executed ticks,
episode id, flags, termination reason). Those are the same arrays the CPU
``TmnfVectorEnv`` exposes, so the trainer's episode accounting, snapshot
pool and F-guards run unchanged on either env.

Snapshots are ``TmnfEnvSnapshot`` bytes as on the CPU (interchangeable in
both directions). A reset or restore writes the affected rows of the
device observation buffer, mirroring the CPU FFI's patch of
``results[i].observation``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tmnf_rl.env import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    POLICY_OBSERVATION_WIDTH,
    POLICY_TRANSITION_WIDTH,
    RESPAWN_FLAG,
    STEP_RESULT_SIZE,
    _StepResult,
)
from tmnf_rl.tracks import project_root, track_spec


# Flat-row columns of the 11 race fields (input_steer .. completed_lap_fraction).
RACE_COLUMNS = slice(35, 46)
RACE_WIDTH = 11
# Result tail: everything after the two embedded observations.
TAIL_OFFSET = 2 * OBSERVATION_SIZE
TAIL_SIZE = STEP_RESULT_SIZE - TAIL_OFFSET
TAIL_DTYPE = np.dtype(
    {
        "names": [
            "reward",
            "transition_discount",
            "completed_episode_return",
            "completed_episode_ticks",
            "race_time_ms",
            "executed_ticks",
            "episode_id",
            "terminated",
            "truncated",
            "final_observation_valid",
            "reset_only",
            "termination_reason",
        ],
        "formats": [
            np.float32,
            np.float32,
            np.float32,
            np.uint32,
            np.uint32,
            np.uint32,
            np.uint64,
            np.uint8,
            np.uint8,
            np.uint8,
            np.uint8,
            np.int32,
        ],
        "offsets": [
            getattr(_StepResult, name).offset - TAIL_OFFSET
            for name in (
                "reward",
                "transition_discount",
                "completed_episode_return",
                "completed_episode_ticks",
                "race_time_ms",
                "executed_ticks",
                "episode_id",
                "terminated",
                "truncated",
                "final_observation_valid",
                "reset_only",
                "termination_reason",
            )
        ],
        "itemsize": TAIL_SIZE,
    }
)
# TmnfAnalogAction: float steer, then gas, brake, respawn, reserved bytes;
# packed as two little-endian int32 words per environment.
ANALOG_ACTION_WORDS = 2
# Per-thread device stack by compute capability. The kernel's frame is
# architecture dependent (`cuobjdump -res-usage` on the device-linked
# object): 14,160 B on sm_120 (RTX 5090), 39,072 B on sm_86 (RTX 3060). A
# frame larger than the limit is an illegal memory access in the template
# kernel at Create; a limit far above the frame wastes the reservation the
# driver makes for every resident thread of the card.
STACK_BYTES_BY_CAPABILITY = {(12, 0): 16_384, (8, 6): 40_960}


class _DevicePointer:
    """``__cuda_array_interface__`` over a raw device pointer the env owns."""

    def __init__(self, address: int, shape: tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (address, False),
            "strides": None,
            "version": 3,
        }


def _load_library(path: Path) -> ctypes.CDLL:
    if not path.is_file():
        raise FileNotFoundError(f"TMNF CUDA shared library does not exist: {path}")
    lib = ctypes.CDLL(str(path))
    if not hasattr(lib, "tmnf_cuda_env_create"):
        raise RuntimeError(f"{path} has no CUDA environment binding (tmnf_cuda_env_create)")
    lib.tmnf_cuda_env_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.tmnf_cuda_env_create.restype = ctypes.c_void_p
    lib.tmnf_cuda_env_last_error.restype = ctypes.c_char_p
    lib.tmnf_cuda_env_destroy.argtypes = [ctypes.c_void_p]
    lib.tmnf_cuda_env_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    for name in ("tmnf_cuda_env_step_discrete", "tmnf_cuda_env_step_analog"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    for name in (
        "tmnf_cuda_env_device_results",
        "tmnf_cuda_env_device_observations",
        "tmnf_cuda_env_device_final_observations",
        "tmnf_cuda_env_device_transitions",
    ):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
        getattr(lib, name).restype = ctypes.c_void_p
    lib.tmnf_cuda_env_capture.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.tmnf_cuda_env_restore.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tmnf_cuda_env_count.argtypes = [ctypes.c_void_p]
    lib.tmnf_cuda_env_count.restype = ctypes.c_uint32
    lib.tmnf_cuda_env_route_length.argtypes = [ctypes.c_void_p]
    lib.tmnf_cuda_env_route_length.restype = ctypes.c_float
    lib.tmnf_cuda_env_checkpoint_count.argtypes = [ctypes.c_void_p]
    lib.tmnf_cuda_env_checkpoint_count.restype = ctypes.c_uint32
    lib.tmnf_cuda_env_lap_count.argtypes = [ctypes.c_void_p]
    lib.tmnf_cuda_env_lap_count.restype = ctypes.c_uint32
    lib.tmnf_env_snapshot_size.restype = ctypes.c_size_t
    lib.tmnf_step_result_size.restype = ctypes.c_size_t
    lib.tmnf_observation_size.restype = ctypes.c_size_t
    return lib


def pack_discrete_actions(actions: torch.Tensor, num_envs: int, respawn_action: bool) -> torch.Tensor:
    """Env-space discrete actions ([num_envs] integers) as the native bytes."""
    if actions.shape != (num_envs,) or actions.is_floating_point():
        raise ValueError(f"discrete actions must be an integer tensor of shape ({num_envs},)")
    values = actions.to(torch.int64)
    if respawn_action:
        values = torch.where(values >= ACTION_COUNT, (values - ACTION_COUNT) | RESPAWN_FLAG, values)
    return values.to(torch.uint8)


def pack_analog_actions(actions: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Env-space analog actions ([num_envs, 3] float32: squashed steer in
    [-1, 1], gas, brake) as ``TmnfAnalogAction`` words. The finite check is
    the F37 guard the CPU binding applies; it costs one stream sync."""
    if actions.shape != (num_envs, 3) or actions.dtype != torch.float32:
        raise ValueError(f"analog actions must be a float32 tensor of shape ({num_envs}, 3)")
    steer = actions[:, 0].contiguous()
    if not bool(torch.isfinite(steer).all()):
        raise ValueError("steer must be finite")
    pedals = actions[:, 1:].to(torch.int32)
    words = pedals[:, 0] | (pedals[:, 1] << 8)
    return torch.stack((steer.view(torch.int32), words), dim=1).contiguous()


class TmnfCudaVectorEnv:
    """Fixed-size CUDA TMNF vector environment with native same-step autoreset.

    The interface is the subset of ``TmnfVectorEnv`` the trainer uses, with
    the policy inputs as device tensors: ``device_observations``,
    ``device_final_observations`` (``[num_envs, 81]`` float32),
    ``device_transitions`` (``[num_envs, 5]``) and ``device_results``
    (``[num_envs, 688]`` uint8). ``observations`` and ``final_observations``
    carry the host ``race`` block only.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        action_repeat: int,
        action_space: str,
        max_race_ticks: int,
        horizon_ticks: int,
        gate_observations: bool = False,
        off_track_grace_ticks: int = 100,
        stuck_grace_ticks: int = 500,
        discount_per_tick: float = 0.9999,
        reference_speed: float = 50.0,
        stuck_progress_epsilon: float = 0.001,
        respawn_action: bool = False,
        track: str = "a01",
        root: str | Path | None = None,
        library_path: str | Path,
        device: torch.device,
    ) -> None:
        if num_envs <= 0 or action_repeat <= 0:
            raise ValueError("num_envs and action_repeat must be positive")
        if action_space not in {"discrete", "analog"}:
            raise ValueError("action_space must be discrete or analog")
        if max_race_ticks <= 0 or horizon_ticks <= 0:
            raise ValueError(
                "the CUDA env takes the resolved race budgets; pass a CPU env's "
                "max_race_ticks and horizon_ticks"
            )
        if device.type != "cuda":
            raise ValueError("device must be a CUDA device")
        capability = torch.cuda.get_device_capability(device)
        if capability not in STACK_BYTES_BY_CAPABILITY:
            raise RuntimeError(
                f"no device stack size is recorded for compute capability {capability}; "
                "measure the kernel frame with cuobjdump -res-usage and add it to "
                "STACK_BYTES_BY_CAPABILITY"
            )
        self.stack_bytes = STACK_BYTES_BY_CAPABILITY[capability]
        root_path = Path(root).resolve() if root is not None else project_root()
        spec = track_spec(track, root_path)
        track_path = spec.track_path(root_path)
        vehicle_path = spec.vehicle_path(root_path)
        route_path = spec.route_path(root_path)
        for path in (track_path, vehicle_path, route_path):
            if not path.is_file():
                raise FileNotFoundError(f"TMNF fixture does not exist: {path}")
        sha256_bytes = bytes.fromhex(spec.sha256)
        expected_sha256 = (ctypes.c_uint8 * len(sha256_bytes)).from_buffer_copy(sha256_bytes)

        self.device = device
        self.library_path = Path(library_path).resolve()
        self._lib = _load_library(self.library_path)
        if int(self._lib.tmnf_step_result_size()) != STEP_RESULT_SIZE:
            raise RuntimeError("native result layout differs from the Python layout")
        if int(self._lib.tmnf_observation_size()) != OBSERVATION_SIZE:
            raise RuntimeError("native observation layout differs from the Python layout")
        # The env orders every launch and copy on this stream; torch work on
        # the same stream before a step (the actions) and after it (reading
        # the buffers) is therefore ordered without further synchronisation.
        self._stream = torch.cuda.current_stream(device)
        # The env's worlds live in the kernel's local memory at the address the
        # template kernel recorded at Create. The first torch kernel launched
        # in a context relocates local memory once (measured on sm_86: a step
        # after Create fails with "local world address differs from the
        # template's" unless torch has launched a kernel before Create; later
        # cuBLAS, backward and optimizer kernels do not move it). So torch's
        # first launch happens here, before Create.
        torch.zeros(1, device=device).add_(1)
        self._stream.synchronize()
        self._handle = self._lib.tmnf_cuda_env_create(
            str(track_path).encode(),
            str(vehicle_path).encode(),
            str(route_path).encode(),
            expected_sha256,
            num_envs,
            0 if action_space == "discrete" else 1,
            max_race_ticks,
            horizon_ticks,
            off_track_grace_ticks,
            stuck_grace_ticks,
            discount_per_tick,
            reference_speed,
            stuck_progress_epsilon,
            1 if respawn_action else 0,
            self.stack_bytes,
            ctypes.c_void_p(self._stream.cuda_stream),
        )
        if not self._handle:
            reason = self._lib.tmnf_cuda_env_last_error()
            raise RuntimeError(
                "tmnf_cuda_env_create failed: "
                + (reason.decode() if reason else "unknown device error")
            )

        self.num_envs = num_envs
        self.track_id = track
        self.track_name = spec.name
        self.track_sha256 = spec.sha256
        self.restore_count = 0
        self.action_space_mode = action_space
        self.respawn_action = bool(respawn_action)
        self.action_count = ACTION_COUNT * (2 if respawn_action else 1)
        self.action_repeat = action_repeat
        self.discount_per_tick = np.float32(discount_per_tick)
        self.route_length = float(self._lib.tmnf_cuda_env_route_length(self._handle))
        self.checkpoint_count = int(self._lib.tmnf_cuda_env_checkpoint_count(self._handle))
        self.lap_count = int(self._lib.tmnf_cuda_env_lap_count(self._handle))
        self.max_race_ticks = int(max_race_ticks)
        self.horizon_ticks = int(horizon_ticks)
        self.snapshot_size = int(self._lib.tmnf_env_snapshot_size())
        self._snapshot_buffer = (ctypes.c_uint8 * (self.snapshot_size * num_envs))()
        self._flat_rows = np.empty((num_envs, POLICY_OBSERVATION_WIDTH), dtype=np.float32)
        self._flat_rows_pointer = self._flat_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        n = num_envs
        self.device_results = torch.as_tensor(
            _DevicePointer(
                int(self._lib.tmnf_cuda_env_device_results(self._handle)), (n, STEP_RESULT_SIZE), "|u1"
            ),
            device=device,
        )
        self._native_observations = torch.as_tensor(
            _DevicePointer(
                int(self._lib.tmnf_cuda_env_device_observations(self._handle)),
                (n, POLICY_OBSERVATION_WIDTH),
                "<f4",
            ),
            device=device,
        )
        self._native_final_observations = torch.as_tensor(
            _DevicePointer(
                int(self._lib.tmnf_cuda_env_device_final_observations(self._handle)),
                (n, POLICY_OBSERVATION_WIDTH),
                "<f4",
            ),
            device=device,
        )
        self.device_transitions = torch.as_tensor(
            _DevicePointer(
                int(self._lib.tmnf_cuda_env_device_transitions(self._handle)),
                (n, POLICY_TRANSITION_WIDTH),
                "<f4",
            ),
            device=device,
        )

        self.gate_observations = bool(gate_observations)
        if self.gate_observations:
            self._lib.tmnf_gate_observations_size.restype = ctypes.c_size_t
            if self._lib.tmnf_gate_observations_size() != 656:
                raise RuntimeError("native gate layout mismatch")
            self._lib.tmnf_cuda_env_observe_gates.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._lib.tmnf_cuda_env_observe_gates.restype = None
            self._lib.tmnf_cuda_env_step_with_gates.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
            self._lib.tmnf_cuda_env_step_with_gates.restype = None
            self._gate_buffer = torch.zeros((n, 164), dtype=torch.float32, device=device)
            self._final_gate_buffer = torch.zeros_like(self._gate_buffer)
            self.device_gates = self._gate_buffer[:, :160].view(n, 8, 20)[:, :, :19]
            self.device_final_gates = self._final_gate_buffer[:, :160].view(n, 8, 20)[:, :, :19]
            self.device_gate_counts = self._gate_buffer.view(torch.int32)[:, 160:]
            self.device_final_gate_counts = self._final_gate_buffer.view(torch.int32)[:, 160:]

        self.device_observations = self._native_observations
        self.device_final_observations = self._native_final_observations
        if self.gate_observations:
            self.device_observations = torch.empty((n, 237), dtype=torch.float32, device=device)
            self.device_final_observations = torch.empty_like(self.device_observations)

        # Pinned host mirrors of the per-step bookkeeping.
        self._tail_host = torch.empty((n, TAIL_SIZE), dtype=torch.uint8, pin_memory=True)
        self._race_host = torch.empty((n, RACE_WIDTH), dtype=torch.float32, pin_memory=True)
        self._final_race_host = torch.empty((n, RACE_WIDTH), dtype=torch.float32, pin_memory=True)
        self._tail = np.ndarray((n,), dtype=TAIL_DTYPE, buffer=self._tail_host.numpy())
        self.observations = {"race": self._race_host.numpy()}
        self.final_observations = {"race": self._final_race_host.numpy()}
        self.rewards = self._tail["reward"]
        self.transition_discounts = self._tail["transition_discount"]
        self.completed_episode_returns = self._tail["completed_episode_return"]
        self.completed_episode_ticks = self._tail["completed_episode_ticks"]
        self.race_times_ms = self._tail["race_time_ms"]
        self.executed_ticks = self._tail["executed_ticks"]
        self.episode_ids = self._tail["episode_id"]
        self.terminations = self._tail["terminated"].view(np.bool_)
        self.truncations = self._tail["truncated"].view(np.bool_)
        self.final_observation_valid = self._tail["final_observation_valid"].view(np.bool_)
        self.reset_only = self._tail["reset_only"].view(np.bool_)
        self.termination_reasons = self._tail["termination_reason"]
        self._step_info: dict[str, Any] = {
            "transition_discount": self.transition_discounts,
            "completed_episode_return": self.completed_episode_returns,
            "completed_episode_ticks": self.completed_episode_ticks,
            "race_time_ms": self.race_times_ms,
            "executed_ticks": self.executed_ticks,
            "episode_id": self.episode_ids,
            "termination_reason": self.termination_reasons,
            "reset_only": self.reset_only,
        }
        self.closed = False

    # --------------------------------------------------------------- helpers

    def _assemble_gate_inputs(self) -> None:
        if self.gate_observations:
            for target, native, gates, counts in (
                (self.device_observations, self._native_observations, self.device_gates, self.device_gate_counts),
                (self.device_final_observations, self._native_final_observations, self.device_final_gates, self.device_final_gate_counts)):
                target[:, :81].copy_(native)
                target[:, 81:233].copy_(gates.reshape(self.num_envs, 152))
                target[:, 233:].copy_(counts)

    def _download_step(self) -> None:
        """Result tail and both race blocks to the pinned mirrors."""
        self._tail_host.copy_(self.device_results[:, TAIL_OFFSET:], non_blocking=True)
        self._race_host.copy_(self._native_observations[:, RACE_COLUMNS], non_blocking=True)
        self._final_race_host.copy_(
            self._native_final_observations[:, RACE_COLUMNS], non_blocking=True
        )
        self._stream.synchronize()

    def _snapshot_indices(self, indices: Sequence[int] | np.ndarray | None) -> np.ndarray:
        if indices is None:
            result = np.arange(self.num_envs, dtype=np.uint32)
        else:
            candidate = np.asarray(indices)
            if candidate.ndim != 1 or candidate.size == 0:
                raise ValueError("snapshot indices must be a nonempty vector")
            if not np.issubdtype(candidate.dtype, np.integer):
                raise TypeError("snapshot indices must have an integer dtype")
            if np.any((candidate < 0) | (candidate >= self.num_envs)):
                raise ValueError("snapshot index is out of range")
            result = np.ascontiguousarray(candidate, dtype=np.uint32)
        if np.unique(result).size != result.size:
            raise ValueError("snapshot indices must be unique")
        return result

    @property
    def policy_observations(self) -> np.ndarray:
        """Host copy of the flat observation batch (checkpoints, guards)."""
        return self.device_observations.cpu().numpy()

    # ------------------------------------------------------------------- api

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.closed:
            raise RuntimeError("cannot reset a closed CUDA environment")
        self._lib.tmnf_cuda_env_reset(self._handle, self._flat_rows_pointer)
        self._native_observations.copy_(torch.from_numpy(self._flat_rows))
        self._native_final_observations.zero_()
        if self.gate_observations:
            self._lib.tmnf_cuda_env_observe_gates(self._handle, self._gate_buffer.data_ptr())
            self._final_gate_buffer.zero_()
        self._tail_host.zero_()
        self._race_host.copy_(torch.from_numpy(self._flat_rows[:, RACE_COLUMNS]))
        self._final_race_host.zero_()
        self._assemble_gate_inputs()
        self._stream.synchronize()
        return self.observations, {}

    def capture(self, indices: Sequence[int] | np.ndarray | None = None) -> tuple[bytes, ...]:
        """Copy packed opaque snapshots for the selected environments."""
        selected = self._snapshot_indices(indices)
        address = ctypes.addressof(self._snapshot_buffer)
        self._lib.tmnf_cuda_env_capture(
            self._handle,
            selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            selected.size,
            address,
        )
        return tuple(
            ctypes.string_at(address + i * self.snapshot_size, self.snapshot_size)
            for i in range(selected.size)
        )

    def restore(
        self, indices: Sequence[int] | np.ndarray, snapshots: Sequence[bytes]
    ) -> dict[str, np.ndarray]:
        """Restore selected environments from exact opaque snapshots; their
        observation rows (device and host race block) follow the restored state."""
        selected = self._snapshot_indices(indices)
        if len(snapshots) != selected.size:
            raise ValueError("snapshot count must match index count")
        address = ctypes.addressof(self._snapshot_buffer)
        for i, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, bytes):
                raise TypeError("snapshots must be bytes")
            if len(snapshot) != self.snapshot_size:
                raise ValueError("snapshot has the wrong byte size")
            ctypes.memmove(address + i * self.snapshot_size, snapshot, self.snapshot_size)
        self._lib.tmnf_cuda_env_restore(
            self._handle,
            selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            selected.size,
            address,
            self._flat_rows_pointer,
        )
        rows = self._flat_rows[: selected.size]
        index_tensor = torch.from_numpy(selected.astype(np.int64)).to(self.device)
        self._native_observations[index_tensor] = torch.from_numpy(rows).to(self.device)
        self._race_host[torch.from_numpy(selected.astype(np.int64))] = torch.from_numpy(
            np.ascontiguousarray(rows[:, RACE_COLUMNS])
        )
        if self.gate_observations:
            self._lib.tmnf_cuda_env_observe_gates(self._handle, self._gate_buffer.data_ptr())
            self._final_gate_buffer[index_tensor] = 0
        self.restore_count += 1
        self._assemble_gate_inputs()
        # Only restored slots acquire a new transition boundary. Unselected
        # host rows may have been cleared by reset and must remain untouched.
        host_indices = torch.from_numpy(selected.astype(np.int64))
        self._tail_host[host_indices] = self.device_results[index_tensor, TAIL_OFFSET:].cpu()
        self._final_race_host[host_indices] = 0
        self._stream.synchronize()
        return self.observations

    def step(
        self, actions: torch.Tensor | np.ndarray | dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step with env-space actions: a device tensor (discrete ``[N]``
        integers, analog ``[N, 3]`` float32 with squashed steer) or the host
        forms ``TmnfVectorEnv`` takes, which are uploaded first."""
        if self.closed:
            raise RuntimeError("cannot step a closed CUDA environment")
        if not isinstance(actions, torch.Tensor):
            actions = self._upload_host_actions(actions)
        if self.action_space_mode == "discrete":
            packed = pack_discrete_actions(actions, self.num_envs, self.respawn_action)
            if not self.gate_observations:
                self._lib.tmnf_cuda_env_step_discrete(self._handle, packed.data_ptr(), self.action_repeat)
        else:
            packed = pack_analog_actions(actions, self.num_envs)
            if not self.gate_observations:
                self._lib.tmnf_cuda_env_step_analog(self._handle, packed.data_ptr(), self.action_repeat)
        if self.gate_observations:
            self._lib.tmnf_cuda_env_step_with_gates(self._handle, packed.data_ptr(), self.action_repeat,
                self._gate_buffer.data_ptr(), self._final_gate_buffer.data_ptr())
        self._assemble_gate_inputs()
        self._download_step()
        if not np.array_equal(self.final_observation_valid, self.terminations | self.truncations):
            raise RuntimeError("native final-observation invariant failed")
        return self.observations, self.rewards, self.terminations, self.truncations, self._step_info

    def _upload_host_actions(self, actions: np.ndarray | dict[str, np.ndarray]) -> torch.Tensor:
        if self.action_space_mode == "discrete":
            if isinstance(actions, dict):
                raise TypeError("discrete actions must be an integer array")
            array = np.asarray(actions)
            if array.shape != (self.num_envs,) or not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"actions must be an integer array of shape ({self.num_envs},)")
            if np.any((array < 0) | (array >= self.action_count)):
                raise ValueError(f"actions must be in [0, {self.action_count})")
            return torch.from_numpy(array.astype(np.int64)).to(self.device)
        if not isinstance(actions, dict) or set(actions) != {"steer", "gas", "brake"}:
            raise ValueError("analog actions require steer, gas, brake")
        stacked = np.stack(
            [np.asarray(actions[name], dtype=np.float32) for name in ("steer", "gas", "brake")], axis=1
        )
        if stacked.shape != (self.num_envs, 3):
            raise ValueError(f"analog actions must have shape ({self.num_envs},) each")
        if not np.isfinite(stacked).all() or np.any(np.abs(stacked[:, 0]) > 1.0):
            raise ValueError("steer must be finite and in [-1, 1]")
        return torch.from_numpy(stacked).to(self.device)

    def close(self) -> None:
        if not self.closed:
            self._lib.tmnf_cuda_env_destroy(self._handle)
            self._handle = None
            self.closed = True

    def __del__(self) -> None:
        if hasattr(self, "closed"):
            self.close()


__all__ = [
    "RACE_COLUMNS",
    "TmnfCudaVectorEnv",
    "pack_analog_actions",
    "pack_discrete_actions",
]
