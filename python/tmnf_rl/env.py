"""Zero-copy Gymnasium vector binding for the native TMNF environment."""

from __future__ import annotations

import ctypes
import operator
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from tmnf_rl.tracks import project_root, track_spec


ACTION_COUNT = 12
# respawn_action=True doubles the discrete set: action a + 12 is action a with
# the Enter press on the first tick of the repeat (native byte a | 0x80). The
# default set is unchanged.
RESPAWN_FLAG = 0x80
OBSERVATION_VERSION = 2
OBSERVATION_SIZE = 324
STEP_RESULT_SIZE = 688
POLICY_OBSERVATION_WIDTH = 81
POLICY_TRANSITION_WIDTH = 5
LOOKAHEAD_METERS = np.array(
    [5.0, 10.0, 20.0, 35.0, 55.0, 80.0, 110.0, 150.0], dtype=np.float32
)
FINISH_REASON = 1
TERMINATION_NAMES = {
    1: "finish",
    2: "timeout",
    3: "off_track",
    4: "stuck",
    5: "fell",
    6: "restart",
}


class _Vec3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class _Quat(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


class _CenterlineSample(ctypes.Structure):
    _fields_ = [("position", _Vec3), ("half_width", ctypes.c_float)]


class _Observation(ctypes.Structure):
    _fields_ = [
        ("position", _Vec3),
        ("rotation", _Quat),
        ("linear_speed", _Vec3),
        ("angular_speed", _Vec3),
        ("wheel_speed", ctypes.c_float * 4),
        ("wheel_damper", ctypes.c_float * 4),
        ("wheel_contact", ctypes.c_float * 4),
        ("wheel_sliding", ctypes.c_float * 4),
        ("wheel_material", ctypes.c_float * 4),
        ("engine_rpm", ctypes.c_float),
        ("gear", ctypes.c_int32),
        ("input_steer", ctypes.c_float),
        ("input_gas", ctypes.c_float),
        ("input_brake", ctypes.c_float),
        ("arc_length", ctypes.c_float),
        ("unwrapped_progress", ctypes.c_float),
        ("lateral_offset", ctypes.c_float),
        ("track_half_width", ctypes.c_float),
        ("remaining_distance", ctypes.c_float),
        ("elapsed_fraction", ctypes.c_float),
        ("next_checkpoint_fraction", ctypes.c_float),
        ("completed_lap_fraction", ctypes.c_float),
        ("turbo_active", ctypes.c_float),
        ("turbo_type", ctypes.c_float),
        ("turbo_remaining_progress", ctypes.c_float),
        ("centerline_lookahead", _CenterlineSample * 8),
    ]


class _AnalogAction(ctypes.Structure):
    _fields_ = [
        ("steer", ctypes.c_float),
        ("gas", ctypes.c_uint8),
        ("brake", ctypes.c_uint8),
        ("respawn", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class _StepResult(ctypes.Structure):
    _fields_ = [
        ("observation", _Observation),
        ("final_observation", _Observation),
        ("reward", ctypes.c_float),
        ("transition_discount", ctypes.c_float),
        ("completed_episode_return", ctypes.c_float),
        ("completed_episode_ticks", ctypes.c_uint32),
        ("race_time_ms", ctypes.c_uint32),
        ("executed_ticks", ctypes.c_uint32),
        ("episode_id", ctypes.c_uint64),
        ("terminated", ctypes.c_uint8),
        ("truncated", ctypes.c_uint8),
        ("final_observation_valid", ctypes.c_uint8),
        ("reset_only", ctypes.c_uint8),
        ("termination_reason", ctypes.c_int32),
    ]


def _load_library(path: Path) -> ctypes.CDLL:
    if not path.is_file():
        raise FileNotFoundError(f"TMNF shared library does not exist: {path}")
    lib = ctypes.CDLL(str(path))
    lib.tmnf_env_create.argtypes = [
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
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint32,
    ]
    lib.tmnf_env_create.restype = ctypes.c_void_p
    if hasattr(lib, "tmnf_env_creation_error"):
        lib.tmnf_env_creation_error.argtypes = []
        lib.tmnf_env_creation_error.restype = ctypes.c_char_p
    if hasattr(lib, "tmnf_env_create_with_budget"):
        lib.tmnf_env_create_with_budget.argtypes = [*lib.tmnf_env_create.argtypes, ctypes.c_float]
        lib.tmnf_env_create_with_budget.restype = ctypes.c_void_p
    lib.tmnf_env_destroy.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_reset.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_step_discrete.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.tmnf_env_step_analog.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.tmnf_env_capture.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.tmnf_env_restore.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.tmnf_env_discrete_actions.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_discrete_actions.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.tmnf_env_analog_actions.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_analog_actions.restype = ctypes.c_void_p
    lib.tmnf_env_results.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_results.restype = ctypes.c_void_p
    lib.tmnf_env_count.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_count.restype = ctypes.c_uint32
    lib.tmnf_env_route_length.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_route_length.restype = ctypes.c_float
    lib.tmnf_env_checkpoint_count.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_checkpoint_count.restype = ctypes.c_uint32
    lib.tmnf_env_lap_count.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_lap_count.restype = ctypes.c_uint32
    lib.tmnf_env_horizon_ticks.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_horizon_ticks.restype = ctypes.c_uint32
    lib.tmnf_env_max_race_ticks.argtypes = [ctypes.c_void_p]
    lib.tmnf_env_max_race_ticks.restype = ctypes.c_uint32
    lib.TmnfVecEnv_FlattenStepResults.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tmnf_observation_version.restype = ctypes.c_uint32
    lib.tmnf_observation_size.restype = ctypes.c_size_t
    lib.tmnf_step_result_size.restype = ctypes.c_size_t
    lib.tmnf_env_snapshot_size.restype = ctypes.c_size_t
    return lib


class TmnfVectorEnv(VectorEnv[dict[str, np.ndarray], np.ndarray, np.ndarray]):
    """A fixed-size TMNF vector environment with native same-step autoreset.

    Every array returned by ``reset`` and ``step`` aliases the persistent native
    ``TmnfStepResult`` batch. A later call overwrites those arrays in place.

    Observation fields:
    ``vehicle`` has 34 native float fields through engine RPM, ``gear`` is the
    native int32 gear, ``race`` has controls followed by eight race-progress
    fields, ``turbo`` is active/type/normalized remaining progress, and
    ``centerline`` is ``(8, 4)`` local x/y/z plus half-width at
    ``LOOKAHEAD_METERS``.
    """

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_envs: int,
        *,
        thread_count: int = 28,
        action_repeat: int = 4,
        action_space: str = "discrete",
        max_race_ticks: int = 0,
        horizon_ticks: int = 0,
        off_track_grace_ticks: int = 100,
        stuck_grace_ticks: int = 500,
        discount_per_tick: float = 0.9999,
        reference_speed: float = 50.0,
        budget_reference_speed: float = 0.0,
        stuck_progress_epsilon: float = 0.001,
        respawn_action: bool = False,
        gate_observations: bool = False,
        track: str = "a01",
        root: str | Path | None = None,
        library_path: str | Path | None = None,
    ) -> None:
        # Gym exposes a class-level `closed`; establish instance ownership
        # before validation so failed construction never closes a missing handle.
        self.closed = True
        self._active_operation: str | None = None
        self._handle = None
        for name, value, minimum in (
            ("num_envs", num_envs, 1), ("thread_count", thread_count, 1),
            ("action_repeat", action_repeat, 1), ("max_race_ticks", max_race_ticks, 0),
            ("horizon_ticks", horizon_ticks, 0),
            ("off_track_grace_ticks", off_track_grace_ticks, 1),
            ("stuck_grace_ticks", stuck_grace_ticks, 1),
        ):
            try:
                valid = minimum <= operator.index(value) <= 0xFFFFFFFF
            except TypeError:
                valid = False
            if not valid:
                sign = "positive" if minimum else "nonnegative"
                raise ValueError(f"{name} must be a {sign} integer in [{minimum}, 4294967295]")
        if action_space not in {"discrete", "analog"}:
            raise ValueError("action_space must be discrete or analog")
        budget_pace = ctypes.c_float(budget_reference_speed).value
        if (not np.isfinite(budget_pace) or budget_reference_speed < 0
                or (budget_reference_speed > 0 and budget_pace == 0)):
            raise ValueError("budget_reference_speed must be zero or a finite positive float32")
        root_path = (
            Path(root).resolve() if root is not None else project_root()
        )
        shared_library = (
            Path(library_path).resolve()
            if library_path is not None
            else root_path / "build" / "libtmnf_physics.so"
        )
        spec = track_spec(track, root_path)
        track_name = spec.name
        track_path = spec.track_path(root_path)
        vehicle_path = spec.vehicle_path(root_path)
        route_path = spec.route_path(root_path)
        for path in (track_path, vehicle_path, route_path):
            if not path.is_file():
                raise FileNotFoundError(f"TMNF fixture does not exist: {path}")
        sha256_bytes = bytes.fromhex(spec.sha256)
        expected_sha256 = (ctypes.c_uint8 * len(sha256_bytes)).from_buffer_copy(
            sha256_bytes
        )

        self._lib = _load_library(shared_library)
        create = self._lib.tmnf_env_create
        extra = ()
        if budget_pace > 0:
            if not hasattr(self._lib, "tmnf_env_create_with_budget"):
                raise RuntimeError("this physics library does not support budget_reference_speed")
            create = self._lib.tmnf_env_create_with_budget
            extra = (budget_pace,)
        self._handle = create(
            str(track_path).encode(),
            str(vehicle_path).encode(),
            str(route_path).encode(),
            expected_sha256,
            num_envs,
            thread_count,
            0 if action_space == "discrete" else 1,
            max_race_ticks,
            horizon_ticks,
            off_track_grace_ticks,
            stuck_grace_ticks,
            discount_per_tick,
            reference_speed,
            stuck_progress_epsilon,
            1 if respawn_action else 0,
            *extra,
        )
        if not self._handle:
            if hasattr(self._lib, "tmnf_env_creation_error"):
                error = self._lib.tmnf_env_creation_error()
                if error:
                    raise ValueError(error.decode("utf-8"))
            raise RuntimeError("tmnf_env_create returned a null handle")
        self.closed = False

        self.gate_observations = bool(gate_observations)
        self.num_envs = num_envs
        self.track_id = track
        self.track_name = track_name
        self.track_sha256 = spec.sha256
        self.library_path = shared_library
        self.restore_count = 0
        self.action_space_mode = action_space
        self.respawn_action = bool(respawn_action)
        self.action_count = ACTION_COUNT * (2 if respawn_action else 1)
        self.action_repeat = action_repeat
        self.discount_per_tick = np.float32(discount_per_tick)
        self.route_length = float(self._lib.tmnf_env_route_length(self._handle))
        self.checkpoint_count = int(
            self._lib.tmnf_env_checkpoint_count(self._handle)
        )
        self.lap_count = int(self._lib.tmnf_env_lap_count(self._handle))
        # 0 asks the native side for the per-track budget (2 x laps x length at
        # reference_speed, no clamp) for either value; these are the resolved
        # values. race[8] is elapsed_ticks / max_race_ticks, so every tick
        # decode must use the resolved timeout, never the config field. Any
        # budget below the speed-cap bound aborts in tmnf_env_create.
        self.max_race_ticks = int(self._lib.tmnf_env_max_race_ticks(self._handle))
        self.horizon_ticks = int(self._lib.tmnf_env_horizon_ticks(self._handle))

        native_observation_version = int(self._lib.tmnf_observation_version())
        native_observation_size = int(self._lib.tmnf_observation_size())
        native_result_size = int(self._lib.tmnf_step_result_size())
        expected_offsets = {
            "gear": 136,
            "input_steer": 140,
            "turbo_active": 184,
            "centerline_lookahead": 196,
        }
        actual_offsets = {
            name: getattr(_Observation, name).offset for name in expected_offsets
        }
        if native_observation_version != OBSERVATION_VERSION:
            raise RuntimeError(
                f"native observation version {native_observation_version} "
                f"differs from Python version {OBSERVATION_VERSION}"
            )
        if (
            ctypes.sizeof(_Observation) != OBSERVATION_SIZE
            or actual_offsets != expected_offsets
        ):
            raise RuntimeError("Python ctypes observation offsets or size changed")
        if ctypes.sizeof(_StepResult) != STEP_RESULT_SIZE:
            raise RuntimeError("Python ctypes result offsets or size changed")
        if native_observation_size != OBSERVATION_SIZE:
            raise RuntimeError(
                "ctypes observation layout differs from the native C layout"
            )
        if native_result_size != STEP_RESULT_SIZE:
            raise RuntimeError("ctypes result layout differs from the native C layout")
        self.observation_size = native_observation_size
        self.result_size = native_result_size
        self.snapshot_size = int(self._lib.tmnf_env_snapshot_size())
        if self.snapshot_size <= 0:
            raise RuntimeError("native snapshot size must be positive")
        self._snapshot_buffer = (
            ctypes.c_uint8 * (self.snapshot_size * self.num_envs)
        )()

        result_address = int(self._lib.tmnf_env_results(self._handle))
        self._result_buffer = (
            ctypes.c_uint8 * (self.result_size * self.num_envs)
        ).from_address(result_address)
        if self.action_space_mode == "discrete":
            action_pointer = self._lib.tmnf_env_discrete_actions(self._handle)
            self._actions = np.ctypeslib.as_array(
                action_pointer, shape=(self.num_envs,)
            )
        else:
            action_address = int(self._lib.tmnf_env_analog_actions(self._handle))
            if action_address == 0:
                raise RuntimeError("native analog action buffer is null")
            analog_action_size = ctypes.sizeof(_AnalogAction)
            self._analog_action_buffer = (
                ctypes.c_uint8 * (analog_action_size * self.num_envs)
            ).from_address(action_address)
            self._analog_steer = np.ndarray(
                shape=(self.num_envs,),
                dtype=np.float32,
                buffer=self._analog_action_buffer,
                offset=_AnalogAction.steer.offset,
                strides=(analog_action_size,),
            )
            self._analog_gas = np.ndarray(
                shape=(self.num_envs,),
                dtype=np.uint8,
                buffer=self._analog_action_buffer,
                offset=_AnalogAction.gas.offset,
                strides=(analog_action_size,),
            )
            self._analog_brake = np.ndarray(
                shape=(self.num_envs,),
                dtype=np.uint8,
                buffer=self._analog_action_buffer,
                offset=_AnalogAction.brake.offset,
                strides=(analog_action_size,),
            )
            self._analog_respawn = np.ndarray(
                shape=(self.num_envs,),
                dtype=np.uint8,
                buffer=self._analog_action_buffer,
                offset=_AnalogAction.respawn.offset,
                strides=(analog_action_size,),
            )

        self.raw_results = self._view(
            0,
            (self.num_envs, self.result_size),
            np.uint8,
            (self.result_size, 1),
        )
        self.observations = self._observation_views(
            _StepResult.observation.offset
        )
        self.final_observations = self._observation_views(
            _StepResult.final_observation.offset
        )
        self.raw_observations = self._view(
            _StepResult.observation.offset,
            (self.num_envs, self.observation_size),
            np.uint8,
            (self.result_size, 1),
        )
        self.raw_final_observations = self._view(
            _StepResult.final_observation.offset,
            (self.num_envs, self.observation_size),
            np.uint8,
            (self.result_size, 1),
        )
        self.policy_observations = np.empty(
            (self.num_envs, POLICY_OBSERVATION_WIDTH), dtype=np.float32
        )
        self.policy_final_observations = np.empty_like(self.policy_observations)
        self.policy_transitions = np.empty(
            (self.num_envs, POLICY_TRANSITION_WIDTH), dtype=np.float32
        )
        self._policy_observation_pointer = self.policy_observations.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        self._policy_final_observation_pointer = (
            self.policy_final_observations.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )
        )
        self._policy_transition_pointer = self.policy_transitions.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        self.rewards = self._scalar_view(_StepResult.reward.offset, np.float32)
        self.transition_discounts = self._scalar_view(
            _StepResult.transition_discount.offset, np.float32
        )
        self.completed_episode_returns = self._scalar_view(
            _StepResult.completed_episode_return.offset, np.float32
        )
        self.completed_episode_ticks = self._scalar_view(
            _StepResult.completed_episode_ticks.offset, np.uint32
        )
        self.race_times_ms = self._scalar_view(
            _StepResult.race_time_ms.offset, np.uint32
        )
        self.executed_ticks = self._scalar_view(
            _StepResult.executed_ticks.offset, np.uint32
        )
        self.episode_ids = self._scalar_view(
            _StepResult.episode_id.offset, np.uint64
        )
        self.terminations = self._scalar_view(
            _StepResult.terminated.offset, np.bool_
        )
        self.truncations = self._scalar_view(
            _StepResult.truncated.offset, np.bool_
        )
        self.final_observation_valid = self._scalar_view(
            _StepResult.final_observation_valid.offset, np.bool_
        )
        self.reset_only = self._scalar_view(
            _StepResult.reset_only.offset, np.bool_
        )
        self.termination_reasons = self._scalar_view(
            _StepResult.termination_reason.offset, np.int32
        )

        if self.gate_observations:
            self._lib.tmnf_gate_observations_size.restype = ctypes.c_size_t
            if self._lib.tmnf_gate_observations_size() != 656:
                raise RuntimeError("native gate observation layout mismatch")
            self._lib.tmnf_env_observe_gates.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._lib.tmnf_env_observe_gates.restype = None
            self._lib.tmnf_env_step_with_gates.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
            self._lib.tmnf_env_step_with_gates.restype = None
            self._gate_buffer = np.zeros((self.num_envs, 656), dtype=np.uint8)
            self._final_gate_buffer = np.zeros_like(self._gate_buffer)
            for buffer, observations in ((self._gate_buffer, self.observations),
                                         (self._final_gate_buffer, self.final_observations)):
                # The diagnostic checkpoint index at byte 76 is deliberately
                # excluded; actors receive geometry, flags and counts only.
                observations["gates"] = np.ndarray((self.num_envs, 8, 19),
                    dtype=np.float32, buffer=buffer, strides=(656, 80, 4))
                observations["gate_counts"] = np.ndarray((self.num_envs, 4),
                    dtype=np.uint32, buffer=buffer, offset=640, strides=(656, 4))

        self._native_policy_observations = self.policy_observations
        self._native_policy_final_observations = self.policy_final_observations
        if self.gate_observations:
            self.policy_observations = np.empty((self.num_envs, 237), dtype=np.float32)
            self.policy_final_observations = np.empty_like(self.policy_observations)

        self._final_obs_objects = np.empty(self.num_envs, dtype=object)
        for index in range(self.num_envs):
            self._final_obs_objects[index] = {
                key: value[index] for key, value in self.final_observations.items()
            }
        self._step_info: dict[str, Any] = {
            "final_obs": self._final_obs_objects,
            "_final_obs": self.final_observation_valid,
            "transition_discount": self.transition_discounts,
            "completed_episode_return": self.completed_episode_returns,
            "completed_episode_ticks": self.completed_episode_ticks,
            "race_time_ms": self.race_times_ms,
            "executed_ticks": self.executed_ticks,
            "episode_id": self.episode_ids,
            "termination_reason": self.termination_reasons,
            "reset_only": self.reset_only,
        }

        self.single_observation_space = gym.spaces.Dict(
            {
                "vehicle": gym.spaces.Box(
                    -np.inf, np.inf, shape=(34,), dtype=np.float32
                ),
                "gear": gym.spaces.Box(-1, 8, shape=(), dtype=np.int32),
                "race": gym.spaces.Box(
                    -np.inf, np.inf, shape=(11,), dtype=np.float32
                ),
                "turbo": gym.spaces.Box(
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([1.0, 2.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "centerline": gym.spaces.Box(
                    -np.inf, np.inf, shape=(8, 4), dtype=np.float32
                ),
            }
        )
        if self.gate_observations:
            self.single_observation_space.spaces["gates"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(8, 19), dtype=np.float32)
            self.single_observation_space.spaces["gate_counts"] = gym.spaces.Box(
                0, np.iinfo(np.uint32).max, shape=(4,), dtype=np.uint32)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        if self.action_space_mode == "discrete":
            self.single_action_space = gym.spaces.Discrete(self.action_count)
        else:
            analog_spaces = {
                "steer": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
                "gas": gym.spaces.Discrete(2),
                "brake": gym.spaces.Discrete(2),
            }
            if self.respawn_action:
                analog_spaces["respawn"] = gym.spaces.Discrete(2)
            self.single_action_space = gym.spaces.Dict(analog_spaces)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.closed = False
        self._active_operation: str | None = None

    def _view(
        self,
        offset: int,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | type[np.generic],
        strides: tuple[int, ...],
    ) -> np.ndarray:
        return np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=self._result_buffer,
            offset=offset,
            strides=strides,
        )

    def _scalar_view(
        self, offset: int, dtype: np.dtype[Any] | type[np.generic]
    ) -> np.ndarray:
        return self._view(
            offset, (self.num_envs,), dtype, (self.result_size,)
        )

    def _observation_views(self, base: int) -> dict[str, np.ndarray]:
        float_size = ctypes.sizeof(ctypes.c_float)
        vehicle_count = _Observation.gear.offset // float_size
        race_offset = _Observation.input_steer.offset
        turbo_offset = _Observation.turbo_active.offset
        centerline_offset = _Observation.centerline_lookahead.offset
        race_count = (turbo_offset - race_offset) // float_size
        turbo_count = (centerline_offset - turbo_offset) // float_size
        return {
            "vehicle": self._view(
                base,
                (self.num_envs, vehicle_count),
                np.float32,
                (self.result_size, float_size),
            ),
            "gear": self._view(
                base + _Observation.gear.offset,
                (self.num_envs,),
                np.int32,
                (self.result_size,),
            ),
            "race": self._view(
                base + race_offset,
                (self.num_envs, race_count),
                np.float32,
                (self.result_size, float_size),
            ),
            "turbo": self._view(
                base + turbo_offset,
                (self.num_envs, turbo_count),
                np.float32,
                (self.result_size, float_size),
            ),
            "centerline": self._view(
                base + centerline_offset,
                (self.num_envs, 8, 4),
                np.float32,
                (
                    self.result_size,
                    ctypes.sizeof(_CenterlineSample),
                    float_size,
                ),
            ),
        }

    def _begin_operation(self, operation: str) -> None:
        if self.closed:
            raise RuntimeError(f"cannot {operation} a closed TMNF environment")
        if self._active_operation is not None:
            raise RuntimeError(
                f"{operation} overlaps active {self._active_operation} operation"
            )
        self._active_operation = operation

    def _end_operation(self) -> None:
        self._active_operation = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if options is not None:
            raise ValueError("TMNF reset does not accept options")
        if seed not in (None, 0):
            raise ValueError(
                "TMNF reset seed must be zero until randomized reset exists"
            )
        self._begin_operation("reset")
        try:
            self._lib.tmnf_env_reset(self._handle)
            if self.gate_observations:
                self._lib.tmnf_env_observe_gates(self._handle, self._gate_buffer.ctypes.data)
                self._final_gate_buffer.fill(0)
            self._flatten_policy_inputs()
            return self.observations, {}
        finally:
            self._end_operation()

    def _flatten_policy_inputs(self) -> None:
        self._lib.TmnfVecEnv_FlattenStepResults(
            ctypes.addressof(self._result_buffer),
            self.num_envs,
            self._policy_observation_pointer,
            self._policy_final_observation_pointer,
            self._policy_transition_pointer,
        )
        if self.gate_observations:
            for target, native, source in (
                (self.policy_observations, self._native_policy_observations, self.observations),
                (self.policy_final_observations, self._native_policy_final_observations, self.final_observations)):
                target[:, :81] = native
                target[:, 81:233] = source["gates"].reshape(self.num_envs, 152)
                target[:, 233:] = source["gate_counts"]

    def _snapshot_indices(
        self, indices: Sequence[int] | np.ndarray | None
    ) -> np.ndarray:
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

    def capture(
        self, indices: Sequence[int] | np.ndarray | None = None
    ) -> tuple[bytes, ...]:
        """Copy packed opaque snapshots for the selected environments."""
        selected = self._snapshot_indices(indices)
        self._begin_operation("capture")
        try:
            self._lib.tmnf_env_capture(
                self._handle,
                selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                selected.size,
                ctypes.addressof(self._snapshot_buffer),
            )
            address = ctypes.addressof(self._snapshot_buffer)
            return tuple(
                ctypes.string_at(address + i * self.snapshot_size, self.snapshot_size)
                for i in range(selected.size)
            )
        finally:
            self._end_operation()

    def restore(
        self,
        indices: Sequence[int] | np.ndarray,
        snapshots: Sequence[bytes],
    ) -> dict[str, np.ndarray]:
        """Restore selected environments from exact opaque snapshots."""
        selected = self._snapshot_indices(indices)
        if len(snapshots) != selected.size:
            raise ValueError("snapshot count must match index count")
        self._begin_operation("restore")
        try:
            address = ctypes.addressof(self._snapshot_buffer)
            for i, snapshot in enumerate(snapshots):
                if not isinstance(snapshot, bytes):
                    raise TypeError("snapshots must be bytes")
                if len(snapshot) != self.snapshot_size:
                    raise ValueError("snapshot has the wrong byte size")
                ctypes.memmove(
                    address + i * self.snapshot_size,
                    snapshot,
                    self.snapshot_size,
                )
            self._lib.tmnf_env_restore(
                self._handle,
                selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                selected.size,
                address,
            )
            if self.gate_observations:
                self._lib.tmnf_env_observe_gates(self._handle, self._gate_buffer.ctypes.data)
                self._final_gate_buffer[selected] = 0
            self.restore_count += 1
            self._flatten_policy_inputs()
            return self.observations
        finally:
            self._end_operation()

    def step(
        self, actions: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        self._begin_operation("step")
        try:
            if self.action_space_mode == "discrete":
                self._write_discrete_actions(actions)
                if not self.gate_observations:
                    self._lib.tmnf_env_step_discrete(self._handle, self.action_repeat)
            else:
                self._write_analog_actions(actions)
                if not self.gate_observations:
                    self._lib.tmnf_env_step_analog(self._handle, self.action_repeat)
            if self.gate_observations:
                self._lib.tmnf_env_step_with_gates(self._handle, self.action_repeat,
                    self._gate_buffer.ctypes.data, self._final_gate_buffer.ctypes.data)
            if not np.array_equal(
                self.final_observation_valid,
                self.terminations | self.truncations,
            ):
                raise RuntimeError("native final-observation invariant failed")
            self._flatten_policy_inputs()
            return (
                self.observations,
                self.rewards,
                self.terminations,
                self.truncations,
                self._step_info,
            )
        finally:
            self._end_operation()

    def _write_discrete_actions(
        self, actions: np.ndarray | dict[str, np.ndarray]
    ) -> None:
        if isinstance(actions, dict):
            raise TypeError("discrete actions must be an integer array")
        action_array = np.asarray(actions)
        if action_array.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if not np.issubdtype(action_array.dtype, np.integer):
            raise TypeError("actions must have an integer dtype")
        if np.any((action_array < 0) | (action_array >= self.action_count)):
            raise ValueError(f"actions must be in [0, {self.action_count})")
        if self.respawn_action:
            respawn = action_array >= ACTION_COUNT
            action_array = np.where(
                respawn, (action_array - ACTION_COUNT) | RESPAWN_FLAG, action_array
            )
        self._actions[:] = action_array

    def _write_analog_actions(
        self, actions: np.ndarray | dict[str, np.ndarray]
    ) -> None:
        if not isinstance(actions, dict):
            raise TypeError("analog actions must be a steer/gas/brake dictionary")
        names = ("steer", "gas", "brake") + (
            ("respawn",) if self.respawn_action else ()
        )
        if set(actions) != set(names):
            raise ValueError(f"analog actions require {', '.join(names)}")

        steer = np.asarray(actions["steer"])
        if steer.shape != (self.num_envs,):
            raise ValueError(f"steer must have shape ({self.num_envs},)")
        if not np.issubdtype(steer.dtype, np.floating):
            raise TypeError("steer must have a floating dtype")
        if not np.isfinite(steer).all() or np.any((steer < -1.0) | (steer > 1.0)):
            raise ValueError("steer must be finite and in [-1, 1]")

        binaries = {}
        for name in names[1:]:
            value = np.asarray(actions[name])
            if value.shape != (self.num_envs,):
                raise ValueError(f"{name} must have shape ({self.num_envs},)")
            if not (
                np.issubdtype(value.dtype, np.bool_)
                or np.issubdtype(value.dtype, np.integer)
            ):
                raise TypeError(f"{name} must have a boolean or integer dtype")
            if np.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must be binary")
            binaries[name] = value

        self._analog_steer[:] = steer
        self._analog_gas[:] = binaries["gas"]
        self._analog_brake[:] = binaries["brake"]
        if self.respawn_action:
            self._analog_respawn[:] = binaries["respawn"]

    def close(self, **kwargs: Any) -> None:
        del kwargs
        if not self.closed:
            self._begin_operation("close")
            try:
                self._lib.tmnf_env_destroy(self._handle)
                self._handle = None
                self.closed = True
            finally:
                self._end_operation()

    def __del__(self) -> None:
        if hasattr(self, "closed"):
            self.close()


__all__ = [
    "ACTION_COUNT",
    "RESPAWN_FLAG",
    "FINISH_REASON",
    "LOOKAHEAD_METERS",
    "OBSERVATION_VERSION",
    "POLICY_OBSERVATION_WIDTH",
    "TERMINATION_NAMES",
    "TmnfVectorEnv",
]
