"""Fixed physical-scale observation encoder for the 81-float policy observation.

Every feature is a raw observation column divided by a constant taken from the
physics (speed, damper travel, RPM limit, lookahead arc distance) or a
categorical turned into a one-hot / embedding index. There are no running
statistics: two runs with different early trajectories encode identically, a
degenerate first batch cannot poison the scale (the b5b26b7 regression in
analysis/training_bisect.md), and the running-normalizer collapse of the
`ladder5090_mlp_*` pilots (analysis/nn_design_review.md, finding 1) cannot
recur.

The encoder feeds the network nothing that pins the car to a place on the
track: world position (obs 0-2), world yaw, the race clock
(`elapsed_fraction`, obs 43), the remaining-distance fraction (obs 42), the
checkpoint fraction (obs 44) and the lap fraction (obs 45) are read by the
trainer for bookkeeping only. The review found the pilot tokenizer leaking the
clock and the progress fractions, which on a deterministic single-start track
identify the position on the route (finding 5); they are gone here.

Column layout of the flat observation (src/vec_env.c flatten_observation):
  0-2   world position          3-6   quaternion (x, y, z, w)
  7-9   world linear velocity   10-12 world angular velocity
  13-16 wheel speed             17-20 wheel damper
  21-24 wheel contact           25-28 wheel sliding
  29-32 wheel material id       33 engine rpm   34 gear
  35 input_steer 36 input_gas 37 input_brake
  38 arc_length 39 unwrapped_progress 40 lateral_offset 41 track_half_width
  42 remaining_distance 43 elapsed_fraction 44 next_checkpoint_fraction
  45 completed_lap_fraction
  46-77 eight (x, y, z, half_width) car-frame centerline samples
  78 turbo_active 79 turbo_type 80 turbo_remaining_progress

Car frame: +z forward, +y up, +x right (the 5 m lookahead from the A01 start is
(0, 0.45, 4.98) and R(q)^T v_world puts the whole speed on z).

Scales were chosen so that no feature exceeds 3 in magnitude on the six
world-record replays and the exported A01/B05 policy laps
(`test_encoder_features_stay_within_scale_on_record_lines`; the per-feature
maxima are tabulated in docs/RL_PLATFORM.md). Bounded quantities (speeds,
damper, rpm) are divided by a constant. Heavy-tailed geometric ones (car-frame
centerline positions, lateral offset, widths, angular velocity) are divided by
the pilot tokenizer's constant and passed through asinh: slope 1 around zero,
so a 3 m rise 5 m ahead still reads 0.56 as it did for the pilot winner, and
logarithmic beyond, so the 46 m excursions that read 8.7 for the pilot read
2.9. A plain division that bounds the tail at 3 (the first phase 1 launch
divided near positions by 25 m) cut the near-geometry resolution five-fold
and every run lost the A01 ramp within 150 updates (docs/RL_PLATFORM.md F36).
"""

from __future__ import annotations

import torch

from tmnf_rl.spaces import OBSERVATION_WIDTH

LOOKAHEAD_METERS = (5.0, 10.0, 20.0, 35.0, 55.0, 80.0, 110.0, 150.0)
SPEED_SCALE = 65.0  # m/s, linear; the E01 record peaks at 181 m/s (2.8)
ANGULAR_SCALE = 3.0  # rad/s, asinh; record lines spin at up to 12.9 rad/s (A01: 2.2)
WHEEL_SPEED_SCALE = 200.0  # linear; E01 record: 498 (2.5)
DAMPER_SCALE = 0.2  # metres, observed range 0.005 .. 0.200
RPM_SCALE = 11000.0
WIDTH_SCALE = 10.0  # metres, asinh; the A01 policy lap flies 59 m beside the centerline (2.5)
# Half widths above this are open surfaces (E01 reports the 256 m cap on its
# open sections); the policy gains nothing from telling 60 m from 256 m.
HALF_WIDTH_CAP = 60.0
LATERAL_RATIO_MAX = 3.0
# Lookahead positions are divided by the slot distance (the pilot's choice,
# which keeps the 5 m slot at full resolution) and compressed with asinh, so
# the 5 m slot that read x/d = 8.7 for the pilot while the car sat 46 m beside
# the centerline (review, finding 11) reads 2.9.
GEO_FAR_SCALE = 150.0
MATERIAL_CLASSES = 18  # 0..16 observed, 17 = out of range
# src/surface_material.h: physical IDs 0..30, plus an unknown category.
SURFACE_MATERIAL_COUNT = 31


def material_classes(version: int = 1) -> int:
    if version == 1:
        return MATERIAL_CLASSES
    if version in (2, 3):
        return SURFACE_MATERIAL_COUNT + 1
    raise ValueError(f"unsupported encoder version: {version}")


def observation_width(version: int = 1) -> int:
    material_classes(version)
    return OBSERVATION_WIDTH + (156 if version == 3 else 0)


def flat_features(version: int = 1) -> int:
    return FLAT_FEATURES + WHEEL_SLOTS * (material_classes(version) - MATERIAL_CLASSES) + (156 if version == 3 else 0)

GEAR_CLASSES = 10  # gear -1..8 shifted by one

VEHICLE_FEATURES = 18
WHEEL_FEATURES = 4
GEO_FEATURES = 9
TURBO_FEATURES = 5
GEOMETRY_SLOTS = 8
WHEEL_SLOTS = 4
FLAT_FEATURES = (
    VEHICLE_FEATURES
    + GEAR_CLASSES
    + WHEEL_SLOTS * WHEEL_FEATURES
    + WHEEL_SLOTS * MATERIAL_CLASSES
    + GEOMETRY_SLOTS * GEO_FEATURES
    + TURBO_FEATURES
)  # 193

# Names of the 193 flat features in `flatten` order, for reports and tests.
FEATURE_NAMES: tuple[str, ...] = (
    "v_car_x", "v_car_y", "v_car_z",
    "w_car_x", "w_car_y", "w_car_z",
    "up_car_x", "up_car_y", "up_car_z",
    "speed", "slip_angle", "rpm",
    "input_steer", "input_gas", "input_brake",
    "lateral", "half_width", "lateral_ratio",
    *(f"gear_{index - 1}" for index in range(GEAR_CLASSES)),
    *(
        f"wheel{wheel}_{name}"
        for wheel in range(WHEEL_SLOTS)
        for name in ("speed", "damper", "contact", "sliding")
    ),
    *(
        f"wheel{wheel}_material_{material}"
        for wheel in range(WHEEL_SLOTS)
        for material in range(MATERIAL_CLASSES)
    ),
    *(
        f"geo{int(distance)}m_{name}"
        for distance in LOOKAHEAD_METERS
        for name in ("x", "y", "z", "half_width", "distance", "dir_x", "dir_y", "dir_z", "half_width_delta")
    ),
    "turbo_active", "turbo_remaining", "turbo_type_0", "turbo_type_1", "turbo_type_2",
)
assert len(FEATURE_NAMES) == FLAT_FEATURES


def feature_names(version: int = 1) -> tuple[str, ...]:
    start = VEHICLE_FEATURES + GEAR_CLASSES + WHEEL_SLOTS * WHEEL_FEATURES
    end = start + WHEEL_SLOTS * MATERIAL_CLASSES
    names = tuple(f"wheel{wheel}_material_{material}"
                  for wheel in range(WHEEL_SLOTS)
                  for material in range(material_classes(version)))
    result = FEATURE_NAMES[:start] + names + FEATURE_NAMES[end:]
    if version == 3:
        fields = ("x", "y", "z") + tuple(f"axis{i}" for i in range(9)) + (
            "half_x", "half_y", "half_z", "valid", "finish", "ready", "respawnable")
        result += tuple(f"gate{slot}_{field}" for slot in range(8) for field in fields)
        result += ("remaining_checkpoints", "omitted_checkpoints", "remaining_laps", "finished")
    return result


# Observation columns the policy features must never read (absolute position,
# clock, route progress). `test_encoder_ignores_position_clock_and_progress`
# perturbs them and asserts the features are unchanged.
EXCLUDED_COLUMNS = (0, 1, 2, 38, 39, 42, 43, 44, 45)

# The critic alone also reads the remaining distance (obs 42). The reward is
# potential-shaped with phi(s) = -remaining / 50 m/s, so the shaped return
# from s is -phi(s) plus the tick costs and the terminal terms
# (analysis/rl_env.md): a critic without remaining distance cannot fit it.
# Measured on A01 (phase 1, first launch): explained variance 0.25-0.34 at
# updates 50-100 without it, 0.85-0.93 with it in every earlier run. The
# policy head never sees it, so the actor stays free of route coordinates.
CRITIC_FEATURES = 1
REMAINING_SCALE = 2500.0  # metres; A01 0.88, B05 0.74, E01 2.28, A08 (3 laps) 2.17
CRITIC_COLUMNS = (42,)


# Per-device constants, built once: a host-to-device copy per call would
# synchronise the stream (under GPU time-slicing every sync costs a slice).
_CONSTANTS: dict[tuple[torch.device, int], dict[str, torch.Tensor]] = {}


def _constants(obs: torch.Tensor, version: int = 1) -> dict[str, torch.Tensor]:
    count = material_classes(version)
    key = (obs.device, version)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = {
            "up_world": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=obs.device),
            "lookahead": torch.tensor(LOOKAHEAD_METERS, dtype=torch.float32, device=obs.device).view(1, GEOMETRY_SLOTS, 1),
            "gear_classes": torch.arange(GEAR_CLASSES, device=obs.device),
            "material_classes": torch.arange(count, device=obs.device),
            "turbo_classes": torch.arange(3, device=obs.device),
        }
    return _CONSTANTS[key]


def one_hot(index: torch.Tensor, classes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``F.one_hot`` without its range check, which reads the indices back to
    the host (a stream sync) on every call."""
    return (index.unsqueeze(-1) == classes).to(dtype)


def quat_inverse_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors ``v`` into the car frame given (x, y, z, w) quats."""
    xyz = -q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


@torch.no_grad()
def encode(obs: torch.Tensor, version: int = 1) -> dict[str, torch.Tensor]:
    """Raw (N, 81) observations to fixed-scale token features.

    Returns ``vehicle`` (N, 18), ``gear`` (N,) long, ``wheels`` (N, 4, 4),
    ``material`` (N, 4) long, ``geo`` (N, 8, 9), ``turbo`` (N, 5).
    """
    if obs.ndim != 2 or obs.shape[1] != observation_width(version):
        raise ValueError(f"expected (N, {observation_width(version)}) observations, got {tuple(obs.shape)}")
    n = obs.shape[0]
    count = material_classes(version)
    constants = _constants(obs, version)
    q = obs[:, 3:7]
    v_car = quat_inverse_rotate(q, obs[:, 7:10]) / SPEED_SCALE
    w_car = quat_inverse_rotate(q, obs[:, 10:13]) / ANGULAR_SCALE
    up_car = quat_inverse_rotate(q, constants["up_world"].expand(n, 3))
    speed = torch.linalg.vector_norm(v_car, dim=1, keepdim=True)
    slip = torch.atan2(v_car[:, 0:1], v_car[:, 2:3].abs() + 1e-3)
    half_width = obs[:, 41:42].clamp_max(HALF_WIDTH_CAP)
    lateral = obs[:, 40:41]
    lateral_ratio = (lateral / half_width.clamp_min(0.5)).clamp(0.0, LATERAL_RATIO_MAX)
    vehicle = torch.cat(
        (
            v_car,
            torch.asinh(w_car),
            up_car,
            speed,
            slip,
            obs[:, 33:34] / RPM_SCALE,
            obs[:, 35:38],
            torch.asinh(lateral / WIDTH_SCALE),
            torch.asinh(half_width / WIDTH_SCALE),
            lateral_ratio,
        ),
        dim=1,
    )
    gear = (obs[:, 34].round().long() + 1).clamp(0, GEAR_CLASSES - 1)

    wheels = torch.stack(
        (
            obs[:, 13:17] / WHEEL_SPEED_SCALE,
            obs[:, 17:21] / DAMPER_SCALE,
            obs[:, 21:25],
            obs[:, 25:29],
        ),
        dim=2,
    )
    material = obs[:, 29:33].round()
    material = torch.where(
        (material >= 0) & (material < count - 1),
        material,
        torch.full_like(material, count - 1),
    ).long()

    lookahead = constants["lookahead"]
    cl = obs[:, 46:78].reshape(n, GEOMETRY_SLOTS, 4)
    pos = cl[:, :, :3]
    nxt = torch.cat((pos[:, 1:], pos[:, -1:] + (pos[:, -1:] - pos[:, -2:-1])), dim=1)
    direction = nxt - pos
    direction = direction / torch.linalg.vector_norm(direction, dim=2, keepdim=True).clamp_min(1e-3)
    hw = cl[:, :, 3:4].clamp_max(HALF_WIDTH_CAP)
    hw_next = torch.cat((hw[:, 1:], hw[:, -1:]), dim=1)
    geo = torch.cat(
        (
            torch.asinh(pos / lookahead),
            torch.asinh(hw / WIDTH_SCALE),
            lookahead.expand(n, GEOMETRY_SLOTS, 1) / GEO_FAR_SCALE,
            direction,
            torch.asinh((hw_next - hw) / WIDTH_SCALE),
        ),
        dim=2,
    )

    turbo_type = obs[:, 79].round().long().clamp(0, 2)
    turbo = torch.cat(
        (obs[:, 78:79], obs[:, 80:81], one_hot(turbo_type, constants["turbo_classes"], obs.dtype)),
        dim=1,
    )
    result = {
        "vehicle": vehicle,
        "gear": gear,
        "wheels": wheels,
        "material": material,
        "geo": geo,
        "turbo": turbo,
    }
    if version == 3:
        gates = obs[:, 81:233].reshape(n, 8, 19)
        valid = gates[:, :, 15:16] > 0.5
        # Mask before transforms so padding cannot inject NaNs or geometry.
        gates = torch.where(valid, gates, torch.zeros_like(gates))
        result["gates"] = torch.cat((torch.asinh(gates[:, :, :3] / 50.0),
            gates[:, :, 3:12], torch.asinh(gates[:, :, 12:15] / 10.0),
            gates[:, :, 15:]), dim=2)
        result["gate_mask"] = valid.squeeze(-1)
        result["gate_counts"] = torch.cat((torch.asinh(obs[:, 233:236] / 8.0),
                                           obs[:, 236:237]), dim=1)
    return result


def flatten(features: dict[str, torch.Tensor], version: int = 1) -> torch.Tensor:
    """Token features in ``feature_names(version)`` order (193 or 249)."""
    n = features["vehicle"].shape[0]
    dtype = features["vehicle"].dtype
    count = material_classes(version)
    constants = _constants(features["vehicle"], version)
    result = torch.cat(
        (
            features["vehicle"],
            one_hot(features["gear"], constants["gear_classes"], dtype),
            features["wheels"].reshape(n, WHEEL_SLOTS * WHEEL_FEATURES),
            one_hot(features["material"], constants["material_classes"], dtype).reshape(n, WHEEL_SLOTS * count),
            features["geo"].reshape(n, GEOMETRY_SLOTS * GEO_FEATURES),
            features["turbo"],
        ),
        dim=1,
    )

    if version == 3:
        result = torch.cat((result, features["gates"].reshape(n, 152),
                            features["gate_counts"]), dim=1)
    return result


def encode_flat(obs: torch.Tensor, version: int = 1) -> torch.Tensor:
    return flatten(encode(obs, version), version)


class FlatEncoder:
    """Replay the fixed MLP encoder without launching each op from Python.

    Graphs contain only the existing encoder operations: no parameters, RNG,
    fusion or changed arithmetic. Keep a bounded cache for recurring batch
    sizes; small/irregular batches and CPU inputs use the ordinary encoder.
    Returned tensors own their storage, including when several forwards are
    retained for a later backward pass.
    """

    def __init__(self, *, cuda_graphs: bool = True, version: int = 1) -> None:
        material_classes(version)
        self.version = version
        self.cuda_graphs = cuda_graphs
        self._graphs: dict[tuple, tuple] = {}

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[1] != observation_width(self.version):
            raise ValueError(f"expected (N, {observation_width(self.version)}) observations, got {tuple(obs.shape)}")
        if not self.cuda_graphs or obs.device.type != "cuda" or obs.shape[0] < 32:
            return encode_flat(obs, self.version)
        key = (obs.device, obs.dtype, tuple(obs.shape))
        cached = self._graphs.get(key)
        if cached is None:
            if len(self._graphs) >= 4:
                return encode_flat(obs, self.version)
            static = torch.empty_like(obs, memory_format=torch.contiguous_format)
            static.copy_(obs)
            stream = torch.cuda.Stream(device=obs.device)
            current = torch.cuda.current_stream(obs.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    encode_flat(static, self.version)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = encode_flat(static, self.version)
            cached = (static, output, graph)
            self._graphs[key] = cached
        static, output, graph = cached
        static.copy_(obs)
        graph.replay()
        return output.clone()


def critic_context(obs: torch.Tensor) -> torch.Tensor:
    """The (N, 1) critic-only input: remaining distance / REMAINING_SCALE."""
    return obs[:, 42:43] / REMAINING_SCALE


__all__ = [
    "ANGULAR_SCALE",
    "CRITIC_COLUMNS",
    "CRITIC_FEATURES",
    "DAMPER_SCALE",
    "EXCLUDED_COLUMNS",
    "FEATURE_NAMES",
    "FLAT_FEATURES",
    "SURFACE_MATERIAL_COUNT",
    "material_classes",
    "flat_features",
    "feature_names",
    "GEAR_CLASSES",
    "GEO_FAR_SCALE",
    "GEO_FEATURES",
    "GEOMETRY_SLOTS",
    "HALF_WIDTH_CAP",
    "LATERAL_RATIO_MAX",
    "LOOKAHEAD_METERS",
    "MATERIAL_CLASSES",
    "REMAINING_SCALE",
    "RPM_SCALE",
    "SPEED_SCALE",
    "TURBO_FEATURES",
    "VEHICLE_FEATURES",
    "WHEEL_FEATURES",
    "WHEEL_SLOTS",
    "WHEEL_SPEED_SCALE",
    "WIDTH_SCALE",
    "critic_context",
    "encode",
    "encode_flat",
    "flatten",
    "quat_inverse_rotate",
]
