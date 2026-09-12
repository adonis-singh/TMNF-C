"""TD learning on TmnfVectorEnv with the mushroom body as the value function.

Every weight change is the dopamine rule at KC->MBON synapses
(``MushroomBody.dopamine_update``). Dopamine = reward prediction error from
the MBON value readout itself: PAM = relu(+RPE), PPL1 = relu(-RPE).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python taskset -c 0-5 build/venv/bin/python \
        -m tmnf_fly.train_mb --track a04 --minutes 5
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch

from tmnf_fly import FLY_DIR
from tmnf_fly.mushroom_body import CIRCUIT_PATH, MushroomBody, circuit_report, extract_circuit, load_circuit
from tmnf_rl.encoder import FLAT_FEATURES, encode_flat
from tmnf_rl.env import FINISH_REASON, TmnfVectorEnv

OUT_DIR = FLY_DIR
PROGRESS_COLUMN = 39  # unwrapped_progress in the 81-float observation
SPEED_COLUMNS = slice(7, 10)

# Six candidate actions: steer {left, none, right} x {gas, brake}.
# Env index = longitudinal * 3 + steering (longitudinal 1 = gas, 2 = brake).
ACTIONS = np.array([3, 4, 5, 6, 7, 8], dtype=np.int64)
# Factorised action code presented to the PNs: steer one-hot (3) + longitudinal one-hot (2).
ACTION_CODES = np.array(
    [
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1],
    ],
    dtype=np.float32,
)

# Gas-only repertoire: steer {left, none, right}, throttle always on (a fly does not brake).
GAS_ACTIONS = ACTIONS[:3]
GAS_ACTION_CODES = ACTION_CODES[:3]

TERM_KEYS = {2: "term_timeout", 3: "term_off_track", 4: "term_stuck", 5: "term_fell"}

# Calibration driving: gas straight 50%, gas left/right 15% each, brake 20% in total.
CALIB_ACTION_P = np.array([0.15, 0.5, 0.15, 0.05, 0.1, 0.05])

LOG_FIELDS = (
    "minute", "elapsed_s", "steps", "transitions", "episodes", "finishes", "best_lap_ms",
    "term_timeout", "term_off_track", "term_stuck", "term_fell",
    "mean_progress_m", "max_progress_m", "mean_progress_frac", "mean_return", "mean_speed_mps",
    "kc_active_fraction", "kc_used_fraction", "mean_abs_dw", "mean_abs_rpe", "mean_q",
    "value_progress_corr", "value_speed_corr", "synapses_at_floor", "synapses_at_ceiling", "epsilon", "steps_per_s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--track", default="a04")
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--thread-count", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--gain", type=float, default=40.0)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--reward-clip", type=float, default=5.0)
    p.add_argument("--eps-start", type=float, default=0.1)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--kc-active", type=int, default=200)
    p.add_argument("--action-overlap", type=float, default=0.5)
    p.add_argument("--calib-steps", type=int, default=400)
    p.add_argument("--tag", default="")
    p.add_argument("--gas-only", action="store_true", help="3 actions: steer left / none / right, always gas")
    p.add_argument("--pn-action-fraction", type=float, default=0.5, help="fraction of PNs that also read the action code")
    return p.parse_args()


def kc_statistics(mb: MushroomBody, features: torch.Tensor, progress: torch.Tensor, steps: int, n_envs: int) -> dict:
    """KC code statistics on the calibration rollout (features ordered step-major)."""
    codes = mb.action_codes[1].expand(features.shape[0], -1)
    kc = mb.kc_code(features, codes)
    used = kc.any(dim=0)
    seq = kc.view(steps, n_envs, -1)
    consecutive = (seq[1:] & seq[:-1]).sum(dim=2).float() / mb.kc_active
    prog = progress.view(steps, n_envs)
    # Distinct positions: same env, pairs of states more than 50 m apart along the route.
    g = torch.Generator(device="cpu").manual_seed(0)
    i = torch.randint(0, features.shape[0], (20000,), generator=g).to(features.device)
    j = torch.randint(0, features.shape[0], (20000,), generator=g).to(features.device)
    pair_overlap = (kc[i] & kc[j]).sum(dim=1).float() / mb.kc_active
    far = (progress[i] - progress[j]).abs() > 50.0
    near = (progress[i] - progress[j]).abs() < 2.0
    counts = kc.float().sum(dim=0)
    return {
        "active_fraction": mb.kc_active / mb.n_kc,
        "kc_used_fraction": float(used.float().mean()),
        "overlap_consecutive": float(consecutive.mean()),
        "overlap_random_pairs": float(pair_overlap.mean()),
        "overlap_far_positions_gt50m": float(pair_overlap[far].mean()),
        "overlap_near_positions_lt2m": float(pair_overlap[near].mean()),
        "n_far_pairs": int(far.sum()),
        "kc_usage_gini_top10pct_share": float(torch.sort(counts, descending=True).values[: mb.n_kc // 10].sum() / counts.sum()),
        "progress_range_m": (float(prog.min()), float(prog.max())),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.track}{args.tag}"
    log_path = OUT_DIR / f"mb_{tag}.csv"
    weights_path = OUT_DIR / f"mb_{tag}_weights.npz"
    actions_path = OUT_DIR / f"mb_{tag}_best_actions.json"

    if not CIRCUIT_PATH.is_file():
        extract_circuit()
    circuit = load_circuit()
    print(circuit_report(circuit), flush=True)

    env = TmnfVectorEnv(args.num_envs, track=args.track, thread_count=args.thread_count,
                        action_repeat=5, action_space="discrete")
    n = env.num_envs
    print(f"track {env.track_name} route {env.route_length:.1f} m, max_race_ticks {env.max_race_ticks}, "
          f"decisions per episode max {env.max_race_ticks // 5}", flush=True)
    ACTIONS, ACTION_CODES = (GAS_ACTIONS, GAS_ACTION_CODES) if args.gas_only else (globals()["ACTIONS"], globals()["ACTION_CODES"])
    K = len(ACTIONS)
    mb = MushroomBody(circuit, n_features=FLAT_FEATURES, action_codes=ACTION_CODES, device=device,
                      seed=args.seed, kc_active=args.kc_active, gain=args.gain, alpha=args.alpha,
                      pn_action_fraction=args.pn_action_fraction)

    def features_of(obs81: np.ndarray) -> torch.Tensor:
        return encode_flat(torch.as_tensor(obs81, device=device))

    # ---- calibration rollout: persistent biased random driving (mostly gas),
    # fixes the sensory normalisation and covers the first part of the route.
    env.reset()
    calib_features, calib_progress = [], []
    calib_p = CALIB_ACTION_P[:K] / CALIB_ACTION_P[:K].sum()
    calib_actions = rng.choice(K, size=n, p=calib_p)
    for _ in range(args.calib_steps):
        obs81 = env.policy_observations
        calib_features.append(features_of(obs81))
        calib_progress.append(torch.as_tensor(obs81[:, PROGRESS_COLUMN].copy(), device=device))
        switch = rng.random(n) < 0.25
        calib_actions = np.where(switch, rng.choice(K, size=n, p=calib_p), calib_actions)
        env.step(ACTIONS[calib_actions])
    calib_features = torch.cat(calib_features)
    calib_progress = torch.cat(calib_progress)
    calib = mb.calibrate(calib_features, target_action_overlap=args.action_overlap)
    print("calibration:", calib, flush=True)
    stats = kc_statistics(mb, calib_features, calib_progress, args.calib_steps, n)
    print("KC statistics:", stats, flush=True)

    # ---- training loop
    obs, _ = env.reset()
    feats = features_of(env.policy_observations)
    kc_all = mb.kc_all_actions(feats)
    q = mb.value(kc_all)  # (N, K)
    actions_idx = torch.zeros(n, dtype=torch.long, device=device)

    episode_actions: list[list[int]] = [[] for _ in range(n)]
    best_lap_ms = 0
    best_actions: np.ndarray | None = None
    totals = {"episodes": 0, "finishes": 0, "steps": 0}
    minute_acc = {"episodes": 0, "finishes": 0, "progress": 0.0, "max_progress": 0.0, "returns": 0.0,
                  "dw": 0.0, "rpe": 0.0, "q": 0.0, "speed": 0.0, "steps": 0,
                  "term_timeout": 0, "term_off_track": 0, "term_stuck": 0, "term_fell": 0}
    value_samples: list[torch.Tensor] = []
    progress_samples: list[torch.Tensor] = []
    speed_samples: list[torch.Tensor] = []
    kc_used = torch.zeros(mb.n_kc, dtype=torch.bool, device=device)

    start = time.time()
    deadline = start + 60.0 * args.minutes
    next_log = start + 60.0
    minute = 0
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    writer.writeheader()
    epsilon = args.eps_start

    while True:
        now = time.time()
        frac = min(1.0, (now - start) / (60.0 * args.minutes))
        epsilon = args.eps_start + (args.eps_end - args.eps_start) * frac
        greedy = q.argmax(dim=1)
        explore = torch.rand(n, device=device) < epsilon
        random_a = torch.randint(0, K, (n,), device=device)
        actions_idx = torch.where(explore, random_a, greedy)
        a_np = actions_idx.cpu().numpy()
        for e in range(n):
            episode_actions[e].append(int(ACTIONS[a_np[e]]))
        kc_sa = kc_all[torch.arange(n, device=device), actions_idx]  # (N, n_kc)
        q_sa = q[torch.arange(n, device=device), actions_idx]

        obs, reward, terminated, truncated, info = env.step(ACTIONS[a_np])
        done_np = terminated | truncated
        r = torch.as_tensor(np.clip(reward * args.reward_scale, -args.reward_clip, args.reward_clip), device=device)
        done = torch.as_tensor(done_np, device=device)

        feats = features_of(env.policy_observations)
        kc_all = mb.kc_all_actions(feats)
        q = mb.value(kc_all)
        v_next = q.max(dim=1).values
        target = r + args.gamma * (~done).float() * v_next
        rpe = target - q_sa
        mean_dw = mb.dopamine_update(kc_sa, rpe)

        kc_used |= kc_sa.any(dim=0)
        speed = torch.as_tensor(np.linalg.norm(env.policy_observations[:, SPEED_COLUMNS], axis=1), device=device)
        value_samples.append(v_next)
        progress_samples.append(torch.as_tensor(env.policy_observations[:, PROGRESS_COLUMN].copy(), device=device))
        speed_samples.append(speed)
        minute_acc["dw"] += mean_dw
        minute_acc["rpe"] += float(rpe.abs().mean())
        minute_acc["q"] += float(q_sa.mean())
        minute_acc["speed"] += float(speed.mean())
        minute_acc["steps"] += 1
        totals["steps"] += 1

        if done_np.any():
            final_progress = env.policy_final_observations[:, PROGRESS_COLUMN]
            for e in np.flatnonzero(done_np):
                minute_acc["episodes"] += 1
                minute_acc["progress"] += float(final_progress[e])
                minute_acc["max_progress"] = max(minute_acc["max_progress"], float(final_progress[e]))
                minute_acc["returns"] += float(info["completed_episode_return"][e])
                reason = int(info["termination_reason"][e])
                if reason in TERM_KEYS:
                    minute_acc[TERM_KEYS[reason]] += 1
                if reason == FINISH_REASON:
                    minute_acc["finishes"] += 1
                    lap = int(info["race_time_ms"][e])
                    if best_lap_ms == 0 or lap < best_lap_ms:
                        best_lap_ms = lap
                        best_actions = np.array(episode_actions[e], dtype=np.int64)
                        actions_path.write_text(json.dumps({
                            "track": args.track, "race_time_ms": lap, "action_repeat": 5,
                            "actions": [int(a) for a in ACTIONS], "training_minute": minute + 1,
                            "env_actions": best_actions.tolist()}) + "\n")
                episode_actions[e] = []

        if now >= next_log or now >= deadline:
            minute += 1
            elapsed = now - start
            v = torch.cat(value_samples)
            pr = torch.cat(progress_samples)
            sp = torch.cat(speed_samples)
            stack = torch.stack((v, pr, sp))
            corr = torch.corrcoef(stack)
            steps = max(minute_acc["steps"], 1)
            eps_count = max(minute_acc["episodes"], 1)
            floor, ceiling = mb.saturation()
            row = {
                "minute": minute,
                "elapsed_s": round(elapsed, 1),
                "steps": totals["steps"],
                "transitions": totals["steps"] * n,
                "episodes": minute_acc["episodes"],
                "finishes": minute_acc["finishes"],
                "best_lap_ms": best_lap_ms,
                "term_timeout": minute_acc["term_timeout"],
                "term_off_track": minute_acc["term_off_track"],
                "term_stuck": minute_acc["term_stuck"],
                "term_fell": minute_acc["term_fell"],
                "mean_progress_m": round(minute_acc["progress"] / eps_count, 2),
                "max_progress_m": round(minute_acc["max_progress"], 2),
                "mean_progress_frac": round(minute_acc["progress"] / eps_count / env.route_length, 4),
                "mean_return": round(minute_acc["returns"] / eps_count, 3),
                "mean_speed_mps": round(minute_acc["speed"] / steps, 2),
                "kc_active_fraction": round(mb.kc_active / mb.n_kc, 4),
                "kc_used_fraction": round(float(kc_used.float().mean()), 4),
                "mean_abs_dw": f"{minute_acc['dw'] / steps:.3e}",
                "mean_abs_rpe": round(minute_acc["rpe"] / steps, 4),
                "mean_q": round(minute_acc["q"] / steps, 4),
                "value_progress_corr": round(float(corr[0, 1]), 4),
                "value_speed_corr": round(float(corr[0, 2]), 4),
                "synapses_at_floor": round(floor, 4),
                "synapses_at_ceiling": round(ceiling, 4),
                "epsilon": round(epsilon, 4),
                "steps_per_s": round(steps / max(now - (next_log - 60.0), 1e-6), 1),
            }
            totals["episodes"] += minute_acc["episodes"]
            totals["finishes"] += minute_acc["finishes"]
            writer.writerow(row)
            log_file.flush()
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            bins = torch.clamp((pr / env.route_length * 10).long(), 0, 9)
            binned = [(float(v[bins == b].mean()) if (bins == b).any() else float("nan"), int((bins == b).sum())) for b in range(10)]
            print("  V by route decile: " + " ".join(f"{b}:{m:.2f}(n={c})" for b, (m, c) in enumerate(binned)), flush=True)
            minute_acc = {k: 0 if isinstance(v, int) else 0.0 for k, v in minute_acc.items()}
            value_samples, progress_samples, speed_samples = [], [], []
            next_log = now + 60.0
            if now >= deadline:
                break

    log_file.close()
    np.savez_compressed(weights_path, **mb.state_dict(), actions=ACTIONS, calibration=np.array(str(calib)),
                        kc_statistics=np.array(str(stats)))
    print(f"total episodes {totals['episodes']} finishes {totals['finishes']} best lap {best_lap_ms} ms", flush=True)
    print(f"wrote {log_path}, {weights_path}" + (f", {actions_path}" if best_actions is not None else ""), flush=True)
    env.close()


if __name__ == "__main__":
    main()
