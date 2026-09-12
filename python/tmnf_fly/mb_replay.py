"""Showcase episode from frozen mushroom-body weights.

Rolls the frozen policy (no plasticity) out on TmnfVectorEnv built exactly as
train_mb.py builds it, keeps the fastest finishing episode, records every
circuit quantity a renderer needs per decision, writes the per-tick input
schedule and verifies it tick by tick in a fresh single env.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python taskset -c 0-5 build/venv/bin/python \
        -m tmnf_fly.mb_replay local/fly/mb_a04_5min_weights.npz --track a04 \
        --envs 64 --episodes 2000 --epsilon 0.0 --out local/fly/mb_a04_5min_showcase.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tmnf_fly.mushroom_body import CIRCUIT_PATH, PUNISHMENT, REWARD, MushroomBody, load_circuit
from tmnf_fly.pose import car_poses
from tmnf_fly.train_mb import ACTIONS, SPEED_COLUMNS
from tmnf_rl.encoder import FLAT_FEATURES, encode_flat
from tmnf_rl.env import FINISH_REASON, TmnfVectorEnv
from tmnf_rl.inputs import InputSchedule
from tmnf_rl.replays import EXPORT_PADDING_TICKS

PROGRESS_RACE_COLUMN = 4  # unwrapped_progress in observations["race"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("weights", type=Path)
    p.add_argument("--track", default="a04")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--actions", type=Path, help="train_mb best_actions.json: force this decision sequence "
                   "(env action values) and record the circuit's activity along it with the loaded weights")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--thread-count", type=int, default=6)
    # RPE definition, identical to train_mb defaults; the RPE is recorded, never applied.
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--reward-clip", type=float, default=5.0)
    return p.parse_args()


def load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        state = {k: raw[k] for k in raw.files}
    # The MushroomBody that trained these weights allocated pn_u with
    # n_action_dims trailing columns that are never written; the forward pass
    # reads only the first n_features columns. Drop them, insisting they are zero.
    pn_u = state["pn_u"]
    if pn_u[:, FLAT_FEATURES:].any():
        raise ValueError(f"pn_u has non-zero columns beyond feature {FLAT_FEATURES}")
    state["pn_u"] = np.ascontiguousarray(pn_u[:, :FLAT_FEATURES])
    if not np.array_equal(state["actions"], ACTIONS[: len(state["actions"])]):
        raise ValueError(f"saved actions {state['actions']} are not a prefix of train_mb ACTIONS {ACTIONS}")
    return state


def dan_classes(circuit: dict) -> np.ndarray:
    """Per DAN: REWARD (PAM), PUNISHMENT (PPL1) or -1 for DANs without a compartment."""
    comp_class = circuit["comp_class"]
    out = np.full(len(circuit["dan_body"]), -1, dtype=np.int64)
    for i, comps in enumerate(circuit["dan_comp"]):
        classes = set(int(c) for c in comp_class[comps])
        if len(classes) > 1:
            raise ValueError(f"DAN {i} spans compartments of both classes")
        if classes:
            out[i] = classes.pop()
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    circuit = load_circuit(CIRCUIT_PATH)
    state = load_state(args.weights)

    ACTIONS = state["actions"]
    ACTION_CODES = state["action_codes"]
    mb = MushroomBody(circuit, n_features=FLAT_FEATURES, action_codes=ACTION_CODES, device=device,
                      seed=args.seed, kc_active=int(state["kc_active"]), gain=float(state["gain"]))
    mb.load_state_dict(state)
    K = len(ACTIONS)

    dan_class = dan_classes(circuit)
    dan_sign = torch.as_tensor(np.where(dan_class == REWARD, 1.0, np.where(dan_class == PUNISHMENT, -1.0, 0.0)),
                               dtype=torch.float32, device=device)  # (340,)

    env = TmnfVectorEnv(args.envs, track=args.track, thread_count=args.thread_count,
                        action_repeat=5, action_space="discrete")
    n = env.num_envs
    print(f"track {env.track_name} route {env.route_length:.1f} m, max_race_ticks {env.max_race_ticks}", flush=True)

    def features_of(obs81: np.ndarray) -> torch.Tensor:
        return encode_flat(torch.as_tensor(obs81, device=device))

    forced = None
    if args.actions is not None:
        lap = json.loads(args.actions.read_text())
        if lap["track"] != args.track:
            raise SystemExit(f"{args.actions} is a {lap['track']} lap, not {args.track}")
        env_actions = np.asarray(lap["env_actions"], dtype=np.int64)
        idx_of = {int(a): i for i, a in enumerate(ACTIONS)}
        forced = torch.as_tensor([idx_of[int(a)] for a in env_actions], device=device)
        if args.envs != 1 or args.episodes != 1:
            raise SystemExit("--actions needs --envs 1 --episodes 1")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    ar = torch.arange(n, device=device)

    env.reset()
    feats = features_of(env.policy_observations)
    kc_all = mb.kc_all_actions(feats)
    q = mb.value(kc_all)

    records: list[list[dict]] = [[] for _ in range(n)]
    episodes = 0
    finishes = 0
    progress_sum = 0.0
    best_ms = 0
    best: list[dict] | None = None
    best_episode_index = -1
    steps = 0

    with torch.no_grad():
        while episodes < args.episodes:
            greedy = q.argmax(dim=1)
            explore = torch.rand(n, device=device, generator=gen) < args.epsilon
            random_a = torch.randint(0, K, (n,), device=device, generator=gen)
            a_idx = torch.where(explore, random_a, greedy)
            if forced is not None:
                if steps >= len(forced):
                    raise SystemExit(f"forced sequence of {len(forced)} decisions ended before the episode did")
                a_idx = forced[steps].expand(n)
            kc_sa = kc_all[ar, a_idx]
            q_sa = q[ar, a_idx]
            drives = mb.drives(kc_sa)

            progress_before = env.observations["race"][:, PROGRESS_RACE_COLUMN].copy()
            speed_before = np.linalg.norm(env.policy_observations[:, SPEED_COLUMNS], axis=1)
            position, _, _ = car_poses(env)

            a_np = a_idx.cpu().numpy()
            q_all_np = q.cpu().numpy()
            _, reward, terminated, truncated, info = env.step(ACTIONS[a_np])
            done_np = terminated | truncated
            r = torch.as_tensor(np.clip(reward * args.reward_scale, -args.reward_clip, args.reward_clip), device=device)
            done = torch.as_tensor(done_np, device=device)

            feats = features_of(env.policy_observations)
            kc_all = mb.kc_all_actions(feats)
            q = mb.value(kc_all)
            v_next = q.max(dim=1).values
            rpe = r + args.gamma * (~done).float() * v_next - q_sa
            dan = torch.relu(dan_sign.unsqueeze(0) * rpe.unsqueeze(1))  # (N, 340)

            progress_after = np.where(done_np, env.final_observations["race"][:, PROGRESS_RACE_COLUMN],
                                      env.observations["race"][:, PROGRESS_RACE_COLUMN])
            speed_after = np.where(done_np, np.linalg.norm(env.policy_final_observations[:, SPEED_COLUMNS], axis=1),
                                   np.linalg.norm(env.policy_observations[:, SPEED_COLUMNS], axis=1))
            executed = info["executed_ticks"].copy()
            kc_np = kc_sa.cpu().numpy()
            drives_np = drives.cpu().numpy()
            dan_np = dan.cpu().numpy()
            q_sa_np = q_sa.cpu().numpy()
            rpe_np = rpe.cpu().numpy()
            r_np = r.cpu().numpy()
            for e in range(n):
                records[e].append({
                    "action_idx": int(a_np[e]),
                    "env_action": int(ACTIONS[a_np[e]]),
                    "executed_ticks": int(executed[e]),
                    "progress_before": float(progress_before[e]),
                    "progress_after": float(progress_after[e]),
                    "speed": float(speed_before[e]),
                    "speed_after": float(speed_after[e]),
                    "position": position[e].copy(),
                    "kc": kc_np[e],
                    "mbon": drives_np[e],
                    "dan": dan_np[e],
                    "value": float(q_sa_np[e]),
                    "q_all": q_all_np[e],
                    "reward": float(r_np[e]),
                    "rpe": float(rpe_np[e]),
                })
            steps += 1

            for e in np.flatnonzero(done_np):
                episodes += 1
                progress_sum += float(progress_after[e])
                if int(info["termination_reason"][e]) == FINISH_REASON:
                    finishes += 1
                    ms = int(info["race_time_ms"][e])
                    if best_ms == 0 or ms < best_ms:
                        best_ms = ms
                        best = records[e]
                        best_episode_index = episodes - 1
                records[e] = []
            if steps % 500 == 0:
                print(f"steps {steps} episodes {episodes} finishes {finishes} best {best_ms} ms", flush=True)
    env.close()

    finish_rate = finishes / episodes
    mean_progress = progress_sum / episodes
    print(f"episodes {episodes} finishes {finishes} finish_rate {finish_rate:.4f} "
          f"mean_progress {mean_progress:.2f} m ({mean_progress / env.route_length:.4f}) best_lap_ms {best_ms}", flush=True)
    if best is None:
        raise SystemExit(f"no episode finished at epsilon {args.epsilon}; rerun with a larger --epsilon")

    def stack(key: str, dtype) -> np.ndarray:
        return np.asarray([rec[key] for rec in best], dtype=dtype)

    env_action = stack("env_action", np.int64)
    executed_ticks = stack("executed_ticks", np.int64)
    ticks_actions = np.repeat(env_action, executed_ticks).astype(np.int16)
    total_ticks = int(ticks_actions.shape[0])

    inputs_path = args.out.with_name(args.out.name.removesuffix(".npz") + ".inputs.bin")
    # export_viewer_scene reads the race state at the top of iteration N for the
    # state after N steps: a T-tick lap needs T + 1 records to register the finish.
    inputs_sha = InputSchedule("discrete", ticks_actions).write(inputs_path, total_ticks + EXPORT_PADDING_TICKS)

    # Verify tick by tick in a fresh single env with action_repeat=1.
    check = TmnfVectorEnv(1, track=args.track, thread_count=1, action_repeat=1, action_space="discrete")
    check.reset()
    verify_ms = -1
    for t, a in enumerate(ticks_actions.tolist()):
        _, _, term, trunc, cinfo = check.step(np.array([a], dtype=np.int64))
        if term[0] or trunc[0]:
            if t != total_ticks - 1:
                raise RuntimeError(f"tick replay ended at tick {t + 1} of {total_ticks}, reason {int(cinfo['termination_reason'][0])}")
            if int(cinfo["termination_reason"][0]) != FINISH_REASON:
                raise RuntimeError(f"tick replay ended with reason {int(cinfo['termination_reason'][0])}, not finish")
            verify_ms = int(cinfo["race_time_ms"][0])
    check.close()
    if verify_ms < 0:
        raise RuntimeError(f"tick replay did not terminate within {total_ticks} ticks")
    if verify_ms != best_ms:
        raise RuntimeError(f"tick replay finished in {verify_ms} ms, rollout episode in {best_ms} ms")

    out = {
        "action_idx": stack("action_idx", np.int64),
        "env_action": env_action,
        "executed_ticks": executed_ticks,
        "progress_before": stack("progress_before", np.float32),
        "progress_after": stack("progress_after", np.float32),
        "speed": stack("speed", np.float32),
        "speed_after": stack("speed_after", np.float32),
        "position": stack("position", np.float32),
        "kc": stack("kc", np.bool_),
        "mbon": stack("mbon", np.float32),
        "dan": stack("dan", np.float32),
        "value": stack("value", np.float32),
        "q_all": stack("q_all", np.float32),
        "reward": stack("reward", np.float32),
        "rpe": stack("rpe", np.float32),
        "ticks_actions": ticks_actions,
        "kc_body": circuit["kc_body"],
        "mbon_body": circuit["mbon_body"],
        "dan_body": circuit["dan_body"],
        "mbon_group": circuit["mbon_group"],
        "dan_class": dan_class,
        "actions": ACTIONS,
        "action_codes": ACTION_CODES,
        "race_time_ms": np.array(best_ms, np.int64),
        "verified_race_time_ms": np.array(verify_ms, np.int64),
        "total_ticks": np.array(total_ticks, np.int64),
        "epsilon": np.array(args.epsilon),
        "episodes": np.array(episodes, np.int64),
        "finishes": np.array(finishes, np.int64),
        "finish_rate": np.array(finish_rate),
        "mean_progress_m": np.array(mean_progress),
        "route_length_m": np.array(env.route_length),
        "best_episode_index": np.array(best_episode_index, np.int64),
        "track": np.array(args.track),
        "weights": np.array(str(args.weights)),
        "inputs_sha256": np.array(inputs_sha),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"best episode: {len(best)} decisions, {total_ticks} ticks, race_time_ms {best_ms}, "
          f"tick replay race_time_ms {verify_ms}", flush=True)
    print(f"wrote {args.out} and {inputs_path} (sha256 {inputs_sha})", flush=True)
    for k, v in out.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
