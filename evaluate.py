"""
Monte Carlo evaluation + figures (TINTIN-style: compare a classical baseline against the RL policy).

    python evaluate.py                                   # baseline only
    python evaluate.py --model models/best/best_model.zip --algo ppo --episodes 200

Writes to results/:  summary.json, wind_sweep.png, landing_scatter.png, trajectory.png
Success (same rule as the environment): soft touchdown (<3 m/s down, <2 m/s sideways, tilt <10 deg),
stays upright for 1.2 s (<15 deg), lands on the 5 m pad, no hull contact.
"""
import argparse
import json
import os

import numpy as np

from rocket_env import RocketLanderEnv, DT
from rocket_model import PAD_RADIUS, FUEL_MASS


def get_policies(args):
    from baseline import SuicideBurnPD
    pol = {"baseline (PD + suicide burn)": SuicideBurnPD()}
    if args.model:
        from stable_baselines3 import PPO, SAC
        model = {"ppo": PPO, "sac": SAC}[args.algo].load(args.model, device="cpu")
        pol[f"RL ({args.algo.upper()})"] = lambda obs: model.predict(obs, deterministic=True)[0]
    return pol


def run(policy, n, seed0=10_000, record_first=False, **env_kwargs):
    env = RocketLanderEnv(**env_kwargs)
    rows, traj = [], None
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        log = [] if (record_first and ep == 0) else None
        done = False
        while not done:
            obs, _, te, tr, info = env.step(policy(obs))
            done = te or tr
            if log is not None:
                log.append((env.steps * DT, env._height(), env.data.qvel[2], env.thr, env.fuel))
        if log is not None:
            traj = np.array(log)
        t = env.touch
        rows.append(dict(success=bool(env.success), touched=t is not None,
                         vz=-t["vz"] if t else np.nan, vxy=t["vxy"] if t else np.nan,
                         tilt=np.degrees(t["tilt"]) if t else np.nan,
                         x=float(env.data.qpos[0]), y=float(env.data.qpos[1]),
                         miss=float(np.linalg.norm(env.data.qpos[:2])),
                         fuel_used=env.fuel0 - env.fuel))
    return rows, traj


def summarize(rows):
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    ok = g("success").astype(bool)
    f = lambda k: float(np.nanmean(g(k)[ok])) if ok.any() else float("nan")
    return dict(episodes=len(rows), success_rate=float(ok.mean()),
                touchdown_speed_mps=f("vz"), miss_distance_m=f("miss"), fuel_used_kg=f("fuel_used"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--sweep-episodes", type=int, default=40)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pols = get_policies(args)
    summary, rows_all, trajs = {}, {}, {}
    for name, pol in pols.items():
        rows, traj = run(pol, args.episodes, record_first=True)
        rows_all[name], trajs[name], summary[name] = rows, traj, summarize(rows)
        s = summary[name]
        print(f"{name:32s} success {s['success_rate']:.0%} | touchdown {s['touchdown_speed_mps']:.2f} m/s | "
              f"miss {s['miss_distance_m']:.2f} m | fuel {s['fuel_used_kg']:.0f} kg   (n={s['episodes']})")

    winds = [0, 4, 8, 12, 16, 20]
    sweep = {n: [] for n in pols}
    for w in winds:
        for name, pol in pols.items():
            rows, _ = run(pol, args.sweep_episodes, seed0=50_000, wind_max=float(w))
            sweep[name].append(float(np.mean([r["success"] for r in rows])))
    summary["wind_sweep"] = dict(wind_max_mps=winds, success_rate=sweep)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    plt.figure(figsize=(6, 4))
    for name, v in sweep.items():
        plt.plot(winds, np.array(v) * 100, "o-", label=name)
    plt.xlabel("max wind speed (m/s)"); plt.ylabel("landing success (%)"); plt.ylim(0, 105)
    plt.grid(alpha=0.3); plt.legend(); plt.title("Robustness to wind"); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "wind_sweep.png"), dpi=150); plt.close()

    fig, axs = plt.subplots(1, len(pols), figsize=(5.2 * len(pols), 5), squeeze=False)
    for ax, (name, rows) in zip(axs[0], rows_all.items()):
        for r in rows:
            ax.plot(r["x"], r["y"], "o", color="tab:green" if r["success"] else "tab:red", ms=4, alpha=0.7)
        ax.add_patch(plt.Circle((0, 0), PAD_RADIUS, fill=False, lw=2))
        lim = max(15, 1.1 * max(r["miss"] for r in rows)); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_title(f"{name}\ntouchdown points (green = landed)"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "landing_scatter.png"), dpi=150); plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for name, tj in trajs.items():
        axs[0].plot(tj[:, 0], tj[:, 1], label=name); axs[1].plot(tj[:, 0], tj[:, 2])
        axs[2].plot(tj[:, 0], tj[:, 3] * 100)
    axs[0].set_ylabel("height (m)"); axs[1].set_ylabel("vertical speed (m/s)"); axs[2].set_ylabel("throttle (%)")
    axs[2].set_xlabel("time (s)"); axs[0].legend()
    for a in axs:
        a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "trajectory.png"), dpi=150); plt.close()
    print(f"wrote {args.out}/summary.json and 3 figures")


if __name__ == "__main__":
    main()
