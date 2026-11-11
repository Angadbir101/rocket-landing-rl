"""
Failure breakdown for a trained landing policy.

    python diagnose.py --model models/best/best_model.zip --algo ppo --difficulty 0.1 --episodes 30

Runs the policy deterministically and, for every episode, checks each condition that rocket_env.py
requires for success, so we can see exactly which ones fail.  Also prints the vehicle's
thrust-to-weight numbers from the constants in rocket_env.py / rocket_model.py.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO, SAC

import rocket_model as rm
from rocket_env import RocketLanderEnv, G0, P0, A_E, T_VAC, THR_MIN

CONDS = ["no body hit", "touchdown vz < limit", "touchdown vxy < 2 m/s", "touchdown tilt < 10 deg",
         "final tilt < 15 deg", "inside pad tolerance", "final speed < 1 m/s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--difficulty", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true", help="use the final (strict) landing tolerances at any difficulty")
    args = ap.parse_args()

    p_sea = P0 * A_E
    t_min, t_max = THR_MIN * T_VAC - p_sea, T_VAC - p_sea
    print(f"dry mass {rm.DRY_MASS:.0f} kg, fuel {rm.FUEL_MASS:.0f} kg, pad radius {rm.PAD_RADIUS:.1f} m")
    print(f"sea-level thrust: min throttle {t_min:.0f} N, max throttle {t_max:.0f} N")
    print(f"weight: full {(rm.DRY_MASS + rm.FUEL_MASS) * G0:.0f} N, dry {rm.DRY_MASS * G0:.0f} N")
    print(f"thrust/weight at dry mass: min throttle {t_min / (rm.DRY_MASS * G0):.2f}, "
          f"max throttle {t_max / (rm.DRY_MASS * G0):.2f}\n")

    env = RocketLanderEnv(difficulty=args.difficulty)
    if args.strict:
        env.strict_tol = True
    model = (PPO if args.algo == "ppo" else SAC).load(args.model, device="cpu")

    fails = {c: 0 for c in CONDS}
    n_touch, n_success = 0, 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if ep == 0:
            print(f"tolerances in use: pad {getattr(env, 'tol_pad', rm.PAD_RADIUS):.2f} m, "
                  f"touchdown vertical speed {getattr(env, 'tol_vz', 3.0):.2f} m/s\n")
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        t = env.touch
        horiz = float(np.linalg.norm(env.data.qpos[:2]))
        if t is None:
            print(f"ep {ep:2d}: NO TOUCHDOWN  (steps {env.steps}, height {env._height():.1f} m, horiz {horiz:.1f} m, "
                  f"tilt {np.degrees(env._tilt()):.0f} deg)")
            continue
        n_touch += 1
        pad_r, vz_lim = getattr(env, "tol_pad", rm.PAD_RADIUS), getattr(env, "tol_vz", 3.0)
        ok = [not env.body_hit, -t["vz"] < vz_lim, t["vxy"] < 2.0, t["tilt"] < np.radians(10),
              env._tilt() < np.radians(15), horiz < pad_r, np.linalg.norm(env.data.qvel[:3]) < 1.0]
        for c, good in zip(CONDS, ok):
            fails[c] += (not good)
        n_success += bool(env.success)
        bad = [c for c, good in zip(CONDS, ok) if not good]
        print(f"ep {ep:2d}: {'SUCCESS' if env.success else 'fail   '} vz {-t['vz']:5.2f}  vxy {t['vxy']:5.2f}  "
              f"touch tilt {np.degrees(t['tilt']):5.1f}  final tilt {np.degrees(env._tilt()):5.1f}  "
              f"dist {horiz:5.1f} m  fuel {env.fuel:6.1f} kg  failed: {', '.join(bad) or '-'}")

    print(f"\ntouched down: {n_touch}/{args.episodes}   success: {n_success}/{args.episodes}")
    print("conditions failed (among touchdown episodes):")
    for c in CONDS:
        print(f"  {c:26s} {fails[c]}/{n_touch}")
    env.close()


if __name__ == "__main__":
    main()