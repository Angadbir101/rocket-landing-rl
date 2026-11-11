import argparse, glob, json, os, re
import numpy as np
from rocket_env import RocketLanderEnv, DT
from render_video import load_policy, hud, save

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", default="models_story/ckpts")
ap.add_argument("--final", default="models_story/ppo_final.zip")
ap.add_argument("--seeds", type=int, nargs="+", default=[2, 4, 6, 9])
ap.add_argument("--difficulty", type=float, default=1.0)
ap.add_argument("--every", type=int, default=2, help="use every Nth checkpoint")
ap.add_argument("--site", default="site")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
args = ap.parse_args()

CAMS = ["wide", "track"]
for c in CAMS:
    os.makedirs(os.path.join(args.site, "videos", c), exist_ok=True)

step_of = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_*_steps.zip")), key=step_of)[::args.every]
policies = [(0, "random")] + [(step_of(c), c) for c in ckpts]
if os.path.exists(args.final):
    policies.append((6_000_000, args.final))


def record_all(env, policy, seed, hold_s=1.5):
    env.record_trail = True
    obs, _ = env.reset(seed=seed)
    frames = {c: [] for c in CAMS}

    def grab(done):
        d = env.data
        status = "TOUCHDOWN" if env.phase == "settle" else ""
        if done:
            status = "LANDED" if env.success else "CRASHED"
        lines = [f"t = {env.steps * DT:5.1f} s   alt = {max(env._height(), 0):6.1f} m   vz = {d.qvel[2]:6.1f} m/s",
                 f"throttle {env.thr:4.0%}   fuel {env.fuel:5.1f} kg   tilt {np.degrees(env._tilt()):4.1f} deg", status]
        color = (120, 255, 120) if status == "LANDED" else (255, 110, 110) if status == "CRASHED" else (255, 255, 255)
        for c in CAMS:
            frames[c].append(hud(env.render_frame(c, args.width, args.height), lines, color))

    grab(False)
    done = False
    while not done:
        obs, _, te, tr, _ = env.step(policy(obs))
        done = te or tr
        grab(done)
    for c in CAMS:
        frames[c] += [frames[c][-1]] * int(hold_s / DT)
    return frames, bool(env.success)


manifest = []
for seed in args.seeds:
    for step, spec in policies:
        env = RocketLanderEnv(difficulty=args.difficulty)
        env.action_space.seed(seed)
        frames, ok = record_all(env, load_policy(spec, "ppo", env), seed)
        for c in CAMS:
            out = save(frames[c], os.path.join(args.site, "videos", c, f"s{seed}_{step:08d}.mp4"), fps=int(1 / DT))
            manifest.append(dict(seed=seed, step=step, cam=c, result="LANDED" if ok else "CRASHED",
                                 file=os.path.relpath(out, args.site).replace("\\", "/")))
        print(f"seed {seed}  step {step:>8}  {'LANDED' if ok else 'CRASHED'}")

with open(os.path.join(args.site, "manifest.js"), "w") as f:
    f.write("window.CLIPS = " + json.dumps(manifest, indent=1) + ";\n")
print(f"wrote {len(manifest)} clips and manifest.js in {args.site}/")