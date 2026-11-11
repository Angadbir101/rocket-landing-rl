
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from rocket_env import RocketLanderEnv, DT


def load_policy(spec, algo, env):
    if spec == "baseline":
        from baseline import SuicideBurnPD
        return SuicideBurnPD()
    if spec == "random":
        return lambda obs: env.action_space.sample()
    from stable_baselines3 import PPO, SAC
    model = {"ppo": PPO, "sac": SAC}[algo].load(spec, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def hud(img, lines, color=(255, 255, 255)):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        dr.text((11, 9 + 14 * i), line, fill=(0, 0, 0))
        dr.text((10, 8 + 14 * i), line, fill=color)
    return np.asarray(im)


def record(env, policy, seed, width, height, hold_s=2.0):
    env.record_trail = True
    obs, _ = env.reset(seed=seed)
    frames, done, info = [], False, {}

    def grab():
        d = env.data
        status = ""
        if env.phase == "settle":
            status = "TOUCHDOWN"
        if done:
            status = "LANDED" if env.success else "CRASHED"
        lines = [f"t = {env.steps * DT:5.1f} s   alt = {max(env._height(), 0):6.1f} m   vz = {d.qvel[2]:6.1f} m/s",
                 f"throttle {env.thr:4.0%}   fuel {env.fuel:5.1f} kg   tilt {np.degrees(env._tilt()):4.1f} deg", status]
        wide = env.render_frame("wide", width, height)
        near = env.render_frame("track", width, height)
        frames.append(hud(np.concatenate([wide, near], axis=1), lines,
                          (120, 255, 120) if status == "LANDED" else (255, 110, 110) if status == "CRASHED" else (255, 255, 255)))

    grab()
    while not done:
        obs, _, te, tr, info = env.step(policy(obs))
        done = te or tr
        grab()
    for _ in range(int(hold_s / DT)):          # hold the final pose
        frames.append(frames[-1])
    return frames, env.success


def save(frames, out, fps):
    if out.endswith(".mp4"):
        try:
            import imageio
            with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
                for f in frames:
                    w.append_data(f)
            return out
        except Exception as exc:                # no ffmpeg -> fall back to gif
            print(f"mp4 failed ({exc}); writing gif instead")
            out = os.path.splitext(out)[0] + ".gif"
    ims = [Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)) for f in frames[::2]]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(2000 * DT), loop=0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="baseline", help="baseline | random | path/to/model.zip")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--out", default="landing.mp4")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    env = RocketLanderEnv(difficulty=args.difficulty)
    frames, ok = record(env, load_policy(args.policy, args.algo, env), args.seed, args.width, args.height)
    path = save(frames, args.out, fps=int(1 / DT))
    print(f"saved {path}  ({'LANDED' if ok else 'CRASHED'}, {len(frames)} frames)")
