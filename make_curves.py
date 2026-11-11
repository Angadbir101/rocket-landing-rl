import glob, json, os, re
import numpy as np
from stable_baselines3 import PPO
from rocket_env import RocketLanderEnv

EPISODES, SITE = 60, "site"
step_of = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
ckpts = sorted(glob.glob("models_story/ckpts/ckpt_*_steps.zip"), key=step_of)
models = [(0, None)] + [(step_of(c), c) for c in ckpts]
if os.path.exists("models_story/ppo_final.zip"):
    models.append((6_000_000, "models_story/ppo_final.zip"))

succ, miss = [], []
for step, path in models:
    model = PPO.load(path, device="cpu") if path else None
    env = RocketLanderEnv(difficulty=1.0)
    env.action_space.seed(0)
    ok, misses = 0, []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=50_000 + ep)
        done, info = False, {}
        while not done:
            a = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
            obs, _, te, tr, info = env.step(a)
            done = te or tr
        ok += bool(info.get("success", False))
        if info.get("miss", -1) >= 0:
            misses.append(info["miss"])
    succ.append([step, ok / EPISODES])
    miss.append([step, float(np.median(misses)) if misses else 0.0])
    print(f"step {step:>8}  success {ok}/{EPISODES}  median miss {miss[-1][1]:.1f} m")

with open(os.path.join(SITE, "curves.js"), "w") as f:
    f.write("window.CURVES = " + json.dumps(dict(success_rate=succ, miss_m=miss)) + ";\n")
print("wrote", os.path.join(SITE, "curves.js"))