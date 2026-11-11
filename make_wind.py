import json
import numpy as np
from stable_baselines3 import PPO
from baseline import SuicideBurnPD
from evaluate import run

WINDS, N = [0, 4, 8, 12, 16, 20], 50
MODELS = [("PPO, 3.0M steps", "models_story/ckpts/ckpt_2999952_steps.zip"),
          ("PPO, best checkpoint", "models_story/best/best_model.zip"),
          ("PPO, final (6M steps)", "models_story/ppo_final.zip")]

def sweep(pol):
    out = []
    for w in WINDS:
        rows, _ = run(pol, N, seed0=50_000, wind_max=float(w))
        out.append(round(100 * float(np.mean([r["success"] for r in rows])), 1))
    return out

series = []
for name, path in MODELS:
    model = PPO.load(path, device="cpu")
    series.append(dict(name=name, success=sweep(lambda obs, m=model: m.predict(obs, deterministic=True)[0])))
    print(name, series[-1]["success"])
series.append(dict(name="classical reference", success=sweep(SuicideBurnPD()), ref=True))
print("classical reference", series[-1]["success"])

open("site/wind.js", "w").write("window.WIND = " + json.dumps(dict(winds=WINDS, n=N, series=series)) + ";\n")
print("wrote site/wind.js")