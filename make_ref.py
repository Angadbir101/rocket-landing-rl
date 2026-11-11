import json
import numpy as np
from rocket_env import RocketLanderEnv
from baseline import SuicideBurnPD

pol, env, ok, misses, N = SuicideBurnPD(), RocketLanderEnv(difficulty=1.0), 0, [], 60
for ep in range(N):
    obs, _ = env.reset(seed=50_000 + ep)          # same seeds as make_curves.py
    done, info = False, {}
    while not done:
        obs, _, te, tr, info = env.step(pol(obs))
        done = te or tr
    ok += bool(info.get("success", False))
    if info.get("miss", -1) >= 0:
        misses.append(info["miss"])
ref = dict(success=ok / N, miss_m=float(np.median(misses)), n=N)
open("site/ref.js", "w").write("window.REF = " + json.dumps(ref) + ";\n")
print(ref)