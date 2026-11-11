"""
Train a landing policy (PPO or SAC) with a TINTIN-style curriculum + domain randomisation.

    python train.py --algo ppo --steps 5000000
    python train.py --algo sac --steps 1500000 --envs 4
    tensorboard --logdir logs

Curriculum: training starts at difficulty 0.1 (low start, tiny offsets, no wind) and the environment
difficulty is raised by 0.1 whenever the rolling success rate over the last 100 episodes reaches
--promote (default 70 %).  Difficulty 1.0 = full deployment envelope (200-300 m, +-35 m offset, 5 m/s wind).
Every episode also randomises dry mass, fuel, thrust, Isp and drag coefficient.
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from reward_scale import REWARD_SCALE, ScaleReward
from rocket_env import RocketLanderEnv


class CurriculumCallback(BaseCallback):
    def __init__(self, start=0.1, step=0.1, promote=0.7, window=100, check_every=5000):
        super().__init__()
        self.dif, self.step_, self.promote, self.window, self.check_every = start, step, promote, window, check_every

    def _on_training_start(self):
        self.training_env.env_method("set_difficulty", self.dif)

    def _on_step(self):
        if self.n_calls % max(self.check_every // self.training_env.num_envs, 1):
            return True
        buf = [e["success"] for e in self.model.ep_info_buffer if "success" in e]
        if len(buf) >= 50:
            rate = float(np.mean(buf[-self.window:]))
            self.logger.record("curriculum/success_rate", rate)
            if rate >= self.promote and self.dif < 1.0:
                self.dif = min(1.0, round(self.dif + self.step_, 3))
                self.training_env.env_method("set_difficulty", self.dif)
                self.model.ep_info_buffer.clear()          # judge the new level on fresh episodes
                print(f"[curriculum] success {rate:.0%} -> difficulty {self.dif:.1f}  (step {self.num_timesteps})")
        self.logger.record("curriculum/difficulty", self.dif)
        return True


class PrecisionLogger(BaseCallback):
    """Log how close landings are, from the episode-end info (episodes with no touchdown are skipped)."""
    def __init__(self, every=5000):
        super().__init__()
        self.every = every

    def _on_step(self):
        if self.n_calls % max(self.every // self.training_env.num_envs, 1):
            return True
        b = [e for e in self.model.ep_info_buffer if e.get("miss", -1.0) >= 0.0]
        if len(b) >= 20:
            miss = np.array([e["miss"] for e in b])
            self.logger.record("precision/miss_m", float(miss.mean()))
            self.logger.record("precision/miss_median_m", float(np.median(miss)))
            self.logger.record("precision/touch_vz", float(np.mean([e["touch_vz"] for e in b])))
            self.logger.record("precision/touch_vxy", float(np.mean([e["touch_vxy"] for e in b])))
            self.logger.record("precision/n_touchdowns", len(b))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--envs", type=int, default=max(os.cpu_count() or 4, 4))
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--promote", type=float, default=0.7)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cpu for PPO, auto for SAC)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-model", default=None, help="fine-tune from this saved model (.zip)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    start = 1.0 if args.no_curriculum else 0.1
    mk = dict(env_kwargs=dict(difficulty=start), seed=args.seed,
              monitor_kwargs=dict(info_keywords=("success", "miss", "touch_vz", "touch_vxy")),
              wrapper_class=ScaleReward, wrapper_kwargs=dict(scale=REWARD_SCALE))
    venv = make_vec_env(RocketLanderEnv, n_envs=args.envs,
                        vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv, **mk)
    eval_env = make_vec_env(RocketLanderEnv, n_envs=1, seed=args.seed + 999, env_kwargs=dict(difficulty=1.0))

    try:
        import tensorboard  # noqa: F401
        tb = "logs"
    except ImportError:
        tb = None
    arch = dict(net_arch=[256, 256])
    if args.algo == "ppo":
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, policy_kwargs=arch, tensorboard_log=tb,
                    device=args.device or "cpu", seed=args.seed, verbose=1)
    else:
        model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=1_000_000, batch_size=256, tau=0.005,
                    gamma=0.995, train_freq=1, gradient_steps=1, learning_starts=10_000, ent_coef="auto",
                    policy_kwargs=arch, tensorboard_log=tb, device=args.device or "auto", seed=args.seed, verbose=1)

    if args.init_model:
        cls = PPO if args.algo == "ppo" else SAC
        model = cls.load(args.init_model, env=venv, device=args.device or ("cpu" if args.algo == "ppo" else "auto"),
                         tensorboard_log=tb)
        print(f"fine-tuning from {args.init_model}")

    callbacks = [EvalCallback(eval_env, best_model_save_path=os.path.join(args.out, "best"),
                              eval_freq=max(25_000 // args.envs, 1), n_eval_episodes=20, deterministic=True)]
    if not args.no_curriculum:
        callbacks.insert(0, CurriculumCallback(promote=args.promote))
    callbacks.append(PrecisionLogger())
    callbacks.append(CheckpointCallback(save_freq=max(250_000 // args.envs, 1),
                                        save_path=os.path.join(args.out, "ckpts"), name_prefix="ckpt"))
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    path = os.path.join(args.out, f"{args.algo}_final")
    model.save(path)
    print(f"saved {path}.zip   (best on difficulty-1.0 eval: {args.out}/best/best_model.zip)")


if __name__ == "__main__":
    main()
