"""
Classical baseline (the 'PID baseline' role in TINTIN): no learning, same observations as the RL policy.

  * throttle: track a suicide-burn style descent-speed profile  v_des(h) = -(0.85*sqrt(2*a*h) + 0.8)
  * guidance: lateral position/velocity error -> desired tilt of the thrust axis (clipped to ~8.5 deg)
  * gimbal:   PD attitude control on the body up-vector (sign convention matches rocket_env)
"""
import numpy as np

GIMBAL_MAX = np.radians(8.0)


class SuicideBurnPD:
    def __init__(self, a_ref=5.0, k_pos=0.02, k_vel=0.08, kp=1.5, kd=0.8, tilt_cmd_max=0.15):
        self.a_ref, self.k_pos, self.k_vel, self.kp, self.kd, self.tmax = a_ref, k_pos, k_vel, kp, kd, tilt_cmd_max

    def __call__(self, obs):
        pos = obs[0:3] * np.array([50.0, 50.0, 200.0])
        vel = obs[3:6] * np.array([10.0, 10.0, 30.0])
        R = obs[6:15].reshape(3, 3)
        w = obs[15:18]
        h = max(pos[2], 0.1)

        v_des = -(0.85 * np.sqrt(2.0 * self.a_ref * h) + 0.8)
        a_thr = np.clip((v_des - vel[2]) - 0.2, -1.0, 1.0)

        des = np.clip(-self.k_pos * pos[:2] - self.k_vel * vel[:2], -self.tmax, self.tmax)
        e = R[:2, 2] - des                               # world-frame tilt error of the body axis
        ex = R[0, 0] * e[0] + R[1, 0] * e[1]             # ... expressed along the body x / y axes,
        ey = R[0, 1] * e[0] + R[1, 1] * e[1]             # because the gimbal axes are body-fixed (yaw is random)
        alpha = self.kp * ex + self.kd * w[1]            # thrust tilt toward body +x
        beta = self.kp * ey - self.kd * w[0]             # thrust tilt toward body +y
        return np.array([a_thr, np.clip(alpha / GIMBAL_MAX, -1, 1), np.clip(beta / GIMBAL_MAX, -1, 1)],
                        dtype=np.float32)
