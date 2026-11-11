
import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces

import rocket_model as rm

G0 = 9.80665
P0, RHO0, H_SCALE = 101325.0, 1.225, 8500.0
T_VAC, ISP, A_E = 28000.0, 300.0, 0.05      # vacuum thrust (N), vacuum Isp (s), nozzle exit area (m^2)
THR_MIN, THR_TAU = 0.4, 0.1                  # minimum throttle, engine lag (s)
GIMBAL_MAX = np.radians(8.0)
CD, A_REF = 0.8, 0.8                        # axial drag coefficient / frontal area (m^2)
CD_LAT, A_SIDE, Z_CP = 1.0, 8.0, 1.0          # cross-flow drag coefficient / side area (m^2) / centre of pressure above CoM (m)
DT, N_SUB = 0.05, 5                          # control period, physics sub-steps (0.01 s each)
SETTLE_STEPS = 24                            # 1.2 s after touchdown
Z_FEET = rm.FOOT_DROP


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rotvec_quat(rv):
    ang = np.linalg.norm(rv)
    if ang < 1e-9:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * rv / ang])


class RocketLanderEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / DT)}

    def __init__(self, difficulty=1.0, randomize=True, wind_max=5.0, max_steps=600):
        super().__init__()
        self.difficulty, self.randomize, self.wind_max, self.max_steps = difficulty, randomize, wind_max, max_steps
        xml, assets = rm.build_mjcf(GIMBAL_MAX, timestep=DT / N_SUB)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        m = self.model
        nm = lambda kind, name: mujoco.mj_name2id(m, kind, name)
        self.hull = nm(mujoco.mjtObj.mjOBJ_BODY, "rocket")
        self.eng = nm(mujoco.mjtObj.mjOBJ_BODY, "engine")
        self.floor = nm(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = {nm(mujoco.mjtObj.mjOBJ_GEOM, f"foot_{k}") for k in range(4)}
        self.g_plume = nm(mujoco.mjtObj.mjOBJ_GEOM, "plume")
        self.g_core = nm(mujoco.mjtObj.mjOBJ_GEOM, "core")
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (19,), np.float32)
        self._renderer, self._rsize = None, None
        self.record_trail, self.trail = False, []

    # ------------------------------------------------------------------ curriculum
    def set_difficulty(self, d):
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    # ------------------------------------------------------------------ helpers
    def _rot(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return R.reshape(3, 3)

    def _tilt(self):
        return float(np.arccos(np.clip(self._rot()[2, 2], -1.0, 1.0)))

    def _height(self):
        return self.data.qpos[2] - Z_FEET

    def _set_mass(self):
        total = self.dry_mass + self.fuel
        m, hull = self.model, self.hull
        m.body_mass[hull] = total - rm.ENGINE_MASS
        ixx = total * (3 * rm.RADIUS ** 2 + rm.HEIGHT ** 2) / 12.0
        m.body_inertia[hull] = [ixx, ixx, 0.5 * total * rm.RADIUS ** 2]
        m.body_subtreemass[self.hull] = total
        m.body_subtreemass[0] = total

    def _shaping(self):
        d = self.data
        pos, vel, w = d.qpos[:3], d.qvel[:3], d.qvel[3:6]
        h = max(self._height(), 0.0)
        v_ref = np.array([np.clip(-0.12 * pos[0], -4, 4), np.clip(-0.12 * pos[1], -4, 4),
                          -min(np.sqrt(2 * 3.0 * h) * 0.9 + 0.3, 30.0)])   # glide-slope descent profile
        pos_w = 1.0 + 4.0 * np.exp(-h / 40.0)          # a miss matters more near the ground
        return -(100.0 * np.linalg.norm([pos[0], pos[1]]) / 200.0 * pos_w + 100.0 * h / 200.0
                 + 100.0 * np.linalg.norm(vel - v_ref) / 30.0
                 + 50.0 * self._tilt() + 10.0 * np.linalg.norm(w))

    def _obs(self):
        d = self.data
        return np.concatenate([
            [d.qpos[0] / 50.0, d.qpos[1] / 50.0, self._height() / 200.0],
            d.qvel[:3] / np.array([10.0, 10.0, 30.0]),
            self._rot().ravel(), d.qvel[3:6], [self.fuel / rm.FUEL_MASS]]).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r, dif, m, d = self.np_random, self.difficulty, self.model, self.data
        lerp = lambda a, b: a + (b - a) * dif
        U = r.uniform
        t_tol = 1.0 if getattr(self, "strict_tol", False) else float(np.clip((dif - 0.1) / 0.9, 0.0, 1.0))
        self.tol_pad, self.tol_vz = rm.PAD_RADIUS * (2.0 - t_tol), 4.0 - t_tol   # loose at 0.1, strict at 1.0

        self.z0 = U(lerp(100, 200), lerp(120, 300))
        rad, ang = U(0, 1) * lerp(5, 35), U(0, 2 * np.pi)
        pos = np.array([rad * np.cos(ang), rad * np.sin(ang), self.z0])
        vl = U(-1, 1, 2) * lerp(0.5, 3.0)
        vel = np.array([vl[0], vl[1], -U(lerp(8, 20), lerp(12, 30))])
        tilt = U(-1, 1, 3) * lerp(0.02, 0.09)
        tilt[2] = 0.0
        yaw = U(-np.pi, np.pi)
        q = _qmul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), _rotvec_quat(tilt))

        self.dry_mass, self.fuel0 = rm.DRY_MASS, rm.FUEL_MASS
        self.t_scale, self.isp, self.cd = 1.0, ISP, CD
        if self.randomize:                                    # domain randomisation
            self.dry_mass *= U(0.9, 1.1)
            self.fuel0 *= U(0.85, 1.0)
            self.t_scale, self.isp, self.cd = U(0.95, 1.05), ISP * U(0.97, 1.03), CD * U(0.8, 1.2)
        self.wind = np.append(U(-1, 1, 2) * self.wind_max * dif, 0.0)

        mujoco.mj_resetData(m, d)
        d.qpos[:3], d.qpos[3:7] = pos, q
        d.qvel[:3], d.qvel[3:6] = vel, U(-1, 1, 3) * lerp(0.01, 0.05)
        d.ctrl[:] = 0.0
        self.fuel, self.thr = self.fuel0, 0.0
        self._set_mass()
        mujoco.mj_forward(m, d)

        self.steps, self.phase, self.settle_left, self.body_hit = 0, "flight", 0, False
        self.touch, self.success, self.trail = None, False, []
        self.prev_shaping = self._shaping()
        return self._obs(), {"success": False, "fuel": self.fuel}

    def _apply_forces(self, h_sub):
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        alt = max(d.qpos[2] - Z_FEET, 0.0)
        pa, rho = P0 * np.exp(-alt / H_SCALE), RHO0 * np.exp(-alt / H_SCALE)
        zero = np.zeros(3)
        if self.thr > 0 and self.fuel > 0:
            thrust = max(self.thr * T_VAC * self.t_scale - pa * A_E, 0.0)   # pressure-dependent thrust
            axis = d.xmat[self.eng].reshape(3, 3)[:, 2]
            mujoco.mj_applyFT(m, d, thrust * axis, zero, d.xpos[self.eng], self.eng, d.qfrc_applied)
            self.fuel = max(self.fuel - self.thr * T_VAC * self.t_scale / (self.isp * G0) * h_sub, 0.0)
        R = self._rot()
        vb = R.T @ (d.qvel[:3] - self.wind)                       # air-relative velocity in the body frame
        f_body = np.array([-0.5 * rho * CD_LAT * A_SIDE * np.linalg.norm(vb[:2]) * vb[0],
                           -0.5 * rho * CD_LAT * A_SIDE * np.linalg.norm(vb[:2]) * vb[1],
                           -0.5 * rho * self.cd * A_REF * abs(vb[2]) * vb[2]])
        cp = d.xipos[self.hull] + R @ np.array([0.0, 0.0, Z_CP])  # aerodynamic force acts at the centre of pressure
        mujoco.mj_applyFT(m, d, R @ f_body, zero, cp, self.hull, d.qfrc_applied)

    def _ground_contacts(self):
        foot = body = False
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            other = c.geom2 if c.geom1 == self.floor else c.geom1 if c.geom2 == self.floor else -1
            if other >= 0:
                if other in self.feet:
                    foot = True
                else:
                    body = True
        return foot, body

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.phase == "flight":
            thr_cmd = THR_MIN + (1.0 - THR_MIN) * (a[0] + 1.0) / 2.0
            if self.fuel <= 0:
                thr_cmd = 0.0
            self.thr += (1.0 - np.exp(-DT / THR_TAU)) * (thr_cmd - self.thr)
            d.ctrl[0], d.ctrl[1] = a[1] * GIMBAL_MAX, a[2] * GIMBAL_MAX
        else:                                                  # engine cut after touchdown
            self.thr, d.ctrl[:] = 0.0, 0.0
        self._set_mass()

        for _ in range(N_SUB):
            self._apply_forces(DT / N_SUB)
            v_prev = d.qvel[:3].copy()
            mujoco.mj_step(m, d)
            foot, body = self._ground_contacts()
            if self.phase == "flight" and (foot or body):
                self.phase, self.settle_left, self.thr = "settle", SETTLE_STEPS, 0.0
                d.ctrl[:] = 0.0
                self.touch = dict(vz=v_prev[2], vxy=float(np.linalg.norm(v_prev[:2])), tilt=self._tilt(),
                                  xy=d.qpos[:2].copy(), t=self.steps * DT)
            if body and self.phase == "settle":
                self.body_hit = True
        self.steps += 1
        if self.record_trail:
            self.trail.append(d.qpos[:3].copy())

        terminated, reward = False, 0.0
        info = {"success": False, "miss": -1.0, "touch_vz": -1.0, "touch_vxy": -1.0}
        horiz = float(np.linalg.norm(d.qpos[:2]))
        if self.phase == "settle":
            self.settle_left -= 1
            if self.body_hit or self.settle_left <= 0:
                terminated = True
                t = self.touch
                pad_r = getattr(self, "tol_pad", rm.PAD_RADIUS)   # curriculum hooks, set in reset()
                vz_lim = getattr(self, "tol_vz", 3.0)
                ok = (not self.body_hit and -t["vz"] < vz_lim and t["vxy"] < 2.0 and t["tilt"] < np.radians(10)
                      and self._tilt() < np.radians(15) and horiz < pad_r
                      and np.linalg.norm(d.qvel[:3]) < 1.0)
                self.success = bool(ok)
                info.update(miss=float(horiz), touch_vz=float(-t["vz"]), touch_vxy=float(t["vxy"]))
                if ok:
                    reward = 100.0 + 20.0 * self.fuel / rm.FUEL_MASS + 40.0 * float(np.exp(-horiz / rm.PAD_RADIUS))
                else:
                    # graded failure: each missed condition costs in proportion to how badly it was missed
                    # (e = 1 means exactly at the limit, so a near-miss is cheap and a crash is expensive)
                    e = np.array([-t["vz"] / vz_lim, t["vxy"] / 2.0, t["tilt"] / np.radians(10),
                                  self._tilt() / np.radians(15), horiz / pad_r,
                                  np.linalg.norm(d.qvel[:3]) / 1.0])
                    miss = float(np.sum(np.array([40.0, 20.0, 15.0, 15.0, 30.0, 15.0]) * np.clip(e - 1.0, 0.0, 2.0)))
                    miss += 15.0 * float(np.clip(horiz / rm.PAD_RADIUS - 1.0, 0.0, 4.0))
                    reward = -min(100.0, 15.0 + miss + (30.0 if self.body_hit else 0.0))
        else:
            shaping = self._shaping()
            reward = shaping - self.prev_shaping - 0.05 * self.thr
            self.prev_shaping = shaping
            if horiz > 300 or self._height() > 500 or self._tilt() > np.radians(90):
                terminated, reward = True, reward - 100.0
        info.update(success=self.success, fuel=self.fuel)
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ visuals
    def sync_visuals(self):
        """Size the exhaust plume from throttle and ambient pressure (plume grows in thin air)."""
        m = self.model
        pa = P0 * np.exp(-max(self._height(), 0.0) / H_SCALE)
        on = self.thr > 0.01 and self.fuel > 0
        L = (3.0 + 7.0 * self.thr) * (1.0 + 0.9 * (1.0 - pa / P0)) if on else 0.01
        r = 0.30 * (0.6 + 0.4 * self.thr) * (1.0 + 0.5 * (1.0 - pa / P0)) if on else 0.01
        m.geom_size[self.g_plume] = [r, r, L / 2]
        m.geom_pos[self.g_plume] = [0, 0, -(rm.BELL_LEN + L / 2)]
        m.geom_size[self.g_core] = [0.5 * r, 0.5 * r, 0.3 * L]
        m.geom_pos[self.g_core] = [0, 0, -(rm.BELL_LEN + 0.3 * L)]

    def render_frame(self, view="track", width=640, height=480):
        """RGB frame from a tracking camera ('track') or a wide mission camera ('wide')."""
        if self._renderer is None or self._rsize != (width, height):
            self._renderer, self._rsize = mujoco.Renderer(self.model, height, width), (width, height)
        p, h = self.data.qpos[:3], max(self._height(), 0.0)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if view == "track":
            cam.lookat[:], cam.distance = p, float(np.clip(16 + 0.08 * h, 16, 45))
            cam.azimuth, cam.elevation = 140.0, -12.0
        else:
            cam.lookat[:] = [0.0, 0.0, 0.5 * p[2] + 4.0]
            cam.distance, cam.azimuth, cam.elevation = 35.0 + 1.25 * h, 120.0, -8.0
        self.sync_visuals()
        self._renderer.update_scene(self.data, camera=cam)
        sc = self._renderer.scene
        for a_, b_ in zip(self.trail[:-1:3], self.trail[3::3]):          # flight-path trail
            if sc.ngeom >= sc.maxgeom:
                break
            g = sc.geoms[sc.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array([0.2, 0.6, 1.0, 0.9], dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.12 if view == "track" else 0.5, a_, b_)
            sc.ngeom += 1
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
