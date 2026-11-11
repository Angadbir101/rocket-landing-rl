"""
Kestrel-1: our own 3D rocket model for MuJoCo, generated procedurally (no external assets).

  * ogive nose cone, fuselage with boat-tail and a recessed-nozzle engine bell -> revolved STL meshes
  * 4 deployable landing legs (struts, braces, foot pads with real collision)
  * 4 grid fins, painted bands
  * 2-DoF gimbal: two hinge joints driven by position servos (the engine is a real child body)
  * exhaust plume = two ellipsoids whose size the environment updates from throttle/altitude
  * launch-pad scene: sky gradient, checker ground, painted landing pad (5 m radius)

Body frame: origin at the centre of mass, +z along the rocket axis (nose up).

Export for inspection in the MuJoCo viewer:
    python rocket_model.py --export model_out
    python -m mujoco.viewer --mjcf=model_out/rocket_lander.xml
"""
import argparse
import os
import struct

import numpy as np

# ------------------------------------------------------------------ vehicle geometry (m, kg)
DRY_MASS = 1000.0          # includes engine
FUEL_MASS = 300.0
ENGINE_MASS = 40.0         # gimballed part (child body)
RADIUS = 0.5
Z_TOP = 6.0                # nose tip above CoM
Z_BASE = -4.0              # base plane / gimbal pivot below CoM
HEIGHT = Z_TOP - Z_BASE    # 10 m
NOSE_LEN = 1.8
LEG_HUB_R, LEG_HUB_Z = 0.5, -2.4
FOOT_R, FOOT_Z, FOOT_RAD = 2.8, -5.3, 0.25
FOOT_DROP = -FOOT_Z + FOOT_RAD     # CoM height above ground when the feet touch (5.55 m)
BELL_LEN = 1.05            # engine bell below the pivot
PAD_RADIUS = 5.0


# ------------------------------------------------------------------ mesh generation
def revolve(profile, n=56):
    """Revolve a closed (r, z) profile (first and last point on the axis) into a watertight mesh."""
    prof = np.asarray(profile, dtype=float)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    verts, idx = [], {}

    def vid(i, j):
        r = prof[i, 0]
        key = (i, 0 if r == 0 else j)
        if key not in idx:
            idx[key] = len(verts)
            verts.append([r * np.cos(th[j]), r * np.sin(th[j]), prof[i, 1]])
        return idx[key]

    faces = []
    for i in range(len(prof) - 1):
        r0, r1 = prof[i, 0], prof[i + 1, 0]
        if r0 == 0 and r1 == 0:
            continue
        for j in range(n):
            k = (j + 1) % n
            a, b, c, d = vid(i, j), vid(i, k), vid(i + 1, k), vid(i + 1, j)
            if r0 == 0:
                faces.append((a, c, d))
            elif r1 == 0:
                faces.append((a, b, d))
            else:
                faces += [(a, b, c), (a, c, d)]
    v, f = np.array(verts), np.array(faces)
    vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0
    if vol < 0:                       # make triangle winding point outwards
        f = f[:, ::-1]
    return v, f


def stl_bytes(v, f):
    tri = v[f].astype("<f4")
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    return b"\0" * 80 + struct.pack("<I", len(f)) + rec.tobytes()


def make_meshes():
    R, L = RADIUS, NOSE_LEN
    z_nose_base = Z_TOP - L
    rho = (R * R + L * L) / (2 * R)                        # tangent-ogive radius
    xs = np.linspace(0.0, L, 18)
    ogive = [(max(np.sqrt(rho ** 2 - (L - x) ** 2) + R - rho, 0.0), z_nose_base + L - x) for x in xs]
    ogive[0] = (0.0, Z_TOP)
    nose = ogive + [(0.0, z_nose_base)]
    tube = [(0.0, z_nose_base), (R, z_nose_base), (R, -3.6), (0.43, Z_BASE), (0.0, Z_BASE)]
    # engine bell in the gimbal-pivot frame (z = 0 at pivot, pointing down); recessed inner nozzle
    bell = [(0.0, 0.0), (0.17, 0.0), (0.17, -0.10), (0.22, -0.25), (0.30, -0.50), (0.37, -0.75),
            (0.42, -0.95), (0.44, -BELL_LEN), (0.40, -BELL_LEN), (0.30, -0.95), (0.16, -0.80), (0.0, -0.70)]
    return {"nose": revolve(nose), "tube": revolve(tube), "bell": revolve(bell, n=40)}


# ------------------------------------------------------------------ MJCF
def _f(*xs):
    return " ".join(f"{x:.4f}" for x in xs)


def build_mjcf(gimbal_max, timestep=0.01, servo_kp=10000.0, servo_kv=500.0):
    """Return (xml_string, assets_dict) for the full landing scene."""
    meshes = make_meshes()
    assets = {f"{k}.stl": stl_bytes(*v) for k, v in meshes.items()}

    total = DRY_MASS + FUEL_MASS
    ixx = total * (3 * RADIUS ** 2 + HEIGHT ** 2) / 12.0
    izz = 0.5 * total * RADIUS ** 2
    g_rng = 1.3 * gimbal_max

    legs, fins = [], []
    for k in range(4):
        a = np.pi / 4 + k * np.pi / 2
        u = np.array([np.cos(a), np.sin(a), 0.0])
        hub = LEG_HUB_R * u + np.array([0, 0, LEG_HUB_Z])
        foot = FOOT_R * u + np.array([0, 0, FOOT_Z])
        brace_from = LEG_HUB_R * u + np.array([0, 0, -3.7])
        brace_to = hub + 0.68 * (foot - hub)
        legs.append(f'''
      <geom type="capsule" fromto="{_f(*hub, *foot)}" size="0.07" material="steel" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="{_f(*brace_from, *brace_to)}" size="0.045" material="steel" contype="0" conaffinity="0"/>
      <geom name="foot_{k}" type="sphere" size="{FOOT_RAD}" pos="{_f(*foot)}" material="dark"
            friction="1.2 0.01 0.001" contype="1" conaffinity="1"/>''')
        fa = k * np.pi / 2
        fins.append(f'''
      <geom type="box" size="0.30 0.04 0.22" pos="{_f(0.62 * np.cos(fa), 0.62 * np.sin(fa), 3.55)}"
            euler="0 0 {fa:.4f}" material="dark" contype="0" conaffinity="0"/>''')

    xml = f'''
<mujoco model="kestrel_lander">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 -9.81" density="0" viscosity="0"/>
  <size memory="32M"/>
  <statistic center="0 0 20" extent="60"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <rgba haze="0.62 0.74 0.9 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.66 0.9" rgb2="0.03 0.04 0.1" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.36 0.27" rgb2="0.24 0.30 0.22"
             width="512" height="512" mark="edge" markrgb="0.34 0.40 0.31"/>
    <material name="ground" texture="grid" texrepeat="0.1 0.1" texuniform="true" reflectance="0.03"/>
    <material name="hull" rgba="0.94 0.94 0.96 1" specular="0.6" shininess="0.7"/>
    <material name="nosec" rgba="0.12 0.12 0.14 1" specular="0.5" shininess="0.6"/>
    <material name="steel" rgba="0.55 0.57 0.60 1" specular="0.7" shininess="0.8"/>
    <material name="dark" rgba="0.14 0.14 0.16 1" specular="0.3"/>
    <material name="red" rgba="0.80 0.12 0.10 1"/>
    <material name="padc" rgba="0.22 0.23 0.26 1"/>
    <material name="padw" rgba="0.92 0.92 0.92 1"/>
    <material name="plume" rgba="1.0 0.55 0.12 0.55" emission="0.8"/>
    <material name="core" rgba="1.0 0.92 0.65 0.85" emission="1"/>
    <mesh name="nose" file="nose.stl"/>
    <mesh name="tube" file="tube.stl"/>
    <mesh name="bell" file="bell.stl"/>
  </asset>
  <worldbody>
    <light directional="true" pos="60 -80 400" dir="-0.35 0.45 -1" diffuse="1 0.96 0.9" specular="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="800 800 0.1" material="ground" friction="1.0 0.005 0.0001"/>
    <geom type="cylinder" size="{PAD_RADIUS} 0.03" pos="0 0 0" material="padc" contype="0" conaffinity="0"/>
    <geom type="cylinder" size="4.5 0.03" pos="0 0 0.006" material="padw" contype="0" conaffinity="0"/>
    <geom type="cylinder" size="4.2 0.03" pos="0 0 0.012" material="padc" contype="0" conaffinity="0"/>
    <geom type="box" size="3.2 0.18 0.03" pos="0 0 0.018" material="padw" contype="0" conaffinity="0"/>
    <geom type="box" size="0.18 3.2 0.03" pos="0 0 0.018" material="padw" contype="0" conaffinity="0"/>

    <body name="rocket" pos="0 0 100">
      <freejoint name="free"/>
      <inertial pos="0 0 0" mass="{total - ENGINE_MASS}" diaginertia="{_f(ixx, ixx, izz)}"/>
      <geom type="mesh" mesh="tube" material="hull" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="nose" material="nosec" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.506 0.30" pos="0 0 3.55" material="dark" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.506 0.10" pos="0 0 1.0" material="red" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.506 0.45" pos="0 0 -3.2" material="dark" contype="0" conaffinity="0"/>
      {''.join(fins)}
      {''.join(legs)}
      <geom name="col_body" type="capsule" fromto="0 0 -3.6 0 0 4.2" size="{RADIUS}" contype="1" conaffinity="1"
            group="3" rgba="1 0 0 0"/>
      <body name="engine" pos="0 0 {Z_BASE}">
        <inertial pos="0 0 -0.45" mass="{ENGINE_MASS}" diaginertia="8 8 1"/>
        <joint name="tilt_x" type="hinge" axis="0 1 0" range="{-g_rng:.5f} {g_rng:.5f}" armature="0.5"/>
        <joint name="tilt_y" type="hinge" axis="-1 0 0" range="{-g_rng:.5f} {g_rng:.5f}" armature="0.5"/>
        <geom type="mesh" mesh="bell" material="steel" contype="0" conaffinity="0"/>
        <geom name="plume" type="ellipsoid" size="0.3 0.3 0.01" pos="0 0 -1.1" material="plume" contype="0" conaffinity="0"/>
        <geom name="core" type="ellipsoid" size="0.15 0.15 0.01" pos="0 0 -1.1" material="core" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_tilt_x" joint="tilt_x" kp="{servo_kp}" kv="{servo_kv}" ctrlrange="{-gimbal_max:.5f} {gimbal_max:.5f}"/>
    <position name="act_tilt_y" joint="tilt_y" kp="{servo_kp}" kv="{servo_kv}" ctrlrange="{-gimbal_max:.5f} {gimbal_max:.5f}"/>
  </actuator>
</mujoco>'''
    return xml, assets


def export_model(out_dir, gimbal_max=np.radians(8.0)):
    xml, assets = build_mjcf(gimbal_max)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in assets.items():
        with open(os.path.join(out_dir, name), "wb") as fh:
            fh.write(data)
    with open(os.path.join(out_dir, "rocket_lander.xml"), "w") as fh:
        fh.write(xml)
    print(f"wrote {out_dir}/rocket_lander.xml and {len(assets)} STL meshes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="model_out")
    export_model(ap.parse_args().export)
