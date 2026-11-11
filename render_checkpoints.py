import argparse, glob, os, re, subprocess, sys
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", default="models_story/ckpts")
ap.add_argument("--difficulty", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--every", type=int, default=1, help="use every Nth checkpoint")
ap.add_argument("--clips", default="clips")
ap.add_argument("--out", default="story.mp4")
args = ap.parse_args()

PAD = 5.0
t = min(max((args.difficulty - 0.1) / 0.9, 0.0), 1.0)
pad_tol = PAD * (2.0 - t)                      # same formula as rocket_env.py line 120
os.makedirs(args.clips, exist_ok=True)

step_of = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_*_steps.zip")), key=step_of)[::args.every]
jobs = [("untrained (random actions)", "random", os.path.join(args.clips, "clip_00000000.mp4"))]
jobs += [(f"PPO, {step_of(c)//1000}k steps", c, os.path.join(args.clips, f"clip_{step_of(c):08d}.mp4"))
         for c in ckpts]

done = []
for label, pol, out in jobs:
    if not os.path.exists(out):
        print(f"rendering {label} -> {out}")
        subprocess.run([sys.executable, "render_video.py", "--policy", pol, "--algo", "ppo",
                        "--seed", str(args.seed), "--difficulty", str(args.difficulty), "--out", out])
    if os.path.exists(out):
        done.append((label, out))
    else:
        print(f"  !! no file at {out}; check render_video.py's --out handling")
if not done:
    sys.exit("no clips rendered")

cap = cv2.VideoCapture(done[0][1])
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

def label_frame(f, text):
    f = cv2.resize(f, (W, H))
    cv2.rectangle(f, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.putText(f, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return f

for label, path in done:
    text = f"{label}  |  difficulty {args.difficulty:.1f}, pad tolerance {pad_tol:.1f} m"
    cap, last = cv2.VideoCapture(path), None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        last = label_frame(f, text)
        vw.write(last)
    cap.release()
    if last is not None:
        for _ in range(int(fps)):          # hold the final frame for 1 s
            vw.write(last)
vw.release()
print(f"wrote {args.out} ({len(done)} clips)")