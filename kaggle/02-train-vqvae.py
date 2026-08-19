"""Kaggle GPU cell: train the VQ-VAE. Accelerator T4x2, Internet ON.

Sessions end voluntarily at SESSION_STEPS rather than running to the session
cap. G-Images lost 8h11m to a run killed at step 14800 of 15000, and the lesson
generalises: finish while healthy, let the next session resume.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

TOTAL_STEPS = 30000
SESSION_STEPS = 20000
BATCH = 16

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

# Recursive: a Dataset mounts at /kaggle/input/<slug>/ but a notebook attached
# through kernel_sources sits two levels deeper, under notebooks/<owner>/<slug>/.
metas = sorted(glob.glob("/kaggle/input/**/gweird_meta.json", recursive=True))
if not metas:
    raise SystemExit("nie znaleziono shardow — podepnij wyjscia gweird-prep-*")
prefixes = [m[:-len("_meta.json")] for m in metas]
print(f"{len(prefixes)} shardow:", *prefixes, sep="\n  ", flush=True)

resume = []
for c in sorted(glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)):
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["cp", c, f"{OUT}/vqvae.pt"], check=True)
    resume = ["--resume"]
    print(f"wznawiam z {c}", flush=True)
    break

subprocess.run([sys.executable, "train/train_vqvae.py",
                "--data", *prefixes, "--out", OUT,
                "--steps", str(TOTAL_STEPS), "--max-steps", str(SESSION_STEPS),
                "--batch", str(BATCH), "--workers", "2",
                "--log-every", "100", "--ckpt-every", "500"] + resume, check=True)
