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

# Raised after looking at the step-20000 reconstructions: compositions and
# colours survived, all fine detail did not. That matters more than it sounds,
# because reconstruction quality is the CEILING for the whole model — the
# transformer adds its own error on top, so whatever the decoder cannot express
# from 256 tokens, no amount of good token prediction will recover. The
# weirdness should come from the generator getting content wrong, not from the
# compressor blurring everything uniformly.
TOTAL_STEPS = 70000
# Short session on purpose: the discriminator adds a forward and a backward of a
# 2.8M net plus two more discriminator passes, so s/step is no longer the 0.70
# measured without it, and most of this week's quota is already spent. Better a
# session that ends cleanly than one killed at 90%.
SESSION_STEPS = 10000
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
