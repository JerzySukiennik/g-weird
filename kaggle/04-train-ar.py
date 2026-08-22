"""Kaggle GPU cell: train the text-to-image transformer. T4x2, Internet ON.

Trains the caption BPE first, inside this kernel, on the captions that came out
of the encode step. Doing it here rather than at home means the vocabulary is
built from all 1.78M captions rather than the 90k sample the first draft used,
and it costs about a minute of CPU.

Sessions end voluntarily at SESSION_STEPS. Checkpoints are atomic and written
every 500 steps.
"""

import glob
import json
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

TOTAL_STEPS = 60000
SESSION_STEPS = 30000
BATCH, ACCUM = 16, 4

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob("/kaggle/input/**/gwtok_meta.json", recursive=True))
if not metas:
    raise SystemExit("brak tokenow — podepnij wyjscie gweird-encode")
prefix = metas[0][:-len("_meta.json")]
meta = json.load(open(metas[0]))
print(f"korpus: {prefix}\n  {meta['n']:,} par, {meta['per_image']} tokenow na obraz, "
      f"tokenizer z kroku {meta['vqvae_step']}", flush=True)

tok_path = f"{WORK}/text.json"
subprocess.run([sys.executable, "data/text_tokenizer.py",
                "--captions", f"{prefix}_captions.json",
                "--vocab", "8192", "--out", tok_path], check=True)

resume = []
for c in sorted(glob.glob("/kaggle/input/**/gweird.pt", recursive=True)):
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["cp", c, f"{OUT}/gweird.pt"], check=True)
    resume = ["--resume"]
    print(f"wznawiam z {c}", flush=True)
    break

subprocess.run([sys.executable, "train/train_ar.py",
                "--data", prefix, "--tokenizer", tok_path, "--out", OUT,
                "--steps", str(TOTAL_STEPS), "--max-steps", str(SESSION_STEPS),
                "--batch", str(BATCH), "--accum", str(ACCUM),
                "--workers", "4", "--log-every", "100",
                "--ckpt-every", "500"] + resume, check=True)
