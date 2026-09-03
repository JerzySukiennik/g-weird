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
# The corpus was assembled across two kernel runs after the first was killed at
# 83%, so "the file exists" is not enough to trust it. A short file trains on a
# truncated corpus without complaining; a caption list of the wrong length pairs
# every image after the seam with somebody else's words. Both are invisible in
# the loss curve, so they are checked here instead.
per = meta["per_image"]
size = os.path.getsize(f"{prefix}_tokens.u16")
if size % (per * 2):
    raise SystemExit(f"plik tokenow ma {size} B, nie dzieli sie na obrazy — urwany")
# n from the file, not from meta: the resumed encode wrote its own session count
# (400015) into meta for a 1780125-pair corpus, and trusting it would have
# trained on 22% of the data without a word in the log.
n = size // (per * 2)
if n != meta["n"]:
    print(f"UWAGA: meta mowi {meta['n']:,} par, plik ma {n:,} — ufam plikowi", flush=True)
n_caps = sum(1 for _ in open(f"{prefix}_captions.jsonl")) \
    if os.path.exists(f"{prefix}_captions.jsonl") \
    else len(json.load(open(f"{prefix}_captions.json")))
if n_caps != n:
    raise SystemExit(f"{n_caps:,} podpisow na {n:,} obrazow — rozjazd par")
print(f"korpus: {prefix}\n  {n:,} par, {per} tokenow na obraz, "
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
