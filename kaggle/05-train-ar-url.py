"""Kaggle GPU cell: train the text-to-image transformer. T4x2, Internet ON.

Same job as 04-train-ar.py, but the corpus arrives over HTTP instead of being
mounted. The reason is not preference: the API token in use cannot attach kernel
outputs as sources (it is denied kernels.get, which source validation needs),
while signed output URLs work fine. Downloading ~1 GB is the cheaper fix.

The caption BPE is trained here rather than at home, so the vocabulary comes from
all 1.78M captions instead of a 90k sample, for about a minute of CPU.

Sessions end voluntarily at SESSION_STEPS; checkpoints are atomic and written
every 500 steps. To continue, push again with a 'ckpt' URL pointing at the last
run's gweird.pt.
"""

import json
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"
TMP = "/kaggle/tmp"

TOTAL_STEPS = 60000
SESSION_STEPS = 30000
BATCH, ACCUM = 16, 4

URLS = json.loads(os.environ.get("GW_URLS") or "{}")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
os.makedirs(TMP, exist_ok=True)


def grab(key, dest):
    subprocess.run(["curl", "-sSL", "--retry", "3", "-o", dest, URLS[key]], check=True)
    print(f"  {key}: {os.path.getsize(dest)/1e6:.1f} MB", flush=True)
    return dest


prefix = f"{TMP}/gwtok"
for key, suffix in [("meta", "_meta.json"), ("caps", "_captions.json"),
                    ("tokens", "_tokens.u16")]:
    grab(key, prefix + suffix)

meta = json.load(open(f"{prefix}_meta.json"))
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
print(f"korpus: {n:,} par, {per} tokenow na obraz, "
      f"tokenizer z kroku {meta['vqvae_step']}", flush=True)
del caps

tok_path = f"{WORK}/text.json"
subprocess.run([sys.executable, "data/text_tokenizer.py",
                "--captions", f"{prefix}_captions.json",
                "--vocab", "8192", "--out", tok_path], check=True)

resume = []
if "ckpt" in URLS:
    os.makedirs(OUT, exist_ok=True)
    grab("ckpt", f"{OUT}/gweird.pt")
    resume = ["--resume"]

subprocess.run([sys.executable, "train/train_ar.py",
                "--data", prefix, "--tokenizer", tok_path, "--out", OUT,
                "--steps", str(TOTAL_STEPS), "--max-steps", str(SESSION_STEPS),
                "--batch", str(BATCH), "--accum", str(ACCUM),
                "--workers", "4", "--log-every", "100",
                "--ckpt-every", "500"] + resume, check=True)
