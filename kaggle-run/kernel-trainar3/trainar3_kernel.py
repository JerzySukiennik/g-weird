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

import torch

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

TOTAL_STEPS = 60000
SESSION_STEPS = 30000
BATCH, ACCUM = 16, 4

# Kaggle does not always give the accelerator that was asked for, and an
# unsupported one fails deep inside the first embedding lookup with
# "no kernel image is available for execution on the device" — a message that
# says nothing about the cause. A run once landed on a Tesla P100 (sm_60) while
# this PyTorch build supports sm_70 and up. Checking here turns three wasted
# minutes and a cryptic traceback into one line.
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}, compute {cap[0]}.{cap[1]}, sztuk {torch.cuda.device_count()}",
          flush=True)
    if cap[0] < 7:
        raise SystemExit(f"{name} (sm_{cap[0]}{cap[1]}) nie jest wspierana przez ten "
                         f"PyTorch — ustaw machine_shape na NvidiaTeslaT4 i powtorz")
else:
    raise SystemExit("brak GPU — kernel bez akceleratora")


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

# A later session MUST find its checkpoint. The original code fell back to
# starting from scratch when the glob came up empty, which would silently throw
# away a finished 5.52 h session and look, in the logs, exactly like a normal
# first run. Set REQUIRE_RESUME for every session after the first.
REQUIRE_RESUME = os.environ.get("GW_REQUIRE_RESUME", "1") == "1"

resume = []
found = sorted(glob.glob("/kaggle/input/**/gweird.pt", recursive=True))
if found:
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["cp", found[0], f"{OUT}/gweird.pt"], check=True)
    resume = ["--resume"]
    ck_step = torch.load(f"{OUT}/gweird.pt", map_location="cpu",
                         weights_only=False).get("step", "?")
    print(f"wznawiam z {found[0]} (krok {ck_step})", flush=True)
elif REQUIRE_RESUME:
    raise SystemExit("nie znalazlem gweird.pt w zadnym zrodle — podepnij wyjscie "
                     "poprzedniej sesji albo ustaw GW_REQUIRE_RESUME=0 dla startu "
                     "od zera")
else:
    print("start od zera", flush=True)

subprocess.run([sys.executable, "train/train_ar.py",
                "--data", prefix, "--tokenizer", tok_path, "--out", OUT,
                "--steps", str(TOTAL_STEPS), "--max-steps", str(SESSION_STEPS),
                "--batch", str(BATCH), "--accum", str(ACCUM),
                "--workers", "4", "--log-every", "100",
                "--ckpt-every", "500"] + resume, check=True)
