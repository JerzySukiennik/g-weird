"""Kaggle GPU cell: sharpen the decoder. T4x2, Internet ON.

Encoder and codebook stay frozen — the transformer predicts code ids, so moving
them would silently invalidate 11 h of transformer training. Only the decoder
learns, and what it learns is texture, not content.

Budget: 16000 steps, sized to land inside one weekly quota at the measured rate.
Checkpoints every 500 steps and sample PNGs every 1000, because today proved
several times that the numbers can look fine while the picture falls apart.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

TOTAL_STEPS = 16000
BATCH = 16

import torch
if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}, compute {cap[0]}.{cap[1]}, "
      f"sztuk {torch.cuda.device_count()}", flush=True)
if cap[0] < 7:
    raise SystemExit(f"sm_{cap[0]}{cap[1]} nie jest wspierana — ustaw NvidiaTeslaT4")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob("/kaggle/input/**/gweird_meta.json", recursive=True))
if not metas:
    raise SystemExit("brak shardow — podepnij wyjscie gweird-prep-*")
prefixes = [m[:-len("_meta.json")] for m in metas]
print(f"{len(prefixes)} shardow", flush=True)

vq = sorted(glob.glob("/kaggle/input/**/vqvae.pt", recursive=True))
if not vq:
    raise SystemExit("brak vqvae.pt — podepnij wyjscie gweird-vqvae")
print(f"zamrozony tokenizer: {vq[0]}", flush=True)

resume = []
for c in sorted(glob.glob("/kaggle/input/**/decoder.pt", recursive=True)):
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["cp", c, f"{OUT}/decoder.pt"], check=True)
    resume = ["--resume"]
    print(f"wznawiam z {c}", flush=True)
    break

subprocess.run([sys.executable, "train/train_decoder.py",
                "--data", *prefixes, "--vqvae", vq[0], "--out", OUT,
                "--steps", str(TOTAL_STEPS), "--max-steps", str(TOTAL_STEPS),
                "--batch", str(BATCH), "--workers", "2",
                "--log-every", "100", "--ckpt-every", "500",
                "--sample-every", "1000"] + resume, check=True)

for f in sorted(os.listdir(OUT)):
    print(f"  {f}  {os.path.getsize(os.path.join(OUT,f))/1e6:.1f} MB", flush=True)
