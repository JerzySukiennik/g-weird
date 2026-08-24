"""Kaggle GPU cell: sharpen the decoder. T4x2, Internet ON.

Encoder and codebook stay frozen — the transformer predicts code ids, so moving
them would silently invalidate 11 h of transformer training. Only the decoder
learns, and what it learns is texture, not content.

Second attempt. The first ran 16000 steps at adv-max 0.5 and failed in a way the
numbers hid: reconstruction error fell to 10.9/255 — better than the reference
VQGAN's 12.5 — while faces melted and the pictures broke into blotches.

The training batches showed why. The corpus is largely watermarked stock photos,
and a discriminator exists to make outputs match the training distribution, so it
learned to render Shutterstock and Alamy marks crisply instead of subjects.

So: the cap drops to 0.2, between the 0.1 that left everything looking like oil
paint and the 0.5 that destroyed content; and crops always exclude the bottom
band and randomise everything else, so the watermark is no longer a fixed,
easily memorised pattern.
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

TOTAL_STEPS = 8000
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
                "--sample-every", "500", "--crop", "224", "--adv-max", "0.2"]
               + resume, check=True)

for f in sorted(os.listdir(OUT)):
    print(f"  {f}  {os.path.getsize(os.path.join(OUT,f))/1e6:.1f} MB", flush=True)
