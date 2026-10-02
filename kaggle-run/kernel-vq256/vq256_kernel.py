"""Kaggle GPU cell: take the 576-token tokenizer to 256px / 32x32 = 1024 tokens.

Not trained from scratch. The architecture is identical — base 64, mults
(1,2,4), f=8, codebook 8192 — and a convolutional encoder does not care what
resolution it runs at, only what statistics it has seen. Fine-tuning the
40000-step 192px checkpoint at 256px costs a fraction of a fresh run and ends
in the same place: 32x32 instead of 24x24, which is exactly DALL-E 1's spec
(256px, 1024 tokens, 8192 codes).

The codes change meaning, so the whole corpus must be re-encoded afterwards.
That was going to happen anyway.

Data is a deliberate mix rather than everything: the tokenizer learns local
texture, not semantics, so a representative slice beats volume. Weighted toward
the clean object renders the new model is aimed at.
"""

import glob
import json
import os
import subprocess
import sys

import torch

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
STEPS_TOTAL = 20000
MAX_STEPS = 9000           # ~1.3 s/krok przy 256 px -> ~3.3 h, z zapasem do sciany

if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}, "
      f"compute {cap[0]}.{cap[1]}", flush=True)
if cap[0] < 7:
    raise SystemExit("bez rdzeni tensor nie palmy kwoty")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob("/kaggle/input/**/gweird_meta.json", recursive=True))
prefixes = [m[: -len("_meta.json")] for m in metas]
if not prefixes:
    raise SystemExit("brak shardow z obrazami")
total = sum(json.load(open(f"{p}_meta.json"))["n"] for p in prefixes)
for p in prefixes:
    m = json.load(open(f"{p}_meta.json"))
    print(f"  {m['source']}: {m['n']:,} obrazow, {m['res']} px", flush=True)
    if m["res"] < 256:
        raise SystemExit(f"{p}: zapisane w {m['res']} px, nie da sie trenowac na 256")
print(f"{len(prefixes)} shardow, {total:,} obrazow", flush=True)

os.makedirs(f"{WORK}/run", exist_ok=True)
cks = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
if cks:
    steps = {c: torch.load(c, map_location="cpu", weights_only=False)["step"] for c in cks}
    best = max(steps, key=steps.get)
    subprocess.run(["cp", best, f"{WORK}/run/vqvae.pt"], check=True)
    print(f"dostrajam checkpoint z kroku {steps[best]} ({best})", flush=True)
else:
    print("brak checkpointu — trening od zera", flush=True)

# --train-res 256 to jedyna roznica wobec biegu 192 px. Partia 10 zamiast 16,
# bo uwaga w dekoderze siedzi na poziomach 32x32 i 64x64 zamiast 24x24 i 48x48,
# czyli 3.2x wiecej pozycji na tym drugim.
subprocess.run([sys.executable, "train/train_vqvae.py", "--data", *prefixes,
                "--out", f"{WORK}/run", "--res", "256", "--train-res", "256",
                "--mults", "1", "2", "4", "--dec-base", "128", "--dec-res", "3",
                "--dec-attn", "2", "--n-codes", "8192",
                "--steps", str(STEPS_TOTAL), "--max-steps", str(MAX_STEPS),
                "--batch", "10", "--workers", "2", "--lr", "1e-4",
                "--disc-start", "1000", "--adv-max", "0.2", "--fm", "1.0",
                "--log-every", "100", "--ckpt-every", "500",
                "--sample-every", "1000", "--resume"], check=True)

ck = torch.load(f"{WORK}/run/vqvae.pt", map_location="cpu", weights_only=False)
print(f"krok {ck['step']}, arch {ck['arch']}", flush=True)

# Dowod, ze siatka jest ta, o ktora chodzilo: jeden obraz przez kodowanie
# i dekodowanie, z wypisanym ksztaltem id.
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from model.vqvae import VQVAE
vq = VQVAE(**ck["arch"]).eval()
vq.load_state_dict(ck["model"], strict=False)
offs = json.load(open(f"{prefixes[0]}_offsets.json"))
fh = open(f"{prefixes[0]}_images.jpgbin", "rb")
import io
ims = []
for i in range(4):
    fh.seek(offs[i]); ims.append(np.asarray(
        Image.open(io.BytesIO(fh.read(offs[i + 1] - offs[i]))).convert("RGB")))
x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
with torch.no_grad():
    idx = vq.encode(x)
    out = vq.decode(idx)
print(f"siatka id: {tuple(idx.shape)} -> {idx.shape[-1] ** 2} tokenow na obraz", flush=True)
arr = ((out.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
sheet = np.concatenate([np.concatenate(ims, axis=1),
                        np.concatenate(list(arr), axis=1)], axis=0)
Image.fromarray(sheet).save(f"{WORK}/dowod-256.png")
print("gotowe", flush=True)
