"""Kaggle GPU cell: take the 576-token tokenizer to 256px / 32x32 = 1024 tokens.

Not trained from scratch. The architecture is identical — base 64, mults
(1,2,4), f=8, codebook 8192 — and a convolutional encoder does not care what
resolution it runs at, only what statistics it has seen. Fine-tuning the
40000-step 192px checkpoint at 256px ends in 32x32 instead of 24x24, which is
exactly DALL-E 1's spec (256px, 1024 tokens, 8192 codes).

The codes change meaning, so the whole corpus must be re-encoded afterwards.

**Version 1 of this kernel trained nothing and reported success.** It resumed
from step 40000 with a target of 20000, so the loop ran zero iterations, and its
"proof" was the id grid shape — which is simply 256/8 and comes out the same
for an untouched checkpoint. Two guards stop that happening again: the trainer
gets --finetune-from (weights loaded, schedule from zero), and this script
measures reconstruction error on 32 held images before and after and refuses to
finish unless it dropped.

**Result of version 2 (9000 steps, 2.6 h): the fine-tune made it worse, and the
guard said so.** Held-out error 11.66 -> 12.88/255; on four pictures read side by
side 12.87 untouched against 14.02 tuned, with visibly less detail in the tuned
one (samurai, picnic-basket weave, glasses). The untouched 192px checkpoint run
on 256px input already gives a 32x32 grid at ~12/255, so the 1024-token corpus
(kernels enc1024) uses it as it is. Note for anyone reading the training log: `rec`
jumps from 0.07 to 0.33 at step 1000, but from there it includes the
feature-matching term (rec = L1 + fm * feature_match), so that jump is mostly the
extra term appearing, not reconstruction collapsing; the held-out number is the
honest one. Kept for the record and in case a better recipe (lower adversarial
weight, later discriminator) is ever worth trying.
"""

import glob
import io
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
STEPS_TOTAL = 12000        # horyzont harmonogramu liczony od zera
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
sys.path.insert(0, ".")
from model.vqvae import VQVAE   # noqa: E402

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

cks = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
if not cks:
    raise SystemExit("brak checkpointu zrodlowego 192 px — nie trenuje od zera")
steps = {c: torch.load(c, map_location="cpu", weights_only=False)["step"] for c in cks}
src = max(steps, key=steps.get)
print(f"zrodlo: krok {steps[src]} ({src})", flush=True)
if steps[src] != 40000:
    raise SystemExit(f"to nie jest zamrozony tokenizer 192 px (krok {steps[src]})")

# 32 obrazy z roznych shardow, 8 z kazdego, zeby ocena nie zalezala od jednego
# zrodla. Te same obrazy przed i po.
def sample_images(n_per=8):
    ims = []
    for p in prefixes:
        offs = json.load(open(f"{p}_offsets.json"))
        with open(f"{p}_images.jpgbin", "rb") as fh:
            for i in range(0, n_per * 1000, 1000):
                fh.seek(offs[i])
                ims.append(np.asarray(Image.open(io.BytesIO(
                    fh.read(offs[i + 1] - offs[i]))).convert("RGB")))
    return ims


def recon_error(ckpt_path, ims):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vq = VQVAE(**ck["arch"]).cuda().eval()
    vq.load_state_dict(ck["model"], strict=False)
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1.0
    errs = []
    with torch.no_grad():
        for i in range(0, len(ims), 8):
            out = vq.decode(vq.encode(x[i:i + 8]))
            errs.append((out - x[i:i + 8]).abs().mean().item() * 127.5)
    idx_shape = tuple(vq.encode(x[:1]).shape)
    del vq
    torch.cuda.empty_cache()
    return float(np.mean(errs)), idx_shape


ims = sample_images()
before, shape = recon_error(src, ims)
print(f"PRZED: blad rekonstrukcji {before:.2f}/255 na {len(ims)} obrazach 256 px, "
      f"siatka {shape}", flush=True)

os.makedirs(f"{WORK}/run", exist_ok=True)
# --finetune-from: wagi ze zrodla, licznik krokow i harmonogram od zera.
subprocess.run([sys.executable, "train/train_vqvae.py", "--data", *prefixes,
                "--out", f"{WORK}/run", "--finetune-from", src,
                "--res", "256", "--train-res", "256",
                "--mults", "1", "2", "4", "--dec-base", "128", "--dec-res", "3",
                "--dec-attn", "2", "--n-codes", "8192",
                "--steps", str(STEPS_TOTAL), "--max-steps", str(MAX_STEPS),
                "--batch", "10", "--workers", "2", "--lr", "1e-4",
                "--disc-start", "1000", "--adv-max", "0.2", "--fm", "1.0",
                "--log-every", "100", "--ckpt-every", "500",
                "--sample-every", "1000", "--resume"], check=True)

ck = torch.load(f"{WORK}/run/vqvae.pt", map_location="cpu", weights_only=False)
print(f"po treningu: krok {ck['step']}", flush=True)
if ck["step"] < 1000:
    raise SystemExit(f"trening zrobil {ck['step']} krokow — to nie jest dostrojenie")

after, shape = recon_error(f"{WORK}/run/vqvae.pt", ims)
print(f"PO:    blad rekonstrukcji {after:.2f}/255, siatka {shape}", flush=True)
if shape[-1] != 32:
    raise SystemExit(f"siatka {shape}, oczekiwalem 32x32")
if not after < before:
    raise SystemExit(f"blad nie spadl: {before:.2f} -> {after:.2f}")

vq = VQVAE(**ck["arch"]).cuda().eval()
vq.load_state_dict(ck["model"], strict=False)
x = torch.from_numpy(np.stack(ims[:4])).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1.0
with torch.no_grad():
    out = vq.decode(vq.encode(x))
arr = ((out.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
sheet = np.concatenate([np.concatenate(ims[:4], axis=1),
                        np.concatenate(list(arr), axis=1)], axis=0)
Image.fromarray(sheet).save(f"{WORK}/dowod-256.png")
print(f"gotowe: {before:.2f} -> {after:.2f} (spadek {100 * (1 - after / before):.0f}%)",
      flush=True)
