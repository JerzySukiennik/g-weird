"""Kaggle GPU cell: find the learning rate a masked model actually wants.

The one hypothesis never tested about why MaskGIT trailed here. A masked model
gets gradient only on hidden positions — about 64% of tokens under the arccos
schedule — while the autoregressive model gets it on all of them. The learning
rate 3e-4 was copied straight from the autoregressive recipe, and the masked
run's loss then sat at 7.45 for 25000 steps without moving.

Three arms, identical in everything but the rate, judged on the fill-in test
rather than on from-scratch samples: at 4000 steps nothing draws well from an
empty canvas, but refilling a masked image is already measurable, and it is the
task the editing feature performs anyway.

Runs on the existing 192px corpus on purpose. The finding is about optimisation,
not about resolution, so it transfers to the 1024-token model and can be had
now instead of after the new tokenizer.
"""

import glob
import json
import os
import subprocess
import sys

import torch

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
STEPS = 4000
RATES = [3e-4, 8e-4, 1.5e-3]

if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}", flush=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob("/kaggle/input/**/gwtok*_meta.json", recursive=True))
prefixes = [m[: -len("_meta.json")] for m in metas]
if len(prefixes) != 4:
    raise SystemExit(f"oczekiwalem 4 shardow tokenow, widze {len(prefixes)}")
n = sum(os.path.getsize(f"{p}_tokens.u16") // 1152 for p in prefixes)
if n != 4017878:
    raise SystemExit(f"korpus ma {n} par, oczekiwalem 4017878")
print(f"korpus: {n:,} par", flush=True)

txt = glob.glob("/kaggle/input/**/text.json", recursive=True)
vqs = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
jpg = sorted(glob.glob("/kaggle/input/**/gweird_meta.json", recursive=True))
if len(txt) != 1 or len(vqs) != 1 or not jpg:
    raise SystemExit(f"wejscia: text {txt}, vqvae {vqs}, obrazy {jpg}")

os.makedirs(f"{WORK}/data", exist_ok=True)
local = []
for p in prefixes:
    tag = os.path.basename(p)
    for suf in ("tokens.u16", "captions.json", "meta.json"):
        dst = f"{WORK}/data/{tag}_{suf}"
        if not os.path.exists(dst):
            os.symlink(f"{p}_{suf}", dst)
    local.append(f"{WORK}/data/{tag}")
subprocess.run([sys.executable, "data/pack_captions.py", "--data", *local,
                "--tokenizer", txt[0], "--text-len", "64"], check=True)

# fill_test.py szuka text.json obok vqvae.pt, zeby uzyc prawdziwych podpisow.
os.makedirs(f"{WORK}/aux", exist_ok=True)
subprocess.run(["cp", vqs[0], f"{WORK}/aux/vqvae.pt"], check=True)
subprocess.run(["cp", txt[0], f"{WORK}/aux/text.json"], check=True)

for lr in RATES:
    tag = f"lr{lr:g}"
    out = f"{WORK}/run-{tag}"
    os.makedirs(out, exist_ok=True)
    print(f"\n===== tempo uczenia {lr:g} =====", flush=True)
    subprocess.run([sys.executable, "train/train_maskgit.py", "--data", *local,
                    "--out", out, "--steps", str(STEPS), "--batch", "32",
                    "--accum", "2", "--lr", str(lr), "--warmup", "500",
                    "--label-smoothing", "0.1", "--workers", "4",
                    "--log-every", "200", "--ckpt-every", "2000"], check=True)
    subprocess.run([sys.executable, "train/fill_test.py",
                    "--ckpt", f"{out}/maskgit.pt", "--vqvae", f"{WORK}/aux/vqvae.pt",
                    "--data", jpg[0][: -len("_meta.json")],
                    "--fracs", "0.5", "--out", f"{WORK}/fill-{tag}.png"], check=False)
    subprocess.run(["rm", "-f", f"{out}/maskgit.pt"], check=False)  # 745 MB x3

subprocess.run(["rm", "-rf", f"{WORK}/data", f"{WORK}/aux"], check=False)
print("\ngotowe — porownaj fill-lr*.png", flush=True)
