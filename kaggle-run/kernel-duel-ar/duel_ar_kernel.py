"""Pojedynek ar: ta sama polowa korpusu, tyle samo krokow, te same podpisy.

Koszt treningu obu architektur zmierzony wczesniej i praktycznie identyczny
(0,789 kontra 0,891 s na krok, partia 32, T4). Rysowanie jednego obrazu juz nie:
autoregresja z KVCache 7,32 s, MaskGIT 12 przebiegow 0,24 s — trzydziesci razy
szybciej. Zostaje pytanie, ktorego zaden pomiar nie rozstrzyga: ktora robi
lepsze obrazy. Stad ten bieg — i stad osiem stalych podpisow zamiast losowej
partii, zeby ocena byla jeden do jednego, a nie wrazeniem z dwoch roznych scen.

Wygrana rozstrzyga sie na obrazkach. Strata treningowa obu modeli nie jest
porownywalna, bo licza zupelnie co innego: autoregresja przewiduje kazdy token
po kolei, MaskGIT tylko te zaslonione.
"""

import glob
import json
import os
import subprocess
import sys

import torch

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
STEPS = 12000
PROMPTS = ["a horse standing in a field", "a red double decker bus on a street", "a cat wearing sunglasses", "portrait of an old man with a beard", "a bowl of soup on a wooden table", "a castle on a mountain at sunset", "two people riding bicycles", "a robot playing a piano"]

if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}, sztuk {torch.cuda.device_count()}",
      flush=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = glob.glob("/kaggle/input/**/gwtok*_meta.json", recursive=True)
if len(metas) != 1:
    raise SystemExit(f"korpus: {len(metas)} sztuk, oczekiwalem jednej: {metas}")
data = metas[0][: -len("_meta.json")]
per = json.load(open(metas[0]))["per_image"]
print(f"korpus {data}, {per} tokenow na obraz", flush=True)
if per != 576:
    raise SystemExit(f"to nie jest korpus nowego tokenizera ({per} tokenow)")

txt = glob.glob("/kaggle/input/**/text.json", recursive=True)
vqs = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
if len(txt) != 1 or len(vqs) != 1:
    raise SystemExit(f"tokenizer tekstu {txt}, vqvae {vqs}")

subprocess.run([sys.executable, "train/train_ar.py",
                "--data", data, "--tokenizer", txt[0], "--out", f"{WORK}/run",
                "--steps", str(STEPS), "--batch", "32", "--accum", "1",
                "--workers", "4", "--log-every", "200", "--ckpt-every", "2000"],
               check=True)

subprocess.run([sys.executable, "train/sample.py",
                "--ckpt", f"{WORK}/run/gweird.pt", "--vqvae", vqs[0],
                "--tokenizer", txt[0], "--prompts", *PROMPTS,
                "--out", f"{WORK}/duel-ar.png", "--cfg-scale", "4.0"],
               check=True)
print("gotowe", flush=True)
