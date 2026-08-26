"""Pojedynek maskgit: ta sama polowa korpusu, tyle samo krokow, te same podpisy.

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

subprocess.run([sys.executable, "train/train_maskgit.py",
                "--data", data, "--tokenizer", txt[0], "--out", f"{WORK}/run",
                "--steps", str(STEPS), "--batch", "32", "--accum", "1",
                "--workers", "4", "--log-every", "200", "--ckpt-every", "2000"],
               check=True)

# MaskGIT nie ma wlasnego skryptu do rysowania, wiec tu jest jego odpowiednik
# train/sample.py — te same podpisy, ta sama sila prowadzenia, ten sam arkusz.
import math

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from model.maskgit import MaskGITConfig, MaskGIT, generate as mg_gen
from model.vqvae import VQVAE

dev = "cuda"
ck = torch.load(f"{WORK}/run/maskgit.pt", map_location="cpu", weights_only=False)
saved = ck.get("cfg")
if saved is None:
    raise SystemExit("checkpoint bez konfiguracji")
cfg = MaskGITConfig(**{k: v for k, v in saved.items()
                       if k in MaskGITConfig.__dataclass_fields__})
model = MaskGIT(cfg).to(dev).eval()
model.load_state_dict(ck["model"])
print(f"maskgit z kroku {ck.get('step', '?')}", flush=True)

tok = Tokenizer.from_file(txt[0])
rows = []
for p in PROMPTS:
    ids = tok.encode(p).ids[: cfg.text_len]
    rows.append([cfg.text_token(i) for i in ids] + [cfg.PAD] * (cfg.text_len - len(ids)))
text_rows = torch.tensor(rows, dtype=torch.long, device=dev)

with torch.no_grad(), torch.cuda.amp.autocast():
    codes = mg_gen(model, text_rows, cfg, steps=12, scale=4.0)

vk = torch.load(vqs[0], map_location="cpu", weights_only=False)
vq = VQVAE(**vk["arch"]).to(dev).eval()
vq.load_state_dict(vk["model"], strict=False)
grid = int(round(math.sqrt(cfg.image_len)))
with torch.no_grad():
    imgs = vq.decode(codes.view(-1, grid, grid).to(dev))
imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
h, w = imgs.shape[1:3]
sheet = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
for i, im in enumerate(imgs):
    r, c = divmod(i, 4)
    sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
Image.fromarray(sheet).save(f"{WORK}/duel-maskgit.png")
print("gotowe", flush=True)
