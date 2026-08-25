"""Kaggle GPU cell: encode half the corpus with the 576-token tokenizer.

The shards live on Kaggle already, as outputs of the prep kernels, so attaching
them costs nothing and downloads nothing. That is the whole reason encoding
happens here and not on Colab: the images are 49.5 GB and pulling them across
the network would take longer than the encoding itself.

**--enc-res 192 is not a tuning knob.** The tokenizer was trained at 192px with
three downsamples, so its grid is 24x24 = 576. Feeding it the stored 256px would
produce 1024 tokens per image from statistics it has never seen, and since the
transformer is permanently bound to whatever ids it is trained on, that mistake
would only surface as bad images weeks later.

Half the corpus per kernel so the two halves run side by side; the training run
concatenates them, which is safe because the token file is a flat array in
corpus order and the captions are a plain list in the same order.
"""

import glob
import json
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
SHARDS = ["gweird-ddb-4", "gweird-ddb-5", "gweird-ddb-6", "gweird-ddb-7", "gweird-jdb-0"]
TAG = "gwtokB"

import numpy as np
import torch

if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}, compute {cap[0]}.{cap[1]}, "
      f"sztuk {torch.cuda.device_count()}", flush=True)
if cap[0] < 7:
    raise SystemExit(f"sm_{cap[0]}{cap[1]} nie ma rdzeni tensor, nie tracmy na to kwoty")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

# Kolejnosc z SHARDS, nie z glob: podpisy i tokeny musza isc w tej samej
# kolejnosci, a glob sortuje alfabetycznie i cicho by ja przestawil.
prefixes = []
for s in SHARDS:
    hits = glob.glob(f"/kaggle/input/{s}/**/gweird_meta.json", recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"{s}: znalazlem {len(hits)} meta.json, oczekiwalem jednego")
    prefixes.append(hits[0][: -len("_meta.json")])
total = sum(json.load(open(f"{p}_meta.json"))["n"] for p in prefixes)
print(f"{len(prefixes)} shardow, {total:,} par do zakodowania", flush=True)

ckpts = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
if len(ckpts) != 1:
    raise SystemExit(f"vqvae.pt: {len(ckpts)} sztuk, oczekiwalem jednej")
step = torch.load(ckpts[0], map_location="cpu", weights_only=False)["step"]
print(f"tokenizer z kroku {step}", flush=True)
if step != 40000:
    raise SystemExit(f"to nie jest ukonczony tokenizer (krok {step})")

out = f"{WORK}/{TAG}"
subprocess.run([sys.executable, "train/encode_corpus.py",
                "--data", *prefixes, "--ckpt", ckpts[0],
                "--out-prefix", out, "--res", "256", "--enc-res", "192",
                "--batch", "128", "--workers", "16"], check=True)

# Rozmiar pliku jest jedynym dowodem, ze zakodowal sie caly korpus. meta.json
# potrafil juz kiedys zapisac licznik sesji zamiast korpusu.
size = os.path.getsize(f"{out}_tokens.u16")
want = total * 576 * 2
if size != want:
    raise SystemExit(f"tokeny maja {size} B, oczekiwano {want}")
print(f"tokeny OK: {size/1e9:.2f} GB", flush=True)

# Dowod, ze te liczby cokolwiek znacza: cztery pierwsze obrazy odtworzone
# z samych id i zapisane jako PNG. Rozmiar pliku moze sie zgadzac dla smieci.
sys.path.insert(0, ".")
from model.vqvae import VQVAE
from PIL import Image

ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
vq = VQVAE(**ck["arch"]).eval()
vq.load_state_dict(ck["model"], strict=False)
toks = np.memmap(f"{out}_tokens.u16", dtype=np.uint16, mode="r", shape=(total, 576))
ids = torch.from_numpy(np.array(toks[:4], dtype=np.int64))
with torch.no_grad():
    img = vq.decode(ids.view(4, 24, 24))
arr = ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
Image.fromarray(np.concatenate(list(arr), axis=1)).save(f"{WORK}/{TAG}-dowod.png")
caps = json.load(open(f"{out}_captions.json"))
print("podpisy do dowodu:", [c[:60] for c in caps[:4]], flush=True)
print("gotowe", flush=True)
