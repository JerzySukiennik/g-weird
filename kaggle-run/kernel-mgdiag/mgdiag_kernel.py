"""Czy MaskGIT sie nie nauczyl, czy tylko nie umie rysowac od zera?

Po 12000 krokach MaskGIT dal plaskie plamy koloru, a autoregresja na tych samych
danych i tej samej liczbie krokow dala rozpoznawalne sceny. To sa dwie zupelnie
rozne diagnozy i nie da sie ich rozroznic patrzac na wynik generowania:

  - jesli model UMIE uzupelniac czesciowo zaslonienty obraz, to trening dziala,
    a zepsute jest rysowanie od zera (harmonogram odslaniania, sila prowadzenia);
  - jesli nie umie nawet tego, problem siedzi w treningu albo w samej stracie.

Wiec: bierzemy prawdziwe obrazy z korpusu, zaslaniamy im 25/50/75% tokenow i
kazemy modelowi je uzupelnic. Pierwszy wiersz to oryginal odkodowany z pelnych
id — punkt odniesienia, ktory pokazuje sufit mozliwosci tokenizera.
"""

import glob
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
sys.path.insert(0, ".")

from model.maskgit import MaskGITConfig, MaskGIT     # noqa: E402
from model.vqvae import VQVAE                        # noqa: E402
from tokenizers import Tokenizer                     # noqa: E402

dev = "cuda"
ckpts = glob.glob("/kaggle/input/**/maskgit.pt", recursive=True)
vqs = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
txt = glob.glob("/kaggle/input/**/text.json", recursive=True)
metas = glob.glob("/kaggle/input/**/gwtok*_meta.json", recursive=True)
if not (ckpts and vqs and txt and metas):
    raise SystemExit(f"brakuje wejsc: {ckpts} {vqs} {txt} {metas}")

ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
cfg = MaskGITConfig(**{k: v for k, v in ck["cfg"].items()
                       if k in MaskGITConfig.__dataclass_fields__})
model = MaskGIT(cfg).to(dev).eval()
model.load_state_dict(ck["model"])
print(f"maskgit z kroku {ck['step']}", flush=True)

vk = torch.load(vqs[0], map_location="cpu", weights_only=False)
vq = VQVAE(**vk["arch"]).to(dev).eval()
vq.load_state_dict(vk["model"], strict=False)
tok = Tokenizer.from_file(txt[0])

prefix = metas[0][: -len("_meta.json")]
n = json.load(open(metas[0]))["n"]
toks = np.memmap(f"{prefix}_tokens.u16", dtype=np.uint16, mode="r",
                 shape=(n, cfg.image_len))
caps = json.load(open(f"{prefix}_captions.json"))

PICK = [7, 1000, 50000, 300000]
ids = torch.from_numpy(np.array(toks[PICK], dtype=np.int64)).to(dev)
rows = []
for i in PICK:
    e = tok.encode(caps[i]).ids[: cfg.text_len]
    rows.append([cfg.text_token(t) for t in e] + [cfg.PAD] * (cfg.text_len - len(e)))
text_rows = torch.tensor(rows, dtype=torch.long, device=dev)
print("podpisy:", [caps[i][:50] for i in PICK], flush=True)

grid = int(round(math.sqrt(cfg.image_len)))
sheets = [vq.decode(ids.view(-1, grid, grid))]     # wiersz odniesienia

g = torch.Generator(device="cpu").manual_seed(0)
for frac in (0.25, 0.5, 0.75):
    img = ids + cfg.image_token(0)
    k = int(frac * cfg.image_len)
    for r in range(len(PICK)):
        cut = torch.randperm(cfg.image_len, generator=g)[:k]
        img[r, cut.to(dev)] = cfg.MASK
    with torch.no_grad(), torch.cuda.amp.autocast():
        seq = torch.cat([text_rows, img], dim=1)
        logits = model(seq)[:, cfg.text_len:]
        lo, hi = cfg.image_token(0), cfg.image_token(cfg.n_image - 1)
        logits[..., :lo] = -float("inf")
        logits[..., hi + 1:] = -float("inf")
        filled = torch.where(img == cfg.MASK, logits.argmax(-1), img)
    with torch.no_grad():
        sheets.append(vq.decode((filled - lo).view(-1, grid, grid)))
    print(f"zaslonione {int(frac*100)}% — uzupelnione", flush=True)

out = torch.cat(sheets, 0)
arr = ((out.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
h, w = arr.shape[1:3]
sheet = np.zeros((4 * h, 4 * w, 3), dtype=np.uint8)
for i, im in enumerate(arr):
    r, c = divmod(i, 4)
    sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
Image.fromarray(sheet).save(f"{WORK}/mgdiag.png")
print("gotowe — wiersze: oryginal, 25%, 50%, 75% zaslonienia", flush=True)
