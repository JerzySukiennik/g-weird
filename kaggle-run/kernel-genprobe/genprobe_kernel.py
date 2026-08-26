"""Ile trwa narysowanie jednego obrazu — uczciwie, obiema drogami.

Pierwsza proba porownala autoregresje BEZ cache'u z MaskGIT-em, co bylo
bezuzyteczne: produkcyjny sampler uzywa KVCache, wiec porownanie bez niego
zawyzalo autoregresje kilkukrotnie. Tu obie metody dostaja to, czego naprawde
uzywaja.

Trening obu architektur zmierzony juz wczesniej i wyszedl praktycznie tak samo
(0,789 kontra 0,891 s na krok przy partii 32 na T4). Roznica miedzy nimi siedzi
wiec nie w treningu, tylko tutaj — i to ona rozstrzyga wybor.
"""

import json
import os
import subprocess
import sys
import time

import torch

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
IMAGE_LEN = 576

if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
sys.path.insert(0, ".")

from model.transformer import WeirdConfig, WeirdGPT              # noqa: E402
from model.maskgit import MaskGITConfig, MaskGIT, generate as mg_gen  # noqa: E402
from model.gpt_base import KVCache                               # noqa: E402

dev = "cuda"
res = {}

# --- autoregresja z cache'em, dokladnie jak train/sample.py -----------------
cfg = WeirdConfig(image_len=IMAGE_LEN)
ar = WeirdGPT(cfg).to(dev).eval()
prefix = torch.randint(0, 100, (2, cfg.text_len + 1), device=dev)   # 2 = para CFG


def ar_raz():
    kv = KVCache(cfg.n_layer)
    with torch.no_grad(), torch.cuda.amp.autocast():
        logits = ar(prefix, kv=kv, pos=0)[:, -1]
        for s in range(IMAGE_LEN):
            nxt = logits.argmax(-1, keepdim=True)
            logits = ar(nxt, kv=kv, pos=cfg.text_len + 1 + s)[:, -1]


ar_raz()                                    # rozgrzewka
torch.cuda.synchronize(); t0 = time.time()
for _ in range(3):
    ar_raz()
torch.cuda.synchronize()
res["ar_z_cache_s"] = round((time.time() - t0) / 3, 2)
print(f"autoregresja z KVCache: {res['ar_z_cache_s']} s "
      f"({IMAGE_LEN} przebiegow po jednym tokenie)", flush=True)
del ar; torch.cuda.empty_cache()

# --- maskgit ---------------------------------------------------------------
mcfg = MaskGITConfig(image_len=IMAGE_LEN)
mg = MaskGIT(mcfg).to(dev).eval()
# Szerokosc dokladnie text_len: poprzednia proba podala 33 kolumny (z BOS,
# jak w autoregresji) i generate rozjechal sie o jeden token. MaskGIT nie ma
# BOS, bo nie zaczyna od lewej — widzi caly obraz naraz.
text_rows = torch.randint(0, 100, (1, mcfg.text_len), device=dev)

for steps in (8, 12, 16):
    with torch.no_grad(), torch.cuda.amp.autocast():
        mg_gen(mg, text_rows, mcfg, steps=steps)          # rozgrzewka
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        with torch.no_grad(), torch.cuda.amp.autocast():
            mg_gen(mg, text_rows, mcfg, steps=steps)
    torch.cuda.synchronize()
    res[f"maskgit_{steps}_krokow_s"] = round((time.time() - t0) / 3, 3)
    print(f"maskgit {steps} przebiegow: {res[f'maskgit_{steps}_krokow_s']} s",
          flush=True)

json.dump(res, open(f"{WORK}/genprobe.json", "w"), indent=2)
print("\n" + json.dumps(res, indent=2), flush=True)
