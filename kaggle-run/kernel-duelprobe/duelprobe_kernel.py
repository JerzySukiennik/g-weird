"""Zmierz obie architektury zanim ktorakolwiek dostanie godziny GPU.

Sekwencja urosla z 289 do 609 tokenow (576 obrazu + 32 tekstu + 1), a uwaga
rosnie z kwadratem dlugosci, wiec kazde oszacowanie kosztu sprzed nowego
tokenizera jest nieaktualne. Zamiast zgadywac: forward+backward na prawdziwej
karcie, dla obu modeli, przy kilku rozmiarach partii, plus pomiar samego
generowania — bo "szybsze rysowanie" jest jednym z dwoch celow 1.1 i akurat je
da sie rozstrzygnac w kilkanascie minut zamiast w kilkanascie godzin.

Dane sa tu nieistotne — mierzymy koszt ksztaltu, nie jakosc — wiec partie sa
losowe. Zaden wniosek o jakosci obrazow z tego nie plynie i nie powinien.
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
print(f"GPU: {torch.cuda.get_device_name(0)}, sztuk {torch.cuda.device_count()}",
      flush=True)

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
sys.path.insert(0, ".")

from model.transformer import WeirdConfig, WeirdGPT      # noqa: E402
from model.maskgit import MaskGITConfig, MaskGIT, generate as mg_generate  # noqa: E402

dev = "cuda"
out = {}


def krok(model, seq, batch, causal):
    """Jeden pelny krok treningowy: forward, strata, backward."""
    x = torch.randint(0, 100, (batch, seq), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sc = torch.cuda.amp.GradScaler()
    for i in range(6):                      # dwa pierwsze to rozgrzewka
        if i == 2:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
        with torch.cuda.amp.autocast():
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), x.reshape(-1))
        sc.scale(loss).backward()
        sc.step(opt)
        sc.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 4
    mem = torch.cuda.max_memory_allocated() / 1e9
    return dt, mem


for name, mk in [("ar", lambda: WeirdGPT(WeirdConfig(image_len=IMAGE_LEN))),
                 ("maskgit", lambda: MaskGIT(MaskGITConfig(image_len=IMAGE_LEN)))]:
    m = mk().to(dev)
    seq = m.config.block_size
    par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"\n=== {name}: {par:.1f}M parametrow, sekwencja {seq}", flush=True)
    out[name] = {"params_m": round(par, 1), "seq": seq, "batches": {}}
    for b in (8, 16, 24, 32):
        torch.cuda.empty_cache()
        try:
            dt, mem = krok(m, seq, b, causal=(name == "ar"))
            ips = b / dt
            out[name]["batches"][b] = {"s_per_step": round(dt, 3),
                                       "img_per_s": round(ips, 1),
                                       "peak_gb": round(mem, 2)}
            print(f"  partia {b:>2}: {dt:.3f} s/krok, {ips:.1f} obr/s, "
                  f"szczyt {mem:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  partia {b:>2}: brak pamieci", flush=True)
            torch.cuda.empty_cache()
            break
    del m
    torch.cuda.empty_cache()

# Rysowanie jednego obrazu — to jest ta liczba, ktora widzi uzytkownik strony.
print("\n=== generowanie jednego obrazu", flush=True)
gen = {}
ar = WeirdGPT(WeirdConfig(image_len=IMAGE_LEN)).to(dev).eval()
ctx = torch.randint(0, 100, (2, 33), device=dev)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(), torch.cuda.amp.autocast():
    seqs = ctx
    for _ in range(IMAGE_LEN):
        lg = ar(seqs)
        lg = lg[0] if isinstance(lg, tuple) else lg
        nxt = lg[:, -1].argmax(-1, keepdim=True)
        seqs = torch.cat([seqs, nxt], 1)
torch.cuda.synchronize()
gen["ar_bez_cache_s"] = round(time.time() - t0, 2)
print(f"  autoregresja, {IMAGE_LEN} przebiegow bez cache: "
      f"{gen['ar_bez_cache_s']} s", flush=True)
del ar; torch.cuda.empty_cache()

mgcfg = MaskGITConfig(image_len=IMAGE_LEN)
mg = MaskGIT(mgcfg).to(dev).eval()
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(), torch.cuda.amp.autocast():
    mg_generate(mg, ctx[:1], mgcfg, steps=12)
torch.cuda.synchronize()
gen["maskgit_12_krokow_s"] = round(time.time() - t0, 2)
print(f"  maskgit, 12 przebiegow: {gen['maskgit_12_krokow_s']} s", flush=True)

out["generowanie"] = gen
json.dump(out, open(f"{WORK}/probe.json", "w"), indent=2)
print("\n" + json.dumps(out, indent=2), flush=True)
