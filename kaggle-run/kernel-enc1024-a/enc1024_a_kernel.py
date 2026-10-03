"""Kaggle GPU cell: encode part of the corpus at 256px / 1024 tokens.

Shards: ddb 0-3. Written for kernel enc1024-a; the four kernels together
cover the same 4,017,878 pairs the 576-token corpus has, so the new model trains
on identical data and the two tokenizers differ in nothing else.

The tokenizer is the UNTOUCHED 40000-step 192px checkpoint run on 256px input.
A convolutional encoder does not care about resolution, so this gives a 32x32
grid for free. We tried fine-tuning it at 256px (kernel vq256, 9000 steps, 2.6 h)
and it got worse: held-out reconstruction error 11.66 -> 12.88/255, and on the
same four pictures 12.87 untouched against 14.02 tuned, with visibly less detail
in the tuned one. The first 1000 steps before the discriminator also did not
improve it, so the adversarial terms are not the whole story.

The ids still differ from the 576-token corpus — a different grid — so nothing
encoded earlier can be mixed in.

Checked before and after, because this project has been burned by "finished"
meaning "ran": the tokenizer must be exactly the frozen 40000-step checkpoint,
the token file must be
exactly n x 1024 x 2 bytes, and a held sample must decode back to something
close to its original, with the error printed.
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
INPUT = os.environ.get("GW_INPUT", "/kaggle/input")
WORK = os.environ.get("GW_WORK", "/kaggle/working")
SHARDS = ["gweird-ddb-0", "gweird-ddb-1", "gweird-ddb-2", "gweird-ddb-3"]
TAG = "gwtok1024A"
GRID, PER = 32, 1024
LO, HI = [int(x) for x in os.environ.get("GW_STEP_RANGE", "40000,40000").split(",")]

if not torch.cuda.is_available() and not os.environ.get("GW_ALLOW_CPU"):
    raise SystemExit("brak GPU")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}, "
          f"compute {cap[0]}.{cap[1]}", flush=True)
    if cap[0] < 7:
        raise SystemExit("bez rdzeni tensor nie palmy kwoty")

if os.environ.get("GW_REPO_DIR"):
    os.chdir(os.environ["GW_REPO_DIR"])
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
    os.chdir(f"{WORK}/g-weird")
sys.path.insert(0, ".")
from model.vqvae import VQVAE   # noqa: E402

# Kazdy kernel podpina TYLKO swoje shardy, wiec bierzemy wszystko, co widac, i
# sprawdzamy liczbe. Posortowana lista daje powtarzalna kolejnosc, a to jedyne,
# czego wymagaja tokeny i podpisy.
metas = sorted(glob.glob(f"{INPUT}/**/gweird_meta.json", recursive=True))
if len(metas) != len(SHARDS):
    raise SystemExit(f"widze {len(metas)} shardow, oczekiwalem {len(SHARDS)}: {metas}")
prefixes = [m[: -len("_meta.json")] for m in metas]
total = 0
for p in prefixes:
    m = json.load(open(f"{p}_meta.json"))
    print(f"  {m['source']}: {m['n']:,} obrazow, {m['res']} px", flush=True)
    if m["res"] != 256:
        raise SystemExit(f"{p}: {m['res']} px, a tokenizer jest na 256")
    total += m["n"]
print(f"{len(prefixes)} shardow, {total:,} par do zakodowania", flush=True)

ckpts = glob.glob(f"{INPUT}/**/vqvae.pt", recursive=True)
if len(ckpts) != 1:
    raise SystemExit(f"vqvae.pt: {len(ckpts)} sztuk, oczekiwalem jednej: {ckpts}")
ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
print(f"tokenizer z kroku {ck['step']}, arch {ck['arch']}", flush=True)
if not LO <= ck["step"] <= HI:
    raise SystemExit(f"krok {ck['step']} poza {LO}..{HI} — oczekiwany "
                     f"zamrozony tokenizer z kroku 40000")

out = f"{WORK}/{TAG}"
subprocess.run([sys.executable, "train/encode_corpus.py",
                "--data", *prefixes, "--ckpt", ckpts[0], "--out-prefix", out,
                "--res", "256", "--batch", "128", "--workers", "16"], check=True)

size = os.path.getsize(f"{out}_tokens.u16")
want = total * PER * 2
if size != want:
    raise SystemExit(f"tokeny maja {size} B, oczekiwano {want} (n x {PER} x 2)")
caps = json.load(open(f"{out}_captions.json"))
if len(caps) != total:
    raise SystemExit(f"{len(caps)} podpisow na {total} obrazow")
print(f"tokeny OK: {size / 1e9:.2f} GB, {total:,} par", flush=True)

# Dowod: pierwsze 16 obrazow z pierwszego sharda z powrotem z samych id, z bledem
# wzgledem oryginalu. Rozmiar pliku moze sie zgadzac dla smieci.
vq = VQVAE(**ck["arch"]).eval()
vq.load_state_dict(ck["model"], strict=False)
offs = json.load(open(f"{prefixes[0]}_offsets.json"))
ims = []
with open(f"{prefixes[0]}_images.jpgbin", "rb") as fh:
    for i in range(min(16, total)):
        fh.seek(offs[i])
        ims.append(np.asarray(Image.open(io.BytesIO(
            fh.read(offs[i + 1] - offs[i]))).convert("RGB")))
toks = np.memmap(f"{out}_tokens.u16", dtype=np.uint16, mode="r", shape=(total, PER))
ids = torch.from_numpy(np.array(toks[:len(ims)], dtype=np.int64))
with torch.no_grad():
    rec = vq.decode(ids.view(-1, GRID, GRID))
arr = ((rec.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
err = float(np.abs(arr.astype(np.float32) - np.stack(ims).astype(np.float32)).mean())
print(f"blad rekonstrukcji z zapisanych id: {err:.2f}/255 na {len(ims)} obrazach", flush=True)
if err > 25:
    raise SystemExit(f"blad {err:.1f}/255 — tokeny nie odtwarzaja obrazow")
sheet = np.concatenate([np.concatenate(ims[:4], axis=1), np.concatenate(list(arr[:4]), axis=1)], axis=0)
Image.fromarray(sheet).save(f"{WORK}/{TAG}-dowod.png")
print("podpisy do dowodu:", [c[:60] for c in caps[:4]], flush=True)
print("gotowe", flush=True)
