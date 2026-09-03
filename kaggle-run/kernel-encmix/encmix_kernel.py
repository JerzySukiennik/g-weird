"""Kaggle GPU cell: turn the mixed corpus into token ids. T4x2, Internet ON.

Nine shards mount here — eight of DiffusionDB and one of COCO — and they are
encoded as a single corpus by the frozen VQ-VAE. The frozen part is not a
detail: the transformer learns what each id means, so the encoder and codebook
must be the same ones used for every later run, and the same ones the decoder
is trained against.

Shard order comes from a sorted glob, so it is reproducible between runs. That
matters because the caption file is written in the same order as the tokens, and
a different order would pair every image after the seam with somebody else's
words — the failure the length check at the end of encode_corpus.py exists to
catch.

Encoding leaves the VQ-VAE untouched and is therefore a one-time cost for the
whole project: the transformer duel, the full training and any retraining all
read this same file.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

import torch
if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}, compute {cap[0]}.{cap[1]}, "
      f"sztuk {torch.cuda.device_count()}", flush=True)
if cap[0] < 7:
    raise SystemExit(f"sm_{cap[0]}{cap[1]} nie jest wspierana — ustaw NvidiaTeslaT4")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob("/kaggle/input/**/gweird_meta.json", recursive=True))
if not metas:
    raise SystemExit("brak shardow — podepnij wyjscia gweird-ddb-* i gweird-coco")
prefixes = [m[:-len("_meta.json")] for m in metas]

import json
total = 0
for m in metas:
    d = json.load(open(m))
    total += d["n"]
    print(f"  {os.path.basename(os.path.dirname(m))}: {d['n']:,} par "
          f"({d.get('source', '?')})", flush=True)
print(f"{len(prefixes)} shardow, {total:,} par razem", flush=True)

ckpts = sorted(glob.glob("/kaggle/input/**/vqvae.pt", recursive=True))
if not ckpts:
    raise SystemExit("brak vqvae.pt — podepnij wyjscie gweird-vqvae")
print(f"zamrozony tokenizer: {ckpts[0]}", flush=True)

subprocess.run([sys.executable, "train/encode_corpus.py",
                "--data", *prefixes, "--ckpt", ckpts[0],
                "--out-prefix", f"{WORK}/gwtok", "--batch", "64"], check=True)

for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
