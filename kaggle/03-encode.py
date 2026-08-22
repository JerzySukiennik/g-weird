"""Kaggle GPU cell: turn 1.78M images into token ids. T4, Internet ON.

One-time cost that pays for the whole transformer stage: the corpus goes from
31.6 GB of pixels to about 0.9 GB of uint16, which is what lets the next model
memory-map the lot instead of streaming shards.

The encoder and codebook are frozen from this point. See train/encode_corpus.py
for why that matters — in short, the transformer learns what each id means, so
changing either of them later silently invalidates it. The decoder stays free to
improve.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob("/kaggle/input/**/gweird_meta.json", recursive=True))
if not metas:
    raise SystemExit("brak shardow — podepnij wyjscia gweird-prep-*")
prefixes = [m[:-len("_meta.json")] for m in metas]
print(f"{len(prefixes)} shardow", flush=True)

ckpts = sorted(glob.glob("/kaggle/input/**/vqvae.pt", recursive=True))
if not ckpts:
    raise SystemExit("brak vqvae.pt — podepnij wyjscie gweird-vqvae")
print(f"tokenizer: {ckpts[0]}", flush=True)

subprocess.run([sys.executable, "train/encode_corpus.py",
                "--data", *prefixes, "--ckpt", ckpts[0],
                "--out-prefix", f"{WORK}/gwtok", "--batch", "64"], check=True)

for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
