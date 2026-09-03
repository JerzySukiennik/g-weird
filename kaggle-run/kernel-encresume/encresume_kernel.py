"""Kaggle GPU cell: finish the corpus encode that was killed at 83%. T4, Internet ON.

The first attempt (gweird-encode) ran 4.65 h and died with no traceback at all.
That silence is the diagnosis: a Python exception prints, an OOM kill does not.
The cause was in this pipeline's own code — every caption was accumulated in a
list and written only at the end, so memory grew monotonically for the whole run.

What survived is the part that was written as it went: gwtok_tokens.u16, 753 MB,
exactly 1470798 images x 256 tokens x 2 B, with no partial image at the tail. So
this kernel does not redo 4.65 h of GPU. It downloads that file, appends the
remaining ~310k images, and re-derives the captions for the whole corpus from the
shards (strings, no GPU).

Two things must hold or the token stream silently misaligns with its captions:
  * the SAME prep shards, mounted in the same set — the prefix list comes from a
    sorted glob, so the set fixes the order;
  * the SAME vqvae.pt — a different codebook would make the second half of the
    file mean something different from the first, with nothing to signal it.
Both are pinned by kernel_sources. The length check at the end of
train/encode_corpus.py is the last line of defence.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

# Images already encoded and present in the partial file. Derived from its size,
# not from the log: the log's last line said 1460046, but the process wrote more
# before it died. The file is the truth.
DONE = 1470798
PARTIAL_URL = 'https://www.kaggleusercontent.com/kf/344193492/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..-sIniborOXhBi_2vpFAAig.9-M9pp4COftQ-g6m1KeaSGsiTm8pSaasTfQcGAyYCi5ZxVXRP84G7KT4VlIRDlALSuAMskxFewSpuL9Q-V6rp6yGVD3W67fDAUoUnZi_ncu-7MYI771U9R9gOu6BCw5A_P_pMFa75Q331tg6p3AhvcYzF8T2XYjPIgSn35iLSaF-kPQqjXszZWVXtHqFxXyTssiaRI_flNF1V7rq8I_fNd6Sh3Kx4WJ9kJgXb1adqt3qv_tTC7PASwaaRVSnkRaPQ4es2UITH2WRCdVv0siIUlo4exKWgRjHExjKQPeTInQV189bZBzw9fyTUU3jdeYHNxO6XfUMvOzMa0IilwXTyp77byL-_yh4qGdUY_78uH6NjtPNe1QRkiolIZMWwVD3mlhYWOZItO8dfIQu6p4Bsy8ttSFP-ua941tRRR2jzdyYbeXYczM_eCCzhhGgn-Dmpd9bAhY5yA09lA3SKO7DmKHNZYKwFpZcleQ6gcq-rfwFTK2Zt8ZZNLAZB7WX-xahRCQHLihcBPuVR9X85UlIS4oj45vCMIqVVYFDE31NArH9W9NNlKxfpXegdjtvDKyQpQMe6RQW_hhkejjlV0XYPFIFb-NMz33T1CVbZUGgEBWtCCLxrb2DWrOX27iDUeHd.7tbVOU2Xs1SzXYxK3dqfrw/gwtok_tokens.u16'

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

tok = f"{WORK}/gwtok_tokens.u16"
if not PARTIAL_URL:
    raise SystemExit("brak GW_PARTIAL_URL — wklej podpisany link do czesciowych tokenow")
print("pobieram czesciowe tokeny...", flush=True)
subprocess.run(["curl", "-sSL", "-o", tok, PARTIAL_URL], check=True)

size = os.path.getsize(tok)
want = DONE * 256 * 2
if size != want:
    raise SystemExit(f"czesciowy plik ma {size} B, oczekiwano {want} B "
                     f"({size/512:.1f} obrazow) — link wygasl albo wskazuje co innego")
print(f"  {size/1e6:.0f} MB = {DONE} obrazow, dopisuje od tego miejsca", flush=True)

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
                "--out-prefix", f"{WORK}/gwtok", "--batch", "64",
                "--skip", str(DONE)], check=True)

for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
