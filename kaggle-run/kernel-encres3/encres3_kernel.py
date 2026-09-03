"""Kaggle GPU cell: finish the mixed-corpus encode. T4x2, Internet ON.

The first attempt died at 1,467,585 of 2,357,878 with no inner traceback — the
signature of a kill rather than an exception, and the second time that has
happened at almost exactly 1.46M images despite a completely different corpus.
That reproducibility is why the progress line now carries resident memory: two
identical deaths without a number to look at is two theories and no evidence.

The tokens written so far are byte-exact (751,403,520 = 1,467,585 x 512, no
partial image), so they are downloaded and appended to rather than recomputed.
Captions are rewritten from scratch for the whole corpus, which costs no GPU and
makes the result independent of what the dead run left behind.
"""

import glob
import json
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

DONE = 1467585
PARTIAL_URL = 'https://www.kaggleusercontent.com/kf/344790285/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..UbrjRwxs8xc95oqTNiWE_Q.IkK-l60G3sRVfyE9JdWYsQwh9e8pw3yTbWrZejctm6t5QvDzFbPVCiB_u3lgSt8qVPS_h2mHYcDxjIdhB720MLW13S_jOQQtQAreDatYyJXalTkNv0GWYd88tWasOMizn-W3QMt1VE8-7w-jaHAe45E9CgLTF7BvunyClD2XKBPQO1qJZHKpbScMk0GJ8vRNWxLjyyjMLSzqJg1jTjdxhEvXU9RvnxUsWiSmrk6a_gKShB6hTPUZXOvfNLvneejiSWstx3ruYCRfSFlqre2wenITeBqAkgMPLYHKO1xmEH9zu3JBi5eAn-jSXIaFgc-51HOTfQETRJMJXs24VA07Pf4U6CTHjDcoXKprvNlQyyp-WAfxMii7MCwM6xZXhan3nBfysrNnGgmnWb1ceZfaYUpR8uzwmmR8397adaFvgthyp-cDpae96gE_vZ0v_MAd3R1vvXNOLsemwKJi9PqyzN9qUQCwZjN1B7AfLOCEZBg7NvvJS3iPROnX6wyPGKjaFzfb7oItP2XdQpTqGvaMf4y5ThXInwwVwqU_S5SCc4ktRWpWS8_bmsMpni8PafJvVXMPXQMAkutnxsFOhxFz5X1Rhg-vZXm0QwgHTxgfyaauQh52Jrrq4Gd42lJiEpOG.CvtBA-IY8a_60XMR-cFiFw/gwtok_tokens.u16'

import torch
if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}, compute {cap[0]}.{cap[1]}, "
      f"sztuk {torch.cuda.device_count()}", flush=True)
if cap[0] < 7:
    raise SystemExit(f"sm_{cap[0]}{cap[1]} nie jest wspierana")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

tok = f"{WORK}/gwtok_tokens.u16"
print("pobieram czesciowe tokeny...", flush=True)
subprocess.run(["curl", "-sSL", "--http1.1", "--continue-at", "-", "--retry", "5",
                "--retry-all-errors", "-o", tok, PARTIAL_URL], check=True)
size, want = os.path.getsize(tok), DONE * 256 * 2
if size != want:
    raise SystemExit(f"czesciowy plik ma {size} B, oczekiwano {want} — "
                     f"link wygasl albo wskazuje co innego")
print(f"  {size/1e6:.0f} MB = {DONE:,} obrazow", flush=True)

metas = sorted(glob.glob("/kaggle/input/**/gweird_meta.json", recursive=True))
prefixes = [m[:-len("_meta.json")] for m in metas]
total = sum(json.load(open(m))["n"] for m in metas)
print(f"{len(prefixes)} shardow, {total:,} par razem, do zrobienia "
      f"{total - DONE:,}", flush=True)

ckpts = sorted(glob.glob("/kaggle/input/**/vqvae.pt", recursive=True))
if not ckpts:
    raise SystemExit("brak vqvae.pt")

subprocess.run([sys.executable, "train/encode_corpus.py",
                "--data", *prefixes, "--ckpt", ckpts[0],
                "--out-prefix", f"{WORK}/gwtok", "--batch", "64",
                "--skip", str(DONE)], check=True)

for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
