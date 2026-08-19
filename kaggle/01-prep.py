"""Kaggle CPU prep for G-Weird. GPU off, Internet ON. Expect ~9h per shard.

SHARD comes from the environment so each bootstrap kernel differs by one line,
and each takes a disjoint slice of Conceptual Captions by skipping the rows the
earlier shards consumed. The skip is not bookkeeping: without it every kernel
starts at row zero and downloads the same pictures, which is exactly how three
G-Images preps once produced 150000 pairs of which 60000 were unique.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

# 400k now that images are stored as JPEG rather than raw uint8. Measured on the
# first two shards: raw cost 196 kB an image, which capped a shard at 90k pairs
# against Kaggle's 20 GB output limit. JPEG at quality 88 costs ~25 kB, so the
# same disk holds seven or eight times as many pairs — and pair count is what
# buys prompt obedience at this scale.
WANT = 400000
SHARD = int(os.environ.get("GWEIRD_SHARD", "0"))

# Measured on shards 0 and 1, not guessed: 62% of the links still resolve, so a
# shard consumes want/0.62 rows. The first two shards took rows 0-145729 and
# 130000-275000 respectively at the old 90k quota; anything from 300000 up is
# untouched.
SKIP = 300000 + (SHARD - 2) * 650000

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets", "pillow", "requests"], check=True)

print(f"shard {SHARD}: chce {WANT} par, pomijam {SKIP} wierszy", flush=True)
subprocess.run([sys.executable, "data/fetch_captions.py",
                "--want", str(WANT), "--skip", str(SKIP),
                "--res", "256", "--workers", "64",
                "--out-prefix", f"{WORK}/gweird"], check=True)

for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
