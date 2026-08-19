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

# 200k pairs per shard at 256px is ~39 GB of raw uint8 — too much for Kaggle's
# 20 GB output. 90k lands at ~17.7 GB, inside it with room to spare. The pixels
# are an intermediate anyway: once the VQ-VAE exists they become 512 bytes an
# image and the shards can be re-encoded and dropped.
WANT = 90000
SHARD = int(os.environ.get("GWEIRD_SHARD", "0"))

# Conceptual Captions rows are cheap to skip (streaming metadata) and roughly
# 70-80% of the links still resolve, so budget ~130k rows consumed per shard.
SKIP = SHARD * 130000

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
