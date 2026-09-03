"""Kaggle CPU cell: build a DiffusionDB shard. GPU OFF, Internet ON.

CPU only, so this costs nothing from the weekly GPU quota — which matters,
because the quota is what the training will need and this is pure I/O.

SHARD picks a disjoint range of parts, so several of these can run without
fetching the same images twice. Each part is ~500 MB of PNG and becomes ~25 MB
of 256px JPEG, and parts are deleted as they are consumed, so peak disk is one
archive rather than the whole range.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

SHARD = int(os.environ.get("DDB_SHARD", "0"))
PARTS_PER_SHARD = 2

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "pillow"], check=True)

first = 1 + SHARD * PARTS_PER_SHARD
print(f"shard {SHARD}: czesci {first}-{first + PARTS_PER_SHARD - 1}", flush=True)

subprocess.run([sys.executable, "data/fetch_diffusiondb.py",
                "--first-part", str(first), "--parts", str(PARTS_PER_SHARD),
                "--res", "256", "--out-prefix", f"{WORK}/gweird"], check=True)

for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
