"""Kaggle CPU cell: JourneyDB into a G-Weird shard. GPU OFF, Internet ON.

CPU only, so it costs nothing from the GPU quota.

The token is injected at push time and this file never carries one. That is not
caution for its own sake: a Kaggle token committed to this repository sat
publicly readable on GitHub for a day, and every prep kernel that cloned the repo
copied it into its own output.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

FIRST = int(os.environ.get("JDB_FIRST", "0"))
ARCHIVES = int(os.environ.get("JDB_ARCHIVES", "20"))
PER = int(os.environ.get("JDB_PER", "7500"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow", "requests"],
               check=True)

env = dict(os.environ, HF_TOKEN=HF_TOKEN)
print(f"archiwa {FIRST}-{FIRST+ARCHIVES-1}, po {PER} obrazow z kazdego", flush=True)
subprocess.run([sys.executable, "data/fetch_journeydb.py",
                "--first", str(FIRST), "--archives", str(ARCHIVES),
                "--per-archive", str(PER), "--res", "256",
                "--out-prefix", f"{WORK}/gweird"], check=True, env=env)

for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
